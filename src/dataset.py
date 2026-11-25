"""
src/dataset.py

MIL bag dataset for ordinal senescence/age score prediction.

Consumes outputs from data/prepare_data.py:
    data/processed/features_manual.npy
    data/processed/features_latent.npy
    data/processed/labels.csv
    data/processed/manual_cols.pkl
    data/processed/latent_cols.pkl

Key design decisions:
    - Features and metadata are kept as separate aligned arrays throughout
    - Bags are constructed on-the-fly per __getitem__ call, acting as
      stochastic data augmentation across training epochs
    - Per-well capping prevents well-density confounds from dominating bags
    - Train/val split is performed at the group (line × passage) level to
      prevent information leakage between splits
    - Scalers are fit on train split only and applied to val/inference splits
"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from src.utils import resolve_block_weights, filter_features
from torch.utils.data import DataLoader, Dataset
from src.split_loco import split_leave_one_line_out
from src.binning import prepare_binned_labels, split_binned_kfold


# ── Load processed data ────────────────────────────────────────────────────────
def load_processed(processed_dir: str) -> tuple:
    """
    Load all outputs from data/prepare_data.py.

    Returns
    -------
    features_manual : np.ndarray, float32, shape (n_cells, n_manual)
    features_latent : np.ndarray, float32, shape (n_cells, n_latent)
    labels_df       : pd.DataFrame, shape (n_cells, n_meta_cols)
    manual_cols     : list[str]
    latent_cols     : list[str]
    """
    d = Path(processed_dir)

    features_manual = np.load(d / 'features_manual.npy')   # (n_cells, n_manual)
    features_latent = np.load(d / 'features_latent.npy')   # (n_cells, n_latent)
    labels_df       = pd.read_csv(d / 'labels.csv')

    with open(d / 'manual_cols.pkl', 'rb') as f:
        manual_cols = pickle.load(f)
    with open(d / 'latent_cols.pkl', 'rb') as f:
        latent_cols = pickle.load(f)

    assert len(features_manual) == len(features_latent) == len(labels_df), (
        f"Row count mismatch — manual: {len(features_manual)}, "
        f"latent: {len(features_latent)}, labels: {len(labels_df)}"
    )

    print(f"Loaded processed data:")
    print(f"  features_manual : {features_manual.shape}")
    print(f"  features_latent : {features_latent.shape}")
    print(f"  labels          : {labels_df.shape}")
    print(f"  manual_cols     : {len(manual_cols)}")
    print(f"  latent_cols     : {len(latent_cols)}")

    return features_manual, features_latent, labels_df, manual_cols, latent_cols


# ── Group-level train / val split ──────────────────────────────────────────────
def split_by_group(
    labels_df: pd.DataFrame,
    val_fraction: float = 0.2,
    random_seed: int = 42
) -> tuple[np.ndarray, np.ndarray]:
    """
    Split cell indices into train and val subsets by holding out entire
    line × passage groups.

    Splitting at the group level is essential: cell-level splitting would
    leak the feature distribution of a passage into both splits, producing
    an optimistically biased validation loss.

    Stratification ensures both splits contain cells from each line where
    possible, preventing a single line from being entirely absent in val.

    Returns
    -------
    train_idx : np.ndarray of int — row indices into features/labels arrays
    val_idx   : np.ndarray of int — row indices into features/labels arrays
    """
    rng = np.random.default_rng(random_seed)

    # Build list of (line, passage) group keys
    groups      = labels_df.groupby(['cell_line', 'passage'])
    group_keys  = list(groups.groups.keys())

    # Stratify by line so val contains at least one group per line
    line_groups = {}
    for key in group_keys:
        cell_line = key[0]
        line_groups.setdefault(cell_line, []).append(key)

    val_keys = []
    for line, keys in line_groups.items():
        keys_shuffled = list(keys)
        rng.shuffle(keys_shuffled)
        n_val = max(1, int(np.floor(val_fraction * len(keys_shuffled))))
        val_keys.extend(keys_shuffled[:n_val])

    val_key_set = set(val_keys)
    train_keys  = [k for k in group_keys if k not in val_key_set]

    # REPLACE the two apply() blocks in split_by_group with:
    group_key_col = list(zip(labels_df['cell_line'], labels_df['passage']))
    train_key_set = set(train_keys)
    val_key_set = set(val_keys)

    train_idx = np.array([
        i for i, k in enumerate(group_key_col) if k in train_key_set
    ], dtype=np.int64)

    val_idx = np.array([
        i for i, k in enumerate(group_key_col) if k in val_key_set
    ], dtype=np.int64)

    print(f"\nGroup-level train/val split:")
    print(f"  Train groups : {len(train_keys)}  ({len(train_idx)} cells)")
    print(f"  Val groups   : {len(val_keys)}   ({len(val_idx)} cells)")
    print(f"  Val groups   : {sorted(val_keys)}")

    return train_idx, val_idx


# ── Scaler fitting and application ────────────────────────────────────────────
# ── Utility scaler for zero-width feature blocks ──────────────────────────────
class NoOpScaler:
    """A drop-in replacement scaler for zero-column feature blocks."""
    def fit(self, X):
        return self
    def transform(self, X):
        return X


def fit_scalers(
    features_manual: np.ndarray,
    features_latent: np.ndarray,
    train_idx: np.ndarray
) -> tuple[StandardScaler, StandardScaler]:
    """
    Fit independent StandardScalers on the training split only.
    Manual and latent features are scaled independently so that per-feature
    variance is equalised within each block before concatenation.
    Fitting on train_idx only prevents data leakage from val into scaling.
    """
    if features_manual.shape[1] == 0:
        scaler_manual = NoOpScaler()
    else:
        scaler_manual = StandardScaler()
        scaler_manual.fit(features_manual[train_idx])

    if features_latent.shape[1] == 0:
        scaler_latent = NoOpScaler()
    else:
        scaler_latent = StandardScaler()
        scaler_latent.fit(features_latent[train_idx])

    print(f"\nScalers fit on {len(train_idx)} training cells")
    return scaler_manual, scaler_latent


def apply_scalers(
    features_manual: np.ndarray,
    features_latent: np.ndarray,
    scaler_manual: StandardScaler,
    scaler_latent: StandardScaler
) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply pre-fitted scalers to the full feature arrays.
    Returns scaled float32 arrays.
    """
    # When a block has zero columns, NoOpScaler returns input unchanged
    scaled_manual = scaler_manual.transform(features_manual).astype(np.float32)
    scaled_latent = scaler_latent.transform(features_latent).astype(np.float32)
    return scaled_manual, scaled_latent


def save_scalers(
    scaler_manual: StandardScaler,
    scaler_latent: StandardScaler,
    processed_dir: str,
    scaler_dir: str = None
):
    """
    Persist fitted scalers to disk.
    Must be reloaded and applied to any future treatment/inference data
    — never refit scalers on new data.
    """
    d = Path(processed_dir)
    with open(d / 'scaler_manual.pkl', 'wb') as f:
        pickle.dump(scaler_manual, f)
    with open(d / 'scaler_latent.pkl', 'wb') as f:
        pickle.dump(scaler_latent, f)
    print(f"Scalers saved to {d}")

    if scaler_dir:
        sd = Path(scaler_dir)
        sd.mkdir(parents=True, exist_ok=True)
        with open(sd / 'scaler_manual.pkl', 'wb') as f:
            pickle.dump(scaler_manual, f)
        with open(sd / 'scaler_latent.pkl', 'wb') as f:
            pickle.dump(scaler_latent, f)
        print(f"Scalers also saved to {sd}")


def load_scalers(processed_dir: str, scaler_dir: str = None) -> tuple[StandardScaler, StandardScaler]:
    """Load previously fitted scalers for inference on new/treatment data."""
    d = Path(scaler_dir) if scaler_dir else Path(processed_dir)
    with open(d / 'scaler_manual.pkl', 'rb') as f:
        scaler_manual = pickle.load(f)
    with open(d / 'scaler_latent.pkl', 'rb') as f:
        scaler_latent = pickle.load(f)
    return scaler_manual, scaler_latent


# ── CellBagDataset ─────────────────────────────────────────────────────────────
class CellBagDataset(Dataset):
    """
    Multiple Instance Learning dataset for ordinal senescence prediction.

    Each __getitem__ returns one synthetic bag: a random subsample of cells
    drawn from all wells belonging to a single line × passage group.
    Bags are constructed on-the-fly, providing stochastic augmentation
    across epochs without any repeated fixed bag structure.

    Parameters
    ----------
    features_manual : np.ndarray, float32 (n_cells, n_manual) — pre-scaled
    features_latent : np.ndarray, float32 (n_cells, n_latent) — pre-scaled
    labels_df       : pd.DataFrame — full labels dataframe (all cells)
    cell_idx        : np.ndarray   — row indices for this split (train or val)
    cells_per_bag   : int          — fixed bag size
    bags_per_group  : int          — synthetic bags generated per line × passage group
    max_well_fraction : float      — max fraction of bag from any single well
    w_manual        : float        — block-level weight applied to manual features
    w_latent        : float        — block-level weight applied to latent features
    random_seed     : int
    """

    def __init__(
        self,
        features_manual: np.ndarray,
        features_latent: np.ndarray,
        labels_df: pd.DataFrame,
        cell_idx: np.ndarray,
        cells_per_bag: int = 200,
        bags_per_group: int = 8,
        max_well_fraction: float = 0.35,
        w_manual: float = 1.0,
        w_latent: float = 1.0,
        random_seed: int = 42,
    ):
        self.cells_per_bag     = cells_per_bag
        self.bags_per_group    = bags_per_group
        self.max_well_fraction = max_well_fraction
        self.w_manual          = w_manual
        self.w_latent          = w_latent
        self.rng               = np.random.default_rng(random_seed)

        # Subset to this split's indices only
        self.features_manual = features_manual[cell_idx]   # (n_split_cells, n_manual)
        self.features_latent = features_latent[cell_idx]   # (n_split_cells, n_latent)
        self.labels_df       = labels_df.iloc[cell_idx].reset_index(drop=True)

        # ── Build group index ──────────────────────────────────────────────────
        # group_index: list of (cell_line, passage, s) — one entry per group
        # group_well_cells: (cell_line, passage) → {well_name: array of local row indices}
        self.group_index     = []
        self.group_well_cells = {}

        for (cell_line, passage), grp in self.labels_df.groupby(['cell_line', 'passage']):
            s = float(grp['s'].iloc[0])
            self.group_index.append((cell_line, passage, s))

            well_dict = {}
            for well_name, well_grp in grp.groupby('well_name'):
                well_dict[well_name] = well_grp.index.values  # local indices into split arrays
            self.group_well_cells[(cell_line, passage)] = well_dict

        self.n_groups  = len(self.group_index)
        n_manual_feats = features_manual.shape[1]
        n_latent_feats = features_latent.shape[1]
        self.n_features = n_manual_feats + n_latent_feats

        # ASSERTION: Effective weight of every non-empty block must be > 0
        if n_manual_feats > 0:
            assert self.w_manual > 0, f"Manual features present ({n_manual_feats}) but w_manual is {self.w_manual}"
        if n_latent_feats > 0:
            assert self.w_latent > 0, f"Latent features present ({n_latent_feats}) but w_latent is {self.w_latent}"

        # Warn if any group has fewer cells than cells_per_bag
        for (cell_line, passage, s) in self.group_index:
            well_dict  = self.group_well_cells[(cell_line, passage)]
            total_cells = sum(len(v) for v in well_dict.values())
            if total_cells < cells_per_bag:
                print(f"  ⚠ cell_line {cell_line} P{passage}: only {total_cells} cells "
                      f"< cells_per_bag={cells_per_bag} — bags will sample with replacement")

        print(f"\nCellBagDataset initialised:")
        print(f"  Groups          : {self.n_groups}")
        print(f"  Bags per group  : {bags_per_group}")
        print(f"  Total bags      : {self.n_groups * bags_per_group}")
        print(f"  Cells per bag   : {cells_per_bag}")
        print(f"  Feature dim     : {self.n_features} "
              f"({features_manual.shape[1]} manual × {w_manual} + "
              f"{features_latent.shape[1]} latent × {w_latent})")

    def __len__(self) -> int:
        return self.n_groups * self.bags_per_group

    def __getitem__(self, idx: int):
        """
        Returns
        -------
        bag     : torch.Tensor, shape (cells_per_bag, n_features), float32
        label   : torch.Tensor, scalar float32 — senescence score s ∈ [0, 1]
        meta    : dict — group metadata for logging (not consumed by the model)
        """
        group_idx               = idx % self.n_groups
        cell_line, passage, s        = self.group_index[group_idx]
        well_dict               = self.group_well_cells[(cell_line, passage)]

        sampled_idx             = self._sample_bag(well_dict)

        manual_block = self.features_manual[sampled_idx] * self.w_manual
        latent_block = self.features_latent[sampled_idx] * self.w_latent
        bag          = np.concatenate([manual_block, latent_block], axis=1)

        # ASSERTION: Sampled bag has non-zero variance across cells
        # We check if any feature has variance > 0 across the bag
        assert np.any(np.var(bag, axis=0) > 1e-12), "Sampled bag has zero variance across all features."

        return (
            torch.from_numpy(bag),
            torch.tensor(s, dtype=torch.float32),
            {'cell_line': cell_line, 'passage': passage, 's': s}
        )

    def _sample_bag(self, well_dict: dict) -> np.ndarray:
        """
        Sample cells_per_bag cell indices from well_dict.

        Per-well cap: each well contributes at most
            ceil(max_well_fraction × cells_per_bag) cells per bag.
        This prevents wells with high seeding density from dominating
        the bag's feature distribution.

        If the total capped pool is smaller than cells_per_bag,
        sampling is done with replacement on the full pool.
        """
        max_per_well = int(np.ceil(self.max_well_fraction * self.cells_per_bag))
        pool_parts   = []

        for well_idx in well_dict.values():
            n_available = len(well_idx)
            n_draw      = min(n_available, max_per_well)
            chosen      = self.rng.choice(n_available, size=n_draw, replace=False)
            pool_parts.append(well_idx[chosen])

        pool    = np.concatenate(pool_parts)
        n_pool  = len(pool)
        replace = n_pool < self.cells_per_bag

        sampled = self.rng.choice(n_pool, size=self.cells_per_bag, replace=replace)
        return pool[sampled]


# ── DataLoader factory ─────────────────────────────────────────────────────────
def build_dataloaders(processed_dir: str, cfg: dict) -> tuple[DataLoader, DataLoader, StandardScaler, StandardScaler] | tuple[DataLoader, DataLoader, DataLoader, StandardScaler, StandardScaler, pd.DataFrame]:
    """
    Full pipeline from processed directory to ready-to-use DataLoaders.

    Steps:
        1. Load processed numpy arrays and labels
        2. Split indices at the group level (no cell-level leakage)
        3. Fit scalers on train indices only, apply to full arrays
        4. Persist scalers for later inference on treatment data
        5. Construct CellBagDataset for train and val splits
        6. Wrap in DataLoaders

    Returns
    -------
    If inner_val is False:
        train_loader  : DataLoader
        val_loader    : DataLoader
        scaler_manual : fitted StandardScaler (for inference reuse)
        scaler_latent : fitted StandardScaler (for inference reuse)
    If inner_val is True:
        train_loader  : DataLoader
        inner_val_loader : DataLoader
        heldout_loader : DataLoader
        scaler_manual : fitted StandardScaler (for inference reuse)
        scaler_latent : fitted StandardScaler (for inference reuse)
        inner_wells_df : pd.DataFrame
    """
    bags_cfg     = cfg['bags']
    training_cfg = cfg['training']

    # 1. Load
    features_manual, features_latent, labels_df, manual_cols, latent_cols = load_processed(processed_dir)

    # Apply feature_set selection by dropping unused blocks
    feature_set = cfg['data'].get('feature_set', 'both').lower()
    if feature_set == 'manual':
        features_latent = np.zeros((features_latent.shape[0], 0), dtype=features_latent.dtype)
        latent_cols = []
    elif feature_set == 'latent':
        features_manual = np.zeros((features_manual.shape[0], 0), dtype=features_manual.dtype)
        manual_cols = []

    # Apply feature_include
    feature_include = cfg['data'].get('feature_include')
    features_manual, features_latent, _, _ = filter_features(
        features_manual, features_latent, manual_cols, latent_cols, feature_include
    )

    # 2. Split
    val_cfg = cfg.get('validation', {})
    scheme = val_cfg.get('scheme', 'group').lower()
    inner_val = val_cfg.get('inner_val', False)
    inner_wells_df = None

    if scheme == 'binned_kfold':
        labels_df = prepare_binned_labels(labels_df, val_cfg.get('n_bins', 10))
        train_idx, val_idx = split_binned_kfold(
            labels_df,
            n_folds=val_cfg.get('n_folds', 5),
            fold=val_cfg.get('fold', 0),
            fold_seed=val_cfg.get('fold_seed', 42),
        )

    elif scheme == 'loco':
        if val_cfg.get('n_bins'):
            labels_df = prepare_binned_labels(labels_df, val_cfg['n_bins'])
        line_column = 'cell_line_orig' if 'cell_line_orig' in labels_df.columns else 'cell_line'

        if inner_val:
            train_idx, inner_idx, val_idx, inner_wells_df = split_leave_one_line_out(
                labels_df,
                held_out_line=val_cfg['held_out_line'],
                line_column=line_column,
                inner_val=True,
                inner_val_seed=val_cfg.get('inner_val_seed', bags_cfg['random_seed'])
            )
        else:
            train_idx, val_idx = split_leave_one_line_out(
                labels_df,
                held_out_line=val_cfg['held_out_line'],
                line_column=line_column,
            )

    elif scheme == 'group':
        train_idx, val_idx = split_by_group(labels_df,
            val_fraction = training_cfg['val_fraction'],
            random_seed = bags_cfg['random_seed'],
        )
    else:
        raise ValueError(f"Unknown validation.scheme '{scheme}'. Use 'group' or 'loco'.")

    # 3. Fit and apply scalers
    scaler_manual, scaler_latent = fit_scalers(features_manual, features_latent, train_idx)
    features_manual, features_latent = apply_scalers(
        features_manual, features_latent, scaler_manual, scaler_latent
    )

    # 4. Persist scalers
    scaler_dir = None
    if 'outputs' in cfg and 'log_dir' in cfg['outputs']:
        scaler_dir = Path(cfg['outputs']['log_dir']) / 'scalers'

    save_scalers(scaler_manual, scaler_latent, processed_dir, scaler_dir=scaler_dir)

    # 5. Compute block-level weights (respect zero-width blocks)
    n_manual = features_manual.shape[1]
    n_latent = features_latent.shape[1]
    w_manual, w_latent = resolve_block_weights(n_manual, n_latent, cfg)

    # 6. Construct datasets
    train_dataset = CellBagDataset(
        features_manual=features_manual,
        features_latent=features_latent,
        labels_df=labels_df,
        cell_idx=train_idx,
        cells_per_bag=bags_cfg['cells_per_bag'],
        bags_per_group=bags_cfg['bags_per_group'],
        max_well_fraction=bags_cfg['max_well_fraction'],
        w_manual=w_manual,
        w_latent=w_latent,
        random_seed=bags_cfg['random_seed'],
    )

    if inner_val:
        inner_dataset = CellBagDataset(
            features_manual=features_manual,
            features_latent=features_latent,
            labels_df=labels_df,
            cell_idx=inner_idx,
            cells_per_bag=bags_cfg['cells_per_bag'],
            bags_per_group=bags_cfg.get('val_bags_per_group', 30),
            max_well_fraction=bags_cfg['max_well_fraction'],
            w_manual=w_manual,
            w_latent=w_latent,
            random_seed=bags_cfg['random_seed'] + 1,
        )
        heldout_dataset = CellBagDataset(
            features_manual=features_manual,
            features_latent=features_latent,
            labels_df=labels_df,
            cell_idx=val_idx,
            cells_per_bag=bags_cfg['cells_per_bag'],
            bags_per_group=bags_cfg.get('val_bags_per_group', 30),
            max_well_fraction=bags_cfg['max_well_fraction'],
            w_manual=w_manual,
            w_latent=w_latent,
            random_seed=bags_cfg['random_seed'] + 2,
        )
    else:
        val_dataset = CellBagDataset(
            features_manual=features_manual,
            features_latent=features_latent,
            labels_df=labels_df,
            cell_idx=val_idx,
            cells_per_bag=bags_cfg['cells_per_bag'],
            bags_per_group=bags_cfg.get('val_bags_per_group', 30),
            max_well_fraction=bags_cfg['max_well_fraction'],
            w_manual=w_manual,
            w_latent=w_latent,
            random_seed=bags_cfg['random_seed'] + 1,
        )

    # 7. Wrap in DataLoaders
    # num_workers=0: required on Windows — multiprocessing with custom Dataset
    # causes deadlocks with spawn start method
    train_loader = DataLoader(
        train_dataset,
        batch_size=training_cfg['batch_size'],
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        drop_last=True,
    )

    if inner_val:
        inner_val_loader = DataLoader(
            inner_dataset,
            batch_size=training_cfg['batch_size'],
            shuffle=False,
            num_workers=0,
            pin_memory=True,
            drop_last=False,
        )
        heldout_loader = DataLoader(
            heldout_dataset,
            batch_size=training_cfg['batch_size'],
            shuffle=False,
            num_workers=0,
            pin_memory=True,
            drop_last=False,
        )
        return train_loader, inner_val_loader, heldout_loader, scaler_manual, scaler_latent, inner_wells_df
    else:
        val_loader = DataLoader(
            val_dataset,
            batch_size=training_cfg['batch_size'],
            shuffle=False,
            num_workers=0,
            pin_memory=True,
            drop_last=False,
        )
        return train_loader, val_loader, scaler_manual, scaler_latent
