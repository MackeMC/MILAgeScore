"""
data/prepare_data.py

Loads raw feature and label data, validates and preprocesses them,
and saves processed outputs ready for CellBagDataset.

Supports two input modes controlled by config.yaml:
    Mode A — Combined: features_path points to a single CSV containing both
             manual and latent features (latent cols identified by 'latent' in name).
    Mode B — Separated: manual_features_path and latent_embeddings_path point
             to two separate CSVs.

Run with:
    python data/prepare_data.py --config config/config.yaml

Outputs written to data/processed/:
    features_manual.npy     — filtered manual features, float32, shape (n_cells, n_manual)
    features_latent.npy     — active latent features, float32, shape (n_cells, n_latent)
    labels.csv              — cleaned metadata with lowercase columns
    manual_cols.pkl         — list of retained manual feature column names
    latent_cols.pkl         — list of retained latent featurae column names
    dataset_summary.csv     — per-group cell counts and senescence scores
"""

import argparse
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.feature_selection import VarianceThreshold

warnings.filterwarnings('ignore')


# ── Config ─────────────────────────────────────────────────────────────────────
def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


# ── Load raw data ──────────────────────────────────────────────────────────────
def load_raw(cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load raw data according to input mode.

    Mode A (combined): features_path is set, manual/latent paths are 'na'.
        Single CSV containing both manual and latent features.
        Latent columns are identified by having 'latent' anywhere in their name.

    Mode B (separated): manual_features_path and latent_embeddings_path are set.
        Two separate CSVs for manual and latent features respectively.

    Returns
    -------
    manual_df : pd.DataFrame  — manual morphological features only
    latent_df : pd.DataFrame  — VAE latent features only
    labels_df : pd.DataFrame  — raw metadata labels
    """
    data_cfg = cfg['data']
    features_path  = data_cfg.get('features_path', 'na')
    manual_path    = data_cfg.get('manual_features_path', 'na')
    latent_path    = data_cfg.get('latent_embeddings_path', 'na')
    labels_path    = data_cfg['labels_path']

    # ── Load labels ────────────────────────────────────────────────────────────
    print(f"Loading labels   : {labels_path}")
    labels_df = pd.read_csv(labels_path)
    print(f"  Labels shape   : {labels_df.shape}")

    # ── Mode A: combined CSV ───────────────────────────────────────────────────
    if features_path != 'na':
        print(f"\nMode A — combined features file")
        print(f"Loading features : {features_path}")
        full_df = pd.read_csv(features_path)
        print(f"  Features shape : {full_df.shape}")

        _check_row_alignment(full_df, labels_df, 'features', 'labels')

        latent_cols = [c for c in full_df.columns if 'latent' in c.lower()]
        manual_cols = [c for c in full_df.columns if 'latent' not in c.lower()]

        manual_df = full_df[manual_cols].reset_index(drop=True)
        latent_df = full_df[latent_cols].reset_index(drop=True)

        print(f"  Identified {len(manual_cols)} manual cols, "
              f"{len(latent_cols)} latent cols from combined file")

    # ── Mode B: separate CSVs ──────────────────────────────────────────────────
    elif manual_path != 'na' and latent_path != 'na':
        print(f"\nMode B — separated feature files")
        print(f"Loading manual   : {manual_path}")
        manual_df = pd.read_csv(manual_path).reset_index(drop=True)
        print(f"  Manual shape   : {manual_df.shape}")

        print(f"Loading latent   : {latent_path}")
        latent_df = pd.read_csv(latent_path).reset_index(drop=True)
        print(f"  Latent shape   : {latent_df.shape}")

        _check_row_alignment(manual_df, labels_df, 'manual features', 'labels')
        _check_row_alignment(latent_df, labels_df, 'latent features', 'labels')

    else:
        raise ValueError(
            "Invalid data config: set either 'features_path' (Mode A) "
            "or both 'manual_features_path' and 'latent_embeddings_path' (Mode B)."
        )

    return manual_df, latent_df, labels_df.reset_index(drop=True)


def _check_row_alignment(df_a: pd.DataFrame, df_b: pd.DataFrame,
                          name_a: str, name_b: str):
    if len(df_a) != len(df_b):
        raise ValueError(
            f"Row count mismatch — {name_a}: {len(df_a)} rows, "
            f"{name_b}: {len(df_b)} rows. "
            f"Files must be in identical row order."
        )


# ── Standardise labels ─────────────────────────────────────────────────────────
def standardise_labels(labels_df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """
    Normalise column names to lowercase and validate required columns exist.

    Expected columns (after lowercasing):
        well_name  — unique well identifier per image
        cell_line       — numerical cell line identifier
        roi        — cell identifier within a well (Region Of Interest)
        treatment  — treatment condition (stored, not used in training)
        passage    — raw passage number (int)
        repeat     — experimental repeat identifier

    All names are lowercased so downstream code never needs to guess
    capitalisation convention.
    """
    # Lowercase all column names
    original_cols = labels_df.columns.tolist()
    labels_df = labels_df.copy()
    labels_df.columns = [c.lower().strip() for c in labels_df.columns]
    
    # Map 'well' to 'well_name' if 'well_name' is missing but 'well' exists
    if 'well' in labels_df.columns and 'well_name' not in labels_df.columns:
        labels_df = labels_df.rename(columns={'well': 'well_name'})
        print("  Renamed 'well' column to 'well_name'")

    renamed = {o: n for o, n in zip(original_cols, labels_df.columns) if o != n}
    if renamed:
        print(f"\nColumn names lowercased: {renamed}")

    # Validate required columns are present
    required = ['well_name', 'cell_line', 'roi', 'treatment', 'passage', 'repeat']
    missing  = [c for c in required if c not in labels_df.columns]
    if missing:
        raise ValueError(
            f"Required label columns missing after lowercasing: {missing}\n"
            f"Available columns: {labels_df.columns.tolist()}"
        )

    # Enforce correct dtypes
    labels_df['passage']   = labels_df['passage'].astype(int)
    labels_df['well_name'] = labels_df['well_name'].astype(str).str.strip()
    labels_df['roi']       = labels_df['roi'].astype(str).str.strip()
    labels_df['treatment'] = labels_df['treatment'].astype(str).str.strip()
    labels_df['repeat']    = labels_df['repeat'].astype(str).str.strip()
    labels_df['cell_line'] = labels_df['cell_line'].astype(float).astype(int).astype(str).str.strip()

    # Apply age grouping if requested
    grouping_cfg = cfg['data'].get('grouping', {})
    mode = grouping_cfg.get('mode', 'individual')
    
    if mode == 'binned':
        n_bins = grouping_cfg.get('n_bins', 5)
        print(f"\nGrouping passages into {n_bins} age groups (mode='binned')...")
        
        # We MUST compute senescence score BEFORE remapping passages to bins,
        # otherwise infer_line_config and compute_senescence_scores will use the bins.
        # Actually, it's better to keep the original passages for s computation,
        # then remap them for the discrete labels.
        
        # Save original passages for s computation
        labels_df['passage_raw'] = labels_df['passage'].copy()

        for line in labels_df['cell_line'].unique():
            line_mask = labels_df['cell_line'] == line
            p = labels_df.loc[line_mask, 'passage']
            p_min, p_max = p.min(), p.max()
            
            bins = ((p - p_min) / (p_max - p_min) * n_bins).astype(int)
            bins = bins.clip(upper=n_bins - 1)
            labels_df.loc[line_mask, 'passage'] = bins
            
        print(f"  Passages remapped to bins 0-{n_bins-1}")

    print(f"\nLabel columns validated: {labels_df.columns.tolist()}")
    print(f"  Cell lines  : {sorted(labels_df['cell_line'].unique())}")
    print(f"  Passages    : {sorted(labels_df['passage'].unique())}")
    print(f"  Treatments  : {sorted(labels_df['treatment'].unique())}")
    print(f"  Total cells : {len(labels_df)}")

    return labels_df


def infer_line_config(labels_df: pd.DataFrame) -> dict:
    """
    Automatically infer line_config from the data.

    Assumes:
        - 'line' column contains values '1' and '2' (as strings after standardisation)
        - p_min  = minimum passage observed for that line
        - p_senescent = maximum passage observed for that line
          (the final passage is confirmed senescent by experimental design)

    Returns
    -------
    line_config : dict
        {
          '1': {'p_min': int, 'p_senescent': int},
          '2': {'p_min': int, 'p_senescent': int}
        }
    """
    lines_found = sorted(labels_df['cell_line'].unique())
    expected    = {'1', '2'}
    unexpected  = set(lines_found) - expected

    if unexpected:
        raise ValueError(
            f"Unexpected line identifiers found: {unexpected}. "
            f"Expected '1' and/or '2' only."
        )

    # Use passage_raw if available (binned mode)
    p_col = 'passage_raw' if 'passage_raw' in labels_df.columns else 'passage'

    line_config = {}
    print(f"\nLine config inferred from data (using {p_col}):")
    for line in lines_found:
        passages = labels_df[labels_df['cell_line'] == line][p_col]
        p_min        = int(passages.min())
        p_senescent  = int(passages.max())
        if p_senescent ==p_min:
            raise ValueError(
                f"Line {line} has only one passage: ({p_min});cannot normalise "
                f"senescence score (division by zero). Need >= 2 passages per line."
            )
        line_config[line] = {
            'p_min'       : p_min,
            'p_senescent' : p_senescent
        }
        print(f"  Line {line}: p_min={p_min}  p_senescent={p_senescent}  "
              f"passages={sorted(passages.unique().tolist())}")

    return line_config


# ── Senescence score ───────────────────────────────────────────────────────────
def compute_senescence_scores(labels_df: pd.DataFrame, cfg: dict) -> pd.Series:
    """
    Compute normalised senescence progression score s ∈ [0, 1] per cell.

        s = (passage - p_min) / (p_senescent - p_min)

    s = 0 at the earliest passage, s = 1 at confirmed senescence.
    Computed from labels only and stored back in labels_df as column 's'.
    """
    line_config = cfg['data']['line_config']
    p_col = 'passage_raw' if 'passage_raw' in labels_df.columns else 'passage'

    def _score(row):
        lc = line_config[row['cell_line']]
        return (row[p_col] - lc['p_min']) / (lc['p_senescent'] - lc['p_min'])

    s = labels_df.apply(_score, axis=1).astype(np.float32)
    print(f"\nSenescence scores computed (using {p_col}): min={s.min():.3f}  max={s.max():.3f}")
    return s


# ── Feature column identification ──────────────────────────────────────────────
def identify_feature_columns(
    manual_df: pd.DataFrame,
    latent_df: pd.DataFrame
) -> tuple[list, list]:
    """
    Identify numeric feature columns in each dataframe.
    Non-numeric columns are dropped with a warning.
    """
    # Explicitly exclude index / identifier columns that are numeric but carry
    # no morphology — these leak acquisition/passage order into the model.
    BLOCKLIST = {
            'unnamed: 0', 'unnamed:0', 'index', 'level_0',
            'roi', 'cell_id', 'id', 'object_id', 'well_id',
            'passage', 'cell_line', 'line', 's', 'repeat', 'dose',
    }
    manual_cols = [c for c in manual_df.columns
                   if pd.api.types.is_numeric_dtype(manual_df[c]) and c.lower().strip() not in BLOCKLIST]
    latent_cols = [c for c in latent_df.columns
                   if pd.api.types.is_numeric_dtype(latent_df[c]) and c.lower().strip() not in BLOCKLIST]

    n_dropped_manual = len(manual_df.columns) - len(manual_cols)
    n_dropped_latent = len(latent_df.columns) - len(latent_cols)

    if n_dropped_manual > 0:
        dropped = set(manual_df.columns) - set(manual_cols)
        print(f"\n  ⚠ Dropped {n_dropped_manual} non-numeric manual cols: {dropped}")
    if n_dropped_latent > 0:
        dropped = set(latent_df.columns) - set(latent_cols)
        print(f"  ⚠ Dropped {n_dropped_latent} non-numeric latent cols: {dropped}")

    print(f"\nNumeric features identified:")
    print(f"  Manual : {len(manual_cols)}")
    print(f"  Latent : {len(latent_cols)}")

    return manual_cols, latent_cols


# ── Manual feature filtering ───────────────────────────────────────────────────
def filter_manual_features(
    manual_df: pd.DataFrame,
    manual_cols: list,
    cfg: dict
) -> list:
    """
    Two-stage filtering of manual features:

    Stage 1 — Variance threshold:
        Remove features with variance below threshold. Features with near-zero
        variance carry no discriminative signal and add noise to distance metrics.

    Stage 2 — Correlation filter:
        Greedily remove one feature from each highly correlated pair (|r| > threshold).
        Correlated features inflate the effective weight of a shared underlying
        signal in both UMAP and MIL attention.
    """
    var_threshold  = cfg['data'].get('variance_threshold', 0.01)
    corr_threshold = cfg['data'].get('correlation_threshold', 0.95)
    X              = manual_df[manual_cols].values

    # Stage 1
    selector   = VarianceThreshold(threshold=var_threshold)
    selector.fit(X)
    passed_var    = [c for c, keep in zip(manual_cols, selector.get_support()) if keep]
    n_removed_var = len(manual_cols) - len(passed_var)
    print(f"\n  Variance filter  (threshold={var_threshold}): "
          f"removed {n_removed_var}, retained {len(passed_var)}")

    # Stage 2
    if cfg['data'].get('correlation_filter') is True:
        # pandas .corr() does pairwise-complete (NaN-robust); np.corrcoef would
        # return all-NaN if any value is NaN, silently disabling the filter.
        corr_matrix = manual_df[passed_var].corr().abs().values
        to_drop     = set()
        for i in range(len(passed_var)):
            if i in to_drop:
                continue
            for j in range(i + 1, len(passed_var)):
                if j not in to_drop and corr_matrix[i, j] > corr_threshold:
                    to_drop.add(j)

        passed_corr    = [c for idx, c in enumerate(passed_var) if idx not in to_drop]
        n_removed_corr = len(passed_var) - len(passed_corr)
        print(f"  Correlation filter (threshold={corr_threshold}): "
              f"removed {n_removed_corr}, retained {len(passed_corr)}")
        print(f"  Final manual feature count: {len(passed_corr)}")
    else:
        print(f"  Correlation filter skipped")
        passed_corr = passed_var

    return passed_corr


# ── Latent collapse screening ──────────────────────────────────────────────────
def check_latent_collapse(
    latent_df: pd.DataFrame,
    latent_cols: list,
    cfg: dict
) -> list:
    """
    Remove latent dimensions with near-zero variance — posterior collapse.
    Collapsed dimensions contribute no cell-level information and dilute
    pairwise distances in both UMAP and MIL attention computation.
    """
    threshold  = cfg['data'].get('latent_collapse_threshold', 0.01)
    latent_var = latent_df[latent_cols].var(axis=0)
    active     = latent_var[latent_var >= threshold].index.tolist()
    n_collapsed = len(latent_cols) - len(active)

    if n_collapsed > 0:
        collapsed = latent_var[latent_var < threshold].index.tolist()
        print(f"\n  ⚠ Posterior collapse: {n_collapsed} dims removed "
              f"(var < {threshold}): {collapsed}")
    else:
        print(f"\n  ✔ All {len(latent_cols)} latent dims active (var ≥ {threshold})")

    return active


# ── NaN handling ───────────────────────────────────────────────────────────────
def drop_nan_rows(
    manual_df: pd.DataFrame,
    latent_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    manual_cols: list,
    latent_cols: list
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Drop rows where any selected feature is NaN. Applied identically to
    all three dataframes to preserve row alignment.
    """
    nan_mask = (
        manual_df[manual_cols].isna().any(axis=1) |
        latent_df[latent_cols].isna().any(axis=1)
    )
    n_dropped = nan_mask.sum()

    if n_dropped > 0:
        print(f"\n  Dropping {n_dropped} rows with NaN in selected feature columns")
        manual_df = manual_df[~nan_mask].reset_index(drop=True)
        latent_df = latent_df[~nan_mask].reset_index(drop=True)
        labels_df = labels_df[~nan_mask].reset_index(drop=True)
    else:
        print(f"\n  ✔ No NaN rows found in selected features")

    return manual_df, latent_df, labels_df


# ── Dataset summary ────────────────────────────────────────────────────────────
def build_dataset_summary(labels_df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Per group (line × passage) cell counts, well counts, and senescence scores."""
    line_config = cfg['data']['line_config']
    records     = []

    grouping_mode = cfg['data'].get('grouping', {}).get('mode', 'individual')

    for (line, passage), grp in labels_df.groupby(['cell_line', 'passage']):
        lc = line_config[line]
        # In binned mode, 'passage' is the bin index, so we need a different score calculation
        # if we want it to represent the bin's relative position.
        # But compute_senescence_scores already ran and filled labels_df['s'].
        # We can just take the mean 's' of the group.
        s_mean = grp['s'].mean()
        
        records.append({
            'cell_line'           : line,
            'passage'             : passage,
            'age_group'           : passage if grouping_mode == 'binned' else 'na',
            's_mean'              : round(float(s_mean), 4),
            'n_cells'             : len(grp),
            'n_wells'             : grp['well_name'].nunique(),
            'n_repeats'           : grp['repeat'].nunique(),
            'cells_per_well_mean' : round(len(grp) / grp['well_name'].nunique(), 1),
        })

    return pd.DataFrame(records).sort_values(['cell_line', 'passage']).reset_index(drop=True)


# ── Save outputs ───────────────────────────────────────────────────────────────
def save_processed(
    manual_df: pd.DataFrame,
    latent_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    manual_cols: list,
    latent_cols: list,
    summary: pd.DataFrame,
    cfg: dict
):
    """
    Save processed outputs to processed_dir.

    features_manual.npy  — float32 array (n_cells, n_manual)
    features_latent.npy  — float32 array (n_cells, n_latent)
    labels.csv           — cleaned metadata CSV with lowercase columns + 's' column
    manual_cols.pkl      — retained manual column names
    latent_cols.pkl      — retained latent column names
    dataset_summary.csv  — per-group summary for inspection and thesis reporting
    """
    out_dir = Path(cfg['data']['processed_dir'])
    out_dir.mkdir(parents=True, exist_ok=True)

    manual_array = manual_df[manual_cols].values.astype(np.float32)
    latent_array = latent_df[latent_cols].values.astype(np.float32)

    # Save as .npy files
    np.save(out_dir / 'features_manual.npy', manual_array)
    np.save(out_dir / 'features_latent.npy', latent_array)

    # Save as .csv files
    pd.DataFrame(manual_array, columns=manual_cols).to_csv(out_dir / 'features_manual.csv', index=False)
    pd.DataFrame(latent_array, columns=latent_cols).to_csv(out_dir / 'features_latent.csv', index=False)

    labels_df.to_csv(out_dir / 'labels.csv', index=False)
    summary.to_csv(out_dir / 'dataset_summary.csv', index=False)

    with open(out_dir / 'manual_cols.pkl', 'wb') as f:
        pickle.dump(manual_cols, f)
    with open(out_dir / 'latent_cols.pkl', 'wb') as f:
        pickle.dump(latent_cols, f)

    print(f"\nProcessed outputs saved to '{out_dir}':")
    print(f"  features_manual.npy   {manual_array.shape}  "
          f"({manual_array.nbytes / 1e6:.1f} MB)")
    print(f"  features_latent.npy   {latent_array.shape}  "
          f"({latent_array.nbytes / 1e6:.1f} MB)")
    print(f"  labels.csv            ({len(labels_df)} rows, "
          f"cols: {labels_df.columns.tolist()})")
    print(f"  manual_cols.pkl       ({len(manual_cols)} features)")
    print(f"  latent_cols.pkl       ({len(latent_cols)} features)")
    print(f"  dataset_summary.csv   ({len(summary)} groups)")


# ── Main ───────────────────────────────────────────────────────────────────────
def main(config_path: str):
    cfg = load_config(config_path)

    # 1. Load raw files
    manual_df, latent_df, labels_df = load_raw(cfg)

    # 2. Standardise label column names to lowercase
    labels_df = standardise_labels(labels_df, cfg)

    # 3. Infer line_config automatically from data — no manual config needed
    line_config = infer_line_config(labels_df)
    cfg['data']['line_config'] = line_config          # inject so downstream fns work

    # 4. Compute and attach senescence scores
    labels_df['s'] = compute_senescence_scores(labels_df, cfg)

    # 5. Identify numeric feature columns
    manual_cols, latent_cols = identify_feature_columns(manual_df, latent_df)

    # 6. Filter manual features
    manual_cols = filter_manual_features(manual_df, manual_cols, cfg)

    # 7. Screen latent dims for posterior collapse
    latent_cols = check_latent_collapse(latent_df, latent_cols, cfg)

    # 8. Drop NaN rows consistently across all three dataframes
    manual_df, latent_df, labels_df = drop_nan_rows(
        manual_df, latent_df, labels_df, manual_cols, latent_cols
    )

    # 9. Build per-group summary
    summary = build_dataset_summary(labels_df, cfg)
    print(f"\nDataset summary:\n{summary.to_string(index=False)}")

    # 10. Save all processed outputs
    save_processed(manual_df, latent_df, labels_df,
                   manual_cols, latent_cols, summary, cfg)

    print("\n✔ Data preparation complete.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Prepare raw data for MILAgeScore training'
    )
    parser.add_argument(
        '--config', type=str, default='config/config.yaml',
        help='Path to config.yaml'
    )
    args = parser.parse_args()
    main(args.config)
