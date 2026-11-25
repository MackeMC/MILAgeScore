"""
src/binning.py

Age-group binning + stratified well-level K-fold for MILAgeScore.

prepare_binned_labels : adds `age_group`, re-labels s as bin/(n_bins-1), and
                        pools both cell lines so one bin == one MIL "group".
split_binned_kfold    : every bin appears in every fold's validation set.
"""
import numpy as np
import pandas as pd


def prepare_binned_labels(labels_df: pd.DataFrame, n_bins: int = 10) -> pd.DataFrame:
    """
    Equal-width bins on the normalised score s in [0, 1] (aligned across lines).

    Overwrites, so CellBagDataset / build_loss / train.py work unchanged:
        s          -> age_group / (n_bins - 1)       (n_bins ordinal ranks)
        cell_line  -> 'all'                          (lines pooled in a bin)
        passage    -> age_group                      (group key = the bin)
        well_name  -> '<line>_P<passage>_<well>'     (unique across the pool)
    Originals are kept in *_orig columns / s_continuous.
    """
    df = labels_df.copy()
    if 'age_group' in df.columns and 'cell_line_orig' in df.columns:
        return df                                    # already prepared
    s = pd.to_numeric(df['s'], errors='coerce').to_numpy(dtype=float)
    ok = np.isfinite(s)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins = np.full(len(df), -1, dtype=int)
    bins[ok] = np.clip(np.digitize(s[ok], edges[1:-1]), 0, n_bins - 1)

    df['s_continuous'] = s
    df['cell_line_orig'] = df['cell_line'].astype(str)
    df['passage_orig'] = df['passage']
    df['well_orig'] = df['well_name'].astype(str)
    df['age_group'] = bins
    df['well_name'] = (df['cell_line_orig'] + '_P' + df['passage_orig'].astype(str)
                       + '_' + df['well_orig'])
    df['s'] = np.where(ok, bins / (n_bins - 1), np.nan)
    df['cell_line'] = 'all'
    df['passage'] = bins
    return df


def assign_folds(labels_df: pd.DataFrame, n_folds: int = 5, fold_seed: int = 42) -> dict:
    """well_name -> fold id. Wells of each bin are dealt round-robin (shuffled
    within each source passage first), so every bin is in every fold and each
    fold mixes both lines and many passages."""
    wells = (labels_df.loc[labels_df['age_group'] >= 0,
                           ['age_group', 'cell_line_orig', 'passage_orig', 'well_name']]
             .drop_duplicates('well_name'))
    rng = np.random.default_rng(fold_seed)
    fold_of = {}
    for b, gb in wells.groupby('age_group'):
        if gb['well_name'].nunique() < n_folds:
            raise ValueError(f"age_group {b} has only {gb['well_name'].nunique()} "
                             f"wells (< n_folds={n_folds}); use fewer folds/bins.")
        order = []
        for _, gp in gb.sort_values(['cell_line_orig', 'passage_orig']).groupby(
                ['cell_line_orig', 'passage_orig'], sort=False):
            w = gp['well_name'].to_numpy().copy()
            rng.shuffle(w)
            order.extend(w.tolist())
        shift = int(rng.integers(n_folds))           # avoid always filling fold 0 first
        for i, w in enumerate(order):
            fold_of[w] = (i + shift) % n_folds
    return fold_of


def split_binned_kfold(labels_df: pd.DataFrame, n_folds: int = 5, fold: int = 0,
                       fold_seed: int = 42):
    """Return (train_idx, val_idx) positional indices for the requested fold."""
    if not 0 <= fold < n_folds:
        raise ValueError(f"fold must be in [0, {n_folds - 1}]; got {fold}")
    fold_of = assign_folds(labels_df, n_folds, fold_seed)
    wf = labels_df['well_name'].map(fold_of).to_numpy(dtype=float)   # NaN = unlabelled
    valid = np.isfinite(wf)
    val_idx = np.where(valid & (wf == fold))[0]
    train_idx = np.where(valid & (wf != fold))[0]

    tr_b = set(labels_df['age_group'].iloc[train_idx])
    va_b = set(labels_df['age_group'].iloc[val_idx])
    assert tr_b == va_b, f"bins missing from a split: train={sorted(tr_b)} val={sorted(va_b)}"

    print(f"\nBinned stratified K-fold  (fold {fold + 1}/{n_folds}, fold_seed={fold_seed})")
    print(f"  Train: {len(train_idx)} cells | Val: {len(val_idx)} cells | bins: {len(va_b)}")
    summ = (labels_df.iloc[val_idx].groupby('age_group')
            .agg(val_wells=('well_name', 'nunique'), val_cells=('well_name', 'size')))
    print(summ.to_string())
    return train_idx, val_idx
