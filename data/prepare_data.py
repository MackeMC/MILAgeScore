"""
data/prepare_data.py

Row-level data preparation for MILAgeScore. Every optional operation is an
explicit switch in config.yaml (data.preprocess.*). No fitted scaling happens
here: scaling / block weighting belong to the runtime pipeline so that fitted
statistics only ever see the training split.

Run with:
    python data/prepare_data.py --config config/config.yaml

Pipeline (each stage is marked [always] or [switch]):
    1. load_raw                      [always]  Mode A (combined CSV) or Mode B (two CSVs)
    2. standardise_labels            [always]  lowercase columns, dtypes, required-column check
    3. line_config + senescence s    [always]  inferred from data, or explicit override
    4. identify_feature_columns      [always]  numeric only, minus blocklist
    5. NaN / inf policy              [switch]  preprocess.nan_policy = drop | error
    6. variance filter               [switch]  preprocess.variance_filter.enabled
    7. correlation filter            [switch]  preprocess.correlation_filter.enabled
    8. latent collapse screen        [switch]  preprocess.latent_collapse.enabled
    9. dataset summary + save        [always]  .npy/.pkl/labels.csv/summary/report;
                                               .csv feature copies [switch]

Config (all keys optional; defaults shown):

data:
  features_path: "na"
  manual_features_path: "na"
  latent_embeddings_path: "na"
  labels_path: ...
  processed_dir: "data/processed/"
  latent_substring: "latent"          # Mode A: columns containing this are latent
  line_config: null                   # optional {"1": {p_min: 4, p_senescent: 15}, ...}
  preprocess:
    nan_policy: drop                  # drop | error
    extra_blocklist: []               # extra column names to exclude from features
    variance_filter:    {enabled: true,  threshold: 0.001, blocks: [manual]}
    correlation_filter: {enabled: false, threshold: 0.95,  blocks: [manual]}
    latent_collapse:    {enabled: false, threshold: 0.01}
    save_feature_csv: false           # also write features_*.csv copies

Legacy flat keys (data.variance_threshold, data.correlation_filter,
data.correlation_threshold, data.latent_collapse_threshold) are still honoured
when the matching preprocess.* entry is absent.

Outputs written to processed_dir:
    features_manual.npy, features_latent.npy, labels.csv,
    manual_cols.pkl, latent_cols.pkl, dataset_summary.csv,
    preprocess_report.json   (which stages ran, what each dropped)

NOTE (LOCO): filters here use statistics from ALL cells. With the filters
disabled (or on pre-pruned data) this is moot; otherwise treat it as a mild
whole-dataset step and say so in the methods.
"""

import argparse
import json
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

warnings.filterwarnings('ignore')

REQUIRED_LABEL_COLS = ['well_name', 'cell_line', 'roi', 'treatment', 'passage', 'repeat']

DEFAULT_BLOCKLIST = {
    'unnamed: 0', 'unnamed:0', 'index', 'level_0',
    'roi', 'cell_id', 'id', 'object_id', 'well_id',
    'passage', 'cell_line', 'line', 's', 'repeat', 'dose', 'label',
}


# ── Config ─────────────────────────────────────────────────────────────────────
def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def resolve_preprocess_cfg(cfg: dict) -> dict:
    """Merge data.preprocess.* with legacy flat keys and defaults."""
    data = cfg['data']
    pp = dict(data.get('preprocess') or {})

    def sect(name, defaults, legacy=None):
        s = dict(defaults)
        if legacy:
            s.update({k: v for k, v in legacy.items() if v is not None})
        s.update(pp.get(name) or {})
        return s

    out = {
        'nan_policy': str(pp.get('nan_policy', 'drop')).lower(),
        'extra_blocklist': [str(c).lower().strip() for c in pp.get('extra_blocklist', [])],
        'save_feature_csv': bool(pp.get('save_feature_csv', False)),
        'variance_filter': sect(
            'variance_filter',
            {'enabled': True, 'threshold': 0.001, 'blocks': ['manual']},
            {'threshold': data.get('variance_threshold')}),
        'correlation_filter': sect(
            'correlation_filter',
            {'enabled': False, 'threshold': 0.95, 'blocks': ['manual']},
            {'enabled': data.get('correlation_filter'),
             'threshold': data.get('correlation_threshold')}),
        'latent_collapse': sect(
            'latent_collapse',
            {'enabled': False, 'threshold': 0.01},
            {'threshold': data.get('latent_collapse_threshold')}),
    }
    if out['nan_policy'] not in {'drop', 'error'}:
        raise ValueError(f"preprocess.nan_policy must be 'drop' or 'error'; got '{out['nan_policy']}'")
    for key in ('variance_filter', 'correlation_filter'):
        bad = set(out[key]['blocks']) - {'manual', 'latent'}
        if bad:
            raise ValueError(f"preprocess.{key}.blocks has invalid entries: {bad}")
    return out


# ── 1. Load raw data ───────────────────────────────────────────────────────────
def _check_row_alignment(df_a, df_b, name_a, name_b):
    if len(df_a) != len(df_b):
        raise ValueError(
            f"Row count mismatch — {name_a}: {len(df_a)} rows, {name_b}: {len(df_b)} rows. "
            f"Files must be in identical row order."
        )


def load_raw(cfg: dict):
    """
    Mode A: features_path set -> one CSV; latent columns contain data.latent_substring.
    Mode B: manual_features_path + latent_embeddings_path -> two CSVs.
    Returns manual_df, latent_df, labels_df (all reset_index).
    """
    d = cfg['data']
    features_path = d.get('features_path', 'na')
    manual_path = d.get('manual_features_path', 'na')
    latent_path = d.get('latent_embeddings_path', 'na')
    sub = str(d.get('latent_substring', 'latent')).lower()

    print(f"Loading labels : {d['labels_path']}")
    labels_df = pd.read_csv(d['labels_path'])
    print(f"  Labels shape : {labels_df.shape}")

    if features_path not in (None, 'na'):
        print("\nMode A — combined features file")
        full_df = pd.read_csv(features_path)
        print(f"  Features shape : {full_df.shape}")
        _check_row_alignment(full_df, labels_df, 'features', 'labels')

        latent_cols = [c for c in full_df.columns if sub in c.lower()]
        manual_cols = [c for c in full_df.columns if sub not in c.lower()]
        manual_df = full_df[manual_cols].reset_index(drop=True)
        latent_df = full_df[latent_cols].reset_index(drop=True)
        print(f"  Identified {len(manual_cols)} manual cols, {len(latent_cols)} latent cols")

    elif manual_path not in (None, 'na') and latent_path not in (None, 'na'):
        print("\nMode B — separated feature files")
        manual_df = pd.read_csv(manual_path).reset_index(drop=True)
        latent_df = pd.read_csv(latent_path).reset_index(drop=True)
        print(f"  Manual shape : {manual_df.shape}\n  Latent shape : {latent_df.shape}")
        _check_row_alignment(manual_df, labels_df, 'manual features', 'labels')
        _check_row_alignment(latent_df, labels_df, 'latent features', 'labels')
    else:
        raise ValueError(
            "Invalid data config: set either 'features_path' (Mode A) "
            "or both 'manual_features_path' and 'latent_embeddings_path' (Mode B)."
        )

    return manual_df, latent_df, labels_df.reset_index(drop=True)


# ── 2. Standardise labels ──────────────────────────────────────────────────────
def standardise_labels(labels_df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """
    Lowercase columns, validate required columns, enforce dtypes.
    """
    labels_df = labels_df.copy()
    original = labels_df.columns.tolist()
    labels_df.columns = [c.lower().strip() for c in labels_df.columns]

    if 'well' in labels_df.columns and 'well_name' not in labels_df.columns:
        labels_df = labels_df.rename(columns={'well': 'well_name'})
        print("  Renamed 'well' column to 'well_name'")

    renamed = {o: n for o, n in zip(original, labels_df.columns) if o != n}
    if renamed:
        print(f"\nColumn names lowercased: {renamed}")

    missing = [c for c in REQUIRED_LABEL_COLS if c not in labels_df.columns]
    if missing:
        raise ValueError(
            f"Required label columns missing after lowercasing: {missing}\n"
            f"Available columns: {labels_df.columns.tolist()}"
        )

    n_null = labels_df[REQUIRED_LABEL_COLS].isna().sum()
    if n_null.any():
        raise ValueError(f"Null values in required label columns:\n{n_null[n_null > 0]}")

    labels_df['passage'] = labels_df['passage'].astype(int)
    for c in ('well_name', 'roi', 'treatment', 'repeat'):
        labels_df[c] = labels_df[c].astype(str).str.strip()
    labels_df['cell_line'] = (
        labels_df['cell_line'].astype(float).astype(int).astype(str).str.strip()
    )

    print(f"\nLabel columns validated: {labels_df.columns.tolist()}")
    print(f"  Cell lines : {sorted(labels_df['cell_line'].unique())}")
    print(f"  Passages   : {sorted(labels_df['passage'].unique())}")
    print(f"  Treatments : {sorted(labels_df['treatment'].unique())}")
    print(f"  Total cells: {len(labels_df)}")
    return labels_df


# ── 3. Line config and senescence score ────────────────────────────────────────
def infer_line_config(labels_df: pd.DataFrame, cfg: dict) -> dict:
    """
    p_min / p_senescent per line. Uses data.line_config if given, otherwise
    infers from the data (min and max observed passage; the final passage is
    assumed senescent by experimental design).
    """
    p_col = 'passage'
    lines_found = sorted(labels_df['cell_line'].unique())

    override = cfg['data'].get('line_config')
    if override:
        line_config = {}
        for k, v in override.items():
            line_config[str(k)] = {'p_min': int(v['p_min']), 'p_senescent': int(v['p_senescent'])}
        missing = set(lines_found) - set(line_config)
        if missing:
            raise ValueError(f"data.line_config is missing lines present in data: {sorted(missing)}")
        print("\nLine config taken from data.line_config (override):")
    else:
        line_config = {}
        print(f"\nLine config inferred from data (using {p_col}):")
        for line in lines_found:
            passages = labels_df.loc[labels_df['cell_line'] == line, p_col]
            line_config[line] = {'p_min': int(passages.min()), 'p_senescent': int(passages.max())}

    for line in lines_found:
        lc = line_config[line]
        if lc['p_senescent'] <= lc['p_min']:
            raise ValueError(
                f"Line {line}: p_senescent ({lc['p_senescent']}) must exceed p_min ({lc['p_min']})."
            )
        obs = sorted(labels_df.loc[labels_df['cell_line'] == line, p_col].unique().tolist())
        print(f"  Line {line}: p_min={lc['p_min']} p_senescent={lc['p_senescent']} passages={obs}")
    return line_config


def compute_senescence_scores(labels_df: pd.DataFrame, line_config: dict) -> pd.Series:
    """
    s = (passage - p_min) / (p_senescent - p_min), vectorised per line.
    """
    p_col = 'passage'
    p_min = labels_df['cell_line'].map(lambda l: line_config[l]['p_min']).astype(float)
    p_sen = labels_df['cell_line'].map(lambda l: line_config[l]['p_senescent']).astype(float)
    s = (labels_df[p_col].astype(float) - p_min) / (p_sen - p_min)

    s = s.astype(np.float32)
    print(f"\nSenescence scores computed (using {p_col}): min={s.min():.3f} max={s.max():.3f}")
    return s


# ── 4. Feature column identification ───────────────────────────────────────────
def identify_feature_columns(manual_df, latent_df, pp: dict):
    """Numeric columns only, excluding index/identifier/label columns."""
    blocklist = DEFAULT_BLOCKLIST | set(pp['extra_blocklist'])

    def pick(df, name):
        keep, dropped_nonnum, dropped_block = [], [], []
        for c in df.columns:
            if c.lower().strip() in blocklist:
                dropped_block.append(c)
            elif not pd.api.types.is_numeric_dtype(df[c]):
                dropped_nonnum.append(c)
            else:
                keep.append(c)
        if dropped_block:
            print(f"  ⚠ {name}: blocklisted cols removed: {dropped_block}")
        if dropped_nonnum:
            print(f"  ⚠ {name}: non-numeric cols removed: {dropped_nonnum}")
        return keep, dropped_block, dropped_nonnum

    manual_cols, mb, mn = pick(manual_df, 'manual')
    latent_cols, lb, ln = pick(latent_df, 'latent')
    print(f"\nNumeric features identified:\n  Manual : {len(manual_cols)}\n  Latent : {len(latent_cols)}")
    report = {'blocklisted': mb + lb, 'non_numeric': mn + ln}
    return manual_cols, latent_cols, report


# ── 5. NaN / inf handling ──────────────────────────────────────────────────────
def handle_nan_rows(manual_df, latent_df, labels_df, manual_cols, latent_cols, policy: str):
    """
    Infinite values are treated as NaN. policy='drop' removes affected rows from
    all three frames identically; policy='error' raises if any are found.
    """
    manual_df = manual_df.copy()
    latent_df = latent_df.copy()
    manual_df[manual_cols] = manual_df[manual_cols].replace([np.inf, -np.inf], np.nan)
    latent_df[latent_cols] = latent_df[latent_cols].replace([np.inf, -np.inf], np.nan)

    nan_mask = (
        manual_df[manual_cols].isna().any(axis=1) | latent_df[latent_cols].isna().any(axis=1)
    ).values
    n_dropped = int(nan_mask.sum())

    if n_dropped == 0:
        print("\n  ✔ No NaN/inf rows in selected features")
    elif policy == 'error':
        raise ValueError(f"{n_dropped} rows contain NaN/inf in selected features (nan_policy='error').")
    else:
        print(f"\n  Dropping {n_dropped} rows with NaN/inf in selected feature columns")
        keep = ~nan_mask
        manual_df = manual_df[keep].reset_index(drop=True)
        latent_df = latent_df[keep].reset_index(drop=True)
        labels_df = labels_df[keep].reset_index(drop=True)
    return manual_df, latent_df, labels_df, n_dropped


# ── 6-8. Optional filters ──────────────────────────────────────────────────────
def variance_filter(df, cols, threshold, block):
    """Remove columns with population variance (ddof=0) <= threshold."""
    var = df[cols].var(axis=0, ddof=0)
    keep = var[var > threshold].index.tolist()
    removed = [c for c in cols if c not in keep]
    print(f"  Variance filter [{block}] (threshold={threshold}): removed {len(removed)}, retained {len(keep)}")
    return keep, removed


def correlation_filter(df, cols, threshold, block):
    """Greedy: for each pair with |r| > threshold, drop the later column."""
    if len(cols) < 2:
        return list(cols), []
    corr = df[cols].corr().abs().values
    to_drop = set()
    for i in range(len(cols)):
        if i in to_drop:
            continue
        for j in range(i + 1, len(cols)):
            if j not in to_drop and corr[i, j] > threshold:
                to_drop.add(j)
    keep = [c for k, c in enumerate(cols) if k not in to_drop]
    removed = [c for k, c in enumerate(cols) if k in to_drop]
    print(f"  Correlation filter [{block}] (|r|>{threshold}): removed {len(removed)}, retained {len(keep)}")
    return keep, removed


def latent_collapse_screen(df, cols, threshold):
    """Remove latent dims with variance below threshold (posterior collapse)."""
    var = df[cols].var(axis=0)
    active = var[var >= threshold].index.tolist()
    removed = [c for c in cols if c not in active]
    if removed:
        print(f"  ⚠ Latent collapse (var < {threshold}): removed {removed}")
    else:
        print(f"  ✔ Latent collapse screen: all {len(cols)} dims active (var ≥ {threshold}); "
              f"min var = {var.min():.4f}")
    return active, removed


def apply_filters(manual_df, latent_df, manual_cols, latent_cols, pp: dict):
    report = {}
    frames = {'manual': (manual_df, manual_cols), 'latent': (latent_df, latent_cols)}
    cols = {'manual': list(manual_cols), 'latent': list(latent_cols)}

    print("\nFilters:")
    vf = pp['variance_filter']
    if vf['enabled']:
        for b in vf['blocks']:
            cols[b], rem = variance_filter(frames[b][0], cols[b], vf['threshold'], b)
            report[f'variance_{b}'] = rem
    else:
        print("  Variance filter: disabled")

    cf = pp['correlation_filter']
    if cf['enabled']:
        for b in cf['blocks']:
            cols[b], rem = correlation_filter(frames[b][0], cols[b], cf['threshold'], b)
            report[f'correlation_{b}'] = rem
    else:
        print("  Correlation filter: disabled")

    lc = pp['latent_collapse']
    if lc['enabled']:
        cols['latent'], rem = latent_collapse_screen(latent_df, cols['latent'], lc['threshold'])
        report['latent_collapse'] = rem
    else:
        print("  Latent collapse screen: disabled")

    # Always-on diagnostic: constant columns are harmful for any later scaler.
    for b, (df, _) in frames.items():
        const = [c for c in cols[b] if df[c].var(ddof=0) == 0]
        if const:
            print(f"  ⚠ {b}: {len(const)} constant column(s) retained: {const}")
            report[f'constant_retained_{b}'] = const

    print(f"  Final: {len(cols['manual'])} manual, {len(cols['latent'])} latent")
    return cols['manual'], cols['latent'], report


# ── 9. Summary and save ────────────────────────────────────────────────────────
def build_dataset_summary(labels_df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    records = []
    for (line, passage), grp in labels_df.groupby(['cell_line', 'passage']):
        n_wells = grp['well_name'].nunique()
        records.append({
            'cell_line': line,
            'passage': passage,
            's_mean': round(float(grp['s'].mean()), 4),
            'n_cells': len(grp),
            'n_wells': n_wells,
            'n_repeats': grp['repeat'].nunique(),
            'cells_per_well_mean': round(len(grp) / n_wells, 1),
        })
    return pd.DataFrame(records).sort_values(['cell_line', 'passage']).reset_index(drop=True)


def save_processed(manual_df, latent_df, labels_df, manual_cols, latent_cols,
                   summary, report, cfg, pp):
    out_dir = Path(cfg['data']['processed_dir'])
    out_dir.mkdir(parents=True, exist_ok=True)

    manual_array = manual_df[manual_cols].values.astype(np.float32)
    latent_array = latent_df[latent_cols].values.astype(np.float32)
    assert len(manual_array) == len(latent_array) == len(labels_df)

    np.save(out_dir / 'features_manual.npy', manual_array)
    np.save(out_dir / 'features_latent.npy', latent_array)
    if pp['save_feature_csv']:
        pd.DataFrame(manual_array, columns=manual_cols).to_csv(out_dir / 'features_manual.csv', index=False)
        pd.DataFrame(latent_array, columns=latent_cols).to_csv(out_dir / 'features_latent.csv', index=False)

    labels_df.to_csv(out_dir / 'labels.csv', index=False)
    summary.to_csv(out_dir / 'dataset_summary.csv', index=False)
    with open(out_dir / 'manual_cols.pkl', 'wb') as f:
        pickle.dump(manual_cols, f)
    with open(out_dir / 'latent_cols.pkl', 'wb') as f:
        pickle.dump(latent_cols, f)

    report = dict(report)
    report.update({
        'n_cells': int(len(labels_df)),
        'n_manual': len(manual_cols),
        'n_latent': len(latent_cols),
        'preprocess_config': pp,
        'line_config': cfg['data'].get('line_config_resolved'),
    })
    with open(out_dir / 'preprocess_report.json', 'w') as f:
        json.dump(report, f, indent=2, default=str)

    print(f"\nProcessed outputs saved to '{out_dir}':")
    print(f"  features_manual.npy {manual_array.shape} ({manual_array.nbytes / 1e6:.1f} MB)")
    print(f"  features_latent.npy {latent_array.shape} ({latent_array.nbytes / 1e6:.1f} MB)")
    print(f"  labels.csv ({len(labels_df)} rows, cols: {labels_df.columns.tolist()})")
    print(f"  manual_cols.pkl ({len(manual_cols)}), latent_cols.pkl ({len(latent_cols)})")
    print(f"  dataset_summary.csv ({len(summary)} groups), preprocess_report.json")
    if pp['save_feature_csv']:
        print("  features_*.csv copies written")


# ── Main ───────────────────────────────────────────────────────────────────────
def main(config_path: str):
    cfg = load_config(config_path)
    pp = resolve_preprocess_cfg(cfg)
    report = {}

    manual_df, latent_df, labels_df = load_raw(cfg)                          # 1
    labels_df = standardise_labels(labels_df, cfg)                           # 2

    line_config = infer_line_config(labels_df, cfg)                          # 3
    cfg['data']['line_config_resolved'] = line_config
    labels_df['s'] = compute_senescence_scores(labels_df, line_config)

    manual_cols, latent_cols, id_report = identify_feature_columns(          # 4
        manual_df, latent_df, pp)
    report['column_selection'] = id_report

    manual_df, latent_df, labels_df, n_nan = handle_nan_rows(                # 5
        manual_df, latent_df, labels_df, manual_cols, latent_cols, pp['nan_policy'])
    report['nan_rows_dropped'] = n_nan

    manual_cols, latent_cols, f_report = apply_filters(                      # 6-8
        manual_df, latent_df, manual_cols, latent_cols, pp)
    report['filters'] = f_report

    summary = build_dataset_summary(labels_df, cfg)                          # 9
    print(f"\nDataset summary:\n{summary.to_string(index=False)}")
    save_processed(manual_df, latent_df, labels_df, manual_cols, latent_cols,
                   summary, report, cfg, pp)
    print("\n✔ Data preparation complete.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Prepare raw data for MILAgeScore training')
    parser.add_argument('--config', type=str, default='config/config.yaml', help='Path to config.yaml')
    args = parser.parse_args()
    main(args.config)
