"""
src/split_loco.py

Leave-one-cell-line-out (LOCO) validation split for MILAgeScore.

────────────────────────────────────────────────────────────────────────────
Why this exists
────────────────────────────────────────────────────────────────────────────
The default `split_by_group` (in dataset.py) holds out whole (cell_line,
passage) groups but keeps BOTH cell lines in train AND val. Because the label
s is a deterministic function of passage within a line, that split only tests
interpolation along a passage axis the model has already seen for each line —
it cannot demonstrate generalisation to unseen biology.

LOCO trains on every line EXCEPT one and validates on the entirely held-out
line. With two lines this is the strongest generalisation test available:
train on line 1 → predict line 2 (and vice-versa). Expect Spearman ρ to drop
relative to the within-line split; that drop is the honest generalisation
number. Leave-one-group-out is the standard scheme for batch/donor-confounded
biological data (scikit-learn, "LeaveOneGroupOut").

────────────────────────────────────────────────────────────────────────────
The encoder subtlety (critical)
────────────────────────────────────────────────────────────────────────────
The two lines have different passage spans (line 1: P4–P15; line 2: P4–P18),
so their normalised s-grids differ. The OrdinalLabelEncoder maps each unique
s-value to a rank. If the encoder were built only on the TRAINING line, the
held-out line's s-values would be out-of-vocabulary and `encode()` would raise.

Therefore, under LOCO the encoder MUST be built on the FULL dataset's s-values
(both lines), even though only one line is used for training. `build_loco_encoder`
does exactly this. The model still only ever trains on one line, but the rank
vocabulary spans both lines so validation labels are always in-vocabulary.

A consequence: because rank indices are not aligned across lines (the same
rank index can mean different absolute passages), report the held-out metric in
CONTINUOUS s-space (Spearman ρ and MAE on s_true vs s_hat), not in rank space.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ── Leave-one-cell-line-out index split ─────────────────────────────────────────
def split_leave_one_line_out(
    labels_df: pd.DataFrame,
    held_out_line,
    random_seed: int = 42, # accepted for signature parity with split_by_group; unused
    line_column: str = 'cell_line',
    inner_val: bool = False,
    inner_val_seed: int = 42,
    ) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Train on all cell lines except `held_out_line`; validate on `held_out_line`.

    Drop-in replacement for `split_by_group` — returns the same
    (train_idx, val_idx) tuple of positional row indices into the
    features/labels arrays.

    If inner_val is True, the training set is further split: one well per
    (cell_line, passage) group is held out for inner validation.

    Parameters
    ----------
    labels_df     : pd.DataFrame — must contain the `line_column`. Note that
                    after standardise_labels the cell_line column is a STRING
                    ('1', '2'), so pass held_out_line accordingly (e.g. '2').
    held_out_line : the cell line to hold out for validation. Compared as a
                    string to match the standardised column dtype; ints are
                    coerced (2 -> '2') for convenience.
    random_seed   : unused — LOCO is deterministic. Present only so this can be
                    swapped in wherever split_by_group(..., random_seed=...) is
                    called.
    line_column   : the column used to identify cell lines (default: 'cell_line').
    inner_val     : if True, split the training lines into (train, inner_val) by
                    holding out one well per (cell_line, passage) group.
    inner_val_seed: seed for the inner validation well selection.

    Returns
    -------
    If inner_val is False:
        train_idx : np.ndarray of int — positional indices for the training lines
        val_idx   : np.ndarray of int — positional indices for the held-out line
    If inner_val is True:
        train_idx : np.ndarray of int — training-line indices MINUS inner-val wells
        inner_idx : np.ndarray of int — one well per (line, passage) from training lines
        val_idx   : np.ndarray of int — positional indices for the held-out line
        inner_wells_df : pd.DataFrame — record of which well was held out per group
    """
    if line_column not in labels_df.columns:
        raise ValueError(f"labels_df must contain a '{line_column}' column.")

    # Coerce to string to match the standardised column dtype ('1', '2').
    line_series = labels_df[line_column].astype(str).str.strip()
    held        = str(held_out_line).strip()

    present = sorted(line_series.unique())
    if held not in present:
        raise ValueError(
            f"held_out_line='{held}' not found in {line_column}. "
            f"Available values: {present}"
        )
    if len(present) < 2:
        raise ValueError(
            f"LOCO requires >= 2 lines in '{line_column}'; found only {present}. "
            f"With a single line there is nothing to hold out."
        )

    val_mask  = (line_series == held).values
    val_idx   = np.where(val_mask)[0]
    train_total_idx = np.where(~val_mask)[0]

    if not inner_val:
        train_lines = sorted(line_series[~val_mask].unique())
        print(f"\nLeave-one-cell-line-out split:")
        print(f"  Held-out (val) line : {held}  ({len(val_idx)} cells)")
        print(f"  Train lines         : {train_lines}  ({len(train_total_idx)} cells)")
        return train_total_idx, val_idx

    # Inner validation: within training lines, hold out one well per group
    train_df = labels_df.iloc[train_total_idx].copy()
    train_df['original_index'] = train_total_idx

    inner_indices = []
    inner_wells_record = []
    rng = np.random.default_rng(inner_val_seed)

    # Group by cell_line and passage (or passage_raw if present)
    group_cols = [line_column, 'passage']
    if 'passage_raw' in train_df.columns:
        group_cols = [line_column, 'passage_raw']

    for (line, passage), group in train_df.groupby(group_cols):
        wells = sorted(group['well_name'].unique())
        chosen_well = rng.choice(wells)
        well_idx = group[group['well_name'] == chosen_well]['original_index'].values
        inner_indices.extend(well_idx)
        inner_wells_record.append({
            'cell_line': line,
            'passage_raw': passage, # might be binned 'passage' or real 'passage_raw'
            'well': chosen_well
        })

    inner_idx = np.array(inner_indices)
    train_idx = np.array([i for i in train_total_idx if i not in inner_idx])
    inner_wells_df = pd.DataFrame(inner_wells_record)

    train_lines = sorted(line_series[~val_mask].unique())
    print(f"\nLeave-one-cell-line-out split with INNER VALIDATION:")
    print(f"  Held-out (loco) line : {held}  ({len(val_idx)} cells)")
    print(f"  Inner-val (wells)    : {len(inner_idx)} cells from {len(inner_wells_df)} wells")
    print(f"  Train (remaining)    : {len(train_idx)} cells from {train_lines}")

    return train_idx, inner_idx, val_idx, inner_wells_df


# ── Full-vocabulary encoder for LOCO ─────────────────────────────────────────────
def build_loco_encoder(labels_df: pd.DataFrame):
    """
    Build an OrdinalLabelEncoder on the FULL dataset's s-values (both lines).

    Under LOCO the model trains on a single line, but the rank vocabulary must
    span ALL lines so the held-out line's s-values are in-vocabulary at
    validation time. Building the encoder on the training subset alone would
    make held-out s-values raise in `encode()`.

    Returns
    -------
    encoder : OrdinalLabelEncoder built on labels_df['s'].unique() over all rows.
    """
    # Imported here to avoid a hard module-level dependency / circular import.
    from src.losses import OrdinalLabelEncoder

    if 's' not in labels_df.columns:
        raise ValueError(
            "labels_df has no 's' column — run prepare_data so senescence "
            "scores are attached before building the LOCO encoder."
        )

    s_values = labels_df['s'].unique()   # FULL dataset, both lines
    encoder  = OrdinalLabelEncoder(s_values)
    print(f"  LOCO encoder built on FULL-dataset s-vocabulary: "
          f"{encoder.num_classes} ranks (spans all lines)")
    return encoder


# ── Continuous-space held-out metrics ────────────────────────────────────────────
def loco_metrics(s_true: np.ndarray, s_hat: np.ndarray) -> dict:
    """
    Held-out metrics computed in CONTINUOUS s-space (not rank space).

    Rank indices are not comparable across lines under LOCO, so the honest
    held-out summary is Spearman ρ and MAE between true and predicted s.

    Parameters
    ----------
    s_true : np.ndarray — true senescence scores for the held-out line
    s_hat  : np.ndarray — predicted continuous scores (e.g. from
             src.scoring.score_from_logits)

    Returns
    -------
    dict with 'loco/spearman_rho', 'loco/spearman_p', 'loco/mae_s', 'loco/n'.
    """
    from scipy.stats import spearmanr

    s_true = np.asarray(s_true, dtype=np.float64).ravel()
    s_hat  = np.asarray(s_hat,  dtype=np.float64).ravel()

    if s_true.shape != s_hat.shape:
        raise ValueError(
            f"s_true and s_hat shape mismatch: {s_true.shape} vs {s_hat.shape}"
        )

    # spearmanr returns nan if either input is constant; guard explicitly.
    if np.unique(s_true).size < 2 or np.unique(s_hat).size < 2:
        rho, p = float('nan'), float('nan')
    else:
        rho, p = spearmanr(s_true, s_hat)

    return {
        'loco/spearman_rho': float(rho),
        'loco/spearman_p'  : float(p),
        'loco/mae_s'       : float(np.mean(np.abs(s_hat - s_true))),
        'loco/n'           : int(s_true.size),
    }


# ── Convenience: iterate every LOCO fold ─────────────────────────────────────────
def iter_loco_folds(labels_df: pd.DataFrame):
    """
    Yield (held_out_line, train_idx, val_idx) for every cell line in turn.

    With two lines this yields two folds: hold out '1', then hold out '2'.
    Use to run a full leave-one-line-out cross-validation and report mean ± SD
    of the continuous-space metrics across folds.
    """
    line_col = labels_df['cell_line'].astype(str).str.strip()
    for line in sorted(line_col.unique()):
        train_idx, val_idx = split_leave_one_line_out(labels_df, line)
        yield line, train_idx, val_idx
