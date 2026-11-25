"""
src/evaluate.py

Post-training inference and evaluation for MILAgeScoreModel.

Responsibilities:
    1. Stable age score estimation per passage group — multiple bag subsamples
       are averaged to reduce variance from stochastic bag construction
    2. Per-passage score distributions for visual inspection of the
       aging trajectory
    3. Treatment group comparison — statistical tests comparing age scores
       between treated and untreated samples
    4. Full results export to CSV for thesis reporting

All inference is done with the model in eval() mode and torch.no_grad().
Scalers loaded from processed_dir are applied to any new data (treatment
samples) before inference — never refit.
"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from coral_pytorch.dataset import corn_label_from_logits
from src.scoring import rank_from_logits, score_from_logits
from scipy.stats import mannwhitneyu, spearmanr
from tqdm import tqdm


# ── Load model from checkpoint ─────────────────────────────────────────────────
def load_model_from_checkpoint(
    ckpt_path: str,
    model: nn.Module,
    device: torch.device,
) -> tuple[nn.Module, dict, object]:
    """
    Load model weights, encoder and config from a saved checkpoint.

    Returns
    -------
    model   : nn.Module — weights loaded, moved to device, set to eval()
    cfg     : dict      — config used during training
    encoder : OrdinalLabelEncoder
    """
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state'])
    model.to(device)
    model.eval()

    print(f"Model loaded from checkpoint:")
    print(f"  Epoch                  : {ckpt['epoch']}")
    print(f"  Inner-validation MAE   : {ckpt['val_metrics'].get('val/mae', 'n/a'):.4f}")
    print(f"  Inner-validation Spearman ρ : {ckpt['val_metrics'].get('val/spearman_rho', 'n/a'):.4f}")

    return model, ckpt['cfg'], ckpt['encoder']


# ── Stable age score for one group ────────────────────────────────────────────
@torch.no_grad()
def predict_group_age_score(
    model: nn.Module,
    cell_features: np.ndarray,
    well_names: np.ndarray,
    encoder,
    device: torch.device,
    cells_per_bag: int = 200,
    n_bag_samples: int = 30,
    max_well_fraction: float = 0.35,
    random_seed: int = 0,
) -> dict:
    """
    Estimate the age score for a single group (passage × cell_line or treatment)
    by averaging predictions across multiple randomly subsampled bags.

    Averaging over n_bag_samples bags reduces variance from stochastic sampling
    and produces a stable, reportable age score per sample.

    Parameters
    ----------
    cell_features  : np.ndarray (n_cells, n_features) — pre-scaled
    well_names     : np.ndarray (n_cells,) str        — for per-well capping
    encoder        : OrdinalLabelEncoder
    n_bag_samples  : int — number of bags to average over (30 default)

    Returns
    -------
    result : dict with keys:
        'mean_score'   — mean s_hat across bags ∈ [0, 1]
        'std_score'    — std of s_hat across bags (uncertainty estimate)
        'mean_rank'    — mean predicted rank
        'all_scores'   — np.ndarray of all bag-level s_hat values
        'all_ranks'    — np.ndarray of all bag-level predicted ranks
    """
    rng          = np.random.default_rng(random_seed)
    max_per_well = int(np.ceil(max_well_fraction * cells_per_bag))

    # Build well → index lookup
    unique_wells = np.unique(well_names)
    well_idx     = {w: np.where(well_names == w)[0] for w in unique_wells}

    all_scores = []
    all_ranks  = []

    for _ in range(n_bag_samples):
        # Per-well capped sampling
        pool_parts = []
        for w_idx in well_idx.values():
            n_draw = min(len(w_idx), max_per_well)
            chosen = rng.choice(len(w_idx), size=n_draw, replace=False)
            pool_parts.append(w_idx[chosen])

        pool    = np.concatenate(pool_parts)
        n_pool  = len(pool)
        replace = n_pool < cells_per_bag
        sampled = rng.choice(n_pool, size=cells_per_bag, replace=replace)
        bag_idx = pool[sampled]

        bag = torch.from_numpy(
            cell_features[bag_idx].astype(np.float32)
        ).unsqueeze(0).to(device)                          # (1, N, D)

        logits, _ = model(bag)
        # Continuous expected-value score (granular) + discrete rank (reference).
        pred_score = float(score_from_logits(logits, encoder.num_classes).item())
        pred_rank = float(rank_from_logits(logits).item())

        all_scores.append(pred_score)
        all_ranks.append(pred_rank)

    all_scores = np.array(all_scores, dtype=np.float32)
    all_ranks  = np.array(all_ranks,  dtype=np.float32)     # now continuous-capable

    return {
        'mean_score' : float(all_scores.mean()),
        'std_score'  : float(all_scores.std()),
        'mean_rank'  : float(all_ranks.mean()),
        'all_scores' : all_scores,
        'all_ranks'  : all_ranks,
    }


# ── Evaluate full training dataset (passage trajectory) ───────────────────────
@torch.no_grad()
def evaluate_passage_trajectory(
    model: nn.Module,
    features_manual: np.ndarray,
    features_latent: np.ndarray,
    labels_df: pd.DataFrame,
    encoder,
    device: torch.device,
    cfg: dict,
    w_manual: float = 1.0,
    w_latent: float = 1.0,
) -> pd.DataFrame:
    """
    Compute mean predicted age score for every passage × cell_line group.

    This is the primary validation of the aging trajectory — if the model
    has learned the ordinal aging signal, predicted scores should correlate
    strongly with true senescence scores s (Spearman ρ → 1.0).

    Returns
    -------
    trajectory_df : pd.DataFrame, one row per group, sorted by cell_line + passage
        Columns: cell_line, passage, s_true, s_hat_mean, s_hat_std,
                 n_cells, spearman_rho (per-line), mae
    """
    bags_cfg     = cfg['bags']
    cells_per_bag = bags_cfg['cells_per_bag']
    n_bag_samples = cfg['evaluate'].get('n_bag_samples', 30)
    max_well_frac = bags_cfg['max_well_fraction']

    # Apply block weights
    feats = np.concatenate([
        features_manual * w_manual,
        features_latent * w_latent
    ], axis=1).astype(np.float32)

    records = []
    # Group by cell_line and passage (or passage_raw if present)
    group_cols = ['cell_line', 'passage']
    if 'passage_raw' in labels_df.columns:
        group_cols = ['cell_line', 'passage_raw']
    
    groups = labels_df.groupby(group_cols)

    for group_vals, grp in tqdm(groups, desc="Evaluating passage trajectory"):
        if len(group_cols) == 2:
            cell_line, passage_val = group_vals
        else:
            cell_line = group_vals
            passage_val = grp['passage'].iloc[0]
        
        idx        = grp.index.values
        group_feats = feats[idx]
        well_names  = grp['well_name'].values
        s_true      = float(grp['s'].iloc[0])
        passage_raw = grp['passage_raw'].iloc[0] if 'passage_raw' in grp.columns else passage_val

        # Seed selection for reproduction
        # TASK 1: "Use a different RNG seed per group." 
        # Deterministic but different per group: use hash of group ID + random_seed
        group_seed = int(pd.util.hash_pandas_object(pd.Series([cell_line, passage_raw])).iloc[0] % (2**32))
        eval_seed = cfg.get('bags', {}).get('random_seed', 42) + group_seed

        result = predict_group_age_score(
            model=model,
            cell_features=group_feats,
            well_names=well_names,
            encoder=encoder,
            device=device,
            cells_per_bag=cells_per_bag,
            n_bag_samples=n_bag_samples,
            max_well_fraction=max_well_frac,
            random_seed=eval_seed
        )

        records.append({
            'cell_line'  : cell_line,
            'passage'    : grp['passage'].iloc[0],
            'passage_raw': passage_raw,
            's_true'     : round(s_true, 4),
            's_hat_mean' : round(result['mean_score'], 4),
            's_hat_std'  : round(result['std_score'],  4),
            's_hat_q05'  : round(float(np.percentile(result['all_scores'], 5)), 4),
            's_hat_q95'  : round(float(np.percentile(result['all_scores'], 95)), 4),
            'rank_mean'  : round(result['mean_rank'],  3),
            'n_cells'    : len(idx),
            'all_scores' : result['all_scores'],
            'all_ranks'  : result['all_ranks'],
        })

    trajectory_df = pd.DataFrame(records).sort_values(
        ['cell_line', 'passage_raw']
    ).reset_index(drop=True)

    # Compute Spearman ρ across all groups
    rho, pval = spearmanr(trajectory_df['s_true'], trajectory_df['s_hat_mean'])
    mae        = float(np.mean(np.abs(
        trajectory_df['s_hat_mean'] - trajectory_df['s_true']
    )))

    print(f"\nHeld-out line Trajectory Evaluation:")
    print(f"  Spearman ρ : {rho:.4f}  (p={pval:.4f})")
    print(f"  MAE (s)    : {mae:.4f}")
    # Print a summary table without the large array columns
    display_df = trajectory_df.drop(columns=['all_scores', 'all_ranks'])
    print(f"\n{display_df.to_string(index=False)}")

    return trajectory_df


# ── Treatment group comparison ─────────────────────────────────────────────────
@torch.no_grad()
def compare_treatment_groups(
    model: nn.Module,
    features_manual: np.ndarray,
    features_latent: np.ndarray,
    labels_df: pd.DataFrame,
    encoder,
    device: torch.device,
    cfg: dict,
    w_manual: float = 1.0,
    w_latent: float = 1.0,
) -> pd.DataFrame:
    """
    Compute age scores per unique treatment group and compare them.

    For each treatment group present in labels_df, the age score is computed
    as the mean predicted s_hat across n_bag_samples bags. Treatment groups
    are compared pairwise using the Mann-Whitney U test (non-parametric,
    appropriate for small group sizes) on the distribution of per-bag scores.

    This is the core analysis for anti-senescence treatment effectiveness:
    a treatment that rejuvenates cells should produce a lower mean s_hat
    than an untreated control imaged at the same passage.

    Returns
    -------
    results_df : pd.DataFrame — one row per treatment group
        Columns: treatment, passage, cell_line, n_cells,
                 s_hat_mean, s_hat_std, s_hat_median
    comparisons_df : pd.DataFrame — pairwise Mann-Whitney results
    """
    bags_cfg      = cfg['bags']
    cells_per_bag = bags_cfg['cells_per_bag']
    n_bag_samples = cfg['evaluate'].get('n_bag_samples', 30)
    max_well_frac = bags_cfg['max_well_fraction']

    feats = np.concatenate([
        features_manual * w_manual,
        features_latent * w_latent
    ], axis=1).astype(np.float32)

    records    = []
    group_bags = {}   # treatment → all_scores array for pairwise testing

    groups = labels_df.groupby(['treatment', 'passage', 'cell_line'])

    for (treatment, passage, cell_line), grp in tqdm(
        groups, desc="Scoring treatment groups"
    ):
        idx         = grp.index.values
        group_feats = feats[idx]
        well_names  = grp['well_name'].values

        result = predict_group_age_score(
            model=model,
            cell_features=group_feats,
            well_names=well_names,
            encoder=encoder,
            device=device,
            cells_per_bag=cells_per_bag,
            n_bag_samples=n_bag_samples,
            max_well_fraction=max_well_frac,
        )

        records.append({
            'treatment'  : treatment,
            'passage'    : passage,
            'cell_line'  : cell_line,
            'n_cells'    : len(idx),
            's_hat_mean' : round(result['mean_score'],  4),
            's_hat_std'  : round(result['std_score'],   4),
            's_hat_median': round(float(np.median(result['all_scores'])), 4),
        })
        group_bags[f"{treatment}|P{passage}|L{cell_line}"] = result['all_scores']

    results_df = pd.DataFrame(records).sort_values(
        ['cell_line', 'passage', 'treatment']
    ).reset_index(drop=True)

    # Pairwise Mann-Whitney U tests
    comparisons = []
    keys = list(group_bags.keys())
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            a, b   = group_bags[keys[i]], group_bags[keys[j]]
            stat, p = mannwhitneyu(a, b, alternative='two-sided')
            comparisons.append({
                'group_a'   : keys[i],
                'group_b'   : keys[j],
                'mw_stat'   : round(float(stat), 2),
                'p_value'   : round(float(p), 6),
                'mean_a'    : round(float(a.mean()), 4),
                'mean_b'    : round(float(b.mean()), 4),
                'delta_s'   : round(float(b.mean() - a.mean()), 4),
            })

    comparisons_df = pd.DataFrame(comparisons)
    print(f"\nTreatment group age scores:\n{results_df.to_string(index=False)}")

    return results_df, comparisons_df


# ── Save evaluation results ────────────────────────────────────────────────────
def save_eval_results(
    trajectory_df: pd.DataFrame,
    results_df: pd.DataFrame,
    comparisons_df: pd.DataFrame,
    cfg: dict,
    run_name: str = 'eval',
):
    out_dir = Path(cfg['outputs']['log_dir'])
    out_dir.mkdir(parents=True, exist_ok=True)

    trajectory_df.to_csv( out_dir / f'{run_name}_trajectory.csv',   index=False)
    if results_df is not None:
        results_df.to_csv(    out_dir / f'{run_name}_treatments.csv',    index=False)
    if comparisons_df is not None:
        comparisons_df.to_csv(out_dir / f'{run_name}_comparisons.csv',   index=False)

    print(f"\nEvaluation results saved to {out_dir}")
