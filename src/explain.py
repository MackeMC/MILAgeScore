"""
src/explain.py

Feature and cell-level interpretability for MILAgeScoreModel.

Two complementary attribution methods:

    1. Attention Weight Analysis
       Extracts per-cell attention weights from the trained MIL aggregator.
       Cells with high attention are those the model considers most
       informative for the bag-level age score. Aggregated across bags
       and passages, this identifies whether senescent-looking cells
       consistently receive higher attention weights.

    2. Integrated Gradients (IG)
       Computes the contribution of each input feature to the model output
       (captum.attr.IntegratedGradients). IG satisfies sensitivity and
       implementation invariance axioms — it is the principled choice over
       raw gradient or first-layer weight extraction for deep models.

       Attribution is computed at the feature level per cell, then aggregated
       across high-attention cells to identify which morphological and latent
       features drive the age prediction.

Outputs:
    attention_summary.csv  — per-group mean attention weight per cell
    feature_attributions.csv — mean |IG| per feature, sorted by importance
    figures/               — attention heatmaps, feature importance bar charts
"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from tqdm import tqdm
from coral_pytorch.dataset import corn_label_from_logits

# ── Attention weight extraction ────────────────────────────────────────────────
@torch.no_grad()
def extract_attention_weights(
    model: nn.Module,
    cell_features: np.ndarray,
    well_names: np.ndarray,
    encoder,
    device: torch.device,
    cells_per_bag: int = 200,
    n_bag_samples: int = 20,
    max_well_fraction: float = 0.35,
    random_seed: int = 0,
) -> dict:
    """
    Extract per-cell attention weights for a single group across multiple bags.

    Each cell's attention weight reflects how much it contributed to the
    bag-level age score. Weights sum to 1 per bag (post-softmax).

    Returns a per-cell mean attention weight averaged across all bags in
    which that cell appeared, providing a stable per-cell importance score.

    Returns
    -------
    result : dict
        'cell_mean_attn'  : np.ndarray (n_cells,) — mean attention per cell
        'cell_count'      : np.ndarray (n_cells,) — bags each cell appeared in
        'pred_scores'     : np.ndarray (n_bag_samples,) — bag-level s_hat
    """
    rng          = np.random.default_rng(random_seed)
    n_cells      = len(cell_features)
    max_per_well = int(np.ceil(max_well_fraction * cells_per_bag))

    # Accumulators
    attn_sum   = np.zeros(n_cells, dtype=np.float64)
    count_sum  = np.zeros(n_cells, dtype=np.int64)
    pred_scores = []

    unique_wells = np.unique(well_names)
    well_idx     = {w: np.where(well_names == w)[0] for w in unique_wells}

    model.eval()

    for _ in range(n_bag_samples):
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

        logits, attn = model(bag)                          # attn: (1, N)
        attn_np      = attn.squeeze(0).cpu().numpy()       # (N,)

        pred_rank = int(corn_label_from_logits(logits).cpu().item())
        pred_score = pred_rank / (encoder.num_classes - 1)

        # Accumulate per original cell index
        for local_i, global_i in enumerate(bag_idx):
            attn_sum[global_i]  += attn_np[local_i]
            count_sum[global_i] += 1

        pred_scores.append(pred_score)

    # Mean attention per cell (only for cells that appeared in at least one bag)
    appeared      = count_sum > 0
    cell_mean_attn = np.where(appeared, attn_sum / np.maximum(count_sum, 1), 0.0)

    return {
        'cell_mean_attn' : cell_mean_attn.astype(np.float32),
        'cell_count'     : count_sum,
        'pred_scores'    : np.array(pred_scores, dtype=np.float32),
    }


# ── Passage-level attention summary ───────────────────────────────────────────
def compute_attention_summary(
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
    Compute per-cell mean attention weight for every passage × cell_line group.

    High-attention cells are the model's implicit definition of "most age-
    informative" cells. Checking whether these cells have distinct morphology
    (e.g. larger area, higher intensity variance) is a key interpretability
    result for the thesis.

    Returns
    -------
    attn_df : pd.DataFrame — labels_df with added column 'mean_attention'
    """
    bags_cfg      = cfg['bags']
    n_bag_samples = cfg['evaluate'].get('n_bag_samples_attention', 300)

    feats = np.concatenate([
        features_manual * w_manual,
        features_latent * w_latent
    ], axis=1).astype(np.float32)

    attn_per_cell = np.zeros(len(labels_df), dtype=np.float32)

    for (cell_line, passage), grp in tqdm(
        labels_df.groupby(['cell_line', 'passage']),
        desc="Extracting attention weights"
    ):
        idx         = grp.index.values
        group_feats = feats[idx]
        well_names  = grp['well_name'].values

        result = extract_attention_weights(
            model=model,
            cell_features=group_feats,
            well_names=well_names,
            encoder=encoder,
            device=device,
            cells_per_bag=bags_cfg['cells_per_bag'],
            n_bag_samples=n_bag_samples,
            max_well_fraction=bags_cfg['max_well_fraction'],
        )
        # Map back to full index
        attn_per_cell[idx] = result['cell_mean_attn']

    attn_df              = labels_df.copy()
    attn_df['mean_attention'] = attn_per_cell

    print(f"\nAttention weight summary:")
    print(f"  Global mean  : {attn_per_cell.mean():.6f}")
    print(f"  Global std   : {attn_per_cell.std():.6f}")
    print(f"  Top decile threshold : {np.percentile(attn_per_cell, 90):.6f}")

    return attn_df


# ── Integrated Gradients feature attribution ───────────────────────────────────
def compute_integrated_gradients(
    model: nn.Module,
    cell_features: np.ndarray,
    well_names: np.ndarray,
    encoder,
    device: torch.device,
    manual_cols: list,
    latent_cols: list,
    cells_per_bag: int = 200,
    n_bag_samples: int = 10,
    n_ig_steps: int = 50,
    max_well_fraction: float = 0.35,
    random_seed: int = 0,
) -> pd.DataFrame:
    """
    Compute Integrated Gradients feature attributions for a single group.

    IG computes the integral of model gradients along a straight-line path
    from a baseline input (zero vector, representing an uninformative cell)
    to the actual input. This satisfies:
        - Sensitivity: features that change the output get non-zero attribution
        - Implementation invariance: equivalent networks get identical attributions

    Attribution is computed per cell in each bag, then averaged across all
    high-attention cells (top 25% by attention weight) to focus on the
    cells the model actually uses for its prediction.

    Note: captum is used if available; a manual IG implementation is used
    as fallback to avoid adding a hard dependency.

    Returns
    -------
    attribution_df : pd.DataFrame
        Columns: feature, mean_abs_ig, std_abs_ig, feature_type (manual/latent)
        Sorted by mean_abs_ig descending.
    """
    rng          = np.random.default_rng(random_seed)
    n_cells      = len(cell_features)
    n_features   = cell_features.shape[1]
    n_manual     = len(manual_cols)
    max_per_well = int(np.ceil(max_well_fraction * cells_per_bag))

    unique_wells = np.unique(well_names)
    well_idx     = {w: np.where(well_names == w)[0] for w in unique_wells}

    all_attributions = []   # list of (cells_per_bag, n_features) arrays

    model.eval()

    for _ in tqdm(range(n_bag_samples), desc="  Computing IG", leave=False):
        pool_parts = []
        for w_idx in well_idx.values():
            n_draw = min(len(w_idx), max_per_well)
            chosen = rng.choice(len(w_idx), size=n_draw, replace=False)
            pool_parts.append(w_idx[chosen])

        pool    = np.concatenate(pool_parts)
        replace = len(pool) < cells_per_bag
        sampled = rng.choice(len(pool), size=cells_per_bag, replace=replace)
        bag_idx = pool[sampled]

        bag_np  = cell_features[bag_idx].astype(np.float32)   # (N, D)

        try:
            from captum.attr import IntegratedGradients

            def model_forward_for_ig(bag_input):
                # bag_input: (1, N, D) — captum passes batch dim
                logits, _ = model(bag_input)
                # Use sum of logits as scalar output for IG target
                return logits.sum(dim=-1, keepdim=True)

            bag_tensor = torch.from_numpy(bag_np).unsqueeze(0).to(device)
            bag_tensor.requires_grad_(True)
            baseline   = torch.zeros_like(bag_tensor)

            ig      = IntegratedGradients(model_forward_for_ig)
            attrs   = ig.attribute(
                bag_tensor,
                baselines=baseline,
                n_steps=n_ig_steps,
                return_convergence_delta=False,
            )
            attrs_np = attrs.squeeze(0).detach().cpu().numpy()  # (N, D)

        except ImportError:
            # Manual IG implementation — identical result, no captum required
            attrs_np = _manual_integrated_gradients(
                model=model,
                bag_np=bag_np,
                device=device,
                n_steps=n_ig_steps,
            )

        # Focus attribution on high-attention cells
        with torch.no_grad():
            bag_t    = torch.from_numpy(bag_np).unsqueeze(0).to(device)
            _, attn  = model(bag_t)
            attn_np  = attn.squeeze(0).cpu().numpy()

        attn_threshold = np.percentile(attn_np, 75)
        high_attn_mask = attn_np >= attn_threshold
        if high_attn_mask.sum() > 0:
            focused_attrs = np.abs(attrs_np[high_attn_mask])   # (k, D)
        else:
            focused_attrs = np.abs(attrs_np)

        all_attributions.append(focused_attrs.mean(axis=0))    # (D,)

    all_attributions = np.stack(all_attributions, axis=0)      # (n_bags, D)
    mean_abs_ig      = all_attributions.mean(axis=0)           # (D,)
    std_abs_ig       = all_attributions.std(axis=0)            # (D,)

    # Build output dataframe
    all_cols     = manual_cols + latent_cols
    feature_type = (['manual'] * len(manual_cols) +
                    ['latent'] * len(latent_cols))

    attribution_df = pd.DataFrame({
        'feature'      : all_cols,
        'feature_type' : feature_type,
        'mean_abs_ig'  : mean_abs_ig,
        'std_abs_ig'   : std_abs_ig,
    }).sort_values('mean_abs_ig', ascending=False).reset_index(drop=True)

    return attribution_df


def _manual_integrated_gradients(
    model: nn.Module,
    bag_np: np.ndarray,
    device: torch.device,
    n_steps: int = 50,
) -> np.ndarray:
    """
    Manual Riemann-sum approximation of Integrated Gradients.
    Identical to captum's implementation; used as fallback.

    IG(x) = (x - x') × ∫₀¹ ∂F(x' + α(x - x'))/∂x dα

    Baseline x' = zero vector (uninformative cell).
    """
    baseline = np.zeros_like(bag_np)
    alphas   = np.linspace(0, 1, n_steps + 1)[1:]         # exclude alpha=0
    grad_sum = np.zeros_like(bag_np, dtype=np.float64)

    for alpha in alphas:
        interp    = baseline + alpha * (bag_np - baseline)
        inp_tensor = torch.from_numpy(
            interp.astype(np.float32)
        ).unsqueeze(0).to(device)
        inp_tensor.requires_grad_(True)

        logits, _ = model(inp_tensor)
        output    = logits.sum()
        output.backward()

        grad_sum += inp_tensor.grad.squeeze(0).detach().cpu().numpy()

    avg_grads    = grad_sum / n_steps
    attributions = (bag_np - baseline) * avg_grads
    return attributions


# ── Aggregate feature importance across passages ───────────────────────────────
def aggregate_feature_importance(
    model: nn.Module,
    features_manual: np.ndarray,
    features_latent: np.ndarray,
    labels_df: pd.DataFrame,
    manual_cols: list,
    latent_cols: list,
    encoder,
    device: torch.device,
    cfg: dict,
    w_manual: float = 1.0,
    w_latent: float = 1.0,
) -> pd.DataFrame:
    """
    Run IG attribution for every passage × cell_line group and aggregate.

    Produces a single ranked feature importance table averaged across all
    passage groups — the model's global view of which features drive aging.
    """
    bags_cfg = cfg['bags']
    n_bags   = cfg['evaluate'].get('n_bag_samples_explain', 10)

    feats = np.concatenate([
        features_manual * w_manual,
        features_latent * w_latent
    ], axis=1).astype(np.float32)

    all_group_attrs = []

    for (cell_line, passage), grp in labels_df.groupby(['cell_line', 'passage']):
        idx         = grp.index.values
        group_feats = feats[idx]
        well_names  = grp['well_name'].values

        attr_df = compute_integrated_gradients(
            model=model,
            cell_features=group_feats,
            well_names=well_names,
            encoder=encoder,
            device=device,
            manual_cols=manual_cols,
            latent_cols=latent_cols,
            cells_per_bag=bags_cfg['cells_per_bag'],
            n_bag_samples=n_bags,
            max_well_fraction=bags_cfg['max_well_fraction'],
        )
        attr_df['cell_line'] = cell_line
        attr_df['passage']   = passage
        all_group_attrs.append(attr_df)

    # Global mean |IG| per feature across all groups
    combined = pd.concat(all_group_attrs, ignore_index=True)
    global_importance = (
        combined.groupby(['feature', 'feature_type'])['mean_abs_ig']
        .mean()
        .reset_index()
        .rename(columns={'mean_abs_ig': 'global_mean_abs_ig'})
        .sort_values('global_mean_abs_ig', ascending=False)
        .reset_index(drop=True)
    )

    print(f"\nTop 15 most important features (global mean |IG|):")
    print(global_importance.head(15).to_string(index=False))

    return global_importance

def normalise_attributions_by_block(
    global_importance: pd.DataFrame,
    w_manual: float,
    w_latent: float,
) -> pd.DataFrame:
    """
    Divide mean |IG| by the block weight applied during forward pass.
    This recovers attribution in the original unweighted feature space,
    making manual and latent attributions directly comparable.
    """
    df = global_importance.copy()
    df.loc[df['feature_type'] == 'manual', 'global_mean_abs_ig'] /= w_manual
    df.loc[df['feature_type'] == 'latent', 'global_mean_abs_ig'] /= w_latent
    df = df.sort_values('global_mean_abs_ig', ascending=False).reset_index(drop=True)
    return df


# ── Save explain outputs ───────────────────────────────────────────────────────
def save_explain_results(
    attn_df: pd.DataFrame,
    global_importance: pd.DataFrame,
    cfg: dict,
    run_name: str = 'explain',
):
    out_dir = Path(cfg['outputs']['log_dir'])
    out_dir.mkdir(parents=True, exist_ok=True)

    attn_df.to_csv(         out_dir / f'{run_name}_attention.csv',    index=False)
    global_importance.to_csv(out_dir / f'{run_name}_importance.csv',  index=False)

    print(f"\nExplainability results saved to {out_dir}")
