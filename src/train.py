"""
src/train.py

Training and validation loops for MILAgeScoreModel.

Responsibilities:
    - One epoch train loop with gradient clipping and wandb batch logging
    - One epoch validation loop with full metric computation
    - Early stopping with configurable patience
    - Checkpoint saving (best val MAE and final epoch)
    - Learning rate scheduling (ReduceLROnPlateau on val MAE)
    - Full run orchestration via train() entry point

Scoring is delegated to src.scoring (the single source of truth shared with
evaluate.py and the analysis notebook):
    - rank_from_logits_np  : discrete CORN rank (for rank-space MAE / MZOE)
    - score_from_logits_np : continuous expected-value score s_hat ∈ [0, 1]

LOCO (leave-one-cell-line-out) support
--------------------------------------
When cfg['validation']['scheme'] == 'loco', rank indices are NOT comparable
across cell lines, so the HONEST held-out metric is reported in continuous
s-space: Spearman ρ and MAE_s between true and predicted s. These are computed
via src.split_loco.loco_metrics and logged under the 'loco/*' namespace, and
the run additionally early-stops on continuous-s MAE rather than rank MAE.

All model, data, loss, and config objects are passed in explicitly.
"""

import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import wandb
from tqdm import tqdm

from src.losses import compute_metrics, compute_score_metrics
from src.scoring import rank_from_logits_np, score_from_logits_np
from src.split_loco import loco_metrics


# ── Single training epoch ──────────────────────────────────────────────────────
def train_one_epoch(
    model: nn.Module,
    loader,
    criterion,
    optimizer: torch.optim.Optimizer,
    encoder,
    device: torch.device,
    epoch: int,
    cfg: dict,
) -> dict:
    """
    Run one full pass over the training DataLoader.

    Gradient clipping is applied before each optimiser step to prevent
    exploding gradients — common in attention mechanisms when early
    attention weights are near-uniform and gradients flow through
    the softmax.

    Returns
    -------
    epoch_metrics : dict — averaged metrics across all batches
    """
    model.train()
    max_grad_norm = cfg['training'].get('max_grad_norm', 1.0)

    running = {
        'loss/corn'  : [],
        'loss/aux'   : [],
        'loss/total' : [],
        'mae'        : [],
    }

    pbar = tqdm(loader, desc=f"Epoch {epoch:03d} [train]", leave=False)

    for batch_idx, (bags, s_labels, _) in enumerate(pbar):
        bags     = bags.to(device, non_blocking=True)       # (B, N, D)
        s_labels = s_labels.to(device, non_blocking=True)   # (B,)

        optimizer.zero_grad(set_to_none=True)

        logits, attn = model(bags)                          # (B, K-1), (B, N)
        loss, loss_components = criterion(logits, s_labels)

        loss.backward()

        # Gradient clipping — guards attention softmax instability
        nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

        optimizer.step()

        # Batch-level MAE for progress monitoring
        with torch.no_grad():
            pred_ranks = rank_from_logits_np(logits)
            true_ranks = encoder.encode(s_labels.cpu()).numpy()
            batch_mae  = float(np.mean(np.abs(pred_ranks - true_ranks)))

        # Accumulate
        for k, v in loss_components.items():
            running[k].append(v)
        running['mae'].append(batch_mae)

        # Update tqdm postfix
        pbar.set_postfix({
            'loss': f"{loss_components['loss/total']:.4f}",
            'mae' : f"{batch_mae:.3f}"
        })

        # Log every batch to wandb for fine-grained loss curves
        global_step = epoch * len(loader) + batch_idx
        wandb.log({
            'batch/train_loss' : loss_components['loss/total'],
            'batch/train_mae'  : batch_mae,
            'batch/train_corn' : loss_components['loss/corn'],
        }, step=global_step)

    epoch_metrics = {
        'train/loss_total' : float(np.mean(running['loss/total'])),
        'train/loss_corn'  : float(np.mean(running['loss/corn'])),
        'train/loss_aux'   : float(np.mean(running['loss/aux'])),
        'train/mae'        : float(np.mean(running['mae'])),
    }

    return epoch_metrics


# ── Single validation epoch ────────────────────────────────────────────────────
@torch.no_grad()
def validate_one_epoch(
    model: nn.Module,
    loader,
    criterion,
    encoder,
    device: torch.device,
    epoch: int,
) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Run one full pass over the validation DataLoader.

    Collects all predictions across the epoch before computing metrics
    so that global MAE, MZOE, RMSE, and Spearman ρ are computed on the
    full validation set rather than averaged over batches (which would
    be incorrect for MZOE and Spearman).

    Returns
    -------
    val_metrics     : dict
    all_pred_ranks  : np.ndarray (n_val_bags,)
    all_true_ranks  : np.ndarray (n_val_bags,)
    all_pred_scores : np.ndarray (n_val_bags,) — s_hat ∈ [0, 1]
    all_true_scores : np.ndarray (n_val_bags,) — true s ∈ [0, 1]
    """
    model.eval()

    all_pred_ranks  = []
    all_true_ranks  = []
    all_pred_scores = []
    all_true_scores = []
    running_loss    = []

    pbar = tqdm(loader, desc=f"Epoch {epoch:03d} [val]  ", leave=False)

    for bags, s_labels, _ in pbar:
        bags     = bags.to(device, non_blocking=True)
        s_labels = s_labels.to(device, non_blocking=True)

        logits, _ = model(bags)
        loss, loss_components = criterion(logits, s_labels)
        running_loss.append(loss_components['loss/total'])

        pred_ranks  = rank_from_logits_np(logits)
        true_ranks  = encoder.encode(s_labels.cpu()).numpy()
        # Continuous expected-value score — granular, single source of truth.
        pred_scores = score_from_logits_np(logits, encoder.num_classes).astype(np.float32)
        true_scores = s_labels.cpu().numpy()

        all_pred_ranks.append(pred_ranks)
        all_true_ranks.append(true_ranks)
        all_pred_scores.append(pred_scores)
        all_true_scores.append(true_scores)

        pbar.set_postfix({'val_loss': f"{loss_components['loss/total']:.4f}"})

    # Concatenate across all batches
    all_pred_ranks  = np.concatenate(all_pred_ranks)
    all_true_ranks  = np.concatenate(all_true_ranks)
    all_pred_scores = np.concatenate(all_pred_scores)
    all_true_scores = np.concatenate(all_true_scores)

    # Compute global metrics on full val set
    rank_metrics  = compute_metrics(all_pred_ranks, all_true_ranks)
    score_metrics = compute_score_metrics(all_pred_scores, all_true_scores)

    val_metrics = {
        'val/loss'          : float(np.mean(running_loss)),
        **{f'val/{k.split("/")[1]}': v for k, v in rank_metrics.items()},
        **{f'val/{k.split("/")[1]}': v for k, v in score_metrics.items()},
    }

    return val_metrics, all_pred_ranks, all_true_ranks, all_pred_scores, all_true_scores


# ── Checkpoint utilities ───────────────────────────────────────────────────────
def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch: int,
    val_metrics: dict,
    encoder,
    cfg: dict,
    tag: str = 'best',
):
    """
    Save model checkpoint to outputs/checkpoints/.

    Saves model state, optimizer state, scheduler state, encoder,
    epoch number, and val metrics. Full state is saved so training
    can be resumed from any checkpoint if needed.
    """
    ckpt_dir = Path(cfg['outputs']['checkpoint_dir'])
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    run_name = wandb.run.name if wandb.run else 'local'
    path     = ckpt_dir / f"{run_name}_{tag}.pt"

    torch.save({
        'epoch'        : epoch,
        'model_state'  : model.state_dict(),
        'optim_state'  : optimizer.state_dict(),
        'sched_state'  : scheduler.state_dict() if scheduler else None,
        'val_metrics'  : val_metrics,
        'encoder'      : encoder,
        'cfg'          : cfg,
    }, path)

    print(f"  Checkpoint saved → {path}")
    return str(path)


def load_checkpoint(path: str, model: nn.Module, optimizer=None, scheduler=None):
    """
    Load a checkpoint. Returns epoch and val_metrics for resuming.
    optimizer and scheduler states are loaded in-place if provided.
    """
    ckpt = torch.load(path, weights_only=False)
    model.load_state_dict(ckpt['model_state'])
    if optimizer and ckpt.get('optim_state'):
        optimizer.load_state_dict(ckpt['optim_state'])
    if scheduler and ckpt.get('sched_state'):
        scheduler.load_state_dict(ckpt['sched_state'])
    print(f"Loaded checkpoint from epoch {ckpt['epoch']}  "
          f"val_mae={ckpt['val_metrics'].get('val/mae', 'n/a'):.4f}")
    return ckpt['epoch'], ckpt['val_metrics'], ckpt.get('encoder')


# ── Early stopping ─────────────────────────────────────────────────────────────
class EarlyStopping:
    """
    Stops training when the monitored metric has not improved for `patience`
    epochs. Lower = better, so improvement means strictly lower monitored value.

    Parameters
    ----------
    patience  : int   — epochs to wait before stopping after last improvement
    min_delta : float — minimum absolute improvement to count as improvement
    """

    def __init__(self, patience: int = 15, min_delta: float = 1e-4):
        self.patience    = patience
        self.min_delta   = min_delta
        self.best_value  = np.inf
        self.counter     = 0
        self.should_stop = False

    def step(self, value: float) -> bool:
        """Returns True if training should stop."""
        if value < self.best_value - self.min_delta:
            self.best_value = value
            self.counter    = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop


# ── LOCO held-out metric helper ──────────────────────────────────────────────────
def _compute_loco_block(
    pred_scores: np.ndarray,
    true_scores: np.ndarray,
    val_labels_df: pd.DataFrame | None,
) -> tuple[dict, pd.DataFrame | None]:
    """
    Compute the honest continuous-s held-out metrics for a LOCO fold, plus a
    per-passage residual table (the young-passage uncertainty the endpoint
    alignment predicts).

    Because val bags are aggregated per (line, passage) group during bagging,
    the per-passage residual table is only produced when a per-bag label frame
    is available; otherwise only the scalar metrics are returned.

    Parameters
    ----------
    pred_scores   : predicted continuous s for the held-out bags
    true_scores   : true continuous s for the held-out bags
    val_labels_df : optional per-bag frame with a 'passage' column aligned to
                    pred_scores/true_scores. May be None.

    Returns
    -------
    metrics       : dict in the 'loco/*' namespace
    per_passage   : pd.DataFrame or None — residuals grouped by passage
    """
    metrics = loco_metrics(true_scores, pred_scores)

    per_passage = None
    if val_labels_df is not None and 'passage' in val_labels_df.columns \
            and len(val_labels_df) == len(pred_scores):
        df = pd.DataFrame({
            'passage' : val_labels_df['passage'].values,
            's_true'  : true_scores,
            's_hat'   : pred_scores,
            'abs_err' : np.abs(pred_scores - true_scores),
        })
        per_passage = (
            df.groupby('passage')
              .agg(s_true=('s_true', 'mean'),
                   s_hat_mean=('s_hat', 'mean'),
                   s_hat_std=('s_hat', 'std'),
                   mae_s=('abs_err', 'mean'),
                   n=('abs_err', 'size'))
              .reset_index()
        )
    return metrics, per_passage


# ── Main training orchestration ────────────────────────────────────────────────
def train(
    model: nn.Module,
    train_loader,
    val_loader,
    criterion,
    encoder,
    cfg: dict,
    device: torch.device,
    val_labels_df: pd.DataFrame | None = None,
):
    """
    Full training run with wandb logging, LR scheduling, early stopping,
    and best-model checkpointing.

    Parameters
    ----------
    model         : MILAgeScoreModel — already moved to device
    train_loader  : DataLoader
    val_loader    : DataLoader
    criterion     : MILAgeLoss
    encoder       : OrdinalLabelEncoder (full-vocabulary under LOCO)
    cfg           : full config dict
    device        : torch.device
    val_labels_df : optional per-bag label frame for the val split, used to
                    build the LOCO per-passage residual table. Pass the
                    held-out line's bag-level labels if available; otherwise
                    only scalar LOCO metrics are reported.

    Returns
    -------
    best_val_metrics : dict — metrics from the epoch with the best monitored value
    """
    training_cfg = cfg['training']

    # ── Validation scheme ───────────────────────────────────────────────────────
    scheme   = cfg.get('validation', {}).get('scheme', 'group').lower()
    is_loco  = (scheme == 'loco')
    held_out = cfg.get('validation', {}).get('held_out_line', None)

    # Under LOCO the honest, comparable metric is continuous-s MAE, NOT rank MAE
    # (rank indices are not aligned across lines). Monitor that for early stop
    # and checkpoint selection. Otherwise keep the original rank-MAE behaviour.
    monitor_key = 'loco/mae_s' if is_loco else 'val/mae'

    # ── Optimiser ──────────────────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_cfg['learning_rate'],
        weight_decay=training_cfg['weight_decay'],
    )

    # ── LR scheduler ───────────────────────────────────────────────────────────
    # ReduceLROnPlateau monitors the same metric as early stopping; reduces LR
    # by factor 0.5 after `scheduler_patience` epochs without improvement.
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=training_cfg.get('scheduler_patience', 7),
        min_lr=training_cfg.get('min_lr', 1e-6)
    )

    # ── Early stopping ─────────────────────────────────────────────────────────
    early_stopper = EarlyStopping(
        patience=training_cfg['patience'],
        min_delta=training_cfg.get('early_stop_min_delta', 1e-4),
    )

    # ── wandb watch — log gradient and weight histograms ──────────────────────
    wandb.watch(model, log='gradients', log_freq=50)

    best_monitor      = np.inf
    best_val_metrics  = {}
    best_ckpt_path    = None
    best_per_passage  = None
    n_epochs          = training_cfg['epochs']

    print(f"\nStarting training — {n_epochs} epochs max  |  "
          f"early stop patience={training_cfg['patience']}  |  "
          f"device={device}")
    if is_loco:
        print(f"Validation scheme: LOCO  (held-out line {held_out})  "
              f"|  monitoring '{monitor_key}' (continuous-s MAE)")
    else:
        print(f"Validation scheme: group within-line  |  monitoring '{monitor_key}'")
    print(f"{'Epoch':>6}  {'Train Loss':>11}  {'Train MAE':>10}  "
          f"{'Val Loss':>9}  {'Val MAE':>8}  {'Val ρ':>7}  "
          f"{'LOCO ρ':>7}  {'MAE_s':>7}  {'LR':>9}  {'ES':>4}")
    print("─" * 104)

    for epoch in range(1, n_epochs + 1):
        t0 = time.time()

        # ── Train ──────────────────────────────────────────────────────────────
        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer,
            encoder, device, epoch, cfg
        )

        # ── Validate ───────────────────────────────────────────────────────────
        (val_metrics, pred_ranks, true_ranks,
         pred_scores, true_scores) = validate_one_epoch(
            model, val_loader, criterion, encoder, device, epoch
        )

        # ── LOCO honest continuous-s metrics ─────────────────────────────────────
        per_passage = None
        if is_loco:
            loco_block, per_passage = _compute_loco_block(
                pred_scores, true_scores, val_labels_df
            )
            val_metrics.update(loco_block)

        # ── Monitored value (continuous-s MAE under LOCO, else rank MAE) ─────────
        monitor_value = val_metrics.get(monitor_key, val_metrics['val/mae'])

        # ── LR scheduler step ──────────────────────────────────────────────────
        scheduler.step(monitor_value)
        current_lr = optimizer.param_groups[0]['lr']

        # ── wandb epoch-level logging ──────────────────────────────────────────
        wandb.log({
            **train_metrics,
            **val_metrics,
            'epoch': epoch,
            'learning_rate': current_lr,
        })

        # ── Console output ─────────────────────────────────────────────────────
        elapsed  = time.time() - t0
        rho      = val_metrics.get('val/spearman_rho', float('nan'))
        loco_rho = val_metrics.get('loco/spearman_rho', float('nan'))
        loco_mae = val_metrics.get('loco/mae_s', float('nan'))
        es_ctr   = early_stopper.counter

        print(
            f"{epoch:>6}  "
            f"{train_metrics['train/loss_total']:>11.4f}  "
            f"{train_metrics['train/mae']:>10.3f}  "
            f"{val_metrics['val/loss']:>9.4f}  "
            f"{val_metrics['val/mae']:>8.3f}  "
            f"{rho:>7.3f}  "
            f"{loco_rho:>7.3f}  "
            f"{loco_mae:>7.3f}  "
            f"{current_lr:>9.2e}  "
            f"{es_ctr:>3}/{training_cfg['patience']}"
            f"  [{elapsed:.1f}s]"
        )

        # ── Save best checkpoint (by monitored value) ───────────────────────────
        if monitor_value < best_monitor:
            best_monitor     = monitor_value
            best_val_metrics = val_metrics.copy()
            best_per_passage = per_passage
            best_ckpt_path   = save_checkpoint(
                model, optimizer, scheduler, epoch,
                val_metrics, encoder, cfg, tag='best'
            )
            wandb.run.summary['best_monitor_metric']  = monitor_key
            wandb.run.summary['best_monitor_value']   = best_monitor
            wandb.run.summary['best_val_mae']         = val_metrics['val/mae']
            wandb.run.summary['best_val_spearman_rho'] = rho
            if is_loco:
                wandb.run.summary['best_loco_spearman_rho'] = loco_rho
                wandb.run.summary['best_loco_mae_s']        = loco_mae
            wandb.run.summary['best_epoch'] = epoch

        # ── Early stopping check (on monitored value) ────────────────────────────
        if early_stopper.step(monitor_value):
            print(f"\n  Early stopping triggered at epoch {epoch} "
                  f"(no improvement in '{monitor_key}' for "
                  f"{training_cfg['patience']} epochs)")
            break

    # ── Save final epoch checkpoint ────────────────────────────────────────────
    save_checkpoint(
        model, optimizer, scheduler, epoch,
        val_metrics, encoder, cfg, tag='final'
    )

    print(f"\n{'─' * 104}")
    print(f"Training complete.")
    print(f"  Monitored metric      : {monitor_key}  (best {best_monitor:.4f})")
    print(f"  Best val MAE (rank)   : {best_val_metrics.get('val/mae', float('nan')):.4f}")
    print(f"  Best val Spearman ρ   : "
          f"{best_val_metrics.get('val/spearman_rho', float('nan')):.4f}")
    print(f"  Best val MZOE         : "
          f"{best_val_metrics.get('val/mzoe', float('nan')):.4f}")
    if is_loco:
        print(f"  ── LOCO held-out (continuous-s, the honest generalisation number) ──")
        print(f"  Held-out line         : {held_out}")
        print(f"  Best LOCO Spearman ρ  : "
              f"{best_val_metrics.get('loco/spearman_rho', float('nan')):.4f}")
        print(f"  Best LOCO MAE_s       : "
              f"{best_val_metrics.get('loco/mae_s', float('nan')):.4f}")
        if best_per_passage is not None:
            print(f"\n  Per-passage residuals on held-out line "
                  f"(expect larger MAE_s at young passages):")
            print(best_per_passage.to_string(index=False))
            # Persist for the thesis figure.
            out_dir = Path(cfg['outputs'].get('log_dir', '.'))
            out_dir.mkdir(parents=True, exist_ok=True)
            run_name = wandb.run.name if wandb.run else 'local'
            csv_path = out_dir / f"{run_name}_loco_per_passage.csv"
            best_per_passage.to_csv(csv_path, index=False)
            print(f"\n  Per-passage residuals saved → {csv_path}")
    print(f"  Best checkpoint       : {best_ckpt_path}")

    wandb.log({
        'final/best_monitor_value'  : best_monitor,
        'final/best_val_mae'        : best_val_metrics.get('val/mae', float('nan')),
        'final/best_val_spearman'   : best_val_metrics.get('val/spearman_rho', float('nan')),
        **({'final/best_loco_spearman': best_val_metrics.get('loco/spearman_rho', float('nan')),
            'final/best_loco_mae_s'   : best_val_metrics.get('loco/mae_s', float('nan'))}
           if is_loco else {}),
    })

    return best_val_metrics
