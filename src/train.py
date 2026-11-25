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
    inner_val_loader=None,
):
    """
    Full training run with wandb logging, LR scheduling, early stopping,
    and best-model checkpointing.

    Parameters
    ----------
    model            : MILAgeScoreModel — already moved to device
    train_loader     : DataLoader
    val_loader       : DataLoader (primary validation, or held-out in inner_val mode)
    criterion        : MILAgeLoss
    encoder          : OrdinalLabelEncoder (full-vocabulary under LOCO)
    cfg              : full config dict
    device           : torch.device
    val_labels_df    : optional per-bag label frame for the val split, used to
                       build the LOCO per-passage residual table.
    inner_val_loader : optional DataLoader for inner validation (one well per group).
                       If provided, monitoring/checkpointing uses this.

    Returns
    -------
    best_val_metrics : dict — metrics from the epoch with the best monitored value
    """
    training_cfg = cfg['training']
    val_cfg = cfg.get('validation', {})

    # ── Validation scheme ───────────────────────────────────────────────────────
    scheme   = val_cfg.get('scheme', 'group').lower()
    is_loco  = (scheme == 'loco')
    held_out = val_cfg.get('held_out_line', None)
    inner_val_enabled = val_cfg.get('inner_val', False)

    # Monitor inner validation if enabled, otherwise loco/mae_s or val/mae
    if inner_val_enabled:
        monitor_key = 'val/mae' # calculated on inner_val_loader
    else:
        monitor_key = 'loco/mae_s' if is_loco else 'val/mae'

    # ── Optimiser ──────────────────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_cfg['learning_rate'],
        weight_decay=training_cfg['weight_decay'],
    )

    # ── LR scheduler ───────────────────────────────────────────────────────────
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

    # ── Setup per-epoch CSV logging ───────────────────────────────────────────
    epoch_csv_rows = []
    log_dir = Path(cfg['outputs'].get('log_dir', '.'))
    log_dir.mkdir(parents=True, exist_ok=True)
    run_name = wandb.run.name if wandb.run else 'local'
    epoch_csv_path = log_dir / f"epochs_{run_name}.csv"

    print(f"\nStarting training — {n_epochs} epochs max  |  "
          f"early stop patience={training_cfg['patience']}  |  "
          f"device={device}")
    if inner_val_enabled:
        print(f"Validation: INNER (seeded wells) | Monitoring '{monitor_key}' on inner-val set")
    elif is_loco:
        print(f"Validation: LOCO (held-out line {held_out}) | Monitoring '{monitor_key}'")
    else:
        print(f"Validation: group within-line | Monitoring '{monitor_key}'")

    print(f"{'Epoch':>6}  {'Train Loss':>11}  {'T-MAE':>7}  "
          f"{'V-MAE':>7}  {'L-MAE':>7}  {'L-ρ':>7}  {'LR':>9}  {'ES':>4}")
    print("─" * 90)

    for epoch in range(1, n_epochs + 1):
        t0 = time.time()

        # ── Train ──────────────────────────────────────────────────────────────
        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer,
            encoder, device, epoch, cfg
        )

        # ── Validation (Inner or Standard) ──────────────────────────────────────
        # If inner_val_enabled, val_loader is the held-out line. We use inner_val_loader for monitoring.
        if inner_val_enabled:
            # Monitoring set
            (innerval_metrics, _, _, _, _) = validate_one_epoch(
                model, inner_val_loader, criterion, encoder, device, epoch
            )
            # Held-out set (logged but not monitored)
            (heldout_metrics, _, _, pred_scores, true_scores) = validate_one_epoch(
                model, val_loader, criterion, encoder, device, epoch
            )
            val_metrics = innerval_metrics
            loco_val_metrics = heldout_metrics
        else:
            (val_metrics, _, _, pred_scores, true_scores) = validate_one_epoch(
                model, val_loader, criterion, encoder, device, epoch
            )
            loco_val_metrics = val_metrics

        # ── LOCO metrics on held-out line ────────────────────────────────────────
        per_passage = None
        if is_loco:
            loco_block, per_passage = _compute_loco_block(
                pred_scores, true_scores, val_labels_df
            )
            loco_val_metrics.update(loco_block)

        # ── Monitored value ──────────────────────────────────────────────────────
        monitor_value = val_metrics.get(monitor_key, val_metrics['val/mae'])

        # ── LR scheduler step ──────────────────────────────────────────────────
        scheduler.step(monitor_value)
        current_lr = optimizer.param_groups[0]['lr']

        # ── wandb epoch-level logging ──────────────────────────────────────────
        log_dict = {
            **train_metrics,
            'epoch': epoch,
            'learning_rate': current_lr,
        }
        if inner_val_enabled:
            log_dict.update({f"innerval/{k.split('/')[-1]}": v for k, v in val_metrics.items()})
            log_dict.update({f"heldout/{k.split('/')[-1]}": v for k, v in loco_val_metrics.items()})
            # For backward compatibility with monitoring logic in main.py
            log_dict['val/mae'] = val_metrics['val/mae']
        else:
            log_dict.update(val_metrics)

        wandb.log(log_dict)

        # ── Per-epoch CSV row ──────────────────────────────────────────────────
        csv_row = {
            'epoch': epoch,
            'lr': current_lr,
            'train_loss': train_metrics['train/loss_total'],
            'train_mae': train_metrics['train/mae'],
        }
        if inner_val_enabled:
            csv_row.update({
                'innerval_loss': val_metrics['val/loss'],
                'innerval_mae': val_metrics['val/mae'],
                'heldout_loss': loco_val_metrics['val/loss'],
                'heldout_mae_s': loco_val_metrics.get('loco/mae_s', float('nan')),
                'heldout_spearman': loco_val_metrics.get('loco/spearman_rho', float('nan')),
            })
        else:
            csv_row.update({
                'val_loss': val_metrics['val/loss'],
                'val_mae': val_metrics['val/mae'],
                'loco_mae_s': val_metrics.get('loco/mae_s', float('nan')),
                'loco_spearman': val_metrics.get('loco/spearman_rho', float('nan')),
            })

        # ── Console output ─────────────────────────────────────────────────────
        elapsed  = time.time() - t0
        v_mae    = val_metrics.get('val/mae', float('nan'))
        l_mae    = loco_val_metrics.get('loco/mae_s', float('nan'))
        l_rho    = loco_val_metrics.get('loco/spearman_rho', float('nan'))
        es_ctr   = early_stopper.counter

        print(
            f"{epoch:>6}  "
            f"{train_metrics['train/loss_total']:>11.4f}  "
            f"{train_metrics['train/mae']:>7.3f}  "
            f"{v_mae:>7.3f}  "
            f"{l_mae:>7.3f}  "
            f"{l_rho:>7.3f}  "
            f"{current_lr:>9.2e}  "
            f"{es_ctr:>3}/{training_cfg['patience']}"
            f"  [{elapsed:.1f}s]"
        )

        # ── Save best checkpoint (by monitored value) ───────────────────────────
        is_best = False
        if monitor_value < best_monitor:
            is_best = True
            best_monitor     = monitor_value
            best_val_metrics = (loco_val_metrics.copy() if not inner_val_enabled 
                                else {**val_metrics, **loco_val_metrics})
            best_per_passage = per_passage
            best_ckpt_path   = save_checkpoint(
                model, optimizer, scheduler, epoch,
                val_metrics, encoder, cfg, tag='best'
            )
            wandb.run.summary['best_monitor_metric']  = monitor_key
            wandb.run.summary['best_monitor_value']   = best_monitor
            wandb.run.summary['best_epoch'] = epoch
        
        csv_row['is_best'] = int(is_best)
        epoch_csv_rows.append(csv_row)

        # ── Early stopping check (on monitored value) ────────────────────────────
        if early_stopper.step(monitor_value):
            print(f"\n  Early stopping triggered at epoch {epoch}")
            break

    # Save per-epoch CSV
    pd.DataFrame(epoch_csv_rows).to_csv(epoch_csv_path, index=False)

    # ── Save final and last checkpoints ────────────────────────────────────────
    save_checkpoint(model, optimizer, scheduler, epoch, val_metrics, encoder, cfg, tag='last')
    save_checkpoint(model, optimizer, scheduler, epoch, val_metrics, encoder, cfg, tag='final')

    print(f"\n{'─' * 90}")
    print(f"Training complete. Best inner-validation {monitor_key}: {best_monitor:.4f}")

    if is_loco:
        l_rho = best_val_metrics.get('loco/spearman_rho', float('nan'))
        l_mae = best_val_metrics.get('loco/mae_s', float('nan'))
        print(f"Best held-out line metrics (at best inner-val epoch): Spearman ρ: {l_rho:.4f}  |  MAE_s: {l_mae:.4f}")
        if best_per_passage is not None:
            csv_path = log_dir / f"{run_name}_loco_per_passage.csv"
            best_per_passage.to_csv(csv_path, index=False)

    return best_val_metrics
