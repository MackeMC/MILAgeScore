"""
main.py

Entry point for MILAgeScore training and evaluation.

Usage:
    # Train
    python main.py --config config/config.yaml --mode train

    # Evaluate passage trajectory with best checkpoint
    python main.py --config config/config.yaml --mode evaluate

    # Feature attribution (explain)
    python main.py --config config/config.yaml --mode explain

    # All three sequentially
    python main.py --config config/config.yaml --mode all

    # lATENT features only
    python main.py --config config/config_latent.yaml --mode all

wandb project and run name are set in config.yaml under the wandb section.
"""

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import wandb

from src.dataset import build_dataloaders, load_processed, split_by_group
from src.dataset import apply_scalers, fit_scalers, save_scalers, load_scalers
from src.evaluate import (
    compare_treatment_groups,
    evaluate_passage_trajectory,
    load_model_from_checkpoint,
    save_eval_results,
)
from src.explain import (
    aggregate_feature_importance,
    compute_attention_summary,
    save_explain_results,
)
from src.losses import build_loss
from src.model import build_model
from src.train import train
from src.utils import (
    get_device,
    get_logger,
    load_config,
    print_config,
    resolve_block_weights,
    resolve_model_dims,
    set_seed,
    setup_output_dirs,
)
from src.split_loco import build_loco_encoder
from src.explain import normalise_attributions_by_block

# ── Training run ───────────────────────────────────────────────────────────────
def run_train(cfg: dict, device: torch.device, logger):
    logger.info("Mode: TRAIN")

    # ── wandb init ─────────────────────────────────────────────────────────────
    wandb.init(
        project=cfg['wandb']['project'],
        name=cfg['wandb'].get('run_name', None),
        config=cfg,
        tags=cfg['wandb'].get('tags', []),
        notes=cfg['wandb'].get('notes', ''),
    )

    wandb.define_metric("batch/*", step_metric="batch_step")
    wandb.define_metric("train/*", step_metric="epoch")
    wandb.define_metric("val/*", step_metric="epoch")

    # ── Resolve dims ───────────────────────────────────────────────────────────
    in_features, n_manual, n_latent = resolve_model_dims(
        cfg['data']['processed_dir'], cfg['data'].get('feature_set', 'both')
    )
    w_manual, w_latent              = resolve_block_weights(n_manual, n_latent, cfg)

    # ── Build dataloaders ──────────────────────────────────────────────────────
    # build_dataloaders handles: split → fit scalers → save scalers → wrap datasets
    train_loader, val_loader, scaler_manual, scaler_latent = build_dataloaders(
        processed_dir=cfg['data']['processed_dir'],
        cfg=cfg,
    )

    # ── Load labels for loss/encoder construction ──────────────────────────────
    labels_df          = pd.read_csv(Path(cfg['data']['processed_dir']) / 'labels.csv')

    # ── Held-out per-bag labels for the LOCO per-passage residual table ─────────
    val_labels_df = None
    if scheme_for_val := cfg.get('validation', {}).get('scheme', 'group').lower():
        if scheme_for_val == 'loco':
            from src.split_loco import split_leave_one_line_out
            _, _val_idx = split_leave_one_line_out(labels_df, held_out_line=cfg['validation']['held_out_line'])
            # Note: this is cell-level; the per-passage table groups by passage,
            # so cell-level rows aggregate correctly even though val metrics are
            # bag-level. Pass only if row counts can be aligned in train.py;
            # otherwise train.py falls back to scalar LOCO metrics safely.
            val_labels_df = labels_df.iloc[_val_idx].reset_index(drop=True)

    # ── Build loss and encoder ─────────────────────────────────────────────────
    # Under LOCO the encoder MUST span the FULL s-vocabulary (both lines) so the
    # held-out line's s-values are in-vocabulary at validation. build_loss already
    # builds from the full labels_df; build_loco_encoder makes that span explicit
    # and verifies it. The criterion's CORN loss reuses the same encoder object.
    scheme = cfg.get('validation', {}).get('scheme', 'group').lower()
    criterion, encoder = build_loss(labels_df, cfg)

    if scheme == 'loco':
        full_vocab = build_loco_encoder(labels_df)  # built on ALL lines
        assert full_vocab.num_classes == encoder.num_classes, (
            f"LOCO encoder vocab ({full_vocab.num_classes}) != criterion encoder "
            f"({encoder.num_classes}). labels_df passed to build_loss must be the "
            f"FULL unsplit frame."
        )
        logger.info(
            f"LOCO active: held-out line {cfg['validation']['held_out_line']}, "
            f"full-vocabulary encoder with {encoder.num_classes} ranks."
        )

    # ── Define number of classes from the encoder ─────────────────────────────────────────────────
    num_classes = encoder.num_classes

    # ── Build model ────────────────────────────────────────────────────────────
    model = build_model(cfg, in_features=in_features, num_classes=num_classes)
    model = model.to(device)

    # Log model architecture to wandb
    wandb.config.update({
        'in_features' : in_features,
        'n_manual'    : n_manual,
        'n_latent'    : n_latent,
        'num_classes' : num_classes,
        'w_manual'    : w_manual,
        'w_latent'    : w_latent,
    })

    # ── Train ──────────────────────────────────────────────────────────────────
    best_val_metrics = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        encoder=encoder,
        cfg=cfg,
        device=device,
        val_labels_df=val_labels_df,
    )

    logger.info(f"Training complete. Best val MAE: {best_val_metrics['val/mae']:.4f}")
    wandb.finish()

    return best_val_metrics


# ── Evaluation run ─────────────────────────────────────────────────────────────
def run_evaluate(cfg: dict, device: torch.device, logger):
    logger.info("Mode: EVALUATE")

    # ── Resolve dims and load data ─────────────────────────────────────────────
    in_features, n_manual, n_latent = resolve_model_dims(
        cfg['data']['processed_dir'], cfg['data'].get('feature_set', 'both')
    )
    w_manual, w_latent              = resolve_block_weights(n_manual, n_latent, cfg)

    features_manual = np.load(
        Path(cfg['data']['processed_dir']) / 'features_manual.npy'
    )
    features_latent = np.load(
        Path(cfg['data']['processed_dir']) / 'features_latent.npy'
    )
    # Respect feature_set selection by zeroing out the unused block(s)
    feature_set = cfg['data'].get('feature_set', 'both').lower()
    if feature_set == 'manual':
        features_latent = np.zeros((features_latent.shape[0], 0), dtype=features_latent.dtype)
    elif feature_set == 'latent':
        features_manual = np.zeros((features_manual.shape[0], 0), dtype=features_manual.dtype)
    labels_df = pd.read_csv(Path(cfg['data']['processed_dir']) / 'labels.csv')

    # ── Apply saved scalers ────────────────────────────────────────────────────
    scaler_manual, scaler_latent = load_scalers(cfg['data']['processed_dir'])
    features_manual, features_latent = apply_scalers(
        features_manual, features_latent, scaler_manual, scaler_latent
    )

    # ── Load model ─────────────────────────────────────────────────────────────
    ckpt_path = cfg['evaluate'].get('checkpoint_path') or \
        str(Path(cfg['outputs']['checkpoint_dir']) / f"{cfg['wandb'].get('run_name', 'run')}_best.pt")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    encoder = ckpt['encoder']
    num_classes = encoder.num_classes
    model = build_model(cfg,in_features=in_features, num_classes=num_classes)
    model.load_state_dict(ckpt['model_state'])
    model.to(device)

    # ── Passage trajectory ─────────────────────────────────────────────────────
    trajectory_df = evaluate_passage_trajectory(
        model=model,
        features_manual=features_manual,
        features_latent=features_latent,
        labels_df=labels_df,
        encoder=encoder,
        device=device,
        cfg=cfg,
        w_manual=w_manual,
        w_latent=w_latent,
    )

    # ── Treatment comparison ───────────────────────────────────────────────────
    results_df, comparisons_df = compare_treatment_groups(
        model=model,
        features_manual=features_manual,
        features_latent=features_latent,
        labels_df=labels_df,
        encoder=encoder,
        device=device,
        cfg=cfg,
        w_manual=w_manual,
        w_latent=w_latent,
    )

    run_name = cfg['wandb'].get('run_name', 'eval')
    save_eval_results(trajectory_df, results_df, comparisons_df, cfg, run_name)
    logger.info("Evaluation complete.")


# ── Explainability run ─────────────────────────────────────────────────────────
def run_explain(cfg: dict, device: torch.device, logger):
    logger.info("Mode: EXPLAIN")

    in_features, n_manual, n_latent = resolve_model_dims(cfg['data']['processed_dir'], cfg['data'].get('feature_set', 'both'))
    w_manual, w_latent              = resolve_block_weights(n_manual, n_latent, cfg)


    features_manual = np.load(
        Path(cfg['data']['processed_dir']) / 'features_manual.npy'
    )
    features_latent = np.load(
        Path(cfg['data']['processed_dir']) / 'features_latent.npy'
    )

    # ── FIX: zero out unused feature block (mirrors run_evaluate) ───────────
    feature_set = cfg['data'].get('feature_set', 'both').lower()
    if feature_set == 'manual':
        features_latent = np.zeros((features_latent.shape[0], 0), dtype=features_latent.dtype)
    elif feature_set == 'latent':
        features_manual = np.zeros((features_manual.shape[0], 0), dtype=features_manual.dtype)

    labels_df = pd.read_csv(Path(cfg['data']['processed_dir']) / 'labels.csv')

    scaler_manual, scaler_latent = load_scalers(cfg['data']['processed_dir'])
    features_manual, features_latent = apply_scalers(
        features_manual, features_latent, scaler_manual, scaler_latent
    )

    with open(Path(cfg['data']['processed_dir']) / 'manual_cols.pkl', 'rb') as f:
        manual_cols = pickle.load(f)
    with open(Path(cfg['data']['processed_dir']) / 'latent_cols.pkl', 'rb') as f:
        latent_cols = pickle.load(f)

    feature_set = cfg['data'].get('feature_set', 'both').lower()
    if feature_set == 'latent':
        manual_cols = []
    elif feature_set == 'manual':
        latent_cols = []

    ckpt_path = cfg['evaluate'].get('checkpoint_path') or \
                str(Path(cfg['outputs']['checkpoint_dir']) / f"{cfg['wandb'].get('run_name', 'run')}_best.pt")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    encoder = ckpt['encoder']
    num_classes = encoder.num_classes
    model = build_model(cfg, in_features=in_features, num_classes=num_classes)
    model.load_state_dict(ckpt['model_state'])
    model.to(device).eval()

    # ── Attention weights ──────────────────────────────────────────────────────
    attn_df = compute_attention_summary(
        model=model,
        features_manual=features_manual,
        features_latent=features_latent,
        labels_df=labels_df,
        encoder=encoder,
        device=device,
        cfg=cfg,
        w_manual=w_manual,
        w_latent=w_latent,
    )

    # ── Integrated Gradients feature importance ────────────────────────────────
    global_importance = aggregate_feature_importance(
        model=model,
        features_manual=features_manual,
        features_latent=features_latent,
        labels_df=labels_df,
        manual_cols=manual_cols,
        latent_cols=latent_cols,
        encoder=encoder,
        device=device,
        cfg=cfg,
        w_manual=w_manual,
        w_latent=w_latent,
    )

    global_importance_raw = global_importance.copy()
    global_importance_norm = normalise_attributions_by_block(
        global_importance, w_manual, w_latent
    )

    run_name = cfg['wandb'].get('run_name', 'explain')
    save_explain_results(attn_df, global_importance, cfg, run_name)
    logger.info("Explainability analysis complete.")

    print(f"\nTop 15 features (block-weight normalised |IG|):")
    print(global_importance_norm.head(15).to_string(index=False))

    # Save both versions
    global_importance_raw.to_csv(
        Path(cfg['outputs']['log_dir']) / f'{run_name}_importance_raw.csv', index=False
    )
    global_importance_norm.to_csv(
        Path(cfg['outputs']['log_dir']) / f'{run_name}_importance_normalised.csv', index=False
    )

# ── CLI entry point ────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description='MILAgeScore — train/evaluate/explain')
    parser.add_argument(
        '--config', type=str, default='config/config.yaml',
        help='Path to config.yaml'
    )
    parser.add_argument(
        '--mode', type=str, default='train',
        choices=['train', 'evaluate', 'explain', 'all'],
        help='Run mode'
    )
    args = parser.parse_args()

    # ── Setup ──────────────────────────────────────────────────────────────────
    cfg    = load_config(args.config)
    seed   = cfg['bags'].get('random_seed', 42)
    set_seed(seed)
    setup_output_dirs(cfg)

    logger = get_logger(
        name='MILAgeScore',
        log_dir=cfg['outputs']['log_dir']
    )

    print_config(cfg)
    device = get_device(cfg)

    # ── Dispatch ───────────────────────────────────────────────────────────────
    if args.mode == 'train':
        run_train(cfg, device, logger)

    elif args.mode == 'evaluate':
        run_evaluate(cfg, device, logger)

    elif args.mode == 'explain':
        run_explain(cfg, device, logger)

    elif args.mode == 'all':
        run_train(cfg, device, logger)
        run_evaluate(cfg, device, logger)
        run_explain(cfg, device, logger)


if __name__ == '__main__':
    main()
