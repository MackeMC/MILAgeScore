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
from src.evaluate import (compare_treatment_groups, evaluate_passage_trajectory, load_model_from_checkpoint, save_eval_results,)
from src.explain import (aggregate_feature_importance, compute_attention_summary, save_explain_results)
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
    filter_features,
    set_seed,
    setup_output_dirs,
)
from src.split_loco import build_loco_encoder
from src.explain import normalise_attributions_by_block
from src.binning import prepare_binned_labels
import json

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
        cfg['data']['processed_dir'], 
        cfg['data'].get('feature_set', 'both'),
        cfg['data'].get('feature_include')
    )
    w_manual, w_latent              = resolve_block_weights(n_manual, n_latent, cfg)

    # ── Build dataloaders ──────────────────────────────────────────────────────
    # build_dataloaders handles: split → fit scalers → save scalers → wrap datasets
    inner_val_loader = None
    inner_wells_df = None
    
    loaders = build_dataloaders(
        processed_dir=cfg['data']['processed_dir'],
        cfg=cfg,
    )
    
    if len(loaders) == 6:
        train_loader, inner_val_loader, val_loader, scaler_manual, scaler_latent, inner_wells_df = loaders
        # Save inner_val_wells.csv
        out_dir = Path(cfg['outputs']['log_dir'])
        out_dir.mkdir(parents=True, exist_ok=True)
        run_name = wandb.run.name if wandb.run else 'local'
        inner_wells_df.to_csv(out_dir / f"inner_val_wells_{run_name}.csv", index=False)
    else:
        train_loader, val_loader, scaler_manual, scaler_latent = loaders

    # ── Load labels for loss/encoder construction ──────────────────────────────
    labels_df = pd.read_csv(Path(cfg['data']['processed_dir']) / 'labels.csv')
    val_cfg = cfg.get('validation', {})
    scheme = val_cfg.get('scheme', 'group').lower()
    if scheme == 'binned_kfold' or (scheme == 'loco' and val_cfg.get('n_bins')):
        labels_df = prepare_binned_labels(labels_df, val_cfg.get('n_bins', 10))

    # ── Held-out per-bag labels for the LOCO per-passage residual table ─────────
    val_labels_df = None
    if scheme == 'loco':
        from src.split_loco import split_leave_one_line_out
        line_column = 'cell_line_orig' if 'cell_line_orig' in labels_df.columns else 'cell_line'
        _, _val_idx = split_leave_one_line_out(
            labels_df,
            held_out_line=val_cfg['held_out_line'],
            line_column=line_column
        )
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
        inner_val_loader=inner_val_loader,
    )

    logger.info(f"Training complete. Best val MAE: {best_val_metrics['val/mae']:.4f}")
    wandb.finish()

    out_dir = Path(cfg['outputs']['log_dir']);
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / 'best_val_metrics.json', 'w') as f:
        json.dump({k: float(v) for k, v in best_val_metrics.items() if np.isscalar(v)}, f, indent=2)
    return best_val_metrics


# ── Evaluation run ─────────────────────────────────────────────────────────────
def run_evaluate(cfg: dict, device: torch.device, logger):
    logger.info("Mode: EVALUATE")

    # ── Resolve dims and load data ─────────────────────────────────────────────
    in_features, n_manual, n_latent = resolve_model_dims(
        cfg['data']['processed_dir'], 
        cfg['data'].get('feature_set', 'both'),
        cfg['data'].get('feature_include')
    )
    w_manual, w_latent              = resolve_block_weights(n_manual, n_latent, cfg)

    features_manual = np.load(
        Path(cfg['data']['processed_dir']) / 'features_manual.npy'
    )
    features_latent = np.load(
        Path(cfg['data']['processed_dir']) / 'features_latent.npy'
    )
    # Also load column names to allow filtering
    with open(Path(cfg['data']['processed_dir']) / 'manual_cols.pkl', 'rb') as f:
        manual_cols = pickle.load(f)
    with open(Path(cfg['data']['processed_dir']) / 'latent_cols.pkl', 'rb') as f:
        latent_cols = pickle.load(f)

    # Respect feature_set selection by zeroing out the unused block(s)
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

    labels_df = pd.read_csv(Path(cfg['data']['processed_dir']) / 'labels.csv')

    # ── Apply saved scalers ────────────────────────────────────────────────────
    scaler_dir = Path(cfg['outputs']['log_dir']) / 'scalers'
    if not scaler_dir.exists():
        logger.warning(f"Run-specific scalers not found at {scaler_dir}. Falling back to processed_dir.")
        scaler_dir = None
    
    scaler_manual, scaler_latent = load_scalers(cfg['data']['processed_dir'], scaler_dir=scaler_dir)
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
    model.eval()

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
    if not cfg.get('evaluate', {}).get('skip_treatments', False):
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
    else:
        run_name = cfg['wandb'].get('run_name', 'eval')
        save_eval_results(trajectory_df, None, None, cfg, run_name)

    # ── TASK 1: Export Bag-Level and Group-Level Results ──────────────────────
    if cfg.get('validation', {}).get('inner_val', False) or 'outputs/review' in cfg['outputs'].get('log_dir', ''):
        out_dir = Path(cfg['outputs']['log_dir'])
        
        # 1. Bag-level predictions
        bag_rows = []
        for _, row in trajectory_df.iterrows():
            scores = row['all_scores']
            ranks  = row['all_ranks']
            for i, (s_hat, r_hat) in enumerate(zip(scores, ranks)):
                bag_rows.append({
                    'run_id': run_name,
                    'seed': cfg['bags']['random_seed'],
                    'train_line': '1' if cfg['validation']['held_out_line'] == '2' else '2',
                    'held_out_line': cfg['validation']['held_out_line'],
                    'cell_line': row['cell_line'],
                    'passage_raw': row['passage_raw'],
                    'bin_label': int(row['passage']), # Assuming 'passage' column holds the binned label
                    's_bin': row['s_true'],
                    'set': 'heldout' if str(row['cell_line']) == str(cfg['validation']['held_out_line']) else 'train_or_val',
                    'bag_idx': i,
                    's_hat': round(float(s_hat), 4),
                    'rank_hat': int(r_hat),
                    'model': 'mil',
                    'pooling': cfg['model'].get('pooling', 'attention'),
                    'feature_set': cfg['data'].get('feature_set', 'both')
                })
        pd.DataFrame(bag_rows).to_csv(out_dir / f"bag_predictions_{run_name}.csv", index=False)
        
        # 2. Group-level predictions (already mostly in trajectory_df, but format for requirement)
        group_rows = []
        for _, row in trajectory_df.iterrows():
            group_rows.append({
                'run_id': run_name,
                'seed': cfg['bags']['random_seed'],
                'train_line': '1' if cfg['validation']['held_out_line'] == '2' else '2',
                'held_out_line': cfg['validation']['held_out_line'],
                'cell_line': row['cell_line'],
                'passage_raw': row['passage_raw'],
                'set': 'heldout' if str(row['cell_line']) == str(cfg['validation']['held_out_line']) else 'train_or_val',
                's_hat_mean': row['s_hat_mean'],
                's_hat_sd': row['s_hat_std'],
                's_hat_median': round(float(np.median(row['all_scores'])), 4)
            })
        pd.DataFrame(group_rows).to_csv(out_dir / f"group_predictions_{run_name}.csv", index=False)

    logger.info("Evaluation complete.")


# ── Explainability run ─────────────────────────────────────────────────────────
def run_explain(cfg: dict, device: torch.device, logger):
    logger.info("Mode: EXPLAIN")

    in_features, n_manual, n_latent = resolve_model_dims(
        cfg['data']['processed_dir'], 
        cfg['data'].get('feature_set', 'both'),
        cfg['data'].get('feature_include')
    )
    w_manual, w_latent              = resolve_block_weights(n_manual, n_latent, cfg)


    features_manual = np.load(
        Path(cfg['data']['processed_dir']) / 'features_manual.npy'
    )
    features_latent = np.load(
        Path(cfg['data']['processed_dir']) / 'features_latent.npy'
    )

    with open(Path(cfg['data']['processed_dir']) / 'manual_cols.pkl', 'rb') as f:
        manual_cols = pickle.load(f)
    with open(Path(cfg['data']['processed_dir']) / 'latent_cols.pkl', 'rb') as f:
        latent_cols = pickle.load(f)

    # ── FIX: zero out unused feature block (mirrors run_evaluate) ───────────
    feature_set = cfg['data'].get('feature_set', 'both').lower()
    if feature_set == 'manual':
        features_latent = np.zeros((features_latent.shape[0], 0), dtype=features_latent.dtype)
        latent_cols = []
    elif feature_set == 'latent':
        features_manual = np.zeros((features_manual.shape[0], 0), dtype=features_manual.dtype)
        manual_cols = []

    # Apply feature_include
    feature_include = cfg['data'].get('feature_include')
    features_manual, features_latent, manual_cols, latent_cols = filter_features(
        features_manual, features_latent, manual_cols, latent_cols, feature_include
    )

    labels_df = pd.read_csv(Path(cfg['data']['processed_dir']) / 'labels.csv')

    scaler_dir = Path(cfg['outputs']['log_dir']) / 'scalers'
    if not scaler_dir.exists():
        logger.warning(f"Run-specific scalers not found at {scaler_dir}. Falling back to processed_dir.")
        scaler_dir = None
        
    scaler_manual, scaler_latent = load_scalers(cfg['data']['processed_dir'], scaler_dir=scaler_dir)
    features_manual, features_latent = apply_scalers(
        features_manual, features_latent, scaler_manual, scaler_latent
    )

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
        if not cfg.get('evaluate', {}).get('skip_treatments', False):
            run_explain(cfg, device, logger)
        else:
            logger.info("evaluate.skip_treatments is true: skipping EXPLAIN mode.")

    elif args.mode == 'all':
        run_train(cfg, device, logger)
        run_evaluate(cfg, device, logger)
        if not cfg.get('evaluate', {}).get('skip_treatments', False):
            run_explain(cfg, device, logger)
        else:
            logger.info("evaluate.skip_treatments is true: skipping EXPLAIN mode.")


if __name__ == '__main__':
    main()
