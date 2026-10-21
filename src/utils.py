"""
src/utils.py

Stateless utility functions used across the project.

Responsibilities:
    - Reproducibility: seeding all random number generators
    - Device resolution: GPU/CPU selection with capability check
    - Config loading and validation
    - Logging setup
    - Runtime computation of in_features and num_classes from processed data
      (these are passed to build_model rather than hardcoded in config)
"""

import logging
import os
import pickle
import random
from pathlib import Path

import numpy as np
import torch
import yaml


# ── Reproducibility ────────────────────────────────────────────────────────────
def set_seed(seed: int = 42):
    """
    Seed all relevant RNGs for reproducibility.

    PyTorch, numpy, and Python random are all seeded. CUDNN is set to
    deterministic mode. Note: deterministic mode can reduce throughput
    slightly — acceptable for this workload size.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False
    os.environ['PYTHONHASHSEED']       = str(seed)


# ── Device resolution ──────────────────────────────────────────────────────────
def get_device(cfg: dict) -> torch.device:
    """
    Resolve the compute device.

    Checks CUDA availability and performs a small smoke test to confirm
    the GPU can actually execute kernels — detecting driver-visible but
    compute-incapable configurations.

    Falls back to CPU with a clear warning if CUDA is unavailable or fails
    the smoke test.
    """
    force_cpu = cfg.get('training', {}).get('force_cpu', False)

    if force_cpu:
        print("Device: CPU (forced via config)")
        return torch.device('cpu')

    if not torch.cuda.is_available():
        print("Device: CPU (CUDA not available)")
        return torch.device('cpu')

    try:
        device = torch.device('cuda')
        # Smoke test — catches driver-visible but compute-incapable GPUs
        x = torch.randn(64, 64, device=device)
        _ = torch.mm(x, x)
        cap = torch.cuda.get_device_capability(0)
        name = torch.cuda.get_device_name(0)
        print(f"Device: {name}  |  Compute capability: sm_{cap[0]}{cap[1]}")
        return device
    except RuntimeError as e:
        print(f"CUDA smoke test failed ({e}). Falling back to CPU.")
        return torch.device('cpu')


# ── Config loading and validation ──────────────────────────────────────────────
def load_config(config_path: str) -> dict:
    """
    Load and validate config.yaml.

    Checks that all required top-level sections are present.
    Does not validate every key — missing optional keys use defaults
    defined at their call sites via .get().
    """
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)

    required_sections = ['data', 'bags', 'model', 'training', 'evaluate', 'outputs']
    missing = [s for s in required_sections if s not in cfg]
    if missing:
        raise ValueError(
            f"config.yaml is missing required sections: {missing}\n"
            f"Present sections: {list(cfg.keys())}"
        )

    return cfg


def print_config(cfg: dict):
    """Pretty-print the active config for run reproducibility logging."""
    print("\n" + "═" * 60)
    print("Active configuration:")
    print("═" * 60)
    print(yaml.dump(cfg, default_flow_style=False, sort_keys=False))
    print("═" * 60 + "\n")


# ── Logger setup ───────────────────────────────────────────────────────────────
def get_logger(name: str, log_dir: str = None, level=logging.INFO) -> logging.Logger:
    """
    Create a logger that writes to both stdout and (optionally) a log file.

    Parameters
    ----------
    name    : str — logger name, typically __name__ of the calling module
    log_dir : str — if provided, also writes to {log_dir}/{name}.log
    """
    logger    = logging.getLogger(name)
    logger.setLevel(level)

    fmt       = logging.Formatter(
        '%(asctime)s  %(levelname)-8s  %(name)s  %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Console handler
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # File handler
    if log_dir:
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(Path(log_dir) / f"{name}.log")
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger


# ── Runtime data shape resolution ─────────────────────────────────────────────
def resolve_model_dims(processed_dir: str, feature_set: str = 'both') -> tuple[int, int, int]:
    """
    Compute in_features and num_classes at runtime from processed data.

    These values depend on the actual data and cannot be hardcoded in config:
        - in_features  = n_manual + n_latent (after filtering)
        - n_manual     = len(manual_cols) as saved by prepare_data.py
        - n_latent     = len(latent_cols) as saved by prepare_data.py
        - num_classes  = number of unique passage × cell_line groups (27 here)

    Parameters
    ----------
    processed_dir : str — path to data/processed/

    Returns
    -------
    in_features  : int
    n_manual     : int
    n_latent     : int
    """
    d = Path(processed_dir)

    with open(d / 'manual_cols.pkl', 'rb') as f:
        manual_cols = pickle.load(f)
    with open(d / 'latent_cols.pkl', 'rb') as f:
        latent_cols = pickle.load(f)

    # Original counts
    n_manual_all = len(manual_cols)
    n_latent_all = len(latent_cols)

    # Apply feature set selection
    feature_set = (feature_set or 'both').lower()
    if feature_set not in {'both', 'manual', 'latent'}:
        raise ValueError(f"Invalid data.feature_set='{feature_set}'. Expected one of: both | manual | latent")

    n_manual = n_manual_all if feature_set in {'both', 'manual'} else 0
    n_latent = n_latent_all if feature_set in {'both', 'latent'} else 0
    in_features = n_manual + n_latent

    print(f"Model input dimensions resolved:")
    print(f"  feature_set : {feature_set}")
    print(f"  n_manual    : {n_manual}  (of {n_manual_all})")
    print(f"  n_latent    : {n_latent}  (of {n_latent_all})")
    print(f"  in_features : {in_features}")

    return in_features, n_manual, n_latent



# ── Block weight resolution ────────────────────────────────────────────────────
def resolve_block_weights(
    n_manual: int,
    n_latent: int,
    cfg: dict,
) -> tuple[float, float]:
    """
    Resolve manual and latent block-level weights.

    Default behaviour: w_manual = sqrt(n_latent / n_manual) to equalise
    total variance contribution across both feature blocks after per-feature
    z-scoring. Overridden by explicit values in config if provided.
    """
    ratio     = n_latent / n_manual if n_manual > 0 else 1.0
    default_w = float(np.sqrt(ratio))

    w_manual  = float(cfg['bags'].get('w_manual', default_w))
    w_latent  = float(cfg['bags'].get('w_latent', 1.0))

    # If a block is unused (0 features), force its weight to 0 to avoid leaks
    if n_manual == 0:
        w_manual = 0.0
    if n_latent == 0:
        w_latent = 0.0

    print(f"\nBlock weights:")
    print(f"  n_manual={n_manual}  n_latent={n_latent}  ratio={ratio:.2f}x")
    print(f"  w_manual={w_manual:.4f}  w_latent={w_latent:.4f}  "
          f"({'default sqrt(ratio)' if 'w_manual' not in cfg['bags'] else 'config override'})")

    return w_manual, w_latent


# ── Output directory setup ─────────────────────────────────────────────────────
def setup_output_dirs(cfg: dict):
    """Create all output directories defined in config if they do not exist."""
    for key, path in cfg['outputs'].items():
        Path(path).mkdir(parents=True, exist_ok=True)
    print(f"Output directories ready: {list(cfg['outputs'].values())}")
