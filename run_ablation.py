#!/usr/bin/env python
"""
run_ablation.py  -  MILAgeScore 3-way ablation (bag_size x bags_per_group x batch_size)

main.py only accepts --config and --mode, so for every run this script writes a
per-run config (base config + overrides) and calls:

    python main.py --config <run_dir>/config.yaml --mode all

Layout:
    ablation/<study_name>/
        ablation_config.json
        summary.csv
        bs025_bpg0100_batch016/
            config.yaml  stdout.log  .done
            checkpoints/  logs/  figures/

Usage (from the project root):
    python run_ablation.py --dry-run
    python run_ablation.py --name study1                 # start a study
    python run_ablation.py --name study1 --resume        # continue, skipping finished runs
    python run_ablation.py --resume                      # continue the most recent study
    python run_ablation.py --only-bag-sizes 25 --only-bags-per-group 100 --only-batch-sizes 32
"""

import argparse
import copy
import csv
import itertools
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import yaml

# ── Study definition ────────────────────────────────────────────────────────────
BAG_SIZES          = [25, 50, 100]
BAGS_PER_GROUP     = [100, 250, 500, 1000, 2000]
BATCH_SIZES        = [16, 32, 64, 128]

WANDB_PROJECT      = "MILAgeScore_Ablation"
VAL_BAGS_PER_GROUP = 30          # pinned: identical validation set size for every run
MAIN_SCRIPT        = "main.py"
DEFAULT_ROOT       = "ablation"
# ────────────────────────────────────────────────────────────────────────────────

PATH_KEYS_DATA = ("features_path", "manual_features_path", "latent_embeddings_path",
                  "labels_path", "processed_dir")


def run_name(bag_size, bpg, batch):
    return f"bs{bag_size:03d}_bpg{bpg:04d}_batch{batch:03d}"


def absolutise_data_paths(cfg, project_root):
    data = cfg.setdefault("data", {})
    for key in PATH_KEYS_DATA:
        val = data.get(key)
        if val is None or str(val).strip().lower() in {"na", "none", "null", ""}:
            continue
        p = Path(str(val))
        if not p.is_absolute():
            data[key] = str((project_root / p).resolve())


def build_run_config(base, bag_size, bpg, batch, run_dir, name, study, project_root, seed):
    cfg = copy.deepcopy(base)
    for section in ("bags", "training", "wandb", "outputs", "evaluate"):
        cfg.setdefault(section, {})

    cfg["bags"]["cells_per_bag"]      = bag_size
    cfg["bags"]["bags_per_group"]     = bpg
    cfg["bags"]["val_bags_per_group"] = VAL_BAGS_PER_GROUP
    if seed is not None:
        cfg["bags"]["random_seed"] = seed
    cfg["training"]["batch_size"] = batch

    cfg["wandb"]["project"]  = WANDB_PROJECT
    cfg["wandb"]["run_name"] = name
    cfg["wandb"]["tags"] = list(base.get("wandb", {}).get("tags", [])) + [
        "ablation", f"bag_size_{bag_size}", f"bags_per_group_{bpg}", f"batch_size_{batch}"]
    cfg["wandb"]["notes"] = (f"Ablation [{study}]: cells_per_bag={bag_size}, "
                             f"bags_per_group={bpg}, batch_size={batch}")

    # Every outputs.*_dir goes inside this run's folder (keeps leaf name).
    out = cfg["outputs"]
    out.setdefault("checkpoint_dir", "checkpoints")
    out.setdefault("log_dir", "logs")
    for key, val in list(out.items()):
        if key.endswith("_dir"):
            leaf = Path(str(val)).name or key[:-4]
            out[key] = str((run_dir / leaf).resolve())

    # None => main.py loads <checkpoint_dir>/<run_name>_best.pt for THIS run.
    cfg["evaluate"]["checkpoint_path"] = None

    absolutise_data_paths(cfg, project_root)
    return cfg


def latest_study(root_parent):
    if not root_parent.exists():
        return None
    studies = [p for p in root_parent.iterdir() if p.is_dir()]
    return max(studies, key=lambda p: p.stat().st_mtime).name if studies else None


def main():
    ap = argparse.ArgumentParser(description="MILAgeScore 3-way ablation runner")
    ap.add_argument("--base-config", default="config/config.yaml")
    ap.add_argument("--root", default=DEFAULT_ROOT)
    ap.add_argument("--name", default=None, help="Study folder name")
    ap.add_argument("--mode", default="all", choices=["train", "evaluate", "explain", "all"])
    ap.add_argument("--seed", type=int, default=None,
                    help="Override bags.random_seed for every run (default: keep base config)")
    ap.add_argument("--resume", action="store_true", help="Skip runs that have a .done marker")
    ap.add_argument("--dry-run", action="store_true", help="Write configs and print commands only")
    ap.add_argument("--stop-on-error", action="store_true")
    ap.add_argument("--only-bag-sizes", type=int, nargs="+", default=None)
    ap.add_argument("--only-bags-per-group", type=int, nargs="+", default=None)
    ap.add_argument("--only-batch-sizes", type=int, nargs="+", default=None)
    args = ap.parse_args()

    project_root = Path(__file__).resolve().parent
    base_path = Path(args.base_config)
    if not base_path.is_absolute():
        base_path = project_root / base_path
    with open(base_path, encoding="utf-8") as f:
        base = yaml.safe_load(f)

    root_parent = project_root / args.root
    study = args.name
    if study is None and args.resume:
        study = latest_study(root_parent)
        if study:
            print(f"--resume without --name: continuing latest study '{study}'")
    if study is None:
        study = f"{WANDB_PROJECT}_{datetime.now():%Y%m%d_%H%M%S}"
    root = (root_parent / study).resolve()
    root.mkdir(parents=True, exist_ok=True)

    bag_sizes = args.only_bag_sizes or BAG_SIZES
    bpgs      = args.only_bags_per_group or BAGS_PER_GROUP
    batches   = args.only_batch_sizes or BATCH_SIZES
    grid = list(itertools.product(bag_sizes, bpgs, batches))

    with open(root / "ablation_config.json", "w", encoding="utf-8") as f:
        json.dump({"wandb_project": WANDB_PROJECT, "base_config": str(base_path),
                   "bag_sizes": bag_sizes, "bags_per_group": bpgs, "batch_sizes": batches,
                   "val_bags_per_group": VAL_BAGS_PER_GROUP, "mode": args.mode,
                   "seed_override": args.seed, "n_runs": len(grid)}, f, indent=2)

    print(f"Study folder : {root}")
    print(f"W&B project  : {WANDB_PROJECT}")
    print(f"Validation   : scheme={base.get('validation', {}).get('scheme')}  "
          f"(held_out_line={base.get('validation', {}).get('held_out_line')})")
    print(f"Total runs   : {len(grid)}\n")

    # Windows-safe: main.py/print_config prints Unicode box characters, which crash
    # under the default cp1252 encoding when stdout is redirected to a file.
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    env["WANDB_RUN_GROUP"] = study      # groups all runs of this study in W&B

    summary_path = root / "summary.csv"
    new_file = not summary_path.exists()
    summary_f = open(summary_path, "a", newline="", encoding="utf-8")
    writer = csv.writer(summary_f)
    if new_file:
        writer.writerow(["run_name", "bag_size", "bags_per_group", "batch_size",
                         "status", "return_code", "minutes"])

    n_ok = n_fail = n_skip = 0
    t_study = time.time()

    try:
        for i, (bag_size, bpg, batch) in enumerate(grid, 1):
            name = run_name(bag_size, bpg, batch)
            run_dir = root / name
            done = run_dir / ".done"
            tag = f"[{i}/{len(grid)}] {name}"

            if args.resume and done.exists():
                print(f"{tag}  -> skipped (already done)")
                n_skip += 1
                continue

            run_dir.mkdir(parents=True, exist_ok=True)
            cfg = build_run_config(base, bag_size, bpg, batch, run_dir, name,
                                   study, project_root, args.seed)
            cfg_path = run_dir / "config.yaml"
            with open(cfg_path, "w", encoding="utf-8") as f:
                yaml.safe_dump(cfg, f, sort_keys=False)

            cmd = [sys.executable, str(project_root / MAIN_SCRIPT),
                   "--config", str(cfg_path), "--mode", args.mode]
            print(f"{tag}\n    {' '.join(cmd)}")
            if args.dry_run:
                continue

            if done.exists():
                done.unlink()
            t0 = time.time()
            stdout_path = run_dir / "stdout.log"
            with open(stdout_path, "w", encoding="utf-8") as log:
                proc = subprocess.run(cmd, cwd=project_root, env=env,
                                      stdout=log, stderr=subprocess.STDOUT)
            minutes = (time.time() - t0) / 60

            if proc.returncode == 0:
                done.touch()
                status = "ok"
                n_ok += 1
                print(f"    -> OK ({minutes:.1f} min)")
            else:
                status = "failed"
                n_fail += 1
                print(f"    -> FAILED, return code {proc.returncode} ({minutes:.1f} min)")
                print(f"       See: {stdout_path}")

            writer.writerow([name, bag_size, bpg, batch, status, proc.returncode,
                             f"{minutes:.2f}"])
            summary_f.flush()

            if proc.returncode != 0 and args.stop_on_error:
                print("Stopping on error.")
                break
    except KeyboardInterrupt:
        print("\nInterrupted. Re-run with --resume to continue.")
    finally:
        summary_f.close()

    print(f"\nFinished: {n_ok} ok, {n_fail} failed, {n_skip} skipped "
          f"in {(time.time() - t_study) / 60:.1f} min")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
