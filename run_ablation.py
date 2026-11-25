#!/usr/bin/env python
"""
run_ablation.py - MILAgeScore ablation with binned stratified K-fold and seed repeats

Grid (bag_size x bags_per_group x batch_size) x folds x seeds.  Every run uses
validation.scheme = "binned_kfold" (see src/binning.py): age_group is the binned
column, each bin appears in every fold's validation set.

Layout:
  ablation/<study>/
    ablation_config.json
    runs.csv                  one row per (config, seed, fold)  [rebuilt by --aggregate]
    aggregate.csv             one row per config: mean/SD over folds and seeds
    bs050_bpg0100_batch064/seed0/fold0/{config.yaml, stdout.log, .done, metrics.json, ...}

Usage (project root):
  python run_ablation.py --dry-run
  python run_ablation.py --name cv1                           # screen: 1 seed x 5 folds
  python run_ablation.py --name cv1 --resume
  python run_ablation.py --name cv2 --seeds 0 1 2 --configs-from ablation/cv1/aggregate.csv --top-k 4
  python run_ablation.py --name cv1 --aggregate               # re-aggregate only
  python run_ablation.py --only-bag-sizes 50 --only-bags-per-group 100 --only-batch-sizes 64 --folds 0

  MAX RUN:
  python run_ablation.py --name cv1 --mode all

"""
import argparse, copy, csv, itertools, json, os, re, subprocess, sys, time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

# -- Study definition ----------------------------------------------------------
BAG_SIZES      = [50, 100]
BAGS_PER_GROUP = [50, 100, 200, 400]     # now per BIN (10 groups, not 22+)
BATCH_SIZES    = [32, 64, 128]
N_BINS         = 10
N_FOLDS        = 5
SEEDS          = [0]
FOLD_SEED      = 42                      # fixed -> identical folds for every config/seed
VAL_BAGS_PER_GROUP = 30
RANK_BY        = "val/mae"               # lower = better
WANDB_PROJECT  = "MILAgeScore_Ablation_v3"
MAIN_SCRIPT    = "main.py"
DEFAULT_ROOT   = "ablation"
METRICS_JSON   = "best_val_metrics.json" # written by main.run_train (patch below)
# ------------------------------------------------------------------------------

PATH_KEYS_DATA = ("features_path", "manual_features_path", "latent_embeddings_path",
                  "labels_path", "processed_dir")


def config_name(bag_size, bpg, batch):
    return f"bs{bag_size:03d}_bpg{bpg:04d}_batch{batch:03d}"


def parse_config_name(name):
    m = re.fullmatch(r"bs(\d+)_bpg(\d+)_batch(\d+)", name)
    return tuple(int(x) for x in m.groups())


def absolutise_data_paths(cfg, project_root):
    data = cfg.setdefault("data", {})
    for key in PATH_KEYS_DATA:
        val = data.get(key)
        if val is None or str(val).strip().lower() in {"na", "none", "null", ""}:
            continue
        p = Path(str(val))
        if not p.is_absolute():
            data[key] = str((project_root / p).resolve())


def build_run_config(base, bag_size, bpg, batch, seed, fold, run_dir, run_name, study,
                     project_root, n_bins, n_folds):
    cfg = copy.deepcopy(base)
    for section in ("bags", "training", "wandb", "outputs", "evaluate"):
        cfg.setdefault(section, {})

    cfg["bags"]["cells_per_bag"] = bag_size
    cfg["bags"]["bags_per_group"] = bpg
    cfg["bags"]["val_bags_per_group"] = VAL_BAGS_PER_GROUP
    cfg["bags"]["random_seed"] = seed
    cfg["training"]["batch_size"] = batch

    cfg["validation"] = {"scheme": "binned_kfold", "n_bins": n_bins,
                         "n_folds": n_folds, "fold": fold, "fold_seed": FOLD_SEED}

    cfg["wandb"]["project"] = WANDB_PROJECT
    cfg["wandb"]["run_name"] = run_name
    cfg["wandb"]["tags"] = list(base.get("wandb", {}).get("tags", [])) + [
        "ablation", "binned_kfold", f"bag_size_{bag_size}", f"bags_per_group_{bpg}",
        f"batch_size_{batch}", f"seed_{seed}", f"fold_{fold}"]
    cfg["wandb"]["notes"] = (f"Ablation [{study}]: cells_per_bag={bag_size}, bags_per_group={bpg}, "
                             f"batch_size={batch}, seed={seed}, fold={fold}/{n_folds}, bins={n_bins}")

    out = cfg["outputs"]
    out.setdefault("checkpoint_dir", "checkpoints")
    out.setdefault("log_dir", "logs")
    for key, val in list(out.items()):
        if key.endswith("_dir"):
            leaf = Path(str(val)).name or key[:-4]
            out[key] = str((run_dir / leaf).resolve())

    cfg["evaluate"]["checkpoint_path"] = None
    absolutise_data_paths(cfg, project_root)
    return cfg


def latest_study(root_parent):
    if not root_parent.exists():
        return None
    studies = [p for p in root_parent.iterdir() if p.is_dir()]
    return max(studies, key=lambda p: p.stat().st_mtime).name if studies else None


# -- Results collection ---------------------------------------------------------
def collect_runs(root: Path) -> pd.DataFrame:
    rows = []
    for mpath in root.glob("bs*_bpg*_batch*/seed*/fold*/metrics.json"):
        fold_dir = mpath.parent
        seed = int(fold_dir.parent.name.replace("seed", ""))
        fold = int(fold_dir.name.replace("fold", ""))
        cname = fold_dir.parent.parent.name
        bs, bpg, batch = parse_config_name(cname)
        with open(mpath, encoding="utf-8") as f:
            m = json.load(f)
        rows.append({"config": cname, "bag_size": bs, "bags_per_group": bpg,
                     "batch_size": batch, "seed": seed, "fold": fold, **m})
    return pd.DataFrame(rows)


def aggregate(root: Path, rank_by: str = RANK_BY):
    runs = collect_runs(root)
    if runs.empty:
        print("No finished runs with metrics.json found.")
        return None
    runs.sort_values(["config", "seed", "fold"]).to_csv(root / "runs.csv", index=False)

    keys = ["config", "bag_size", "bags_per_group", "batch_size"]
    metric_cols = [c for c in runs.columns
                   if c not in keys + ["seed", "fold"] and pd.api.types.is_numeric_dtype(runs[c])]
    out = []
    for cfg_key, g in runs.groupby(keys):
        row = dict(zip(keys, cfg_key))
        row["n_runs"] = len(g)
        row["n_seeds"] = g["seed"].nunique()
        row["n_folds"] = g["fold"].nunique()
        for c in metric_cols:
            per_seed = g.groupby("seed")[c].mean()      # average folds within a seed
            per_fold = g.groupby("fold")[c].mean()      # average seeds within a fold
            row[f"{c}_mean"] = g[c].mean()
            row[f"{c}_sd_folds"] = per_fold.std(ddof=1) if len(per_fold) > 1 else np.nan
            row[f"{c}_sd_seeds"] = per_seed.std(ddof=1) if len(per_seed) > 1 else np.nan
        out.append(row)
    agg = pd.DataFrame(out)

    sort_col = f"{rank_by}_mean"
    if sort_col in agg.columns:
        agg = agg.sort_values(sort_col).reset_index(drop=True)
        agg.insert(0, "rank", np.arange(1, len(agg) + 1))
    agg.to_csv(root / "aggregate.csv", index=False)

    show = [c for c in ["rank", "config", "n_runs", f"{rank_by}_mean", f"{rank_by}_sd_folds",
                        f"{rank_by}_sd_seeds", "val/spearman_rho_mean", "val/mae_s_mean"]
            if c in agg.columns]
    print("\nAggregate (lower is better for", rank_by + "):")
    print(agg[show].head(15).to_string(index=False))
    print(f"\nWrote {root / 'runs.csv'} and {root / 'aggregate.csv'}")
    return agg


# -- Main ------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="MILAgeScore binned K-fold ablation runner")
    ap.add_argument("--base-config", default="config/config.yaml")
    ap.add_argument("--root", default=DEFAULT_ROOT)
    ap.add_argument("--name", default=None, help="Study folder name")
    ap.add_argument("--mode", default="train", choices=["train", "evaluate", "explain", "all"],
                    help="default 'train' - evaluate/explain (IG) per fold is wasteful in a sweep")
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS, help="bag/init seeds")
    ap.add_argument("--folds", type=int, nargs="+", default=None, help="subset of fold ids")
    ap.add_argument("--n-folds", type=int, default=N_FOLDS)
    ap.add_argument("--n-bins", type=int, default=N_BINS)
    ap.add_argument("--rank-by", default=RANK_BY)
    ap.add_argument("--resume", action="store_true", help="Skip runs that have a .done marker")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--stop-on-error", action="store_true")
    ap.add_argument("--aggregate", action="store_true", help="Only (re)build runs.csv/aggregate.csv")
    ap.add_argument("--only-bag-sizes", type=int, nargs="+", default=None)
    ap.add_argument("--only-bags-per-group", type=int, nargs="+", default=None)
    ap.add_argument("--only-batch-sizes", type=int, nargs="+", default=None)
    ap.add_argument("--configs-from", default=None,
                    help="aggregate.csv from a previous study; restrict grid to its top-k configs")
    ap.add_argument("--top-k", type=int, default=4)
    args = ap.parse_args()

    project_root = Path(__file__).resolve().parent
    base_path = Path(args.base_config)
    if not base_path.is_absolute():
        base_path = project_root / base_path
    with open(base_path, encoding="utf-8") as f:
        base = yaml.safe_load(f)

    root_parent = project_root / args.root
    study = args.name
    if study is None and (args.resume or args.aggregate):
        study = latest_study(root_parent)
        if study:
            print(f"Using latest study '{study}'")
    if study is None:
        study = f"{WANDB_PROJECT}_{datetime.now():%Y%m%d_%H%M%S}"
    root = (root_parent / study).resolve()
    root.mkdir(parents=True, exist_ok=True)

    if args.aggregate:
        aggregate(root, args.rank_by)
        return

    # configs to run
    if args.configs_from:
        prev = pd.read_csv(args.configs_from).sort_values("rank").head(args.top_k)
        configs = [parse_config_name(c) for c in prev["config"]]
        print(f"Restricting to top-{args.top_k} configs from {args.configs_from}:")
        for c in prev["config"]:
            print("  ", c)
    else:
        bag_sizes = args.only_bag_sizes or BAG_SIZES
        bpgs = args.only_bags_per_group or BAGS_PER_GROUP
        batches = args.only_batch_sizes or BATCH_SIZES
        configs = list(itertools.product(bag_sizes, bpgs, batches))

    folds = args.folds if args.folds is not None else list(range(args.n_folds))
    bad = [x for x in folds if not 0 <= x < args.n_folds]
    if bad:
        sys.exit(f"--folds {bad} outside [0, {args.n_folds - 1}]")
    grid = [(c, s, k) for c in configs for s in args.seeds for k in folds]

    with open(root / "ablation_config.json", "w", encoding="utf-8") as f:
        json.dump({"wandb_project": WANDB_PROJECT, "base_config": str(base_path),
                   "configs": [config_name(*c) for c in configs], "seeds": args.seeds,
                   "folds": folds, "n_folds": args.n_folds, "n_bins": args.n_bins,
                   "fold_seed": FOLD_SEED, "val_bags_per_group": VAL_BAGS_PER_GROUP,
                   "mode": args.mode, "n_runs": len(grid)}, f, indent=2)

    print(f"Study folder : {root}")
    print(f"W&B project  : {WANDB_PROJECT}")
    print(f"Validation   : binned_kfold  bins={args.n_bins}  folds={folds} of {args.n_folds}")
    print(f"Configs x seeds x folds : {len(configs)} x {len(args.seeds)} x {len(folds)} = {len(grid)} runs\n")

    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"

    n_ok = n_fail = n_skip = 0
    t_study = time.time()
    log_path = root / "run_log.csv"
    new_log = not log_path.exists()
    log_f = open(log_path, "a", newline="", encoding="utf-8")
    writer = csv.writer(log_f)
    if new_log:
        writer.writerow(["config", "seed", "fold", "status", "return_code", "minutes"])

    try:
        for i, ((bag_size, bpg, batch), seed, fold) in enumerate(grid, 1):
            cname = config_name(bag_size, bpg, batch)
            run_name = f"{cname}_s{seed}_f{fold}"
            run_dir = root / cname / f"seed{seed}" / f"fold{fold}"
            done = run_dir / ".done"
            tag = f"[{i}/{len(grid)}] {run_name}"

            if args.resume and done.exists():
                print(f"{tag} -> skipped (already done)")
                n_skip += 1
                continue

            run_dir.mkdir(parents=True, exist_ok=True)
            cfg = build_run_config(base, bag_size, bpg, batch, seed, fold, run_dir, run_name,
                                   study, project_root, args.n_bins, args.n_folds)
            cfg_path = run_dir / "config.yaml"
            with open(cfg_path, "w", encoding="utf-8") as f:
                yaml.safe_dump(cfg, f, sort_keys=False)

            cmd = [sys.executable, str(project_root / MAIN_SCRIPT),
                   "--config", str(cfg_path), "--mode", args.mode]
            print(f"{tag}\n  {' '.join(cmd)}")
            if args.dry_run:
                continue

            if done.exists():
                done.unlink()
            run_env = dict(env, WANDB_RUN_GROUP=f"{study}/{cname}")   # one W&B group per config
            t0 = time.time()
            with open(run_dir / "stdout.log", "w", encoding="utf-8") as log:
                proc = subprocess.run(cmd, cwd=project_root, env=run_env,
                                      stdout=log, stderr=subprocess.STDOUT)
            minutes = (time.time() - t0) / 60

            metrics_src = Path(cfg["outputs"]["log_dir"]) / METRICS_JSON
            if proc.returncode == 0 and metrics_src.exists():
                (run_dir / "metrics.json").write_text(metrics_src.read_text(encoding="utf-8"),
                                                      encoding="utf-8")
                done.touch()
                status = "ok"
                n_ok += 1
                print(f"  -> OK ({minutes:.1f} min)")
            elif proc.returncode == 0:
                status = "no_metrics"
                n_fail += 1
                print(f"  -> finished but {METRICS_JSON} missing (main.py patch applied?)")
            else:
                status = "failed"
                n_fail += 1
                print(f"  -> FAILED, return code {proc.returncode} ({minutes:.1f} min)\n"
                      f"     See: {run_dir / 'stdout.log'}")

            writer.writerow([cname, seed, fold, status, proc.returncode, f"{minutes:.2f}"])
            log_f.flush()
            if status != "ok" and args.stop_on_error:
                print("Stopping on error.")
                break
    except KeyboardInterrupt:
        print("\nInterrupted. Re-run with --resume to continue.")
    finally:
        log_f.close()

    print(f"\nFinished: {n_ok} ok, {n_fail} failed, {n_skip} skipped in "
          f"{(time.time() - t_study) / 60:.1f} min")
    if not args.dry_run:
        aggregate(root, args.rank_by)


if __name__ == "__main__":
    main()
