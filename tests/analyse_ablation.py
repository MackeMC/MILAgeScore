# %% 1. Config
import re, warnings, itertools
from pathlib import Path
import numpy as np, pandas as pd, matplotlib.pyplot as plt
warnings.filterwarnings("ignore"); pd.set_option("display.width", 220, "display.max_columns", 60)
STUDY_DIR = Path(r"ablation/study1"); WANDB_ENTITY = None; WANDB_PROJECT = "MILAgeScore_Ablation"; USE_WANDB = True; TOP_K = 15
TL, VL, VM, VMS, VR, EP = "train/loss_corn", "val/loss", "val/mae", "val/mae_s", "val/spearman_rho", "epoch"
OUT = STUDY_DIR / "analysis"; OUT.mkdir(parents=True, exist_ok=True)
FACT = [("bag_size", "Bag size"), ("bpg", "Bags per group"), ("batch", "Batch size")]
def save(fig, name): fig.savefig(OUT / name, dpi=200, bbox_inches="tight"); plt.show()

# %% 2. Discover runs
pat = re.compile(r"bs(\d+)_bpg(\d+)_batch(\d+)")
runs = pd.DataFrame([dict(run=p.name, bag_size=int(m[1]), bpg=int(m[2]), batch=int(m[3]), dir=p, done=(p / ".done").exists()) for p in sorted(STUDY_DIR.iterdir()) if p.is_dir() and (m := pat.fullmatch(p.name))])
print(f"{len(runs)} runs, {runs.done.sum()} done")

# %% 3. Split sizes per run (from stdout.log)
def split_info(r):
    t = (r.dir / "stdout.log").read_text(encoding="utf-8", errors="ignore"); g = lambda p: (re.search(p, t) or [None, np.nan])[1]
    return dict(run=r.run, train_groups=float(g(r"Train groups\s*:\s*(\d+)")), val_groups=float(g(r"Val groups\s*:\s*(\d+)")), val_list=g(r"Val groups\s*:\s*(\[.*\])"))
splits = pd.DataFrame([split_info(r) for r in runs.itertuples()])
print(splits[["train_groups", "val_groups"]].drop_duplicates().to_string(index=False)); print("val groups (run 1):", splits.val_list.iloc[0])

# %% 4. Pull wandb histories
histories = {}
if USE_WANDB:
    import wandb
    api = wandb.Api(); ent = WANDB_ENTITY or api.default_entity; wb = {}
    for r in api.runs(f"{ent}/{WANDB_PROJECT}"):
        if r.name in set(runs.run) and (r.name not in wb or r.created_at > wb[r.name].created_at): wb[r.name] = r
    histories = {n: r.history(samples=100000, pandas=True) for n, r in wb.items()}
    assert histories, "No matching wandb runs - check entity/project/login."
    first = next(iter(histories.values())); miss = [c for c in (TL, VL, VM, VMS, VR, EP) if c not in first.columns]
    assert not miss, f"Missing wandb columns {miss}. Available: {[c for c in first.columns if not c.startswith('_')]}"
    print(f"matched {len(histories)}/{len(runs)} runs; metrics OK")

# %% 4b. Per-run summary (best epoch = lowest val/mae, same rule as train.py)
def epoch_table(h): return h.groupby(EP)[[TL, VL, VM, VMS, VR]].first().dropna(subset=[VM])
def summarise(n, h):
    ep = epoch_table(h); b = ep.loc[ep[VM].idxmin()]
    return dict(run=n, n_epochs=len(ep), best_epoch=ep[VM].idxmin(), train_loss=b[TL], val_loss=b[VL], val_mae=b[VM], val_mae_s=b[VMS], val_rho=b[VR], gap=b[VL] - b[TL])
if histories:
    d = runs.drop(columns="dir").merge(pd.DataFrame([summarise(n, h) for n, h in histories.items()]), on="run").merge(splits.drop(columns="val_list"), on="run")
    d["steps_per_epoch"] = d.train_groups * d.bpg // d.batch; d["steps_to_best"] = d.steps_per_epoch * d.best_epoch; d["cells_per_epoch"] = d.train_groups * d.bpg * d.bag_size
    d["rank_mae"], d["rank_rho"] = d.val_mae.rank(), (-d.val_rho).rank(); d["combined_rank"] = d.rank_mae + d.rank_rho
    d.to_csv(OUT / "run_summary.csv", index=False); print(f"{len(d)} usable runs")

# %% 5. Best runs
if histories:
    show = ["run", "train_loss", "val_loss", "val_mae", "val_mae_s", "val_rho", "gap", "best_epoch", "steps_per_epoch"]
    print("== Best by val MAE + Spearman rank =="); print(d.sort_values("combined_rank")[show].head(10).to_string(index=False))
    print("\n== Best by val MAE_s =="); print(d.sort_values("val_mae_s")[show].head(5).to_string(index=False))
    print("\n== Best by val CORN loss (secondary) =="); print(d.sort_values("val_loss")[show].head(5).to_string(index=False))
    print("\nNote: train loss has dropout ON, val has it OFF, so the gap is not a clean overfitting measure.")

# %% 6. Marginal effects + standardised log2 fit
if histories:
    mets = ["val_mae", "val_mae_s", "val_rho", "val_loss", "train_loss", "gap", "best_epoch", "steps_per_epoch"]
    for f, lab in FACT: t = d.groupby(f)[mets].mean(); print(f"\n--- {lab} (mean over other factors) ---\n{t.round(4)}"); t.to_csv(OUT / f"marginal_{f}.csv")
    X = np.log2(d[["bag_size", "bpg", "batch"]].astype(float)); X = (X - X.mean()) / X.std(); A = np.c_[np.ones(len(X)), X.values]
    for tg in ("val_mae", "val_rho", "val_loss", "gap"): y = d[tg].values; c = np.linalg.lstsq(A, (y - y.mean()) / y.std(), rcond=None)[0]; print(f"{tg:9s} std. coef:", dict(zip(["bag_size", "bpg", "batch"], c[1:].round(3))))
    print("(linear, single seed: indicative only)")

# %% 7. Step-count confound
if histories:
    print(d[["bag_size", "bpg", "batch", "steps_per_epoch", "cells_per_epoch", "steps_to_best", "best_epoch", "val_mae", "val_mae_s", "val_rho", "val_loss"]].corr(method="spearman")[["val_mae", "val_mae_s", "val_rho", "val_loss"]].round(2))
    print("\nbest_epoch by bpg x batch:"); print(d.pivot_table(index="bpg", columns="batch", values="best_epoch").round(1))
    fig, ax = plt.subplots(1, 3, figsize=(15, 3.8))
    for a, m in zip(ax, ("val_mae", "val_rho", "val_loss")):
        for bs, g in d.groupby("bag_size"): a.scatter(g.steps_per_epoch, g[m], label=f"bag {bs}", alpha=.8)
        a.set(xscale="log", xlabel="steps per epoch", ylabel=m); a.grid(alpha=.3); a.legend()
    save(fig, "steps_confound.png")

# %% 8. Heatmaps
def heat(metric, title, cmap="viridis_r"):
    bs = sorted(d.batch.unique()); fig, axes = plt.subplots(1, len(bs), figsize=(4.2 * len(bs), 3.6), sharey=True); axes = np.atleast_1d(axes)
    for ax, b in zip(axes, bs):
        p = d[d.batch == b].pivot(index="bpg", columns="bag_size", values=metric).sort_index(); im = ax.imshow(p.values, cmap=cmap, vmin=d[metric].min(), vmax=d[metric].max(), aspect="auto")
        ax.set_xticks(range(p.shape[1]), p.columns); ax.set_yticks(range(p.shape[0]), p.index); ax.set_title(f"batch {b}"); ax.set_xlabel("bag size")
        for i, j in itertools.product(range(p.shape[0]), range(p.shape[1])): ax.text(j, i, f"{p.values[i, j]:.3f}", ha="center", va="center", color="w", fontsize=8)
    axes[0].set_ylabel("bags per group"); fig.suptitle(title); fig.colorbar(im, ax=list(axes), shrink=.8); save(fig, f"heatmap_{metric}.png")
if histories:
    S = "at best val-MAE epoch"
    for m, t, c in [("val_mae", "Val rank MAE", "viridis_r"), ("val_mae_s", "Val MAE (s units)", "viridis_r"), ("val_rho", "Val Spearman rho", "viridis"), ("val_loss", "Val CORN loss", "viridis_r"), ("train_loss", "Train CORN loss", "viridis_r"), ("gap", "Val - train CORN loss", "magma"), ("best_epoch", "Best epoch", "viridis")]: heat(m, f"{t} {S}", c)

# %% 9. Factor lines
if histories:
    fig, ax = plt.subplots(2, 3, figsize=(15, 7))
    for r_, m in enumerate(("val_mae", "val_rho")):
        for a, (f, lab) in zip(ax[r_], FACT):
            for bs, g in d.groupby("bag_size"): s = g.groupby(f)[m].mean(); a.plot(s.index, s.values, "o-", label=f"bag {bs}")
            a.set(xscale="log", xlabel=lab, ylabel=m); a.grid(alpha=.3); a.legend()
    save(fig, "factor_lines.png")

# %% 10. Learning curves (best 3 vs worst 2 by combined rank)
if histories:
    o = d.sort_values("combined_rank").run; fig, ax = plt.subplots(1, 3, figsize=(16, 3.8))
    for n in list(o.head(3)) + list(o.tail(2)):
        e = epoch_table(histories[n])
        for a, c in zip(ax, (TL, VL, VM)): a.plot(e.index, e[c], label=n)
    for a, t in zip(ax, ("train CORN loss", "val CORN loss", "val rank MAE")): a.set_title(t); a.set_xlabel("epoch"); a.grid(alpha=.3)
    ax[2].legend(fontsize=7); save(fig, "learning_curves.png")

# %% 11. Attention - inspect
def load_att(r): p = r.dir / "logs" / f"{r.run}_attention.csv"; return pd.read_csv(p) if p.exists() else None
a0 = load_att(next(runs.itertuples())); print(list(a0.columns), len(a0)); print(a0.head())

# %% 11b. Attention metrics (pin ATT_W / ATT_BAG if auto-detect is wrong)
ATT_W, ATT_BAG = None, None
nc = a0.select_dtypes("number").columns.tolist()
ATT_W = ATT_W or next((c for c in nc if re.search(r"attn|attention|weight|alpha", c, re.I)), None); ATT_BAG = ATT_BAG or next((c for c in a0.columns if re.search(r"bag|sample|draw", c, re.I)), None)
print("weight col:", ATT_W, "| bag col:", ATT_BAG); assert ATT_W, "Set ATT_W to the attention-weight column."
def bag_stats(w):
    w = np.asarray(w, float); w = w / w.sum(); h = -(w * np.log(w + 1e-12)).sum(); k = max(1, int(np.ceil(.1 * len(w))))
    return h / np.log(len(w)), w.max() * len(w), np.sort(w)[-k:].sum(), np.exp(h) / len(w)
def att_metrics(r):
    a = load_att(r)
    if a is None: return None
    n = r.bag_size; w = a[ATT_W].astype(float).values
    if ATT_BAG and ATT_BAG in a and a.groupby(ATT_BAG).size().median() == n: gs, mode = [g[ATT_W].values for _, g in a.groupby(ATT_BAG)], "per-bag"
    elif not ATT_BAG and len(a) % n == 0: gs, mode = list(w.reshape(-1, n)), "per-bag(blocks)"
    else: gs, mode = [w], "per-cell"
    s = np.array([bag_stats(g) for g in gs]).mean(0); return dict(run=r.run, mode=mode, n_bags=len(gs), norm_entropy=s[0], max_w_x_uniform=s[1], top10pct_mass=s[2], eff_cell_fraction=s[3])
att = runs.drop(columns="dir").merge(pd.DataFrame([m for m in map(att_metrics, runs.itertuples()) if m]), on="run")
if histories: att = att.merge(d[["run", "val_mae", "val_rho", "val_loss"]], on="run", how="left")
att.to_csv(OUT / "attention_metrics.csv", index=False); print(att["mode"].value_counts().to_dict()); print(att.sort_values("norm_entropy").head(8).to_string(index=False))

# %% 11c. Attention vs factors
AM = ["norm_entropy", "max_w_x_uniform", "top10pct_mass", "eff_cell_fraction"]
for f, _ in FACT: print(f"\n--- {f} ---\n{att.groupby(f)[AM].mean().round(3)}")
if histories: print("\nSpearman vs outcomes:\n", att[AM + ["val_mae", "val_rho", "val_loss"]].corr(method="spearman")[["val_mae", "val_rho", "val_loss"]].round(2))
fig, ax = plt.subplots(1, 3, figsize=(15, 3.8))
for a, (f, _) in zip(ax, FACT): a.scatter(att[f].astype(str), att.norm_entropy, alpha=.7); a.set(xlabel=f, ylabel="norm. attention entropy (1 = uniform)"); a.grid(alpha=.3)
save(fig, "attention_entropy.png")

# %% 12. Feature importance - inspect
def load_imp(r):
    for s in ("importance_normalised", "importance"):
        p = r.dir / "logs" / f"{r.run}_{s}.csv"
        if p.exists(): return pd.read_csv(p)
i0 = load_imp(next(runs.itertuples())); print(list(i0.columns)); print(i0.head())

# %% 12b. Feature x run matrix + consensus
FEAT, VALC = None, None
FEAT = FEAT or next((c for c in i0.columns if i0[c].dtype == object), i0.columns[0]); num_ = i0.select_dtypes("number").columns.tolist()
VALC = VALC or next((c for c in num_ if re.search(r"import|attr|ig|norm", c, re.I)), num_[0]); print("feature col:", FEAT, "| value col:", VALC)
def share(t): s = t.set_index(FEAT)[VALC].abs(); return s / s.sum()
M = pd.DataFrame({r.run: share(t) for r in runs.itertuples() if (t := load_imp(r)) is not None}).fillna(0); rank = M.rank(ascending=False)
cons = pd.DataFrame({"mean_share": M.mean(axis=1), "sd_share": M.std(axis=1), "cv": M.std(axis=1) / M.mean(axis=1), "mean_rank": rank.mean(axis=1), "rank_sd": rank.std(axis=1), "in_top_k_frac": (rank <= TOP_K).mean(axis=1)}).sort_values("mean_rank")
cons.to_csv(OUT / "feature_consensus.csv"); print(M.shape); print(cons.head(25).round(4))

# %% 12c. Run-to-run agreement
sp = M.corr(method="spearman"); top = {c: set(rank[c].nsmallest(TOP_K).index) for c in M}; iu = np.triu_indices(len(M.columns), 1)
jac = np.array([[len(top[a] & top[b]) / len(top[a] | top[b]) for b in M] for a in M])
print(f"Pairwise Spearman: mean {sp.values[iu].mean():.3f}, min {sp.values[iu].min():.3f} | top-{TOP_K} Jaccard: mean {jac[iu].mean():.3f}, min {jac[iu].min():.3f}")
print(f"In the top-{TOP_K} of EVERY run: {list(cons.index[cons.in_top_k_frac == 1])}")
meta = runs.drop(columns="dir").set_index("run").loc[M.columns].copy(); meta["corr_to_consensus"] = [M[c].corr(M.mean(axis=1), method="spearman") for c in M]
if histories: meta = meta.join(d.set_index("run")[["val_mae", "val_rho"]])
for f, _ in FACT: print(f"\nAgreement with consensus by {f}:\n{meta.groupby(f).corr_to_consensus.mean().round(3).to_string()}")
if histories: print("\nSpearman(agreement, outcome):", meta[["corr_to_consensus", "val_mae", "val_rho"]].corr(method="spearman").iloc[0, 1:].round(3).to_dict())

# %% 12d. Importance plots
tf = list(cons.index[:20]); order = [o for o in runs.sort_values(["bag_size", "bpg", "batch"]).run if o in M.columns]
fig, ax = plt.subplots(1, 2, figsize=(16, 6), gridspec_kw=dict(width_ratios=[1, 2.2]))
ax[0].barh(tf[::-1], cons.loc[tf, "mean_share"].values[::-1], xerr=cons.loc[tf, "sd_share"].values[::-1]); ax[0].set_title("Top 20: mean share +/- SD")
im = ax[1].imshow(M.loc[tf, order].values, aspect="auto", cmap="viridis"); ax[1].set_yticks(range(len(tf)), tf); ax[1].set_xticks([]); ax[1].set_xlabel("runs (bag size, bpg, batch)"); fig.colorbar(im, ax=ax[1], label="importance share"); save(fig, "feature_importance_consistency.png")
fig, ax = plt.subplots(figsize=(6, 5)); im = ax.imshow(sp.loc[order, order].values, vmin=sp.values.min(), vmax=1, cmap="magma"); ax.set_title("Run-vs-run Spearman of importance"); fig.colorbar(im); save(fig, "run_similarity.png")
print("saved to", OUT)
