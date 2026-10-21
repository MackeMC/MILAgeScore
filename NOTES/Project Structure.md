
## File Structure

```
MILAgeScore/
│
├── config/
│   └── config.yaml              # all hyperparameters and data paths in one place
│
├── data/
│   ├── raw/                     # original CSVs, .npy latent embeddings — never modified
│   └── processed/               # cached scaled feature arrays, scaler .pkl files
│
├── src/
│   ├── __init__.py
│   ├── dataset.py               # bag construction, BagDataset, DataLoader factory
│   ├── model.py                 # AttentionMIL aggregator + CORN ordinal head
│   ├── losses.py                # CORN loss wrapper, any auxiliary losses
│   ├── train.py                 # training loop + validation loop
│   ├── evaluate.py              # age score inference, MAE, Spearman ρ, concordance
│   ├── explain.py               # attention weight extraction, Integrated Gradients
│   └── utils.py                 # seeding, logging, normalization, checkpoint I/O
│
├── notebooks/
│   ├── 01_eda.ipynb             # feature distributions, passage separation, correlation
│   └── 02_visualise.ipynb       # UMAP with attention-weighted features, age score plots
│
├── outputs/
│   ├── checkpoints/             # saved model .pt files keyed by run name
│   ├── logs/                    # CSV or TensorBoard training logs
│   └── figures/                 # exported plots (UMAP, age score per passage, attention)
│
├── tests/
│   ├── test_dataset.py          # bag construction correctness, no well-leakage
│   └── test_model.py            # forward pass shape checks, CORN output monotonicity
│
├── main.py                      # CLI entry point — wires config → data → train → eval
├── requirements.txt
└── README.md
```
## `src/dataset.py`

Handles all data wrangling. Its two responsibilities are completely separate from the model:

- **Preprocessing:** load raw data, z-score manual and latent features independently, persist scalers to `data/processed/` so they can be reapplied to treatment samples later without refitting
- **Bag construction:** pool cells by passage × cell line, subsample bags with per-well capping, return a `BagDataset` compatible with `torch.utils.data.DataLoader`

---

## `src/model.py`

Contains two `nn.Module` classes and one wrapper:

- **`AttentionMILAggregator`:** takes a bag of instance vectors, returns a single bag-level embedding and the per-instance attention weights \(a\)
- **`CORNHead`:** the CORN ordinal output head (from `coral-pytorch`) that maps the bag embedding to ordinal logits
- **`MILAgeScoreModel`:** composes the above two; its `forward()` returns `(logits, attention_weights)`

---

## `src/losses.py`

A thin wrapper around `coral_pytorch.losses.corn_loss` to handle the normalized passage score \(s\) → discrete rank mapping and expose a clean interface to the training loop.

---

## `src/train.py`

The training and validation loops only. Accepts a model, dataloaders, optimizer, and config. Writes metrics to the log directory. Has no knowledge of data loading or model architecture details — only calls `model(bag)` and `loss(logits, labels)`.

---

## `src/evaluate.py`

Post-training inference. Key functions:

- `compute_age_score(model, bag)` → scalar predicted \(s\) per bag, averaged over multiple subsampled bags for stability
- `compare_groups(scores_treated, scores_control)` → statistical test (Mann–Whitney U for non-parametric, given expected small group sizes)
- `spearman_by_passage(model, dataset)` → rank correlation of predicted vs. true \(s\) — your primary validation metric

---

## `src/explain.py`

Feature attribution after training:

- Extract per-instance attention weights \(a_i\) — these tell you which cells within a sample are driving the age score
- Use `captum.attr.IntegratedGradients` on the `MILAgeScoreModel` to get per-feature attributions across the input vector — these tell you which features within those cells are driving attention

---

## `src/utils.py`

Stateless helper functions: `set_seed()`, `save_checkpoint()`, `load_checkpoint()`, `get_logger()`, and the feature scaling logic.

---

## `main.py`

The single entry point. Parses `config.yaml`, calls `dataset.py` to build loaders, instantiates the model from `model.py`, calls `train.py`, then `evaluate.py`. Should be runnable as:

```bash
python main.py --config config/config.yaml
```
