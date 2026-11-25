# Project Guidelines
    
Project: attention-MIL + CORN ordinal model that scores replicative age of MSC populations from
single-cell features. Two cell lines: line "1" (passages P4–P15) and line "2" (P4–P18). Each
(line, passage) was imaged in its own session, 5–6 wells each. Labels: s = (passage - p_min) /
(p_max - p_min) per line, then binned into 10 equal-width bins (binning.py). The key experiment is
"LOCO": train on one line, predict the other (config_loco.yaml holds out line 2, config_loco2.yaml
holds out line 1).

Ground rules for every task:
1. Work on a new git branch. Do not change behaviour of existing configs: every new config key must
   default to the current behaviour.
2. Never overwrite existing output folders. Write everything to outputs/review/<task_id>/...
3. No hyperparameter tuning. Do not change model sizes, learning rate, or dropout unless the task says so.
4. Each run must save: the resolved config (yaml), the git commit hash, the seed, and a per-epoch
   metrics CSV.
5. All result tables must be plain CSV with one row per unit and explicit columns for
   run_id, seed, train_line, held_out_line, cell_line, passage_raw (the real passage number, 4–18).
6. At the end, write outputs/review/<task_id>/REPORT.md listing: what was changed (files/functions),
   the exact commands run, any deviations from this brief, and any warnings or errors seen.
7. If something in the brief conflicts with the code, stop and say so in REPORT.md. Do not guess.
