
Based on paper: **Expanding Multiple Instance Learning Methodology to Ordinal
Outcomes and Distributional Instances** Sean Kent, (https://asset.library.wisc.edu/1711.dl/KELXLGXS4LL6F8E/R/file-69b29.pdf)

This paper is directly relevant and validates our architecture. Key findings from Chapter 3 that inform losses.py:

CORN and CORAL consistently dominate CLM-QWK on both MAE and RMSE metrics across all datasets
​

Our exact architecture (MI-NET with gated attention → CORN head) is the best-performing combination on tabular TMA data
​

The paper evaluates three metrics — MAE, MZOE (mean zero-one error), and RMSE — and recommends MAE as the primary metric for ordinal problems with many rank levels
​

Two ordinal-MIL architectures are explored: bag-level CORN (our approach, Architecture A in Figure 3.5) and instance-level CORN then pooled (Architecture B). Architecture A is what we have implemented, which is the standard and correct approach for our bag-label-only setting
​

There is also one important architectural nuance to note: the paper's best-performing models on tabular TMA data used CORN with MI-NET ATTENTION and found that attention pooling outperformed max-pooling specifically on tabular data — directly validating our gated attention choice.
​

**Key Design Decisions from the Paper**
CORN over CORAL: Kent's large-scale experiment (7,200 models) found CORN and CORAL perform similarly on MAE, but CORAL dominates MZOE. Since our primary thesis metric is biological interpretability of the age score (continuous, MAE-like), CORN is preferred — its lack of weight-sharing constraint gives the head more expressive freedom for the relatively small 27-class ordinal problem.

Non-uniform s values require explicit encoding: Because the two lines have denominators of 11 and 14, their s grids do not align (e.g., line 1 has s=0.0909 while line 2 has s=0.0714). The OrdinalLabelEncoder jointly sorts all 27 unique s values and assigns them sequential rank indices 0–26 — this is the correct treatment since CORN operates on integer ranks, not continuous scores.
​

MAE as primary metric: Kent explicitly favours MAE over MZOE for datasets with many ordinal levels, since MZOE can be misleadingly low by predicting high-frequency classes. With 27 ranks, this is directly applicable to your dataset. Spearman ρ is added as a secondary metric for biological interpretability of the age trajectory.
​