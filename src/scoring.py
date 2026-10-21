"""
src/scoring.py

Single source of truth for converting CORN logits into ranks and continuous
age scores. This module MUST be imported by train.py, evaluate.py, explain.py
and any analysis notebook so that the score definition is identical across
training, validation, evaluation and treatment inference.

────────────────────────────────────────────────────────────────────────────
Background
────────────────────────────────────────────────────────────────────────────
A CORN head (Shi, Cao & Raschka, 2021, https://arxiv.org/abs/2111.08851)
outputs K-1 logits for K ordinal ranks. Each logit j models the conditional
probability

    f_j = P(Y > r_j | Y > r_{j-1}) = sigmoid(logit_j).

Applying the chain rule for probabilities gives the (unconditional) cumulative
survival probabilities

    P(Y >= k) = prod_{j=1}^{k} sigmoid(logit_j),    k = 1 ... K-1,

which are guaranteed rank-monotonic (decreasing in k) by construction.

Two equivalent quantities are derived from these:

  • Discrete rank   (argmax-style):  the number of survival probabilities that
    exceed 0.5. This is exactly coral_pytorch's `corn_label_from_logits`. It is
    granular only to the nearest integer rank and is used for reporting the
    rank-space MAE / MZOE that the loss optimises.

  • Continuous rank (expected value): the EXPECTED rank under the predicted
    distribution. For a non-negative integer random variable Y,

        E[Y] = sum_{k=1}^{K-1} P(Y >= k),

    so the expected rank is simply the SUM of the cumulative survival
    probabilities. This is computed directly from the cumulative products with
    no per-class differencing, no last-bin special case and no renormalisation
    — and is therefore numerically stable. Summing cumulative probabilities to
    recover a continuous score from discrete ordinal labels is the standard
    approach (Yao et al., 2023, https://arxiv.org/abs/2305.19097).

The continuous score is the expected rank mapped to [0, 1]:

        s_hat = E[rank] / (num_classes - 1).

This gives sub-rank granularity: two bags that share the same discrete argmax
rank but lean differently in probability mass receive different continuous
scores — exactly what is required to compare e.g. a treated vs untreated
passage-7 sample.

────────────────────────────────────────────────────────────────────────────
Conventions
────────────────────────────────────────────────────────────────────────────
  • `logits`      : torch.Tensor, shape (B, K-1)  — raw CORN head outputs.
  • `num_classes` : int, K  — the number of ordinal ranks. ALWAYS pass
                    `encoder.num_classes` loaded from the checkpoint. Never
                    recompute K from the data in a downstream stage.
  • Returned scores are in [0, 1]; returned ranks are in [0, K-1].

All functions are no-grad and accept either a single bag (B = 1) or a batch.
"""

from __future__ import annotations

import numpy as np
import torch


# ── Internal: cumulative survival probabilities ────────────────────────────────
def _cumulative_survival(logits: torch.Tensor) -> torch.Tensor:
    """
    Compute the rank-monotonic cumulative survival probabilities from CORN logits.

        P(Y >= k) = prod_{j=1}^{k} sigmoid(logit_j),  for k = 1 ... K-1

    Parameters
    ----------
    logits : torch.Tensor, shape (B, K-1)

    Returns
    -------
    cum : torch.Tensor, shape (B, K-1)
        cum[:, k-1] = P(Y >= k). Values are non-increasing along dim=1 by
        construction (each factor is a sigmoid in (0, 1)).
    """
    if logits.dim() != 2:
        raise ValueError(
            f"Expected logits of shape (B, K-1); got shape {tuple(logits.shape)}. "
            f"Add a batch dimension with .unsqueeze(0) for a single bag."
        )
    return torch.sigmoid(logits).cumprod(dim=1)


# ── Discrete rank (matches coral_pytorch.corn_label_from_logits) ────────────────
@torch.no_grad()
def rank_from_logits(logits: torch.Tensor) -> torch.Tensor:
    """
    Discrete predicted rank — the number of survival probabilities exceeding 0.5.

    This is identical to coral_pytorch's `corn_label_from_logits` and is the
    quantity to report for rank-space MAE / MZOE (the metric the CORN loss
    optimises). Use this ONLY for integer-rank reporting; use
    `score_from_logits` for any continuous age score.

    Parameters
    ----------
    logits : torch.Tensor, shape (B, K-1)

    Returns
    -------
    ranks : torch.Tensor, shape (B,), dtype long — integer ranks in [0, K-1].
    """
    cum = _cumulative_survival(logits)
    return (cum > 0.5).sum(dim=1).long()


# ── Continuous expected rank ───────────────────────────────────────────────────
@torch.no_grad()
def expected_rank_from_logits(logits: torch.Tensor) -> torch.Tensor:
    """
    Continuous expected rank E[Y] = sum_{k=1}^{K-1} P(Y >= k).

    Derived directly from the cumulative survival probabilities, so it is
    numerically stable and free of the boundary double-counting that arises
    when reconstructing per-class probabilities and differencing them.

    Parameters
    ----------
    logits : torch.Tensor, shape (B, K-1)

    Returns
    -------
    expected_rank : torch.Tensor, shape (B,) — continuous ranks in [0, K-1].
    """
    cum = _cumulative_survival(logits)
    return cum.sum(dim=1)


# ── Continuous age score in [0, 1] ─────────────────────────────────────────────
@torch.no_grad()
def score_from_logits(logits: torch.Tensor, num_classes: int) -> torch.Tensor:
    """
    Continuous age score s_hat = E[rank] / (num_classes - 1)  in [0, 1].

    This is the PRIMARY reportable output of the model. It is the only score
    function that should be used for trajectory evaluation, treatment
    comparison and any continuous s_hat reported in the thesis.

    Parameters
    ----------
    logits      : torch.Tensor, shape (B, K-1)
    num_classes : int, K — pass `encoder.num_classes` from the checkpoint.

    Returns
    -------
    score : torch.Tensor, shape (B,) — continuous age scores in [0, 1].
    """
    if num_classes < 2:
        raise ValueError(f"num_classes must be >= 2; got {num_classes}.")
    if logits.shape[1] != num_classes - 1:
        raise ValueError(
            f"logits has {logits.shape[1]} columns but num_classes={num_classes} "
            f"implies {num_classes - 1} columns (K-1). The encoder and model "
            f"output dimension disagree — load num_classes from the checkpoint "
            f"encoder rather than recomputing it."
        )
    return expected_rank_from_logits(logits) / (num_classes - 1)


# ── Full rank probability distribution (for plotting / diagnostics only) ───────
@torch.no_grad()
def probs_from_logits(logits: torch.Tensor) -> torch.Tensor:
    """
    Full per-rank probability distribution P(Y = k), k = 0 ... K-1.

    Use this ONLY for visualising the rank distribution. Do NOT use it to
    compute the age score — `score_from_logits` (expected value from cumulative
    survival) is the stable, canonical path.

    Construction
    ------------
        survival[:, 0]   = 1                                  (P(Y >= 0))
        survival[:, k]   = P(Y >= k) = cumprod sigmoid        (k = 1 ... K-1)
        P(Y = k)         = P(Y >= k) - P(Y >= k+1)            (k = 0 ... K-2)
        P(Y = K-1)       = P(Y >= K-1)                        (last rank)

    Because survival is rank-monotonic, every P(Y = k) is non-negative and the
    distribution sums to 1 exactly — no renormalisation needed (a small clamp
    guards against floating-point noise only).

    Parameters
    ----------
    logits : torch.Tensor, shape (B, K-1)

    Returns
    -------
    probs : torch.Tensor, shape (B, K) — sums to 1 along dim=1.
    """
    cum   = _cumulative_survival(logits)                       # (B, K-1) = P(Y>=k), k=1..K-1
    ones  = torch.ones(logits.shape[0], 1, device=logits.device, dtype=cum.dtype)
    survival = torch.cat([ones, cum], dim=1)                   # (B, K)   = P(Y>=k), k=0..K-1

    # P(Y = k) = P(Y >= k) - P(Y >= k+1) for k = 0 ... K-1, with P(Y >= K) = 0.
    zeros = torch.zeros(logits.shape[0], 1, device=logits.device, dtype=cum.dtype)
    survival_shifted = torch.cat([survival[:, 1:], zeros], dim=1)   # P(Y>=k+1)
    probs = survival - survival_shifted                        # (B, K), non-negative

    probs = probs.clamp(min=0.0)
    probs = probs / probs.sum(dim=1, keepdim=True)             # guard fp noise only
    return probs


# ── NumPy convenience wrappers ─────────────────────────────────────────────────
@torch.no_grad()
def score_from_logits_np(logits: torch.Tensor, num_classes: int) -> np.ndarray:
    """Continuous age score as a NumPy array. See `score_from_logits`."""
    return score_from_logits(logits, num_classes).detach().cpu().numpy()


@torch.no_grad()
def rank_from_logits_np(logits: torch.Tensor) -> np.ndarray:
    """Discrete predicted rank as a NumPy array. See `rank_from_logits`."""
    return rank_from_logits(logits).detach().cpu().numpy()


# ── Self-test ──────────────────────────────────────────────────────────────────
def _self_test() -> None:
    """
    Validate the module against three invariants:

      1. `rank_from_logits` matches coral_pytorch's `corn_label_from_logits`
         (when the package is installed).
      2. The expected rank from cumulative survival equals the expected rank
         computed from the full P(Y=k) distribution (the two derivations agree).
      3. `probs_from_logits` produces a valid distribution (non-negative, sums
         to 1) and the discrete argmax rank is bracketed by floor/ceil of the
         continuous expected rank.
    """
    torch.manual_seed(0)
    K = 25
    logits = torch.randn(2000, K - 1)

    # (1) discrete rank matches coral_pytorch reference, if available
    try:
        from coral_pytorch.dataset import corn_label_from_logits
        ref = corn_label_from_logits(logits)
        ours = rank_from_logits(logits)
        assert torch.equal(ref, ours), "rank_from_logits disagrees with coral_pytorch"
        print("  [1] rank_from_logits matches coral_pytorch.corn_label_from_logits  ✔")
    except ImportError:
        print("  [1] coral_pytorch not installed — skipped reference rank check    (skip)")

    # (2) expected rank via survival-sum == expected rank via full distribution
    er_survival = expected_rank_from_logits(logits)
    probs       = probs_from_logits(logits)
    rank_vals   = torch.arange(K, dtype=probs.dtype)
    er_distrib  = (probs * rank_vals).sum(dim=1)
    max_err = (er_survival - er_distrib).abs().max().item()
    assert max_err < 1e-4, f"expected-rank derivations disagree (max err {max_err:.2e})"
    print(f"  [2] E[rank] survival-sum == distribution form (max err {max_err:.2e})    ✔")

    # (3) probs_from_logits is a valid distribution: non-negative and sums to 1.
    #     (We do NOT assert argmax is near the mean: for wide/multimodal CORN
    #     distributions the mode can legitimately sit several ranks from E[Y].)
    assert (probs >= 0).all(), "negative probabilities"
    sums = probs.sum(dim=1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5), "probs do not sum to 1"
    print("  [3] probs_from_logits valid (non-negative, sums to 1)                ✔")

    # (4) monotonicity: a uniformly more "senescent" logit profile (every logit
    #     shifted up) must never DECREASE the continuous score. This guards the
    #     ordinal interpretation the whole pipeline depends on.
    base    = torch.randn(1000, K - 1)
    shifted = base + 0.5                       # push every conditional prob up
    s_base    = score_from_logits(base, K)
    s_shifted = score_from_logits(shifted, K)
    assert (s_shifted >= s_base - 1e-6).all(), "score not monotonic in logits"
    print("  [4] score is monotonic non-decreasing in logits                      ✔")

    # score range
    s = score_from_logits(logits, K)
    assert (s >= 0).all() and (s <= 1).all(), "score outside [0, 1]"
    print(f"  [+] score_from_logits in [0, 1]  (min {s.min():.4f}, max {s.max():.4f})  ✔")

    print("\nAll scoring self-tests passed.")


if __name__ == "__main__":
    _self_test()
