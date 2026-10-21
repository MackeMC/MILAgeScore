"""
src/losses.py

Loss functions, label encoding, and evaluation metrics for MILAgeScore.

Design follows Kent (2023) Chapter 3 — Ordinal Multiple Instance Deep Learning,
which empirically established CORN as the strongest ordinal loss for tabular
MIL data with attention pooling, evaluated using MAE, MZOE, and RMSE.

Key responsibilities:
    1. OrdinalLabelEncoder — maps continuous senescence scores s ∈ [0,1] to
       integer rank indices required by the CORN loss. This is non-trivial
       because the two cell lines have different passage spacings, producing
       non-uniform s values. All unique s values across both lines are jointly
       sorted and indexed.

    2. corn_loss_fn — thin wrapper around coral_pytorch.losses.corn_loss,
       handling device placement and label encoding cleanly.

    3. Evaluation metrics — MAE, MZOE, RMSE computed at the bag level on
       predicted vs. true rank indices, consistent with Kent (2023) Table 3.2.

    4. ContinuousAgeLoss — optional auxiliary MSE loss on the continuous
       predicted score s_hat vs true s, for use as a secondary training signal
       if CORN loss alone shows slow convergence on the ordinal regression task.
"""

import numpy as np
import torch
import torch.nn as nn
from coral_pytorch.losses import corn_loss
from coral_pytorch.dataset import corn_label_from_logits
from scipy.stats import spearmanr

# ── Ordinal Label Encoder ──────────────────────────────────────────────────────
class OrdinalLabelEncoder:
    """
    Maps continuous senescence scores s ∈ [0, 1] to integer rank indices
    required by corn_loss and corn_label_from_logits.

    The two cell lines have different senescence denominators (11 and 14
    passage intervals respectively), producing non-uniform s values:
        Line 1: s ∈ {0/11, 1/11, ..., 11/11}
        Line 2: s ∈ {0/14, 1/14, ..., 14/14}

    After pooling all unique s values and sorting them, each gets a unique
    rank index 0, 1, ..., K-1 where K = number of unique groups.
    This index is what CORN trains on.

    Parameters
    ----------
    s_values : array-like
        All unique senescence scores present in the dataset.
        Typically extracted from labels_df['s'].unique().
    tol : float
        Floating point tolerance for matching s values to rank indices.
    """

    def __init__(self, s_values: np.ndarray, tol: float = 1e-4):
        unique_s         = np.unique(np.array(s_values, dtype=np.float32))
        self.unique_s    = unique_s
        self.num_classes = len(unique_s)
        self.tol         = tol

        self._s_to_rank  = {round(float(s), 6): idx
                            for idx, s in enumerate(unique_s)}

        print(f"OrdinalLabelEncoder initialised:")
        print(f"  Unique s values (ranks) : {self.num_classes}")
        print(f"  s range                 : [{unique_s.min():.4f}, {unique_s.max():.4f}]")
        print(f"  Rank 0  → s={unique_s[0]:.4f}")
        print(f"  Rank {self.num_classes-1} → s={unique_s[-1]:.4f}")

    def encode(self, s_tensor: torch.Tensor) -> torch.Tensor:
        """
        Convert a batch of continuous s values to integer rank indices.

        Parameters
        ----------
        s_tensor : torch.Tensor, shape (batch_size,), float32

        Returns
        -------
        ranks : torch.Tensor, shape (batch_size,), int64
        """
        s_np  = s_tensor.detach().cpu().numpy()
        ranks = np.array([self._encode_single(s) for s in s_np], dtype=np.int64)
        return torch.from_numpy(ranks)

    def _encode_single(self, s: float) -> int:
        key = round(float(s), 6)
        if key in self._s_to_rank:
            return self._s_to_rank[key]
        # Fallback: nearest unique_s within tolerance
        diffs = np.abs(self.unique_s - s)
        idx   = int(np.argmin(diffs))
        if diffs[idx] > self.tol:
            raise ValueError(
                f"Senescence score s={s:.6f} does not match any known rank "
                f"within tolerance={self.tol}. "
                f"Nearest known s={self.unique_s[idx]:.6f}. "
                f"Check that this sample comes from the same dataset split."
            )
        return idx

    def decode(self, ranks: torch.Tensor) -> torch.Tensor:
        """
        Convert integer rank indices back to continuous senescence scores.

        Parameters
        ----------
        ranks : torch.Tensor, shape (batch_size,), int64

        Returns
        -------
        scores : torch.Tensor, shape (batch_size,), float32
        """
        ranks_np = ranks.detach().cpu().long().numpy()
        scores   = self.unique_s[ranks_np]
        return torch.from_numpy(scores)

    def rank_to_score(self, rank: int) -> float:
        """Return the senescence score for a given rank index."""
        return float(self.unique_s[rank])

    def score_to_rank(self, s: float) -> int:
        """Return the rank index for a given senescence score."""
        return self._encode_single(s)


# ── CORN Loss Wrapper ──────────────────────────────────────────────────────────
class CORNLoss(nn.Module):
    """
    Wrapper around coral_pytorch.losses.corn_loss for use in the training loop.

    CORN (Shi et al. 2021) achieves rank consistency via conditional training
    subsets and the chain rule of probability, without architectural constraints.
    Empirically outperforms CORAL and CLM-QWK on MAE and RMSE for tabular
    ordinal MIL data (Kent 2023, Chapter 3).

    Parameters
    ----------
    encoder : OrdinalLabelEncoder
        Used to convert continuous s labels to integer ranks before loss
        computation. Must be the same encoder used throughout training.
    """

    def __init__(self, encoder: OrdinalLabelEncoder):
        super().__init__()
        self.encoder     = encoder
        self.num_classes = encoder.num_classes

    def forward(
        self,
        logits: torch.Tensor,
        s_labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        logits   : torch.Tensor, shape (batch_size, num_classes - 1)
                   Raw CORN logits from CORNHead.forward()
        s_labels : torch.Tensor, shape (batch_size,), float32
                   Continuous senescence scores from the DataLoader

        Returns
        -------
        loss : torch.Tensor, scalar
        """
        rank_labels = self.encoder.encode(s_labels).to(logits.device)
        return corn_loss(logits, rank_labels, self.num_classes)


# ── Optional Auxiliary Loss ────────────────────────────────────────────────────
class ContinuousAgeLoss(nn.Module):
    """
    Auxiliary MSE loss between predicted continuous score s_hat and true s.

    Can be combined with CORN loss as a secondary signal:
        total_loss = corn_loss + lambda_aux * mse_loss

    Useful during early training when CORN loss gradients are sparse due
    to the conditional subset structure of the CORN formulation.
    Set lambda_aux = 0.0 to disable entirely.

    The predicted s_hat is derived from CORN logits via:
        s_hat = predicted_rank / (num_classes - 1)
    which maps rank indices linearly back onto [0, 1].
    """

    def __init__(self, encoder: OrdinalLabelEncoder):
        super().__init__()
        self.encoder     = encoder
        self.num_classes = encoder.num_classes
        self.mse         = nn.MSELoss()

    def forward(
        self,
        logits: torch.Tensor,
        s_labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        logits   : (batch_size, num_classes - 1)
        s_labels : (batch_size,) float32 — true continuous s

        Returns
        -------
        loss : scalar MSE between predicted and true s
        """

        pred_ranks  = corn_label_from_logits(logits).float()
        pred_scores = pred_ranks / (self.num_classes - 1)
        return self.mse(pred_scores, s_labels.to(logits.device))


# ── Combined Loss ──────────────────────────────────────────────────────────────
class MILAgeLoss(nn.Module):
    """
    Combined training loss: CORN + optional auxiliary continuous MSE.

        L = corn_loss + lambda_aux * mse_loss

    lambda_aux = 0.0 (default) disables the auxiliary term entirely,
    falling back to pure CORN. Set to 0.1–0.5 if early training is slow
    to converge.

    Parameters
    ----------
    encoder    : OrdinalLabelEncoder
    lambda_aux : float — weight on auxiliary MSE loss
    """

    def __init__(self, encoder: OrdinalLabelEncoder, lambda_aux: float = 0.0):
        super().__init__()
        self.corn_loss  = CORNLoss(encoder)
        self.aux_loss   = ContinuousAgeLoss(encoder) if lambda_aux > 0.0 else None
        self.lambda_aux = lambda_aux

    def forward(
        self,
        logits: torch.Tensor,
        s_labels: torch.Tensor,
    ) -> tuple[torch.Tensor, dict]:
        """
        Returns
        -------
        total_loss      : scalar torch.Tensor
        loss_components : dict — 'loss/corn', 'loss/aux', 'loss/total'
                          for granular wandb logging
        """
        l_corn = self.corn_loss(logits, s_labels)

        if self.aux_loss is not None and self.lambda_aux > 0.0:
            l_aux  = self.aux_loss(logits, s_labels)
            total  = l_corn + self.lambda_aux * l_aux
        else:
            l_aux  = torch.tensor(0.0)
            total  = l_corn

        return total, {
            'loss/corn'  : l_corn.item(),
            'loss/aux'   : l_aux.item() if isinstance(l_aux, torch.Tensor) else 0.0,
            'loss/total' : total.item(),
        }


# ── Evaluation Metrics ─────────────────────────────────────────────────────────
def compute_metrics(
    pred_ranks: np.ndarray,
    true_ranks: np.ndarray,
) -> dict:
    """
    Bag-level evaluation metrics consistent with Kent (2023) Chapter 3.

    MAE  — Mean Absolute Error on rank indices (primary metric)
    MZOE — Mean Zero-One Error: fraction of bags with incorrect rank
    RMSE — Root Mean Squared Error on rank indices

    Metrics are computed on the full concatenated val set — not averaged
    over batches — to ensure MZOE and Spearman are computed correctly.

    Parameters
    ----------
    pred_ranks : np.ndarray (n_bags,) int
    true_ranks : np.ndarray (n_bags,) int

    Returns
    -------
    metrics : dict — keys 'eval/mae', 'eval/mzoe', 'eval/rmse'
    """
    pred = np.array(pred_ranks)
    true = np.array(true_ranks)
    diff = pred - true

    return {
        'eval/mae'  : float(np.mean(np.abs(diff))),
        'eval/mzoe' : float(np.mean(pred != true)),
        'eval/rmse' : float(np.sqrt(np.mean(diff ** 2))),
    }


def compute_score_metrics(
    pred_scores: np.ndarray,
    true_scores: np.ndarray,
) -> dict:
    """
    Metrics in continuous senescence score space s ∈ [0, 1].

    Spearman ρ — rank correlation of predicted vs true s.
                 Primary for thesis reporting: does not assume linearity
                 and is robust to non-uniform s spacing across the two lines.
    MAE_s      — mean absolute error in s units (biologically interpretable)

    Parameters
    ----------
    pred_scores : np.ndarray (n_bags,) float32
    true_scores : np.ndarray (n_bags,) float32
    """

    rho, pval = spearmanr(pred_scores, true_scores)

    return {
        'eval/spearman_rho' : float(rho),
        'eval/spearman_p'   : float(pval),
        'eval/mae_s'        : float(np.mean(np.abs(pred_scores - true_scores))),
    }


# ── Factory ────────────────────────────────────────────────────────────────────
def build_loss(
    labels_df,
    cfg: dict,
) -> tuple[MILAgeLoss, OrdinalLabelEncoder]:
    """
    Build the combined loss and encoder from the labels dataframe.

    The encoder is built from all unique s values in the full dataset
    (not just the training split) so that val/test s values are never
    out-of-vocabulary at encode time.

    Parameters
    ----------
    labels_df  : pd.DataFrame with column 's'
    cfg        : full config dict

    Returns
    -------
    criterion : MILAgeLoss
    encoder   : OrdinalLabelEncoder — also used in train.py and evaluate.py
    """
    s_values   = labels_df['s'].unique()
    encoder    = OrdinalLabelEncoder(s_values)
    lambda_aux = cfg['training'].get('lambda_aux', 0.0)
    criterion  = MILAgeLoss(encoder=encoder, lambda_aux=lambda_aux)

    print(f"\nLoss: CORN  +  {lambda_aux} × MSE auxiliary")
    return criterion, encoder
