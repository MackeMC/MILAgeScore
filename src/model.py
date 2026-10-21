"""
src/model.py

MILAgeScore model architecture.

Two composable nn.Module classes plus one wrapper:

    InstanceEncoder
        Maps each cell's concatenated feature vector to a learned embedding.
        Applied identically and independently to every instance in a bag.

    AttentionMILAggregator
        Implements gated attention-based MIL pooling (Ilse et al. 2018).
        Learns a scalar attention weight per instance and returns a
        weighted sum of instance embeddings as the bag-level representation.
        Gated variant is used throughout as it is empirically more stable
        than the simple tanh attention for regression tasks.

    MILAgeScoreModel
        Composes InstanceEncoder → AttentionMILAggregator → CORN ordinal head.
        forward() returns (logits, attention_weights) so attention weights
        are always available for cell-level interpretability without a
        separate forward pass.

Reference:
    Ilse M, Tomczak J, Welling M. Attention-based deep multiple instance
    learning. ICML 2018. https://proceedings.mlr.press/v80/ilse18a/ilse18a.pdf
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from coral_pytorch.dataset import corn_label_from_logits


# ── Instance Encoder ───────────────────────────────────────────────────────────
class InstanceEncoder(nn.Module):
    """
    Encodes each cell's feature vector into a fixed-size embedding.

    Applied independently and identically to every instance in a bag
    (weight sharing across instances). This is the standard formulation
    in deep MIL — the encoder learns a task-relevant representation of
    individual cell morphology before aggregation.

    Architecture: Linear → LayerNorm → GELU → Dropout → Linear → LayerNorm

    LayerNorm is preferred over BatchNorm here because bag sizes are fixed
    but the effective batch dimension during training is
    (batch_size × cells_per_bag), and statistics should be computed
    per-instance rather than across instances from different bags.

    Parameters
    ----------
    in_features   : int   — dimensionality of concatenated input features
    hidden_dim    : int   — width of the hidden layer
    embedding_dim : int   — output embedding dimensionality
    dropout       : float — dropout rate applied after first activation
    """

    def __init__(
        self,
        in_features: int,
        hidden_dim: int = 256,
        embedding_dim: int = 128,
        dropout: float = 0.3,
    ):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (batch_size, cells_per_bag, in_features)

        Returns
        -------
        h : (batch_size, cells_per_bag, embedding_dim)
        """
        B, N, D = x.shape
        # Flatten to (B*N, D), encode, reshape back
        h = self.encoder(x.view(B * N, D))
        return h.view(B, N, -1)


# ── Gated Attention MIL Aggregator ────────────────────────────────────────────
class AttentionMILAggregator(nn.Module):
    """
    Gated attention-based MIL pooling (Ilse et al. 2018, Eq. 9).

    Computes a scalar attention weight per instance:

        a_i = softmax( w^T (tanh(V h_i) ⊙ sigmoid(U h_i)) )

    The gating mechanism (sigmoid branch) allows the network to suppress
    uninformative instances multiplicatively, which is more expressive than
    simple tanh attention and better suited to regression tasks where only
    a subset of cells may carry aging-relevant signal.

    The bag-level representation is the attention-weighted sum of embeddings:

        z = Σ_i  a_i * h_i

    Parameters
    ----------
    embedding_dim    : int   — dimensionality of instance embeddings (from encoder)
    attention_dim    : int   — internal attention projection dimensionality
    dropout          : float — dropout on attention weights before softmax
    """

    def __init__(
        self,
        embedding_dim: int = 128,
        attention_dim: int = 64,
        dropout: float = 0.25,
    ):
        super().__init__()

        # Tanh branch
        self.V = nn.Linear(embedding_dim, attention_dim, bias=False)
        # Sigmoid (gating) branch
        self.U = nn.Linear(embedding_dim, attention_dim, bias=False)
        # Attention scoring vector
        self.w = nn.Linear(attention_dim, 1, bias=False)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self, h: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        h : (batch_size, cells_per_bag, embedding_dim)

        Returns
        -------
        z       : (batch_size, embedding_dim)  — bag-level representation
        weights : (batch_size, cells_per_bag)  — attention weights (sum to 1 per bag)
        """
        # Gated attention scores: (B, N, attention_dim)
        gate = torch.tanh(self.V(h)) * torch.sigmoid(self.U(h))

        # Scalar score per instance: (B, N, 1)
        scores = self.w(self.dropout(gate))

        # Normalise across instances: (B, N, 1)
        weights = torch.softmax(scores, dim=1)

        # Weighted sum to bag embedding: (B, embedding_dim)
        z = (weights * h).sum(dim=1)

        return z, weights.squeeze(-1)   # weights: (B, N)


# ── CORN Ordinal Head ──────────────────────────────────────────────────────────
class CORNHead(nn.Module):
    """
    Rank-consistent ordinal regression output head using the CORN formulation
    (Shi et al. 2021, doi:10.1016/j.patrec.2021.01.010).

    CORN achieves rank consistency via conditional training subsets and the
    chain rule of probability, without restricting network architecture.
    The output layer has (num_classes - 1) units, one per rank boundary.

    The predicted rank label is obtained via corn_label_from_logits() which
    returns the number of rank boundaries exceeded — equivalent to the
    predicted ordinal class index.

    To convert the predicted rank index to a continuous senescence score s:
        s_hat = predicted_rank / (num_classes - 1)

    Parameters
    ----------
    in_features  : int — bag embedding dimensionality (from aggregator output)
    hidden_dim   : int — width of intermediate FC layer
    num_classes  : int — number of ordinal classes (unique passage groups)
    dropout      : float
    """

    def __init__(
        self,
        in_features: int,
        hidden_dim: int = 64,
        num_classes: int = 27,
        dropout: float = 0.3,
    ):
        super().__init__()

        self.num_classes = num_classes

        self.head = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes - 1),  # CORN: output dim = K-1
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        z : (batch_size, in_features) — bag-level embedding

        Returns
        -------
        logits : (batch_size, num_classes - 1) — raw CORN logits
        """
        return self.head(z)

    def predict_rank(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Convert CORN logits to predicted rank labels (integer class indices).

        Returns
        -------
        ranks : (batch_size,) int64
        """
        return corn_label_from_logits(logits)

    def predict_score(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Convert CORN logits to a continuous senescence score s_hat ∈ [0, 1].
        Maps predicted rank index linearly onto [0, 1].

        Returns
        -------
        scores : (batch_size,) float32
        """
        ranks = self.predict_rank(logits).float()
        return ranks / (self.num_classes - 1)


# ── Full MILAgeScore Model ─────────────────────────────────────────────────────
class MILAgeScoreModel(nn.Module):
    """
    Full MIL ordinal age scoring model.

    Pipeline:
        bag (B, N, D_in)
            → InstanceEncoder     → (B, N, embedding_dim)
            → AttentionMILAggregator → (B, embedding_dim), (B, N) attention weights
            → CORNHead            → (B, num_classes - 1) logits

    The model returns both logits and attention weights from forward().
    Attention weights are available at inference time without any additional
    computation, enabling direct cell-level interpretability.

    Parameters (all surfaced in config.yaml under model:)
    ----------
    in_features      : int   — D_in, total feature dimensionality per cell
    encoder_hidden   : int   — InstanceEncoder hidden layer width
    embedding_dim    : int   — shared embedding dimensionality
    attention_dim    : int   — gated attention internal projection dim
    head_hidden      : int   — CORNHead hidden layer width
    num_classes      : int   — number of ordinal classes
    dropout_encoder  : float
    dropout_attention: float
    dropout_head     : float
    """

    def __init__(
        self,
        in_features: int,
        encoder_hidden: int = 256,
        embedding_dim: int = 128,
        attention_dim: int = 64,
        head_hidden: int = 64,
        num_classes: int = 27,
        dropout_encoder: float = 0.3,
        dropout_attention: float = 0.25,
        dropout_head: float = 0.3,
    ):
        super().__init__()

        self.encoder     = InstanceEncoder(
            in_features=in_features,
            hidden_dim=encoder_hidden,
            embedding_dim=embedding_dim,
            dropout=dropout_encoder,
        )

        self.aggregator  = AttentionMILAggregator(
            embedding_dim=embedding_dim,
            attention_dim=attention_dim,
            dropout=dropout_attention,
        )

        self.corn_head   = CORNHead(
            in_features=embedding_dim,
            hidden_dim=head_hidden,
            num_classes=num_classes,
            dropout=dropout_head,
        )

    def forward(
        self, bag: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        bag : (batch_size, cells_per_bag, in_features)

        Returns
        -------
        logits          : (batch_size, num_classes - 1)
        attention_weights: (batch_size, cells_per_bag)
        """
        h       = self.encoder(bag)
        z, attn = self.aggregator(h)
        logits  = self.corn_head(z)
        return logits, attn

    def predict_score(self, bag: torch.Tensor) -> torch.Tensor:
        """
        Convenience method for inference. Returns continuous senescence
        score s_hat ∈ [0, 1] per bag without needing to handle logits externally.

        Parameters
        ----------
        bag : (batch_size, cells_per_bag, in_features) or (cells_per_bag, in_features)
              — single bag input is automatically unsqueezed

        Returns
        -------
        scores : (batch_size,) float32
        """
        if bag.dim() == 2:
            bag = bag.unsqueeze(0)
        self.eval()
        with torch.no_grad():
            logits, attn = self.forward(bag)
        return self.corn_head.predict_score(logits), attn


# ── Model factory ──────────────────────────────────────────────────────────────
def build_model(cfg: dict, in_features: int, num_classes: int) -> MILAgeScoreModel:
    """
    Instantiate MILAgeScoreModel from config.yaml model section.

    Parameters
    ----------
    cfg         : full config dict (model section accessed internally)
    in_features : total feature dim per cell — computed at runtime from data
    num_classes : number of ordinal classes — computed at runtime from dataset

    Returns
    -------
    model : MILAgeScoreModel on CPU (moved to device in train.py)
    """
    model_cfg = cfg['model']

    model = MILAgeScoreModel(
        in_features=in_features,
        encoder_hidden=model_cfg.get('encoder_hidden', 256),
        embedding_dim=model_cfg.get('embedding_dim', 128),
        attention_dim=model_cfg.get('attention_dim', 64),
        head_hidden=model_cfg.get('head_hidden', 64),
        num_classes=num_classes,
        dropout_encoder=model_cfg.get('dropout_encoder', 0.3),
        dropout_attention=model_cfg.get('dropout_attention', 0.25),
        dropout_head=model_cfg.get('dropout_head', 0.3),
    )

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nMILAgeScoreModel built:")
    print(f"  in_features   : {in_features}")
    print(f"  num_classes   : {num_classes}")
    print(f"  Parameters    : {n_params:,}")
    print(f"  Architecture  :")
    print(f"    InstanceEncoder   : {in_features} → {model_cfg.get('encoder_hidden', 256)}"
          f" → {model_cfg.get('embedding_dim', 128)}")
    print(f"    AttentionAggregator: {model_cfg.get('embedding_dim', 128)}"
          f" → attention_dim={model_cfg.get('attention_dim', 64)} → "
          f"{model_cfg.get('embedding_dim', 128)}")
    print(f"    CORNHead          : {model_cfg.get('embedding_dim', 128)}"
          f" → {model_cfg.get('head_hidden', 64)} → {num_classes - 1}")

    return model
