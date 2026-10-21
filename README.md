# MILAgeScore Project Overview

## Purpose
The MILAgeScore project is designed to predict biological age (or "Age Score") from high-content imaging data of cells using a **Multiple Instance Learning (MIL)** framework combined with **Ordinal Regression**. 

Instead of predicting the age of individual cells (which is noisy and lacks ground truth at the single-cell level), this project groups cells from the same biological sample (well) into "bags" and predicts the sample-level age. This approach allows the model to learn which cell phenotypes are most representative of aging and senescence.

---

## Technical Architecture

The model is built as a deep neural network with three primary components:

### 1. Instance Encoder (`InstanceEncoder`)
*   **Function:** Processes individual cell features (both manual morphological features and latent embeddings from pre-trained models).
*   **Architecture:** A multi-layer perceptron (MLP) with residual-like skip connections (depending on implementation details) and dropout for regularisation.
*   **Role:** Transforms raw high-dimensional cell features into a shared latent embedding space.

### 2. Attention MIL Aggregator (`AttentionMILAggregator`)
*   **Type of Network:** **Gated Attention Mechanism**.
*   **Function:** Aggregates embeddings from all cells in a bag into a single bag-level representation.
*   **Mechanism:** It learns a weight (attention score) for each cell. Cells that display strong morphological markers of aging receive higher weights, meaning they contribute more to the final bag-level prediction.
*   **Benefit:** Provides interpretability by identifying which specific cells the model is "looking at" to determine the age score.

### 3. CORN Head (`CORNHead`)
*   **Type of Network:** **Conditional Ordinal Regression for Deep Learning (CORN)**.
*   **Function:** Maps the bag-level embedding to an ordinal rank.
*   **Logic:** Aging is a continuous but often discretised process (e.g., passage numbers or age brackets). Standard classification ignores the ordering, while regression assumes linear distances. CORN treats the problem as a series of binary classification tasks (is age > k?), ensuring the predicted probabilities are monotonic and respect the ordinal nature of aging.
*   **Output:** Produces logits that are converted into a "continuous age score" through the expected value of the predicted ranks.

---

## Data Pipeline

### Feature Sets
The project supports three types of feature configurations:
*   **Manual:** Human-engineered morphological features (size, shape, texture).
*   **Latent:** Embeddings derived from a self-supervised or pre-trained image encoder (e.g., DINO, MAE).
*   **Both:** A concatenated set of both manual and latent features, often weighted to balance their influence.

### Bag Construction (`CellBagDataset`)
*   Cells are sampled from biological wells.
*   **`cells_per_bag`**: The number of cells sampled to represent a sample.
*   **`max_well_fraction`**: Limits the percentage of cells taken from a single well to ensure bag diversity.

---

## Validation Strategies

### 1. Group-based Validation
Standard random split of samples into training and validation sets.

### 2. Leave-One-Cell-Line-Out (LOCO)
*   **Function:** Trains the model on all cell lines except one, which is held out for validation.
*   **Purpose:** Tests the model's ability to generalise to unseen biological contexts, ensuring it learns universal aging markers rather than line-specific artifacts.

---

## Interpretability Tools (`explain.py`)

*   **Attention Mapping:** Visualises which cells in a well are driving the age prediction.
*   **Integrated Gradients (IG):** Attributes the final age score prediction back to specific input features (e.g., "cell area" or "nuclear intensity"), showing *why* a particular sample was rated as older.

---

## Implementation Details
*   **Framework:** PyTorch
*   **Logging:** Weights & Biases (W&B)
*   **Loss Function:** Custom `MILAgeLoss` combining CORN loss with optional auxiliary constraints.
