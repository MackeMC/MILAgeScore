Key Design Decisions
Global validation metrics: Val metrics are computed over all concatenated validation bags before calling compute_metrics, not averaged over batches. This matters for MZOE and Spearman ρ — batch-averaging these is mathematically incorrect since MZOE is a proportion over the full set and Spearman requires the full rank ordering to be present.


ReduceLROnPlateau on val MAE: The scheduler monitors val MAE directly rather than loss, because CORN loss and MAE can diverge in behaviour — particularly in early training when logits are near-zero and the conditional subset structure of CORN produces flat loss surfaces. Halving LR after 7 stagnant epochs gives the model time to escape these plateaus before triggering early stopping at 15.


AdamW over Adam: AdamW decouples weight decay from the gradient update, which is the correct formulation for L2 regularisation in adaptive optimisers. For a tabular model with dropout, AdamW with weight_decay=1e-4 provides meaningful regularisation without interfering with the attention mechanism's learning dynamics.


set_to_none=True in zero_grad: Slightly faster than zeroing gradients as it releases memory back to the allocator between steps — meaningful at scale with 512 latent dims and 200 cells per bag.
