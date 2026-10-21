'''Leave-one-line-out cross-validation. Train on Line 2 only, predict Line 1 passage ordering.
Then train on Line 1 only, predict Line 2.
Add this to main.py as a --mode crossval before the meeting if time permits:'''

# In main.py, add run_crossval():
def run_crossval(cfg: dict, device: torch.device, logger):
    """
    Leave-one-line-out evaluation.
    Train on one cell line entirely, evaluate ordinal ranking on the other.
    This tests whether learned morphological features transfer across lines —
    the strongest available test of generalisation with this dataset.
    """
    logger.info("Mode: CROSS-VALIDATION (leave-one-line-out)")
    # Load full processed data
    features_manual = np.load(...)
    features_latent = np.load(...)
    labels_df       = pd.read_csv(...)

    for held_out_line in ['1', '2']:
        train_line = '2' if held_out_line == '1' else '1'

        train_idx = labels_df[labels_df['cell_line'] == train_line].index.values
        val_idx   = labels_df[labels_df['cell_line'] == held_out_line].index.values

        # Re-fit scalers on train_idx only
        # Build datasets, train fresh model, evaluate trajectory on held_out_line
        logger.info(f"  Train on Line {train_line}, evaluate on Line {held_out_line}")
        ...
