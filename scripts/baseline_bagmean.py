import os
import pickle
import sys
from pathlib import Path
# Add project root to sys.path so 'src' can be imported when running from scripts/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import src
import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

def load_data(processed_dir):
    features_manual = np.load(Path(processed_dir) / 'features_manual.npy')
    features_latent = np.load(Path(processed_dir) / 'features_latent.npy')
    labels_df = pd.read_csv(Path(processed_dir) / 'labels.csv')
    with open(Path(processed_dir) / 'manual_cols.pkl', 'rb') as f:
        manual_cols = pickle.load(f)
    with open(Path(processed_dir) / 'latent_cols.pkl', 'rb') as f:
        latent_cols = pickle.load(f)
    return features_manual, features_latent, labels_df, manual_cols, latent_cols

def get_bag_representation(features, bag_indices, use_std=False):
    bag_feats = features[bag_indices]
    mean_feat = np.mean(bag_feats, axis=0)
    if not use_std:
        return mean_feat
    std_feat = np.std(bag_feats, axis=0)
    return np.concatenate([mean_feat, std_feat])

def run_ridge_baseline(held_out_line, seed, use_std=False, task_id="task3"):
    processed_dir = "data/processed/"
    features_manual, features_latent, labels_df, manual_cols, latent_cols = load_data(processed_dir)
    
    # Standardise labels (binning)
    from src.binning import prepare_binned_labels
    labels_df = prepare_binned_labels(labels_df, 10)
    
    # LOCO Split
    from src.split_loco import split_leave_one_line_out
    line_column = 'cell_line_orig' if 'cell_line_orig' in labels_df.columns else 'cell_line'
    train_idx, inner_idx, heldout_idx, inner_wells_df = split_leave_one_line_out(
        labels_df, held_out_line=held_out_line, line_column=line_column,
        inner_val=True, inner_val_seed=seed
    )
    
    # Scale features
    scaler = StandardScaler()
    scaler.fit(np.concatenate([features_manual[train_idx], features_latent[train_idx]], axis=1))
    
    full_features = np.concatenate([features_manual, features_latent], axis=1)
    full_features_scaled = scaler.transform(full_features)
    
    # Sampler params (from Task 3 brief)
    cells_per_bag = 100
    bags_per_group_train = 200
    bags_per_group_eval = 30
    max_well_fraction = 0.35

    def sample_bags(indices, bags_per_group, seed):
        rng = np.random.default_rng(seed)
        df = labels_df.iloc[indices].copy()
        df['orig_idx'] = indices
        bag_data = []
        bag_labels = []
        bag_meta = []

        group_cols = ['cell_line', 'passage_orig']
        for name, group in df.groupby(group_cols):
            well_dict = {w: g['orig_idx'].values for w, g in group.groupby('well_name')}
            s_val = group['s'].iloc[0]
            
            for b in range(bags_per_group):
                # Sample cells with well cap
                well_names = list(well_dict.keys())
                n_wells = len(well_names)
                n_per_well = int(np.ceil(cells_per_bag * max_well_fraction))
                
                bag_indices = []
                # Mix wells
                shuffled_wells = rng.permutation(well_names)
                for w in shuffled_wells:
                    w_pool = well_dict[w]
                    n_sample = min(len(w_pool), n_per_well, cells_per_bag - len(bag_indices))
                    if n_sample > 0:
                        bag_indices.extend(rng.choice(w_pool, n_sample, replace=False))
                    if len(bag_indices) >= cells_per_bag:
                        break
                
                # If not enough cells, fill with replacement from full group pool
                if len(bag_indices) < cells_per_bag:
                    group_pool = group['orig_idx'].values
                    needed = cells_per_bag - len(bag_indices)
                    bag_indices.extend(rng.choice(group_pool, needed, replace=True))
                
                bag_indices = np.array(bag_indices)[:cells_per_bag]
                rep = get_bag_representation(full_features_scaled, bag_indices, use_std=use_std)
                bag_data.append(rep)
                bag_labels.append(s_val)
                bag_meta.append({
                    'cell_line': name[0],
                    'passage_raw': name[1],
                    's_true': s_val
                })
        return np.array(bag_data), np.array(bag_labels), bag_meta

    print(f"Sampling bags for LOCO HO L{held_out_line} seed {seed} (std={use_std})...")
    X_train, y_train, _ = sample_bags(train_idx, bags_per_group_train, seed)
    X_inner, y_inner, _ = sample_bags(inner_idx, bags_per_group_eval, seed + 100)
    X_heldout, y_heldout, meta_heldout = sample_bags(heldout_idx, bags_per_group_eval, seed + 200)
    
    # Tune Alpha
    alphas = [0.1, 1, 10, 100]
    best_alpha = 1.0
    best_mae = float('inf')
    for a in alphas:
        model = Ridge(alpha=a)
        model.fit(X_train, y_train)
        preds = np.clip(model.predict(X_inner), 0, 1)
        mae = np.mean(np.abs(preds - y_inner))
        if mae < best_mae:
            best_mae = mae
            best_alpha = a
    
    # Final Fit
    model = Ridge(alpha=best_alpha)
    model.fit(X_train, y_train)
    preds = np.clip(model.predict(X_heldout), 0, 1)
    
    # Save results
    run_id = f"ridge_{'meanstd' if use_std else 'mean'}_HO_L{held_out_line}_seed{seed}"
    out_dir = Path(f"outputs/review/{task_id}/{run_id}")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Predictions CSV
    pred_records = []
    for i, m in enumerate(meta_heldout):
        pred_records.append({
            'run_id': run_id,
            'seed': seed,
            'train_line': '2' if held_out_line == '1' else '1',
            'held_out_line': held_out_line,
            'cell_line': m['cell_line'],
            'passage_raw': m['passage_raw'],
            's_true': m['s_true'],
            's_hat': preds[i],
            'model': 'ridge',
            'pooling': 'meanstd' if use_std else 'mean',
            'feature_set': 'both'
        })
    pd.DataFrame(pred_records).to_csv(out_dir / "bag_predictions.csv", index=False)
    
    # Coefficients CSV
    feat_names = manual_cols + latent_cols
    if use_std:
        feat_names = [f"{fn}_mean" for fn in feat_names] + [f"{fn}_std" for fn in feat_names]
    
    coef_df = pd.DataFrame({
        'feature': feat_names,
        'coefficient': model.coef_,
        'run_id': run_id
    })
    coef_df.to_csv(out_dir / "ridge_coefficients.csv", index=False)
    
    print(f"Done {run_id}. Alpha={best_alpha}, MAE={best_mae:.4f}")
    return pred_records, coef_df

if __name__ == "__main__":
    all_preds = []
    all_coefs = []
    for ho_line in ["2", "1"]:
        for seed in [0, 1, 2]:
            for use_std in [False, True]:
                p, c = run_ridge_baseline(ho_line, seed, use_std=use_std)
                all_preds.extend(p)
                all_coefs.append(c)
    
    # Aggregated outputs
    out_root = Path("outputs/review/task3")
    out_root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(all_preds).to_csv(out_root / "bag_predictions_ridge.csv", index=False)
    pd.concat(all_coefs).to_csv(out_root / "ridge_coefficients.csv", index=False)
