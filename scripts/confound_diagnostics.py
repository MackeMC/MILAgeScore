import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from pathlib import Path
import pickle
import torch
import yaml
from tqdm import tqdm
import re

def categorize_feature(name):
    shape_features = ['Area', 'Perimeter', 'Eccentricity', 'Circularity', 'MinAxisLength', 'MaxAxisLength', 'AspectRatio', 'Extent']
    ratio_features = ['ratio3', 'ratio4', 'ratio5', 'ratio6', 'ratio8']
    if name in shape_features:
        return 'shape'
    if name in ratio_features:
        return 'ratio'
    if name.startswith('latent_'):
        return 'latent'
    if any(x in name for x in ['mean', 'rawIntDen']):
        return 'intensity'
    if any(x in name for x in ['var', 'skew', 'kurt']):
        return 'texture-distribution'
    return 'other'

def fit_trend_and_get_residuals(x, y, degree=2):
    poly = PolynomialFeatures(degree=degree)
    x_poly = poly.fit_transform(x.reshape(-1, 1))
    model = LinearRegression().fit(x_poly, y)
    y_pred = model.predict(x_poly)
    residuals = y - y_pred
    return residuals, y_pred, model

def permutation_test(res1, res2, n_perms=5000):
    obs_r, _ = pearsonr(res1, res2)
    count = 0
    res2_shuffled = res2.copy()
    for _ in range(n_perms):
        np.random.shuffle(res2_shuffled)
        r, _ = pearsonr(res1, res2_shuffled)
        if abs(r) >= abs(obs_r):
            count += 1
    return obs_r, count / n_perms

def main():
    task_id = "task4"
    out_dir = Path(f"outputs/review/{task_id}")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    labels = pd.read_csv("data/processed/labels.csv")
    features_manual = pd.read_csv("data/processed/features_manual.csv")
    features_latent = pd.read_csv("data/processed/features_latent.csv")

    # Combine features
    features = pd.concat([features_manual, features_latent], axis=1)
    
    # Standardize over all cells
    scaler = StandardScaler()
    features_scaled = pd.DataFrame(scaler.fit_transform(features), columns=features.columns)
    
    # Merge with labels for grouping
    data = pd.concat([labels[['cell_line', 'passage', 'well_name']], features_scaled], axis=1)
    
    # Part A: Shared-session component, per feature
    print("Part A: Shared-session component...")
    group_medians = data.groupby(['cell_line', 'passage']).median(numeric_only=True).reset_index()
    
    results_a = []
    passages_p4_p15 = list(range(4, 16))
    
    for feat in features.columns:
        block = categorize_feature(feat)
        
        feat_res = {'feature': feat, 'block': block}
        
        for deg_name, deg in [('resid', 2), ('lin', 1), ('cub', 3)]:
            # Line 1
            l1_data = group_medians[(group_medians['cell_line'] == 1) & (group_medians['passage'].isin(passages_p4_p15))].sort_values('passage')
            x1 = l1_data['passage'].values
            y1 = l1_data[feat].values
            
            # Line 2
            l2_data = group_medians[(group_medians['cell_line'] == 2) & (group_medians['passage'].isin(passages_p4_p15))].sort_values('passage')
            x2 = l2_data['passage'].values
            y2 = l2_data[feat].values
            
            if len(x1) < 12 or len(x2) < 12:
                continue
                
            res1, _, _ = fit_trend_and_get_residuals(x1, y1, degree=deg)
            res2, _, _ = fit_trend_and_get_residuals(x2, y2, degree=deg)
            
            r, p_perm = permutation_test(res1, res2, n_perms=5000)
            rho, _ = spearmanr(res1, res2)
            
            if deg == 2:
                feat_res['resid_r'] = r
                feat_res['resid_rho'] = rho
                feat_res['perm_p'] = p_perm
                
                # Additional metrics for quadratic
                rho_l1, _ = spearmanr(x1, y1)
                rho_l2, _ = spearmanr(x2, y2)
                feat_res['trend_rho_L1'] = rho_l1
                feat_res['trend_rho_L2'] = rho_l2
                
                feat_res['resid_var_frac_L1'] = np.var(res1) / np.var(y1) if np.var(y1) > 0 else 0
                feat_res['resid_var_frac_L2'] = np.var(res2) / np.var(y2) if np.var(y2) > 0 else 0
            else:
                feat_res[f'{deg_name}_r'] = r
                feat_res[f'{deg_name}_rho'] = rho
                feat_res[f'{deg_name}_p'] = p_perm
        
        results_a.append(feat_res)

    df_a = pd.DataFrame(results_a)
    df_a.to_csv(out_dir / "session_component.csv", index=False)
    
    # Figure A
    plt.figure(figsize=(12, 8))
    df_a_sorted = df_a.sort_values('resid_r')
    sns.scatterplot(data=df_a_sorted, x=range(len(df_a_sorted)), y='resid_r', hue='block')
    plt.title('Features sorted by line-residual correlation (quadratic trend)')
    plt.xlabel('Feature Rank')
    plt.ylabel('Pearson r of residuals')
    plt.savefig(out_dir / "fig_session_component.png")
    plt.close()

    # Part B: Between-line contrast within each session
    print("Part B: Between-line contrast...")
    # 1. For each passage P4-P15: vector of (line-1 median - line-2 median) over all features.
    
    contrast_results = []
    
    # Ageing axis: pooled P4-P5 to pooled P13-P15
    p45_data = group_medians[group_medians['passage'].isin([4, 5])]
    p45_median = p45_data[features.columns].mean().values # mean of the medians
    
    p1315_data = group_medians[group_medians['passage'].isin([13, 14, 15])]
    p1315_median = p1315_data[features.columns].mean().values
    
    ageing_axis = p1315_median - p45_median
    ageing_axis_unit = ageing_axis / np.linalg.norm(ageing_axis)
    
    def get_contrast_stats(df_well_medians):
        res = []
        for p in passages_p4_p15:
            l1_p = df_well_medians[(df_well_medians['cell_line'] == 1) & (df_well_medians['passage'] == p)]
            l2_p = df_well_medians[(df_well_medians['cell_line'] == 2) & (df_well_medians['passage'] == p)]
            
            if l1_p.empty or l2_p.empty:
                continue
            
            # Medians per line-passage
            m1 = l1_p[features.columns].median().values
            m2 = l2_p[features.columns].median().values
            
            diff = m1 - m2
            norm = np.linalg.norm(diff)
            proj = np.dot(diff, ageing_axis_unit)
            res.append({'passage': p, 'norm': norm, 'proj': proj})
        return pd.DataFrame(res)

    # Need well-level medians for bootstrap
    well_medians = data.groupby(['cell_line', 'passage', 'well_name'])[features.columns].median().reset_index()
    
    obs_contrast = get_contrast_stats(well_medians)
    
    # Bootstrap
    boot_norms = []
    boot_projs = []
    
    for _ in tqdm(range(1000), desc="Bootstrapping Part B"):
        # Resample wells within each group
        boot_well_medians = []
        for (cl, p), group in well_medians.groupby(['cell_line', 'passage']):
            resampled = group.sample(n=len(group), replace=True)
            boot_well_medians.append(resampled)
        
        boot_df = pd.concat(boot_well_medians)
        stats = get_contrast_stats(boot_df)
        boot_norms.append(stats.set_index('passage')['norm'])
        boot_projs.append(stats.set_index('passage')['proj'])
        
    df_boot_norms = pd.concat(boot_norms, axis=1)
    df_boot_projs = pd.concat(boot_projs, axis=1)
    
    obs_contrast['norm_low'] = obs_contrast['passage'].map(df_boot_norms.quantile(0.025, axis=1))
    obs_contrast['norm_high'] = obs_contrast['passage'].map(df_boot_norms.quantile(0.975, axis=1))
    obs_contrast['proj_low'] = obs_contrast['passage'].map(df_boot_projs.quantile(0.025, axis=1))
    obs_contrast['proj_high'] = obs_contrast['passage'].map(df_boot_projs.quantile(0.975, axis=1))
    
    obs_contrast.to_csv(out_dir / "line_contrast.csv", index=False)
    
    # Figure B
    fig, ax = plt.subplots(1, 2, figsize=(15, 6))
    ax[0].errorbar(obs_contrast['passage'], obs_contrast['norm'], yerr=[obs_contrast['norm'] - obs_contrast['norm_low'], obs_contrast['norm_high'] - obs_contrast['norm']], fmt='-o')
    ax[0].set_title('Euclidean Norm of Line Contrast (L1-L2)')
    ax[0].set_xlabel('Passage')
    ax[0].set_ylabel('Norm')
    
    ax[1].errorbar(obs_contrast['passage'], obs_contrast['proj'], yerr=[obs_contrast['proj'] - obs_contrast['proj_low'], obs_contrast['proj_high'] - obs_contrast['proj']], fmt='-o')
    ax[1].axhline(0, color='grey', linestyle='--')
    ax[1].set_title('Projection of Line Contrast onto Ageing Axis')
    ax[1].set_xlabel('Passage')
    ax[1].set_ylabel('Projection')
    
    plt.savefig(out_dir / "fig_line_contrast.png")
    plt.close()

    # Part C: Density
    print("Part C: Density analysis...")
    
    # 1. Per well: n_cells, and median num_neighbours if it exists
    well_stats = labels.groupby(['cell_line', 'passage', 'well_name']).size().reset_index(name='n_cells')
    if 'num_neighbours' in features.columns:
        well_nn = data.groupby('well_name')['num_neighbours'].median().reset_index()
        well_stats = well_stats.merge(well_nn, on='well_name')
    else:
        well_stats['median_num_neighbours'] = np.nan

    # 2. Per well age score
    print("Scoring wells...")
    well_scores = []
    
    in_features = features.shape[1]
    num_classes = 10
    
    # Define Task 1 seed 0 checkpoints
    checkpoints = {
        1: "outputs/review/task1/loco_HO_L1_seed0/checkpoints/loco_HO_L1_seed0_best.pt",
        2: "outputs/review/task1/loco_HO_L2_seed0/checkpoints/loco_HO_L2_seed0_best.pt"
    }
    
    from src.model import build_model
    
    for held_out_line, ckpt_path in checkpoints.items():
        if not os.path.exists(ckpt_path):
            print(f"Warning: Checkpoint not found at {ckpt_path}")
            continue
            
        print(f"Scoring Line {held_out_line} using {ckpt_path}")
        cfg = {
            'model': {
                'encoder_hidden': 64, 'embedding_dim': 128, 'attention_dim': 64,
                'head_hidden': 64, 'dropout_encoder': 0.3, 'dropout_attention': 0.25,
                'dropout_head': 0.3, 'pooling': 'attention'
            }
        }
        model = build_model(cfg, in_features, num_classes)
        checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        state_dict = checkpoint['model_state'] if 'model_state' in checkpoint else checkpoint
        model.load_state_dict(state_dict)
        model.eval()
        
        line_wells = labels[labels['cell_line'] == held_out_line]['well_name'].unique()
        
        for well in tqdm(line_wells, desc=f"Well scores Line {held_out_line}"):
            well_data = data[data['well_name'] == well][features.columns].values
            n_cells = len(well_data)
            flagged = n_cells < 100
            
            well_info = labels[labels['well_name'] == well].iloc[0]
            
            s_hats = []
            with torch.no_grad():
                for _ in range(30):
                    # Draw 100 cells with replacement
                    indices = np.random.choice(n_cells, 100, replace=True)
                    bag = torch.from_numpy(well_data[indices]).float()
                    s_hat, _ = model.predict_score(bag)
                    s_hats.append(s_hat.item())
            
            # Extract row/col for Part D
            # well_name format "... - R02-C06.tif"
            match = re.search(r'R(\d+)-C(\d+)', well)
            row = int(match.group(1)) if match else np.nan
            col = int(match.group(2)) if match else np.nan
            
            well_scores.append({
                'run_id': f'loco_HO_L{held_out_line}_s0',
                'cell_line': held_out_line,
                'passage_raw': well_info['passage'],
                'well_name': well,
                'row': row,
                'col': col,
                'n_cells': n_cells,
                'median_num_neighbours': well_stats[well_stats['well_name'] == well]['median_num_neighbours'].values[0],
                's_hat_mean': np.mean(s_hats),
                's_hat_sd': np.std(s_hats),
                'flagged': flagged,
                's_orig': well_info['s']
            })
            
    df_well_scores = pd.DataFrame(well_scores)
    
    # Check if any wells were missed
    print(f"Scored {len(df_well_scores)} wells.")
    
    df_well_scores.to_csv(out_dir / "well_scores.csv", index=False)
    
    # 3. Pooled within-group correlation
    print("Calculating within-group correlations...")
    # Centre n_cells and well score on the group mean
    df_well_scores['n_cells_centred'] = df_well_scores.groupby(['cell_line', 'passage_raw'])['n_cells'].transform(lambda x: x - x.mean())
    df_well_scores['score_centred'] = df_well_scores.groupby(['cell_line', 'passage_raw'])['s_hat_mean'].transform(lambda x: x - x.mean())
    
    def get_corrs(df):
        r, _ = pearsonr(df['n_cells_centred'], df['score_centred'])
        rho, _ = spearmanr(df['n_cells_centred'], df['score_centred'])
        return r, rho

    obs_r, obs_rho = get_corrs(df_well_scores)
    
    # Bootstrap CI over groups
    groups = list(df_well_scores.groupby(['cell_line', 'passage_raw']))
    boot_rs, boot_rhos = [], []
    for _ in tqdm(range(1000), desc="Bootstrapping within-group corrs"):
        resampled_indices = np.random.choice(len(groups), len(groups), replace=True)
        boot_df = pd.concat([groups[i][1] for i in resampled_indices])
        r, rho = get_corrs(boot_df)
        boot_rs.append(r)
        boot_rhos.append(rho)
        
    # 4. Between groups: correlation of group total n_cells with group mean score residual
    # s_hat - T_own_bin. T_own_bin = bin rank / (K-1)
    # Actually labels['s'] is already the bin value in the processed labels? 
    # Let's check binning.py: df['s'] = np.where(ok, bins / (n_bins - 1), np.nan)
    # So s_orig in our df_well_scores should be T_own_bin.
    
    df_well_scores['score_resid'] = df_well_scores['s_hat_mean'] - df_well_scores['s_orig']
    group_stats = df_well_scores.groupby(['cell_line', 'passage_raw']).agg({
        'n_cells': 'sum',
        'score_resid': 'mean'
    }).reset_index()
    
    between_corrs = []
    for cl in [1, 2]:
        cl_data = group_stats[group_stats['cell_line'] == cl]
        r, _ = pearsonr(cl_data['n_cells'], cl_data['score_resid'])
        rho, _ = spearmanr(cl_data['n_cells'], cl_data['score_resid'])
        between_corrs.append({'cell_line': cl, 'between_r': r, 'between_rho': rho})
        
    df_between = pd.DataFrame(between_corrs)
    
    summary_c = {
        'within_r': obs_r,
        'within_r_low': np.quantile(boot_rs, 0.025),
        'within_r_high': np.quantile(boot_rs, 0.975),
        'within_rho': obs_rho,
        'within_rho_low': np.quantile(boot_rhos, 0.025),
        'within_rho_high': np.quantile(boot_rhos, 0.975),
    }
    pd.DataFrame([summary_c]).to_csv(out_dir / "density_summary.csv", index=False)
    df_between.to_csv(out_dir / "density_between_groups.csv", index=False) # Additional output
    
    # Figure C
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df_well_scores, x='n_cells_centred', y='score_centred', hue='cell_line', style='cell_line')
    plt.title('Within-group centred density vs centred score')
    plt.savefig(out_dir / "fig_density.png")
    plt.close()

    # Part D: Plate position
    print("Part D: Plate position...")
    # Plate layout
    plate_layout = df_well_scores.groupby(['cell_line', 'passage_raw', 'col']).size().unstack(fill_value=0)
    plate_layout.to_csv(out_dir / "plate_layout.csv")
    
    # Test whether well score (centred within group) varies with row
    row_effects = []
    for cl in [1, 2]:
        cl_data = df_well_scores[df_well_scores['cell_line'] == cl]
        # Pearson/Spearman correlation between row and score_centred
        r, p = pearsonr(cl_data['row'], cl_data['score_centred'])
        rho, pr = spearmanr(cl_data['row'], cl_data['score_centred'])
        row_effects.append({'cell_line': cl, 'row_r': r, 'row_p': p, 'row_rho': rho, 'row_rho_p': pr})
        
    pd.DataFrame(row_effects).to_csv(out_dir / "row_effect.csv", index=False)

if __name__ == "__main__":
    main()
