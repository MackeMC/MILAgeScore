import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, kendalltau
from pathlib import Path
import os
import argparse

def bootstrap_metric(data_dict, passages, metric_fn, n_resamples=200, level='bag'):
    resampled_values = []
    for _ in range(n_resamples):
        if level == 'bag':
            resampled_s_hat = []
            resampled_target = []
            for p in passages:
                arr = data_dict[p]
                idx = np.random.randint(0, len(arr), size=len(arr))
                resampled_s_hat.append(arr[idx, 0])
                resampled_target.append(arr[idx, 1])
            val = metric_fn(np.concatenate(resampled_s_hat), np.concatenate(resampled_target))
        else: # group
            means_s_hat = []
            targets = []
            for p in passages:
                arr = data_dict[p]
                idx = np.random.randint(0, len(arr), size=len(arr))
                means_s_hat.append(arr[idx, 0].mean())
                targets.append(arr[0, 1])
            val = metric_fn(np.array(means_s_hat), np.array(targets))
        resampled_values.append(val)
    
    resampled_values = np.array(resampled_values)
    mask = ~np.isnan(resampled_values)
    if not np.any(mask):
        return np.nan, np.nan
    ci_low = np.percentile(resampled_values[mask], 2.5)
    ci_high = np.percentile(resampled_values[mask], 97.5)
    return ci_low, ci_high

def get_metric_functions_np():
    def spearman_fn(s_hat, target):
        if len(np.unique(target)) <= 1: return np.nan
        return spearmanr(s_hat, target)[0]
    
    def kendall_fn(s_hat, target):
        if len(np.unique(target)) <= 1: return np.nan
        return kendalltau(s_hat, target)[0]
    
    def mae_fn(s_hat, target):
        return np.abs(s_hat - target).mean()
    
    def bias_fn(s_hat, target):
        return (s_hat - target).mean()
    
    return {
        'spearman_rho': spearman_fn,
        'kendall_tau': kendall_fn,
        'mae': mae_fn,
        'bias': bias_fn
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="outputs/review/task1/bag_predictions.csv")
    parser.add_argument("--output", type=str, default="outputs/review/task2/rescoring_summary.csv")
    parser.add_argument("--n_boot", type=int, default=200)
    args = parser.parse_args()

    input_file = Path(args.input)
    output_path = Path(args.output)
    output_dir = output_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_file.exists():
        print(f"Input file not found: {input_file}")
        return

    df = pd.read_csv(input_file)
    df = df[df['set'] == 'heldout'].copy()

    # Define P_MAX per line
    P_MAX = {1: 15, 2: 18}
    P_MIN = 4
    N_BINS = 10

    # Helper for binned label: bin / (N_BINS - 1)
    def get_bin(p, line):
        if p < P_MIN or p > P_MAX[line]: return np.nan
        s = (p - P_MIN) / (P_MAX[line] - P_MIN)
        edges = np.linspace(0.0, 1.0, N_BINS + 1)
        b = np.clip(np.digitize(s, edges[1:-1]), 0, N_BINS - 1)
        return b / (N_BINS - 1)

    long_results = []
    metrics_fns = get_metric_functions_np()

    for run_id, run_df in df.groupby('run_id'):
        print(f"Processing run_id: {run_id}")
        seed = run_df['seed'].iloc[0]
        held_out_line = int(run_df['held_out_line'].iloc[0])
        train_line = int(run_df['train_line'].iloc[0])
        
        # Add targets to run_df
        run_df = run_df.copy()
        passages = sorted(run_df['passage_raw'].unique())
        
        target_map = {}
        for p in passages:
            t_own_bin = get_bin(p, held_out_line)
            t_own_frac = (p - P_MIN) / (P_MAX[held_out_line] - P_MIN)
            t_abs = get_bin(p, train_line)
            
            # T_remaining: aligned by passages before growth arrest
            p_rem = p - (P_MAX[held_out_line] - P_MAX[train_line])
            t_remaining = get_bin(p_rem, train_line)
            
            target_map[p] = {
                'T_own_bin': t_own_bin,
                'T_own_frac': t_own_frac,
                'T_abs': t_abs,
                'T_remaining': t_remaining
            }
        
        for t_name in ['T_own_bin', 'T_own_frac', 'T_abs', 'T_remaining']:
            run_df[t_name] = run_df['passage_raw'].map(lambda p: target_map[p][t_name])

        # Define ranges
        ranges = {
            'all': passages,
            'overlap': [p for p in passages if p <= 15],
            'extrapolation': [p for p in passages if p >= 16]
        }

        for r_name, r_ps in ranges.items():
            if not r_ps: continue
            range_df = run_df[run_df['passage_raw'].isin(r_ps)].copy()
            if range_df.empty: continue
            
            for t_name in ['T_own_bin', 'T_own_frac', 'T_abs', 'T_remaining']:
                valid_range_df = range_df.dropna(subset=[t_name])
                if valid_range_df.empty: continue
                
                # Pre-split for bootstrap
                passages_in_range = sorted(valid_range_df['passage_raw'].unique())
                data_dict = {p: valid_range_df[valid_range_df['passage_raw'] == p][['s_hat', t_name]].values 
                             for p in passages_in_range}

                for m_name, m_fn in metrics_fns.items():
                    # Bag level point estimate
                    val_bag = m_fn(valid_range_df['s_hat'].values, valid_range_df[t_name].values)
                    ci_low_bag, ci_high_bag = bootstrap_metric(data_dict, passages_in_range, m_fn, n_resamples=args.n_boot, level='bag')
                    long_results.append({
                        'run_id': run_id, 'seed': seed, 'held_out_line': held_out_line,
                        'target': t_name, 'range': r_name, 'level': 'bag',
                        'metric': m_name, 'value': val_bag, 'ci_low': ci_low_bag, 'ci_high': ci_high_bag, 'n': len(valid_range_df)
                    })

                    # Group level point estimate
                    g = valid_range_df.groupby('passage_raw').agg({'s_hat': 'mean', t_name: 'first'}).dropna()
                    val_group = m_fn(g['s_hat'].values, g[t_name].values)
                    ci_low_group, ci_high_group = bootstrap_metric(data_dict, passages_in_range, m_fn, n_resamples=args.n_boot, level='group')
                    long_results.append({
                        'run_id': run_id, 'seed': seed, 'held_out_line': held_out_line,
                        'target': t_name, 'range': r_name, 'level': 'group',
                        'metric': m_name, 'value': val_group, 'ci_low': ci_low_group, 'ci_high': ci_high_group, 'n': len(g)
                    })

        # Monotonicity summary
        group_all = run_df.groupby('passage_raw')['s_hat'].mean().sort_index()
        diffs = group_all.diff().dropna()
        decreases = diffs[diffs < 0]
        num_decreases = len(decreases)
        max_decrease = abs(decreases.min()) if not decreases.empty else 0
        
        long_results.append({
            'run_id': run_id, 'seed': seed, 'held_out_line': held_out_line,
            'target': 'monotonicity', 'range': 'all', 'level': 'group',
            'metric': 'num_decreases', 'value': num_decreases, 'ci_low': np.nan, 'ci_high': np.nan, 'n': len(diffs)
        })
        long_results.append({
            'run_id': run_id, 'seed': seed, 'held_out_line': held_out_line,
            'target': 'monotonicity', 'range': 'all', 'level': 'group',
            'metric': 'max_decrease', 'value': max_decrease, 'ci_low': np.nan, 'ci_high': np.nan, 'n': len(diffs)
        })

    long_df = pd.DataFrame(long_results)
    long_df.to_csv(output_dir / "rescoring_long.csv", index=False)

    # Summary
    summary_df = long_df.groupby(['held_out_line', 'target', 'range', 'level', 'metric'])['value'].agg(['mean', 'std']).reset_index()
    summary_df.to_csv(output_path, index=False)

    # Plots
    for held_out_line, line_df in df.groupby('held_out_line'):
        fig, ax = plt.subplots(figsize=(10, 6))
        
        passages = sorted(line_df['passage_raw'].unique())
        train_line = 2 if held_out_line == 1 else 1
        t_own_bin = [get_bin(p, held_out_line) for p in passages]
        t_abs = [get_bin(p, train_line) for p in passages]
        
        ax.step(passages, t_own_bin, where='mid', label='T_own_bin', color='black', linestyle='--')
        ax.step(passages, t_abs, where='mid', label='T_abs', color='gray', linestyle=':')
        
        for seed, seed_df in line_df.groupby('seed'):
            stats = seed_df.groupby('passage_raw')['s_hat'].agg(['mean', 'std'])
            ax.errorbar(stats.index, stats['mean'], yerr=stats['std'], label=f'Seed {seed}', alpha=0.7)
        
        ax.set_xlabel('Passage (raw)')
        ax.set_ylabel('Predicted age score (s_hat)')
        ax.set_title(f'Trajectory: Held-out Line {held_out_line}')
        ax.legend()
        plt.tight_layout()
        plt.savefig(output_dir / f"fig_trajectory_{held_out_line}.png")
        plt.close()

if __name__ == "__main__":
    main()
