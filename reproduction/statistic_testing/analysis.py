#!/usr/bin/env python3
import os
import glob
import json
import argparse
import pandas as pd
from scipy.stats import ttest_rel
from statsmodels.stats.weightstats import DescrStatsW
from statsmodels.stats.contingency_tables import mcnemar
from statsmodels.stats.contingency_tables import Table2x2

def load_jsonl(path):
    with open(path, 'r') as f:
        records = [json.loads(line) for line in f]
    return pd.DataFrame.from_records(records)

def compute_tok_per_sec(df):
    def calc(row):
        times = row['time_metric']
        out_len = row['output_len']

        # tok/s
        return out_len / (times[out_len-1] - times[0])
    return df.apply(calc, axis=1)


def analyze_pair(orig_path, tree_path):
    """Perform paired t-tests on time_taken and input_kv_memory, and McNemar's test on score."""
    df_o = load_jsonl(orig_path)
    df_t = load_jsonl(tree_path)
    assert len(df_o) == len(df_t), f"Sample count mismatch: {orig_path} vs {tree_path}"

    # Recompute time_taken using time_metrics and output length
    df_o['tok_per_sec'] = compute_tok_per_sec(df_o)
    df_t['tok_per_sec'] = compute_tok_per_sec(df_t)

    results = {}
    # 1) Paired t-test on continuous metrics
    for metric in ['input_kv_memory', 'tok_per_sec']:
        x = df_o[metric]
        y = df_t[metric]
        stat, pval = ttest_rel(x, y, nan_policy='omit')
        results[f'{metric}_orig_mean'] = x.mean()
        results[f'{metric}_tree_mean'] = y.mean()
        results[f'{metric}_t_stat'] = stat
        results[f'{metric}_t_pval'] = pval

    # 2) 95% CI on the *difference* (tree - orig)
    diff = (y - x).dropna()
    ds = DescrStatsW(diff)
    ci_low, ci_high = ds.tconfint_mean(alpha=0.05)
    results[f'{metric}_diff_mean']    = diff.mean()
    results[f'{metric}_ci_lower']     = ci_low
    results[f'{metric}_ci_upper']     = ci_high
    
    # 3) McNemar's test on binary score
    orig_scores = df_o['score'].astype(int)
    tree_scores = df_t['score'].astype(int)
    b = ((orig_scores == 1) & (tree_scores == 0)).sum()
    c = ((orig_scores == 0) & (tree_scores == 1)).sum()
    
    table = [[0, b], [c, 0]]
    tbl   = Table2x2(table)
    
    m_result = mcnemar(table, exact=False)

    # 95% CI for the odds‐ratio (tree vs orig)
    or_low, or_high = tbl.oddsratio_confint()

    # 95% CI for the risk difference P(tree=1)−P(orig=1)
    rd_low, rd_high = tbl.riskratio_confint()

    results.update({
        'mcnemar_b': b,
        'mcnemar_c': c,
        'mcnemar_stat': m_result.statistic,
        'mcnemar_pval': m_result.pvalue,
        'mcnemar_or_ci_low':  or_low,
        'mcnemar_or_ci_high': or_high,
        'mcnemar_rd_ci_low':  rd_low,
        'mcnemar_rd_ci_high': rd_high
    })
    return results

def main(base_dir, output_csv):
    summary = []
    # Iterate over each model directory
    for model_name in os.listdir(base_dir):
        orig_base = os.path.join(base_dir, model_name, 'origin')
        tree_base = os.path.join(base_dir, model_name, 'tree')
        if not os.path.isdir(orig_base) or not os.path.isdir(tree_base):
            continue
        
        # For each dataset under orig/
        for dataset_name in os.listdir(orig_base):
            orig_dataset_dir = os.path.join(orig_base, dataset_name)
            tree_dataset_dir = os.path.join(tree_base, dataset_name)
            if not os.path.isdir(tree_dataset_dir):
                print(f"[WARN] Missing tree/ directory for {model_name}/{dataset_name}, skipping.")
                continue
            
            # For each beam_samples.jsonl in orig_dataset_dir
            for orig_path in glob.glob(os.path.join(orig_dataset_dir, '*.jsonl')):
                filename = os.path.basename(orig_path)                # e.g. "3_1000.jsonl"
                beam_str, samples_str = os.path.splitext(filename)[0].split('_', 1)
                tree_path = os.path.join(tree_dataset_dir, filename)
                if not os.path.exists(tree_path):
                    print(f"[WARN] Missing tree file {tree_path}, skipping.")
                    continue
                
                stats = analyze_pair(orig_path, tree_path)
                stats.update({
                    'model': model_name,
                    'dataset': dataset_name,
                    'beam': int(beam_str),
                    'samples': int(samples_str)
                })
                summary.append(stats)
    
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(output_csv, index=False)
    print(f"Analysis complete. Results saved to {output_csv}")
    print(summary_df)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare orig vs tree beam-search decoding results.")
    parser.add_argument("--base_dir", required=True,
                        help="Root directory (e.g. 'final_out/') containing {model_name}/orig/{dataset}/ and {model_name}/tree/{dataset}/ subfolders.")
    parser.add_argument("--output_csv", default="beam_search_comparison.csv",
                        help="Output CSV file for summary results.")
    args = parser.parse_args()
    main(args.base_dir, args.output_csv)

    """
    Example usage:
    python -m reproduction.statistic_testing.analysis --base_dir reproduction/final_out --output_csv reproduction/statistic_testing_results.csv
    """
