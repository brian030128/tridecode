#!/usr/bin/env python3
import os, glob, json, argparse
import pandas as pd
from typing import Tuple
from scipy.stats import ttest_rel
from statsmodels.stats.weightstats import DescrStatsW
from statsmodels.stats.contingency_tables import mcnemar, Table2x2

def _is_binary(series: pd.Series) -> bool:
    """True if all non-NaN values are in {0,1,True,False}."""
    vals = series.dropna().unique()
    if len(vals) == 0:
        return False
    py_vals = {v.item() if hasattr(v, "item") else v for v in vals}
    return py_vals.issubset({0, 1, True, False})

def _paired_t_with_ci(x: pd.Series, y: pd.Series, prefix: str, res: dict) -> None:
    """Paired t-test + 95 % CI on (y-x) and record both means."""
    stat, pval       = ttest_rel(x, y, nan_policy='omit')
    res[f'{prefix}_orig_mean'] = x.mean()
    res[f'{prefix}_tree_mean'] = y.mean()
    res[f'{prefix}_t_stat']    = stat
    res[f'{prefix}_t_pval']    = pval

    diff = (y - x).dropna()
    ci_low, ci_high = DescrStatsW(diff).tconfint_mean(alpha=0.05)
    res[f'{prefix}_diff_mean'] = diff.mean()
    res[f'{prefix}_ci_lower']  = ci_low
    res[f'{prefix}_ci_upper']  = ci_high

def _describe(series: pd.Series, prefix: str, res: dict) -> None:
    """Mean, std and 95 % CI for a single sample series."""
    s = series.dropna()
    res[f'{prefix}_mean'] = s.mean()
    res[f'{prefix}_std']  = s.std(ddof=1)
    ci_low, ci_high      = DescrStatsW(s).tconfint_mean(alpha=0.05)
    res[f'{prefix}_ci_lower'] = ci_low
    res[f'{prefix}_ci_upper'] = ci_high

def _coerce_numeric(s: pd.Series) -> pd.Series:
    """
    Convert to numeric; non-convertible entries → NaN.
    This keeps honest 0/1 scores, floats, ints, etc.,
    but silently discards weird strings.
    """
    return pd.to_numeric(s, errors='coerce')

def load_jsonl(path: str) -> pd.DataFrame:
    with open(path) as f:
        return pd.DataFrame.from_records(json.loads(ln) for ln in f)

def compute_tok_per_sec(df: pd.DataFrame) -> pd.Series:
    """tokens / second over the output sequence."""
    return df.apply(
        lambda row: row['output_len'] /
        (row['time_metric'][row['output_len'] - 1] - row['time_metric'][0]), axis=1
    )

def compute_mem_per_token(df: pd.DataFrame) -> pd.Series:
    """
    (peak_mem - model_mem) / (total tokens).
    `peak_mem` is taken as max(memory_usage) for that sample.
    """
    peak = df['memory_usage'].apply(max)
    lowest = df['memory_usage'].apply(min)
    model_mem = df['model_memory']

    tokens = df['input_len'] + df['output_len']

    return (peak - model_mem) / tokens

    # return (peak - lowest) / tokens


def analyze_pair(orig_path: str, tree_path: str) -> dict:
    df_o, df_t = load_jsonl(orig_path), load_jsonl(tree_path)
    assert len(df_o) == len(df_t), f"Sample count mismatch: {orig_path} vs {tree_path}"

    # Derived columns
    for df in (df_o, df_t):
        df['tok_per_sec']   = compute_tok_per_sec(df)
        df['mem_per_token'] = compute_mem_per_token(df)

    res: dict = {}

    # mean input / output lengths (they are identical in the pair)
    res['mean_input_len']  = df_o['input_len'].mean()
    res['mean_output_len'] = df_o['output_len'].mean()

    # numeric paired metrics
    for m in ['input_kv_memory', 'tok_per_sec', 'mem_per_token']:
        _paired_t_with_ci(df_o[m], df_t[m], m, res)

    # ------------------  score (binary OR continuous)  ------------------ #
    s_o = _coerce_numeric(df_o['score'])
    s_t = _coerce_numeric(df_t['score'])

    # Always store the means that you asked for
    res['score_orig_mean'] = s_o.mean()
    res['score_tree_mean'] = s_t.mean()

    if _is_binary(s_o) and _is_binary(s_t):
        b = ((s_o == 1) & (s_t == 0)).sum()
        c = ((s_o == 0) & (s_t == 1)).sum()
        res['mcnemar_b'], res['mcnemar_c'] = b, c

        if b + c == 0:          # no discordant pairs
            res.update({'mcnemar_stat': 0.0, 'mcnemar_pval': 1.0,
                        'mcnemar_or_ci_low': float('nan'),
                        'mcnemar_or_ci_high': float('nan')})
        else:
            n00 = ((s_o == 0) & (s_t == 0)).sum()
            n11 = ((s_o == 1) & (s_t == 1)).sum()
            table = [[n00, c], [b, n11]]

            mc = mcnemar(table, exact=False, correction=True)
            or_low, or_high = Table2x2(table).oddsratio_confint()
            res.update({'mcnemar_stat': mc.statistic, 'mcnemar_pval': mc.pvalue,
                        'mcnemar_or_ci_low': or_low, 'mcnemar_or_ci_high': or_high})

            # Optional: CI on the paired diff in proportions
            diff = (s_t - s_o).dropna()
            d_low, d_high = DescrStatsW(diff).tconfint_mean()
            res.update({'mcnemar_diff_ci_low': d_low,
                        'mcnemar_diff_ci_high': d_high})
    else:
        _paired_t_with_ci(s_o.astype(float), s_t.astype(float), 'score', res)

    return res


# beam = 1
def analyze_origin_only(orig_path: str) -> dict:
    df = load_jsonl(orig_path)
    df['tok_per_sec']   = compute_tok_per_sec(df)
    df['mem_per_token'] = compute_mem_per_token(df)
    df['score'] = _coerce_numeric(df['score'])

    res: dict = {
        'mean_input_len':  df['input_len'].mean(),
        'mean_output_len': df['output_len'].mean()
    }

    for m in ['input_kv_memory', 'tok_per_sec', 'mem_per_token', 'score']:
        _describe(df[m].astype(float), f"{m}_orig", res)

    # binary/continuous score mean already handled by _describe
    return res


def main(base_dir: str, output_csv: str):
    summary = []

    for model in os.listdir(base_dir):
        obase = os.path.join(base_dir, model, 'origin')
        tbase = os.path.join(base_dir, model, 'tree')
        if not (os.path.isdir(obase) and os.path.isdir(tbase)):
            continue

        for dataset in os.listdir(obase):
            o_dir = os.path.join(obase, dataset)
            t_dir = os.path.join(tbase, dataset)
            if not os.path.isdir(o_dir):
                continue

            for o_path in glob.glob(os.path.join(o_dir, '*.jsonl')):
                fname = os.path.basename(o_path)        # e.g. "3_1000.jsonl"
                beam, samples = map(int, os.path.splitext(fname)[0].split('_', 1))
                t_path = os.path.join(t_dir, fname)

                if os.path.exists(t_path):              # paired origin-vs-tree
                    stats = analyze_pair(o_path, t_path)
                elif beam == 1:                         # origin-only baseline
                    stats = analyze_origin_only(o_path)
                else:
                    print(f"[WARN] Missing tree file {t_path}, skipping.")
                    continue

                stats.update({'model': model, 'dataset': dataset,
                              'beam': beam, 'samples': samples})
                summary.append(stats)

    pd.DataFrame(summary).to_csv(output_csv, index=False)
    print(f"Analysis complete. Results saved to {output_csv}")


if __name__ == '__main__':
    p = argparse.ArgumentParser(
        description="Compare origin vs tree decoding results and gather beam-1 baselines."
    )
    p.add_argument('--base_dir',   required=True,
                   help="Root dir with {model}/origin/{dataset}/ etc.")
    p.add_argument(
        '--output_csv',
        default='analysis/results/statistic_testing_results.csv',
        help="Destination CSV."
    )
    args = p.parse_args()
    main(args.base_dir, args.output_csv)


    """
    Example usage:
    python -m analysis.statistic_testing.analysis \
        --base_dir=./reproduction/final_out \
        --output_csv=./analysis/results/statistic_testing_results.csv
    """
