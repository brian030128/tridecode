"""Ablation study summary for trie decoding modifications.

This script summarizes time and token savings for three configurations
evaluated on the PHI model with the HumanEval dataset using beam search
with width 3:

1. No trie, No garbage collection (baseline beam search)
2. Yes trie, No garbage collection
3. Yes trie, Yes garbage collection (full trie decoding)

The second configuration is recorded in ``modification_test.csv`` while the
baseline and full trie results are loaded from the ``final_out`` directory.

Timing information in ``modification_test.csv`` only contains timestamps for
the first 400 generated tokens under the "Yes trie, No GC" setting. To keep
the comparison fair, we truncate all timing arrays to at most 400 tokens when
computing total runtime across the three configurations.
"""

import csv
import json
import math
import statistics
from argparse import ArgumentParser
from pathlib import Path


def _mean_ci(data):
    if not data:
        return float('nan'), float('nan'), float('nan')
    mean = statistics.mean(data)
    if len(data) < 2:
        return mean, float('nan'), float('nan')
    sd = statistics.stdev(data)
    margin = 1.96 * sd / math.sqrt(len(data))
    return mean, mean - margin, mean + margin


def load_rows(path, limit_tokens=400):
    """Load CSV rows and compute total time up to ``limit_tokens`` tokens."""
    rows = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            times = json.loads(row['time_took'])
            if times:
                end = min(len(times) - 1, limit_tokens)
                total_time = times[end] - times[0]
            else:
                total_time = 0.0
            rows.append({
                'total_saved': int(row['total_saved']),
                'gc_saved': int(row['gc_saved']),
                'trie_saved': int(row['trie_atten_saved']),
                'time': total_time,
            })
    return rows


def load_times(jsonl_path, limit_tokens=400):
    """Return a list of total times up to ``limit_tokens`` tokens from a JSONL output file."""
    times = []
    with open(jsonl_path) as f:
        for line in f:
            data = json.loads(line)
            stamps = data.get("time_metric")
            if stamps:
                end = min(len(stamps) - 1, limit_tokens)
                total = stamps[end] - stamps[0]
            else:
                total = float(data.get("time_taken", 0.0))
            times.append(total)
    return times


def analyze(
    origin_times,
    trie_csv_rows,
    tree_times,
    out_csv,
):
    base_tokens = [0.0 for _ in origin_times]
    trie_tokens = [r["trie_saved"] for r in trie_csv_rows]
    full_tokens = [r["total_saved"] for r in trie_csv_rows]
    trie_times = [r["time"] for r in trie_csv_rows]

    configs = {
        "No trie, No GC": (base_tokens, origin_times),
        "Yes trie, No GC": (trie_tokens, trie_times),
        "Yes trie, Yes GC": (full_tokens, tree_times),
    }

    results = []
    for name, (token_arr, time_arr) in configs.items():
        mean_s, ci_low_s, ci_high_s = _mean_ci(token_arr)
        mean_t, ci_low_t, ci_high_t = _mean_ci(time_arr)
        results.append({
            "config": name,
            "saved_tokens_mean": mean_s,
            "saved_tokens_ci_low": ci_low_s,
            "saved_tokens_ci_high": ci_high_s,
            "time_mean": mean_t,
            "time_ci_low": ci_low_t,
            "time_ci_high": ci_high_t,
        })

    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)


def main():
    p = ArgumentParser(description='Ablation study for trie and GC techniques')
    p.add_argument('--csv', default='reproduction/final_out/modification_test.csv',
                   help='CSV with trie/no-GC results')
    p.add_argument(
        '--origin_jsonl',
        default='reproduction/final_out/PHI35/origin/HUMAN_EVAL/3_1000.jsonl',
        help='JSONL file for baseline results',
    )
    p.add_argument(
        '--tree_jsonl',
        default='reproduction/final_out/PHI35/tree/HUMAN_EVAL/3_1000.jsonl',
        help='JSONL file for trie+GC results',
    )
    p.add_argument('--out_csv', default='analysis/results/modification_ablation.csv')
    p.add_argument('--limit_tokens', type=int, default=400,
                   help='Number of tokens to use when computing timing metrics')
    args = p.parse_args()

    rows = load_rows(args.csv, args.limit_tokens)
    origin_times = load_times(args.origin_jsonl, args.limit_tokens)
    tree_times = load_times(args.tree_jsonl, args.limit_tokens)
    analyze(origin_times, rows, tree_times, args.out_csv)
    print(f'Results saved to {args.out_csv}')


if __name__ == '__main__':
    main()

    """
    Example usage:
    python -m analysis.modification_ablation \
        --csv=reproduction/final_out/modification_test.csv \
        --origin_jsonl=reproduction/final_out/PHI35/origin/HUMAN_EVAL/3_1000.jsonl \
        --tree_jsonl=reproduction/final_out/PHI35/tree/HUMAN_EVAL/3_1000.jsonl \
        --out_csv=analysis/results/modification_ablation.csv
    """