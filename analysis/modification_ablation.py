"""Ablation study summary for trie decoding modifications."""

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


def load_rows(path):
    rows = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            times = json.loads(row['time_took'])
            total_time = times[-1] - times[0] if len(times) >= 2 else 0.0
            rows.append({
                'total_saved': int(row['total_saved']),
                'gc_saved': int(row['gc_saved']),
                'trie_saved': int(row['trie_atten_saved']),
                'time': total_time,
            })
    return rows


def analyze(rows, out_csv):
    base_tokens = [0 for _ in rows]
    trie_tokens = [r['trie_saved'] for r in rows]
    full_tokens = [r['total_saved'] for r in rows]
    times = [r['time'] for r in rows]

    results = []
    for name, arr in [
        ('No trie, No GC', base_tokens),
        ('Yes trie, No GC', trie_tokens),
        ('Yes trie, Yes GC', full_tokens),
    ]:
        mean_s, ci_low_s, ci_high_s = _mean_ci(arr)
        mean_t, ci_low_t, ci_high_t = _mean_ci(times)
        results.append({
            'config': name,
            'saved_tokens_mean': mean_s,
            'saved_tokens_ci_low': ci_low_s,
            'saved_tokens_ci_high': ci_high_s,
            'time_mean': mean_t,
            'time_ci_low': ci_low_t,
            'time_ci_high': ci_high_t,
        })

    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)


def main():
    p = ArgumentParser(description='Ablation study for trie and GC techniques')
    p.add_argument('--csv', default='reproduction/final_out/modification_test.csv')
    p.add_argument('--out_csv', default='analysis/results/modification_ablation.csv')
    args = p.parse_args()

    rows = load_rows(args.csv)
    analyze(rows, args.out_csv)
    print(f'Results saved to {args.out_csv}')


if __name__ == '__main__':
    main()

    """
    Example usage:
    python -m analysis.modification_ablation \
        --csv=reproduction/final_out/modification_test.csv \
        --out_csv=analysis/results/modification_ablation.csv
    """
