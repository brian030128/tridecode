import csv
import json
import ast
from pathlib import Path
import matplotlib.pyplot as plt


def load_no_gc(path: Path, limit: int) -> list:
    times = []
    with path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            ts = ast.literal_eval(row['time_took'])
            times.extend(ts)
            if len(times) >= limit:
                return times[:limit]
    return times


def load_with_gc(path: Path, limit: int) -> list:
    times = []
    with path.open() as f:
        for line in f:
            data = json.loads(line)
            ts = data.get('time_metric')
            if ts:
                times.extend(ts)
                if len(times) >= limit:
                    return times[:limit]
    return times


def main():
    no_gc_path = Path('reproduction/final_out/modification_test.csv')
    with_gc_path = Path('reproduction/final_out/PHI35/tree/HUMAN_EVAL/3_1000.jsonl')
    token_limit = 400

    no_gc_times = load_no_gc(no_gc_path, token_limit)
    with_gc_times = load_with_gc(with_gc_path, token_limit)

    if len(no_gc_times) != len(with_gc_times):
        raise ValueError('Mismatched length after loading')

    diffs_no_gc = [j - i for i, j in zip(no_gc_times[:-1], no_gc_times[1:])]
    diffs_with_gc = [j - i for i, j in zip(with_gc_times[:-1], with_gc_times[1:])]
    ratio = [b / a if a != 0 else 0.0 for a, b in zip(diffs_no_gc, diffs_with_gc)]

    out_csv = Path('analysis/results/token_speed.csv')
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['step', 'no_gc', 'with_gc', 'ratio'])
        for i, (a, b, r) in enumerate(zip(diffs_no_gc, diffs_with_gc, ratio), 1):
            writer.writerow([i, a, b, r])
    print(f'Saved speed data to {out_csv}')

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(14, 4))

    steps = range(1, len(diffs_no_gc) + 1)
    ax0.plot(steps, diffs_no_gc, label='trie w/o GC')
    ax0.plot(steps, diffs_with_gc, label='trie + GC')
    ax0.set_xlabel('Token index')
    ax0.set_ylabel('Time per token (s)')
    ax0.set_title('Token generation time')
    ax0.legend(frameon=False)
    ax0.grid(alpha=0.3)

    ax1.plot(steps, ratio, color='tab:red')
    ax1.set_xlabel('Token index')
    ax1.set_ylabel('GC / no-GC ratio')
    ax1.set_title('Relative speed')
    ax1.grid(alpha=0.3)

    fig.tight_layout()
    out_fig = Path('analysis/results/figs/token_speed_ratio.png')
    fig.savefig(out_fig, dpi=300)
    plt.close(fig)
    print(f'Figure saved to {out_fig}')


if __name__ == '__main__':
    main()

    """
    Example usage:
    python -m analysis.token_speed
    """