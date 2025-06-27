import os
import json
import argparse
import csv
from pathlib import Path
from typing import Dict, List, Tuple
import statistics

CUSTOM_COLORS = [
    "#e41a1c",  # red
    "#377eb8",  # blue
    "#4daf4a",  # green
    "#984ea3",  # purple
    "#ff7f00",  # orange
    "#ffff33",  # yellow
    "#a65628",  # brown
    "#f781bf",  # pink
    "#999999",  # gray
]


def _assign_colors(beams: List[int]) -> Dict[int, str]:
    """Map each beam in the list to a color from CUSTOM_COLORS."""
    return {b: CUSTOM_COLORS[i % len(CUSTOM_COLORS)] for i, b in enumerate(sorted(beams))}


def choose_beams(trends: Dict[Tuple[str, str, int], List[float]], model: str, dataset: str, n_max: int = 3) -> List[int]:
    beams = sorted(b for (m, d, b) in trends.keys() if m == model and d == dataset)
    if len(beams) <= n_max:
        return beams
    # pick smallest(greedy), second largest, and largest
    return [beams[0], beams[-2], beams[-1]]


def _load_memory(path: str) -> Dict[str, List[float]]:
    """Load memory usage per sample minus model memory."""
    data = {}
    with open(path) as f:
        for line in f:
            obj = json.loads(line)
            mem = [float(m) - float(obj.get("model_memory", 0)) for m in obj.get("memory_usage", [])]
            data[obj["id"]] = mem
    return data

def _load_memory(path: str) -> Dict[str, List[float]]:
    """Load memory usage per sample minus model memory."""
    data = {}
    with open(path) as f:
        for line in f:
            obj = json.loads(line)
            mem = [float(m) - float(obj.get("model_memory", 0)) for m in obj.get("memory_usage", [])]
            data[obj["id"]] = mem
    return data


def _step_ratio(orig_path: str, tree_path: str) -> Tuple[List[float], float]:
    orig = _load_memory(orig_path)
    tree = _load_memory(tree_path)
    ids = set(orig) & set(tree)
    if not ids:
        return [], float('nan')

    step_sums: List[float] = []
    step_counts: List[int] = []
    for sid in ids:
        o = orig[sid]
        t = tree[sid]
        n = min(len(o), len(t))
        for i in range(n):
            r = (o[i] - t[i]) / o[i] if o[i] != 0 else 0.0
            if i >= len(step_sums):
                step_sums.append(r)
                step_counts.append(1)
            else:
                step_sums[i] += r
                step_counts[i] += 1
    ratios = [s / c for s, c in zip(step_sums, step_counts)]
    avg = statistics.mean(ratios) if ratios else float('nan')
    return ratios, avg


def analyze(base_dir: str):
    summary = []
    trends: Dict[Tuple[str, str, int], List[float]] = {}
    for model in sorted(os.listdir(base_dir)):
        model_dir = os.path.join(base_dir, model)
        if not os.path.isdir(model_dir):
            continue
        origin_base = os.path.join(model_dir, 'origin')
        tree_base = os.path.join(model_dir, 'tree')
        if not os.path.isdir(origin_base) or not os.path.isdir(tree_base):
            continue
        for dataset in sorted(os.listdir(origin_base)):
            o_dir = os.path.join(origin_base, dataset)
            t_dir = os.path.join(tree_base, dataset)
            if not os.path.isdir(t_dir):
                continue
            for fname in os.listdir(o_dir):
                if not fname.endswith('.jsonl'):
                    continue
                beam = int(fname.split('_', 1)[0])
                o_path = os.path.join(o_dir, fname)
                t_path = os.path.join(t_dir, fname)
                if not os.path.exists(t_path):
                    continue
                step_ratio, avg = _step_ratio(o_path, t_path)
                summary.append({
                    'model': model,
                    'dataset': dataset,
                    'beam': beam,
                    'avg_ratio': avg,
                })
                trends[(model, dataset, beam)] = step_ratio
    return summary, trends


def save_trends(trends: Dict[Tuple[str, str, int], List[float]], out_csv: str) -> None:
    with open(out_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['model', 'dataset', 'beam', 'step', 'ratio'])
        for (model, dataset, beam), arr in trends.items():
            for step, val in enumerate(arr, 1):
                writer.writerow([model, dataset, beam, step, val])


def plot_trends(trends: Dict[Tuple[str, str, int], List[float]], out_dir: str) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print('matplotlib not installed; skipping plots.')
        return

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    combos = {(m, d) for (m, d, _) in trends.keys()}
    for model, dataset in sorted(combos):
        plt.figure(figsize=(6, 4))
        beams = sorted(b for (m, d, b) in trends.keys() if m == model and d == dataset)
        for beam in beams:
            arr = trends.get((model, dataset, beam))
            if not arr:
                continue
            steps = list(range(1, len(arr) + 1))
            plt.plot(steps, arr, label=f'beam={beam}')
        plt.axhline(0, color='black', lw=0.8, ls='--')
        plt.title(f'(origin - trie)/origin\n{model} · {dataset}')
        plt.xlabel('Decoding step')
        plt.ylabel('Relative memory diff')
        plt.legend(frameon=False, fontsize=8)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_path / f'{model}_{dataset}_trend.png', dpi=300)
        plt.close()

def plot_combined_trends(
    trends: Dict[Tuple[str, str, int], List[float]],
    out_path: Path = Path("./reproduction/figs/combined_overlap_trend.png"),
    orientation: str = "vertical",
) -> None:
    """Draw a grid of overlap trend curves similar to combined memory plot."""
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("matplotlib not installed; skipping combined plot.")
        return

    models = sorted({m for (m, _, _) in trends.keys()})
    datasets = sorted({d for (_, d, _) in trends.keys()})

    if orientation == "horizontal":
        nrows, ncols = len(datasets), len(models)
    elif orientation == "vertical":
        nrows, ncols = len(models), len(datasets)
    else:
        raise ValueError("orientation must be 'horizontal' or 'vertical'")

    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows), squeeze=False)

    for i in range(nrows):
        for j in range(ncols):
            if orientation == "horizontal":
                dataset = datasets[i]
                model = models[j]
            else:
                model = models[i]
                dataset = datasets[j]

            ax = axes[i, j]
            beams = choose_beams(trends, model, dataset)
            if not beams:
                fig.delaxes(ax)
                continue

            beam_colors = _assign_colors(beams)
            for beam in beams:
                arr = trends.get((model, dataset, beam))
                if not arr:
                    continue
                steps = np.arange(1, len(arr) + 1)
                ax.plot(
                    steps,
                    arr,
                    color=beam_colors[beam],
                    label=f"beam={beam}",
                    linewidth=1.2,
                )

            ax.axhline(0, color="black", lw=0.8, ls="--")

            d_label = "HumanEval" if dataset == "HUMAN_EVAL" else dataset

            if model == "PHI35":
                m_label = "Phi-3.5"
            elif model == "LLAMA3":
                m_label = "Llama 3.1"
            elif model == "MISTRAL":
                m_label = "Mistral Small"
            else:
                m_label = model

            title = f"{m_label} · {d_label}"
            ax.set_title(title, fontsize=10)
            ax.set_xlabel("Decoding step", fontsize=8)
            ax.set_ylabel("Relative memory diff", fontsize=8)
            ax.grid(alpha=0.3)
            ax.legend(frameon=False, fontsize=6)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)

def main():
    parser = argparse.ArgumentParser(description='Analyze memory overlap between origin and trie decoding')
    parser.add_argument('--base_dir', default='reproduction/out', help='Directory with model outputs')
    parser.add_argument('--output_csv', default='memory_overlap.csv', help='Summary CSV')
    parser.add_argument('--trend_csv', default='memory_overlap_trend.csv', help='Step-wise ratio CSV')
    parser.add_argument('--fig_dir', default='reproduction/figs', help='Directory to save trend plots')
    args = parser.parse_args()

    summary, trends = analyze(args.base_dir)
    if not summary:
        print('No results found.')
        return

    with open(args.output_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        writer.writeheader()
        writer.writerows(summary)
    print(f'Saved summary to {args.output_csv}')

    save_trends(trends, args.trend_csv)
    print(f'Saved trend data to {args.trend_csv}')

    plot_trends(trends, args.fig_dir)
    print(f'Trend figures saved to {args.fig_dir}')

    plot_combined_trends(trends, Path(args.fig_dir) / 'combined_overlap_trend.png')
    print(f'Combined trend figure saved to {Path(args.fig_dir) / "combined_overlap_trend.png"}')


if __name__ == '__main__':
    main()
