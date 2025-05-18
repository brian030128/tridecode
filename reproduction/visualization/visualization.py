"""
Plot average memory-usage-per-decoding-step for final_out/{model}/{origin|tree}/{dataset}/{beam_samples}.jsonl
and save the figures to ./figs/.
"""

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

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
    """
    Map each beam in the list to a color from CUSTOM_COLORS.
    """
    return {b: CUSTOM_COLORS[i % len(CUSTOM_COLORS)] for i, b in enumerate(sorted(beams))}


def _parse_path(p: Path) -> Tuple[str, str, str, int]:
    """
    Return (model, variant, dataset, beam) for a path like
    final_out/LLAMA3/origin/CNN/1_1000.jsonl
    """
    # ensure we are looking after .../final_out/...
    parts = p.parts[p.parts.index("final_out") + 1 :]
    model, variant, dataset, fname = parts[:4]
    beam = int(fname.split("_")[0])  # '1_1000.jsonl' -> 1
    return model, variant, dataset, beam


def _mean_across_samples(mem_lists: List[List[float]], min_frac: float = 0.8) -> np.ndarray:
    """
    Average memory usage across all samples per step,
    truncating at the last step where the count of samples >= min_frac of total.
    """
    # pad to equal length
    total = len(mem_lists)
    max_len = max(len(lst) for lst in mem_lists)
    padded = np.full((total, max_len), np.nan, dtype=float)
    for i, lst in enumerate(mem_lists):
        padded[i, : len(lst)] = lst
    # count non-nan per column
    counts = np.sum(~np.isnan(padded), axis=0)
    # find last index where count >= threshold
    threshold = total * min_frac
    valid = counts >= threshold
    if not np.any(valid):
        return np.array([])
    last = int(np.max(np.where(valid))) + 1
    # compute nanmean up to last
    return np.nanmean(padded[:, :last], axis=0)


def load_all(base_dir: str = "final_out") -> Dict:
    """
    Returns a dictionary:
        keyed[(model, dataset, beam, variant)] -> 1-D np.ndarray (avg mem per step)
    """
    agg: Dict[Tuple[str, str, int, str], List[List[float]]] = defaultdict(list)

    for path in Path(base_dir).rglob("*.jsonl"):
        model, variant, dataset, beam = _parse_path(path)

        if dataset == "GSM8K" or dataset == "WMT":
            continue

        with path.open() as f:
            for line in f:
                mem = json.loads(line)["memory_usage"]
                agg[(model, dataset, beam, variant)].append(mem)

    # convert lists->average curves
    curves = {
        key: _mean_across_samples(mem_lists) for key, mem_lists in agg.items()
    }
    return curves

def choose_beams(curves, model, dataset, n_max=3) -> List[int]:
    beams = sorted(
        {b for (m, d, b, _), _curve in curves.items() if m == model and d == dataset}
    )

    if len(beams) <= n_max:
        return beams
    
    # largest, second largest, smallest(greedy)
    return [beams[0], beams[-2], beams[-1]]


def plot_all(curves: Dict, out_dir: Path = Path("./reproduction/figs")):
    out_dir.mkdir(parents=True, exist_ok=True)
    cmap = plt.get_cmap("tab10")

    for model, dataset in sorted({(m, d) for (m, d, _, _) in curves.keys()}):
        plt.figure(figsize=(6, 4))
        beams = choose_beams(curves, model, dataset)
        # assign a distinct color per beam in this figure
        beam_colors = _assign_colors(beams)

        for beam in beams:
            for variant, ls in [("origin", "--"), ("trie", "-")]:
                key = (model, dataset, beam, variant)
                if key not in curves or len(curves[key]) == 0:
                    continue
                mem_gb = curves[key] / 1024.0
                steps = np.arange(len(mem_gb))
                beam_label = f"beam={beam}" if beam > 1 else "greedy (beam=1)"

                plt.plot(
                    steps,
                    mem_gb,
                    linestyle=ls,
                    color=beam_colors[beam],
                    label=f"{variant} ({beam_label})",
                    linewidth=1.8,
                )
        plt.title(f"Average GPU memory over decoding steps\n{model} · {dataset}")
        plt.xlabel("Decoding step")
        plt.ylabel("Memory (GB)")
        plt.legend(frameon=False, fontsize=8)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_dir / f"{model}_{dataset}_memory.png", dpi=300)
        plt.close()


def plot_combined(
    curves: Dict,
    out_path: Path = Path("./reproduction/figs/combined_memory.png"),
):
    models = sorted({m for (m, _, _, _) in curves.keys()})
    ds_map = {m: sorted({d for (m2, d, _, _) in curves.keys() if m2 == m}) for m in models}
    ncols = len(models)
    nrows = max(len(ds_map[m]) for m in models)
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows))
    cmap = plt.get_cmap("tab10")

    for col, model in enumerate(models):
        for row, dataset in enumerate(ds_map[model]):
            ax = axes[row, col] if nrows > 1 else axes[col]
            beams = choose_beams(curves, model, dataset)
            # assign colors per beam for this subplot
            beam_colors = _assign_colors(beams)

            for beam in beams:
                beam_label = f"beam={beam}" if beam > 1 else "greedy (beam=1)"  
                for variant, ls in [("origin", "--"), ("tree", "-")]:
                    key = (model, dataset, beam, variant)
                    if key not in curves or len(curves[key]) == 0:
                        continue
                    mem_gb = curves[key] / 1024.0
                    steps = np.arange(len(mem_gb))
                    ax.plot(
                        steps,
                        mem_gb,
                        linestyle=ls,
                        color=beam_colors[beam],
                        label=f"{'trie' if variant == 'tree' else variant} ({beam_label})",
                        linewidth=1.2,
                    )
            ax.set_title(f"{model} · {dataset}", fontsize=10)
            ax.set_xlabel("Decoding step", fontsize=8)
            ax.set_ylabel("Memory (GB)", fontsize=8)
            ax.grid(alpha=0.3)
            ax.legend(frameon=False, fontsize=6)

    # remove extra axes
    for r in range(nrows):
        for c in range(ncols):
            if r >= len(ds_map[models[c]]):
                fig.delaxes(axes[r, c])

    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"Combined figure saved to {out_path}")

if __name__ == "__main__":
    curves = load_all("./reproduction/final_out")
    save_path = "./reproduction/figs"
    plot_all(curves, Path(save_path))
    plot_combined(curves, Path("./reproduction/figs/combined_memory.png"))
    print(f"Figures saved in {save_path}")

    """
    Example usage:
    python -m reproduction.visualization.visualization
    """
