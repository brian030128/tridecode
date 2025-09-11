"""
Compare memory usage and token speed over time for REASONING runs.

Reads JSONL outputs under reproduction/final_out/REASONING/{origin,tree,sample}/
and plots a large, multi-panel figure:
  - Columns: separate beams (no averaging across beams)
  - Rows: memory (GB) over steps, time per token (s) over steps
  - Lines in each subplot: origin (this beam), tree (this beam if exists), and sample (overlaid)

Saves figures to analysis/results/figs by default.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, DefaultDict

import numpy as np
import matplotlib.pyplot as plt


def _mean_across_samples(mem_lists: List[List[float]], min_frac: float = 0.8) -> np.ndarray:
    """
    Average sequences across samples per step, truncating at the last step
    where at least `min_frac` of samples still have data.
    """
    if not mem_lists:
        return np.array([])

    total = len(mem_lists)
    max_len = max(len(lst) for lst in mem_lists if lst is not None)
    if max_len == 0:
        return np.array([])

    padded = np.full((total, max_len), np.nan, dtype=float)
    for i, lst in enumerate(mem_lists):
        if lst is None:
            continue
        n = min(len(lst), max_len)
        padded[i, :n] = lst[:n]

    counts = np.sum(~np.isnan(padded), axis=0)
    threshold = total * min_frac
    valid = counts >= threshold
    if not np.any(valid):
        # fallback: nanmean across all positions
        return np.nanmean(padded, axis=0)
    last = int(np.max(np.where(valid))) + 1
    return np.nanmean(padded[:, :last], axis=0)


def _parse_meta(path: Path) -> Tuple[str, str, Optional[int]]:
    """
    Given a path like REASONING/origin/MATH500/3_2000.jsonl, return
    (variant, dataset, beam). For sample variant, beam is None.
    """
    # find index of 'REASONING' component
    parts = path.parts
    try:
        i = parts.index("REASONING")
    except ValueError:
        raise ValueError(f"Path does not contain 'REASONING': {path}")
    # expected: .../REASONING/<variant>/<dataset>/<file>
    variant = parts[i + 1]
    dataset = parts[i + 2]
    beam: Optional[int] = None
    if variant != "sample":
        fname = parts[i + 3]
        try:
            beam = int(fname.split("_")[0])
        except Exception:
            beam = None
    return variant, dataset, beam


def _load_series_by_beam(base_dir: Path):
    """
    Load per-sample sequences grouped by dataset and beam for origin/tree,
    and dataset-level sequences for sample.

    Returns two dicts:
      - by_beam[(dataset, beam)][variant]['memory'|'times'] -> List[List[float]]
      - sample_by_dataset[dataset]['memory'|'times'] -> List[List[float]]
    """
    from collections import defaultdict

    by_beam: DefaultDict[Tuple[str, int], Dict[str, Dict[str, List[List[float]]]]] = defaultdict(
        lambda: {"origin": {"memory": [], "times": []}, "tree": {"memory": [], "times": []}}
    )
    sample_by_dataset: DefaultDict[str, Dict[str, List[List[float]]]] = defaultdict(
        lambda: {"memory": [], "times": []}
    )

    for variant in ("origin", "tree", "sample"):
        vdir = base_dir / variant
        if not vdir.exists():
            continue
        for path in vdir.rglob("*.jsonl"):
            try:
                v, dataset, beam = _parse_meta(path)
            except Exception:
                continue
            with path.open() as f:
                for line in f:
                    if not line.strip():
                        continue
                    obj = json.loads(line)
                    mem = obj.get("memory_usage") or []
                    ts = obj.get("time_metric") or []
                    if not mem or not ts:
                        continue
                    if v == "sample":
                        sample_by_dataset[dataset]["memory"].append(mem)
                        sample_by_dataset[dataset]["times"].append(ts)
                    else:
                        if beam is None:
                            continue
                        by_beam[(dataset, beam)][v]["memory"].append(mem)
                        by_beam[(dataset, beam)][v]["times"].append(ts)

    return by_beam, sample_by_dataset


def compute_aggregates_by_beam(
    by_beam, sample_by_dataset
):
    """
    Compute averaged memory and time-per-token per variant, grouped by (dataset, beam).

    Returns:
      - mem_curves[(dataset, beam)][variant] -> np.ndarray (GB per step)
      - tpt_curves[(dataset, beam)][variant] -> np.ndarray (seconds per token)
      - sample_curves[dataset] -> dict with 'memory' (GB) and 'tpt' arrays
    """
    mem_curves: Dict[Tuple[str, int], Dict[str, np.ndarray]] = {}
    tpt_curves: Dict[Tuple[str, int], Dict[str, np.ndarray]] = {}

    for (dataset, beam), variants in by_beam.items():
        mem_curves[(dataset, beam)] = {}
        tpt_curves[(dataset, beam)] = {}
        for variant, data in variants.items():
            mem_lists = data.get("memory", [])
            time_lists = data.get("times", [])

            mem_avg = _mean_across_samples(mem_lists, min_frac=0.8)
            mem_curves[(dataset, beam)][variant] = (
                mem_avg / 1024.0 if mem_avg.size > 0 else mem_avg
            )

            deltas: List[List[float]] = []
            for ts in time_lists:
                if len(ts) >= 2:
                    deltas.append(np.diff(ts).tolist())
            avg_delta = _mean_across_samples(deltas, min_frac=0.8)
            tpt_curves[(dataset, beam)][variant] = avg_delta

    sample_curves: Dict[str, Dict[str, np.ndarray]] = {}
    for dataset, data in sample_by_dataset.items():
        mem_avg = _mean_across_samples(data.get("memory", []), min_frac=0.8)
        deltas: List[List[float]] = []
        for ts in data.get("times", []):
            if len(ts) >= 2:
                deltas.append(np.diff(ts).tolist())
        tpt_avg = _mean_across_samples(deltas, min_frac=0.8)
        sample_curves[dataset] = {
            "memory": mem_avg / 1024.0 if mem_avg.size > 0 else mem_avg,
            "tpt": tpt_avg,
        }

    return mem_curves, tpt_curves, sample_curves


def plot_reasoning_by_beam(
    mem_curves: Dict[Tuple[str, int], Dict[str, np.ndarray]],
    tpt_curves: Dict[Tuple[str, int], Dict[str, np.ndarray]],
    sample_curves: Dict[str, Dict[str, np.ndarray]],
    out_path: Path,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # consistent variant colors/styles
    colors = {
        "origin": "#377eb8",  # blue
        "tree": "#4daf4a",    # green
    }
    sample_color = "#999999"     # gray

    # group by dataset, collect beams
    datasets = sorted({ds for (ds, _b) in mem_curves.keys()})
    for dataset in datasets:
        beams = sorted({b for (ds, b) in mem_curves.keys() if ds == dataset})
        if not beams:
            continue

        fig, axes = plt.subplots(2, len(beams), figsize=(4 * len(beams), 6), squeeze=False)

        for j, beam in enumerate(beams):
            key = (dataset, beam)
            # Memory (row 0)
            ax_mem = axes[0, j]
            for variant in ("origin", "tree"):
                mem = mem_curves.get(key, {}).get(variant)
                if mem is None or len(mem) == 0:
                    continue
                steps = np.arange(1, len(mem) + 1)
                label = "trie" if variant == "tree" else variant
                ax_mem.plot(steps, mem, label=label, color=colors.get(variant), linewidth=1.6)

            # overlay sample if available (same across beams)
            samp = sample_curves.get(dataset, {}).get("memory")
            if samp is not None and len(samp) > 0:
                ax_mem.plot(
                    np.arange(1, len(samp) + 1),
                    samp,
                    label="sample",
                    color=sample_color,
                    linestyle="--",
                    linewidth=1.4,
                )

            ax_mem.set_title(f"{dataset} · beam={beam} · memory")
            ax_mem.set_xlabel("Decoding step")
            ax_mem.set_ylabel("Memory (GB)")
            ax_mem.grid(alpha=0.3)
            if j == len(beams) - 1:
                ax_mem.legend(frameon=False, fontsize=8)

            # Time per token (row 1)
            ax_tpt = axes[1, j]
            for variant in ("origin", "tree"):
                tpt = tpt_curves.get(key, {}).get(variant)
                if tpt is None or len(tpt) == 0:
                    continue
                steps = np.arange(1, len(tpt) + 1)
                label = "trie" if variant == "tree" else variant
                ax_tpt.plot(steps, tpt, label=label, color=colors.get(variant), linewidth=1.6)

            samp_tpt = sample_curves.get(dataset, {}).get("tpt")
            if samp_tpt is not None and len(samp_tpt) > 0:
                ax_tpt.plot(
                    np.arange(1, len(samp_tpt) + 1),
                    samp_tpt,
                    label="sample",
                    color=sample_color,
                    linestyle="--",
                    linewidth=1.4,
                )

            ax_tpt.set_title(f"{dataset} · beam={beam} · time/token")
            ax_tpt.set_xlabel("Token index")
            ax_tpt.set_ylabel("Time per token (s)")
            ax_tpt.grid(alpha=0.3)
            if j == len(beams) - 1:
                ax_tpt.legend(frameon=False, fontsize=8)

        fig.suptitle(f"REASONING performance by beam — {dataset}", fontsize=12)
        fig.tight_layout(rect=[0, 0, 1, 0.96])

        ds_sanitized = dataset.replace("/", "_")
        file_path = out_path.with_name(f"reasoning_by_beam_{ds_sanitized}.png")
        fig.savefig(file_path, dpi=300)
        plt.close(fig)
        print(f"Saved figure to {file_path}")


def main():
    base_dir = Path("reproduction/final_out/REASONING")
    if not base_dir.exists():
        raise SystemExit(f"Base directory not found: {base_dir}")

    by_beam, sample_by_dataset = _load_series_by_beam(base_dir)
    mem_curves, tpt_curves, sample_curves = compute_aggregates_by_beam(by_beam, sample_by_dataset)

    out_fig = Path("analysis/results/figs/reasoning_by_beam.png")
    plot_reasoning_by_beam(mem_curves, tpt_curves, sample_curves, out_fig)


if __name__ == "__main__":
    main()

"""
Usage:
  python -m analysis.visualization.reasoning_perf

Outputs:
  analysis/results/figs/reasoning_by_beam_<DATASET>.png
"""
