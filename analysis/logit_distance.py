import argparse
import json
from pathlib import Path
import numpy as np
from scipy.spatial.distance import cosine


def compute_distance(a: np.ndarray, b: np.ndarray, metric: str) -> float:
    """Return distance between two 1-D arrays using the selected metric."""
    if metric == "mse":
        return float(np.mean((a - b) ** 2))
    return float(cosine(a, b))


def _step_tree_distance(step_a: np.ndarray, step_b: np.ndarray, metric: str) -> float:
    """Compute distance when beam structures differ.

    Each input is shaped ``(beam, vocab)``. The distance is the average of the
    minimal pairwise distances between beams from ``step_a`` and ``step_b``.
    """
    dist = np.zeros((step_a.shape[0], step_b.shape[0]), dtype=float)
    for i, a in enumerate(step_a):
        for j, b in enumerate(step_b):
            dist[i, j] = compute_distance(a.ravel(), b.ravel(), metric)
    row_min = dist.min(axis=1)
    col_min = dist.min(axis=0)
    return float((row_min.mean() + col_min.mean()) / 2)


def distance_different_tree(tree: list[np.ndarray], base: list[np.ndarray], metric: str) -> float:
    """Average distance for two sequences of beam logits with mismatched trees."""
    steps = min(len(tree), len(base))
    if steps == 0:
        return 0.0
    dists = [_step_tree_distance(tree[i], base[i], metric) for i in range(steps)]
    return float(np.mean(dists))


def main():
    parser = argparse.ArgumentParser(description="Compute distance between logit distributions")
    parser.add_argument("logit_file", help="JSON file produced by logit_test.py")
    parser.add_argument("--metric", choices=["mse", "cosine"], default="mse")
    parser.add_argument(
        "--mode",
        choices=["index", "tree"],
        default="index",
        help="Comparison strategy when beam structures differ",
    )
    args = parser.parse_args()

    data = json.loads(Path(args.logit_file).read_text())
    distances = []
    for sample in data:
        tree = [np.array(t) for t in sample["tree"]]
        base = [np.array(b) for b in sample["baseline"]]
        if args.mode == "tree":
            distances.append(distance_different_tree(tree, base, args.metric))
            continue
        steps = min(len(tree), len(base))
        for i in range(steps):
            distances.append(compute_distance(tree[i].ravel(), base[i].ravel(), args.metric))

    if distances:
        print(f"Average {args.metric}: {np.mean(distances):.6f}")
    else:
        print("No distance computed")


if __name__ == "__main__":
    main()
