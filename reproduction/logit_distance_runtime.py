"""
Explanation:

### Notation:
* B = beam width (rows in each logit matrix)
* V = vocabulary size (columns)

At a decoding step t:
* T[t] ∈ ℝ^{BxV} - trie-search logits
* B[t] ∈ ℝ^{BxV} - baseline logits
* Ts[t], Bs[t] - tree-state dicts describing which tokens/paths occupy each beam

### Comparison modes:
1. Index distance:	
* compute_distance(T[t].ravel(), B[t].ravel()) → treats each (B,V) matrix as one long vector of length B·V. Fast, but assumes row i in the trie corresponds to row i in the baseline.

2. Tree distance:
* _step_tree_distance(T[t], B[t]) → builds a BxB distance matrix between beam rows, takes symmetric Chamfer average (mean row-min + mean col-min)/2. Robust to permuted or partially mismatched beams.
"""

import argparse
import csv
import json
from typing import List
import os

import numpy as np
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from reproduction.logit_test import (
    MODEL_CHOICES,
    DATASET_CHOICES,
    set_seed,
    record_baseline_logits,
    record_trie_logits,
)

METRICS = ["mse", "cosine", "kl"]


def compute_distance(a: np.ndarray, b: np.ndarray, metric: str) -> float:
    """Return distance between two 1-D arrays using the selected metric."""
    if metric == "mse":
        return float(np.mean((a - b) ** 2))
    if metric == "cosine":
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        return float(1 - np.dot(a, b) / denom) if denom != 0 else 0.0
    if metric == "kl":
        pa = np.exp(a)
        return float(np.sum(pa * (a - b)))
    raise ValueError(f"Unknown metric: {metric}")


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


def _distance_after_diverge(
    tree: list[np.ndarray],
    base: list[np.ndarray],
    tree_steps: list[dict],
    base_steps: list[dict],
    metric: str,
    use_tree: bool,
) -> float:
    """Compute average distance after decoding trees diverge."""
    steps = min(len(tree_steps), len(base_steps), len(tree), len(base))
    if steps == 0:
        return 0.0

    diverge = None
    for i in range(steps):
        if tree_steps[i] != base_steps[i]:
            diverge = i
            break
    if diverge is None:
        return 0.0

    dists: list[float] = []
    for j in range(diverge, steps):
        if use_tree:
            dists.append(_step_tree_distance(tree[j], base[j], metric))
        else:
            dists.append(compute_distance(tree[j].ravel(), base[j].ravel(), metric))
    return float(np.mean(dists)) if dists else 0.0


def _distance_until_diverge(
    tree: list[np.ndarray],
    base: list[np.ndarray],
    tree_steps: list[dict],
    base_steps: list[dict],
    metric: str,
    use_tree: bool,
) -> float:
    """Compute average distance up to the point where decoding trees diverge."""
    steps = min(len(tree_steps), len(base_steps), len(tree), len(base))
    if steps == 0:
        return 0.0
    dists: list[float] = []
    for i in range(steps):
        if tree_steps[i] != base_steps[i]:
            break
        if use_tree:
            dists.append(_step_tree_distance(tree[i], base[i], metric))
        else:
            dists.append(compute_distance(tree[i].ravel(), base[i].ravel(), metric))
    return float(np.mean(dists)) if dists else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute distance between trie and baseline beam search logits on the fly"
    )
    parser.add_argument("--model", choices=MODEL_CHOICES.keys(), help="Model choice")
    parser.add_argument("--dataset", choices=DATASET_CHOICES.keys(), help="Dataset choice")
    parser.add_argument(
        "--samples", type=int, default=None, help="Number of samples to evaluate (omit to use all)"
    )
    parser.add_argument("--beam_width", type=int, default=3)
    parser.add_argument("--max_new_tokens", type=int, default=50)
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional path to save per-step distances (.json or .csv)",
    )
    parser.add_argument(
        "--mode",
        choices=[
            "index",
            "tree",
            "index_until_diverge",
            "tree_until_diverge",
            "index_after_diverge",
            "tree_after_diverge",
        ],
        default="tree_until_diverge",
        help="Comparison strategy when beam structures differ",
    )
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    set_seed(args.seed)

    model_name = MODEL_CHOICES[args.model]
    ds_info = DATASET_CHOICES[args.dataset]

    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model.eval()

    dataset = load_dataset(ds_info["path"], ds_info["config"], split=ds_info["split"])
    if args.samples is not None:
        n = min(args.samples, len(dataset))
        dataset = dataset.select(range(n))

    eos: List[int] = [tokenizer.eos_token_id]
    distances = {m: [] for m in METRICS}

    for sample in dataset:
        prompt = sample[ds_info["text_column"]]
        tree_logits, tree_steps = record_trie_logits(
            model,
            tokenizer,
            prompt,
            args.beam_width,
            args.max_new_tokens,
            eos,
        )
        base_logits, base_steps = record_baseline_logits(
            model,
            tokenizer,
            prompt,
            args.beam_width,
            args.max_new_tokens,
            eos,
        )

        tree = [t.numpy() for t in tree_logits]
        base = [b.numpy() for b in base_logits]

        for i in range(min(len(tree), len(base))):
            print(
                f"Step {i}: trie logits shape {tree[i].shape}, baseline logits shape {base[i].shape}"
            )
            max_diff = np.max(np.abs(tree[i] - base[i]))
            print(f"Step {i}: max abs diff {max_diff}")
            if i < min(len(tree_steps), len(base_steps)) and tree_steps[i] != base_steps[i]:
                print(f"Beam divergence at step {i}:")
                print(f"  tree_steps[{i}]: {tree_steps[i]}")
                print(f"  base_steps[{i}]: {base_steps[i]}")

        if args.mode == "tree":
            for m in METRICS:
                distances[m].append(distance_different_tree(tree, base, m))
            continue
        if args.mode == "index_until_diverge":
            for m in METRICS:
                distances[m].append(
                    _distance_until_diverge(tree, base, tree_steps, base_steps, m, False)
                )
            continue
        if args.mode == "tree_until_diverge":
            for m in METRICS:
                distances[m].append(
                    _distance_until_diverge(tree, base, tree_steps, base_steps, m, True)
                )
            continue
        if args.mode == "index_after_diverge":
            for m in METRICS:
                distances[m].append(
                    _distance_after_diverge(tree, base, tree_steps, base_steps, m, False)
                )
            continue
        if args.mode == "tree_after_diverge":
            for m in METRICS:
                distances[m].append(
                    _distance_after_diverge(tree, base, tree_steps, base_steps, m, True)
                )
            continue

        steps = min(len(tree), len(base))
        for i in range(steps):
            for m in METRICS:
                distances[m].append(
                    compute_distance(tree[i].ravel(), base[i].ravel(), m)
                )

    if any(distances.values()):
        for m, vals in distances.items():
            if vals:
                print(f"Average {m}: {np.mean(vals):.6f}")
            else:
                print(f"No {m} computed")
    else:
        print("No distance computed")

    if args.output:
        records = [
            {m: distances[m][i] for m in METRICS}
            for i in range(len(next(iter(distances.values()), [])))
        ]
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        if args.output.endswith(".json"):
            with open(args.output, "w") as f:
                json.dump(records, f)
        elif args.output.endswith(".csv"):
            with open(args.output, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=METRICS)
                writer.writeheader()
                for row in records:
                    writer.writerow(row)
        else:
            raise ValueError("Output file must end with .json or .csv")


if __name__ == "__main__":
    main()

    """
    Example usage:  
    python -m reproduction.logit_distance_runtime \
        --model llama3 \
        --dataset human_eval \
        --samples 10 \
        --output analysis/results/logits/llama3/human_eval.json

    Available models: llama3, phi35, mistral, llama3_70b
    Available datasets: human_eval, gsm8k, cnn, wmt

    Test case:
    1. llama3, human_eval, 10 samples
    2. phi35, human_eval, 10 samples
    3. mistral, gsm8k, 10 samples
    ---
    4. llama3_70b, cnn, 10 samples
    5. phi35, cnn, 10 samples
    6. mistral, cnn, 10 samples
    """