#!/usr/bin/env python3
"""Aggregate token usage for MATH500 across models and decoding methods."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

# Map directory names to the model labels requested by the user.
MODEL_DIRS: Dict[str, str] = {
    "LLAMA3": "Llama",
    "MISTRAL": "Mistral",
    "PHI35": "PHI3.5",
}

# Map decoding directory names to the labels we want in the CSV.
DECODE_DIRS: Dict[str, str] = {
    "origin": "origin",
    "tree": "tree",
    "sample": "sampling",
}


def coerce_int(value: Optional[int]) -> Optional[int]:
    """Attempt to coerce the value to an int, returning None on failure."""

    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def iter_records(base_dir: Path) -> Iterator[Dict[str, object]]:
    """Yield per-problem records with token statistics."""

    for model_dir, model_label in MODEL_DIRS.items():
        for decode_dir, decode_label in DECODE_DIRS.items():
            math500_dir = base_dir / model_dir / decode_dir / "MATH500"
            if not math500_dir.exists():
                continue

            for jsonl_path in sorted(math500_dir.glob("*.jsonl")):
                with jsonl_path.open("r", encoding="utf-8") as handle:
                    for line_number, raw_line in enumerate(handle, start=1):
                        raw_line = raw_line.strip()
                        if not raw_line:
                            continue

                        try:
                            record = json.loads(raw_line)
                        except json.JSONDecodeError:
                            print(
                                f"Skipping invalid JSON in {jsonl_path}:{line_number}",
                                file=sys.stderr,
                            )
                            continue

                        input_tokens = coerce_int(record.get("input_len"))
                        output_tokens = coerce_int(record.get("output_len"))

                        if input_tokens is None and output_tokens is None:
                            continue

                        total_tokens = None
                        if input_tokens is not None or output_tokens is not None:
                            total_tokens = (input_tokens or 0) + (output_tokens or 0)

                        problem_id = record.get("id")
                        if isinstance(problem_id, str) and problem_id.strip():
                            problem_key = problem_id.strip()
                        else:
                            problem_key = f"{jsonl_path.stem}#{line_number}"

                        yield {
                            "model": model_label,
                            "decoding_method": decode_label,
                            "problem_id": problem_key,
                            "input_tokens": input_tokens,
                            "output_tokens": output_tokens,
                            "total_tokens": total_tokens,
                            "score": record.get("score"),
                            "source_file": str(jsonl_path.relative_to(base_dir)),
                            "row_index": line_number,
                        }


def write_detailed_csv(rows: List[Dict[str, object]], output_path: Path) -> None:
    """Write the per-problem rows to CSV."""

    fieldnames = [
        "model",
        "decoding_method",
        "problem_id",
        "input_tokens",
        "output_tokens",
        "total_tokens",
        "score",
        "source_file",
        "row_index",
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            sanitized = {key: ("" if value is None else value) for key, value in row.items()}
            writer.writerow(sanitized)


def write_summary_csv(rows: List[Dict[str, object]], output_path: Path) -> None:
    """Create a summary CSV aggregating token counts per model and decoding method."""

    aggregates: Dict[Tuple[str, str], Dict[str, float]] = defaultdict(
        lambda: {"count": 0, "avg_input_tokens": 0.0, "avg_output_tokens": 0.0, "avg_total_tokens": 0.0}
    )

    for row in rows:
        key = (row["model"], row["decoding_method"])
        agg = aggregates[key]
        agg["count"] += 1

        for field, avg_key in (
            ("input_tokens", "avg_input_tokens"),
            ("output_tokens", "avg_output_tokens"),
            ("total_tokens", "avg_total_tokens"),
        ):
            token_value = row.get(field)
            if isinstance(token_value, int):
                agg[avg_key] += token_value

    summary_rows = []
    for (model, decoding_method), agg in sorted(aggregates.items()):
        count = agg["count"]
        if not count:
            continue
        summary_rows.append(
            {
                "model": model,
                "decoding_method": decoding_method,
                "num_problems": count,
                "avg_input_tokens": round(agg["avg_input_tokens"] / count, 2),
                "avg_output_tokens": round(agg["avg_output_tokens"] / count, 2),
                "avg_total_tokens": round(agg["avg_total_tokens"] / count, 2),
            }
        )

    if not summary_rows:
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "model",
        "decoding_method",
        "num_problems",
        "avg_input_tokens",
        "avg_output_tokens",
        "avg_total_tokens",
    ]

    with output_path.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    default_base = project_root / "reproduction" / "final_out"
    default_output = Path(__file__).resolve().parent / "math500_token_usage.csv"
    default_summary = Path(__file__).resolve().parent / "math500_token_usage_summary.csv"

    parser = argparse.ArgumentParser(
        description=(
            "Export per-problem token usage for the MATH500 dataset "
            "across Llama, Mistral, and PHI3.5 with origin/tree/sample decoding."
        )
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=default_base,
        help="Path to the root directory containing the final_out results.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=default_output,
        help="Path to the detailed CSV output file.",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=default_summary,
        help="Optional path for the aggregated summary CSV.",
    )
    parser.add_argument(
        "--skip-summary",
        action="store_true",
        help="Disable writing the summary CSV.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_dir: Path = args.base_dir

    if not base_dir.exists():
        raise SystemExit(f"Base directory {base_dir} does not exist")

    rows = list(iter_records(base_dir))
    if not rows:
        raise SystemExit("No token usage data found.")

    write_detailed_csv(rows, args.output)

    if not args.skip_summary:
        write_summary_csv(rows, args.summary_output)

    print(f"Wrote {len(rows)} rows to {args.output}")
    if not args.skip_summary:
        print(f"Wrote summary to {args.summary_output}")


if __name__ == "__main__":
    main()
