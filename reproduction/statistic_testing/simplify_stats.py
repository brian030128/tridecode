#!/usr/bin/env python3
import argparse
import pandas as pd

ALPHA = 0.05

def verdict(row, metric, higher_is_better=True):
    diff = row.get(f"{metric}_diff_mean", float("nan"))
    pval = row.get(f"{metric}_t_pval", float("nan"))
    if pd.isna(diff) or pd.isna(pval):
        return "n/a"
    if pval >= ALPHA:
        return "no_sig"
    return "better" if (diff > 0) == higher_is_better else "worse"

def verdict_binary(row):
    b = row.get("mcnemar_b", float("nan"))
    c = row.get("mcnemar_c", float("nan"))
    pval = row.get("mcnemar_pval", float("nan"))
    if pd.isna(b) or pd.isna(c) or pd.isna(pval):
        return "n/a"
    if pval >= ALPHA:
        return "no_sig"

    return "better" if c > b else "worse"

def simplify(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in df.iterrows():
        common = {
            "model":   r["model"],
            "dataset": r["dataset"],
            "beam":    r["beam"],
            "samples": r["samples"],
        }

        # input_kv_memory (lower is better)
        rows.append({
            **common,
            "metric":  "input_kv_memory",
            "verdict": verdict(r, "input_kv_memory", higher_is_better=False),
            "ci_lower": r.get("input_kv_memory_ci_lower", ""),
            "ci_upper": r.get("input_kv_memory_ci_upper", ""),
        })

        # tok_per_sec (higher is better)
        rows.append({
            **common,
            "metric":  "tok_per_sec",
            "verdict": verdict(r, "tok_per_sec", higher_is_better=True),
            "ci_lower": r.get("tok_per_sec_ci_lower", ""),
            "ci_upper": r.get("tok_per_sec_ci_upper", ""),
        })

        # score
        if "mcnemar_pval" in r and not pd.isna(r["mcnemar_pval"]):
            # binary case: use odds‐ratio CI
            rows.append({
                **common,
                "metric":  "score",
                "verdict": verdict_binary(r),
                "ci_lower": r.get("mcnemar_or_ci_low", ""),
                "ci_upper": r.get("mcnemar_or_ci_high", ""),
            })
        else:
            # continuous case: paired‐t
            rows.append({
                **common,
                "metric":  "score",
                "verdict": verdict(r, "score", higher_is_better=True),
                "ci_lower": r.get("score_ci_lower", ""),
                "ci_upper": r.get("score_ci_upper", ""),
            })

    return pd.DataFrame(rows)

def main():
    parser = argparse.ArgumentParser(
        description="Simplify a detailed beam-search comparison CSV into verdict + CI only"
    )
    parser.add_argument(
        "detailed_csv",
        help="Path to the detailed results CSV (with _diff_mean, _ci_lower, mcnemar_*, etc.)"
    )
    parser.add_argument(
        "simplified_csv",
        help="Output path for the simplified CSV"
    )
    args = parser.parse_args()

    df = pd.read_csv(args.detailed_csv)
    simple = simplify(df)
    simple.to_csv(args.simplified_csv, index=False)
    print(f"Simplified CSV written to {args.simplified_csv}")

if __name__ == "__main__":
    main()

    """
    Example usage:
    python -m reproduction.statistic_testing.simplify_stats reproduction/statistic_testing_results.csv reproduction/simplified_results.csv
    """
