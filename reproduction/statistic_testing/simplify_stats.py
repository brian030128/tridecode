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

def compute_margin(ci_low, ci_high):
    """Compute half-width of a confidence interval."""
    try:
        low = float(ci_low)
        high = float(ci_high)
        return (high - low) / 2
    except Exception:
        return ""


def simplify(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in df.iterrows():
        common = {
            "model":   r["model"],
            "dataset": r["dataset"],
            "beam":    r["beam"],
            "samples": r["samples"],
        }

        # 1) mem_per_token_ (paired t‐test, lower is better)
        metric = "mem_per_token"
        ci_l = r.get(f"{metric}_ci_lower", "")
        ci_u = r.get(f"{metric}_ci_upper", "")
        rows.append({
            **common,
            "metric":       metric,
            "origin_mean":  r.get(f"{metric}_orig_mean", ""),
            "tree_mean":    r.get(f"{metric}_tree_mean", ""),
            "margin":       compute_margin(ci_l, ci_u),
            "verdict":      verdict(r, metric, higher_is_better=False),
            "p_value":      r.get(f"{metric}_t_pval", ""),
            "test":         "paired t-test",
            "ci_lower":     ci_l,
            "ci_upper":     ci_u,
        })

        # 2) tok_per_sec (paired t‐test, higher is better)
        metric = "tok_per_sec"
        ci_l = r.get(f"{metric}_ci_lower", "")
        ci_u = r.get(f"{metric}_ci_upper", "")
        rows.append({
            **common,
            "metric":       metric,
            "origin_mean":  r.get(f"{metric}_orig_mean", ""),
            "tree_mean":    r.get(f"{metric}_tree_mean", ""),
            "margin":       compute_margin(ci_l, ci_u),
            "verdict":      verdict(r, metric, higher_is_better=True),
            "p_value":      r.get(f"{metric}_t_pval", ""),
            "test":         "paired t-test",
            "ci_lower":     ci_l,
            "ci_upper":     ci_u,
        })

        # 3) score: choose McNemar for binary, paired‐t for continuous
        if not pd.isna(r.get("mcnemar_pval")):
            # binary
            ci_l = r.get("mcnemar_or_ci_low", "")
            ci_u = r.get("mcnemar_or_ci_high", "")
            rows.append({
                **common,
                "metric":       "score",
                "origin_mean":  r.get("score_orig_mean", ""),
                "tree_mean":    r.get("score_tree_mean", ""),
                "margin":       compute_margin(ci_l, ci_u),
                "verdict":      verdict_binary(r),
                "p_value":      r.get("mcnemar_pval", ""),
                "test":         "McNemar test",
                "ci_lower":     ci_l,
                "ci_upper":     ci_u,
            })
        else:
            # continuous
            metric = "score"
            ci_l = r.get(f"{metric}_ci_lower", "")
            ci_u = r.get(f"{metric}_ci_upper", "")
            rows.append({
                **common,
                "metric":       metric,
                "origin_mean":  r.get(f"{metric}_orig_mean", ""),
                "tree_mean":    r.get(f"{metric}_tree_mean", ""),
                "margin":       compute_margin(ci_l, ci_u),
                "verdict":      verdict(r, metric, higher_is_better=True),
                "p_value":      r.get(f"{metric}_t_pval", ""),
                "test":         "paired t-test",
                "ci_lower":     ci_l,
                "ci_upper":     ci_u,
            })

    return pd.DataFrame(rows)

def main():
    parser = argparse.ArgumentParser(
        description="Simplify detailed beam-search results into verdict, ±margin, means, p-value, and test"
    )
    parser.add_argument("detailed_csv", help="Path to the detailed CSV")
    parser.add_argument("simplified_csv", help="Output path for the simplified CSV")
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

