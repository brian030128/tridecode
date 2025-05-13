import sys
import pandas as pd
import argparse
ALPHA = 0.05 # significance level

def verdict(row, metric, higher_is_better=True):
    diff   = row[f"{metric}_diff_mean"]
    pval   = row[f"{metric}_t_pval"]
    if pd.isna(diff) or pd.isna(pval):
        return "n/a"               # metric not present
    if pval >= ALPHA:
        return "no_sig"
    if higher_is_better:
        return "better" if diff > 0 else "worse"
    else:                          # lower is better
        return "better" if diff < 0 else "worse"

def verdict_binary(row):
    if pd.isna(row.get("mcnemar_pval")):
        return "n/a"
    if row["mcnemar_pval"] >= ALPHA:
        return "no_sig"
    return "better" if row["mcnemar_c"] > row["mcnemar_b"] else "worse"

def main(in_csv, out_csv):
    df = pd.read_csv(in_csv)

    rows = []
    for _, r in df.iterrows():
        base = {
            "model"  : r["model"],
            "dataset": r["dataset"],
            "beam"   : r["beam"],
            "samples": r["samples"],
        }

        for metric, higher_is_better in [
            ("tok_per_sec",  True),   # faster is better
            ("input_kv_memory", False)  # less memory is better
        ]:
            if f"{metric}_diff_mean" in r:
                rows.append({
                    **base,
                    "metric" : metric,
                    "verdict": verdict(r, metric, higher_is_better),
                    "ci_low" : r[f"{metric}_ci_lower"],
                    "ci_high": r[f"{metric}_ci_upper"],
                })

        if "score_t_pval" in r:                    
            rows.append({
                **base,
                "metric" : "score",
                "verdict": verdict(r, "score", higher_is_better=True),
                "ci_low" : r["score_ci_lower"],
                "ci_high": r["score_ci_upper"],
            })
        elif "mcnemar_pval" in r:                
            rows.append({
                **base,
                "metric" : "score (binary)",
                "verdict": verdict_binary(r),
                "ci_low" : r["mcnemar_rd_ci_low"],
                "ci_high": r["mcnemar_rd_ci_high"],
            })

    out = pd.DataFrame(rows)
    out = out[["model", "dataset", "beam", "metric", "verdict", "ci_low", "ci_high"]]
    out.to_csv(out_csv, index=False)
    print(f"Simplified results written to {out_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simplify and summarize results.")
    parser.add_argument("--source", required=True)
    parser.add_argument("--output_csv", required=True)
    args = parser.parse_args()

    main(args.source, args.output_csv)

    """
    Example usage:
    python -m reproduction.statistic_testing.simplify_stats --source reproduction/statistic_testing_results.csv --output_csv reproduction/simplified_results.csv
    """
