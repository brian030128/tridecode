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
    try:
        low = float(ci_low); high = float(ci_high)
        return (high - low) / 2
    except Exception:
        return ""

def _has_mean(row, metric, prefix):
    return not pd.isna(row.get(f"{metric}_{prefix}_mean", float("nan")))

def _mode_and_prefix(row, metric):
    if _has_mean(row, metric, "orig"):
        return "beam", "orig"
    if _has_mean(row, metric, "sample"):
        return "topk", "sample"
    if not pd.isna(row.get(f"{metric}_mean", float("nan"))):
        return "unknown", ""
    return "unknown", ""

def _ci_bounds(row, metric, prefix):
    ci_l = row.get(f"{metric}_{prefix}_ci_lower", None) if prefix else None
    ci_u = row.get(f"{metric}_{prefix}_ci_upper", None) if prefix else None
    if pd.isna(ci_l) or ci_l is None:
        ci_l = row.get(f"{metric}_ci_lower", "")
    if pd.isna(ci_u) or ci_u is None:
        ci_u = row.get(f"{metric}_ci_upper", "")
    return ci_l, ci_u

def _append_metric(rows, common, r, metric, higher_is_better):
    mode, prefix = _mode_and_prefix(r, metric)
    ci_l, ci_u = _ci_bounds(r, metric, prefix)
    p_val = r.get(f"{metric}_t_pval", "")
    rows.append({
        **common,
        "mode":        mode,                        # 'beam' or 'topk' (or 'unknown')
        "metric":      metric,
        "origin_mean": r.get(f"{metric}_{prefix}_mean", r.get(f"{metric}_mean", "")),
        "tree_mean":   r.get(f"{metric}_tree_mean", ""),
        "margin":      compute_margin(ci_l, ci_u),
        "verdict":     verdict(r, metric, higher_is_better=higher_is_better),
        "p_value":     p_val if not pd.isna(p_val) else "",
        "test":        "paired t-test" if (not pd.isna(p_val) and p_val != "") else "n/a",
        "ci_lower":    ci_l,
        "ci_upper":    ci_u,
        "is_topk":     (mode == "topk"),
        "prefix_used": prefix or "n/a",
    })

def simplify(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in df.iterrows():
        common = {
            "model":   r.get("model", ""),
            "dataset": r.get("dataset", ""),
            "beam":    r.get("beam", ""),
            "samples": r.get("samples", ""),
        }
        # 1) mem_per_token (lower is better)
        _append_metric(rows, common, r, "mem_per_token", higher_is_better=False)
        # 2) tok_per_sec (higher is better)
        _append_metric(rows, common, r, "tok_per_sec", higher_is_better=True)

        # 3) score
        if not pd.isna(r.get("mcnemar_pval", float("nan"))):
            ci_l = r.get("mcnemar_or_ci_low", "")
            ci_u = r.get("mcnemar_or_ci_high", "")
            mode, prefix = _mode_and_prefix(r, "score")
            rows.append({
                **common,
                "mode":        mode,
                "metric":      "score",
                "origin_mean": r.get(f"score_{prefix}_mean", r.get("score_mean", "")),
                "tree_mean":   r.get("score_tree_mean", ""),
                "margin":      compute_margin(ci_l, ci_u),
                "verdict":     verdict_binary(r),
                "p_value":     r.get("mcnemar_pval", ""),
                "test":        "McNemar test",
                "ci_lower":    ci_l,
                "ci_upper":    ci_u,
                "is_topk":     (mode == "topk"),
                "prefix_used": prefix or "n/a",
            })
        else:
            _append_metric(rows, common, r, "score", higher_is_better=True)

        # 4) Optional: include if present
        if any(col.startswith("input_kv_memory") for col in df.columns):
            _append_metric(rows, common, r, "input_kv_memory", higher_is_better=False)

    out = pd.DataFrame(rows)
    order = ["model","dataset","beam","samples","mode","metric","origin_mean","tree_mean",
             "margin","verdict","p_value","test","ci_lower","ci_upper","is_topk","prefix_used"]
    return out.reindex(columns=order)

def main():
    parser = argparse.ArgumentParser(
        description="Simplify results (beam & top-k). Produces unified CSV, plus split files if applicable."
    )
    parser.add_argument("--detailed_csv", help="Path to the detailed CSV")
    parser.add_argument("--simplified_csv", help="Output path for the unified simplified CSV")
    args = parser.parse_args()

    df = pd.read_csv(args.detailed_csv)
    simple = simplify(df)
    simple.to_csv(args.simplified_csv, index=False)
    print(f"[all] {args.simplified_csv}")

    # Optional split
    stem = args.simplified_csv.rsplit(".", 1)[0]
    beam_df = simple[simple["mode"] == "beam"]
    topk_df = simple[simple["mode"] == "topk"]
    if not beam_df.empty:
        p = f"{stem}_beam.csv"; beam_df.to_csv(p, index=False); print(f"[beam] {p}")
    if not topk_df.empty:
        p = f"{stem}_topk.csv"; topk_df.to_csv(p, index=False); print(f"[topk] {p}")


if __name__ == "__main__":
    main()

    """
    Example usage:
    python -m analysis.statistic_testing.simplify_stats \
        --detailed_csv analysis/results/statistic_testing_results.csv \
        --simplified_csv analysis/results/simplified_results.csv
    """

