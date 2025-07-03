import csv
import json
import ast
from pathlib import Path
import matplotlib.pyplot as plt


def _extend_until_limit(dest: list, new: list, limit: int) -> bool:
    remaining = limit - len(dest)
    if remaining <= 0:
        return True
    dest.extend(new[:remaining])
    return len(dest) >= limit


def load_times(path: Path, limit: int) -> list:
    times: list[float] = []
    suffix = path.suffix.lower()

    if suffix == ".csv":
        with path.open(newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if "time_took" not in row:
                    raise KeyError(
                        f"'time_took' column missing in {path.name}"
                    )
                ts = ast.literal_eval(row["time_took"])
                if _extend_until_limit(times, ts, limit):
                    break

    elif suffix in {".jsonl", ".json"}:
        with path.open() as f:
            for ln in f:
                if not ln.strip():
                    continue  # skip blank lines
                obj = json.loads(ln)
                ts = (
                    obj.get("time_metric")
                    or obj.get("time_took")
                    or []
                )
                if _extend_until_limit(times, ts, limit):
                    break
    else:
        raise ValueError(
            f"Unsupported file extension '{suffix}'. "
            "Use .csv, .jsonl, or .json."
        )

    return times


def main() -> None:
    no_gc_path = Path(
        "reproduction/final_out/ablation_nogc_mistral_humaneval_3.csv"
    )
    with_gc_path = Path(
        "reproduction/final_out/ablation_gc_mistral_humaneval_3.csv"
    )  # can be .csv, .jsonl, or .json
    token_limit = 400

    no_gc_times = load_times(no_gc_path, token_limit)
    with_gc_times = load_times(with_gc_path, token_limit)

    if len(no_gc_times) != len(with_gc_times):
        raise ValueError("Mismatched length after loading")

    diffs_no_gc = [
        j - i for i, j in zip(no_gc_times[:-1], no_gc_times[1:])
    ]
    diffs_with_gc = [
        j - i for i, j in zip(with_gc_times[:-1], with_gc_times[1:])
    ]
    ratio = [
        b / a if a != 0 else 0.0
        for a, b in zip(diffs_no_gc, diffs_with_gc)
    ]

    out_csv = Path("analysis/results/token_speed_mistral.csv")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "no_gc", "with_gc", "ratio"])
        for i, (a, b, r) in enumerate(zip(diffs_no_gc, diffs_with_gc, ratio), 1):
            writer.writerow([i, a, b, r])
    print(f"Saved speed data to {out_csv}")

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(14, 4))

    steps = range(1, len(diffs_no_gc) + 1)
    ax0.plot(steps, diffs_no_gc, label="trie w/o GC")
    ax0.plot(steps, diffs_with_gc, label="trie + GC")
    ax0.set_xlabel("Token index")
    ax0.set_ylabel("Time per token (s)")
    ax0.set_title("Token generation time")
    ax0.legend(frameon=False)
    ax0.grid(alpha=0.3)

    ax1.plot(steps, ratio)
    ax1.set_xlabel("Token index")
    ax1.set_ylabel("GC / no-GC ratio")
    ax1.set_title("Relative Generation Time")
    ax1.grid(alpha=0.3)

    fig.tight_layout()
    out_fig = Path("analysis/results/figs/token_speed_ratio_mistral.png")
    fig.savefig(out_fig, dpi=300)
    plt.close(fig)
    print(f"Figure saved to {out_fig}")


if __name__ == "__main__":
    main()

"""
Example usage:
python -m analysis.token_speed
"""
