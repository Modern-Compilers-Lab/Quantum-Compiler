"""
Print a comparison summary from the deps-ablation results.

Reads results/deps-ablation/{variant}/{folder}/{circuit}/metrics.json
and prints aggregate statistics across all 4 metrics + time,
with variant 'default' as the baseline.
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

RESULTS_DIR = Path(__file__).parent / "results" / "deps-ablation"
VARIANTS = ["default", "unrolled", "no_deps"]
METRICS = ["swaps", "depth", "latency", "error_rate", "time_s"]
METRIC_LABELS = {
    "swaps":      "Swaps",
    "depth":      "Depth",
    "latency":    "Latency (µs)",
    "error_rate": "Avg Error",
    "time_s":     "Time (s)",
}


def load_all_metrics():
    """
    Returns {variant: {leaf_depth_folder: {circuit: metrics_dict}}}
    """
    data = {}
    for vname in VARIANTS:
        vdir = RESULTS_DIR / vname
        if not vdir.exists():
            print(f"⚠️  No results for variant '{vname}' at {vdir}")
            data[vname] = {}
            continue
        vdata = {}
        for folder in sorted(vdir.iterdir()):
            if not folder.is_dir():
                continue
            for circ_dir in sorted(folder.iterdir()):
                if not circ_dir.is_dir():
                    continue
                mfile = circ_dir / "metrics.json"
                if mfile.exists():
                    with open(mfile) as f:
                        metrics = json.load(f)
                    key = f"{folder.name}/{circ_dir.name}"
                    vdata[key] = metrics
        data[vname] = vdata
    return data


def avg(vals):
    nums = [v for v in vals if v is not None]
    return sum(nums) / len(nums) if nums else float("nan")


def pct_change(baseline, other):
    if baseline == 0:
        return 0.0
    return (other - baseline) / baseline * 100


def fmt_val(metric, val):
    if val is None:
        return "ERR"
    if metric == "error_rate":
        return f"{val:.6f}"
    elif metric == "latency":
        return f"{val:.2f}"
    elif metric == "time_s":
        return f"{val:.3f}"
    else:
        return f"{val:.1f}"


def main():
    data = load_all_metrics()

    # Check we have results
    if not data.get("default"):
        print("❌ No 'default' results found. Run run_comparison.py first.")
        sys.exit(1)

    # Align circuits: only include circuits that exist across all variants
    all_circuits = set(data["default"].keys())
    for v in VARIANTS:
        all_circuits &= set(data[v].keys())
    all_circuits = sorted(all_circuits)

    if not all_circuits:
        print("❌ No common circuits found across all variants.")
        sys.exit(1)

    print(f"\n{'='*90}")
    print(f"  DEPS-ABLATION COMPARISON  —  {len(all_circuits)} circuits")
    print(f"{'='*90}")

    # ---- Aggregate summary ----
    avgs = {}
    for v in VARIANTS:
        avgs[v] = {}
        for m in METRICS:
            avgs[v][m] = avg([data[v][c][m] for c in all_circuits])

    print(f"\n{'AGGREGATE AVERAGES':^90}")
    col_w = 16
    header = f"{'Metric':<16}"
    for v in VARIANTS:
        header += f"  {v:>{col_w}}"
    header += f"  {'Δ unrolled':>{col_w}}  {'Δ no_deps':>{col_w}}"
    print(header)
    print("-" * len(header))

    for m in METRICS:
        row = f"{METRIC_LABELS[m]:<16}"
        baseline = avgs["default"][m]
        for v in VARIANTS:
            row += f"  {fmt_val(m, avgs[v][m]):>{col_w}}"
        for comp in ["unrolled", "no_deps"]:
            delta = pct_change(baseline, avgs[comp][m])
            sign = "-" if delta >= 0 else "+"
            row += f"  {sign}{delta:>{col_w-1}.2f}%"
        print(row)

    # ---- Per leaf-depth breakdown ----
    print(f"\n{'='*90}")
    print(f"  PER LEAF-DEPTH BREAKDOWN")
    print(f"{'='*90}")

    # Group circuits by leaf-depth folder
    depth_groups = defaultdict(list)
    for c in all_circuits:
        folder = c.split("/")[0]
        depth_groups[folder].append(c)

    for folder in sorted(depth_groups.keys()):
        circuits = depth_groups[folder]
        print(f"\n  📂 {folder}  ({len(circuits)} circuits)")

        ld_avgs = {}
        for v in VARIANTS:
            ld_avgs[v] = {}
            for m in METRICS:
                ld_avgs[v][m] = avg([data[v][c][m] for c in circuits])

        header2 = f"  {'Metric':<16}"
        for v in VARIANTS:
            header2 += f"  {v:>{col_w}}"
        header2 += f"  {'Δ unrolled':>{col_w}}  {'Δ no_deps':>{col_w}}"
        print(header2)
        print("  " + "-" * (len(header2) - 2))

        for m in METRICS:
            row = f"  {METRIC_LABELS[m]:<16}"
            baseline = ld_avgs["default"][m]
            for v in VARIANTS:
                row += f"  {fmt_val(m, ld_avgs[v][m]):>{col_w}}"
            for comp in ["unrolled", "no_deps"]:
                delta = pct_change(baseline, ld_avgs[comp][m])
                sign = "+" if delta >= 0 else ""
                row += f"  {sign}{delta:>{col_w-1}.2f}%"
            print(row)

    # ---- Per-circuit detail (compact) ----
    print(f"\n{'='*90}")
    print(f"  PER-CIRCUIT DETAIL  (swaps only, Δ vs default)")
    print(f"{'='*90}")

    header3 = f"  {'Circuit':<55}  {'default':>8}  {'unrolled':>8}  {'Δ':>7}  {'no_deps':>8}  {'Δ':>7}"
    print(header3)
    print("  " + "-" * (len(header3) - 2))

    for c in all_circuits:
        circ_label = c.split("/")[-1]
        folder_label = c.split("/")[0].split("leaf-depth-")[-1] if "leaf-depth-" in c else ""
        label = f"LD-{folder_label}/{circ_label}"

        s_def = data["default"][c]["swaps"]
        s_unr = data["unrolled"][c]["swaps"]
        s_nod = data["no_deps"][c]["swaps"]

        d1 = pct_change(s_def, s_unr) if s_def else 0
        d2 = pct_change(s_def, s_nod) if s_def else 0

        sign1 = "+" if d1 >= 0 else ""
        sign2 = "+" if d2 >= 0 else ""

        print(f"  {label:<55}  {s_def:>8.0f}  {s_unr:>8.0f}  {sign1}{d1:>5.1f}%  {s_nod:>8.0f}  {sign2}{d2:>5.1f}%")

    print()


if __name__ == "__main__":
    main()
