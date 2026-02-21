"""
Compare Qlosure vs SABRE on Surface Code benchmarks.

Reads trace.json files from both methods, computes metrics, prints
a summary table, and optionally generates comparison plots (PDF).

Usage:
    python compare_surface_code.py --backend=ibm_kingston
    python compare_surface_code.py --backend=ibm_kingston --plot
    python compare_surface_code.py --backend=ibm_flamingo --distances 3 5 7 9
"""

import argparse
import json
import os
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "dynamic-qlosure"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.evaluation import compute_max_swaps_count, compute_quantum_depth, estimate_dynamic_circuit
from src.results_utils import RESULTS_ROOT, RESULTS_SUMMARY_ROOT, load_topology

LOOP_ITERATIONS = 10  # unroll iterations for metric estimation


def collect_results(results_dir: Path):
    """Walk a results tree and return {(distance, rounds, circ): [(swaps, depth, latency, error, time), ...]}"""
    data = defaultdict(list)
    if not results_dir.exists():
        return data

    for d_dir in sorted(results_dir.iterdir()):
        if not d_dir.is_dir() or not d_dir.name.startswith("d"):
            continue
        distance = int(d_dir.name[1:])

        for bench_dir in sorted(d_dir.iterdir()):
            if not bench_dir.is_dir():
                continue
            # surface_code_d{d}_r{rounds}
            parts = bench_dir.name.split("_")
            try:
                rounds = int(parts[-1][1:])
            except (ValueError, IndexError):
                continue

            for circ_dir in sorted(bench_dir.iterdir()):
                if not circ_dir.is_dir():
                    continue
                circ_name = circ_dir.name

                for seed_dir in sorted(circ_dir.iterdir()):
                    if not seed_dir.is_dir() or not seed_dir.name.startswith("SEED_"):
                        continue
                    trace_path = seed_dir / "trace.json"
                    time_path = seed_dir / "time.txt"
                    if not trace_path.exists():
                        continue

                    with open(trace_path) as f:
                        trace = json.load(f)

                    elapsed = 0.0
                    if time_path.exists():
                        with open(time_path) as f:
                            elapsed = float(f.read().strip())

                    swaps = compute_max_swaps_count(trace, loop_iterations=LOOP_ITERATIONS)
                    depth = compute_quantum_depth(trace, loop_iterations=LOOP_ITERATIONS)

                    data[(distance, rounds, circ_name)].append({
                        "swaps": swaps,
                        "depth": depth,
                        "time": elapsed,
                    })

    return data


def compute_summary(data):
    """Aggregate per (distance, rounds) with mean ± std across circuits and seeds."""
    agg = defaultdict(lambda: {"swaps": [], "depth": [], "time": []})
    for (d, r, _circ), entries in data.items():
        for e in entries:
            agg[(d, r)]["swaps"].append(e["swaps"])
            agg[(d, r)]["depth"].append(e["depth"])
            agg[(d, r)]["time"].append(e["time"])

    rows = []
    for (d, r), vals in sorted(agg.items()):
        rows.append({
            "distance": d,
            "rounds": r,
            "swaps_mean": np.mean(vals["swaps"]),
            "swaps_std": np.std(vals["swaps"]),
            "depth_mean": np.mean(vals["depth"]),
            "depth_std": np.std(vals["depth"]),
            "time_mean": np.mean(vals["time"]),
            "time_std": np.std(vals["time"]),
            "n_samples": len(vals["swaps"]),
        })
    return pd.DataFrame(rows)


def print_comparison(df_ours, df_sabre, backend):
    """Print a formatted comparison table."""
    if df_ours.empty and df_sabre.empty:
        print("No results found for either method.")
        return

    # Merge on (distance, rounds)
    df = pd.merge(df_ours, df_sabre, on=["distance", "rounds"], suffixes=("_ours", "_sabre"), how="outer")
    df = df.sort_values(["distance", "rounds"])

    print(f"\n{'='*100}")
    print(f"  Surface Code Routing Comparison: Qlosure vs SABRE on {backend}")
    print(f"{'='*100}")

    print(f"\n  {'d':>3s}  {'rounds':>6s}  {'swaps_ours':>12s}  {'swaps_sabre':>12s}  {'impr%':>7s}  "
          f"{'depth_ours':>12s}  {'depth_sabre':>12s}  {'impr%':>7s}  "
          f"{'time_ours':>10s}  {'time_sabre':>10s}")
    print(f"  {'-'*3}  {'-'*6}  {'-'*12}  {'-'*12}  {'-'*7}  {'-'*12}  {'-'*12}  {'-'*7}  {'-'*10}  {'-'*10}")

    for _, row in df.iterrows():
        d = int(row["distance"])
        r = int(row["rounds"])

        so = row.get("swaps_mean_ours", np.nan)
        ss = row.get("swaps_mean_sabre", np.nan)
        si = (ss - so) / ss * 100 if pd.notna(ss) and pd.notna(so) and ss != 0 else np.nan

        do_ = row.get("depth_mean_ours", np.nan)
        ds = row.get("depth_mean_sabre", np.nan)
        di = (ds - do_) / ds * 100 if pd.notna(ds) and pd.notna(do_) and ds != 0 else np.nan

        to = row.get("time_mean_ours", np.nan)
        ts = row.get("time_mean_sabre", np.nan)

        def fmt(v, w=12):
            return f"{v:>{w}.1f}" if pd.notna(v) else f"{'N/A':>{w}s}"

        def fmti(v):
            return f"{v:>+6.1f}%" if pd.notna(v) else f"{'N/A':>7s}"

        print(f"  {d:>3d}  {r:>6d}  {fmt(so)}  {fmt(ss)}  {fmti(si)}  "
              f"{fmt(do_)}  {fmt(ds)}  {fmti(di)}  "
              f"{fmt(to, 10)}  {fmt(ts, 10)}")

    # Overall averages
    print(f"\n  --- Overall Average Improvement ---")
    for metric in ["swaps", "depth"]:
        ours_col = f"{metric}_mean_ours"
        sabre_col = f"{metric}_mean_sabre"
        valid = df.dropna(subset=[ours_col, sabre_col])
        if not valid.empty:
            avg_ours = valid[ours_col].mean()
            avg_sabre = valid[sabre_col].mean()
            avg_impr = (avg_sabre - avg_ours) / avg_sabre * 100 if avg_sabre != 0 else 0
            print(f"    {metric:>8s}: Ours={avg_ours:>10.1f}  SABRE={avg_sabre:>10.1f}  Improvement={avg_impr:>+.1f}%")
    print()


def generate_plots(df_ours, df_sabre, backend, output_dir):
    """Generate comparison plots grouped by distance."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping plots.")
        return

    plt.rcParams.update({
        "font.size": 16, "axes.labelsize": 18, "xtick.labelsize": 14,
        "ytick.labelsize": 14, "legend.fontsize": 14, "lines.linewidth": 2.5,
        "lines.markersize": 8, "figure.dpi": 300, "pdf.fonttype": 42,
    })

    df = pd.merge(df_ours, df_sabre, on=["distance", "rounds"], suffixes=("_ours", "_sabre"), how="outer")
    df = df.sort_values(["distance", "rounds"])

    os.makedirs(output_dir, exist_ok=True)

    distances = sorted(df["distance"].unique())

    for metric, ylabel in [("swaps", "SWAP Count"), ("depth", "Quantum Depth")]:
        fig, ax = plt.subplots(figsize=(10, 6))

        for d in distances:
            sub = df[df["distance"] == d].sort_values("rounds")
            rounds = sub["rounds"]
            ours = sub[f"{metric}_mean_ours"]
            sabre = sub[f"{metric}_mean_sabre"]

            ax.plot(rounds, ours, "o-", label=f"Qlosure d={d}")
            ax.plot(rounds, sabre, "s--", label=f"SABRE d={d}", alpha=0.7)

        ax.set_xlabel("Syndrome Extraction Rounds")
        ax.set_ylabel(ylabel)
        ax.set_title(f"Surface Code {ylabel}: Qlosure vs SABRE ({backend})")
        ax.legend(loc="upper left", ncol=2)
        ax.grid(True, alpha=0.3)

        pdf_path = os.path.join(output_dir, f"surface_code_{metric}_{backend}.pdf")
        fig.tight_layout()
        fig.savefig(pdf_path)
        plt.close(fig)
        print(f"📄 Saved: {pdf_path}")

    # Improvement plot
    fig, ax = plt.subplots(figsize=(10, 6))
    for d in distances:
        sub = df[df["distance"] == d].sort_values("rounds")
        rounds = sub["rounds"]
        for metric, ls in [("swaps", "-"), ("depth", "--")]:
            ours = sub[f"{metric}_mean_ours"]
            sabre = sub[f"{metric}_mean_sabre"]
            impr = (sabre - ours) / sabre * 100
            ax.plot(rounds, impr, f"o{ls}", label=f"d={d} {metric}")

    ax.set_xlabel("Syndrome Extraction Rounds")
    ax.set_ylabel("Improvement over SABRE (%)")
    ax.set_title(f"Surface Code: Qlosure Improvement ({backend})")
    ax.axhline(y=0, color="gray", linestyle=":", linewidth=1)
    ax.legend(loc="best", ncol=2, fontsize=11)
    ax.grid(True, alpha=0.3)

    pdf_path = os.path.join(output_dir, f"surface_code_improvement_{backend}.pdf")
    fig.tight_layout()
    fig.savefig(pdf_path)
    plt.close(fig)
    print(f"📄 Saved: {pdf_path}")


def main():
    parser = argparse.ArgumentParser(description="Compare Qlosure vs SABRE on Surface Code")
    parser.add_argument("--backend", type=str, default="ibm_kingston",
                        help="Backend name (default: ibm_kingston)")
    parser.add_argument("--distances", type=int, nargs="+", default=None,
                        help="Filter to specific distances")
    parser.add_argument("--plot", action="store_true",
                        help="Generate PDF comparison plots")
    parser.add_argument("--save-csv", action="store_true",
                        help="Save summary CSV")
    args = parser.parse_args()

    ours_dir = RESULTS_ROOT / "qlosure" / "surface_code" / args.backend
    sabre_dir = RESULTS_ROOT / "sabre" / "surface_code" / args.backend

    print(f"Collecting Qlosure results from: {ours_dir}")
    data_ours = collect_results(ours_dir)
    print(f"  Found {sum(len(v) for v in data_ours.values())} trace files")

    print(f"Collecting SABRE results from: {sabre_dir}")
    data_sabre = collect_results(sabre_dir)
    print(f"  Found {sum(len(v) for v in data_sabre.values())} trace files")

    df_ours = compute_summary(data_ours)
    df_sabre = compute_summary(data_sabre)

    # Filter by distance if requested
    if args.distances:
        if not df_ours.empty:
            df_ours = df_ours[df_ours["distance"].isin(args.distances)]
        if not df_sabre.empty:
            df_sabre = df_sabre[df_sabre["distance"].isin(args.distances)]

    print_comparison(df_ours, df_sabre, args.backend)

    if args.save_csv:
        csv_dir = str(RESULTS_SUMMARY_ROOT / "surface_code")
        os.makedirs(csv_dir, exist_ok=True)
        csv_path = os.path.join(csv_dir, f"surface_code_{args.backend}_comparison.csv")
        merged = pd.merge(df_ours, df_sabre, on=["distance", "rounds"], suffixes=("_ours", "_sabre"), how="outer")
        merged.to_csv(csv_path, index=False)
        print(f"✅ CSV saved: {csv_path}")

    if args.plot:
        plot_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "..", "dynamic-qlosure",
            "paper-images", "US", "surface-code", args.backend
        )
        generate_plots(df_ours, df_sabre, args.backend, plot_dir)


if __name__ == "__main__":
    main()
