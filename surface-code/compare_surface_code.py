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
from qpu.src.load_backend import load_backend_data

LOOP_ITERATIONS = 10  # unroll iterations for metric estimation (default)


def collect_results(results_dir: Path, qubit_props: dict, loop_iterations: int = None):
    """Walk a results tree and return {(distance, rounds, circ): [(swaps, depth, latency, error, time), ...]}"""
    iters = loop_iterations if loop_iterations is not None else LOOP_ITERATIONS
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

                    swaps = compute_max_swaps_count(trace, loop_iterations=iters)
                    depth = compute_quantum_depth(trace, loop_iterations=iters)

                    # Compute latency and error via dynamic scheduling
                    dyn = estimate_dynamic_circuit(trace, qubit_props, loop_iterations=iters)
                    latency = dyn["max_time"]
                    error = dyn["max_error"]

                    data[(distance, rounds, circ_name)].append({
                        "swaps": swaps,
                        "depth": depth,
                        "latency": latency,
                        "error": error,
                        "time": elapsed,
                    })

    return data


METRICS = ["swaps", "depth", "latency", "error"]


def compute_summary(data):
    """Aggregate per (distance, rounds) with mean ± std across circuits and seeds."""
    agg = defaultdict(lambda: {m: [] for m in METRICS + ["time"]})
    for (d, r, _circ), entries in data.items():
        for e in entries:
            for m in METRICS + ["time"]:
                agg[(d, r)][m].append(e[m])

    rows = []
    for (d, r), vals in sorted(agg.items()):
        row = {"distance": d, "rounds": r, "n_samples": len(vals["swaps"])}
        for m in METRICS + ["time"]:
            row[f"{m}_mean"] = np.mean(vals[m])
            row[f"{m}_std"] = np.std(vals[m])
        rows.append(row)
    return pd.DataFrame(rows)


METRIC_LABELS = {
    "swaps": "SWAPs",
    "depth": "Depth",
    "latency": "Latency(µs)",
    "error": "Error",
}


def print_comparison(df_ours, df_sabre, backend):
    """Print a formatted comparison table."""
    if df_ours.empty and df_sabre.empty:
        print("No results found for either method.")
        return

    # Merge on (distance, rounds)
    df = pd.merge(df_ours, df_sabre, on=["distance", "rounds"], suffixes=("_ours", "_sabre"), how="outer")
    df = df.sort_values(["distance", "rounds"])

    print(f"\n{'='*140}")
    print(f"  Surface Code Routing Comparison: Qlosure vs SABRE on {backend}")
    print(f"{'='*140}")

    # Header
    hdr = f"  {'d':>3s}  {'rnds':>4s}"
    sep = f"  {'-'*3}  {'-'*4}"
    for m in METRICS:
        lbl = METRIC_LABELS[m]
        hdr += f"  {'Ours_'+lbl:>14s}  {'SABRE_'+lbl:>14s}  {'impr%':>7s}"
        sep += f"  {'-'*14}  {'-'*14}  {'-'*7}"
    hdr += f"  {'time_ours':>10s}  {'time_sabre':>10s}"
    sep += f"  {'-'*10}  {'-'*10}"
    print(f"\n{hdr}")
    print(sep)

    def fmt(v, w=14, prec=1):
        return f"{v:>{w}.{prec}f}" if pd.notna(v) else f"{'N/A':>{w}s}"

    def fmti(v):
        return f"{v:>+6.1f}%" if pd.notna(v) else f"{'N/A':>7s}"

    for _, row in df.iterrows():
        d = int(row["distance"])
        r = int(row["rounds"])
        line = f"  {d:>3d}  {r:>4d}"

        for m in METRICS:
            ours_v = row.get(f"{m}_mean_ours", np.nan)
            sabre_v = row.get(f"{m}_mean_sabre", np.nan)
            impr = ((sabre_v - ours_v) / sabre_v * 100
                    if pd.notna(sabre_v) and pd.notna(ours_v) and sabre_v != 0 else np.nan)
            prec = 4 if m == "error" else 1
            line += f"  {fmt(ours_v, 14, prec)}  {fmt(sabre_v, 14, prec)}  {fmti(impr)}"

        to = row.get("time_mean_ours", np.nan)
        ts = row.get("time_mean_sabre", np.nan)
        line += f"  {fmt(to, 10)}  {fmt(ts, 10)}"
        print(line)

    # Overall averages
    print(f"\n  --- Overall Average Improvement ---")
    for m in METRICS:
        ours_col = f"{m}_mean_ours"
        sabre_col = f"{m}_mean_sabre"
        valid = df.dropna(subset=[ours_col, sabre_col])
        if not valid.empty:
            avg_ours = valid[ours_col].mean()
            avg_sabre = valid[sabre_col].mean()
            avg_impr = (avg_sabre - avg_ours) / avg_sabre * 100 if avg_sabre != 0 else 0
            print(f"    {METRIC_LABELS[m]:>14s}: Ours={avg_ours:>12.2f}  SABRE={avg_sabre:>12.2f}  Improvement={avg_impr:>+.1f}%")
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

    metric_plots = [
        ("swaps", "SWAP Count"),
        ("depth", "Quantum Depth"),
        ("latency", "Latency (µs)"),
        ("error", "Accumulated Error"),
    ]

    for metric, ylabel in metric_plots:
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
        print(f"  Saved: {pdf_path}")

    # Improvement plot (all 4 metrics)
    fig, ax = plt.subplots(figsize=(12, 7))
    linestyles = {
        "swaps": "-", "depth": "--", "latency": "-.", "error": ":",
    }
    for d in distances:
        sub = df[df["distance"] == d].sort_values("rounds")
        rounds = sub["rounds"]
        for metric in METRICS:
            ours = sub[f"{metric}_mean_ours"]
            sabre = sub[f"{metric}_mean_sabre"]
            impr = (sabre - ours) / sabre * 100
            ax.plot(rounds, impr, f"o{linestyles[metric]}",
                    label=f"d={d} {METRIC_LABELS[metric]}")

    ax.set_xlabel("Syndrome Extraction Rounds")
    ax.set_ylabel("Improvement over SABRE (%)")
    ax.set_title(f"Surface Code: Qlosure Improvement ({backend})")
    ax.axhline(y=0, color="gray", linestyle=":", linewidth=1)
    ax.legend(loc="best", ncol=2, fontsize=10)
    ax.grid(True, alpha=0.3)

    pdf_path = os.path.join(output_dir, f"surface_code_improvement_{backend}.pdf")
    fig.tight_layout()
    fig.savefig(pdf_path)
    plt.close(fig)
    print(f"  Saved: {pdf_path}")


def main():
    parser = argparse.ArgumentParser(description="Compare Qlosure vs SABRE on Surface Code")
    parser.add_argument("--backend", type=str, default="ibm_kingston",
                        help="Backend name (default: ibm_kingston)")
    parser.add_argument("--distances", type=int, nargs="+", default=None,
                        help="Filter to specific distances")
    parser.add_argument("--results-tag", type=str, default="surface_code",
                        help="Results sub-folder tag (default: surface_code)")
    parser.add_argument("--plot", action="store_true",
                        help="Generate PDF comparison plots")
    parser.add_argument("--save-csv", action="store_true",
                        help="Save summary CSV")
    parser.add_argument("--loop-iterations", type=int, default=None,
                        help="Loop unroll iterations for metrics (default: 10)")
    args = parser.parse_args()

    # Load backend qubit properties for latency / error estimation
    print(f"Loading backend: {args.backend}")
    backend_data = load_backend_data(args.backend)
    qubit_props = backend_data.get("qubits", {})

    ours_dir = RESULTS_ROOT / "qlosure" / args.results_tag / args.backend
    sabre_dir = RESULTS_ROOT / "sabre" / args.results_tag / args.backend

    li = args.loop_iterations
    if li is not None:
        print(f"Using loop_iterations = {li}")

    print(f"Collecting Qlosure results from: {ours_dir}")
    data_ours = collect_results(ours_dir, qubit_props, loop_iterations=li)
    print(f"  Found {sum(len(v) for v in data_ours.values())} trace files")

    print(f"Collecting SABRE results from: {sabre_dir}")
    data_sabre = collect_results(sabre_dir, qubit_props, loop_iterations=li)
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
        csv_dir = str(RESULTS_SUMMARY_ROOT / args.results_tag)
        os.makedirs(csv_dir, exist_ok=True)
        csv_path = os.path.join(csv_dir, f"{args.results_tag}_{args.backend}_comparison.csv")
        merged = pd.merge(df_ours, df_sabre, on=["distance", "rounds"], suffixes=("_ours", "_sabre"), how="outer")
        merged.to_csv(csv_path, index=False)
        print(f"✅ CSV saved: {csv_path}")

    if args.plot:
        plot_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "..", "dynamic-qlosure",
            "paper-images", "US", args.results_tag, args.backend
        )
        generate_plots(df_ours, df_sabre, args.backend, plot_dir)


if __name__ == "__main__":
    main()
