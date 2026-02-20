"""
Print a summary of routing results for heavy_hexagon topology.

Usage:
    python print_summary.py                          # all qubit counts
    python print_summary.py 54                       # single qubit count
    python print_summary.py 54 81                    # specific qubit counts
    python print_summary.py --backend ibm_brisbane_old 54 81 121
"""

import argparse
import os
import sys
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from src.results_utils import RESULTS_SUMMARY_ROOT


def print_summary(backend: str, qubit_counts: list[int], loop_iterations: int, template: str):
    for nq in qubit_counts:
        csv_path = RESULTS_SUMMARY_ROOT / f"{backend}_{nq}qbt_{loop_iterations}iter_metrics.csv"
        if not csv_path.exists():
            print(f"\n[SKIP] CSV not found: {csv_path}")
            continue

        df = pd.read_csv(csv_path)
        n_leaf_depths = len(df)

        print(f"\n{'=' * 72}")
        print(f"  {backend} | {nq} qubits | {loop_iterations} loop iters | template={template}")
        print(f"  CSV: {csv_path.name}  ({n_leaf_depths} leaf-depths)")
        print(f"{'=' * 72}")
        print(f"  {'Metric':>10s}  {'Ours':>12s}  {'SABRE':>12s}  {'Improvement':>12s}")
        print(f"  {'-'*10}  {'-'*12}  {'-'*12}  {'-'*12}")

        for metric, unit in [("swaps", ""), ("depth", ""), ("latency", " us"), ("error", "")]:
            ours = df[f"{metric}_mean_ours"].mean()
            sabre = df[f"{metric}_mean_sabre"].mean()
            if sabre != 0:
                impr = (sabre - ours) / sabre * 100
            else:
                impr = 0.0
            print(f"  {metric:>10s}  {ours:>12.1f}  {sabre:>12.1f}  {impr:>+11.1f}%")

        # Per leaf-depth breakdown
        print(f"\n  {'leaf_depth':>10s}  {'swaps_ours':>10s}  {'swaps_sabre':>11s}  {'depth_ours':>10s}  {'depth_sabre':>11s}  {'impr_swaps':>10s}  {'impr_depth':>10s}")
        print(f"  {'-'*10}  {'-'*10}  {'-'*11}  {'-'*10}  {'-'*11}  {'-'*10}  {'-'*10}")
        for _, row in df.iterrows():
            ld = int(row["leaf_depth"])
            so = row["swaps_mean_ours"]
            ss = row["swaps_mean_sabre"]
            do = row["depth_mean_ours"]
            ds = row["depth_mean_sabre"]
            si = (ss - so) / ss * 100 if ss else 0
            di = (ds - do) / ds * 100 if ds else 0
            print(f"  {ld:>10d}  {so:>10.1f}  {ss:>11.1f}  {do:>10.1f}  {ds:>11.1f}  {si:>+9.1f}%  {di:>+9.1f}%")

    print()


def main():
    parser = argparse.ArgumentParser(description="Print routing results summary")
    parser.add_argument("qubits", nargs="*", type=int, default=[54, 81, 121],
                        help="Qubit counts to show (default: 54 81 121)")
    parser.add_argument("--backend", type=str, default="heavy_hexagon",
                        help="Backend/topology name (default: heavy_hexagon)")
    parser.add_argument("--iterations", type=int, default=10,
                        help="Number of loop iterations (default: 10)")
    parser.add_argument("--template", type=str, default="nest0",
                        help="Template name (default: nest0)")
    args = parser.parse_args()

    print_summary(args.backend, args.qubits, args.iterations, args.template)


if __name__ == "__main__":
    main()
