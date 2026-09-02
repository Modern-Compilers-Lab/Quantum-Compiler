"""Table 3 - average relative improvement (%) of DynamiQ over Qiskit Sabre.

Improvement is computed per leaf depth as (Sabre - Ours) / Sabre, then averaged
over the nine leaf depths.
"""

from __future__ import annotations

import pandas as pd

from common import (METRICS, BACKEND_LABELS, banner, load_pairwise_summary,
                    saved, table_dir, warn_partial)
from csv_sources import resolve

BACKENDS = ["ibm_brisbane_old", "ibm_kingston"]
WIDTHS = [54, 81, 121]
HINT = "python artifact/generate.py main"


def improvement(df, metric):
    s = df[f"{metric}_mean_sabre"].astype(float)
    o = df[f"{metric}_mean_ours"].astype(float)
    return float(((s - o) / s * 100.0).replace([float("inf"), float("-inf")], pd.NA).mean())


def build(backends, widths, iters):
    rows, origins = [], set()
    for backend in backends:
        for nq in widths:
            path, origin = resolve(f"main/{backend}_{nq}qbt_{iters}iter_metrics.csv", HINT)
            origins.add(origin)
            df = load_pairwise_summary(path)
            rows.append({"backend": backend, "qubits": nq, "n_leaf_depths": len(df),
                         **{m: improvement(df, m) for m in METRICS}})
    return pd.DataFrame(rows), origins


def render(df, backends, widths):
    print(f"  {'Bench.':<8s}", end="")
    for backend in backends:
        print(f"  |{BACKEND_LABELS.get(backend, backend).center(32)}", end="")
    print()
    print(f"  {'':<8s}", end="")
    for _ in backends:
        print("  |" + "".join(f"{k:>8s}" for k in ("S", "D", "L", "E")), end="")
    print()
    print("  " + "-" * (8 + 34 * len(backends)))
    for nq in widths:
        print(f"  {'dQ-' + str(nq):<8s}", end="")
        for backend in backends:
            r = df[(df.backend == backend) & (df.qubits == nq)].iloc[0]
            print("  |" + "".join(f"{r[m]:>8.2f}" for m in METRICS), end="")
        print()


def run(args):
    banner("Table 3 - DynamiQ vs Qiskit Sabre on d-QUEKO (Sec. 7.2)",
           "higher is better")
    df, origins = build(BACKENDS, WIDTHS, args.loop_iterations)
    print(f"  source: {'/'.join(sorted(origins))} CSVs\n")
    render(df, BACKENDS, WIDTHS)
    print("\n  S=SWAPs D=Depth L=Latency E=Error   "
          f"loops x{args.loop_iterations}")
    warn_partial(df, "backend", "qubits")

    out = table_dir("tab03") / "table03_main_improvements.csv"
    df.to_csv(out, index=False)
    print()
    saved(out)

    return None
