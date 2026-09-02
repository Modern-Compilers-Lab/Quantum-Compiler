"""Table 4 - routing results on chiplet-based QPUs.

Absolute cells are column means over the leaf depths. "Overall imp." takes the
improvement between the two column means per benchmark, then averages over the
benchmarks of that topology.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import paper_values as pv
from common import (METRICS, banner, load_pairwise_summary, saved, table_dir,
                    warn_partial)
from csv_sources import resolve

BENCHMARKS = [("heavy_hexagon", 81), ("heavy_hexagon", 121), ("ibm_flamingo", 256)]
DISPLAY = {"heavy_hexagon": "Heavy H.", "ibm_flamingo": "IBM F."}
UNITS = {"swaps": "x10^3", "depth": "x10^3", "latency": "x10^3 us", "error": ""}
HINT = "python artifact/generate.py chiplet"


def csv_rel(topology, nq, iters):
    if topology == "heavy_hexagon":
        return f"{topology}_{nq}qbt_{iters}iter_metrics.csv"
    return f"main/{topology}_{nq}qbt_{iters}iter_metrics.csv"


def build(iters):
    rows, origins = [], set()
    for topology, nq in BENCHMARKS:
        path, origin = resolve(csv_rel(topology, nq, iters), HINT)
        origins.add(origin)
        df = load_pairwise_summary(path)
        row = {"topology": topology, "qubits": nq, "n_leaf_depths": len(df)}
        for m in METRICS:
            o = float(df[f"{m}_mean_ours"].mean())
            s = float(df[f"{m}_mean_sabre"].mean())
            row[f"{m}_ours"], row[f"{m}_sabre"] = o, s
            row[f"{m}_impr"] = (s - o) / s * 100.0 if s else float("nan")
        rows.append(row)
    return pd.DataFrame(rows), origins


def overall(df, topology):
    sub = df[df.topology == topology]
    return {m: float(np.mean(sub[f"{m}_impr"])) for m in METRICS}


def render(df):
    scale = pv.TABLE4_SCALE
    print(f"  {'Topology':<11s}{'Qubits':>7s}", end="")
    for m in METRICS:
        print(f"  |{(m.upper() + ' ' + UNITS[m]).center(21)}", end="")
    print()
    print(f"  {'':<11s}{'':>7s}", end="")
    for _ in METRICS:
        print(f"  |{'Ours':>10s}{'Sabre':>11s}", end="")
    print()
    print("  " + "-" * 110)
    for topology in df.topology.unique():
        for _, r in df[df.topology == topology].iterrows():
            print(f"  {DISPLAY.get(topology, topology):<11s}{int(r.qubits):>7d}", end="")
            for m in METRICS:
                print(f"  |{r[f'{m}_ours'] / scale[m]:>10.2f}"
                      f"{r[f'{m}_sabre'] / scale[m]:>11.2f}", end="")
            print()
        ov = overall(df, topology)
        print(f"  {'Overall imp.':<18s}", end="")
        for m in METRICS:
            print(f"  |{f'{ov[m]:+.1f}%':>21s}", end="")
        print()
        print("  " + "-" * 110)


def run(args):
    banner("Table 4 - Chiplet-based QPUs (Sec. 7.3)",
           "+P% improvement, -P% deliberate degradation")
    df, origins = build(args.loop_iterations)
    print(f"  source: {'/'.join(sorted(origins))} CSVs\n")
    render(df)
    print("  Flamingo trades SWAPs and depth for latency and error "
          "to avoid the 7.4x inter-chip penalty.")
    warn_partial(df, "topology", "qubits")

    out = table_dir("tab04") / "table04_chiplet.csv"
    df.to_csv(out, index=False)
    print()
    saved(out)

    return None
