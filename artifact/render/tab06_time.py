"""Table 6 - average DynamiQ mapping time (s) per leaf-depth category.

Per leaf depth, average the seeds; per category, average the three leaf depths.
Wall-clock, so machine dependent.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from common import (SUMMARY_ROOT, BACKEND_LABELS, banner, saved, table_dir)
import csv_sources
from csv_sources import GENERATED_ROOT

BACKENDS = ["ibm_kingston", "ibm_brisbane_old"]
WIDTHS = [54, 81, 121]
SEEDS = [3, 21, 42, 63, 84, 105, 126, 147, 168, 189]
CATEGORIES = {"Small (10-30)": [10, 20, 30],
              "Med. (40-60)": [40, 50, 60],
              "Large (70-90)": [70, 80, 90]}

GENERATED_CSV = GENERATED_ROOT / "time" / "mapping_time.csv"
COMMITTED_ROOT = SUMMARY_ROOT / "time" / "qroqi" / "one_loop"


def from_generated():
    df = pd.read_csv(GENERATED_CSV)
    out = {}
    for (backend, nq), sub in df.groupby(["backend", "qubits"]):
        out[(backend, int(nq))] = sub.groupby("leaf_depth")["time_s"].mean().to_dict()
    return out


def from_committed():
    out = {}
    for backend in BACKENDS:
        for nq in WIDTHS:
            base = COMMITTED_ROOT / backend / f"{nq}qbt"
            by_depth = {}
            for d in range(10, 91, 10):
                cfg = base / f"queko-{nq:03d}qbt_nest_00_nodes001_leaf-depth-{d}"
                vals = [float((cfg / f"SEED_{s}" / "time.txt").read_text().strip())
                        for s in SEEDS if (cfg / f"SEED_{s}" / "time.txt").exists()]
                if vals:
                    by_depth[d] = float(np.mean(vals))
            if by_depth:
                out[(backend, nq)] = by_depth
    return out


def categorise(by_depth):
    return {label: float(np.mean([by_depth[d] for d in depths if d in by_depth]))
            if any(d in by_depth for d in depths) else float("nan")
            for label, depths in CATEGORIES.items()}


def run(args):
    # timings come from time.txt, not a summary CSV, but --source is honoured
    if args.source == "generated":
        if not GENERATED_CSV.exists():
            raise SystemExit(
                f"no generated timings at {GENERATED_CSV}\n\n"
                "  Generate it with:\n      python artifact/generate.py timing")
        data, origin = from_generated(), "generated"
    elif args.source == "committed":
        data, origin = from_committed(), "committed"
    elif GENERATED_CSV.exists():
        data, origin = from_generated(), "generated"
    else:
        data, origin = from_committed(), "committed"
    if not data:
        raise SystemExit("no timing data; run: python artifact/generate.py timing")
    csv_sources._record("time/mapping_time.csv", origin)

    banner("Table 6 - DynamiQ mapping time (Sec. 7.5)",
           "recorded i7-10750H timings" if origin == "committed"
           else "measured on this machine by generate.py")
    print(f"  source: {origin}\n")

    rows = [{"backend": b, "qubits": q, **categorise(v)} for (b, q), v in sorted(data.items())]
    df = pd.DataFrame(rows)

    print(f"  {'Qubits':<8s}", end="")
    for backend in BACKENDS:
        print(f"  |{BACKEND_LABELS.get(backend, backend).center(32)}", end="")
    print()
    print(f"  {'':<8s}", end="")
    for _ in BACKENDS:
        print("  |" + "".join(f"{c.split(' ')[0]:>10s}" for c in CATEGORIES).rjust(32), end="")
    print()
    print("  " + "-" * (8 + 34 * len(BACKENDS)))
    for nq in WIDTHS:
        print(f"  {nq:<8d}", end="")
        for backend in BACKENDS:
            sub = df[(df.backend == backend) & (df.qubits == nq)]
            cells = "".join(f"{sub.iloc[0][c]:>10.2f}" for c in CATEGORIES) if not sub.empty \
                else f"{'-':>30s}"
            print("  |" + cells.rjust(32), end="")
        print()
    print("\n  Seconds. Leaf Depth is the depth of the static circuit in the innermost loop.")

    out = table_dir("tab06") / f"table06_mapping_time_{origin}.csv"
    df.to_csv(out, index=False)
    print()
    saved(out)

    return None
