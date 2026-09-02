"""Table 5 - rotated surface-code circuits (memory-Z, generated with Stim).

Absolute cells are per-distance means over rounds, replicates and seeds.
"Overall imp." pools every row of the backend.

Brisbane traces are scored at loop_iterations=10, MECH at 3 (it only has r=3);
scoring MECH at 10 inflates each improvement by roughly 10 points.
"""

from __future__ import annotations

import pandas as pd

import paper_values as pv
from common import METRICS, banner, saved, table_dir
from csv_sources import resolve

BACKENDS = ["ibm_brisbane", "mech_heavy_hex"]
TAGS = {"ibm_brisbane": "surface_code_stim", "mech_heavy_hex": "surface_code_mech"}
DISPLAY = {"ibm_brisbane": "Brisbane", "mech_heavy_hex": "MECH"}
HINT = "python artifact/generate.py surface-code"


def per_distance(df):
    cols = [f"{m}_mean_{w}" for m in METRICS for w in ("ours", "sabre")]
    return df.groupby("distance")[cols].mean()


def pooled_improvement(df, metric):
    o = float(df[f"{metric}_mean_ours"].mean())
    s = float(df[f"{metric}_mean_sabre"].mean())
    return (s - o) / s * 100.0 if s else float("nan")


def render(name, df, iters):
    g = per_distance(df)
    print(f"  {'Backend':<9s}{'d':>2s}{'Qubits':>8s}{'CX/rnd':>8s}", end="")
    for m in METRICS:
        print(f"  |{m.upper().center(21)}", end="")
    print()
    print(f"  {'':<9s}{'':>2s}{'':>8s}{'':>8s}", end="")
    for _ in METRICS:
        print(f"  |{'Ours':>10s}{'Sabre':>11s}", end="")
    print()
    print("  " + "-" * 122)
    for i, d in enumerate(g.index):
        d = int(d)
        shape = pv.TABLE5_SHAPE.get(d, {})
        print(f"  {DISPLAY.get(name, name) if i == 0 else '':<9s}{d:>2d}"
              f"{shape.get('qubits', 2 * d * d - 1):>8d}{shape.get('cx_per_round', 0):>8d}", end="")
        for m in METRICS:
            print(f"  |{g.loc[d, f'{m}_mean_ours']:>10.2f}"
                  f"{g.loc[d, f'{m}_mean_sabre']:>11.2f}", end="")
        print()
    ov = {m: pooled_improvement(df, m) for m in METRICS}
    print(f"  {'Overall imp. Delta%':<27s}", end="")
    for m in METRICS:
        print(f"  |{f'{ov[m]:+.1f}%':>21s}", end="")
    print()
    print("  " + "-" * 122)
    print(f"  {len(df)} (distance, rounds) groups at loop_iterations={iters}. "
          "Latency in microseconds.")
    return ov


def run(args):
    banner("Table 5 - Rotated surface-code circuits (Sec. 7.4)",
           "memory-Z on 2d^2-1 qubits, four-step cx schedule")
    done = 0
    for name in BACKENDS:
        rel = f"{TAGS[name]}/{TAGS[name]}_{name}_comparison.csv"
        try:
            path, origin = resolve(rel, HINT)
        except SystemExit as exc:
            print(f"\n  [SKIP] {DISPLAY[name]}: {exc}")
            continue
        df = pd.read_csv(path)
        iters = pv.TABLE5_LOOP_ITERATIONS[name]
        print(f"\n  source: {origin} CSV")
        render(name, df, iters)
        out = table_dir("tab05") / f"table05_surface_code_{name}.csv"
        df.to_csv(out, index=False)
        saved(out)
        done += 1

    if done == 0:
        raise SystemExit("no backend could be rendered; see the skips above")
    return {"produced": done, "total": len(BACKENDS)}
