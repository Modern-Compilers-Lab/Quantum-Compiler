"""Table 7 - what drives the improvement on dynamic circuits (Sec. 7.6.2).

Average percentage improvement of each cumulative configuration over the
reconciliation-only baseline (i). Computed per leaf depth as
(baseline - variant) / baseline, then averaged over the nine leaf depths - the
Table 3 rule applied against configuration (i) rather than against Sabre.

Each configuration is generated with generate.py --ablation 1..4.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import paper_values as pv
from common import METRICS, banner, saved, table_dir
from csv_sources import resolve

WIDTHS = [81, 121]
#: The baseline itself is not a row of Table 7.
REPORTED = ["(ii) Recon+Err", "(iii) Recon+Err+Depth", "(iv) Recon+Err+Depth+Remap"]
HINT = "python artifact/generate.py ablation"


def ablation_csv_rel(nq, iters):
    return (f"{pv.ABLATION_STUDY}/"
            f"{pv.ABLATION_BACKEND}_{nq}qbt_{iters}iter_metrics_ablation_study.csv")


def load(nq, iters):
    path, origin = resolve(ablation_csv_rel(nq, iters), HINT)
    df = pd.read_csv(path).sort_values("leaf_depth").reset_index(drop=True)
    missing = [f"{m}_mean_{s}" for m in METRICS for s in pv.ABLATION_COLUMNS.values()
               if f"{m}_mean_{s}" not in df.columns]
    if missing:
        raise SystemExit(f"{path} is missing ablation columns: {missing}")
    return df, origin


def improvements(df, config_key):
    """Mean over leaf depths of the improvement over configuration (i)."""
    suffix = pv.ABLATION_COLUMNS[config_key]
    out = {}
    for m in METRICS:
        base = df[f"{m}_mean_{pv.ABLATION_BASELINE}"].astype(float)
        var = df[f"{m}_mean_{suffix}"].astype(float)
        with np.errstate(divide="ignore", invalid="ignore"):
            imp = (base - var) / base * 100.0
        out[m] = float(np.nanmean(imp.replace([np.inf, -np.inf], np.nan)))
    return out


def render(per_width, widths):
    print(f"  {'Method':<30s}", end="")
    for nq in widths:
        print(f"  |{(str(nq) + ' Qubits').center(32)}", end="")
    print()
    print(f"  {'':<30s}", end="")
    for _ in widths:
        print("  |" + "".join(f"{k:>8s}" for k in ("S", "D", "L", "E")), end="")
    print()
    print("  " + "-" * (30 + 34 * len(widths)))
    for key in REPORTED:
        print(f"  {key:<30s}", end="")
        for nq in widths:
            print("  |" + "".join(f"{per_width[nq][key][m]:>8.1f}" for m in METRICS), end="")
        print()


def run(args):
    banner("Table 7 - Ablation over the reconciliation-only baseline (Sec. 7.6.2)",
           f"improvement % vs configuration (i); backend = {pv.ABLATION_BACKEND}")

    per_width, rows, origins = {}, [], set()
    for nq in WIDTHS:
        try:
            df, origin = load(nq, args.loop_iterations)
        except SystemExit as exc:
            print(f"\n  [SKIP] {nq}qbt: {exc}\n")
            continue
        origins.add(origin)
        per_width[nq] = {k: improvements(df, k) for k in REPORTED}
        for key, vals in per_width[nq].items():
            rows.append({"qubits": nq, "config": key, "n_leaf_depths": len(df), **vals})
    if not per_width:
        raise SystemExit("no ablation CSV for any width")
    widths = [nq for nq in WIDTHS if nq in per_width]
    print(f"  source: {'/'.join(sorted(origins))} CSVs\n")

    render(per_width, widths)
    print("\n  Baseline (i) Recon = reconciliation passes only.   "
          "S=SWAPs D=Depth L=Latency E=Error")

    out = table_dir("tab07") / "table07_ablation.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print()
    saved(out)

    return {"produced": len(widths), "total": len(WIDTHS)}
