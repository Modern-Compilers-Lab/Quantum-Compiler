"""Figure 9 - dynamic nested-loop circuits as a function of nesting depth.

Two panels with twin axes: SWAPs + Depth, and Latency + Error, against the
number of nested while loops. Each w value averages the five i variants.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from common import apply_paper_style, banner, figure_dir, saved
from csv_sources import resolve

HINT = "python artifact/generate.py nested"
CSV_REL = "w_i_rule_metrics.csv"


def aggregate(df):
    df = df.copy()
    df[["w", "i"]] = df["folder"].str.split("_", expand=True).astype(int)
    spec = {}
    for m in ("swaps", "depth", "latency", "error"):
        for method in ("sabre", "qroqi"):
            spec[f"{method}_{m}_mean"] = (f"{method}_mean_{m}", "mean")
            spec[f"{method}_{m}_std"] = (f"{method}_mean_{m}", "std")
    return df.groupby("w").agg(**spec).reset_index().sort_values("w")


def twin_panel(plt, agg, left, right, labels, out_path):
    fig, ax1 = plt.subplots(figsize=(8.5, 6.6))
    ax2 = ax1.twinx()
    x = agg["w"].astype(float).to_numpy()
    dx = [-0.16, -0.05, 0.05, 0.16]
    specs = [
        (ax1, left, "qroqi", dx[0], "o", "tab:blue", None, f"Us {labels[0]}"),
        (ax1, left, "sabre", dx[1], "s", "tab:blue", "white", f"Sabre {labels[0]}"),
        (ax2, right, "qroqi", dx[2], "^", "tab:orange", None, f"Us {labels[1]}"),
        (ax2, right, "sabre", dx[3], "D", "tab:orange", "white", f"Sabre {labels[1]}"),
    ]
    handles = []
    for ax, metric, method, off, marker, color, face, label in specs:
        handles.append(ax.errorbar(
            x + off, agg[f"{method}_{metric}_mean"], yerr=agg[f"{method}_{metric}_std"],
            fmt=marker, linestyle="none", color=color, ecolor=color,
            mfc=face or color, mec=color, capsize=2.5, markersize=16, label=label))

    ax1.set_xlabel("Nested While Loops")
    ax1.set_ylabel(labels[0], color="black")
    ax2.set_ylabel(labels[1], color="black")
    ax1.set_xticks(x.astype(int))
    ax1.set_xlim(x.min() - 0.38, x.max() + 0.38)
    for ax in (ax1, ax2):
        ax.set_yscale("log")
        lo, hi = ax.get_ylim()
        ax.set_ylim(lo, hi * 4)
    ax1.grid(True, axis="y", alpha=0.25)
    ax1.grid(False, axis="x")
    ax1.legend(handles, [h.get_label() for h in handles], loc="upper left",
               bbox_to_anchor=(0, 1.01), frameon=True, fancybox=True, edgecolor="black",
               framealpha=0.5, labelspacing=0.1, handletextpad=0.1, borderpad=0.1)
    plt.savefig(out_path, format="pdf", bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)


def run(args):
    banner("Figure 9 - Nested while loops, 81-qubit circuits (Sec. 7.6.1)",
           "metrics vs number of nested while loops; lower is better")
    path, origin = resolve(CSV_REL, HINT)
    df = pd.read_csv(path)
    agg = aggregate(df)
    print(f"  source: {origin} CSV  ({len(df)} w_i folders, w = 1..{int(agg.w.max())})\n")

    plt = apply_paper_style()
    out_dir = figure_dir("fig09")
    for left, right, labels, stub in (("swaps", "depth", ("SWAPs", "Depth"), "swaps_depth_vs_w"),
                                      ("latency", "error", ("Latency", "Error"),
                                       "latency_error_vs_w")):
        out = out_dir / f"{stub}.pdf"
        twin_panel(plt, agg, left, right, labels, out)
        saved(out)

    print()
    for m in ("swaps", "depth", "latency", "error"):
        imp = float(np.mean((df[f"sabre_mean_{m}"] - df[f"qroqi_mean_{m}"])
                            / df[f"sabre_mean_{m}"] * 100.0))
        print(f"  mean improvement {m:<8s} {imp:+6.1f}%")
    return None
