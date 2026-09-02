"""Figure 1 - motivating comparison, normalised bar charts.

Panel (a): nested w=5, i=5. Panel (b): 121 qubits at leaf depth 60.
Each metric is normalised by the larger of the two values.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from common import banner, apply_paper_style, figure_dir, load_pairwise_summary, saved
from csv_sources import resolve

METRICS = ["swaps", "depth", "latency", "error"]
LABELS = ["S", "D", "L", "E"]


def bar_panel(plt, values, out_path):
    plt.rcParams.update({"font.size": 20, "axes.labelsize": 26,
                         "xtick.labelsize": 24, "ytick.labelsize": 24})
    fig = plt.figure(figsize=(8.5, 4.5))
    x = np.arange(len(METRICS))
    width = 0.28
    colors = list(plt.cm.tab10.colors)
    for i, (method, label, color, hatch) in enumerate(
            [("sabre", "Sabre", colors[1], "*"), ("ours", "Us", colors[0], "\\\\")]):
        plt.bar(x + i * width - width / 2, [values[m][method] for m in METRICS],
                width=width, label=label, color=color, edgecolor="black",
                hatch=hatch, linewidth=0.6)
    plt.xticks(x, LABELS, fontsize=26)
    plt.grid(axis="y", linestyle="--", alpha=0.4)
    plt.legend(fontsize=38, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.25),
               fancybox=True, framealpha=0, borderpad=0, columnspacing=1.3, handlelength=1.8)
    plt.yticks([0, 1])
    plt.tight_layout()
    plt.savefig(out_path, format="pdf", bbox_inches="tight", pad_inches=0, dpi=300)
    plt.close(fig)


def normalise(raw):
    out = {}
    for m in METRICS:
        hi = max(raw[m].values())
        out[m] = {k: (v / hi if hi else v) for k, v in raw[m].items()}
    return out


def nested_panel_values(folder):
    path, origin = resolve("w_i_rule_metrics.csv", "python artifact/generate.py nested")
    df = pd.read_csv(path)
    row = df[df.folder == folder]
    if row.empty:
        raise SystemExit(f"folder {folder!r} not in {path}; have {sorted(df.folder)[:5]}...")
    row = row.iloc[0]
    return {m: {"ours": float(row[f"qroqi_mean_{m}"]),
                "sabre": float(row[f"sabre_mean_{m}"])} for m in METRICS}, origin


def main_panel_values(backend, nq, leaf_depth, iters):
    path, origin = resolve(f"main/{backend}_{nq}qbt_{iters}iter_metrics.csv",
                           "python artifact/generate.py main")
    df = load_pairwise_summary(path)
    row = df[df.leaf_depth == leaf_depth]
    if row.empty:
        raise SystemExit(f"leaf depth {leaf_depth} not in {path}")
    row = row.iloc[0]
    return {m: {"ours": float(row[f"{m}_mean_ours"]),
                "sabre": float(row[f"{m}_mean_sabre"])} for m in METRICS}, origin


def run(args):
    banner("Figure 1 - Motivating comparison (Sec. 1)",
           "normalised per metric; lower is better")
    plt = apply_paper_style()
    out_dir = figure_dir("fig01")

    panels = [
        ("a", "81qbt nested w=5 i=5", lambda: nested_panel_values("5_5"),
         "ibm_kingston_dynamique_81QBT_5w_5i.pdf"),
        ("b", "121qbt leaf depth 60",
         lambda: main_panel_values("ibm_brisbane_old", 121, 60, args.loop_iterations),
         "ibm_brisbane_dynamique_121QBT_60leafdepth.pdf"),
    ]
    done = 0
    for tag, title, getter, filename in panels:
        try:
            raw, origin = getter()
        except SystemExit as exc:
            print(f"\n  [SKIP] panel ({tag}) {title}: {exc}")
            continue
        print(f"\n  ({tag}) {title}   source: {origin}")
        for m in METRICS:
            o, s = raw[m]["ours"], raw[m]["sabre"]
            print(f"      {m:<8s} ours {o:12.2f}   sabre {s:12.2f}   "
                  f"improvement {(s - o) / s * 100:+6.1f}%")
        out = out_dir / filename
        bar_panel(plt, normalise(raw), out)
        saved(out)
        done += 1

    if done == 0:
        raise SystemExit("no panel could be rendered; see the skips above")
    return {"produced": done, "total": len(panels)}
