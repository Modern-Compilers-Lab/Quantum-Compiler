"""Figure 10 - ablation study on the Brisbane back-end (Sec. 7.6.2).

Eight panels: 81 and 121 qubits x {SWAPs, Depth, Latency, Error}, showing all
four configurations as mean with +/-1 sigma across leaf depths.
"""

from __future__ import annotations

import paper_values as pv
from common import apply_paper_style, banner, figure_dir, saved
from render.tab07_ablation import load

WIDTHS = [81, 121]

#: metric -> (display scale, y label, legend location, filename stub)
PANELS = [
    ("swaps", 1e3, "SWAPs (x10^3)", "lower left", "max-swaps_vs-leaf-depth"),
    ("depth", 1e3, "Depth (x10^3)", "upper left", "quantum-depth_vs-leaf-depth"),
    ("latency", 1e3, "Latency (ms)", "upper left", "latency_vs-leaf-depth"),
    ("error", 1.0, "Error Rate", "upper left", "error-rate_vs-leaf-depth"),
]

MARKERS = {"no_remap_no_error": "o", "no_remap": "s", "new_line": "D", "default": "^"}


def panel(plt, df, metric, scale, ylabel, legend_loc, out_path):
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    for suffix, label in pv.ABLATION_PANEL_LABELS.items():
        mean_col, std_col = f"{metric}_mean_{suffix}", f"{metric}_std_{suffix}"
        if mean_col not in df.columns:
            continue
        mean = df[mean_col].astype(float) / scale
        std = df[std_col].astype(float).fillna(0) / scale
        ax.errorbar(df["leaf_depth"], mean, yerr=std, capsize=3,
                    marker=MARKERS.get(suffix, "o"), linewidth=1.8, label=label)

    ax.set_xlabel("Leaf Depth")
    ax.set_ylabel(ylabel)
    if metric == "swaps":
        ax.legend(loc="lower left", ncol=2, frameon=True, fancybox=True,
                  edgecolor="black", framealpha=0.4, borderpad=0.1,
                  columnspacing=0.8, handletextpad=0.4, labelspacing=0.1,
                  bbox_to_anchor=(0.0, -0.05))
    else:
        ax.legend(loc=legend_loc, frameon=True, fancybox=True, edgecolor="black",
                  framealpha=0.3, labelspacing=0.1, bbox_to_anchor=(0.0, 1.05))
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)
    plt.tight_layout()
    plt.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.close(fig)


def run(args):
    banner("Figure 10 - Ablation study on Brisbane (Sec. 7.6.2)",
           "(i) Recon, (ii) +Err, (iii) +Depth, (iv) full DynamiQ; "
           "mean +/-1 sigma across leaf depths")
    plt = apply_paper_style()
    out_dir = figure_dir("fig10")

    done = 0
    for nq in WIDTHS:
        try:
            df, origin = load(nq, args.loop_iterations)
        except SystemExit as exc:
            print(f"\n  [SKIP] {nq}qbt: {exc}")
            continue
        done += 1
        print(f"\n  {nq} qbt  source: {origin}  ({len(df)} leaf depths)")
        for metric, scale, ylabel, loc, stub in PANELS:
            out = (out_dir /
                   f"{pv.ABLATION_BACKEND}_{nq}q_{args.loop_iterations}it_{stub}.pdf")
            panel(plt, df, metric, scale, ylabel, loc, out)
            saved(out)
        for suffix, label in pv.ABLATION_PANEL_LABELS.items():
            col = df[f"swaps_mean_{suffix}"]
            print(f"     {label:<5s} swaps {col.iloc[0]:10.1f} -> {col.iloc[-1]:10.1f}")

    if done == 0:
        raise SystemExit("no ablation CSV for any width")
    return {"produced": done, "total": len(WIDTHS)}
