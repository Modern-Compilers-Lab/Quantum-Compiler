"""Figure 8 - routing performance on IBM Brisbane across leaf depth.

Eight panels: 81 and 121 qubits x {SWAPs, Depth, Latency, Error}, mean with
+/-1 sigma shaded, DynamiQ vs Qiskit Sabre.
"""

from __future__ import annotations

from common import apply_paper_style, banner, figure_dir, load_pairwise_summary, saved
from csv_sources import resolve

BACKEND = "ibm_brisbane_old"
WIDTHS = [81, 121]
HINT = "python artifact/generate.py main"

PANELS = [
    ("swaps", 1e3, "SWAPs (x10^3)", "lower left", "max-swaps_vs-leaf-depth"),
    ("depth", 1e3, "Depth (x10^3)", "upper left", "quantum-depth_vs-leaf-depth"),
    ("latency", 1e6, "Latency (s)", "upper left", "latency_vs-leaf-depth"),
    ("error", 1.0, "Error Rate", "upper left", "error-rate_vs-leaf-depth"),
]


def panel(plt, df, metric, scale, ylabel, legend_loc, out_path):
    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    x = df["leaf_depth"]
    for suffix, label, marker, style in (("ours", "Us", "o", "-"),
                                         ("sabre", "Qiskit (Sabre)", "s", "--")):
        mean = df[f"{metric}_mean_{suffix}"] / scale
        std = df[f"{metric}_std_{suffix}"].fillna(0) / scale
        ax.plot(x, mean, marker=marker, linestyle=style, label=label)
        ax.fill_between(x, mean - std, mean + std, alpha=0.2)
    ax.set_xlabel("Leaf Depth")
    ax.set_ylabel(ylabel)
    anchor = (0.0, -0.05) if legend_loc == "lower left" else (0.0, 1.05)
    ax.legend(loc=legend_loc, bbox_to_anchor=anchor, labelspacing=0.2, frameon=False)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)
    plt.tight_layout()
    plt.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.close(fig)


def run(args):
    banner("Figure 8 - d-QUEKO routing on IBM Brisbane (Sec. 7.2)",
           "mean +/-1 sigma across seeds, vs leaf depth")
    plt = apply_paper_style()
    out_dir = figure_dir("fig08")

    for nq in WIDTHS:
        path, origin = resolve(f"main/{BACKEND}_{nq}qbt_{args.loop_iterations}iter_metrics.csv",
                               HINT)
        df = load_pairwise_summary(path)
        print(f"\n  {nq} qbt  source: {origin}  ({len(df)} leaf depths)")
        for metric, scale, ylabel, loc, stub in PANELS:
            out = out_dir / f"{BACKEND}_{nq}q_{args.loop_iterations}it_{stub}.pdf"
            panel(plt, df, metric, scale, ylabel, loc, out)
            saved(out)
            lo, hi = df[f"{metric}_mean_ours"].iloc[0], df[f"{metric}_mean_ours"].iloc[-1]
            print(f"     {metric:<8s} ours {lo:12.2f} -> {hi:12.2f}   "
                  f"sabre {df[f'{metric}_mean_sabre'].iloc[0]:12.2f} -> "
                  f"{df[f'{metric}_mean_sabre'].iloc[-1]:12.2f}")
    return None
