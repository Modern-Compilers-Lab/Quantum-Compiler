import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_combined_aggregated_from_csv(csv_path, out_dir="paper-images/nested-plots"):


    plt.rcParams.update({
        "font.size": 32,              # base font size
        "axes.titlesize": 34,         # in case titles are used later
        "axes.labelsize": 34,         # axis labels (x/y)
        "xtick.labelsize": 32,
        "ytick.labelsize": 30,
        "legend.fontsize": 28,
        "legend.title_fontsize": 34,
        "lines.linewidth": 2.5,
        "lines.markersize": 16,
        "figure.dpi": 300 ,        # ensure crisp rendering in PDF
    })
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42

    os.makedirs(out_dir, exist_ok=True)

    df = pd.read_csv(csv_path)
    df[["w", "i"]] = df["folder"].str.split("_", expand=True).astype(int)

    agg = (
        df.groupby("w")
        .agg(
            sabre_swaps_mean=("sabre_mean_swaps", "mean"),
            sabre_swaps_std=("sabre_mean_swaps", "std"),
            qroqi_swaps_mean=("qroqi_mean_swaps", "mean"),
            qroqi_swaps_std=("qroqi_mean_swaps", "std"),

            sabre_depth_mean=("sabre_mean_depth", "mean"),
            sabre_depth_std=("sabre_mean_depth", "std"),
            qroqi_depth_mean=("qroqi_mean_depth", "mean"),
            qroqi_depth_std=("qroqi_mean_depth", "std"),

            sabre_latency_mean=("sabre_mean_latency", "mean"),
            sabre_latency_std=("sabre_mean_latency", "std"),
            qroqi_latency_mean=("qroqi_mean_latency", "mean"),
            qroqi_latency_std=("qroqi_mean_latency", "std"),

            sabre_error_mean=("sabre_mean_error", "mean"),
            sabre_error_std=("sabre_mean_error", "std"),
            qroqi_error_mean=("qroqi_mean_error", "mean"),
            qroqi_error_std=("qroqi_mean_error", "std"),
        )
        .reset_index()
        .sort_values("w")
    )

    x = agg["w"].astype(float).to_numpy()

    def combined_plot(
        left_ours, left_ours_std, left_sabre, left_sabre_std, left_label,
        right_ours, right_ours_std, right_sabre, right_sabre_std, right_label,
        fname, left_log=True, right_log=True,
        left_color="tab:blue", right_color="tab:orange"
    ):
        fig, ax1 = plt.subplots(figsize=(8.5, 6.6))
        ax2 = ax1.twinx()

        # offsets
        dx = [-0.16, -0.05, 0.05, 0.16]

        # left metric
        h1 = ax1.errorbar(
            x + dx[0], left_ours, yerr=left_ours_std,
            fmt='o', linestyle='none',
            color=left_color, ecolor=left_color,
            capsize=2.5, markersize=16,
            label=f"Us {left_label}"
        )
        h2 = ax1.errorbar(
            x + dx[1], left_sabre, yerr=left_sabre_std,
            fmt='s', linestyle='none',
            color=left_color, ecolor=left_color,
            mfc='white', mec=left_color,
            capsize=2.5, markersize=16,
            label=f"Sabre {left_label}"
        )

        # right metric
        h3 = ax2.errorbar(
            x + dx[2], right_ours, yerr=right_ours_std,
            fmt='^', linestyle='none',
            color=right_color, ecolor=right_color,
            capsize=2.5, markersize=16,
            label=f"Us {right_label}"
        )
        h4 = ax2.errorbar(
            x + dx[3], right_sabre, yerr=right_sabre_std,
            fmt='D', linestyle='none',
            color=right_color, ecolor=right_color,
            mfc='white', mec=right_color,
            capsize=2.5, markersize=16,
            label=f"Sabre {right_label}"
        )

        # ---- AXES (ALL BLACK) ----
        ax1.set_xlabel("Nested While Loops ")
        ax1.set_ylabel(left_label, color="black")
        ax2.set_ylabel(right_label, color="black")

        ax1.tick_params(axis='y', colors="black")
        ax2.tick_params(axis='y', colors="black")
        ax1.tick_params(axis='x', colors="black")

        # add headroom so legend doesn't overlap

        # ticks
        ax1.set_xticks(x.astype(int))
        ax1.set_xlim(x.min() - 0.38, x.max() + 0.38)

        if left_log:
            ax1.set_yscale("log")
        if right_log:
            ax2.set_yscale("log")
       
        ymin, ymax = ax1.get_ylim()
        ax1.set_ylim(ymin, ymax * 4)

        ymin2, ymax2 = ax2.get_ylim()
        ax2.set_ylim(ymin2, ymax2 * 4)

        ax1.grid(True, axis="y", alpha=0.25)
        ax1.grid(False, axis="x")

        # legend
        handles = [h1, h2, h3, h4]
        labels = [h.get_label() for h in handles]
        ax1.legend(
            handles, labels,
            loc="upper left",
            # frameon=False,
            # ncol=2,
            # columnspacing=0.8,
            # handletextpad=0.4
             frameon=True, fancybox=True, edgecolor="black", framealpha=0.5, labelspacing=.1,    bbox_to_anchor=(0, 1.01)
             ,handletextpad=0.1,borderpad=0.1
        )

        # tight layout
        # fig.subplots_adjust(left=0.16, right=0.84, bottom=0.20, top=0.96)

        out_path = os.path.join(out_dir, f"{fname}.pdf")
        plt.savefig(out_path, format="pdf", bbox_inches="tight", pad_inches=0.01)
        plt.close(fig)

        print(f"Saved: {out_path}")

    # swaps + depth
    combined_plot(
        agg["qroqi_swaps_mean"], agg["qroqi_swaps_std"],
        agg["sabre_swaps_mean"], agg["sabre_swaps_std"],
        "SWAPs",

        agg["qroqi_depth_mean"], agg["qroqi_depth_std"],
        agg["sabre_depth_mean"], agg["sabre_depth_std"],
        "Depth",

        "swaps_depth_vs_w",
        left_color="tab:blue",
        right_color="tab:orange"
    )

    # latency + error
    combined_plot(
        agg["qroqi_latency_mean"], agg["qroqi_latency_std"],
        agg["sabre_latency_mean"], agg["sabre_latency_std"],
        "Latency",

        agg["qroqi_error_mean"], agg["qroqi_error_std"],
        agg["sabre_error_mean"], agg["sabre_error_std"],
        "Error",

        "latency_error_vs_w",
        left_color="tab:blue",
        right_color="tab:orange"
    )
if __name__ == "__main__":
    plot_combined_aggregated_from_csv("results-summary/w_i_rule_metrics.csv")
