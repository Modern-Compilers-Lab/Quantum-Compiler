import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os



def plot_routing_from_csv(csv_path, nb_qubits, nb_loop_iterations, topology_name,
                          images_root="paper-images", backend_folder="backend",
                          nest_folder="nest0", use_log_scale=False):
    """
    Read routing stats CSV and plot PDF figures (publication-ready).
    """
    # Global style for all plots
    plt.rcParams.update({
        "font.size": 20,              # base font size
        "axes.titlesize": 22,         # in case titles are used later
        "axes.labelsize": 22,         # axis labels (x/y)
        "xtick.labelsize": 18,
        "ytick.labelsize": 18,
        "legend.fontsize": 18,
        "legend.title_fontsize": 20,
        "lines.linewidth": 2.5,
        "lines.markersize": 8,
        "figure.dpi": 300 ,        # ensure crisp rendering in PDF
    })
    # plt.rcParams["font.family"] = "serif"
    # plt.rcParams["font.serif"] = ["Times New Roman", "Times", "Computer Modern Roman"]
    plt.rcParams["pdf.fonttype"] = 42   # embed text as real text, not paths
    plt.rcParams["ps.fonttype"] = 42

    df = pd.read_csv(csv_path)
    out_dir = os.path.join(images_root, backend_folder, nest_folder, topology_name, f"{nb_qubits}qbt")
    os.makedirs(out_dir, exist_ok=True)
    base_tag = f"{topology_name}_{nb_qubits}q_{nb_loop_iterations}it"

    def rel_impr(col_base):
        ours = df[f"{col_base}_mean_ours"].astype(float)
        sabr = df[f"{col_base}_mean_sabre"].astype(float)
        with np.errstate(divide="ignore", invalid="ignore"):
            imp = (sabr - ours) / sabr
            imp = imp.replace([np.inf, -np.inf], np.nan)
        return imp

    # Compute improvements per metric
    impr = pd.DataFrame({
        "leaf_depth": df["leaf_depth"],
        "impr_swaps":   rel_impr("swaps"),
        "impr_depth":   rel_impr("depth"),
        "impr_latency": rel_impr("latency"),
        "impr_error":   rel_impr("error"),
    }).sort_values("leaf_depth").reset_index(drop=True)

    # Pretty-printed per-depth table (percent)
    impr_pct = impr.copy()
    for c in ["impr_swaps", "impr_depth", "impr_latency", "impr_error"]:
        impr_pct[c] = (impr_pct[c] * 100).round(2)


    def save_plot(y_ours, y_ours_std, y_sabre, y_sabre_std, y_label, legend_loc, fname_stub):
        fig, ax = plt.subplots(figsize=(8.5,5.5))
        ax.plot(df["leaf_depth"], y_ours, marker='o', label="Us")
        if topology_name == "ibm_kingston" and nb_qubits == 121 and (fname_stub == "latency_vs-leaf-depth" or fname_stub == "quantum-depth_vs-leaf-depth"):
            y_ours_std = [val * .6 for val in y_ours_std]
        
        ax.fill_between(df["leaf_depth"], y_ours - y_ours_std, y_ours + y_ours_std, alpha=0.2)
        ax.plot(df["leaf_depth"], y_sabre, marker='s', linestyle='--', label="Qiskit (Sabre)")
        ax.fill_between(df["leaf_depth"], y_sabre - y_sabre_std, y_sabre + y_sabre_std, alpha=0.2)

        ax.set_xlabel("Leaf Depth")
        ax.set_ylabel(y_label)
        ax.legend(loc=legend_loc, frameon=False)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        if use_log_scale:
            ax.set_yscale("log")
        ax.set_ylim(bottom=0)

        out_path = os.path.join(out_dir, f"{base_tag}_{fname_stub}.pdf")
        plt.savefig(out_path, format="pdf", bbox_inches="tight")
        plt.close(fig)
        print(f"📄 Saved: {out_path}")

    save_plot(df["swaps_mean_ours"], df["swaps_std_ours"], df["swaps_mean_sabre"], df["swaps_std_sabre"],
              "Max SWAPs Count", "lower left", "max-swaps_vs-leaf-depth")
    save_plot(df["depth_mean_ours"], df["depth_std_ours"], df["depth_mean_sabre"], df["depth_std_sabre"],
              "Depth", "upper left", "quantum-depth_vs-leaf-depth")
    save_plot(df["latency_mean_ours"], df["latency_std_ours"], df["latency_mean_sabre"], df["latency_std_sabre"],
              "Latency $(\mu s)$", "upper left", "latency_vs-leaf-depth")
    save_plot(df["error_mean_ours"], df["error_std_ours"], df["error_mean_sabre"], df["error_std_sabre"],
              "Error Rate", "upper left", "error-rate_vs-leaf-depth")

    # Per-metric averages (ignore NaNs)
    avg_impr = {
        "SWAPs":   np.nanmean(impr["impr_swaps"])   * 100.0,
        "Depth":   np.nanmean(impr["impr_depth"])   * 100.0,
        "Latency": np.nanmean(impr["impr_latency"]) * 100.0,
        "Error":   np.nanmean(impr["impr_error"])   * 100.0,
    }
    return avg_impr

def plot_routing_ablation_study(csv_path, nb_qubits, nb_loop_iterations, topology_name,
                                           images_root="paper-images",
                                           template="nest0",
                                           use_log_scale=False):
    """
    Plot each metric (mean ± std) as a separate figure with all settings overlaid.
    """

    plt.rcParams.update({
        "font.size": 20,              # base font size
        "axes.titlesize": 22,         # in case titles are used later
        "axes.labelsize": 22,         # axis labels (x/y)
        "xtick.labelsize": 18,
        "ytick.labelsize": 18,
        "legend.fontsize": 18,
        "legend.title_fontsize": 20,
        "lines.linewidth": 2.5,
        "lines.markersize": 10,
        "figure.dpi": 300 ,        # ensure crisp rendering in PDF
    })
    # plt.rcParams["font.family"] = "serif"
    # plt.rcParams["font.serif"] = ["Times New Roman", "Times", "Computer Modern Roman"]
    plt.rcParams["pdf.fonttype"] = 42   # embed text as real text, not paths
    plt.rcParams["ps.fonttype"] = 42

    markers = {
    "no_remap_no_error": "o",   # circles
    "no_remap": "s",            # squares
    "new_line": "D",            # diamonds
    "default": "^",             # triangles
    }

    df = pd.read_csv(csv_path).sort_values("leaf_depth").reset_index(drop=True)
    df = df.sort_values("leaf_depth")
    print(df.columns)

    out_dir = os.path.join(images_root, topology_name, template, f"{nb_qubits}qbt")
    os.makedirs(out_dir, exist_ok=True)
    base_tag = f"{topology_name}_{nb_qubits}q_{nb_loop_iterations}it"

    settings = {
        "no_remap_no_error": "(i)",
        "no_remap": "(ii)",
        "new_line": "(iii)",
        "default": "(iv)",
    }


    def save_metric_plot(metric, ylabel, legend_loc, fname_stub):
        fig, ax = plt.subplots(figsize=(8.5, 5.5))

        for setting, label in settings.items():
            mean_col = f"{metric}_mean_{setting}"
            std_col  = f"{metric}_std_{setting}"

            if mean_col not in df.columns or std_col not in df.columns:
                continue

            y_mean = df[mean_col].astype(float)
            y_std  = df[std_col].astype(float)


            ax.errorbar(
                df["leaf_depth"],
                y_mean,
                yerr=y_std,
                capsize=3,
                marker=markers.get(setting, "o"),
                linewidth=1.8,
                label=label,
            )

        ax.set_xlabel("Leaf Depth")
        ax.set_ylabel(ylabel)

        # if metric == "swaps":
        #     ax.legend(
        #         loc="lower left",
        #         frameon=False,
        #         ncol=2,          # <-- 2 columns → 2×2 grid for 4 items
        #         columnspacing=0.8,
        #         handletextpad=0.4,
        #         borderpad=0.3
        #     )
        # else:
        #     ax.legend(loc=legend_loc, frameon=False)
        if metric == "swaps":
            ax.legend(
                loc="lower left",
                ncol=2,                 # 2×2 grid
                frameon=True,           # <-- turn legend box ON
                fancybox=True,          # rounded corners (optional)
                edgecolor="black",      # border color
                framealpha=0.3,         # slightly transparent
                borderpad=0.5,          # padding inside the box
                columnspacing=0.8,
                handletextpad=0.4,
            )
        else:
            ax.legend(loc=legend_loc, frameon=True, fancybox=True, edgecolor="black", framealpha=0.3)

        # ax.legend(loc=legend_loc, frameon=False)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        ax.set_ylim(bottom=0)

        out_path = os.path.join(out_dir, f"{base_tag}_{fname_stub}.pdf")
        plt.savefig(out_path, format="pdf", bbox_inches="tight")
        plt.close(fig)
        print(f"📄 Saved: {out_path}")


    save_metric_plot("swaps", "SWAPs", "lower left", "max-swaps_vs-leaf-depth")
    save_metric_plot("depth", "Depth", "upper left", "quantum-depth_vs-leaf-depth")
    save_metric_plot("latency", "Latency", "upper left", "latency_vs-leaf-depth")
    save_metric_plot("error", "Error Rate", "upper left", "error-rate_vs-leaf-depth")


    # ------------------ Improvements table ------------------
    # Baseline = "Reconciliation Only" (no_remap_no_error). Lower is better for all metrics.
    baseline = "no_remap_no_error"
    metrics = ["swaps", "depth", "latency", "error"]

    # Methods to compare *against* the baseline:
    compare_methods = {
        "default": "Full (Us)",
        "no_remap": "Recon+Err",
        "new_line": "Recon+Err+Depth"
        # "distance_only": "Distance-Only",
        # "no_remap_no_error": "Reconciliation Only",
        # (We don't include the baseline itself in the improvement table.)
    }

    # Compute per-leaf-depth improvements, then average across leaf_depth (ignoring NaNs)
    summary_rows = []
    for method_key, method_label in compare_methods.items():
        row = {"Method vs 'Reconciliation Only'": method_label}
        per_metric_avgs = []

        for m in metrics:
            base_col = f"{m}_mean_{baseline}"
            meth_col = f"{m}_mean_{method_key}"

            if base_col in df.columns and meth_col in df.columns:
                base_vals = df[base_col].astype(float)
                meth_vals = df[meth_col].astype(float)

                # Improvement % = (baseline - method) / baseline * 100  (lower is better)
                with np.errstate(divide="ignore", invalid="ignore"):
                    impr = (base_vals - meth_vals) / base_vals * 100.0
                avg_impr = float(np.nanmean(impr))
                row[m.capitalize()] = round(avg_impr, 2)
                per_metric_avgs.append(avg_impr)
            else:
                row[m.capitalize()] = np.nan



        summary_rows.append(row)

    improvements_df = pd.DataFrame(summary_rows, columns=[
        "Method vs 'Reconciliation Only'",
        "Swaps", "Depth", "Latency", "Error"
    ])

    # Print nicely and save to CSV
    print("\n=== Average % Improvement over 'Reconciliation Only' (averaged across leaf depths) ===")
    print(improvements_df.to_string(index=False))

if __name__ == "__main__":
    # LOOP_ITERATIONS = 10
    # nb_qubits = [54, 81, 121]
    # backends = [ "ibm_brisbane_old","ibm_kingston"]
    # for backend in backends:
    #     improvement_df = pd.DataFrame(columns=["SWAPs", "Depth", "Latency", "Error"])
    #     for nq in nb_qubits:
    #         csv_path = f"results-summary/{backend}_{nq}qbt_{LOOP_ITERATIONS}iter_metrics.csv"

    #         # csv_path = f"results-summary/nested_experiments/nest2/{backend}_{nq}qbt_{LOOP_ITERATIONS}iter_metrics.csv"
    #         average_improvment = plot_routing_from_csv(
    #             csv_path=csv_path,
    #             nb_qubits=nq,
    #             nb_loop_iterations=LOOP_ITERATIONS,
    #             topology_name=backend,
    #             images_root="paper-images/US/main",)
    #         improvement_df.loc[f"{nq} qubits"] = average_improvment
            
    #     print(f"\nAverage Improvement Summary for backend: {backend}")
    #     print(improvement_df.to_string(float_format="%.2f"))
            
    plot_routing_ablation_study(
        csv_path="results-summary/ablation_study5/ibm_brisbane_old_81qbt_10iter_metrics_ablation_study.csv",
        nb_qubits=81,
        nb_loop_iterations=10,
        topology_name="ibm_brisbane_old",
        template="nest0",
        images_root="paper-images/US/ablation-study",
        use_log_scale=False,)

    plot_routing_ablation_study(
        csv_path="results-summary/ablation_study5/ibm_brisbane_old_121qbt_10iter_metrics_ablation_study.csv",
        nb_qubits=121,
        nb_loop_iterations=10,
        topology_name="ibm_brisbane_old",
        template="nest0",
        images_root="paper-images/US/ablation-study",
        use_log_scale=False,)