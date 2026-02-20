from pathlib import Path
import os
import csv
import statistics as stats
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def collect_times_by_depth(method_root: Path, qubits: int):
    """
    Walk a method root and return {leaf_depth(int): [times]} for a given qubit size.
    Expects structure like:
      <method_root>/<qubits>qbt/.../leaf-depth-10/SEED_3/time.txt
    and specifically folders:
      queko-0{qubits}qbt_nest_00_nodes001_leaf-depth-{depth}/SEED_{seed}/time.txt
    """
    times_by_depth = {}
    qstr = f"{qubits:03d}"    # zero-padded e.g. 054, 081, 121
    for depth in range(10, 91, 10):
        method_dir = method_root / f"{qubits}qbt" / f"queko-{qstr}qbt_nest_00_nodes001_leaf-depth-{depth}"

        seed_times = []
        # Each leaf-depth-XX contains SEED_* subfolders with time.txt
        for seed in [3, 21, 42, 63, 84, 105, 126, 147, 168, 189]:
            seed_dir = method_dir / f"SEED_{seed}"
            time_file_path = seed_dir / "time.txt"
            try:
                with open(time_file_path, 'r') as time_file:
                    val = float(time_file.read().strip())
                    seed_times.append(val)
            except Exception:
                # Skip missing/malformed entries silently
                pass

        if seed_times:
            times_by_depth.setdefault(depth, []).extend(seed_times)
    return times_by_depth


def mean_std_series(times_by_depth: dict):
    """Convert {depth: [times]} -> (sorted_depths, avg_times, std_times)."""
    depths = sorted(times_by_depth.keys())
    avgs = []
    stds = []
    for d in depths:
        vals = times_by_depth[d]
        avgs.append(stats.mean(vals))
        # population vs sample std: use sample std if >=2 values, else 0
        if len(vals) >= 2:
            stds.append(stats.pstdev(vals))  # or stats.stdev(vals) if you prefer sample
        else:
            stds.append(0.0)
    return depths, avgs, stds


def plot_qroqi_time_three_sizes(
                                sizes=(54, 81, 121),
                                images_root="paper-images",
                                backend_folder="backend",
                                nest_folder="nest0",
                                topology_name="ibm_brisbane_old",
                                use_log_scale=False,
                                write_csv=True):
    """
    Publication-style plot of QROQI average routing time vs leaf depth for 3 qubit sizes.
    - Uses same font/DPI/PDF text embedding/grid/tight_layout style as your other plotting function.
    - Saves a single PDF figure with 3 curves (54/81/121q).
    - Optionally writes a combined CSV with per-depth averages and counts.
    """

    # ---- Global style (matches your publication settings) ----
    plt.rcParams.update({
    # "font.size": 12,              # base font size
    # "axes.titlesize": 14,         # optional title size
    # "axes.labelsize": 14,         # x/y labels
    # "xtick.labelsize": 11,
    # "ytick.labelsize": 11,
    # "legend.fontsize": 11,
    # "legend.title_fontsize": 12,
    # "lines.linewidth": 1.5,
    # "lines.markersize": 5,
    # "figure.dpi": 300,
    "font.size": 20,              # base font size
    "axes.titlesize": 22,         # in case titles are used later
    "axes.labelsize": 22,         # axis labels (x/y)
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
    "legend.fontsize": 18,
    "legend.title_fontsize": 20,
    "lines.linewidth": 2.5,
    "lines.markersize": 8,
    "figure.dpi": 300
    })  

    plt.rcParams["pdf.fonttype"] = 42  # embed text as text
    plt.rcParams["ps.fonttype"] = 42

    method_root = Path(f"results_time/qroqi/one_loop/{topology_name}")


    # ---- Collect data ----
    data = {}
    for n in sizes:
        times = collect_times_by_depth(method_root, n)
        data[n] = times

    # Build union of depths to align series
    all_depths = sorted(set().union(*[set(d.keys()) for d in data.values()]))

    # Prepare means/std aligned to all_depths
    series = {}
    for n in sizes:
        # compute per-depth mean/std first on native order, then align
        dpths, avgs, stds = mean_std_series(data[n])
        d_to_avg = dict(zip(dpths, avgs))
        d_to_std = dict(zip(dpths, stds))
        series[n] = {
            "avg": [d_to_avg.get(d, np.nan) for d in all_depths],
            "std": [d_to_std.get(d, np.nan) for d in all_depths],
        }

    # ---- Optional CSV ----
    if write_csv:
        out_csv = "leaf_depth_qroqi_54_81_121.csv"
        with open(out_csv, "w", newline="") as f:
            w = csv.writer(f)
            header = ["leaf_depth"]
            for n in sizes:
                header += [f"qroqi_{n}q_avg_time", f"qroqi_{n}q_std_time", f"qroqi_{n}q_n"]
            w.writerow(header)

            for d in all_depths:
                row = [d]
                for n in sizes:
                    vals = data[n].get(d, [])
                    avg = series[n]["avg"][all_depths.index(d)]
                    std = series[n]["std"][all_depths.index(d)]
                    row += [("" if np.isnan(avg) else avg),
                            ("" if np.isnan(std) else std),
                            len(vals)]
                w.writerow(row)
        print(f"📝 Wrote CSV: {out_csv}")

    # ---- Plot (single figure, three curves, with std bands) ----
    out_dir = os.path.join(images_root, topology_name)
    os.makedirs(out_dir, exist_ok=True)
    base_tag = f"{topology_name}_qroqi_{'_'.join(str(q) for q in sizes)}q_time"

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    marker_map = {sizes[0]: "o", sizes[1]: "s", sizes[2]: "D"} if len(sizes) >= 3 else {}

    x = np.array(all_depths, dtype=float)
    for n in sizes:
        y = np.array(series[n]["avg"], dtype=float)
        ystd = np.array(series[n]["std"], dtype=float)

        # mask out NaNs (depths missing in a size) for plotting lines
        mask = ~np.isnan(y)
        if not np.any(mask):
            continue
        ax.plot(x[mask], y[mask], marker=marker_map.get(n, "o"), label=f"{n} qubits")
        # shaded band only where both y and ystd are finite
        m2 = mask & ~np.isnan(ystd)
        if np.any(m2):
            ax.fill_between(x[m2], y[m2] - ystd[m2], y[m2] + ystd[m2], alpha=0.2)

    ax.set_xlabel("Leaf Depth")
    ax.set_ylabel("Average time ($s$)")
    ax.legend(loc="upper left", frameon=False)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if use_log_scale:
        ax.set_yscale("log")
    ax.set_ylim(bottom=0)

    out_pdf = os.path.join(out_dir, f"{base_tag}_time-vs-leaf-depth.pdf")
    plt.savefig(out_pdf, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"📄 Saved: {out_pdf}")





# ---------------- Example call ----------------

plot_qroqi_time_three_sizes(
    sizes=(54, 81, 121),
    images_root="paper-images/US/time-analysis",
    backend_folder="backend",
    nest_folder="nest0",
    topology_name="ibm_kingston",
    use_log_scale=False,
    write_csv=False
)
plot_qroqi_time_three_sizes(
    sizes=(54, 81, 121),
    images_root="paper-images/US/time-analysis",
    backend_folder="backend",
    nest_folder="nest0",
    topology_name="ibm_brisbane_old",
    use_log_scale=False,
    write_csv=False
)
