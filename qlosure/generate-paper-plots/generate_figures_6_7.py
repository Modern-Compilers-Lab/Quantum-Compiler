import os
import sys
from typing import Dict, List, Tuple
from collections import defaultdict
import argparse

sys.path.insert(0, os.path.abspath('..'))


if __name__ == "__main__":
    from qlosure.visualization.utils import *

    parser = argparse.ArgumentParser(description="Analyze Qlosure results")
    parser.add_argument("--benchmark", type=str,
                        default="queko-bss-16qbt", choices=["queko-bss-16qbt", "queko-bss-54qbt", "queko-bss-81qbt"], help="Benchmark name")
    parser.add_argument("--backend", type=str,
                        default="ibm_sherbrooke", choices=["ibm_sherbrooke", "ankaa"], help="Name of the backend")

    args = parser.parse_args()

    backend = args.backend
    benchmark = args.benchmark

    paths = {
        "sabre": fr"./baselines/results/{benchmark}/sabre.csv",
        "cirq": fr"./baselines/results/{benchmark}/cirq.csv",
        "qmap": fr"./baselines/results/{benchmark}/qmap.csv",
        "pytket": fr"./baselines/results/{benchmark}/pytket.csv",
        "qlosure": fr"./results/stats/{benchmark}_{backend}_trivial.csv"
    }

    grouped_data_swaps = {}
    grouped_data_depth = {}

    backend = "sherbrooke" if "sherbrooke" in args.backend else "ankaa_3"
    figure_idx = 6 if "sherbrooke" in args.backend else 7
    sub_figure_idx1 = "a"

    if benchmark == "queko-bss-16qbt":
        sub_figure_idx1 = "a"
    elif benchmark == "queko-bss-54qbt":
        sub_figure_idx1 = "b"
    elif benchmark == "queko-bss-81qbt":
        sub_figure_idx1 = "c"

    sub_figure_idx2 = "d"

    if benchmark == "queko-bss-16qbt":
        sub_figure_idx2 = "d"
    elif benchmark == "queko-bss-54qbt":
        sub_figure_idx2 = "e"
    elif benchmark == "queko-bss-81qbt":
        sub_figure_idx2 = "f"

    for name, path in paths.items():
        swaps, depth = group_by_initial_depth(path) if name == "qlosure" else group_by_initial_depth_baseline(
            path, backend=backend, init_mapping_method="trivial")
        grouped_data_swaps[name] = swaps
        grouped_data_depth[name] = depth

    swaps_fig_path = plot_grouped_scatter_with_noise(
        grouped_data_swaps, ylabel="Swaps", title=f"Figure {figure_idx}{sub_figure_idx1} SWAPs for {benchmark} on {backend}",)
    depth_fig_path = plot_grouped_scatter_with_noise(
        grouped_data_depth, ylabel="Depth", title=f"Figure {figure_idx}{sub_figure_idx2} Depth for {benchmark} on {backend}",)

    print(f"SWAPs figure saved to: {swaps_fig_path}")
    print(f"Depth figure saved to: {depth_fig_path}")
