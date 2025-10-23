import csv
import re
from collections import defaultdict
from typing import Dict, List, Tuple
import os


def group_by_initial_depth(csv_path: str) -> Tuple[Dict[int, List[int]], Dict[int, List[int]]]:
    """
    Read the CSV at `csv_path` and return (swaps_grouped, depth_grouped).

    - swaps_grouped[init_depth] -> list of swap_count (ints)
    - depth_grouped[init_depth] -> list of final_depth (ints)
    """

    CYC_REGEX = re.compile(r'(\d+)\s*CYC', re.IGNORECASE)
    swaps_grouped: Dict[int, List[int]] = defaultdict(list)
    depth_grouped: Dict[int, List[int]] = defaultdict(list)

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        # Expect headers: filename,final_depth,swap_count,runtime,compiled_qasm_file
        # start=2 accounts for header line as 1
        for i, row in enumerate(reader, start=2):
            try:
                filename = (row.get("filename") or "").strip()
                if not filename:
                    # Skip rows without a filename
                    continue

                init_depth = int(CYC_REGEX.search(filename).group(1))

                # Parse numeric fields; coerce to int safely
                # Some CSVs may contain floats like "123.0"—handle gracefully
                def to_int(x):
                    if x is None or str(x).strip() == "":
                        return None
                    as_str = str(x).strip()
                    try:
                        return int(as_str)
                    except ValueError:
                        return int(float(as_str))

                final_depth = to_int(row.get("final_depth"))
                swap_count = to_int(row.get("swap_count"))

                if swap_count is not None:
                    swaps_grouped[init_depth].append(swap_count)
                if final_depth is not None:
                    depth_grouped[init_depth].append(final_depth)

            except Exception as e:
                # You can choose to log or collect errors if needed; for now we skip bad rows
                # print(f"Skipping row {i}: {e}")
                continue

    # Convert defaultdicts to plain dicts before returning
    return dict(swaps_grouped), dict(depth_grouped)


def group_by_initial_depth_baseline(
    csv_path: str, backend: str = "sherbrooke", init_mapping_method: str = "trivial"
) -> Tuple[Dict[int, List[int]], Dict[int, List[int]]]:
    """
    Read the CSV at `csv_path` and return (swaps_grouped, depth_grouped) 
    for the given backend and init_mapping_method.

    backend: "sherbrooke" or "ankaa_3"
    init_mapping_method: "trivial" or "default"

    - swaps_grouped[init_depth] -> list of swap counts
    - depth_grouped[init_depth] -> list of depths
    """

    CYC_REGEX = re.compile(r'(\d+)\s*CYC', re.IGNORECASE)

    swaps_grouped: Dict[int, List[int]] = defaultdict(list)
    depth_grouped: Dict[int, List[int]] = defaultdict(list)

    swap_col = f"{backend}_swaps_{init_mapping_method}"
    depth_col = f"{backend}_depth_{init_mapping_method}"

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            try:
                file_path = (row.get("file_path") or "").strip()
                if not file_path:
                    continue

                init_depth = int(CYC_REGEX.search(file_path).group(1))

                swap_val = row.get(swap_col)
                depth_val = row.get(depth_col)

                def to_int(x):
                    if x is None or str(x).strip() == "":
                        return None
                    as_str = str(x).strip()
                    try:
                        return int(as_str)
                    except ValueError:
                        return int(float(as_str))

                swap_val = to_int(swap_val)
                depth_val = to_int(depth_val)

                if swap_val is not None:
                    swaps_grouped[init_depth].append(swap_val)
                if depth_val is not None:
                    depth_grouped[init_depth].append(depth_val)

            except Exception as e:
                # Skip bad rows
                # print(f"Skipping row due to error: {e}")
                continue

    return dict(swaps_grouped), dict(depth_grouped)


def plot_grouped_scatter_with_noise(data_grouped, ylabel, title,
                                    font_size=21,
                                    marker_size=100, save=True,
                                    out_dir="plots",
                                    # optional explicit filename (without extension)
                                    fname=None,
                                    fmt="png",
                                    dpi=300,
                                    show=False):
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.ticker import FuncFormatter

    plt.figure(figsize=(10, 6))
    plt.rcParams.update({'font.size': font_size})

    # Fixed color and marker per method
    method_colors = {
        "sabre": "#1f77b4",    # blue
        "cirq": "#2ca02c",     # orange
        "qmap": "#ff7f0e",     # green
        "pytket": "#d62728",   # red
        "qlosure": "#9467bd",  # purple
    }

    method_markers = {
        "sabre": "o",
        "cirq": "s",
        "qmap": "^",
        "pytket": "*",
        "qlosure": "D",
    }

    for i, (method, depth_dict) in enumerate(data_grouped.items()):
        color = method_colors.get(method, "#333333")  # fallback to gray
        marker = method_markers.get(method, "o")      # fallback to circle

        all_depths = sorted(depth_dict.keys())
        all_vals = [depth_dict[d] for d in all_depths]

        x = []
        y = []
        for d, values in zip(all_depths, all_vals):
            x.extend([d] * len(values))
            y.extend(values)

        plt.scatter(x, y,
                    s=marker_size,
                    alpha=0.5,
                    color=color,
                    marker=marker,
                    label=method)

        # Mean and std band
        means = [np.mean(v) for v in all_vals]
        stds = [np.std(v) for v in all_vals]
        plt.plot(all_depths, means,
                 color=color,
                 linewidth=2)
        plt.fill_between(all_depths,
                         np.array(means) - np.array(stds),
                         np.array(means) + np.array(stds),
                         color=color,
                         alpha=0)

    # Formatter for y-axis (e.g., 1000 -> 1k)
    def thousands_formatter(x, pos):
        return f'{int(x/1000)}k' if x >= 1000 else f'{int(x)}'

    formatter = FuncFormatter(thousands_formatter)
    plt.gca().yaxis.set_major_formatter(formatter)

    # Axis labels and title
    plt.xlabel("Initial depth", fontsize=25)
    plt.ylabel(ylabel, fontsize=25)
    plt.title(title, fontsize=font_size + 2)

    # Styling
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=font_size)
    plt.xticks(fontsize=font_size)
    plt.yticks(fontsize=font_size)
    plt.tight_layout()

    saved_path = None
    if save:
        os.makedirs(out_dir, exist_ok=True)
        base = fname or title
        # sanitize: keep letters, numbers, _.- and replace other runs with _
        safe = re.sub(r'[^A-Za-z0-9_.-]+', '_', base.strip()).strip('_')
        ext = fmt.lstrip('.')
        saved_path = os.path.join(out_dir, f"{safe}.{ext}")
        plt.savefig(saved_path, dpi=dpi, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close()

    return saved_path
