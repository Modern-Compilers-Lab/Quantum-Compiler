import os
import csv
import re
import statistics
import argparse
from collections import defaultdict

results_dir = "results/stats"

ALGORITHMS = ["sabre", "qmap", "pytket", "cirq", "qlosure"]


def extract_cycles(filename):
    """Extract cycle count (CYC) from benchmark filename."""
    match = re.search(r'_(\d+)CYC_', filename)
    if match:
        return int(match.group(1))
    return None


def load_csv(csv_path):
    """Load a CSV file and return rows with filename, depth, swaps, runtime, cycles."""
    rows = []
    with open(csv_path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            filename = row['filename']
            runtime = float(row['runtime'])
            depth = int(row['final_depth'])
            swaps = int(row['swap_count'])
            cycles = extract_cycles(filename)

            if cycles is None or runtime < 0 or depth < 0 or swaps < 0:
                continue

            size = "medium" if cycles < 500 else "large" if cycles > 600 else None
            if size:
                rows.append({
                    "filename": filename,
                    "cycles": cycles,
                    "depth": depth,
                    "swaps": swaps,
                    "runtime": runtime,
                    "size": size
                })
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze Qlosure results")
    parser.add_argument("--benchmark", type=str,
                        default="queko-bss-16qbt", help="Benchmark name")
    parser.add_argument("--backend", type=str,
                        default="ibm_sherbrooke", help="Name of the backend")

    args = parser.parse_args()
    print(f"\n\nShowing results for {args.benchmark} on {args.backend}")

    # Load all CSVs per algorithm
    data = {algo: [] for algo in ALGORITHMS}
    for algorithm in ALGORITHMS:
        csv_filename = f"{args.benchmark}_{args.backend}_trivial_{algorithm}.csv" if algorithm != "qlosure" else f"{args.benchmark}_{args.backend}_trivial.csv"
        csv_path = os.path.join(results_dir, csv_filename)

        if not os.path.exists(csv_path):
            print(f"\n⚠️ No results found for {algorithm}")
            continue

        data[algorithm] = load_csv(csv_path)

    # --- Compute averages ---
    for algorithm in ALGORITHMS:
        if not data[algorithm]:
            continue

        # runtime averages
        runtimes = defaultdict(list)
        depth_factors = defaultdict(list)
        swaps = defaultdict(list)

        for row in data[algorithm]:
            size = row["size"]
            runtimes[size].append(row["runtime"])
            depth_factors[size].append(row["depth"] / row["cycles"])
            swaps[size].append(row["swaps"])

        print(f"\n🚀 === Algorithm: {algorithm.upper()} ===")

        # Depth factor
        print("________________________________________________________")

        print(
            "    📏 Depth Factor (post-mapping depth / optimal depth): [Table 2]")
        for size in ["medium", "large"]:
            if depth_factors[size]:
                print(
                    f"       {size.capitalize()} Depth factor: {statistics.mean(depth_factors[size]):.4f}")
            else:
                print(f"       {size.capitalize()} Depth factor: No data")

        # Swap ratio (only for non-qlosure algorithms)
        if algorithm != "qlosure" and data["qlosure"]:
            print("________________________________________________________")
            print(
                "    🔄 SWAP ratio (Existing mapping method SWAPs / Qlosure SWAPs): [Table 3]")
            swap_ratios = defaultdict(list)
            # index qlosure swaps by filename
            qlosure_map = {row["filename"]: row["swaps"]
                           for row in data["qlosure"]}

            for row in data[algorithm]:
                if row["filename"] in qlosure_map and qlosure_map[row["filename"]] > 0:
                    ratio = row["swaps"] / qlosure_map[row["filename"]]
                    swap_ratios[row["size"]].append(ratio)

            for size in ["medium", "large"]:
                if swap_ratios[size]:
                    print(
                        f"       {size.capitalize()} Swap Ratio (Algo/Qlosure): {statistics.mean(swap_ratios[size]):.4f}")
                else:
                    print(f"       {size.capitalize()} Swap Ratio: No data")
        elif algorithm == "qlosure":
            pass

        # Runtime
        print("________________________________________________________")
        print("    ⏱️ Runtime (sec): [Table 4]")
        for size in ["medium", "large"]:
            if runtimes[size]:
                print(
                    f"        -> {size.capitalize()} : {statistics.mean(runtimes[size]):.4f} sec")
            else:
                print(f"        -> {size.capitalize()} : No data")

        print("________________________________________________________")
