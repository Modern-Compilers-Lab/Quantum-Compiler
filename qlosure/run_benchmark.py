import argparse

from src.utils.isl_data_loader import json_file_to_isl
from src.mapping.routing import Qlosure
from qpu.src.load_backend import load_backend_edges
import os
import csv


parser = argparse.ArgumentParser(
    description="Run Qlosure with optional parameters")
parser.add_argument("--benchmark", type=str,
                    default="queko-bss-16qbt", help="Benchmark name")
parser.add_argument("--backend", type=str,
                    default="ibm_sherbrooke", help="Name of the backend")
parser.add_argument("--initial", type=str, default="trivial",
                    help="Initial mapping method")
parser.add_argument("--verbose", type=int, default=1, help="Verbosity level")
parser.add_argument("--heuristic", type=str, default="qlosure",
                    help="Heuristic to use for mapping")
parser.add_argument("--num_iterations", type=int, default=1,
                    help="number of bidirectional passes")


args = parser.parse_args()


benchmarks_folder_path = f"benchmarks/polyhedral/{args.benchmark}"

edges = load_backend_edges(args.backend)
all_files = os.listdir(benchmarks_folder_path)

results_dir = "results/stats"
os.makedirs(results_dir, exist_ok=True)

csv_filename = f"{args.benchmark}_{args.backend}_{args.initial}.csv"
csv_path = os.path.join(results_dir, csv_filename)

# Remove previous results CSV file if it exists
if os.path.exists(csv_path):
    os.remove(csv_path)

for file_idx, filename in enumerate(all_files):

    if filename.endswith('.json'):
        print(f"Processing file {file_idx+1}/{len(all_files)}: {filename}")
        file_path = os.path.join(benchmarks_folder_path, filename)
        # Process each JSON file

        data = json_file_to_isl(file_path)

        poly_mapper = Qlosure(edges, data)
        swap_count, depth, time = poly_mapper.run(initial_mapping_method=args.initial, verbose=args.verbose,
                                                  heuristic_method=args.heuristic, num_iter=args.num_iterations)

        # Write results to CSV
        with open(csv_path, 'a', newline='') as csvfile:
            fieldnames = ['filename', 'final_depth', 'swap_count', 'runtime']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            # Write header if file is empty
            if os.path.getsize(csv_path) == 0:
                writer.writeheader()

            # Write the results
            writer.writerow({
                'filename': filename,
                'final_depth': depth,
                'swap_count': swap_count,
                'runtime': time
            })
