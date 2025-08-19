import os
import time
import argparse
import csv
from qiskit.qasm2 import dump
from src.utils.isl_data_loader import json_file_to_isl
from qpu.src.load_backend import load_backend_edges
from baselines.sabre import run_sabre
from baselines.qmap import run_qmap
from baselines.pytket import run_pytket
from baselines.cirq import run_cirq

ALGORITHMS = ["sabre", "qmap", "pytket", "cirq"]


parser = argparse.ArgumentParser(
    description="Run Qlosure with optional parameters")
parser.add_argument("--benchmark", type=str,
                    default="queko-bss-16qbt", help="Benchmark name")
parser.add_argument("--backend", type=str,
                    default="ibm_sherbrooke", help="Name of the backend")
parser.add_argument("--verbose", type=int, default=1, help="Verbosity level")
parser.add_argument("--algorithm", type=str, choices=ALGORITHMS + ["all"], default="all",
                    help="Mapping algorithm to use (or 'all' to run all)")

args = parser.parse_args()

benchmarks_folder_path = f"benchmarks/polyhedral/{args.benchmark}"

edges = load_backend_edges(args.backend)
all_files = os.listdir(benchmarks_folder_path)

results_dir = "results/stats"
os.makedirs(results_dir, exist_ok=True)


def run_algorithm(algorithm, data, edges, layout):
    """Dispatch algorithm execution."""
    if algorithm == "sabre":
        return run_sabre(data, edges, layout=layout)
    elif algorithm == "qmap":
        return run_qmap(data, edges, initial_mapping=layout)
    elif algorithm == "pytket":
        return run_pytket(data, edges, initial_mapping=layout)
    elif algorithm == "cirq":
        return run_cirq(data, edges, initial_mapping=layout)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")


# Decide which algorithms to run
algorithms_to_run = ALGORITHMS if args.algorithm == "all" else [args.algorithm]

# Create a CSV file for each algorithm selected
csv_paths = {}
for algorithm in algorithms_to_run:
    csv_filename = f"{args.benchmark}_{args.backend}_{args.initial}_{algorithm}.csv"
    csv_path = os.path.join(results_dir, csv_filename)
    if os.path.exists(csv_path):
        os.remove(csv_path)
    csv_paths[algorithm] = csv_path

for file_idx, filename in enumerate(all_files):
    if filename.endswith('.json'):
        print(f"\nProcessing file {file_idx+1}/{len(all_files)}: {filename}")
        file_path = os.path.join(benchmarks_folder_path, filename)

        for algorithm in algorithms_to_run:
            print(f"  → Running algorithm: {algorithm}")

            final_depth = -1
            swap_count = -1
            execution_time = -1
            qasm_path = "N/A"

            try:
                data = json_file_to_isl(file_path)

                start_time = time.time()
                results = run_algorithm(
                    algorithm, data, edges, layout="trivial")
                execution_time = time.time() - start_time

                final_depth = results["depth"]
                swap_count = results["swaps"]

            except Exception as e:
                print(f"  ❌ Error with {algorithm} on {filename}: {e}")

            # Write results to the algorithm-specific CSV
            csv_path = csv_paths[algorithm]
            with open(csv_path, 'a', newline='') as csvfile:
                fieldnames = ['filename', 'final_depth',
                              'swap_count', 'runtime']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

                # Write header if file is empty
                if os.path.getsize(csv_path) == 0:
                    writer.writeheader()

                writer.writerow({
                    'filename': filename,
                    'final_depth': final_depth,
                    'swap_count': swap_count,
                    'runtime': execution_time,
                })
