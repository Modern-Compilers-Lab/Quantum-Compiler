import argparse
import json
import os
import sys
import time
from pathlib import Path

# Add parent directory to path for shared qpu package
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from qiskit import qasm3

from src.routing import Qlosure
from src.dag import build_dag, extract_multi_qubit_dag
from src.evaluation import compute_max_swaps_count, compute_quantum_depth

from qpu.src.load_backend import load_backend_data
from src.backend import QuantumBackend
from src.results_utils import RESULTS_ROOT, D_QUEKO_BENCHMARKS_DIR, save_trace_results

from tqdm import tqdm



# Argument parser setup
parser = argparse.ArgumentParser(
    description="Run Qlosure with optional parameters")
parser.add_argument("--bench", type=str,
                    default="16qbt", help="Benchmark folder name (e.g., 16qbt, 54qbt)")
parser.add_argument("--backend", type=str,
                    default="ibm_brisbane_old", help="Name of the backend")
parser.add_argument("--initial", type=str, default="trivial",
                    help="Initial mapping method")
parser.add_argument("--verbose", type=int, default=0, help="Verbosity level")
parser.add_argument("--num_iterations", type=int, default=1,
                    help="number of bidirectional passes")
parser.add_argument("--template", type=str, default="nest0",
                    choices=["nest0", "nest1", "nest2", "if_else_inside_for"],
                    help="Template type for D-QuEKO circuits")

args = parser.parse_args()



d_queko_benchmarks_dir = D_QUEKO_BENCHMARKS_DIR / args.template / args.bench
results_root_dir = RESULTS_ROOT / "qlosure" / args.template / args.backend / args.bench


def find_qasm_files(bench_dir):
    """Find all .qasm files in the benchmark directory, organized by folder."""
    qasm_files_by_folder = {}

    for root, dirs, files in os.walk(bench_dir):
        # Sort directories to ensure consistent ordering
        dirs.sort()
        qasm_files = sorted([f for f in files if f.endswith('.qasm')])

        if qasm_files:
            relative_path = os.path.relpath(root, bench_dir)
            if relative_path == '.':
                relative_path = ''
            qasm_files_by_folder[relative_path] = [
                os.path.join(root, f) for f in qasm_files]

    return qasm_files_by_folder


def run_circuit(circuit_path, backend, initial_mapping, num_iterations, verbose):

    try:
        # Load circuit
        qc = qasm3.load(circuit_path)

        dag = build_dag(qc)
        dag2q = extract_multi_qubit_dag(dag)

        # Run Qlosure
        poly_mapper = Qlosure(backend)
        start = time.time()
        qlosure_results = poly_mapper.run(
            dag, dag2q, initial_mapping=initial_mapping, num_iter=num_iterations, verbose=verbose)
        qlosure_end_time = time.time()

        # Get results
        trace = poly_mapper.get_structured_trace()

        # Save results
        circuit_path_obj = Path(circuit_path)
        circuit_name = circuit_path_obj.stem
        relative_path = circuit_path_obj.relative_to(
            d_queko_benchmarks_dir)

        output_dir = results_root_dir / relative_path.parent / circuit_name
        save_trace_results(output_dir, trace, qlosure_end_time - start, circuit_path)

        return True, None, None

    except Exception as e:
        print(f"❌ Error processing {circuit_path}: {str(e)}")
        return False, None, None


# Main execution
bench_dir = d_queko_benchmarks_dir

if not bench_dir.exists():
    print(f"❌ Benchmark directory {bench_dir} does not exist!")
    exit(1)

print(f"Scanning benchmark directory: {bench_dir}")
qasm_files_by_folder = find_qasm_files(bench_dir)

if not qasm_files_by_folder:
    print(f"❌ No .qasm files found in {bench_dir}")
    exit(1)

# Load backend topology
print(f"Loading backend: {args.backend}")
backend_data = load_backend_data(args.backend)
edges = backend_data["coupling_map"]
qubits_props = backend_data.get("qubits", {})
backend = QuantumBackend(edges, qubit_props=qubits_props)
print("✅ Backend topology loaded.")

# Process circuits folder by folder
total_circuits = sum(len(files) for files in qasm_files_by_folder.values())
processed = 0
successful = 0
failed = 0

print(
    f"\nFound {total_circuits} circuits in {len(qasm_files_by_folder)} folders")

for folder_path in sorted(qasm_files_by_folder.keys()):
    circuits = qasm_files_by_folder[folder_path]
    folder_display = folder_path if folder_path else "root"

    # if "leaf-depth-40" not in folder_path:
    #     continue

    print(
        f"\n🗂️ Processing folder: {folder_display} ({len(circuits)} circuits)")

    for circuit_path in tqdm(circuits):
        processed += 1

        success, max_swaps, quant_depth = run_circuit(
            circuit_path, backend, args.initial, args.num_iterations, args.verbose > 0
        )

        if success:
            successful += 1
        else:
            failed += 1
    # break  # Remove this break to process all circuits

print(f"\n{'='*60}")
print(f"BENCHMARK COMPLETED")
print(f"{'='*60}")
print(f"Total circuits processed: {processed}")
print(f"Successful: {successful}")
print(f"Failed: {failed}")
print(f"Success rate: {successful/processed*100:.1f}%")
