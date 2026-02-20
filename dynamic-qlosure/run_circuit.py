import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, Tuple, Any

# Add parent directory to path for shared qpu package
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from qiskit import qasm3

from src.routing import Qlosure
from src.dag import build_dag, extract_multi_qubit_dag
from src.evaluation import compute_max_swaps_count, compute_quantum_depth, estimate_dynamic_circuit

from qpu.src.load_backend import load_backend_data
from src.backend import QuantumBackend
from src.results_utils import RESULTS_ROOT, D_QUEKO_BENCHMARKS_DIR, save_trace_results

from qiskit import QuantumCircuit, QuantumRegister

# Argument parser setup
parser = argparse.ArgumentParser(
    description="Run Qlosure with optional parameters")
parser.add_argument("--circuit", type=str,
                    default="16qbt/queko-016qbt_nest_01_nodes010_leaf-depth-10/circ_00.qasm", help="Path to circuit JSON file")
parser.add_argument("--backend", type=str,
                    default="ibm_brisbane", help="Name of the backend")
parser.add_argument("--initial", type=str, default="trivial",
                    help="Initial mapping method")
parser.add_argument("--verbose", type=int, default=1, help="Verbosity level")
parser.add_argument("--num_iterations", type=int, default=1,
                    help="number of bidirectional passes")

args = parser.parse_args()


# Load circuit data
print(f"Loading circuit from: {args.circuit}")
qasm_file_path = D_QUEKO_BENCHMARKS_DIR / args.circuit
print(f"Full path: {qasm_file_path.resolve()}")
start = time.time()
qc = qasm3.load(qasm_file_path)
end = time.time()
print(f"⏱️ Circuit loaded in {end - start:.2f} seconds.")

# Load backend topology via the centralized loader
print(f"Loading backend: {args.backend}")
backend_data = load_backend_data(args.backend)
edges = backend_data["coupling_map"]
qubits_props = backend_data.get("qubits", {})
print("✅ Backend topology loaded.")

print("Preparing data structures...")
start = time.time()
dag = build_dag(qc)
dag2q = extract_multi_qubit_dag(dag)
print(f"DAG built in {time.time() - start:.2f} seconds.")
# Run Qlosure
backend = QuantumBackend(edges, qubit_props=qubits_props)
poly_mapper = Qlosure(backend)

start = time.time()
qlosure_results = poly_mapper.run(
    dag, dag2q, initial_mapping=args.initial, num_iter=args.num_iterations, verbose=True)
qlosure_end_time = time.time()

print(f"⏱️ Qlosure run completed in {qlosure_end_time - start:.2f} seconds.")

# Get machine-friendly nested structure
trace = poly_mapper.get_structured_trace()

max_swaps = compute_max_swaps_count(trace, loop_iterations=10)
quant_depth = compute_quantum_depth(
    trace, loop_iterations=10, use_physical_qubits=True)

print(f"Max swaps in trace: {max_swaps}")
print(f"Quantum depth in trace: {quant_depth}")

if qubits_props:
    latency, error = estimate_dynamic_circuit(
        trace, qubit_props=qubits_props, loop_iterations=10).values()
    print(f"Latency in trace: {latency}")
    print(f"Error in trace: {error}")

# Save results
circuit_name = Path(args.circuit).stem
circuit_dir = str(Path(args.circuit).parent)
output_dir = RESULTS_ROOT / "qlosure" / args.backend / circuit_dir / circuit_name
save_trace_results(output_dir, trace, qlosure_end_time - start, qasm_file_path)

print(f"Results written to {output_dir}")
