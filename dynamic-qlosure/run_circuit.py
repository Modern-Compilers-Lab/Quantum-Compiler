import argparse
import json
import logging
import os
import time
from pathlib import Path
from typing import Dict, Tuple, Any

from qiskit.qasm2 import dump
from qiskit import qasm3

from src.routing import Qlosure
from src.dag import build_dag, extract_multi_qubit_dag
from src.evaluation import compute_max_swaps_count, compute_quantum_depth, estimate_dynamic_circuit

from qpu.src.load_backend import load_backend_edges
from src.backend import QuantumBackend

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister

from qiskit.circuit.controlflow import IfElseOp


d_queko_benchmarks_dir = "../d-queko/benchmarks/"

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
qasm_file_path = Path(d_queko_benchmarks_dir) / args.circuit
print(f"Full path: {qasm_file_path.resolve()}")
start = time.time()
qc = qasm3.load(qasm_file_path)
end = time.time()
print(f"⏱️ Circuit loaded in {end - start:.2f} seconds.")

# Load backend edges
print(f"Loading backend: {args.backend}")
edges = load_backend_edges(args.backend)

with open("/scratch/mb10324/Quantum-Compiler/d-queko/qpu/topologies/ibm_brisbane_old.json", 'r', encoding="utf-8") as fp:
    ibm_brisbane_topology = json.load(fp)

qubits_props = ibm_brisbane_topology.get("qubits", {})
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
latency, error = estimate_dynamic_circuit(trace, qubit_props=ibm_brisbane_topology["qubits"], loop_iterations=10).values()


print(f"Max swaps in trace: {max_swaps}")
print(f"Quantum depth in trace: {quant_depth}")
print(f"Latency in trace: {latency}")
print(f"Error in trace: {error}")

# # Create output directory based on circuit name
# circuit_name = Path(args.circuit).stem
# circuit_dir = "/".join(args.circuit.split('/')[:-1])
# print(circuit_dir)
# output_dir = Path("results_new") / circuit_dir / circuit_name
# output_dir.mkdir(parents=True, exist_ok=True)

# # Save trace files with meaningful names
# trace_txt_path = output_dir / f"{circuit_name}_trace.txt"
# trace_json_path = output_dir / f"{circuit_name}_trace.json"

# with open(trace_txt_path, "w") as f:
#     f.write(poly_mapper.format_structured_trace(trace))
#     f.write("\n")

# with open(trace_json_path, "w") as f:
#     json.dump(trace, f, indent=2)

# print(f"Trace files written to {output_dir}/")

# print("Trace written to trace.txt")
