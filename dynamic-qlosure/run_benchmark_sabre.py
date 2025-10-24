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
from src.evaluation import compute_max_swaps_count, compute_structural_depth, compute_quantum_depth

from qpu.src.load_backend import load_backend_edges
from src.backend import QuantumBackend

from tqdm import tqdm

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import SabreSwap, SabreLayout, EnlargeWithAncilla, FullAncillaAllocation
from qiskit import transpile
from qiskit import qasm3
from qiskit.circuit.controlflow import IfElseOp
from qiskit.transpiler import PassManager, Layout, CouplingMap
from qiskit.transpiler.passes import SetLayout, ApplyLayout, SabreSwap

from qpu.src.load_backend import load_backend_edges
from src.parser import build_structured_trace_from_circuit, format_structured_trace

d_queko_benchmarks_dir = "../d-queko/benchmarks/"

# Argument parser setup
parser = argparse.ArgumentParser(
    description="Run Qlosure with optional parameters")
parser.add_argument("--bench", type=str,
                    default="16qbt", help="Benchmark folder name (e.g., 16qbt, 54qbt)")
parser.add_argument("--backend", type=str,
                    default="ibm_sherbrooke", help="Name of the backend")
parser.add_argument("--initial", type=str, default="trivial",
                    help="Initial mapping method")
parser.add_argument("--verbose", type=int, default=0, help="Verbosity level")
parser.add_argument("--num_iterations", type=int, default=1,
                    help="number of bidirectional passes")

args = parser.parse_args()


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

        coupling_map = CouplingMap(backend.edges)
        num_physical_qubits = backend.num_qubits - 1
        print(f"Backend has {num_physical_qubits} physical qubits.")
        if num_physical_qubits < len(qc.qubits):
            raise ValueError(
                f"Backend has {num_physical_qubits} qubits but circuit needs {len(qc.qubits)}."
            )

        # 3) Trivial *explicit* mapping: vq_i -> physical p_i
        #    Build a PHYSICAL circuit on a full-size device register
        qr = QuantumRegister(num_physical_qubits, "q")
        mapped_circuit = QuantumCircuit(qr, *qc.cregs, name=qc.name)
        mapped_circuit.global_phase = getattr(qc, "global_phase", 0)
        mapped_circuit.metadata = getattr(qc, "metadata", None)

        #    Create the mapping {virtual_qubit_obj -> physical_index}
        computed_layout = Layout({vq: i for i, vq in enumerate(qc.qubits)})

        # 4) Manually remap each instruction's qubit args onto the physical register
        for instr, qargs, cargs in qc.data:
            new_qargs = [qr[computed_layout[vq]] for vq in qargs]
            mapped_circuit.append(instr, new_qargs, cargs)

        # (Optional) If you have calibrations in qc, bring them over:
        # for gate, qubits, calib in qc.calibrations.items(): ...
        # mapped_circuit.calibrations = qc.calibrations

        # 5) Now run SABRE on the PHYSICAL circuit (allowed in your Terra)
        pm = PassManager([
            SabreSwap(coupling_map=coupling_map, heuristic="decay",
                      seed=42, trials=1)
        ])

        routed_qc = pm.run(mapped_circuit)
        trace = build_structured_trace_from_circuit(
            routed_qc, decompose=False)

        max_swaps = compute_max_swaps_count(trace, loop_iterations=100)
        quant_depth = compute_quantum_depth(
            trace, loop_iterations=100, use_physical_qubits=True)

        # Save results
        circuit_path_obj = Path(circuit_path)
        circuit_name = circuit_path_obj.stem
        relative_path = circuit_path_obj.relative_to(
            Path(d_queko_benchmarks_dir))

        output_dir = Path("results_sabre") / \
            relative_path.parent / circuit_name
        output_dir.mkdir(parents=True, exist_ok=True)

        # trace_txt_path = output_dir / f"{circuit_name}_trace.txt"
        trace_json_path = output_dir / f"{circuit_name}_trace.json"

        with open(trace_json_path, "w") as f:
            json.dump(trace, f, indent=2)

        # with open(trace_txt_path, "w") as f:
        #     f.write(format_structured_trace(trace))
        #     f.write("\n")

        return True, max_swaps, quant_depth

    except Exception as e:
        print(f"❌ Error processing {circuit_path}: {str(e)}")
        print(e)
        return False, None, None


# Main execution
bench_dir = Path(d_queko_benchmarks_dir) / args.bench

if not bench_dir.exists():
    print(f"❌ Benchmark directory {bench_dir} does not exist!")
    exit(1)

print(f"Scanning benchmark directory: {bench_dir}")
qasm_files_by_folder = find_qasm_files(bench_dir)

if not qasm_files_by_folder:
    print(f"❌ No .qasm files found in {bench_dir}")
    exit(1)

# Load backend edges
print(f"Loading backend: {args.backend}")
edges = load_backend_edges(args.backend)
backend = QuantumBackend(edges)
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
        break  # Remove this break to process all circuits

print(f"\n{'='*60}")
print(f"BENCHMARK COMPLETED")
print(f"{'='*60}")
print(f"Total circuits processed: {processed}")
print(f"Successful: {successful}")
print(f"Failed: {failed}")
print(f"Success rate: {successful/processed*100:.1f}%")
