import argparse
import json
import os
import sys
import time
from pathlib import Path

# Add parent directory to path for shared qpu package
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from qiskit import qasm3
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.transpiler import PassManager, Layout, CouplingMap
from qiskit.transpiler.passes import SabreSwap

from qpu.src.load_backend import load_backend_data
from src.backend import QuantumBackend
from src.parser import build_structured_trace_from_circuit, format_structured_trace
from src.results_utils import RESULTS_ROOT, D_QUEKO_BENCHMARKS_DIR, save_trace_results

from tqdm import tqdm


# Argument parser setup
parser = argparse.ArgumentParser(
    description="Run Qlosure with optional parameters")
parser.add_argument("--bench", type=str,
                    default="16qbt", help="Benchmark folder name (e.g., 16qbt, 54qbt)")
parser.add_argument("--backend", type=str,
                    default="ibm_kingston", help="Name of the backend")
parser.add_argument("--initial", type=str, default="trivial",
                    help="Initial mapping method")
parser.add_argument("--verbose", type=int, default=0, help="Verbosity level")
parser.add_argument("--num_iterations", type=int, default=1,
                    help="number of bidirectional passes")
parser.add_argument("--leaf_depth", type=int, default=10, help="Leaf depth to filter benchmarks")     
parser.add_argument("--template", type=str, default="nest0",
                    choices=["nest0", "nest1", "if_else_inside_for","nest2"],
                    help="Template type for D-QuEKO circuits")

args = parser.parse_args()

d_queko_benchmarks_dir = D_QUEKO_BENCHMARKS_DIR / "wi_rule_benchmarks" / args.bench / f"{args.leaf_depth}Leaf_depth"
results_root_dir = RESULTS_ROOT / "sabre" / "wi_rule" / args.backend / args.bench / f"{args.leaf_depth}Leaf_depth"

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


        # 5) Now run SABRE on the PHYSICAL circuit (allowed in your Terra)
        pm = PassManager([
            SabreSwap(coupling_map=coupling_map, heuristic="decay",
                      seed=42, trials=1)
        ])

        start = time.time()
        routed_qc = pm.run(mapped_circuit)
        elapsed = time.time() - start
        trace = build_structured_trace_from_circuit(
            routed_qc, decompose=False)


        # Save results
        circuit_path_obj = Path(circuit_path)
        circuit_name = circuit_path_obj.stem
        relative_path = circuit_path_obj.relative_to(
            d_queko_benchmarks_dir)

        # add circuit parent
        output_dir = results_root_dir / relative_path.parent / circuit_name
        save_trace_results(output_dir, trace, elapsed, circuit_path)

        return True, None, None

    except Exception as e:
        print(f"❌ Error processing {circuit_path}: {str(e)}")
        print(e)
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
