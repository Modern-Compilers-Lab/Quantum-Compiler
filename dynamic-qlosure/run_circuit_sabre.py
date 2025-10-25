import argparse
import json
import time
from pathlib import Path

from qiskit import QuantumCircuit, QuantumRegister
from qiskit import qasm3
from qiskit.transpiler import PassManager, Layout, CouplingMap
from qiskit.transpiler.passes import SabreSwap

from qpu.src.load_backend import load_backend_edges
from src.backend import QuantumBackend
from src.parser import build_structured_trace_from_circuit, format_structured_trace
from src.evaluation import (
    compute_max_swaps_count,
    compute_quantum_depth,
)

# Root directory for D-QUeKO benchmarks
D_QUEKO_BENCHMARKS_DIR = "../d-queko/benchmarks/"


def route_single_circuit(qasm_path: Path, backend: QuantumBackend, verbose: bool = True):
    """Load a QASM3 circuit, map it trivially to the backend, run SABRE swap, and
    return (trace, max_swaps, quant_depth, routed_qc)."""
    # Load circuit
    start = time.time()
    qc = qasm3.load(qasm_path)
    load_secs = time.time() - start
    if verbose:
        print(f"⏱️ Circuit loaded in {load_secs:.2f} seconds.")

    # Build coupling map from backend
    coupling_map = CouplingMap(backend.edges)

    # NOTE: Some topologies include an unused index; match prior code's -1
    num_physical_qubits = backend.num_qubits - 1
    print(f"Backend has {num_physical_qubits} physical qubits.")
    if num_physical_qubits < len(qc.qubits):
        raise ValueError(
            f"Backend has {num_physical_qubits} qubits but circuit needs {len(qc.qubits)}."
        )

    # Build a PHYSICAL circuit on a full-size device register
    qr = QuantumRegister(num_physical_qubits, "q")
    mapped_circuit = QuantumCircuit(qr, *qc.cregs, name=qc.name)
    mapped_circuit.global_phase = getattr(qc, "global_phase", 0)
    mapped_circuit.metadata = getattr(qc, "metadata", None)

    # Trivial explicit layout: vq_i -> physical p_i
    computed_layout = Layout({vq: i for i, vq in enumerate(qc.qubits)})

    # Remap each instruction's qubit args to the physical register
    for instr, qargs, cargs in qc.data:
        new_qargs = [qr[computed_layout[vq]] for vq in qargs]
        mapped_circuit.append(instr, new_qargs, cargs)

    # Run SABRE on the physical circuit
    pm = PassManager([
        SabreSwap(coupling_map=coupling_map,
                  heuristic="decay", seed=42, trials=1)
    ])

    start = time.time()
    routed_qc = pm.run(mapped_circuit)
    sabre_secs = time.time() - start
    if verbose:
        print(f"⏱️ SABRE completed in {sabre_secs:.2f} seconds.")

    # Build structured trace & metrics
    trace = build_structured_trace_from_circuit(routed_qc, decompose=False)
    max_swaps = compute_max_swaps_count(trace, loop_iterations=100)
    quant_depth = compute_quantum_depth(
        trace, loop_iterations=100, use_physical_qubits=True)

    return trace, max_swaps, quant_depth, routed_qc


def main():
    parser = argparse.ArgumentParser(
        description="Route a single circuit with SABRE and emit metrics")
    parser.add_argument(
        "--circuit",
        type=str,
        default="16qbt/queko-016qbt_nest_01_nodes010_leaf-depth-10/circ_00.qasm",
        help="Path to circuit relative to D-QUeKO benchmarks root",
    )
    parser.add_argument("--backend", type=str,
                        default="ibm_sherbrooke", help="Backend name for topology")
    parser.add_argument("--verbose", type=int, default=1,
                        help="Verbosity (0/1)")
    args = parser.parse_args()

    # Resolve circuit path
    qasm_file_path = Path(D_QUEKO_BENCHMARKS_DIR) / args.circuit
    print(f"Loading circuit from: {args.circuit}")
    print(f"Full path: {qasm_file_path.resolve()}")

    # Load backend
    print(f"Loading backend: {args.backend}")
    edges = load_backend_edges(args.backend)
    backend = QuantumBackend(edges)
    print("✅ Backend topology loaded.")

    try:
        trace, max_swaps, quant_depth, _ = route_single_circuit(
            qasm_file_path, backend, verbose=bool(args.verbose))
    except Exception as e:
        print(f"❌ Error while routing: {e}")
        return

    # Report metrics
    print(f"Max swaps in trace: {max_swaps}")
    print(f"Quantum depth in trace: {quant_depth}")

    # Write results next to a deterministic folder mirroring circuit's location
    circuit_path_obj = Path(args.circuit)
    circuit_name = circuit_path_obj.stem
    relative_parent = circuit_path_obj.parent

    output_dir = Path("results_sabre") / relative_parent / circuit_name
    output_dir.mkdir(parents=True, exist_ok=True)

    trace_json_path = output_dir / f"{circuit_name}_trace.json"
    trace_txt_path = output_dir / f"{circuit_name}_trace.txt"
    with open(trace_json_path, "w") as f:
        json.dump(trace, f, indent=2)

    with open(trace_txt_path, "w") as f:
        f.write(format_structured_trace(trace))
        f.write("\n")

    print(f"Trace JSON written to: {trace_json_path}")


if __name__ == "__main__":
    main()
