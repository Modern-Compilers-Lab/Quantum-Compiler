import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, Any, List, Tuple

# Add parent directory to path for shared qpu package
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from qiskit import QuantumCircuit, QuantumRegister, qasm3
from qiskit.transpiler import CouplingMap, Layout, PassManager
from qiskit.transpiler.passes import SabreSwap

from qpu.src.load_backend import load_backend_data
from src.backend import QuantumBackend
from src.dag import build_dag, extract_multi_qubit_dag
from src.evaluation import (
    compute_max_swaps_count,
    compute_quantum_depth,
    estimate_dynamic_circuit,
)
from src.parser import build_structured_trace_from_circuit
from src.results_utils import D_QUEKO_BENCHMARKS_DIR, RESULTS_ROOT, save_trace_results
from src.routing import Qlosure


def resolve_circuit_path(circuit_arg: str) -> Path:
    circuit_path = Path(circuit_arg)
    if circuit_path.exists():
        return circuit_path.resolve()

    relative_candidate = D_QUEKO_BENCHMARKS_DIR / circuit_arg
    if relative_candidate.exists():
        return relative_candidate.resolve()

    raise FileNotFoundError(
        f"Circuit not found. Tried '{circuit_path}' and '{relative_candidate}'."
    )


def compute_metrics(
    trace: List[Dict[str, Any]],
    qubit_props: Dict[int, Dict[str, Any]],
    loop_iterations: int,
    use_physical_qubits: bool = True,
) -> Dict[str, Any]:
    metrics = {
        "swaps": compute_max_swaps_count(trace, loop_iterations=loop_iterations),
        "depth": compute_quantum_depth(
            trace,
            loop_iterations=loop_iterations,
            use_physical_qubits=use_physical_qubits,
        ),
        "latency": None,
        "error": None,
    }

    if qubit_props:
        timing = estimate_dynamic_circuit(
            trace,
            qubit_props=qubit_props,
            loop_iterations=loop_iterations,
            use_physical_qubits=use_physical_qubits,
        )
        metrics["latency"] = timing["max_time"]
        metrics["error"] = timing["max_error"]

    return metrics


def run_qlosure(
    qasm_file_path: Path,
    backend: QuantumBackend,
    qubit_props: Dict[int, Dict[str, Any]],
    initial_mapping: str,
    num_iterations: int,
    loop_iterations: int,
    verbose: bool,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any], float]:
    qc = qasm3.load(qasm_file_path)
    dag = build_dag(qc)
    dag2q = extract_multi_qubit_dag(dag)

    mapper = Qlosure(backend)
    start = time.time()
    mapper.run(
        dag,
        dag2q,
        initial_mapping=initial_mapping,
        num_iter=num_iterations,
        verbose=verbose,
    )
    elapsed = time.time() - start
    trace = mapper.get_structured_trace()
    metrics = compute_metrics(
        trace,
        qubit_props=qubit_props,
        loop_iterations=loop_iterations,
        use_physical_qubits=True,
    )
    return trace, metrics, elapsed


def run_sabre(
    qasm_file_path: Path,
    backend: QuantumBackend,
    qubit_props: Dict[int, Dict[str, Any]],
    loop_iterations: int,
    verbose: bool,
    sabre_seed: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any], float]:
    qc = qasm3.load(qasm_file_path)
    coupling_map = CouplingMap(backend.edges)

    num_physical_qubits = backend.num_qubits - 1
    if num_physical_qubits < len(qc.qubits):
        raise ValueError(
            f"Backend has {num_physical_qubits} qubits but circuit needs {len(qc.qubits)}."
        )

    qr = QuantumRegister(num_physical_qubits, "q")
    mapped_circuit = QuantumCircuit(qr, *qc.cregs, name=qc.name)
    mapped_circuit.global_phase = getattr(qc, "global_phase", 0)
    mapped_circuit.metadata = getattr(qc, "metadata", None)

    computed_layout = Layout({vq: i for i, vq in enumerate(qc.qubits)})
    for instruction in qc.data:
        instr = instruction.operation
        qargs = instruction.qubits
        cargs = instruction.clbits
        new_qargs = [qr[computed_layout[vq]] for vq in qargs]
        mapped_circuit.append(instr, new_qargs, cargs)

    pm = PassManager(
        [SabreSwap(coupling_map=coupling_map, heuristic="decay", seed=sabre_seed, trials=1)]
    )

    start = time.time()
    routed_qc = pm.run(mapped_circuit)
    elapsed = time.time() - start
    if verbose:
        print(f"SABRE wall time: {elapsed:.4f}s")

    trace = build_structured_trace_from_circuit(routed_qc, decompose=False)
    metrics = compute_metrics(
        trace,
        qubit_props=qubit_props,
        loop_iterations=loop_iterations,
        use_physical_qubits=True,
    )
    return trace, metrics, elapsed


def metric_delta(qlosure_value: Any, sabre_value: Any) -> str:
    if qlosure_value is None or sabre_value is None:
        return "n/a"
    delta = qlosure_value - sabre_value
    if isinstance(delta, float):
        return f"{delta:+.6f}"
    return f"{delta:+d}"


def format_value(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def print_comparison(qlosure_metrics: Dict[str, Any], sabre_metrics: Dict[str, Any]) -> None:
    headers = ("metric", "qlosure", "sabre", "qlosure-sabre")
    rows = []
    for metric in ("swaps", "depth", "latency", "error"):
        q_val = qlosure_metrics.get(metric)
        s_val = sabre_metrics.get(metric)
        rows.append(
            (
                metric,
                format_value(q_val),
                format_value(s_val),
                metric_delta(q_val, s_val),
            )
        )

    widths = [
        max(len(headers[idx]), max(len(row[idx]) for row in rows))
        for idx in range(len(headers))
    ]

    header_line = "  ".join(headers[idx].ljust(widths[idx]) for idx in range(len(headers)))
    separator = "  ".join("-" * widths[idx] for idx in range(len(headers)))
    print(header_line)
    print(separator)
    for row in rows:
        print("  ".join(row[idx].ljust(widths[idx]) for idx in range(len(row))))


def build_output_dir(method: str, backend_name: str, circuit_path: Path) -> Path:
    try:
        relative_circuit_path = circuit_path.resolve().relative_to(D_QUEKO_BENCHMARKS_DIR.resolve())
    except ValueError:
        relative_circuit_path = Path(circuit_path.name)

    return RESULTS_ROOT / "single-circuit-comparison" / method / backend_name / relative_circuit_path.parent / relative_circuit_path.stem


def save_comparison_results(
    circuit_path: Path,
    backend_name: str,
    qlosure_trace: List[Dict[str, Any]],
    qlosure_metrics: Dict[str, Any],
    qlosure_time: float,
    sabre_trace: List[Dict[str, Any]],
    sabre_metrics: Dict[str, Any],
    sabre_time: float,
) -> None:
    qlosure_dir = build_output_dir("qlosure", backend_name, circuit_path)
    sabre_dir = build_output_dir("sabre", backend_name, circuit_path)

    save_trace_results(qlosure_dir, qlosure_trace, qlosure_time, circuit_path)
    save_trace_results(sabre_dir, sabre_trace, sabre_time, circuit_path)

    with open(qlosure_dir / "metrics.json", "w", encoding="utf-8") as handle:
        json.dump(qlosure_metrics, handle, indent=2)
    with open(sabre_dir / "metrics.json", "w", encoding="utf-8") as handle:
        json.dump(sabre_metrics, handle, indent=2)

    comparison = {
        "circuit": str(circuit_path),
        "backend": backend_name,
        "qlosure": {**qlosure_metrics, "wall_time_s": qlosure_time},
        "sabre": {**sabre_metrics, "wall_time_s": sabre_time},
    }
    comparison_dir = RESULTS_ROOT / "single-circuit-comparison" / "summary" / backend_name
    comparison_dir.mkdir(parents=True, exist_ok=True)
    summary_file = comparison_dir / f"{circuit_path.stem}.json"
    with open(summary_file, "w", encoding="utf-8") as handle:
        json.dump(comparison, handle, indent=2)

    print(f"Saved Qlosure results to: {qlosure_dir}")
    print(f"Saved SABRE results to: {sabre_dir}")
    print(f"Saved comparison summary to: {summary_file}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Qlosure and SABRE on a single circuit and compare swaps, depth, latency, and error."
    )
    parser.add_argument(
        "--circuit",
        type=str,
        required=True,
        help="Path to a QASM circuit. Accepts an absolute path or a path relative to d-queko/benchmarks.",
    )
    parser.add_argument("--backend", type=str, default="ibm_brisbane_old")
    parser.add_argument("--initial", type=str, default="trivial")
    parser.add_argument("--num-iterations", type=int, default=1)
    parser.add_argument("--loop-iterations", type=int, default=10)
    parser.add_argument("--sabre-seed", type=int, default=42)
    parser.add_argument("--verbose", type=int, default=1)
    parser.add_argument(
        "--save-results",
        action="store_true",
        help="Persist traces, timing, and metrics under dynamic-qlosure/results. Disabled by default.",
    )
    args = parser.parse_args()

    circuit_path = resolve_circuit_path(args.circuit)
    print(f"Circuit: {circuit_path}")
    print(f"Backend: {args.backend}")

    backend_data = load_backend_data(args.backend)
    edges = backend_data["coupling_map"]
    qubit_props = backend_data.get("qubits", {})
    backend = QuantumBackend(edges, qubit_props=qubit_props)

    qlosure_trace, qlosure_metrics, qlosure_time = run_qlosure(
        qasm_file_path=circuit_path,
        backend=backend,
        qubit_props=qubit_props,
        initial_mapping=args.initial,
        num_iterations=args.num_iterations,
        loop_iterations=args.loop_iterations,
        verbose=bool(args.verbose),
    )
    sabre_trace, sabre_metrics, sabre_time = run_sabre(
        qasm_file_path=circuit_path,
        backend=backend,
        qubit_props=qubit_props,
        loop_iterations=args.loop_iterations,
        verbose=bool(args.verbose),
        sabre_seed=args.sabre_seed,
    )

    print()
    print_comparison(qlosure_metrics, sabre_metrics)
    print()
    print(f"Qlosure wall time: {qlosure_time:.6f}s")
    print(f"SABRE wall time:    {sabre_time:.6f}s")

    if args.save_results:
        save_comparison_results(
            circuit_path=circuit_path,
            backend_name=args.backend,
            qlosure_trace=qlosure_trace,
            qlosure_metrics=qlosure_metrics,
            qlosure_time=qlosure_time,
            sabre_trace=sabre_trace,
            sabre_metrics=sabre_metrics,
            sabre_time=sabre_time,
        )


if __name__ == "__main__":
    main()
