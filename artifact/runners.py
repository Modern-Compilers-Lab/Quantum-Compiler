"""Execute DynamiQ and Qiskit Sabre on a circuit and score the resulting trace."""

from __future__ import annotations

import json
import time
from pathlib import Path

from common import RESULTS_ROOT, ensure_dir

METRIC_KEYS = ("swaps", "depth", "latency", "error")

# Sec. 7.1: identity layout and sabre routing, so gate count and initial
# placement are identical for both mappers.
SABRE_HEURISTIC = "decay"
SABRE_TRIALS = 1


def load_backend(name):
    from qpu.src.load_backend import load_backend_data
    from src.backend import QuantumBackend

    data = load_backend_data(name)
    edges = data["coupling_map"]
    props = data.get("qubits", {})
    if not edges:
        raise SystemExit(f"backend {name!r} has no coupling_map")
    return QuantumBackend(edges, qubit_props=props), props, edges


def num_physical_qubits(edges, props):
    if props:
        return max(int(q) for q in props) + 1
    return max(max(a, b) for a, b in edges) + 1


def score_trace(trace, qubit_props, loop_iterations):
    from src.evaluation import (compute_max_swaps_count, compute_quantum_depth,
                                estimate_dynamic_circuit)

    timing = estimate_dynamic_circuit(trace, qubit_props, loop_iterations=loop_iterations)
    return {
        "swaps": compute_max_swaps_count(trace, loop_iterations=loop_iterations),
        "depth": compute_quantum_depth(trace, loop_iterations=loop_iterations),
        "latency": timing["max_time"],
        "error": timing["max_error"],
    }


def run_dynamiq(circuit_path, backend, seed, num_iterations=1, verbose=0, ablation=None):
    from qiskit import qasm3
    from src.dag import build_dag, extract_multi_qubit_dag
    from src.routing import Qlosure

    qc = qasm3.load(str(circuit_path))
    dag = build_dag(qc)
    dag2q = extract_multi_qubit_dag(dag)

    mapper = Qlosure(backend, seed=seed, ablation=ablation)
    t0 = time.time()
    mapper.run(dag, dag2q, initial_mapping="trivial", num_iter=num_iterations, verbose=verbose)
    elapsed = time.time() - t0
    return mapper.get_structured_trace(), elapsed


def run_sabre(circuit_path, coupling_map, n_phys, seed):
    from qiskit import QuantumCircuit, QuantumRegister, qasm3
    from qiskit.transpiler import Layout, PassManager
    from qiskit.transpiler.passes import SabreSwap
    from src.parser import build_structured_trace_from_circuit

    qc = qasm3.load(str(circuit_path))
    if len(qc.qubits) > n_phys:
        raise ValueError(f"circuit needs {len(qc.qubits)} qubits, backend has {n_phys}")

    qr = QuantumRegister(n_phys, "q")
    mapped = QuantumCircuit(qr, *qc.cregs, name=qc.name)
    mapped.global_phase = getattr(qc, "global_phase", 0)
    mapped.metadata = getattr(qc, "metadata", None)
    layout = Layout({vq: i for i, vq in enumerate(qc.qubits)})
    for instr, qargs, cargs in qc.data:
        mapped.append(instr, [qr[layout[vq]] for vq in qargs], cargs)

    pm = PassManager([SabreSwap(coupling_map=coupling_map, heuristic=SABRE_HEURISTIC,
                                seed=seed, trials=SABRE_TRIALS)])
    t0 = time.time()
    routed = pm.run(mapped)
    elapsed = time.time() - t0
    return build_structured_trace_from_circuit(routed, decompose=False), elapsed


def trace_dir(method, *parts):
    return RESULTS_ROOT / method / Path(*[str(p) for p in parts])


def save_trace(out_dir, trace, elapsed, circuit_path):
    out_dir = ensure_dir(Path(out_dir))
    (out_dir / "trace.json").write_text(json.dumps(trace, separators=(",", ":")))
    (out_dir / "time.txt").write_text(f"{elapsed:.6f}\n")
    (out_dir / "path.txt").write_text(str(circuit_path))


def execute_one(job):
    """Run one (method, circuit, seed) job. Returns (ok, message)."""
    out_dir = Path(job["out_dir"])
    if not job.get("force") and (out_dir / "trace.json").exists():
        return True, "cached"
    try:
        if job["method"] == "sabre":
            from qiskit.transpiler import CouplingMap
            _, props, edges = load_backend(job["backend"])
            trace, elapsed = run_sabre(job["circuit"], CouplingMap(edges),
                                       num_physical_qubits(edges, props), job["seed"])
        else:
            backend, _, _ = load_backend(job["backend"])
            trace, elapsed = run_dynamiq(job["circuit"], backend, job["seed"],
                                         job.get("num_iterations", 1),
                                         ablation=job.get("ablation"))
        save_trace(out_dir, trace, elapsed, job["circuit"])
        return True, f"{elapsed:.2f}s"
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"
