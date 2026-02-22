"""
Run SABRE routing on Surface Code benchmarks.

Usage:
    python run_surface_code_sabre.py --backend=ibm_kingston
    python run_surface_code_sabre.py --backend=ibm_kingston --distances 3 5 7
    python run_surface_code_sabre.py --backend=ibm_flamingo --distances 9 11
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Add parent and root directories to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "dynamic-qlosure"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from qiskit import QuantumCircuit, QuantumRegister
from qiskit import qasm3
from qiskit.transpiler import PassManager, Layout, CouplingMap
from qiskit.transpiler.passes import SabreSwap
from tqdm import tqdm

from src.parser import build_structured_trace_from_circuit
from src.evaluation import compute_max_swaps_count, compute_quantum_depth, estimate_dynamic_circuit

from qpu.src.load_backend import load_backend_data
from src.backend import QuantumBackend
from src.results_utils import RESULTS_ROOT, save_trace_results

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BENCHMARKS_DIR = os.path.join(SCRIPT_DIR, "benchmarks")

SEEDS = [3, 21, 42]


def infer_num_qubits(edges, qubits_props):
    if qubits_props:
        return len(qubits_props)
    if not edges:
        return 0
    return max(max(u, v) for u, v in edges) + 1


def run_circuit(circuit_path, output_dir, coupling_map, num_phys_qubits, seed=42, verbose=0):
    """Route a single circuit with SABRE and save results."""
    try:
        qc = qasm3.load(circuit_path)

        if num_phys_qubits < len(qc.qubits):
            raise ValueError(
                f"Backend has {num_phys_qubits} qubits but circuit needs {len(qc.qubits)}.")

        # Build physical circuit on full-size device register
        qr = QuantumRegister(num_phys_qubits, "q")
        mapped_circuit = QuantumCircuit(qr, *qc.cregs, name=qc.name)
        mapped_circuit.global_phase = getattr(qc, "global_phase", 0)
        mapped_circuit.metadata = getattr(qc, "metadata", None)

        # Trivial layout: vq_i → physical p_i
        computed_layout = Layout({vq: i for i, vq in enumerate(qc.qubits)})
        for instr, qargs, cargs in qc.data:
            new_qargs = [qr[computed_layout[vq]] for vq in qargs]
            mapped_circuit.append(instr, new_qargs, cargs)

        pm = PassManager([
            SabreSwap(coupling_map=coupling_map, heuristic="decay", seed=seed, trials=1)
        ])

        start = time.time()
        routed_qc = pm.run(mapped_circuit)
        elapsed = time.time() - start

        trace = build_structured_trace_from_circuit(routed_qc, decompose=False)
        save_trace_results(output_dir, trace, elapsed, circuit_path)
        return True, elapsed
    except Exception as e:
        print(f"  ✘ Error: {e}")
        return False, 0.0


def main():
    parser = argparse.ArgumentParser(description="Run SABRE on Surface Code benchmarks")
    parser.add_argument("--backend", type=str, default="ibm_kingston",
                        help="Backend topology name (default: ibm_kingston)")
    parser.add_argument("--distances", type=int, nargs="+", default=None,
                        help="Code distances to run (default: auto-detect from benchmarks/)")
    parser.add_argument("--rounds", type=int, nargs="+", default=None,
                        help="Rounds to run (default: auto-detect)")
    parser.add_argument("--verbose", type=int, default=0, help="Verbosity level")
    parser.add_argument("--benchmarks-dir", type=str, default=None,
                        help="Benchmarks directory (default: benchmarks/)")
    parser.add_argument("--results-tag", type=str, default="surface_code",
                        help="Results sub-folder tag (default: surface_code)")
    args = parser.parse_args()

    if args.benchmarks_dir:
        global BENCHMARKS_DIR
        BENCHMARKS_DIR = os.path.abspath(args.benchmarks_dir)

    # Load backend
    print(f"Loading backend: {args.backend}")
    backend_data = load_backend_data(args.backend)
    edges = backend_data["coupling_map"]
    qubits_props = backend_data.get("qubits", {})
    num_phys_qubits = infer_num_qubits(edges, qubits_props)
    coupling_map = CouplingMap(edges)
    print(f"  Backend has {num_phys_qubits} qubits")

    # Discover benchmarks
    if not os.path.isdir(BENCHMARKS_DIR):
        print(f"✘ No benchmarks found at {BENCHMARKS_DIR}. Run generate_surface_code.py first.")
        sys.exit(1)

    results_base = RESULTS_ROOT / "sabre" / args.results_tag / args.backend

    total_ok, total_fail = 0, 0

    for d_dir in sorted(os.listdir(BENCHMARKS_DIR)):
        if not d_dir.startswith("d"):
            continue
        d = int(d_dir[1:])
        if args.distances and d not in args.distances:
            continue

        d_path = os.path.join(BENCHMARKS_DIR, d_dir)
        for bench_dir_name in sorted(os.listdir(d_path)):
            if not bench_dir_name.startswith("surface_code_"):
                continue

            parts = bench_dir_name.split("_")
            r_val = int(parts[-1][1:])
            if args.rounds and r_val not in args.rounds:
                continue

            bench_path = os.path.join(d_path, bench_dir_name)

            # Check qubit count
            meta_path = os.path.join(bench_path, "bench.json")
            if os.path.isfile(meta_path):
                with open(meta_path) as f:
                    meta = json.load(f)
                total_q = meta["total_qubits"]
                if total_q >= num_phys_qubits:
                    print(f"  [SKIP] d={d} r={r_val}: needs {total_q} qubits but backend has {num_phys_qubits}")
                    continue

            circ_files = sorted([f for f in os.listdir(bench_path) if f.endswith(".qasm")])
            if not circ_files:
                continue

            print(f"\n--- d={d}, rounds={r_val} ({len(circ_files)} circuits × {len(SEEDS)} seeds) ---")

            for circ_file in circ_files:
                circ_path = os.path.join(bench_path, circ_file)
                circ_name = circ_file.replace(".qasm", "")

                for seed in tqdm(SEEDS, desc=f"  {circ_file}", leave=False):
                    out_dir = results_base / f"d{d}" / bench_dir_name / circ_name / f"SEED_{seed}"
                    ok, elapsed = run_circuit(circ_path, out_dir, coupling_map, num_phys_qubits, seed=seed, verbose=args.verbose)
                    if ok:
                        total_ok += 1
                    else:
                        total_fail += 1

    print(f"\n{'='*60}")
    print(f"  SABRE done: {total_ok} succeeded, {total_fail} failed")
    print(f"  Results: {results_base}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
