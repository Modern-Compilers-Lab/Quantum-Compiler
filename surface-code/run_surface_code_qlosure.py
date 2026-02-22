"""
Run Qlosure routing on Surface Code benchmarks.

Usage:
    python run_surface_code_qlosure.py --backend=ibm_kingston
    python run_surface_code_qlosure.py --backend=ibm_kingston --distances 3 5 7
    python run_surface_code_qlosure.py --backend=ibm_flamingo --distances 9 11
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

from qiskit import qasm3
from tqdm import tqdm

from src.routing import Qlosure
from src.dag import build_dag, extract_multi_qubit_dag
from src.evaluation import compute_max_swaps_count, compute_quantum_depth, estimate_dynamic_circuit

from qpu.src.load_backend import load_backend_data
from src.backend import QuantumBackend
from src.results_utils import RESULTS_ROOT, save_trace_results

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BENCHMARKS_DIR = os.path.join(SCRIPT_DIR, "benchmarks")

SEEDS = [3, 21, 42]


def run_circuit(circuit_path, output_dir, backend, seed=42, verbose=0):
    """Route a single circuit with Qlosure and save results."""
    try:
        qc = qasm3.load(circuit_path)

        dag = build_dag(qc)
        dag2q = extract_multi_qubit_dag(dag)

        poly_mapper = Qlosure(backend, seed=seed)
        start = time.time()
        poly_mapper.run(dag, dag2q, initial_mapping="trivial", num_iter=1, verbose=verbose)
        elapsed = time.time() - start

        trace = poly_mapper.get_structured_trace()
        save_trace_results(output_dir, trace, elapsed, circuit_path)
        return True, elapsed
    except Exception as e:
        print(f"  ✘ Error: {e}")
        return False, 0.0


def main():
    parser = argparse.ArgumentParser(description="Run Qlosure on Surface Code benchmarks")
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
    backend = QuantumBackend(edges, qubit_props=qubits_props)
    backend_qubits = backend.num_qubits
    print(f"  Backend has {backend_qubits} qubits")

    # Discover benchmarks
    if not os.path.isdir(BENCHMARKS_DIR):
        print(f"✘ No benchmarks found at {BENCHMARKS_DIR}. Run generate_surface_code.py first.")
        sys.exit(1)

    results_base = RESULTS_ROOT / "qlosure" / args.results_tag / args.backend

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

            # Parse rounds from dir name: surface_code_d{d}_r{rounds}
            parts = bench_dir_name.split("_")
            r_val = int(parts[-1][1:])  # r{rounds}
            if args.rounds and r_val not in args.rounds:
                continue

            bench_path = os.path.join(d_path, bench_dir_name)

            # Load metadata to check qubit count
            meta_path = os.path.join(bench_path, "bench.json")
            if os.path.isfile(meta_path):
                with open(meta_path) as f:
                    meta = json.load(f)
                total_q = meta["total_qubits"]
                if total_q >= backend_qubits:
                    print(f"  [SKIP] d={d} r={r_val}: needs {total_q} qubits but backend has {backend_qubits}")
                    continue

            # Find circuit files
            circ_files = sorted([f for f in os.listdir(bench_path) if f.endswith(".qasm")])
            if not circ_files:
                continue

            print(f"\n--- d={d}, rounds={r_val} ({len(circ_files)} circuits × {len(SEEDS)} seeds) ---")

            for circ_file in circ_files:
                circ_path = os.path.join(bench_path, circ_file)
                circ_name = circ_file.replace(".qasm", "")

                for seed in tqdm(SEEDS, desc=f"  {circ_file}", leave=False):
                    out_dir = results_base / f"d{d}" / bench_dir_name / circ_name / f"SEED_{seed}"
                    if (out_dir / "trace.json").exists():
                        total_ok += 1
                        continue
                    ok, elapsed = run_circuit(circ_path, out_dir, backend, seed=seed, verbose=args.verbose)
                    if ok:
                        total_ok += 1
                    else:
                        total_fail += 1

    print(f"\n{'='*60}")
    print(f"  Qlosure done: {total_ok} succeeded, {total_fail} failed")
    print(f"  Results: {results_base}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
