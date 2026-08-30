"""
Comparison benchmark: 3 transitive-closure variants on D-QuEKO nest0/54qbt.

Variant 1 (default)  : original compute_transitive_closure_bitset
Variant 2 (unrolled) : loop-unrolling then bitset closure
Variant 3 (no_deps)  : deps_count all zeroed (with_closure_depth=False effect)

Runs only leaf depths 10, 20, 30, 40, 50 (50 circuits total).
Saves traces + metrics under results/deps-ablation/{variant}/...
"""

import argparse
import copy
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

# Add parent directory to path for shared qpu package
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from qiskit import qasm3

from src.routing import Qlosure
from src.dag import build_dag, extract_multi_qubit_dag
from src.evaluation import (
    compute_max_swaps_count,
    compute_quantum_depth,
    estimate_dynamic_circuit,
)
from src.graph import (
    compute_transitive_closure_bitset,
    compute_transitive_closure_bitset_unrolled,
)

from qpu.src.load_backend import load_backend_data
from src.backend import QuantumBackend
from src.results_utils import D_QUEKO_BENCHMARKS_DIR

from tqdm import tqdm

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="3-variant comparison benchmark")
parser.add_argument("--bench", type=str, default="81qbt")
parser.add_argument("--template", type=str, default="nest0")
parser.add_argument("--backend", type=str, default="ibm_brisbane_old")
parser.add_argument("--num_iterations", type=int, default=1)
parser.add_argument("--loop_iterations", type=int, default=10,
                    help="Assumed loop iterations for evaluation metrics")
parser.add_argument("--verbose", type=int, default=0)
args = parser.parse_args()

# Only these leaf depths
LEAF_DEPTHS = [10, 20, 30, 40, 50]
LEAF_DEPTHS = [90]
VARIANTS = ["default", "unrolled", "no_deps"]

# Output root
RESULTS_DIR = Path(__file__).parent / "results" / "deps-ablation"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def find_qasm_files_by_leaf_depth(bench_dir, leaf_depths):
    """Return {leaf_depth: [list of .qasm paths]} for the requested depths."""
    result = {}
    for ld in leaf_depths:
        # Match folder names like queko-054qbt_nest_00_nodes010_leaf-depth-10
        pattern = f"leaf-depth-{ld}"
        for entry in sorted(bench_dir.iterdir()):
            if entry.is_dir() and pattern in entry.name:
                qasm_files = sorted(entry.glob("circ_00.qasm"))
                if qasm_files:
                    result[ld] = [str(f) for f in qasm_files]
                break
    return result


def run_single_variant(
    qc, dag, dag2q, backend_obj, qubit_props,
    variant_name, num_iterations, loop_iterations, verbose,
):
    """
    Run one variant and return (metrics_dict, trace).
    """
    successors2q = dag2q["successors"]
    predecessors2q = dag2q["predecessors"]
    node_data_2q = dag2q["node_data"]

    # ---- Compute dependency counts depending on variant ----
    if variant_name == "default":
        fwd_deps = compute_transitive_closure_bitset(successors2q, predecessors2q)
    elif variant_name == "unrolled":
        fwd_deps = compute_transitive_closure_bitset_unrolled(
            successors2q, predecessors2q,
            node_data=node_data_2q,
            default_while_iterations=10,
        )
    elif variant_name == "no_deps":
        max_id = max(max(successors2q.keys(), default=0),
                     max(predecessors2q.keys(), default=0))
        fwd_deps = [0] * (max_id + 1)
    else:
        raise ValueError(f"Unknown variant: {variant_name}")

    if num_iterations > 1:
        if variant_name == "default":
            bwd_deps = compute_transitive_closure_bitset(predecessors2q, successors2q)
        elif variant_name == "unrolled":
            bwd_deps = compute_transitive_closure_bitset_unrolled(
                predecessors2q, successors2q,
                node_data=node_data_2q,
                default_while_iterations=10,
            )
        else:
            bwd_deps = fwd_deps
    else:
        bwd_deps = fwd_deps

    # ---- Subclass Qlosure to inject our pre-computed deps ----
    class _PatchedQlosure(Qlosure):
        def run(self, dag_, dag2q_, heuristic_method="Qlosure",
                initial_mapping_method="trivial", initial_mapping=None,
                num_iter=1, param=5, decay_max_reset=None, verbose=0):

            self.init_mapping(method=initial_mapping_method,
                              initial_mapping=initial_mapping)
            self.results = {}
            self.swap_history = []
            min_swaps = float("inf")
            min_depth = float("inf")

            self.dag = dag_
            self.dag2q = dag2q_

            s2q = dag2q_["successors"]
            p2q = dag2q_["predecessors"]
            sf  = dag_["successors"]
            pf  = dag_["predecessors"]

            self.node_data = dag_["node_data"]

            dag_forward_dependencies_count = fwd_deps
            dag_backward_dependencies_count = bwd_deps if num_iter > 1 else fwd_deps

            initial_mapping = None
            for i in range(2 * (num_iter - 1) + 1):
                if i % 2 == 0:
                    self.dag_dependencies_count = dag_forward_dependencies_count
                    self.dag_successors2q = s2q
                    self.dag_predecessors2q = p2q
                    self.dag_successors_full = sf
                    self.dag_predecessors_full = (
                        copy.deepcopy(pf) if num_iter > 1 else pf
                    )
                    initial_mapping = copy.deepcopy(self.mapping_dict)
                else:
                    self.dag_dependencies_count = dag_backward_dependencies_count
                    self.dag_successors2q = p2q
                    self.dag_predecessors2q = s2q
                    self.dag_successors_full = pf
                    self.dag_predecessors_full = (
                        copy.deepcopy(sf) if num_iter > 1 else sf
                    )

                self.init_front_layer()
                self.qubit_depth = {q: 0 for q in range(self.num_qubits)}
                swap_count = self.execute_algorithm(
                    heuristic_method, param, decay_max_reset, verbose
                )

                if i % 2 == 0:
                    if swap_count < min_swaps:
                        min_swaps = min(min_swaps, swap_count)
                        min_depth = min(min_depth, self.get_circuit_depth())
                    elif swap_count == min_swaps:
                        min_depth = min(min_depth, self.get_circuit_depth())
            return min_swaps, min_depth, initial_mapping

    # ---- Run ----
    mapper = _PatchedQlosure(backend_obj)
    t0 = time.time()
    swap_count, circ_depth, _ = mapper.run(
        copy.deepcopy(dag), copy.deepcopy(dag2q),
        num_iter=num_iterations, verbose=verbose,
    )
    elapsed = time.time() - t0

    # ---- Evaluate trace ----
    trace = mapper.get_structured_trace()
    swaps = compute_max_swaps_count(trace, loop_iterations=loop_iterations)
    depth = compute_quantum_depth(trace, loop_iterations=loop_iterations)

    timing = estimate_dynamic_circuit(
        trace, qubit_props,
        loop_iterations=loop_iterations,
        use_physical_qubits=True,
    )

    metrics = {
        "swaps": swaps,
        "depth": depth,
        "latency": timing["max_time"],
        "error_rate": timing["max_error"],
        "time_s": elapsed,
    }

    return metrics, trace


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

bench_dir = D_QUEKO_BENCHMARKS_DIR / args.template / args.bench
if not bench_dir.exists():
    print(f"❌  Benchmark dir not found: {bench_dir}")
    sys.exit(1)

print(f"Loading backend: {args.backend}")
backend_data = load_backend_data(args.backend)
edges = backend_data["coupling_map"]
qubit_props = backend_data.get("qubits", {})
backend_obj = QuantumBackend(edges, qubit_props=qubit_props)
print("✅ Backend loaded.\n")

qasm_by_depth = find_qasm_files_by_leaf_depth(bench_dir, LEAF_DEPTHS)
total_circuits = sum(len(v) for v in qasm_by_depth.values())
print(f"Leaf depths: {LEAF_DEPTHS}")
print(f"Total circuits: {total_circuits}\n")

if total_circuits == 0:
    print("❌ No circuits found!")
    sys.exit(1)

# ---- Run all variants ----
for ld in LEAF_DEPTHS:
    circuits = qasm_by_depth.get(ld, [])
    if not circuits:
        print(f"⚠️  No circuits for leaf-depth-{ld}, skipping.")
        continue

    print(f"\n{'='*60}")
    print(f"  Leaf depth {ld}  ({len(circuits)} circuits)")
    print(f"{'='*60}")

    for circ_path in tqdm(circuits, desc=f"  LD-{ld}"):
        circ_name = Path(circ_path).stem
        # Get the folder name (e.g. queko-054qbt_nest_00_nodes010_leaf-depth-10)
        folder_name = Path(circ_path).parent.name

        try:
            qc = qasm3.load(circ_path)
            dag = build_dag(qc)
            dag2q = extract_multi_qubit_dag(dag)
        except Exception as e:
            print(f"\n❌  Skipping {circ_name}: {e}")
            continue

        for vname in VARIANTS:
            try:
                metrics, trace = run_single_variant(
                    qc, dag, dag2q, backend_obj, qubit_props,
                    variant_name=vname,
                    num_iterations=args.num_iterations,
                    loop_iterations=args.loop_iterations,
                    verbose=args.verbose,
                )

                # Save under results/deps-ablation/{variant}/{folder}/{circuit}/
                out_dir = RESULTS_DIR / vname / folder_name / circ_name
                out_dir.mkdir(parents=True, exist_ok=True)

                with open(out_dir / "trace.json", "w") as f:
                    json.dump(trace, f, indent=2)
                with open(out_dir / "metrics.json", "w") as f:
                    json.dump(metrics, f, indent=2)
                (out_dir / "time.txt").write_text(f"{metrics['time_s']:.6f}\n")
                (out_dir / "path.txt").write_text(circ_path)

            except Exception as e:
                print(f"\n❌  {vname}/{circ_name}: {e}")

print(f"\n✅ All results saved to {RESULTS_DIR}")
print(f"   Run 'python print_comparison_summary.py' to view the report.")
