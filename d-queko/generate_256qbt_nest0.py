"""
Generate 256-qubit d-QUEKO benchmarks with nest0 template.

Steps:
  1. Create GML device graph for 256 qubits (if not present).
  2. Call generate-d-queko.py for each leaf-depth (10..90), nest-depth=0.
"""

import subprocess
import sys
import os

import networkx as nx
import random

# ── Config ──────────────────────────────────────────────────────────────────
N_QUBITS = 256
LEAF_DEPTHS = list(range(10, 100, 10))       # 10, 20, ..., 90
REPLICATES = 10
TOP_LEN = 10
NEST_DEPTH = 0
OUTPUT_DIR = "nest0"                         # → benchmarks/nest0/256qbt/...
SEED = 1
GML_SEED = 42
GML_DENSITY = 0.3
# ────────────────────────────────────────────────────────────────────────────

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
GML_DIR = os.path.join(SCRIPT_DIR, "..", "qpu", "dqueko")
GML_FILE = os.path.join(GML_DIR, f"backend_{N_QUBITS}_qubits_seed_{GML_SEED}_density_{GML_DENSITY}.gml")
GENERATOR = os.path.join(SCRIPT_DIR, "generate-d-queko.py")


def ensure_gml():
    """Generate the 256-qubit GML device graph if it doesn't exist."""
    if os.path.isfile(GML_FILE):
        print(f"✔ GML device already exists: {GML_FILE}")
        return

    print(f"Generating {N_QUBITS}-qubit device graph (density={GML_DENSITY}, seed={GML_SEED}) ...")
    os.makedirs(GML_DIR, exist_ok=True)

    G = nx.Graph()
    G.add_nodes_from(range(N_QUBITS))
    rng = random.Random(GML_SEED)
    for i in range(N_QUBITS):
        for j in range(i + 1, N_QUBITS):
            if rng.random() < GML_DENSITY:
                G.add_edge(i, j)

    nx.write_gml(G, GML_FILE)
    print(f"✔ Saved GML: {GML_FILE}  ({G.number_of_nodes()} nodes, {G.number_of_edges()} edges)")


def run_benchmark(leaf_depth: int):
    """Invoke generate-d-queko.py for one leaf-depth."""
    cmd = [
        sys.executable, GENERATOR,
        f"--nest-depth={NEST_DEPTH}",
        f"--leaf-depth={leaf_depth}",
        f"--n-qubits={N_QUBITS}",
        f"--replicates={REPLICATES}",
        f"--top-len={TOP_LEN}",
        f"--output-dir={OUTPUT_DIR}",
        f"--seed={SEED}",
        "--emit-metadata",
    ]

    print(f"\n{'='*60}")
    print(f"  leaf-depth={leaf_depth}  ({REPLICATES} replicates)")
    print(f"  cmd: {' '.join(cmd)}")
    print(f"{'='*60}")

    result = subprocess.run(cmd, cwd=SCRIPT_DIR)
    if result.returncode != 0:
        print(f"✘ FAILED for leaf-depth={leaf_depth} (exit code {result.returncode})")
        return False
    print(f"✔ Done: leaf-depth={leaf_depth}")
    return True


def main():
    ensure_gml()

    ok, fail = 0, 0
    for ld in LEAF_DEPTHS:
        if run_benchmark(ld):
            ok += 1
        else:
            fail += 1

    print(f"\n{'='*60}")
    print(f"  Finished: {ok} succeeded, {fail} failed")
    print(f"  Output:   d-queko/benchmarks/{OUTPUT_DIR}/{N_QUBITS}qbt/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
