"""
Build an IBM Flamingo topology: 3 × IBM Kingston chips connected via
inter-chip links with penalised latency and error rates.

Usage:
    python build_flamingo.py                    # default 7.4× penalty
    python build_flamingo.py --penalty 10.0     # custom penalty factor

Output:
    ../qpu/topologies/ibm_flamingo.json

The inter-chip connections are placed between degree-1 (leaf) qubits of
adjacent chips, mimicking IBM's Flamingo modular architecture.
"""

import argparse
import copy
import json
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TOPOLOGIES_DIR = os.path.join(SCRIPT_DIR, "..", "qpu", "topologies")
KINGSTON_PATH = os.path.join(TOPOLOGIES_DIR, "ibm_kingston.json")
OUTPUT_PATH = os.path.join(TOPOLOGIES_DIR, "ibm_flamingo.json")

# Inter-chip links: connect leaf qubits of adjacent chips.
# Kingston has 10 degree-1 (leaf) qubits:
#   [0, 20, 40, 55, 59, 60, 80, 100, 120, 140]
# We split them into two groups for "left" and "right" connectors:
#   left  connectors: [0, 20, 40, 55, 59]   (connect to previous chip)
#   right connectors: [60, 80, 100, 120, 140] (connect to next chip)
# Chip 0 right <-> Chip 1 left, Chip 1 right <-> Chip 2 left
LEFT_CONNECTORS  = [0, 20, 40, 55, 59]
RIGHT_CONNECTORS = [60, 80, 100, 120, 140]

NUM_CHIPS = 3


def load_kingston():
    with open(KINGSTON_PATH, "r") as f:
        return json.load(f)


def build_flamingo(penalty_factor: float):
    """
    Build a 3-chip Flamingo topology from Kingston.

    Returns a topology dict with:
      - backend_name
      - num_qubits
      - coupling_map  (all intra + inter chip edges)
      - qubits        (qubit properties; inter-chip links get penalised values)
      - inter_chip_edges (list of inter-chip [q1, q2] for reference)
      - penalty_factor
    """
    kingston = load_kingston()
    k_coupling = kingston["coupling_map"]
    k_qubits = kingston["qubits"]
    n_k = len(k_qubits)  # 156

    # ── Find max latency and max error in Kingston ──────────────────────
    max_2q_len = 0.0
    max_2q_err = 0.0
    for qid, props in k_qubits.items():
        for neighbor, length in props.get("two_qubit_len", {}).items():
            max_2q_len = max(max_2q_len, length)
        for neighbor, err in props.get("two_qubit_err", {}).items():
            max_2q_err = max(max_2q_err, err)

    inter_chip_len = max_2q_len * penalty_factor
    inter_chip_err = max_2q_err * penalty_factor

    print(f"Kingston: {n_k} qubits, {len(k_coupling)} edges")
    print(f"Max 2q length : {max_2q_len}")
    print(f"Max 2q error  : {max_2q_err}")
    print(f"Penalty factor: {penalty_factor}×")
    print(f"Inter-chip len: {inter_chip_len}")
    print(f"Inter-chip err: {inter_chip_err}")

    # ── Replicate qubits and coupling maps for each chip ────────────────
    all_coupling = []
    all_qubits = {}

    for chip in range(NUM_CHIPS):
        offset = chip * n_k
        # Coupling map: shift qubit IDs
        for edge in k_coupling:
            all_coupling.append([edge[0] + offset, edge[1] + offset])
        # Qubit properties: shift IDs
        for qid_str, props in k_qubits.items():
            new_id = int(qid_str) + offset
            new_props = {
                "T1": props["T1"],
                "T2": props["T2"],
                "single_qubit_len": props["single_qubit_len"],
                "single_qubit_err": props["single_qubit_err"],
                "two_qubit_len": {str(int(k) + offset): v for k, v in props.get("two_qubit_len", {}).items()},
                "two_qubit_err": {str(int(k) + offset): v for k, v in props.get("two_qubit_err", {}).items()},
            }
            all_qubits[str(new_id)] = new_props

    # ── Add inter-chip connections ──────────────────────────────────────
    inter_chip_edges = []

    for chip_pair in range(NUM_CHIPS - 1):  # 0-1, 1-2
        chip_a = chip_pair
        chip_b = chip_pair + 1
        offset_a = chip_a * n_k
        offset_b = chip_b * n_k

        # Connect chip_a's RIGHT connectors to chip_b's LEFT connectors
        for r_conn, l_conn in zip(RIGHT_CONNECTORS, LEFT_CONNECTORS):
            qa = r_conn + offset_a
            qb = l_conn + offset_b

            # Add edge (bidirectional in coupling map)
            all_coupling.append([qa, qb])
            all_coupling.append([qb, qa])
            inter_chip_edges.append([qa, qb])

            # Update qubit properties for inter-chip link
            all_qubits[str(qa)]["two_qubit_len"][str(qb)] = inter_chip_len
            all_qubits[str(qa)]["two_qubit_err"][str(qb)] = inter_chip_err
            all_qubits[str(qb)]["two_qubit_len"][str(qa)] = inter_chip_len
            all_qubits[str(qb)]["two_qubit_err"][str(qa)] = inter_chip_err

    total_qubits = NUM_CHIPS * n_k
    print(f"\nFlamingo: {total_qubits} qubits, {len(all_coupling)} edges")
    print(f"Inter-chip links: {len(inter_chip_edges)} bidirectional pairs")

    topology = {
        "backend_name": "ibm_flamingo",
        "num_qubits": total_qubits,
        "coupling_map": all_coupling,
        "qubits": all_qubits,
        "inter_chip_edges": inter_chip_edges,
        "penalty_factor": penalty_factor,
    }
    return topology


def main():
    parser = argparse.ArgumentParser(description="Build IBM Flamingo topology (3 × Kingston)")
    parser.add_argument("--penalty", type=float, default=7.4,
                        help="Penalty multiplier for inter-chip latency and error (default: 7.4)")
    parser.add_argument("--output", type=str, default=OUTPUT_PATH,
                        help=f"Output JSON path (default: {OUTPUT_PATH})")
    args = parser.parse_args()

    topology = build_flamingo(args.penalty)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(topology, f, indent=4)
    print(f"\n✅ Saved: {args.output}")


if __name__ == "__main__":
    main()
