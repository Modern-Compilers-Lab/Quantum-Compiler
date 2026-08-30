#!/usr/bin/env python3
"""Generate backend topology JSONs from MECH chiplet configurations.

Uses the MECH Chiplet module to build each lattice, then exports the
coupling map and per-qubit properties in the same JSON format used by
``qpu/topologies/*.json``.

Cross-chip edges receive a 7.4× penalty on two-qubit gate duration and
error (matching the MECH default ``cross_chip_gate_weight``).

Usage
-----
    cd MECH/
    python generate_mech_backends.py          # generate all four
    python generate_mech_backends.py --only heavy_hexagon
"""

import argparse
import builtins
import json
import sys
from pathlib import Path

import Chiplet
from Chiplet import (
    gen_chiplet_array,
    gen_highway_layout,
    gen_qubit_idx_dict,
)

# MECH monkey-patches (same as mech_qpu_report.py)
Chiplet.min = builtins.min
Chiplet.max = builtins.max
Chiplet.sum = builtins.sum

# ── Output directory ────────────────────────────────────────────────────────
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "qpu" / "topologies"

# ── Physical constants (from ibm_brisbane calibrations / MECH defaults) ────
SINGLE_QUBIT_LEN = 0.06       # μs  (≈ 60 ns SX gate)
SINGLE_QUBIT_ERR = 0.0003     # typical 1-qubit error
TWO_QUBIT_LEN    = 0.66       # μs  (≈ 660 ns ECR/CX gate)
TWO_QUBIT_ERR    = 0.001      # typical 2-qubit error
CROSS_CHIP_WEIGHT = 7.4       # latency & error multiplier for cross-chip


# ── Topology configurations (match the 4 MECH report files) ────────────────
CONFIGS = {
    "mech_heavy_hex_3x4_8x8": {
        "structure":  "heavy_hexagon",
        "array":      (3, 4),
        "chiplet":    (8, 8),
        "sparsity":   1,
    },
    "mech_heavy_square_3x3_8x8": {
        "structure":  "heavy_square",
        "array":      (3, 3),
        "chiplet":    (8, 8),
        "sparsity":   1,
    },
    "mech_hex_3x3_8x8": {
        "structure":  "hexagon",
        "array":      (3, 3),
        "chiplet":    (8, 8),
        "sparsity":   1,
    },
    "mech_square_3x3_7x7": {
        "structure":  "square",
        "array":      (3, 3),
        "chiplet":    (7, 7),
        "sparsity":   1,
    },
}


def build_backend_json(name: str, cfg: dict) -> dict:
    """Build a backend topology dict from a MECH configuration."""
    ax, ay = cfg["array"]
    cx, cy = cfg["chiplet"]
    structure = cfg["structure"]
    sparsity  = cfg["sparsity"]

    # 1. Build the MECH chiplet lattice
    chip = gen_chiplet_array(structure, ax, ay, cx, cy,
                             cross_link_sparsity=sparsity)
    gen_highway_layout(chip)

    # 2. Map (x,y) nodes → contiguous integer qubit indices
    node_to_idx = gen_qubit_idx_dict(chip)   # {(x,y): idx}
    idx_to_node = {v: k for k, v in node_to_idx.items()}
    num_qubits = len(chip.nodes)

    # 3. Build bidirectional coupling map
    coupling_map = []
    cross_chip_pairs = set()
    for u, v in chip.edges:
        iu, iv = node_to_idx[u], node_to_idx[v]
        coupling_map.append([iu, iv])
        coupling_map.append([iv, iu])
        if chip.edges[(u, v)].get("type") == "cross_chip":
            cross_chip_pairs.add((iu, iv))
            cross_chip_pairs.add((iv, iu))

    coupling_map.sort()

    # 4. Build per-qubit properties
    qubits = {}
    for idx in range(num_qubits):
        node = idx_to_node[idx]
        neighbors_in_coupling = []
        for u, v in chip.edges(node):
            other = v if u == node else u
            neighbors_in_coupling.append(node_to_idx[other])

        two_qubit_len = {}
        two_qubit_err = {}
        for nidx in neighbors_in_coupling:
            is_cross = (idx, nidx) in cross_chip_pairs
            scale = CROSS_CHIP_WEIGHT if is_cross else 1.0
            two_qubit_len[str(nidx)] = round(TWO_QUBIT_LEN * scale, 6)
            two_qubit_err[str(nidx)] = round(TWO_QUBIT_ERR * scale, 6)

        qubits[str(idx)] = {
            "T1": None,
            "T2": None,
            "single_qubit_len": SINGLE_QUBIT_LEN,
            "single_qubit_err": SINGLE_QUBIT_ERR,
            "two_qubit_len": two_qubit_len,
            "two_qubit_err": two_qubit_err,
        }

    # 5. Count stats for notes
    on_chip_edges = sum(1 for u, v in chip.edges
                        if chip.edges[(u, v)].get("type") != "cross_chip")
    cross_chip_edges = sum(1 for u, v in chip.edges
                           if chip.edges[(u, v)].get("type") == "cross_chip")

    return {
        "backend_name": name,
        "num_qubits": num_qubits,
        "coupling_map": coupling_map,
        "qubits": qubits,
        "notes": {
            "structure": structure,
            "array_dim": list(cfg["array"]),
            "chiplet_size": list(cfg["chiplet"]),
            "cross_link_sparsity": sparsity,
            "on_chip_edges": on_chip_edges,
            "cross_chip_edges": cross_chip_edges,
            "single_qubit_len_default": SINGLE_QUBIT_LEN,
            "two_qubit_len_default": TWO_QUBIT_LEN,
            "cross_chip_gate_weight": CROSS_CHIP_WEIGHT,
            "two_qubit_len_scaled_by_cross_chip_gate_weight": True,
        },
    }


def main():
    parser = argparse.ArgumentParser(
        description="Generate MECH chiplet backend JSONs for qpu/topologies/")
    parser.add_argument("--only", nargs="*", default=None,
                        help="Generate only the listed backend names (default: all)")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    targets = CONFIGS
    if args.only:
        targets = {k: v for k, v in CONFIGS.items() if k in args.only}
        if not targets:
            print(f"ERROR: none of {args.only} matched. Available: {list(CONFIGS.keys())}")
            sys.exit(1)

    for name, cfg in targets.items():
        print(f"Generating {name} ...", end=" ", flush=True)
        backend = build_backend_json(name, cfg)
        out_path = OUTPUT_DIR / f"{name}.json"
        with open(out_path, "w") as f:
            json.dump(backend, f, indent=2)
        n = backend["num_qubits"]
        edges = len(backend["coupling_map"]) // 2
        cross = backend["notes"]["cross_chip_edges"]
        print(f"{n} qubits, {edges} edges ({cross} cross-chip) → {out_path.name}")


if __name__ == "__main__":
    main()
