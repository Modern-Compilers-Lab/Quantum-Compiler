"""
Surface Code Benchmark Generator for Quantum Routing Evaluation
================================================================

Generates OpenQASM 3 circuits implementing rotated Surface Code
quantum memory (repeated syndrome extraction with no logical ops).

This addresses the QEC evaluation gap: circuits have:
  - FOR loops (repeated syndrome extraction rounds)
  - Mid-circuit measurements + conditional corrections (IF blocks)
  - Heavy 2-qubit gate (CNOT) workloads on a local 2D patch
  - Scalable qubit counts via code distance parameter

Rotated Surface Code layout (distance d):
  - d² data qubits
  - (d²-1) / 2  X-stabilizers  (on "faces")
  - (d²-1) / 2  Z-stabilizers  (on "vertices")
  - Total ancilla = d² - 1
  - Total qubits  = 2d² - 1

Qubit counts by distance:
  d=3  →   17 qubits
  d=5  →   49 qubits
  d=7  →   97 qubits
  d=9  →  161 qubits
  d=11 →  241 qubits
  d=13 →  337 qubits

Usage:
    python generate_surface_code.py                      # all defaults
    python generate_surface_code.py --distances 3 5 7
    python generate_surface_code.py --rounds 5 10 15 20
    python generate_surface_code.py --distances 3 5 7 9 --rounds 3 5 10 15 20

Output:
    surface-code/benchmarks/d{d}/surface_code_d{d}_r{rounds}/circ_{rep:02d}.qasm
"""

import argparse
import json
import os
import random
import math
from typing import List, Tuple, Dict, Set

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BENCHMARKS_DIR = os.path.join(SCRIPT_DIR, "benchmarks")


# ─── Rotated Surface Code geometry ──────────────────────────────────────────

def build_rotated_surface_code(d: int):
    """
    Build the rotated surface code layout for distance d.

    Returns:
        data_qubits:   list of (row, col) for data qubits
        x_stabilizers: list of (ancilla_idx, [data neighbors])
        z_stabilizers: list of (ancilla_idx, [data neighbors])
        n_data:        number of data qubits
        n_ancilla:     number of ancilla qubits
    """
    # Data qubits on a d×d grid
    data_coords = []
    coord_to_idx = {}
    for r in range(d):
        for c in range(d):
            idx = r * d + c
            data_coords.append((r, c))
            coord_to_idx[(r, c)] = idx

    n_data = d * d

    # Stabilizers for rotated surface code
    # X-stabilizers are on "even" plaquettes, Z-stabilizers on "odd" plaquettes
    x_stabilizers = []
    z_stabilizers = []
    ancilla_idx = n_data  # ancilla IDs start after data qubits

    for r in range(d - 1):
        for c in range(d - 1):
            # Each plaquette touches 4 data qubits:
            # (r,c), (r,c+1), (r+1,c), (r+1,c+1)
            neighbors = [
                coord_to_idx[(r, c)],
                coord_to_idx[(r, c + 1)],
                coord_to_idx[(r + 1, c)],
                coord_to_idx[(r + 1, c + 1)],
            ]
            if (r + c) % 2 == 0:
                x_stabilizers.append((ancilla_idx, neighbors))
            else:
                z_stabilizers.append((ancilla_idx, neighbors))
            ancilla_idx += 1

    # Boundary stabilizers (weight-2)
    # Top boundary
    for c in range(0, d - 1, 2):
        neighbors = [coord_to_idx[(0, c)], coord_to_idx[(0, c + 1)]]
        z_stabilizers.append((ancilla_idx, neighbors))
        ancilla_idx += 1

    # Bottom boundary
    for c in range((d - 1) % 2, d - 1, 2):
        neighbors = [coord_to_idx[(d - 1, c)], coord_to_idx[(d - 1, c + 1)]]
        z_stabilizers.append((ancilla_idx, neighbors))
        ancilla_idx += 1

    # Left boundary
    for r in range(0, d - 1, 2):
        neighbors = [coord_to_idx[(r, 0)], coord_to_idx[(r + 1, 0)]]
        x_stabilizers.append((ancilla_idx, neighbors))
        ancilla_idx += 1

    # Right boundary
    for r in range((d - 1) % 2, d - 1, 2):
        neighbors = [coord_to_idx[(r, d - 1)], coord_to_idx[(r + 1, d - 1)]]
        x_stabilizers.append((ancilla_idx, neighbors))
        ancilla_idx += 1

    n_ancilla = ancilla_idx - n_data

    return data_coords, x_stabilizers, z_stabilizers, n_data, n_ancilla


def emit_surface_code_qasm(
    d: int,
    rounds: int,
    n_data: int,
    n_ancilla: int,
    x_stabilizers: List[Tuple[int, List[int]]],
    z_stabilizers: List[Tuple[int, List[int]]],
    seed: int = 42,
    with_corrections: bool = True,
) -> str:
    """
    Generate an OpenQASM 3 circuit for Surface Code quantum memory.

    The circuit structure:
        for round in [0:rounds] {
            // Reset ancillas
            // Z-stabilizer extraction (CNOT: data→ancilla)
            // X-stabilizer extraction (H, CNOT: ancilla→data, H)
            // Measure ancillas
            // Conditional corrections (if measurement != 0)
        }
    """
    total_qubits = n_data + n_ancilla
    n_cbits = n_ancilla  # one classical bit per ancilla measurement

    lines = []
    lines.append("OPENQASM 3;")
    lines.append('include "stdgates.inc";')
    lines.append(f"qubit[{total_qubits}] q;")
    lines.append(f"bit[{n_cbits}] c;")
    lines.append("")
    lines.append(f"// Surface Code d={d}, {rounds} syndrome extraction rounds")
    lines.append(f"// {n_data} data qubits + {n_ancilla} ancilla qubits = {total_qubits} total")
    lines.append(f"// X-stabilizers: {len(x_stabilizers)}, Z-stabilizers: {len(z_stabilizers)}")
    lines.append("")

    # Initialize data qubits in |+⟩ state (optional, for a mixed-basis memory)
    rng = random.Random(seed)
    init_qubits = rng.sample(range(n_data), n_data // 3)
    for qi in sorted(init_qubits):
        lines.append(f"h q[{qi}];")
    lines.append("")

    # Main syndrome extraction loop
    lines.append(f"for int round in [0:{rounds}] {{")

    # ─── Reset ancilla qubits ───
    lines.append("  // Reset ancilla qubits")
    for anc_idx, _ in z_stabilizers + x_stabilizers:
        local_idx = anc_idx
        lines.append(f"  reset q[{local_idx}];")
    lines.append("")

    # ─── Z-stabilizer syndrome extraction ───
    lines.append("  // Z-stabilizer syndrome extraction")
    cbit_counter = 0
    z_cbit_start = cbit_counter
    for anc_idx, data_neighbors in z_stabilizers:
        for data_q in data_neighbors:
            lines.append(f"  cx q[{data_q}], q[{anc_idx}];")
        cbit_counter += 1
    lines.append("")

    # ─── X-stabilizer syndrome extraction ───
    lines.append("  // X-stabilizer syndrome extraction")
    x_cbit_start = cbit_counter
    for anc_idx, data_neighbors in x_stabilizers:
        lines.append(f"  h q[{anc_idx}];")
        for data_q in data_neighbors:
            lines.append(f"  cx q[{anc_idx}], q[{data_q}];")
        lines.append(f"  h q[{anc_idx}];")
        cbit_counter += 1
    lines.append("")

    # ─── Measure all ancilla qubits ───
    lines.append("  // Measure syndrome ancillas")
    meas_idx = 0
    for anc_idx, _ in z_stabilizers:
        lines.append(f"  c[{meas_idx}] = measure q[{anc_idx}];")
        meas_idx += 1
    for anc_idx, _ in x_stabilizers:
        lines.append(f"  c[{meas_idx}] = measure q[{anc_idx}];")
        meas_idx += 1
    lines.append("")

    # ─── Conditional corrections based on syndrome ───
    if with_corrections:
        lines.append("  // Conditional corrections")
        # Z-syndrome → X corrections on data
        meas_idx = 0
        for anc_idx, data_neighbors in z_stabilizers:
            if data_neighbors:
                correction_target = data_neighbors[0]
                lines.append(f"  if (c[{meas_idx}]) {{")
                lines.append(f"    x q[{correction_target}];")
                lines.append(f"  }}")
            meas_idx += 1

        # X-syndrome → Z corrections on data
        for anc_idx, data_neighbors in x_stabilizers:
            if data_neighbors:
                correction_target = data_neighbors[0]
                lines.append(f"  if (c[{meas_idx}]) {{")
                lines.append(f"    z q[{correction_target}];")
                lines.append(f"  }}")
            meas_idx += 1
        lines.append("")

    lines.append("}")  # end for loop
    lines.append("")

    # Final data qubit measurement
    lines.append("// Final data qubit readout")
    for i in range(n_data):
        if i < n_cbits:
            lines.append(f"c[{i}] = measure q[{i}];")

    return "\n".join(lines)


def generate_benchmarks(
    distances: List[int],
    rounds_list: List[int],
    replicates: int,
    base_seed: int,
    with_corrections: bool,
):
    """Generate all surface code benchmark circuits."""
    os.makedirs(BENCHMARKS_DIR, exist_ok=True)

    summary = []

    for d in distances:
        data_coords, x_stab, z_stab, n_data, n_ancilla = build_rotated_surface_code(d)
        total_q = n_data + n_ancilla
        n_stab = len(x_stab) + len(z_stab)

        # CNOT count per round: sum of all stabilizer weights
        cnots_per_round = sum(len(nb) for _, nb in z_stab) + sum(len(nb) for _, nb in x_stab)
        # H gates per round: 2 * number of X-stabilizers
        h_per_round = 2 * len(x_stab)

        for rounds in rounds_list:
            bench_name = f"surface_code_d{d}_r{rounds}"
            bench_dir = os.path.join(BENCHMARKS_DIR, f"d{d}", bench_name)
            os.makedirs(bench_dir, exist_ok=True)

            total_cnots = cnots_per_round * rounds
            total_h = h_per_round * rounds
            total_corrections = n_stab * rounds

            print(f"  d={d:2d}, rounds={rounds:3d} → {total_q:4d} qubits, "
                  f"~{total_cnots:6d} CNOTs, ~{total_h} H gates, "
                  f"~{total_corrections} corrections/round")

            for rep in range(replicates):
                seed = base_seed + d * 1000 + rounds * 100 + rep
                qasm = emit_surface_code_qasm(
                    d=d,
                    rounds=rounds,
                    n_data=n_data,
                    n_ancilla=n_ancilla,
                    x_stabilizers=x_stab,
                    z_stabilizers=z_stab,
                    seed=seed,
                    with_corrections=with_corrections,
                )
                circ_path = os.path.join(bench_dir, f"circ_{rep:02d}.qasm")
                with open(circ_path, "w") as f:
                    f.write(qasm)

            # Save metadata
            meta = {
                "code_distance": d,
                "rounds": rounds,
                "n_data_qubits": n_data,
                "n_ancilla_qubits": n_ancilla,
                "total_qubits": total_q,
                "x_stabilizers": len(x_stab),
                "z_stabilizers": len(z_stab),
                "cnots_per_round": cnots_per_round,
                "h_gates_per_round": h_per_round,
                "total_cnots_unrolled": total_cnots,
                "corrections_per_round": n_stab,
                "with_corrections": with_corrections,
                "replicates": replicates,
                "base_seed": base_seed,
            }
            with open(os.path.join(bench_dir, "bench.json"), "w") as f:
                json.dump(meta, f, indent=2)

            summary.append(meta)

    # Summary table
    print(f"\n{'='*70}")
    print(f"  Generated {len(summary)} benchmark configs in {BENCHMARKS_DIR}")
    print(f"{'='*70}")
    print(f"  {'distance':>8s}  {'rounds':>6s}  {'qubits':>6s}  {'CNOTs/round':>11s}  {'total_gates':>11s}")
    print(f"  {'-'*8}  {'-'*6}  {'-'*6}  {'-'*11}  {'-'*11}")
    for m in summary:
        total_gates = m["total_cnots_unrolled"] + m["h_gates_per_round"] * m["rounds"]
        print(f"  {m['code_distance']:>8d}  {m['rounds']:>6d}  {m['total_qubits']:>6d}  "
              f"{m['cnots_per_round']:>11d}  {total_gates:>11d}")

    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Generate Surface Code quantum memory benchmarks (OpenQASM 3)")
    parser.add_argument("--distances", type=int, nargs="+", default=[3, 5, 7, 9],
                        help="Code distances to generate (default: 3 5 7 9)")
    parser.add_argument("--rounds", type=int, nargs="+", default=[3, 5, 10, 15, 20],
                        help="Syndrome extraction rounds (default: 3 5 10 15 20)")
    parser.add_argument("--replicates", type=int, default=3,
                        help="Circuit replicates per config (default: 3)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Base random seed (default: 42)")
    parser.add_argument("--no-corrections", action="store_true",
                        help="Omit conditional corrections (no IF blocks)")
    args = parser.parse_args()

    print(f"Generating Surface Code benchmarks...")
    print(f"  Distances: {args.distances}")
    print(f"  Rounds:    {args.rounds}")
    print(f"  Replicates: {args.replicates}")
    print(f"  Corrections: {not args.no_corrections}")
    print()

    generate_benchmarks(
        distances=args.distances,
        rounds_list=args.rounds,
        replicates=args.replicates,
        base_seed=args.seed,
        with_corrections=not args.no_corrections,
    )


if __name__ == "__main__":
    main()
