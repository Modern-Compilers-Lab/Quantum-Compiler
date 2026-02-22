"""
Surface Code Benchmark Generator for Quantum Routing Evaluation
================================================================

Uses **Stim** (https://github.com/quantumlib/Stim) to produce
the canonical rotated-surface-code memory-Z experiment, then
converts the resulting circuit to OpenQASM 3 for Qiskit-based
mapping / routing tools.

Why Stim?
  Stim's `Circuit.generated("surface_code:rotated_memory_z", ...)`
  is a widely-used, well-tested reference implementation.  Using it
  guarantees correct:
    - stabilizer decomposition (X vs Z plaquettes, boundaries)
    - 4-step CX schedule (minimises hook-error weight)
    - qubit coordinate assignment

What this script adds on top of Stim:
  - Sparse -> contiguous qubit index remapping
  - OpenQASM 3 emission with FOR loops and IF-based corrections
  - Replicate generation with distinct random seeds

Rotated Surface Code qubit counts (2d^2 - 1):
  d=3  ->   17 qubits        d=7  ->   97 qubits
  d=5  ->   49 qubits        d=9  ->  161 qubits

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
from typing import Dict, List, Tuple, Set

import stim

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BENCHMARKS_DIR = os.path.join(SCRIPT_DIR, "benchmarks")


# --- Stim -> internal representation ----------------------------------------

def _extract_surface_code_schedule(d: int, rounds: int = 2):
    """
    Use Stim to generate a rotated surface-code memory-Z experiment
    and extract the gate schedule, qubit roles, and CX pairs.

    Returns
    -------
    data_qubits   : sorted list of (remapped) data qubit indices
    ancilla_qubits: sorted list of (remapped) ancilla qubit indices
    h_qubits      : list of (remapped) ancilla indices that get H gates
                    (these are the X-stabilizer ancillas)
    cx_steps      : list of 4 lists, each a list of (ctrl, tgt) pairs
    total_qubits  : int  (== 2d^2 - 1)
    remap         : dict  sparse_stim_idx -> contiguous_idx
    """
    circuit = stim.Circuit.generated(
        "surface_code:rotated_memory_z",
        rounds=max(rounds, 3),   # need >= 3 so Stim emits a REPEAT block
        distance=d,
        after_clifford_depolarization=0,
        after_reset_flip_probability=0,
        before_measure_flip_probability=0,
        before_round_data_depolarization=0,
    )

    # -- identify data / ancilla qubits from flattened circuit --
    flat = circuit.flattened()
    data_set: Set[int] = set()
    ancilla_set: Set[int] = set()
    for inst in flat:
        if inst.name == "M":          # final data-qubit measurement
            for t in inst.targets_copy():
                data_set.add(t.value)
        elif inst.name == "MR":       # mid-circuit ancilla measure+reset
            for t in inst.targets_copy():
                ancilla_set.add(t.value)

    # Build sparse -> contiguous remap (data first, then ancilla)
    data_sorted = sorted(data_set)
    ancilla_sorted = sorted(ancilla_set)
    remap: Dict[int, int] = {}
    for new_idx, old_idx in enumerate(data_sorted):
        remap[old_idx] = new_idx
    offset = len(data_sorted)
    for new_idx, old_idx in enumerate(ancilla_sorted):
        remap[old_idx] = offset + new_idx

    total_qubits = len(data_set) + len(ancilla_set)
    assert total_qubits == 2 * d * d - 1, (
        f"Expected {2*d*d-1} qubits for d={d}, got {total_qubits}"
    )

    # -- extract one syndrome-extraction round from the REPEAT body --
    body = None
    for inst in circuit:
        if isinstance(inst, stim.CircuitRepeatBlock):
            body = inst.body_copy()
            break
    assert body is not None, "No REPEAT block found in Stim circuit"

    h_qubits: List[int] = []
    cx_steps: List[List[Tuple[int, int]]] = []
    seen_h = False

    for op in body:
        if op.name == "H" and not seen_h:
            # First H layer -> X-stabilizer ancillas
            h_qubits = [remap[t.value] for t in op.targets_copy()]
            seen_h = True
        elif op.name == "CX":
            targets = [t.value for t in op.targets_copy()]
            pairs = [(remap[targets[i]], remap[targets[i + 1]])
                     for i in range(0, len(targets), 2)]
            cx_steps.append(pairs)

    assert len(cx_steps) == 4, f"Expected 4 CX steps, got {len(cx_steps)}"

    data_qubits = [remap[q] for q in data_sorted]
    ancilla_qubits = [remap[q] for q in ancilla_sorted]

    return data_qubits, ancilla_qubits, h_qubits, cx_steps, total_qubits, remap


# --- Identify X / Z stabilizer ancillas and their data neighbors ------------

def _build_stabilizer_info(
    ancilla_qubits: List[int],
    h_qubits: List[int],
    cx_steps: List[List[Tuple[int, int]]],
    data_set: Set[int],
):
    """
    From the CX schedule, figure out which ancillas are X-type (they
    get H gates) vs Z-type, and which data qubits each ancilla touches.

    Returns
    -------
    x_stabilizers : list of (ancilla_idx, [data_neighbors])
    z_stabilizers : list of (ancilla_idx, [data_neighbors])
    """
    h_set = set(h_qubits)

    # Collect all data neighbors of each ancilla
    anc_neighbors: Dict[int, List[int]] = {a: [] for a in ancilla_qubits}
    for step in cx_steps:
        for ctrl, tgt in step:
            if ctrl in anc_neighbors and tgt in data_set:
                anc_neighbors[ctrl].append(tgt)
            elif tgt in anc_neighbors and ctrl in data_set:
                anc_neighbors[tgt].append(ctrl)

    x_stabilizers = []
    z_stabilizers = []
    for a in ancilla_qubits:
        neighbors = list(dict.fromkeys(anc_neighbors[a]))  # deduplicate, keep order
        if a in h_set:
            x_stabilizers.append((a, neighbors))
        else:
            z_stabilizers.append((a, neighbors))

    return x_stabilizers, z_stabilizers


# --- QASM 3 emitter ---------------------------------------------------------

def emit_surface_code_qasm(
    d: int,
    rounds: int,
    seed: int = 42,
    with_corrections: bool = True,
) -> str:
    """
    Generate an OpenQASM 3 circuit for rotated Surface Code memory.

    The circuit structure mirrors Stim's canonical CX schedule:

        // initialise a random subset with H
        for round in [0:rounds] {
            reset ancillas
            H on X-ancillas
            CX step 1 ... 4     (Stim's hook-optimal ordering)
            H on X-ancillas
            measure ancillas -> c[]
            if (c[k]) { X or Z correction on a data qubit }
        }
        measure data qubits
    """
    (data_qubits, ancilla_qubits, h_qubits,
     cx_steps, total_qubits, _remap) = _extract_surface_code_schedule(d, rounds)

    data_set = set(data_qubits)
    x_stabilizers, z_stabilizers = _build_stabilizer_info(
        ancilla_qubits, h_qubits, cx_steps, data_set,
    )

    n_data = len(data_qubits)
    n_ancilla = len(ancilla_qubits)
    n_cbits = n_ancilla

    lines: List[str] = []
    lines.append("OPENQASM 3;")
    lines.append('include "stdgates.inc";')
    lines.append(f"qubit[{total_qubits}] q;")
    lines.append(f"bit[{n_cbits}] c;")
    lines.append("")
    lines.append(f"// Rotated Surface Code d={d}, {rounds} syndrome-extraction rounds")
    lines.append(f"// Generated from Stim {stim.__version__} (surface_code:rotated_memory_z)")
    lines.append(f"// {n_data} data + {n_ancilla} ancilla = {total_qubits} qubits")
    lines.append(f"// X-stabilizers: {len(x_stabilizers)}, Z-stabilizers: {len(z_stabilizers)}")
    lines.append(f"// CX schedule: 4 steps, {sum(len(s) for s in cx_steps)} total CX per round")
    lines.append("")

    # Random initialisation of a subset of data qubits in |+> state
    rng = random.Random(seed)
    init_qubits = sorted(rng.sample(data_qubits, n_data // 3))
    for qi in init_qubits:
        lines.append(f"h q[{qi}];")
    lines.append("")

    # -- Syndrome extraction loop --
    lines.append(f"for int round in [0:{rounds}] {{")

    # Reset ancilla qubits
    lines.append("  // Reset ancilla qubits")
    for a in ancilla_qubits:
        lines.append(f"  reset q[{a}];")
    lines.append("")

    # H on X-stabilizer ancillas (basis change for X-type measurement)
    lines.append("  // Hadamard on X-stabilizer ancillas")
    for a in h_qubits:
        lines.append(f"  h q[{a}];")
    lines.append("")

    # 4-step CX schedule (Stim's hook-optimal ordering)
    for step_idx, pairs in enumerate(cx_steps):
        lines.append(f"  // CX step {step_idx + 1}")
        for ctrl, tgt in pairs:
            lines.append(f"  cx q[{ctrl}], q[{tgt}];")
        lines.append("")

    # H on X-stabilizer ancillas (undo basis change)
    lines.append("  // Undo Hadamard on X-stabilizer ancillas")
    for a in h_qubits:
        lines.append(f"  h q[{a}];")
    lines.append("")

    # Measure ancillas
    lines.append("  // Measure syndrome ancillas")
    meas_idx = 0
    z_meas_indices = {}
    for anc_idx, _ in z_stabilizers:
        lines.append(f"  c[{meas_idx}] = measure q[{anc_idx}];")
        z_meas_indices[anc_idx] = meas_idx
        meas_idx += 1
    x_meas_indices = {}
    for anc_idx, _ in x_stabilizers:
        lines.append(f"  c[{meas_idx}] = measure q[{anc_idx}];")
        x_meas_indices[anc_idx] = meas_idx
        meas_idx += 1
    lines.append("")

    # Conditional corrections
    if with_corrections:
        lines.append("  // Conditional corrections")
        # Z-syndrome -> X correction on first data neighbor
        for anc_idx, data_neighbors in z_stabilizers:
            if data_neighbors:
                mi = z_meas_indices[anc_idx]
                lines.append(f"  if (c[{mi}]) {{")
                lines.append(f"    x q[{data_neighbors[0]}];")
                lines.append(f"  }}")
        # X-syndrome -> Z correction on first data neighbor
        for anc_idx, data_neighbors in x_stabilizers:
            if data_neighbors:
                mi = x_meas_indices[anc_idx]
                lines.append(f"  if (c[{mi}]) {{")
                lines.append(f"    z q[{data_neighbors[0]}];")
                lines.append(f"  }}")
        lines.append("")

    lines.append("}")  # end for loop
    lines.append("")

    # Final data qubit readout
    lines.append("// Final data qubit readout")
    for i, dq in enumerate(data_qubits):
        if i < n_cbits:
            lines.append(f"c[{i}] = measure q[{dq}];")

    return "\n".join(lines)


# --- batch generation -------------------------------------------------------

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
        # Compute structural info once per distance
        (data_q, anc_q, h_q, cx_steps,
         total_q, _) = _extract_surface_code_schedule(d)
        data_set = set(data_q)
        x_stab, z_stab = _build_stabilizer_info(anc_q, h_q, cx_steps, data_set)

        n_data = len(data_q)
        n_ancilla = len(anc_q)
        n_stab = len(x_stab) + len(z_stab)
        cnots_per_round = sum(len(step) for step in cx_steps)
        h_per_round = 2 * len(h_q)

        for rounds in rounds_list:
            bench_name = f"surface_code_d{d}_r{rounds}"
            bench_dir = os.path.join(BENCHMARKS_DIR, f"d{d}", bench_name)
            os.makedirs(bench_dir, exist_ok=True)

            total_cnots = cnots_per_round * rounds
            total_h = h_per_round * rounds
            total_corrections = n_stab * rounds

            print(f"  d={d:2d}, rounds={rounds:3d} -> {total_q:4d} qubits, "
                  f"~{total_cnots:6d} CNOTs, ~{total_h} H gates, "
                  f"~{total_corrections} corrections/round")

            for rep in range(replicates):
                seed = base_seed + d * 1000 + rounds * 100 + rep
                qasm = emit_surface_code_qasm(
                    d=d,
                    rounds=rounds,
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
                "generator": f"stim {stim.__version__} (surface_code:rotated_memory_z)",
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
        description="Generate Surface Code quantum memory benchmarks (OpenQASM 3) "
                    "using Stim's rotated_memory_z generator")
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
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory for benchmarks (default: benchmarks/)")
    args = parser.parse_args()

    if args.output_dir:
        global BENCHMARKS_DIR
        BENCHMARKS_DIR = os.path.abspath(args.output_dir)

    print(f"Generating Surface Code benchmarks (Stim {stim.__version__})...")
    print(f"  Distances:   {args.distances}")
    print(f"  Rounds:      {args.rounds}")
    print(f"  Replicates:  {args.replicates}")
    print(f"  Corrections: {not args.no_corrections}")
    print(f"  Output dir:  {BENCHMARKS_DIR}")
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
