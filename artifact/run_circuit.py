"""Run DynamiQ on a single circuit, optionally comparing against Qiskit Sabre.

    python artifact/run_circuit.py --circuit=<path.qasm> --backend=ibm_brisbane_old
    python artifact/run_circuit.py --circuit=<path.qasm> --compare
    python artifact/run_circuit.py --circuit=<path.qasm> --save-trace out/

--circuit accepts an absolute path, a path relative to the repository root, or
one relative to d-queko/benchmarks/.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import BENCHMARKS_DIR, METRICS, REPO_ROOT, ensure_dir
from runners import (METRIC_KEYS, load_backend, num_physical_qubits, run_dynamiq,
                     run_sabre, score_trace)

UNITS = {"swaps": "", "depth": "", "latency": " us", "error": ""}


def resolve_circuit(arg: str) -> Path:
    for cand in (Path(arg), REPO_ROOT / arg, BENCHMARKS_DIR / arg):
        if cand.exists():
            return cand.resolve()
    raise SystemExit(
        f"circuit not found: {arg}\n"
        f"  tried {Path(arg)}\n        {REPO_ROOT / arg}\n        {BENCHMARKS_DIR / arg}")


def summarise(trace):
    """Count what the mapper emitted. Trace entries are typed gate / swap /
    for / while / if_else; control-flow entries nest under body or branches."""
    counts = {"gates": 0, "swaps": 0, "loops": 0, "conditionals": 0, "max nesting": 0}

    def walk(entries, depth):
        counts["max nesting"] = max(counts["max nesting"], depth)
        for e in entries:
            kind = e.get("type")
            if kind == "gate":
                counts["gates"] += 1
            elif kind == "swap":
                counts["swaps"] += 1
            elif kind in ("for", "while"):
                counts["loops"] += 1
                walk(e.get("body", []), depth + 1)
            elif kind == "if_else":
                counts["conditionals"] += 1
                for branch in e.get("branches", []) or []:
                    walk(branch if isinstance(branch, list) else branch.get("body", []),
                         depth + 1)

    walk(trace, 0)
    return counts


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--circuit", required=True)
    ap.add_argument("--backend", default="ibm_brisbane_old")
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--num-iterations", type=int, default=1,
                    help="forward/backward passes for the initial mapping")
    ap.add_argument("--loop-iterations", type=int, default=10,
                    help="assumed trip count when scoring loops")
    ap.add_argument("--ablation", type=int, choices=[1, 2, 3, 4], default=None)
    ap.add_argument("--compare", action="store_true",
                    help="also route with Qiskit Sabre and report the difference")
    ap.add_argument("--save-trace", metavar="DIR",
                    help="write trace.json (and sabre_trace.json with --compare)")
    ap.add_argument("--verbose", type=int, default=0)
    args = ap.parse_args()

    circuit = resolve_circuit(args.circuit)
    backend, props, edges = load_backend(args.backend)

    print(f"circuit : {circuit.relative_to(REPO_ROOT) if REPO_ROOT in circuit.parents else circuit}")
    print(f"backend : {args.backend}  ({backend.num_qubits} physical qubits, "
          f"{len(edges)} couplings)")
    print(f"settings: seed={args.seed}  num_iterations={args.num_iterations}  "
          f"loop_iterations={args.loop_iterations}"
          f"{'  ablation=' + str(args.ablation) if args.ablation else ''}")

    print("\nrouting with DynamiQ ...", flush=True)
    trace, elapsed = run_dynamiq(circuit, backend, args.seed, args.num_iterations,
                                 args.verbose, ablation=args.ablation)
    ours = score_trace(trace, props, args.loop_iterations)
    print(f"  done in {elapsed:.2f} s")

    print("\ntrace")
    for k, v in summarise(trace).items():
        print(f"  {k:<22} {v}")

    sabre = None
    if args.compare:
        from qiskit.transpiler import CouplingMap
        print("\nrouting with Qiskit Sabre ...", flush=True)
        s_trace, s_elapsed = run_sabre(circuit, CouplingMap(edges),
                                       num_physical_qubits(edges, props), args.seed)
        sabre = score_trace(s_trace, props, args.loop_iterations)
        print(f"  done in {s_elapsed:.2f} s")

    print("\nmetrics" + ("  (lower is better; improvement = (Sabre - Ours) / Sabre)"
                         if sabre else ""))
    header = f"  {'metric':<10}{'DynamiQ':>14}"
    if sabre:
        header += f"{'Sabre':>14}{'improvement':>14}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for m in METRICS:
        line = f"  {m:<10}{ours[m]:>14.2f}"
        if sabre:
            imp = (sabre[m] - ours[m]) / sabre[m] * 100 if sabre[m] else float("nan")
            line += f"{sabre[m]:>14.2f}{imp:>13.1f}%"
        print(line + UNITS[m])

    if args.save_trace:
        out = ensure_dir(Path(args.save_trace))
        (out / "trace.json").write_text(json.dumps(trace, separators=(",", ":")))
        (out / "metrics.json").write_text(json.dumps(
            {"dynamiq": {k: ours[k] for k in METRIC_KEYS}, "time_s": elapsed}, indent=2))
        print(f"\n  saved {out / 'trace.json'}")
        if sabre:
            (out / "sabre_trace.json").write_text(json.dumps(s_trace, separators=(",", ":")))
            print(f"  saved {out / 'sabre_trace.json'}")


if __name__ == "__main__":
    raise SystemExit(main())
