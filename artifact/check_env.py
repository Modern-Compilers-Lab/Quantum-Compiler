"""Check the interpreter, pinned packages and benchmark inputs.

    python artifact/check_env.py
"""

from __future__ import annotations

import importlib.metadata as md
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

#: Must match artifact/environment.yml.
EXPECTED = {
    "qiskit": "2.3.0",
    "qiskit-qasm3-import": "0.6.0",
    "rustworkx": "0.16.0",
    "numpy": "2.3.2",
    "scipy": "1.16.1",
    "networkx": "3.5",
    "symengine": "0.13.0",
    "pandas": "2.2.3",
    "matplotlib": "3.10.5",
    "tqdm": "4.67.1",
}
EXPECTED_PYTHON = "3.13.1"

#: Fixed inputs, never regenerated.
REQUIRED_INPUTS = [
    ("d-queko/benchmarks/nest0", "d-QUEKO circuits for main / chiplet"),
    ("d-queko/benchmarks/one_loop", "circuits for the timing experiment"),
    ("d-queko/benchmarks/wi_rule_benchmarks", "nested (w_i) circuits"),
    ("surface-code/benchmarks_stim", "Stim surface-code circuits"),
    ("qpu/topologies", "backend coupling maps and calibration"),
    ("artifact/circuit_selection.json", "per-backend circuit selection"),
]


def main() -> int:
    root = Path(__file__).resolve().parent.parent
    problems, warnings = [], []

    print("Python")
    got_py = ".".join(str(v) for v in sys.version_info[:3])
    ok = got_py == EXPECTED_PYTHON
    print(f"  {'ok ' if ok else 'DIFF'}  {got_py:<12} expected {EXPECTED_PYTHON}")
    if not ok:
        warnings.append(f"python {got_py} != {EXPECTED_PYTHON}")

    print("\nPackages")
    for pkg, want in EXPECTED.items():
        try:
            got = md.version(pkg)
        except md.PackageNotFoundError:
            print(f"  MISSING  {pkg}")
            problems.append(f"{pkg} not installed")
            continue
        ok = got == want
        print(f"  {'ok  ' if ok else 'DIFF'}  {pkg:<22} {got:<10} expected {want}")
        if not ok:
            warnings.append(f"{pkg} {got} != {want}")

    print("\nInputs")
    for rel, what in REQUIRED_INPUTS:
        p = root / rel
        ok = p.exists()
        print(f"  {'ok  ' if ok else 'MISS'}  {rel:<40} {what}")
        if not ok:
            problems.append(f"missing input: {rel}")

    print("\nMapper")
    try:
        import common  # puts src.* and qpu.* on sys.path
        assert common.REPO_ROOT.exists()
        from qpu.src.load_backend import load_backend_data  # noqa: E402
        from src.backend import QuantumBackend              # noqa: E402
        from src.routing import ABLATION_CONFIGS, Qlosure   # noqa: E402
        from src.dag import build_dag, extract_multi_qubit_dag  # noqa: E402
        from qiskit import qasm3                            # noqa: E402

        d = load_backend_data("ibm_brisbane_old")
        hw = QuantumBackend(d["coupling_map"], qubit_props=d.get("qubits", {}))
        circ = next((root / "d-queko/benchmarks/one_loop/54qbt").rglob("*.qasm"))
        qc = qasm3.load(str(circ))
        dag = build_dag(qc)
        m = Qlosure(hw, seed=3)
        m.run(dag, extract_multi_qubit_dag(dag), initial_mapping="trivial", num_iter=1)
        print(f"  ok    routed {circ.name} on ibm_brisbane_old "
              f"({hw.num_qubits} physical qubits)")
        print(f"  ok    ablation configs available: {sorted(ABLATION_CONFIGS)}")
    except Exception as exc:
        print(f"  FAIL  {type(exc).__name__}: {exc}")
        problems.append(f"mapper smoke test failed: {exc}")

    print()
    if problems:
        print(f"FAILED - {len(problems)} problem(s):")
        for p in problems:
            print(f"  - {p}")
        return 1
    if warnings:
        print(f"OK with {len(warnings)} version difference(s):")
        for w in warnings:
            print(f"  - {w}")
        print("\nThe artifact will run, but numbers may differ.")
        return 0
    print("OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
