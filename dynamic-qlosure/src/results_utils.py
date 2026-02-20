"""Centralised results configuration and I/O helpers.

Standard results directory hierarchy
-------------------------------------
results/{method}/{template}/{backend}/{bench}/{circuit_config}/{circuit_name}/[SEED_{seed}/]
    trace.json  — structured routing trace
    time.txt    — routing wall-clock time (seconds)
    path.txt    — source circuit file path

Methods: qlosure, sabre, baselines
"""

import json
from pathlib import Path

# ── Path anchors (CWD-independent) ──────────────────────────────────────────
_SRC_DIR     = Path(__file__).resolve().parent        # dynamic-qlosure/src/
PROJECT_DIR  = _SRC_DIR.parent                        # dynamic-qlosure/
ROOT_DIR     = PROJECT_DIR.parent                     # quantum-compiler/

# ── Shared path roots ───────────────────────────────────────────────────────
RESULTS_ROOT           = PROJECT_DIR / "results"
RESULTS_SUMMARY_ROOT   = PROJECT_DIR / "results-summary"
D_QUEKO_BENCHMARKS_DIR = ROOT_DIR / "d-queko" / "benchmarks"
TOPOLOGIES_DIR         = ROOT_DIR / "qpu" / "topologies"


# ── I/O helpers ─────────────────────────────────────────────────────────────

def save_trace_results(output_dir, trace, elapsed_time=None, circuit_path=None):
    """Write standardised result files into *output_dir*.

    Always writes ``trace.json``.
    Writes ``time.txt`` when *elapsed_time* is provided.
    Writes ``path.txt`` when *circuit_path* is provided.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "trace.json", "w") as f:
        json.dump(trace, f, indent=2)

    if elapsed_time is not None:
        (output_dir / "time.txt").write_text(f"{elapsed_time:.6f}\n")

    if circuit_path is not None:
        (output_dir / "path.txt").write_text(str(circuit_path))


def load_topology(name):
    """Load a topology JSON by backend name or filename stem.

    Tries ``qpu/topologies/{name}.json`` first; if that doesn't exist it
    falls back to the ``BACKEND_FILE_MAP`` in ``qpu/src/load_backend.py``
    so that short names like ``"heavy_hexagon"`` resolve correctly.
    """
    path = TOPOLOGIES_DIR / f"{name}.json"
    if not path.exists():
        # Fallback: resolve via the shared backend name mapping
        from qpu.src.load_backend import BACKEND_FILE_MAP
        filename = BACKEND_FILE_MAP.get(name)
        if filename:
            path = TOPOLOGIES_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Topology not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
