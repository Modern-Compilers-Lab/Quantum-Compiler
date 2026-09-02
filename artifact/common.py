"""Shared plumbing for the DynamiQ artifact scripts.

Every ``render/*.py`` module imports from here so that path resolution and
plot styling behave identically across the artifact.

Directory anchors
-----------------
    <repo>/dynamic-qlosure/results-summary/   aggregated CSVs (committed)
    <repo>/dynamic-qlosure/results/           raw per-seed traces (git-ignored)
    <repo>/d-queko/benchmarks/                d-QUEKO circuits
    <repo>/qpu/topologies/                    backend coupling maps + calibration
    <repo>/artifact/output/                   everything the artifact produces
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

# ── Path anchors ────────────────────────────────────────────────────────────
ARTIFACT_DIR = Path(__file__).resolve().parent
REPO_ROOT = ARTIFACT_DIR.parent

DQ_DIR = REPO_ROOT / "dynamic-qlosure"
SUMMARY_ROOT = DQ_DIR / "results-summary"
RESULTS_ROOT = DQ_DIR / "results"
BENCHMARKS_DIR = REPO_ROOT / "d-queko" / "benchmarks"
TOPOLOGIES_DIR = REPO_ROOT / "qpu" / "topologies"
SURFACE_CODE_DIR = REPO_ROOT / "surface-code"

OUTPUT_DIR = ARTIFACT_DIR / "output"
FIGURES_OUT = OUTPUT_DIR / "figures"
TABLES_OUT = OUTPUT_DIR / "tables"

# Make the DynamiQ sources importable (src.*, qpu.*) from any artifact script.
for _p in (str(REPO_ROOT), str(DQ_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ── Vocabulary ──────────────────────────────────────────────────────────────
METRICS = ["swaps", "depth", "latency", "error"]
METRIC_LABELS = {"swaps": "SWAPs", "depth": "Depth", "latency": "Latency", "error": "Error"}

#: Backend id used in the results tree -> name used in the paper.
BACKEND_LABELS = {
    "ibm_brisbane_old": "IBM Brisbane (127 qbt)",
    "ibm_kingston": "IBM Kingston (156 qbt)",
    "ibm_flamingo": "IBM Flamingo (3x156 chiplet)",
    "heavy_hexagon": "Heavy-Hexagon 8x8_2x2 (MECH)",
    "mech_heavy_hex": "MECH heavy-hex 3x4 (480 qbt)",
    "ibm_brisbane": "IBM Brisbane (127 qbt, recalibrated)",
}


# ── Output helpers ──────────────────────────────────────────────────────────

def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def figure_dir(fig_id: str) -> Path:
    """``figure_dir("fig08")`` -> ``artifact/output/figures/fig08`` (created)."""
    return ensure_dir(FIGURES_OUT / fig_id)


def table_dir(tab_id: str) -> Path:
    return ensure_dir(TABLES_OUT / tab_id)


def require(path: Path, hint: str = "") -> Path:
    """Fail loudly and actionably when an input the script needs is absent."""
    if not Path(path).exists():
        msg = f"Required input not found:\n    {path}"
        if hint:
            msg += f"\n\n{hint}"
        raise SystemExit(msg)
    return Path(path)


# ── Summary-CSV loading ─────────────────────────────────────────────────────

def load_pairwise_summary(csv_path: Path) -> pd.DataFrame:
    """Load an ours-vs-sabre summary CSV keyed on leaf_depth."""
    require(csv_path, "Run artifact/pipeline steps, or regenerate with\n"
                      "    python dynamic-qlosure/visualization/save-results-csv.py")
    df = pd.read_csv(csv_path)
    missing = [f"{m}_mean_{w}" for m in METRICS for w in ("ours", "sabre")
               if f"{m}_mean_{w}" not in df.columns]
    if missing:
        raise SystemExit(f"{csv_path} is missing columns: {missing}")
    return df.sort_values("leaf_depth").reset_index(drop=True)


# ── Matplotlib styling ──────────────────────────────────────────────────────

PAPER_RCPARAMS = {
    "font.size": 32,
    "axes.titlesize": 34,
    "axes.labelsize": 34,
    "xtick.labelsize": 32,
    "ytick.labelsize": 32,
    "legend.fontsize": 30,
    "legend.title_fontsize": 34,
    "lines.linewidth": 2.5,
    "lines.markersize": 20,
    "figure.dpi": 300,
    "pdf.fonttype": 42,   # embed real text, not outlines
    "ps.fonttype": 42,
}


def apply_paper_style(**overrides):
    """Apply the paper's matplotlib style. Import matplotlib lazily."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(PAPER_RCPARAMS)
    if overrides:
        plt.rcParams.update(overrides)
    return plt


def banner(title: str, subtitle: str = "") -> None:
    print()
    print("=" * 78)
    print(f"  {title}")
    if subtitle:
        print(f"  {subtitle}")
    print("=" * 78)


def saved(path: Path) -> None:
    print(f"  saved  {path.relative_to(REPO_ROOT) if REPO_ROOT in path.parents else path}")


def warn_partial(df, *key_cols, col="n_leaf_depths", expected=9):
    """Report how many leaf depths went into each cell. A narrowed run
    (--widths / --leaf-depths) leaves cells that are not comparable."""
    counts = sorted(set(df[col]))
    if counts == [expected]:
        print(f"  {expected} leaf depths per cell.")
        return
    print(f"  NOTE: cells are averaged over differing numbers of leaf depths "
          f"(expected {expected}).")
    for _, r in df.iterrows():
        n = int(r[col])
        flag = "" if n == expected else "   <- partial"
        label = " / ".join(str(r[k]) for k in key_cols)
        print(f"    {label:<28} {n} leaf depth{'s' if n != 1 else ''}{flag}")
    print("  Regenerate the partial ones for a like-for-like comparison.")
