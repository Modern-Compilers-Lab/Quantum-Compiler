"""Experiment definitions: which circuits run on which backend, and where the
resulting CSV lands.

One entry per paper experiment. ``generate.py`` walks these to build jobs;
``render.py`` reads the CSVs they produce.
"""

from __future__ import annotations

import json

from common import ARTIFACT_DIR, BENCHMARKS_DIR

SELECTION = json.loads((ARTIFACT_DIR / "circuit_selection.json").read_text())

# One selection for both mappers, taken from run_benchmark_seeds.py.
SELECTION_NOTE = ("nest0/ibm_brisbane_old selection taken from "
                  "run_benchmark_seeds.py and applied to both mappers")

SEEDS_10 = [3, 21, 42, 63, 84, 105, 126, 147, 168, 189]

# routing.ABLATION_CONFIGS rung -> column suffix in the ablation CSV
ABLATION_SUFFIX = {1: "no_remap_no_error", 2: "no_remap",
                   3: "new_line", 4: "default"}
SEEDS_3 = [3, 21, 42]
LOOP_ITERATIONS = 10


class Experiment:
    def __init__(self, name, kind, csv_rel, **kw):
        self.name = name
        self.kind = kind
        self.csv_rel = csv_rel
        self.__dict__.update(kw)


def _selected(template, backend, bench):
    try:
        return SELECTION[template][backend][bench]
    except KeyError:
        return {}


def dqueko_jobs(template, backend, bench, seeds, leaf_depths=None):
    """Yield (circuit_path, config_name, seed) for a d-QUEKO sweep."""
    root = BENCHMARKS_DIR / template / bench
    for depth_key, rel in _selected(template, backend, bench).items():
        depth = int(depth_key.rsplit("-", 1)[-1])
        if leaf_depths and depth not in leaf_depths:
            continue
        path = root / rel
        config = rel.split("/")[-2]
        for seed in seeds:
            yield path, config, seed


def wi_rule_jobs(bench, leaf_depth, seeds):
    """Yield (circuit_path, relative_config, seed) for the nested w_i sweep."""
    root = BENCHMARKS_DIR / "wi_rule_benchmarks" / bench / f"{leaf_depth}Leaf_depth"
    for circuit in sorted(root.rglob("*.qasm")):
        rel = circuit.relative_to(root)
        for seed in seeds:
            yield circuit, str(rel.parent / circuit.stem).replace("\\", "/"), seed


EXPERIMENTS = {
    # Table 3, Figure 8, Figure 1
    "main": Experiment(
        "main", "dqueko", "main/{backend}_{bench}_{iters}iter_metrics.csv",
        template="nest0",
        backends=["ibm_brisbane_old", "ibm_kingston"],
        benches=["54qbt", "81qbt", "121qbt"],
        seeds=SEEDS_10,
    ),
    # Table 4
    "chiplet": Experiment(
        "chiplet", "dqueko", "{backend}_{bench}_{iters}iter_metrics.csv",
        template="nest0",
        backends=["heavy_hexagon", "ibm_flamingo"],
        benches={"heavy_hexagon": ["81qbt", "121qbt"], "ibm_flamingo": ["256qbt"]},
        csv_overrides={"ibm_flamingo": "main/{backend}_{bench}_{iters}iter_metrics.csv"},
        seeds=SEEDS_3,
    ),
    # Table 5
    "surface-code": Experiment(
        "surface-code", "surface", None,
        configs=[
            dict(backend="ibm_brisbane", tag="surface_code_stim",
                 benchmarks="benchmarks_stim", rounds=None, loop_iterations=10),
            dict(backend="mech_heavy_hex", tag="surface_code_mech",
                 benchmarks="benchmarks", rounds=[3], loop_iterations=3),
        ],
        seeds=SEEDS_3,
    ),
    # Figure 9
    "nested": Experiment(
        "nested", "wi_rule", "w_i_rule_metrics.csv",
        backend="ibm_kingston", bench="81qbt", leaf_depth=10, seeds=[42],
    ),
    # Table 7, Figure 10
    "ablation": Experiment(
        "ablation", "ablation",
        "ablation_study5/{backend}_{bench}_{iters}iter_metrics_ablation_study.csv",
        template="nest0",
        backends=["ibm_brisbane_old"],
        benches=["81qbt", "121qbt"],
        seeds=SEEDS_10,
        configs=[1, 2, 3, 4],
    ),
    # Table 6
    "timing": Experiment(
        "timing", "dqueko", None,
        template="one_loop",
        backends=["ibm_kingston", "ibm_brisbane_old"],
        benches=["54qbt", "81qbt", "121qbt"],
        seeds=SEEDS_10,
        dynamiq_only=True,
    ),
}

RENDER_DEPENDENCIES = {
    "fig01": ["main"],
    "fig08": ["main"],
    "fig09": ["nested"],
    "tab03": ["main"],
    "tab04": ["chiplet"],
    "tab05": ["surface-code"],
    "tab06": ["timing"],
    "fig10": ["ablation"],
    "tab07": ["ablation"],
}
