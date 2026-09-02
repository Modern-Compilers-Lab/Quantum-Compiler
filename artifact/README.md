# DynamiQ reproduction artifact

Reproduces the figures and tables of *Mapping Dynamic, Hierarchical Quantum
Circuits* (`paper/paper-174.pdf`).

The artifact is two steps:

```
generate.py   run DynamiQ and Qiskit Sabre on the benchmarks  ->  summary CSVs
render.py     turn those CSVs into the paper's figures and tables
```

## Install

```bash
conda env create -f artifact/environment.yml
conda activate dynamiq-artifact
python artifact/check_env.py
```

Run every command from the repository root. `check_env.py` verifies the
interpreter, the pinned packages and the benchmark inputs, and exits non-zero
if anything is missing.

Use `artifact/environment.yml`, not `dynamic-qlosure/requirements.txt` — the
latter is the wider environment for the baseline comparisons and pins an
incompatible NumPy.

## Quick check (seconds)

Render everything from the CSVs committed under
`dynamic-qlosure/results-summary/`, without running any mapping:

```bash
python artifact/render.py all --source committed
```

This renders from the authors' data without re-running the mappers.

## Full reproduction

```bash
python artifact/generate.py all --jobs 8      # hours; see runtimes below
python artifact/render.py all
```

## Choosing the CSV source

`render.py` takes `--source`:

| value | behaviour |
|---|---|
| `auto` (default) | prefer a CSV from `generate.py`, fall back to the committed one |
| `generated` | only your own CSVs; skip items you have not generated |
| `committed` | only the shipped CSVs |

Under `auto` you can regenerate one experiment and still render everything.
The closing summary lists every CSV that fell back, so you can always see
which items came from your own runs and which from the shipped data.

`render.py` does not compare anything against the paper — it renders the
figures and tables from whatever CSVs it is given. The paper's printed
numbers are recorded under `REFERENCE` in `paper_values.py` if you want to
compare by hand.

`generate.py` writes per-run traces to `dynamic-qlosure/results/` and reuses
them on a rerun, so an interrupted job resumes. `--force` recomputes.

Work on one experiment or a slice of it:

```bash
python artifact/generate.py --list
python artifact/generate.py main --widths 54 --leaf-depths 10 20 --jobs 4
python artifact/generate.py surface-code --jobs 8
python artifact/generate.py main --summarise-only     # rebuild CSVs only
```

## What maps to what

| Paper item | Render | Experiment | Backends |
|---|---|---|---|
| Figure 1 | `fig01` | `main`, `nested` | Kingston, Brisbane |
| Figure 8 | `fig08` | `main` | Brisbane |
| Figure 9 | `fig09` | `nested` | Kingston |
| Table 3 | `tab03` | `main` | Brisbane, Kingston |
| Table 4 | `tab04` | `chiplet` | Heavy-Hexagon 8x8_2x2, IBM Flamingo |
| Table 5 | `tab05` | `surface-code` | Brisbane, MECH 3x4 |
| Table 6 | `tab06` | `timing` | Kingston, Brisbane |
| Table 7 | `tab07` | committed ablation CSVs | Brisbane |
| Figure 10 | `fig10` | committed ablation CSVs | Brisbane |

```bash
python artifact/render.py --list
python artifact/render.py tab03
```

Output lands in `artifact/output/`: `csv/` from generate, `figures/` and
`tables/` from render.

## Experiment settings

Fixed by Sec. 7.1 and reflected in `experiments.py`:

- Sabre baseline: `routing_method='sabre'`, `layout_method='identity'`,
  `optimization_level=1`, so gate count and initial placement match DynamiQ's.
- Loops are scored at 10 iterations; conditionals take the worst branch.
- `main` and `timing` sweep 10 seeds; `chiplet` and `surface-code` use 3.
- Surface-code traces are scored at `loop_iterations=10` on Brisbane and 3 on
  MECH (which only has r=3). Scoring MECH at 10 inflates every improvement by
  about 10 points.

## Runtimes

Measured on one core of a laptop-class CPU, per DynamiQ mapping; Sabre is
sub-second throughout.

| Experiment | Runs | Approx. serial time |
|---|---|---|
| `main` | 720 | 6-10 h |
| `chiplet` | 162 | 2-4 h |
| `surface-code` | 324 | 1-2 h |
| `nested` | 150 | 1-2 h |
| `timing` | 270 | 1 h |

`--jobs N` runs N mappings in parallel and scales close to linearly.

## Reproduction fidelity

Regenerated numbers do not land exactly on the committed ones. Expect
differences rather than equality, and compare trends rather than cells.

**Qiskit version.** Sabre comes out a few percent below the committed values,
consistent with LightSABRE changing between 1.3.2 (the paper) and 2.3.0 (what
this artifact has been exercised on). Pin 1.3.2 for the closest match.

**Nondeterminism.** Two runs of the same circuit in different processes gave
4500 and 4515 SWAPs, so results are not bit-reproducible across processes (set
`PYTHONHASHSEED` to test).

**Circuit selection is not the cause.** Where a regenerated DynamiQ number
differs from the committed one, it is not because the wrong circuit was picked:
re-running every candidate in `leaf-depth-10` puts `circ_01`, the one
`circuit_selection.json` names, closest by a wide margin. Nor is it a general
code regression — on `nest0/heavy_hexagon/121qbt` the mapper reproduces a stored
trace to within 0.04%.

## Known gaps and deviations

**Sabre invocation.** Sec. 7.1 describes LightSABRE through
`transpile(routing_method='sabre', layout_method='identity',
optimization_level=1)`. The repo's runners instead apply the bare `SabreSwap`
pass with `heuristic='decay', trials=1` on an identity layout, which is not the
same pipeline. `runners.py` matches the runners, not the prose, so the artifact
reproduces what was actually measured.

**Circuit selection for `nest0` / `ibm_brisbane_old`.** The repo's two runners
disagree: `run_benchmark_seeds.py` lists this backend, `run_benchmark_seeds_sabre.py`
does not, so the Sabre side cannot run for it as shipped. Both mappers must see
the same circuits (Sec. 7.1), so `artifact/circuit_selection.json` holds one
selection, taken from `run_benchmark_seeds.py`, used by both.

**Ablation study.** `tab07` and `fig10` render from the ablation CSVs
committed under `dynamic-qlosure/results-summary/ablation_study5/`, which hold
all four cumulative configurations side by side. They are render-only:
`generate.py` cannot rebuild them, because producing the CSVs needs one mapping
run per configuration and selecting a configuration is still a source-level edit
in `dynamic-qlosure/src/routing.py::_apply_qlosure_score_heuristic` (the
alternative `score = ...` lines there are commented out and swapped by hand).
Exposing that as a flag is the one change needed to close the loop.

**Table 8 is not reproduced.** The alpha/beta sensitivity sweep has no stored
data, and it needs the same configuration switch plus a parameterised cost
function: Eq. 3's alpha and beta are the hardcoded `W_distance = 1.0` and
`tie_breaker_weight = 0.01` inside `heuristic.depth_poly_heuristic`. Note also
that the shipped default scorer is `heuristic.new_qlosure_poly_heuristic`, which
has no additive `C_rate` term at all - so how Table 8's stated evaluation pair
<1.0, 0.2> maps onto the configuration behind Tables 3-7 needs confirming
before the sweep is meaningful. The paper's values are recorded in
`paper_values.TABLE8`.

**Figure 1.** The submitted PDF's absolute numbers are not in any committed
CSV, so they came from an earlier run. Both panels are normalised per metric,
so the ratios are what the figure shows, and those the artifact reproduces.
Panel (a) is drawn from the `w=5, i=5` nested configuration, matching the
"5-nested loops" caption.

**Table 6 is wall-clock.** `--source committed` replays the timings recorded on
the paper's machine (Intel i7-10750H at 2.60 GHz) and matches the table.
`--source generated` reports this machine's timings, where only the scaling
trend is comparable.
