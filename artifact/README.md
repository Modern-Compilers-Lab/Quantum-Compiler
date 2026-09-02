# DynamiQ Artifact

Hello,

Thank you for reviewing the artifact for **DynamiQ**.

This artifact is designed to invoke DynamiQ to run the tests that generated
Figures 1, 8, 9 and 10, as well as Tables 3, 4, 5, 6 and 7 from the main body of
the paper.

The main wrapper script, **`run_all_experiments.py`**, will execute the entire
experimentation pipeline — from benchmark loading, invoking DynamiQ and the
Qiskit Sabre baseline, to plotting — automatically. Reviewers may also run
individual experiments using:

-   **`run_circuit.py`**: to run DynamiQ on a single circuit
-   **`generate.py`**: to run DynamiQ and Sabre on one experiment's benchmarks
-   **`render.py`**: to produce a single figure or table from the results

---

## Repository layout

```
quantum-compiler/
├─ artifact/
│  ├─ render/                  # One module per paper figure and table
│  ├─ output/                  # (Created at runtime) CSVs, figures, tables
│  ├─ run_all_experiments.py   # Whole pipeline: generate + render
│  ├─ run_circuit.py           # CLI entrypoint for a single circuit
│  ├─ generate.py              # CLI entrypoint for running an experiment
│  ├─ render.py                # CLI entrypoint for figures and tables
│  ├─ check_env.py             # Verify the environment before a long run
│  ├─ experiments.py           # Which circuits run on which backend
│  ├─ runners.py               # DynamiQ and Sabre invocation
│  └─ environment.yml
├─ d-queko/benchmarks/         # d-QUEKO dynamic circuits (QASM3)
├─ surface-code/               # Stim surface-code circuits + scripts
├─ qpu/topologies/             # JSON hardware topologies + calibration
├─ dynamic-qlosure/
│  ├─ src/                     # Main DynamiQ framework code
│  └─ results-summary/         # Reference CSVs from the paper's runs
└─ paper/paper-174.pdf
```

---

## Installation

### 1) Create and activate the environment (dynamiq-artifact)

```bash
conda env create -f artifact/environment.yml
conda activate dynamiq-artifact
```

### 2) Verify the installation

```bash
python artifact/check_env.py
```

This checks the interpreter, every pinned package, the benchmark inputs, and
routes one circuit end to end. It exits non-zero if anything is missing, so it
can gate a long run.

> On Windows, clone with long paths enabled — the benchmark directories are
> deeply nested:
>
> ```bash
> git clone -c core.longpaths=true <repo>
> ```

Run every command from the repository root.

---

## Quick start

### Run on a single circuit with the IBM Brisbane backend

```bash
python artifact/run_circuit.py \
  --circuit=d-queko/benchmarks/nest0/54qbt/queko-054qbt_nest_00_nodes010_leaf-depth-10/circ_01.qasm \
  --backend=ibm_brisbane_old
```

This prints the routing time, a summary of the emitted trace (gates, SWAPs,
loops, conditionals, nesting depth) and the four metrics.

### Run the circuit and also compare against the Sabre baseline

```bash
python artifact/run_circuit.py --circuit=<path.qasm> --backend=ibm_brisbane_old --compare
```

```
metrics  (lower is better; improvement = (Sabre - Ours) / Sabre)
  metric           DynamiQ         Sabre   improvement
  ----------------------------------------------------
  swaps            1193.00       2816.00         57.6%
  depth            5898.00       6670.00         11.6%
  latency          3067.68       3484.20         12.0% us
  error              13.71         16.84         18.6%
```

### Save the routed trace

```bash
python artifact/run_circuit.py --circuit=<path.qasm> --compare --save-trace out/
```

### Run one experiment, then plot it

```bash
python artifact/generate.py main --jobs 4
python artifact/render.py tab03 fig08
```

### Run everything

```bash
python artifact/run_all_experiments.py --jobs 4
```

---

## What maps to what

| Paper item | Render | Experiment | Backends |
|---|---|---|---|
| Figure 1 | `fig01` | `main`, `nested` | Kingston, Brisbane |
| Figure 8 | `fig08` | `main` | Brisbane |
| Figure 9 | `fig09` | `nested` | Kingston |
| Figure 10 | `fig10` | `ablation` | Brisbane |
| Table 3 | `tab03` | `main` | Brisbane, Kingston |
| Table 4 | `tab04` | `chiplet` | Heavy-Hexagon 8x8_2x2, IBM Flamingo |
| Table 5 | `tab05` | `surface-code` | Brisbane, MECH 3x4 |
| Table 6 | `tab06` | `timing` | Kingston, Brisbane |
| Table 7 | `tab07` | `ablation` | Brisbane |

```bash
python artifact/generate.py --list
python artifact/render.py --list
```

Output lands in `artifact/output/`: `csv/` from generate, `figures/` and
`tables/` from render.

---

## Experiment settings

Fixed by Sec. 7.1 and reflected in `experiments.py`:

- Sabre baseline: `SabreSwap` with `heuristic='decay'`, `trials=1`, on an
  identity layout, so gate count and initial placement match DynamiQ's.
- Loops are scored at 10 iterations; conditionals take the worst branch.
- `main` and `timing` sweep 10 seeds; `chiplet` and `surface-code` use 3.
- Surface-code traces are scored at `loop_iterations=10` on Brisbane and 3 on
  MECH, which only has r=3.
- The ablation runs the four cumulative configurations of Sec. 7.6.2, selected
  with `--ablation 1..4`: distance only, + error, + depth rate, + loop-entry
  remapping.

---

## Runtimes

Measured on this machine, per DynamiQ mapping; Sabre is sub-second throughout.
`--jobs N` runs N mappings in parallel.

| Experiment | Runs | Approx. time at `--jobs 4` |
|---|---|---|
| `main` | 1080 | 5-6 h |
| `chiplet` | 162 | 3-4 h |
| `surface-code` | 342 | 30 min |
| `nested` | 150 | several h (deep nesting is expensive) |
| `timing` | 540 | 1 h |
| `ablation` | 720 | 4-6 h |

Memory is the binding constraint on the widest circuits: the 256-qubit Flamingo
benchmark can exhaust RAM at `--jobs 8`. `generate.py` recovers by halving the
worker count and retrying, but `--jobs 2` is safer for `chiplet`.

Choosing the CSV source:

| `--source` | behaviour |
|---|---|
| `auto` (default) | prefer a CSV from `generate.py`, fall back to the shipped one |
| `generated` | only your own CSVs; skip items you have not generated |
| `committed` | only the shipped CSVs |

Under `auto` you can regenerate one experiment and still render everything; the
closing summary lists which CSVs fell back.

---

## Notes

- Benchmarks are treated as fixed inputs and are never regenerated. Every trace
  and CSV is produced by the run.
- Table 6 is wall-clock, so it reflects the machine that ran it. Use `--jobs 1`
  for numbers comparable to the paper's single-core measurement.
- Table 8 (the alpha/beta sensitivity sweep) is not covered by this artifact.

Please contact the authors with ANY issues that arise and we will be glad to
help.

Thank you again for reviewing this artifact.
