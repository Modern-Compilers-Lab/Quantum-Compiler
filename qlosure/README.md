# Qlosure Artifact

Hello,

Thank you for reviewing the artifact for **Qlosure**.

This artifact is designed to invoke Qlosure to run the tests that generated Figures 5, 6, and 7, as well as Tables 2, 3, and 4 from the main body of the paper.

The main wrapper script, **`run_all_experiments.py`**, will execute the entire experimentation pipeline ranging from benchmark loading, invoking Qlosure, to plotting—automatically. Reviewers may also run individual experiments using:

-   **`run_circuit.py`**: to run Qlosure on a single circuit
-   **`run_benchmark.py`**: to run Qlosure on all circuits of a specific benchmark

A **`requirements.txt`** file is provided to set up the Python environment.

---

## Repository layout

```
qlosure/
├─ baselines/            # Baseline mappers: SABRE, TKET, QMAP, Cirq
├─ benchmarks/
│  └─ polyhedral/        # IR generated via QRANE
├─ qpu/                  # JSON hardware topologies + loader script
├─ scripts/              # Bash helpers (incl. Slurm job scripts)
├─ src/                  # Main Qlosure framework code
├─ visualization/        # Plotting and analysis of experiment results
├─ tmp_results/          # (Created at runtime) logs & outputs
├─ run_baseline.py       # CLI entrypoint for running the baseline methods
├─ run_benchmark.py      # CLI entrypoint for running Qlosure on a specific benchmark circuits
├─ run_circuit.py        # CLI entrypoint for running Qlosure on a single circuit
├─ requirements.txt
└─ README.md
```

---

## Installation

### 1) Create and activate a virtual environment (qlosure-env)

```bash
conda create -n qlosure-env python=3.10.12 -y
conda activate qlosure-env
```

### 2) Install Python dependencies

```bash
pip install -r requirements.txt
```

> If you run into build issues for packages with native wheels (e.g., `islpy`), you likely need a **C/C++ compiler toolchain** installed (plus recent build tooling). Install it, then retry:
>
> **Linux**: `sudo apt-get update && sudo apt-get install -y build-essential python3-dev`

---

## Quick start

### Run on a single circuit (QUEKO 16 qubits) with the IBM Sherbrooke backend

```bash
python run_circuit.py --circuit=benchmarks/polyhedral/queko-bss-16qbt/16QBT_100CYC_QSE_0.json --backend=ibm_sherbrooke
```

### Run the a 54-qubit QUEKO circuit with multiple iterations (e.g., 3 forward/backward passes) to obtain a good initial mapping

```bash
python run_circuit.py --circuit=benchmarks/polyhedral/queko-bss-54qbt/54QBT_100CYC_QSE_0.json --backend=ibm_sherbrooke --num_iterations=3
```

### Run the circuit and also compare against competitors/baselines

```bash
python run_circuit.py --circuit=benchmarks/polyhedral/queko-bss-54qbt/54QBT_100CYC_QSE_0.json --backend=ibm_sherbrooke --competitors
```

### Run Benchmark on a specific backend topology

```bash
python run_benchmark.py --benchmark=queko-bss-16qbt --backend=ibm_sherbrooke
```

Please contact the authors with ANY issues that arise and we will be glad to help.

Thank you again for reviewing this artifact.
