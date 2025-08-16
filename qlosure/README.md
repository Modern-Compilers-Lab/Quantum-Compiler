# Qlosure

Implementation of a qubit mapping algorithm based on transitive dependence weights using affine abstractions. Evaluated on IBM and Rigetti QPUs with QUEKO and QASMBench, showing improvements in circuit depth and SWAP count over QMAP, Sabre, Cirq, and TKET.

---

## Repository layout

```
qlosure/
├─ baselines/            # Baseline mappers: SABRE, TKET, QMAP, Cirq
├─ benchmarks/
│  ├─ qasm/              # Raw OpenQASM circuits
│  └─ polyhedral/        # IR generated via QRANE
├─ qpu/                  # JSON hardware topologies + loader script
├─ scripts/              # Bash helpers (incl. Slurm job scripts)
├─ src/                  # Main Qlosure framework code
├─ visualization/        # Plotting and analysis of experiment results
├─ tmp_results/          # (Created at runtime) logs & outputs
├─ main.py               # CLI entrypoint for running experiments
├─ requirements.txt
└─ README.md
```

---

## Installation

### 1) Create and activate a virtual environment (venv)

```bash
python3 -m venv .venv
source .venv/bin/activate
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
python main.py --circuit=benchmarks/polyhedral/queko-bss-16qbt/16QBT_100CYC_QSE_0.json --backend=ibm_sherbrooke
```

### Run the a 54-qubit QUEKO circuit with multiple iterations (e.g., 3 forward/backward passes) to obtain a good initial mapping

```bash
python main.py --circuit=benchmarks/polyhedral/queko-bss-54qbt/54QBT_100CYC_QSE_0.json --backend=ibm_sherbrooke --num_iterations=3
```

### Run the circuit and also compare against competitors/baselines

```bash
python main.py --circuit=benchmarks/polyhedral/queko-bss-54qbt/54QBT_100CYC_QSE_0.json --backend=ibm_sherbrooke --competitors
```
