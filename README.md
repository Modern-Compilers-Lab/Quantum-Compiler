# DynamiQ

Qubit mapping and routing for dynamic, hierarchical quantum circuits.

This branch is the reproduction artifact for the paper *Mapping Dynamic,
Hierarchical Quantum Circuits*. It contains the DynamiQ mapper, the benchmark
circuits, and a pipeline that regenerates every experimental figure and table.

**Start here: [`artifact/README.md`](artifact/README.md)**

```bash
conda env create -f artifact/environment.yml
conda activate dynamiq-artifact
python artifact/check_env.py
python artifact/render.py all          # figures and tables, seconds
```

## Layout

| path | contents |
|---|---|
| `artifact/` | reproduction pipeline and entrypoints |
| `dynamic-qlosure/src/` | the DynamiQ mapper |
| `d-queko/benchmarks/` | d-QUEKO dynamic circuits |
| `surface-code/` | Stim surface-code circuits |
| `qpu/topologies/` | backend coupling maps and calibration |
