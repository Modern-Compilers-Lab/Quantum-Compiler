import argparse
import json
import time
from pathlib import Path

from qiskit import qasm3
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.transpiler import PassManager, Layout, CouplingMap
from qiskit.transpiler.passes import SabreSwap

from src.backend import QuantumBackend
from src.parser import build_structured_trace_from_circuit

from tqdm import tqdm

# Argument parser setup
parser = argparse.ArgumentParser(
    description="Run SABRE routing with optional parameters")
parser.add_argument("--bench", type=str,
                    default="54qbt",
                    choices=["54qbt", "81qbt", "121qbt"],
                    help="Benchmark folder name either 54qbt, 81qbt, or 121qbt")
parser.add_argument("--backend", type=str,
                    default="ibm_kingston", choices=["ibm_kingston", "ibm_brisbane", "ibm_brisbane_old"],
                    help="Name of the backend (either ibm_kingston or ibm_brisbane)")
parser.add_argument("--initial", type=str, default="trivial",
                    help="Initial mapping method (currently only 'trivial' supported)")
parser.add_argument("--verbose", type=int, default=0, help="Verbosity level")
parser.add_argument("--num_iterations", type=int, default=1,
                    help="Number of bidirectional passes (unused for SABRE but kept for CLI compat)")
parser.add_argument("--template", type=str, default="nest0",
                    choices=["nest0", "nest1", "nest2", "if_else_inside_for","one_loop","one-if_inside_loop"],
                    help="Template type for D-QuEKO circuits")
args = parser.parse_args()

# Derived paths
d_queko_benchmarks_dir = Path(f"../d-queko/benchmarks/{args.template}/{args.bench}")
backend_dir = Path(f"../d-queko/qpu/topologies/{args.backend}.json")
results_dir = Path(f"results/sabre/{args.template}/{args.backend}/{args.bench}")

# Pre-selected circuits for each backend/bench/depth
# queko-121qbt_nest_00_nodes010_leaf-depth-10/circ_00.qasm
circuits_to_use_0 = {
    "ibm_brisbane": {
        "121qbt": {
            "leaf-depth-10": "queko-121qbt_nest_00_nodes010_leaf-depth-10/circ_03.qasm",
            "leaf-depth-20": "queko-121qbt_nest_00_nodes010_leaf-depth-20/circ_07.qasm",
            "leaf-depth-30": "queko-121qbt_nest_00_nodes010_leaf-depth-30/circ_04.qasm",
            "leaf-depth-40": "queko-121qbt_nest_00_nodes010_leaf-depth-40/circ_08.qasm",
            "leaf-depth-50": "queko-121qbt_nest_00_nodes010_leaf-depth-50/circ_09.qasm",
            "leaf-depth-60": "queko-121qbt_nest_00_nodes010_leaf-depth-60/circ_01.qasm",
            "leaf-depth-70": "queko-121qbt_nest_00_nodes010_leaf-depth-70/circ_03.qasm",
            "leaf-depth-80": "queko-121qbt_nest_00_nodes010_leaf-depth-80/circ_03.qasm",
            "leaf-depth-90": "queko-121qbt_nest_00_nodes010_leaf-depth-90/circ_04.qasm"
        },
        "81qbt": {
            "leaf-depth-10": "queko-081qbt_nest_00_nodes010_leaf-depth-10/circ_08.qasm",
            "leaf-depth-20": "queko-081qbt_nest_00_nodes010_leaf-depth-20/circ_09.qasm",
            "leaf-depth-30": "queko-081qbt_nest_00_nodes010_leaf-depth-30/circ_06.qasm",
            "leaf-depth-40": "queko-081qbt_nest_00_nodes010_leaf-depth-40/circ_05.qasm",
            "leaf-depth-50": "queko-081qbt_nest_00_nodes010_leaf-depth-50/circ_09.qasm",
            "leaf-depth-60": "queko-081qbt_nest_00_nodes010_leaf-depth-60/circ_03.qasm",
            "leaf-depth-70": "queko-081qbt_nest_00_nodes010_leaf-depth-70/circ_02.qasm",
            "leaf-depth-80": "queko-081qbt_nest_00_nodes010_leaf-depth-80/circ_04.qasm",
            "leaf-depth-90": "queko-081qbt_nest_00_nodes010_leaf-depth-90/circ_09.qasm"
        },
        "54qbt": {
            "leaf-depth-10": "queko-054qbt_nest_00_nodes010_leaf-depth-10/circ_00.qasm",
            "leaf-depth-20": "queko-054qbt_nest_00_nodes010_leaf-depth-20/circ_03.qasm",
            "leaf-depth-30": "queko-054qbt_nest_00_nodes010_leaf-depth-30/circ_08.qasm",
            "leaf-depth-40": "queko-054qbt_nest_00_nodes010_leaf-depth-40/circ_02.qasm",
            "leaf-depth-50": "queko-054qbt_nest_00_nodes010_leaf-depth-50/circ_03.qasm",
            "leaf-depth-60": "queko-054qbt_nest_00_nodes010_leaf-depth-60/circ_08.qasm",
            "leaf-depth-70": "queko-054qbt_nest_00_nodes010_leaf-depth-70/circ_01.qasm",
            "leaf-depth-80": "queko-054qbt_nest_00_nodes010_leaf-depth-80/circ_06.qasm",
            "leaf-depth-90": "queko-054qbt_nest_00_nodes010_leaf-depth-90/circ_01.qasm"
        }
    },
    "ibm_kingston": {
        "54qbt": {
            "leaf-depth-10": "queko-054qbt_nest_00_nodes010_leaf-depth-10/circ_09.qasm",
            "leaf-depth-20": "queko-054qbt_nest_00_nodes010_leaf-depth-20/circ_05.qasm",
            "leaf-depth-30": "queko-054qbt_nest_00_nodes010_leaf-depth-30/circ_07.qasm",
            "leaf-depth-40": "queko-054qbt_nest_00_nodes010_leaf-depth-40/circ_01.qasm",
            "leaf-depth-50": "queko-054qbt_nest_00_nodes010_leaf-depth-50/circ_07.qasm",
            "leaf-depth-60": "queko-054qbt_nest_00_nodes010_leaf-depth-60/circ_00.qasm",
            "leaf-depth-70": "queko-054qbt_nest_00_nodes010_leaf-depth-70/circ_09.qasm",
            "leaf-depth-80": "queko-054qbt_nest_00_nodes010_leaf-depth-80/circ_03.qasm",
            "leaf-depth-90": "queko-054qbt_nest_00_nodes010_leaf-depth-90/circ_02.qasm"
        },
        "81qbt": {
            "leaf-depth-10": "queko-081qbt_nest_00_nodes010_leaf-depth-10/circ_01.qasm",
            "leaf-depth-20": "queko-081qbt_nest_00_nodes010_leaf-depth-20/circ_02.qasm",
            "leaf-depth-30": "queko-081qbt_nest_00_nodes010_leaf-depth-30/circ_04.qasm",
            "leaf-depth-40": "queko-081qbt_nest_00_nodes010_leaf-depth-40/circ_04.qasm",
            "leaf-depth-50": "queko-081qbt_nest_00_nodes010_leaf-depth-50/circ_02.qasm",
            "leaf-depth-60": "queko-081qbt_nest_00_nodes010_leaf-depth-60/circ_08.qasm",
            "leaf-depth-70": "queko-081qbt_nest_00_nodes010_leaf-depth-70/circ_07.qasm",
            "leaf-depth-80": "queko-081qbt_nest_00_nodes010_leaf-depth-80/circ_07.qasm",
            "leaf-depth-90": "queko-081qbt_nest_00_nodes010_leaf-depth-90/circ_07.qasm"
        },
        "121qbt": {
            "leaf-depth-10": "queko-121qbt_nest_00_nodes010_leaf-depth-10/circ_09.qasm",
            "leaf-depth-20": "queko-121qbt_nest_00_nodes010_leaf-depth-20/circ_02.qasm",
            "leaf-depth-30": "queko-121qbt_nest_00_nodes010_leaf-depth-30/circ_01.qasm",
            "leaf-depth-40": "queko-121qbt_nest_00_nodes010_leaf-depth-40/circ_00.qasm",
            "leaf-depth-50": "queko-121qbt_nest_00_nodes010_leaf-depth-50/circ_05.qasm",
            "leaf-depth-60": "queko-121qbt_nest_00_nodes010_leaf-depth-60/circ_02.qasm",
            "leaf-depth-70": "queko-121qbt_nest_00_nodes010_leaf-depth-70/circ_09.qasm",
            "leaf-depth-80": "queko-121qbt_nest_00_nodes010_leaf-depth-80/circ_07.qasm",
            "leaf-depth-90": "queko-121qbt_nest_00_nodes010_leaf-depth-90/circ_06.qasm"
        }
    }
}

circuits_to_use_if_else = {
    "ibm_brisbane_old": {
        "81qbt": {
            "leaf-depth-10": "queko-081qbt_nest_00_nodes010_leaf-depth-10/circ_00.qasm",
            "leaf-depth-20": "queko-081qbt_nest_00_nodes010_leaf-depth-20/circ_07.qasm",
            "leaf-depth-30": "queko-081qbt_nest_00_nodes010_leaf-depth-30/circ_08.qasm",
            "leaf-depth-40": "queko-081qbt_nest_00_nodes010_leaf-depth-40/circ_01.qasm",
            "leaf-depth-50": "queko-081qbt_nest_00_nodes010_leaf-depth-50/circ_09.qasm",
            "leaf-depth-60": "queko-081qbt_nest_00_nodes010_leaf-depth-60/circ_01.qasm",
            "leaf-depth-70": "queko-081qbt_nest_00_nodes010_leaf-depth-70/circ_01.qasm",
            "leaf-depth-80": "queko-081qbt_nest_00_nodes010_leaf-depth-80/circ_05.qasm",
            "leaf-depth-90": "queko-081qbt_nest_00_nodes010_leaf-depth-90/circ_05.qasm"
        },
     },
}

circuits_to_use_nest1 = {
    "ibm_brisbane_old": {
        "81qbt": {
            "leaf-depth-10": "queko-081qbt_nest_01_nodes005_leaf-depth-10/circ_08.qasm",
            "leaf-depth-20": "queko-081qbt_nest_01_nodes005_leaf-depth-20/circ_09.qasm",
            "leaf-depth-30": "queko-081qbt_nest_01_nodes005_leaf-depth-30/circ_06.qasm",
            "leaf-depth-40": "queko-081qbt_nest_01_nodes005_leaf-depth-40/circ_00.qasm",
            "leaf-depth-50": "queko-081qbt_nest_01_nodes005_leaf-depth-50/circ_04.qasm",
            "leaf-depth-60": "queko-081qbt_nest_01_nodes005_leaf-depth-60/circ_02.qasm",
            "leaf-depth-70": "queko-081qbt_nest_01_nodes005_leaf-depth-70/circ_01.qasm",
            "leaf-depth-80": "queko-081qbt_nest_01_nodes005_leaf-depth-80/circ_07.qasm",
            "leaf-depth-90": "queko-081qbt_nest_01_nodes005_leaf-depth-90/circ_09.qasm"
        },
     },
}

circuits_to_use_nest2 = {
    "ibm_brisbane_old": {
        "81qbt": {
            "leaf-depth-10": "queko-081qbt_nest_02_nodes002_leaf-depth-10/circ_03.qasm",
            "leaf-depth-20": "queko-081qbt_nest_02_nodes002_leaf-depth-20/circ_02.qasm",
            "leaf-depth-30": "queko-081qbt_nest_02_nodes002_leaf-depth-30/circ_01.qasm",
            "leaf-depth-40": "queko-081qbt_nest_02_nodes002_leaf-depth-40/circ_01.qasm",
            "leaf-depth-50": "queko-081qbt_nest_02_nodes002_leaf-depth-50/circ_00.qasm",
            "leaf-depth-60": "queko-081qbt_nest_02_nodes002_leaf-depth-60/circ_01.qasm",
            "leaf-depth-70": "queko-081qbt_nest_02_nodes002_leaf-depth-70/circ_03.qasm",
            "leaf-depth-80": "queko-081qbt_nest_02_nodes002_leaf-depth-80/circ_02.qasm",
            "leaf-depth-90": "queko-081qbt_nest_02_nodes002_leaf-depth-90/circ_01.qasm"
        },
     },
}

circuits_to_use_default = {
    "ibm_brisbane_old": {
        "81qbt": {
            "leaf-depth-10": "queko-081qbt_nest_00_nodes001_leaf-depth-10/circ_00.qasm",
            "leaf-depth-20": "queko-081qbt_nest_00_nodes001_leaf-depth-20/circ_00.qasm",
            "leaf-depth-30": "queko-081qbt_nest_00_nodes001_leaf-depth-30/circ_00.qasm",
            "leaf-depth-40": "queko-081qbt_nest_00_nodes001_leaf-depth-40/circ_00.qasm",
            "leaf-depth-50": "queko-081qbt_nest_00_nodes001_leaf-depth-50/circ_00.qasm",
            "leaf-depth-60": "queko-081qbt_nest_00_nodes001_leaf-depth-60/circ_00.qasm",
            "leaf-depth-70": "queko-081qbt_nest_00_nodes001_leaf-depth-70/circ_00.qasm",
            "leaf-depth-80": "queko-081qbt_nest_00_nodes001_leaf-depth-80/circ_00.qasm",
            "leaf-depth-90": "queko-081qbt_nest_00_nodes001_leaf-depth-90/circ_00.qasm"
        },
        "54qbt": {
            "leaf-depth-10": "queko-054qbt_nest_00_nodes001_leaf-depth-10/circ_00.qasm",
            "leaf-depth-20": "queko-054qbt_nest_00_nodes001_leaf-depth-20/circ_00.qasm",
            "leaf-depth-30": "queko-054qbt_nest_00_nodes001_leaf-depth-30/circ_00.qasm",
            "leaf-depth-40": "queko-054qbt_nest_00_nodes001_leaf-depth-40/circ_00.qasm",
            "leaf-depth-50": "queko-054qbt_nest_00_nodes001_leaf-depth-50/circ_00.qasm",
            "leaf-depth-60": "queko-054qbt_nest_00_nodes001_leaf-depth-60/circ_00.qasm",
            "leaf-depth-70": "queko-054qbt_nest_00_nodes001_leaf-depth-70/circ_00.qasm",
            "leaf-depth-80": "queko-054qbt_nest_00_nodes001_leaf-depth-80/circ_00.qasm",
            "leaf-depth-90": "queko-054qbt_nest_00_nodes001_leaf-depth-90/circ_00.qasm"
        },
        "121qbt": {
            "leaf-depth-10": "queko-121qbt_nest_00_nodes001_leaf-depth-10/circ_00.qasm",
            "leaf-depth-20": "queko-121qbt_nest_00_nodes001_leaf-depth-20/circ_00.qasm",
            "leaf-depth-30": "queko-121qbt_nest_00_nodes001_leaf-depth-30/circ_00.qasm",
            "leaf-depth-40": "queko-121qbt_nest_00_nodes001_leaf-depth-40/circ_00.qasm",
            "leaf-depth-50": "queko-121qbt_nest_00_nodes001_leaf-depth-50/circ_00.qasm",
            "leaf-depth-60": "queko-121qbt_nest_00_nodes001_leaf-depth-60/circ_00.qasm",
            "leaf-depth-70": "queko-121qbt_nest_00_nodes001_leaf-depth-70/circ_00.qasm",
            "leaf-depth-80": "queko-121qbt_nest_00_nodes001_leaf-depth-80/circ_00.qasm",
            "leaf-depth-90": "queko-121qbt_nest_00_nodes001_leaf-depth-90/circ_00.qasm"
        }
     },
     "ibm_kingston": {
        "81qbt": {
            "leaf-depth-10": "queko-081qbt_nest_00_nodes001_leaf-depth-10/circ_00.qasm",
            "leaf-depth-20": "queko-081qbt_nest_00_nodes001_leaf-depth-20/circ_00.qasm",
            "leaf-depth-30": "queko-081qbt_nest_00_nodes001_leaf-depth-30/circ_00.qasm",
            "leaf-depth-40": "queko-081qbt_nest_00_nodes001_leaf-depth-40/circ_00.qasm",
            "leaf-depth-50": "queko-081qbt_nest_00_nodes001_leaf-depth-50/circ_00.qasm",
            "leaf-depth-60": "queko-081qbt_nest_00_nodes001_leaf-depth-60/circ_00.qasm",
            "leaf-depth-70": "queko-081qbt_nest_00_nodes001_leaf-depth-70/circ_00.qasm",
            "leaf-depth-80": "queko-081qbt_nest_00_nodes001_leaf-depth-80/circ_00.qasm",
            "leaf-depth-90": "queko-081qbt_nest_00_nodes001_leaf-depth-90/circ_00.qasm"
        },
        "54qbt": {
            "leaf-depth-10": "queko-054qbt_nest_00_nodes001_leaf-depth-10/circ_00.qasm",
            "leaf-depth-20": "queko-054qbt_nest_00_nodes001_leaf-depth-20/circ_00.qasm",
            "leaf-depth-30": "queko-054qbt_nest_00_nodes001_leaf-depth-30/circ_00.qasm",
            "leaf-depth-40": "queko-054qbt_nest_00_nodes001_leaf-depth-40/circ_00.qasm",
            "leaf-depth-50": "queko-054qbt_nest_00_nodes001_leaf-depth-50/circ_00.qasm",
            "leaf-depth-60": "queko-054qbt_nest_00_nodes001_leaf-depth-60/circ_00.qasm",
            "leaf-depth-70": "queko-054qbt_nest_00_nodes001_leaf-depth-70/circ_00.qasm",
            "leaf-depth-80": "queko-054qbt_nest_00_nodes001_leaf-depth-80/circ_00.qasm",
            "leaf-depth-90": "queko-054qbt_nest_00_nodes001_leaf-depth-90/circ_00.qasm"
        },
        "121qbt": {
            "leaf-depth-10": "queko-121qbt_nest_00_nodes001_leaf-depth-10/circ_00.qasm",
            "leaf-depth-20": "queko-121qbt_nest_00_nodes001_leaf-depth-20/circ_00.qasm",
            "leaf-depth-30": "queko-121qbt_nest_00_nodes001_leaf-depth-30/circ_00.qasm",
            "leaf-depth-40": "queko-121qbt_nest_00_nodes001_leaf-depth-40/circ_00.qasm",
            "leaf-depth-50": "queko-121qbt_nest_00_nodes001_leaf-depth-50/circ_00.qasm",
            "leaf-depth-60": "queko-121qbt_nest_00_nodes001_leaf-depth-60/circ_00.qasm",
            "leaf-depth-70": "queko-121qbt_nest_00_nodes001_leaf-depth-70/circ_00.qasm",
            "leaf-depth-80": "queko-121qbt_nest_00_nodes001_leaf-depth-80/circ_00.qasm",
            "leaf-depth-90": "queko-121qbt_nest_00_nodes001_leaf-depth-90/circ_00.qasm"
        }
     },


}


circuits_to_use = None
if args.template == "nest1":
    circuits_to_use = circuits_to_use_nest1
elif args.template == "nest2":
    circuits_to_use = circuits_to_use_nest2
elif args.template == "if_else_inside_for":
    circuits_to_use = circuits_to_use_if_else
elif args.template == "one-if_inside_loop":
    circuits_to_use = circuits_to_use_default
elif args.template == "one_loop":
    circuits_to_use = circuits_to_use_default
    

def infer_num_qubits(edges, qubits_props):
    if qubits_props:
        return len(qubits_props)
    # Fallback: infer from edge list
    if not edges:
        return 0
    max_index = 0
    for u, v in edges:
        max_index = max(max_index, u, v)
    return max_index + 1


def run_circuit(circuit_path, circuit_config, coupling_map, num_phys_qubits, seed=40, verbose=0):
    try:
        # Load circuit
        qc = qasm3.load(circuit_path)

        if num_phys_qubits < len(qc.qubits):
            raise ValueError(
                f"Backend has {num_phys_qubits} qubits but circuit needs {len(qc.qubits)}.")

        # Build a PHYSICAL circuit on a full-size device register
        qr = QuantumRegister(num_phys_qubits, "q")
        mapped_circuit = QuantumCircuit(qr, *qc.cregs, name=qc.name)
        mapped_circuit.global_phase = getattr(qc, "global_phase", 0)
        mapped_circuit.metadata = getattr(qc, "metadata", None)

        # Trivial explicit layout: vq_i -> p_i
        computed_layout = Layout({vq: i for i, vq in enumerate(qc.qubits)})

        # Remap each instruction's qubit args onto the physical register
        for instr, qargs, cargs in qc.data:
            new_qargs = [qr[computed_layout[vq]] for vq in qargs]
            mapped_circuit.append(instr, new_qargs, cargs)

        pm = PassManager([
            SabreSwap(coupling_map=coupling_map, heuristic="decay", seed=seed, trials=1)
        ])

        start = time.time()
        routed_qc = pm.run(mapped_circuit)
        end = time.time()

        # Build structured trace from the routed circuit
        trace = build_structured_trace_from_circuit(routed_qc, decompose=False)

        # Save results
        output_dir = results_dir / circuit_config / f"SEED_{seed}"
        output_dir.mkdir(parents=True, exist_ok=True)

        (output_dir / "time.txt").write_text(f"{end - start:.6f}")
        (output_dir / "path.txt").write_text(str(circuit_path))
        with open(output_dir / "trace.json", "w") as f:
            json.dump(trace, f, indent=2)

        return True

    except Exception as e:
        print(f"❌ Error processing {circuit_path}: {str(e)}")
        return False


# Sanity checks on inputs/paths
if not d_queko_benchmarks_dir.exists():
    print(f"❌ Benchmark directory {d_queko_benchmarks_dir} does not exist!")
    raise SystemExit(1)

if not backend_dir.exists():
    print(f"❌ Backend topology file not found: {backend_dir}")
    raise SystemExit(1)

# Load backend topology JSON
with open(backend_dir, 'r', encoding="utf-8") as fp:
    backend_topology = json.load(fp)
    edges = backend_topology.get("coupling_map", [])
    qubits_props = backend_topology.get("qubits", {})

    if not edges:
        print(f"❌ No coupling information found in {backend_dir}")
        raise SystemExit(1)
    if qubits_props is None:
        qubits_props = {}

num_phys_qubits = infer_num_qubits(edges, qubits_props)
coupling_map = CouplingMap(edges)

# Summary and loop
selected = circuits_to_use[args.backend][args.bench]
print(f"Found {len(selected)} circuits in benchmark {args.bench}")

processed = 0
successful = 0
failed = 0

random_seeds = [3, 21, 42, 63, 84, 105, 126, 147, 168, 189]

for depth, circuit_rel_path in selected.items():
    circuit_path = d_queko_benchmarks_dir / circuit_rel_path
    circuit_config = circuit_rel_path.split('/')[-2]

    print(f"Processing circuit at depth {depth}: {circuit_path}")

    for seed in tqdm(random_seeds):
        ok = run_circuit(
            circuit_path, circuit_config, coupling_map, num_phys_qubits, seed=seed, verbose=args.verbose
        )

        processed += 1
        if ok:
            successful += 1
        else:
            failed += 1
            print(f"❌ Failed to process {circuit_path} with seed {seed}.")

print(f"✅ Processing complete. {successful}/{processed} circuits processed successfully, {failed} failures.")
