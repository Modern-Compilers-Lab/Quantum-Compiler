import argparse
import json
import os
import sys
import time
from pathlib import Path

# Add parent directory to path for shared qpu package
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from qiskit import qasm3

from src.routing import Qlosure
from src.dag import build_dag, extract_multi_qubit_dag
from src.evaluation import compute_max_swaps_count, compute_quantum_depth

from qpu.src.load_backend import load_backend_data
from src.backend import QuantumBackend

from tqdm import tqdm


# Argument parser setup
parser = argparse.ArgumentParser(
    description="Run Qlosure with optional parameters")
parser.add_argument("--bench", type=str,
                    default="54qbt" ,
                    choices=["54qbt", "81qbt", "121qbt"],
                    help="Benchmark folder name either 54qbt, 81qbt, or 121qbt")
parser.add_argument("--backend", type=str,
                    default="ibm_kingston", choices=["ibm_kingston", "ibm_brisbane","ibm_brisbane_old"],
                    help="Name of the backend (either ibm_kingston or ibm_brisbane or ibm_brisbane_old)")
parser.add_argument("--initial", type=str, default="trivial",
                    help="Initial mapping method")
parser.add_argument("--verbose", type=int, default=0, help="Verbosity level")
parser.add_argument("--num_iterations", type=int, default=1,
                    help="number of bidirectional passes")
parser.add_argument("--template", type=str, default="nest0",
                    choices=["nest0", "nest1", "nest2", "if_else_inside_for","one_loop","one-if_inside_loop"],
                    help="Template type for D-QuEKO circuits")
args = parser.parse_args()

d_queko_benchmarks_dir = Path(f"../d-queko/benchmarks/{args.template}/{args.bench}")
results_dir = Path(f"results/qroqi/{args.template}/{args.backend}/{args.bench}")

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
    },
    "ibm_brisbane_old": {
        "54qbt": {
            "leaf-depth-10": "queko-054qbt_nest_00_nodes010_leaf-depth-10/circ_01.qasm",
            "leaf-depth-20": "queko-054qbt_nest_00_nodes010_leaf-depth-20/circ_08.qasm",
            "leaf-depth-30": "queko-054qbt_nest_00_nodes010_leaf-depth-30/circ_03.qasm",
            "leaf-depth-40": "queko-054qbt_nest_00_nodes010_leaf-depth-40/circ_02.qasm",
            "leaf-depth-50": "queko-054qbt_nest_00_nodes010_leaf-depth-50/circ_03.qasm",
            "leaf-depth-60": "queko-054qbt_nest_00_nodes010_leaf-depth-60/circ_08.qasm",
            "leaf-depth-70": "queko-054qbt_nest_00_nodes010_leaf-depth-70/circ_01.qasm",
            "leaf-depth-80": "queko-054qbt_nest_00_nodes010_leaf-depth-80/circ_06.qasm",
            "leaf-depth-90": "queko-054qbt_nest_00_nodes010_leaf-depth-90/circ_01.qasm"
        },
        "81qbt": {
            "leaf-depth-10": "queko-081qbt_nest_00_nodes010_leaf-depth-10/circ_06.qasm",
            "leaf-depth-20": "queko-081qbt_nest_00_nodes010_leaf-depth-20/circ_09.qasm",
            "leaf-depth-30": "queko-081qbt_nest_00_nodes010_leaf-depth-30/circ_04.qasm",
            "leaf-depth-40": "queko-081qbt_nest_00_nodes010_leaf-depth-40/circ_05.qasm",
            "leaf-depth-50": "queko-081qbt_nest_00_nodes010_leaf-depth-50/circ_03.qasm",
            "leaf-depth-60": "queko-081qbt_nest_00_nodes010_leaf-depth-60/circ_03.qasm",
            "leaf-depth-70": "queko-081qbt_nest_00_nodes010_leaf-depth-70/circ_02.qasm",
            "leaf-depth-80": "queko-081qbt_nest_00_nodes010_leaf-depth-80/circ_08.qasm",
            "leaf-depth-90": "queko-081qbt_nest_00_nodes010_leaf-depth-90/circ_09.qasm"
        },
        "121qbt": {
            "leaf-depth-10": "queko-121qbt_nest_00_nodes010_leaf-depth-10/circ_03.qasm",
            "leaf-depth-20": "queko-121qbt_nest_00_nodes010_leaf-depth-20/circ_02.qasm",
            "leaf-depth-30": "queko-121qbt_nest_00_nodes010_leaf-depth-30/circ_06.qasm",
            "leaf-depth-40": "queko-121qbt_nest_00_nodes010_leaf-depth-40/circ_08.qasm",
            "leaf-depth-50": "queko-121qbt_nest_00_nodes010_leaf-depth-50/circ_03.qasm",
            "leaf-depth-60": "queko-121qbt_nest_00_nodes010_leaf-depth-60/circ_05.qasm",
            "leaf-depth-70": "queko-121qbt_nest_00_nodes010_leaf-depth-70/circ_07.qasm",
            "leaf-depth-80": "queko-121qbt_nest_00_nodes010_leaf-depth-80/circ_03.qasm",
            "leaf-depth-90": "queko-121qbt_nest_00_nodes010_leaf-depth-90/circ_04.qasm"
        },


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

if args.template == "nest0":
    circuits_to_use = circuits_to_use_0
elif args.template == "nest1":
    circuits_to_use = circuits_to_use_nest1
elif args.template == "nest2":
    circuits_to_use = circuits_to_use_nest2
elif args.template == "if_else_inside_for":
    circuits_to_use = circuits_to_use_if_else
elif args.template == "one-if_inside_loop":
    circuits_to_use = circuits_to_use_default
elif args.template == "one_loop":
    circuits_to_use = circuits_to_use_default
else:
    print(f"❌ Unknown template type: {args.template}")
    exit(1)


def run_circuit(circuit_path,circuit_config, backend, initial_mapping, num_iterations , seed=40, verbose=0):

    try:
        # Load circuit
        qc = qasm3.load(circuit_path)

        dag = build_dag(qc)
        dag2q = extract_multi_qubit_dag(dag)

        # Run Qlosure
        poly_mapper = Qlosure(backend,seed=seed)
        start = time.time()
        qlosure_results = poly_mapper.run(
            dag, dag2q, initial_mapping=initial_mapping, num_iter=num_iterations, verbose=verbose)
        qlosure_end_time = time.time()

        # Get results
        trace = poly_mapper.get_structured_trace()

        # Save results
        circuit_path_obj = Path(circuit_path)
        circuit_name = circuit_path_obj.stem


        output_dir = results_dir / circuit_config /f"SEED_{seed}"
        output_dir.mkdir(parents=True, exist_ok=True)

        trace_json_path = output_dir / f"trace.json"
        time_txt_path = output_dir / f"time.txt"
        file_path = output_dir / f"path.txt"
        # save time

        with open(time_txt_path, "w") as f:
            f.write(f"{qlosure_end_time - start:.6f}\n")

        with open(trace_json_path, "w") as f:
            json.dump(trace, f, indent=2)
        with open(file_path, "w") as f:
            f.write(str(circuit_path))
        


        return True

    except Exception as e:
        print(f"❌ Error processing {circuit_path}: {str(e)}")
        return False, None, None


if not d_queko_benchmarks_dir.exists():
    print(f"❌ Benchmark directory {d_queko_benchmarks_dir} does not exist!")
    exit(1)


backend_data = load_backend_data(args.backend)
edges = backend_data["coupling_map"]
qubits_props = backend_data.get("qubits", {})

if not edges:
    print(f"\u274c No coupling information found for backend '{args.backend}'")
    exit(1)

backend = QuantumBackend(edges, qubit_props=qubits_props)

total_circuits = len(circuits_to_use[args.backend][args.bench])
processed = 0
successful = 0
failed = 0


print(
    f"\nFound {total_circuits} circuits in benchmark {args.bench}")

#  63, 84, 105, 126, 147, 168, 189
random_seeds = [3, 21, 42,]
for depth, circuit_rel_path in circuits_to_use[args.backend][args.bench].items():
    circuit_path = Path(d_queko_benchmarks_dir) / circuit_rel_path
    circuit_config = circuit_rel_path.split('/')[-2]

    print(f"\nProcessing circuit at depth {depth}: {circuit_path}")

    for seed in tqdm(random_seeds):
        success = run_circuit(
            circuit_path,circuit_config, backend, args.initial, args.num_iterations, seed=seed, verbose=args.verbose)

        processed += 1
        if success:
            successful += 1
        else:
            failed += 1
            print(f"❌ Failed to process {circuit_path} with seed {seed}.")

    
print(f"\n✅ Processing complete. {successful}/{processed} circuits processed successfully, {failed} failures.")