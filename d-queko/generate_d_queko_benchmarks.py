import subprocess
import sys


def run_benchmark(n_qubits, leaf_depth,w=1,i=1,dir_name="benchmarks"):
    """Run the d-queko benchmark with specified parameters"""
    cmd = [
        "python", "./generate-d-queko.py",
        "--nest-depth=0",
        f"--leaf-depth={leaf_depth}",
        f"--n-qubits={n_qubits}",
        f"--template=wi_rule",
        "--replicates=3",
        "--top-len=1",
        f"--output-dir=wi_rule_benchmarks/{n_qubits}qbt/{leaf_depth}Leaf_depth/{w}_{i}",
        f"--w={w}",
        f"--i={i}",
    ]

    print(f"Running: {' '.join(cmd)}")
    try:
        result = subprocess.run(
            cmd, check=True, capture_output=True, text=True)
        print(f"Success for {n_qubits} qubits, leaf_depth {leaf_depth}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error for {n_qubits} qubits, leaf_depth {leaf_depth}: {e}")
        print(f"Stdout: {e.stdout}")
        return False


def main():
    # Qubit counts to test
    qubit_counts = [81,121]

    # Leaf depth range
    leaf_depths = [10] # 10, 20, 30, ..., 90
    w_vals = range(1,6)
    i_vals = range(1,6)

    total_runs = len(qubit_counts) * len(leaf_depths)
    current_run = 0

    for n_qubits in qubit_counts:
        for leaf_depth in leaf_depths:
            for w in w_vals:
                for i in i_vals:
                    current_run += 1
                    print(
                        f"\n[{current_run}/{total_runs}] Testing {n_qubits} qubits with leaf_depth {leaf_depth}")
                    run_benchmark(n_qubits, leaf_depth, w=w, i=i)


if __name__ == "__main__":
    main()
