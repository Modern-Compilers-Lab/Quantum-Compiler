import subprocess
import sys


def run_benchmark(n_qubits, leaf_depth):
    """Run the d-queko benchmark with specified parameters"""
    cmd = [
        "python", "./generate-d-queko.py",
        "--nest-depth=1",
        f"--leaf-depth={leaf_depth}",
        "--top-len=10",
        f"--n-qubits={n_qubits}"
    ]

    print(f"Running: {' '.join(cmd)}")
    try:
        result = subprocess.run(
            cmd, check=True, capture_output=True, text=True)
        print(f"Success for {n_qubits} qubits, leaf_depth {leaf_depth}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error for {n_qubits} qubits, leaf_depth {leaf_depth}: {e}")
        return False


def main():
    # Qubit counts to test
    qubit_counts = [54, 81, 127]

    # Leaf depth range
    leaf_depths = range(10, 91, 10)  # 10, 20, 30, ..., 90

    total_runs = len(qubit_counts) * len(leaf_depths)
    current_run = 0

    for n_qubits in qubit_counts:
        for leaf_depth in leaf_depths:
            current_run += 1
            print(
                f"\n[{current_run}/{total_runs}] Testing {n_qubits} qubits with leaf_depth {leaf_depth}")
            run_benchmark(n_qubits, leaf_depth)


if __name__ == "__main__":
    main()
