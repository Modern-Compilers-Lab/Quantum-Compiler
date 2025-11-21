import os
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from src.evaluation import compute_max_swaps_count, compute_quantum_depth, estimate_dynamic_circuit

def compute_routing_metrics_to_csv(nb_qubits, nb_loop_iterations, topology_name, output_csv_path , template="nest0"):
    """
    Compute average + std stats for both methods and save results as CSV.
    """
    with open(f"/scratch/mb10324/Quantum-Compiler/d-queko/qpu/topologies/{topology_name}.json", 'r', encoding="utf-8") as fp:
        ibm_topology = json.load(fp)

    results_dir = f'xxresults_seeds/qroqi/nest2/{topology_name}/{nb_qubits}qbt'
    sabre_dir = f'xxresults_seeds/sabre/nest2/{topology_name}/{nb_qubits}qbt'

    def collect_json_files(root_dir):
        files = []
        for root, _, filenames in os.walk(root_dir):
            for f in filenames:
                if f.endswith(".json"):
                    files.append(os.path.join(root, f))
        return files

    files = collect_json_files(results_dir)
    sabre_files = collect_json_files(sabre_dir)

    def process_files(file_list):
        data = []
        for file in tqdm(file_list):
            with open(file, 'r') as fp:
                trace = json.load(fp)
            swaps = compute_max_swaps_count(trace, loop_iterations=nb_loop_iterations)
            depth = compute_quantum_depth(trace, loop_iterations=nb_loop_iterations)
            latency, error = estimate_dynamic_circuit(trace, qubit_props=ibm_topology["qubits"], loop_iterations=nb_loop_iterations).values()
            for part in file.split(os.sep):
                if "leaf-depth-" in part:
                    leaf_depth = int(part.split("leaf-depth-")[-1])
                    data.append((leaf_depth, swaps, depth, latency, error))
                    break
        return pd.DataFrame(data, columns=["leaf_depth", "swaps", "depth", "latency", "error"])

    df_ours = process_files(files)
    df_sabre = process_files(sabre_files)

    # Compute mean and std for both
    df_summary = df_ours.groupby("leaf_depth").agg(["mean", "std"]).reset_index()
    df_summary.columns = ["leaf_depth"] + [f"{col}_{stat}_ours" for col, stat in df_summary.columns[1:]]

    df_summary_sabre = df_sabre.groupby("leaf_depth").agg(["mean", "std"]).reset_index()
    df_summary_sabre.columns = ["leaf_depth"] + [f"{col}_{stat}_sabre" for col, stat in df_summary_sabre.columns[1:]]

    df_final = pd.merge(df_summary, df_summary_sabre, on="leaf_depth", how="outer").sort_values("leaf_depth")

    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    df_final.to_csv(output_csv_path, index=False)
    print(f"✅ Metrics saved to {output_csv_path}")
    return df_final


def compute_routing_metrics_ablation_to_csv(nb_qubits, nb_loop_iterations, topology_name, output_csv_path , template="nest0"):
    """
    Compute average + std stats for both methods and save results as CSV.
    """
    with open(f"/scratch/mb10324/Quantum-Compiler/d-queko/qpu/topologies/{topology_name}.json", 'r', encoding="utf-8") as fp:
        ibm_topology = json.load(fp)

    results_dir0 = f'results_seeds/{topology_name}/{template}/{nb_qubits}qbt'
    results_dir1 = f'results_seeds_no_remaping/{topology_name}/{template}/{nb_qubits}qbt'
    results_dir2 = f'results_seeds_no_remaping_no_error/{topology_name}/{template}/{nb_qubits}qbt'
    results_dir3 = f'results_no_remap_no_error_with_depth_rate_with_error/qroqi/{template}/{topology_name}/{nb_qubits}qbt'
    # results_dir4 = f'results_seeds_depth_rate_with_error/{topology_name}/{template}/{nb_qubits}qbt'
    # results_dir5 = f'results_seeds_depth_rate_without_error/{topology_name}/{template}/{nb_qubits}qbt'
    # results_dir5 = f'results_seeds_depth_rate_without_error/{topology_name}/{template}/{nb_qubits}qbt'


    def collect_json_files(root_dir):
        files = []
        for root, _, filenames in os.walk(root_dir):
            for f in filenames:
                if f.endswith(".json"):
                    files.append(os.path.join(root, f))
        return files

    files_default = collect_json_files(results_dir0)
    files_no_remap = collect_json_files(results_dir1)
    files_no_remap_no_error = collect_json_files(results_dir2)
    files_new_line = collect_json_files(results_dir3)

    def process_files(file_list):
        data = []
        for file in tqdm(file_list):
            with open(file, 'r') as fp:
                trace = json.load(fp)
            swaps = compute_max_swaps_count(trace, loop_iterations=nb_loop_iterations)
            depth = compute_quantum_depth(trace, loop_iterations=nb_loop_iterations)
            latency, error = estimate_dynamic_circuit(trace, qubit_props=ibm_topology["qubits"], loop_iterations=nb_loop_iterations).values()
            for part in file.split(os.sep):
                if "leaf-depth-" in part:
                    leaf_depth = int(part.split("leaf-depth-")[-1])
                    data.append((leaf_depth, swaps, depth, latency, error))
                    break
        return pd.DataFrame(data, columns=["leaf_depth", "swaps", "depth", "latency", "error"])

    df_default = process_files(files_default)
    df_no_remap = process_files(files_no_remap)
    df_no_remap_no_error = process_files(files_no_remap_no_error)
    df_new_line = process_files(files_new_line)

    # Compute mean and std for both
    df_summary = df_default.groupby("leaf_depth").agg(["mean", "std"]).reset_index()
    df_summary.columns = ["leaf_depth"] + [f"{col}_{stat}_default" for col, stat in df_summary.columns[1:]]

    df_summary_no_remap = df_no_remap.groupby("leaf_depth").agg(["mean", "std"]).reset_index()
    df_summary_no_remap.columns = ["leaf_depth"] + [f"{col}_{stat}_no_remap" for col, stat in df_summary_no_remap.columns[1:]]

    df_summary_no_remap_no_error = df_no_remap_no_error.groupby("leaf_depth").agg(["mean", "std"]).reset_index()
    df_summary_no_remap_no_error.columns = ["leaf_depth"] + [f"{col}_{stat}_no_remap_no_error" for col, stat in df_summary_no_remap_no_error.columns[1:]]

    df_summary_new_line = df_new_line.groupby("leaf_depth").agg(["mean", "std"]).reset_index()
    df_summary_new_line.columns = ["leaf_depth"] + [f"{col}_{stat}_new_line" for col, stat in df_summary_new_line.columns[1:]]

    df_final = pd.merge(df_summary, df_summary_no_remap, on="leaf_depth", how="outer").sort_values("leaf_depth")
    df_final = pd.merge(df_final, df_summary_no_remap_no_error, on="leaf_depth", how="outer").sort_values("leaf_depth")
    df_final = pd.merge(df_final, df_summary_new_line, on="leaf_depth", how="outer").sort_values("leaf_depth")


    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    df_final.to_csv(output_csv_path, index=False)
    print(f"✅ Metrics saved to {output_csv_path}")
    return df_final



if __name__ == "__main__":
    LOOP_ITERATIONS = 10
    nb_qubits = [ 81,121]
    backends = ["ibm_brisbane_old"]
    for backend in backends:
        for nq in nb_qubits:
            compute_routing_metrics_ablation_to_csv(
                nb_qubits=nq,
                nb_loop_iterations=LOOP_ITERATIONS,
                topology_name=backend,
                output_csv_path=f"results-summary/ablation_study5/{backend}_{nq}qbt_{LOOP_ITERATIONS}iter_metrics_ablation_study.csv"
            )