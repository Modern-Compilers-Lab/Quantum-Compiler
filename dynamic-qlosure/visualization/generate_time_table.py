from pathlib import Path
import statistics as stats
import numpy as np

# ---------- Load helpers ----------

def collect_times_by_depth(method_root: Path, qubits: int):
    """Return {leaf_depth(int): [times]} for a given qubit size."""
    times_by_depth = {}
    qstr = f"{qubits:03d}"
    for depth in range(10, 91, 10):
        method_dir = method_root / f"{qubits}qbt" / f"queko-{qstr}qbt_nest_00_nodes001_leaf-depth-{depth}"
        seed_times = []
        for seed in [3, 21, 42, 63, 84, 105, 126, 147, 168, 189]:
            time_file_path = method_dir / f"SEED_{seed}" / "time.txt"
            try:
                with open(time_file_path, "r") as f:
                    seed_times.append(float(f.read().strip()))
            except Exception:
                pass
        if seed_times:
            times_by_depth[depth] = seed_times
    return times_by_depth


# ---------- Aggregation helpers ----------

DEPTH_BINS = {
    "Small (10–30)": [10, 20, 30],
    "Medium (40–60)": [40, 50, 60],
    "Large (70–90)": [70, 80, 90],
}

def average_over_bins(times_by_depth: dict, bins=DEPTH_BINS):
    """Return {bin_label: average_of_means}."""
    out = {}
    for label, depths in bins.items():
        vals = []
        for d in depths:
            if d in times_by_depth and times_by_depth[d]:
                vals.append(stats.mean(times_by_depth[d]))
        out[label] = (sum(vals) / len(vals)) if vals else float("nan")
    return out


# ---------- Main comparison ----------

def print_qroqi_vs_sabre_summary(
    topology_name="ibm_kingston",
    sizes=(54, 81, 121),
    qroqi_root="results_time/qroqi/one_loop",
    sabre_root="results_time/sabre/one_loop",
):
    qroqi_base = Path(qroqi_root) / topology_name
    sabre_base = Path(sabre_root) / topology_name

    print(f"\n=== QROQI vs SABRE Summary for {topology_name} ===\n")
    for q in sizes:
        print(f"--- {q} qubits ---")
        qroqi_depths = collect_times_by_depth(qroqi_base, q)
        sabre_depths = collect_times_by_depth(sabre_base, q)

        qroqi_bins = average_over_bins(qroqi_depths)
        sabre_bins = average_over_bins(sabre_depths)

        for label in DEPTH_BINS.keys():
            q_avg = qroqi_bins.get(label, float("nan"))
            s_avg = sabre_bins.get(label, float("nan"))
            if np.isnan(q_avg) or np.isnan(s_avg):
                print(f"{label:<15}: data missing")
            else:
                imp = (1 - q_avg / s_avg) * 100
                print(f"{label:<15}: SABRE={s_avg:.3f}s  QROQI={q_avg:.3f}s  Δ={imp:.1f}% faster")
        print()


# ---------------- Example usage ----------------
if __name__ == "__main__":
    print_qroqi_vs_sabre_summary("ibm_kingston", sizes=(54, 81, 121))
    print_qroqi_vs_sabre_summary("ibm_brisbane_old", sizes=(54, 81, 121))
