import argparse
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
import os
import json

# Backends you want to process
BACKENDS = ["ibm_sherbrooke", "ankaa",  "imb_sherbrooke2X"]


def load_qops_from_json(json_path):
    """Read QOPS from a JSON file."""
    try:
        with open(json_path, "r") as f:
            data = json.load(f)
        return data.get("Stats", {}).get("Qops", None)
    except Exception as e:
        print(f"⚠️ Could not read {json_path}: {e}")
        return None


def load_backend_data(benchmark: str, backends=BACKENDS):
    """Load CSVs, enrich with QOPS from JSON files, and return as dict of DataFrames."""
    data = {}
    for backend in backends:
        csv_file = f"results/stats/{benchmark}_{backend}_trivial.csv"
        try:
            df = pd.read_csv(csv_file)

            # Add QOPS column by reading JSON files
            qops_values = []
            for fname in df["filename"]:
                # assumes JSON files are in ./benchmarks/
                json_path = os.path.join(
                    f"benchmarks/polyhedral/{benchmark}", fname)
                qops = load_qops_from_json(json_path)
                qops_values.append(qops)
            df["QOPS"] = qops_values

            data[backend] = df
            print(f"✅ Loaded {csv_file} with {len(df)} rows (QOPS added)")
        except FileNotFoundError:
            print(f"⚠️ File not found: {csv_file}")
        except Exception as e:
            print(f"❌ Error reading {csv_file}: {e}")
    return data


# --- plotting function (self-contained) ---
def plot_grouped_scatter_with_noise(data_grouped, xlabel, ylabel, title, fig_name):
    plt.figure(figsize=(10, 4))
    colors = plt.cm.tab10.colors
    plt.rcParams.update({'font.size': 21})
    markers = ['o', 's', '^', 'x', '+', '*', 'D', 'v']

    for i, (method, grouping) in enumerate(data_grouped.items()):
        color = colors[i % len(colors)]
        marker = markers[i % len(markers)]

        xs = sorted(grouping.keys())
        ys = [grouping[x] for x in xs]

        # scatter points
        x_pts, y_pts = [], []
        for xv, vals in zip(xs, ys):
            x_pts.extend([xv] * len(vals))
            y_pts.extend(vals)
        plt.scatter(x_pts, y_pts, alpha=0.5, color=color,
                    marker=marker, label=method)

        # mean ± std line
        means = [np.mean(vals) for vals in ys]
        stds = [np.std(vals) for vals in ys]
        plt.plot(xs, means, color=color, linewidth=2)
        plt.fill_between(xs,
                         np.array(means) - np.array(stds),
                         np.array(means) + np.array(stds),
                         color=color, alpha=0)

    # format thousands on X axis
    def thousands_formatter(x, pos):
        if x >= 1000:
            return f'{int(x/1000)}k'
        else:
            return f'{int(x)}'

    formatter = FuncFormatter(thousands_formatter)
    plt.gca().xaxis.set_major_formatter(formatter)
    plt.gca().tick_params(axis='x', labelsize=16)
    plt.gca().tick_params(axis='y', labelsize=16)
    plt.xlabel(xlabel, fontsize=18)
    plt.ylabel(ylabel, fontsize=18)
    plt.title(title, fontsize=0)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=17)
    plt.tight_layout()

    os.makedirs("plots", exist_ok=True)
    plt.savefig(f"plots/{fig_name}", dpi=300)
    print(f"Figure 5 saved as: plots/{title.replace(' ', '_').lower()}.png")


def main():
    parser = argparse.ArgumentParser(
        description="Plot runtime for given benchmark across backends")
    parser.add_argument("--benchmark", type=str, default="queko-bss-54qbt",
                        help="Benchmark name (e.g., queko-bss-54qbt)")
    args = parser.parse_args()

    # Load data from CSVs
    data_dict = load_backend_data(args.benchmark)

    # Prepare data for plotting
    data_grouped = {}
    for backend, df in data_dict.items():
        grouped = df.groupby("QOPS")["runtime"].apply(list).to_dict()
        data_grouped[backend] = grouped

    # Plot
    plot_grouped_scatter_with_noise(
        data_grouped,
        xlabel="QOPS",
        ylabel="Time (s)",
        title=f"{args.benchmark} - Runtime vs QOPS",
        fig_name=f"{args.benchmark}_runtime_vs_qops"
    )


if __name__ == "__main__":
    main()
