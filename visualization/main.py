import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import matplotlib.pyplot as plt
import os
import csv
import glob

def plot_performance0(benchmark, results_swaps, results_depth, relative=False, baseline="Qlosure"):
    """
    Plot the performance of heuristic methods using Plotly.
    
    This function now offers an option to plot relative performance. When `relative=True`,
    the function computes for each method (including swap counts and circuit depths) the 
    relative difference with respect to the baseline method (default: "Qlosure"). 
    The formula used is: (method - baseline) / baseline.
    
    Parameters:
        benchmark (str): The label for the x-axis.
        results_swaps (dict): Dictionary where keys are method names and values are lists of swap counts.
        results_depth (dict): Dictionary where keys are method names and values are lists of circuit depths.
        relative (bool): If True, compute relative performance with respect to the baseline method.
        baseline (str): The method to use as baseline (default is "Qlosure").
            
    Returns:
        fig (plotly.graph_objects.Figure): Plotly figure with subplots and interactive buttons.
    """
    # Default colors for the methods
    colors = {
        "qmap": "#8c564b",     # blue
        "sabre": "#ff7f0e",    # orange
        "pytket": "#2ca02c",   # green 
        "pytket2": "#2ca02c",
        "cirq": "#9467bd",     # purple
        "Qlosure": "#d62728",  # red
        "no_read": "#1f77b4",  # brown

    }
    
    methods = list(results_swaps.keys())
    total_methods = len(methods)
    
    # If relative performance is requested, compute the relative differences
    if relative:
        relative_swaps = {}
        relative_depth = {}
        baseline_swaps = np.array(results_swaps[baseline])
        baseline_depth = np.array(results_depth[baseline])
        for method in methods:
            method_swaps = np.array(results_swaps[method])
            method_depth = np.array(results_depth[method])
            # For the baseline method, the relative difference is 0
            if method == baseline:
                relative_swaps[method] = [0] * len(method_swaps)
                relative_depth[method] = [0] * len(method_depth)
            else:
                relative_swaps[method] = ((method_swaps - baseline_swaps) / baseline_swaps).tolist()
                relative_depth[method] = ((method_depth - baseline_depth) / baseline_depth).tolist()
        plot_swaps = relative_swaps
        plot_depth = relative_depth
        yaxis_title_swaps = "Relative Swap Counts"
        yaxis_title_depth = "Relative Circuit Depth"
    else:
        plot_swaps = results_swaps
        plot_depth = results_depth
        yaxis_title_swaps = "Swap Counts"
        yaxis_title_depth = "Circuit Depth"
    
    # Create a subplot figure with 2 rows: swaps (row 1) and depths (row 2)
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        subplot_titles=("Swap Counts", "Circuit Depth")
    )
    
    # Add traces for swap counts in the first row
    for method in methods:
        x_vals = list(range(len(plot_swaps[method])))
        fig.add_trace(
            go.Scatter(
                x=x_vals, 
                y=plot_swaps[method],
                mode='lines+markers',
                name=method,
                legendgroup=method,
                marker=dict(color=colors.get(method, "#000000")),
                line=dict(color=colors.get(method, "#000000"))
            ),
            row=1, col=1
        )

    # Add traces for circuit depths in the second row
    for method in methods:
        x_vals = list(range(len(plot_depth[method])))
        fig.add_trace(
            go.Scatter(
                x=x_vals, 
                y=plot_depth[method],
                mode='lines+markers',
                name=method,
                legendgroup=method,
                showlegend=False,  # Avoid duplicate legend entries
                marker=dict(color=colors.get(method, "#000000")),
                line=dict(color=colors.get(method, "#000000"))
            ),
            row=2, col=1
        )

    # Total number of traces is twice the number of methods
    total_traces = total_methods * 2

    # Build interactive buttons
    buttons = [
        {
            "label": "Show All",
            "method": "update",
            "args": [{"visible": [True] * total_traces}]
        },
        {
            "label": "Hide All",
            "method": "update",
            "args": [{"visible": [False] * total_traces}]
        }
    ]

    # For each method, create a button to show its corresponding swap and depth traces only
    for i, method in enumerate(methods):
        visibility = [False] * total_traces
        visibility[i] = True                   # Swap trace in row 1
        visibility[i + total_methods] = True   # Depth trace in row 2
        buttons.append({
            "label": method,
            "method": "update",
            "args": [{"visible": visibility}]
        })

    # Update layout with interactive buttons and axis titles
    fig.update_layout(
        updatemenus=[{
            "buttons": buttons,
            "direction": "down",
            "showactive": True,
            "x": 1.15,
            "y": 0.5
        }],
        title="Performance of Heuristic Methods",
        xaxis_title="benchmark",
        yaxis_title=yaxis_title_swaps,
        yaxis2_title=yaxis_title_depth
    )
    
    return fig

def plot_performance(benchmark, results_swaps, results_depth, qops_values, relative=False, baseline="Qlosure"):
    """
    Plot the performance of heuristic methods using Plotly.
    
    This function now offers an option to plot relative performance. When `relative=True`,
    the function computes for each method (including swap counts and circuit depths) the 
    relative difference with respect to the baseline method (default: "Qlosure"). 
    The formula used is: (method - baseline) / baseline.
    
    Parameters:
        benchmark (str): The label for the x-axis.
        results_swaps (dict): Dictionary where keys are method names and values are lists of swap counts.
        results_depth (dict): Dictionary where keys are method names and values are lists of circuit depths.
        qops_values (list): List of qops values to use for the x-axis.
        relative (bool): If True, compute relative performance with respect to the baseline method.
        baseline (str): The method to use as baseline (default is "Qlosure").
            
    Returns:
        fig (plotly.graph_objects.Figure): Plotly figure with subplots and interactive buttons.
    """
    # Default colors for the methods
    colors = {
        "qmap": "#8c564b",     # blue
        "sabre": "#ff7f0e",    # orange
        "pytket": "#2ca02c",   # green 
        "pytket2": "#2ca02c",
        "cirq": "#9467bd",     # purple
        "Qlosure": "#d62728",  # red
        "no_read": "#1f77b4",  # brown
    }
    
    methods = list(results_swaps.keys())
    total_methods = len(methods)
    
    # If relative performance is requested, compute the relative differences
    if relative:
        relative_swaps = {}
        relative_depth = {}
        baseline_swaps = np.array(results_swaps[baseline])
        baseline_depth = np.array(results_depth[baseline])
        for method in methods:
            method_swaps = np.array(results_swaps[method])
            method_depth = np.array(results_depth[method])
            # For the baseline method, the relative difference is 0
            if method == baseline:
                relative_swaps[method] = [0] * len(method_swaps)
                relative_depth[method] = [0] * len(method_depth)
            else:
                relative_swaps[method] = ((method_swaps - baseline_swaps) / baseline_swaps).tolist()
                relative_depth[method] = ((method_depth - baseline_depth) / baseline_depth).tolist()
        plot_swaps = relative_swaps
        plot_depth = relative_depth
        yaxis_title_swaps = "Relative Swap Counts"
        yaxis_title_depth = "Relative Circuit Depth"
    else:
        plot_swaps = results_swaps
        plot_depth = results_depth
        yaxis_title_swaps = "Swap Counts"
        yaxis_title_depth = "Circuit Depth"
    
    # Create a subplot figure with 2 rows: swaps (row 1) and depths (row 2)
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        subplot_titles=("Swap Counts", "Circuit Depth")
    )
    print(qops_values)

    # Add traces for swap counts in the first row
    for method in methods:
        fig.add_trace(
            go.Scatter(
                x=qops_values,  # Use qops values for x-axis
                y=plot_swaps[method],
                mode='lines+markers',
                name=method,
                legendgroup=method,
                marker=dict(color=colors.get(method, "#000000")),
                line=dict(color=colors.get(method, "#000000"))
            ),
            row=1, col=1
        )

    # Add traces for circuit depths in the second row
    for method in methods:
        fig.add_trace(
            go.Scatter(
                x=qops_values,  # Use qops values for x-axis
                y=plot_depth[method],
                mode='lines+markers',
                name=method,
                legendgroup=method,
                showlegend=False,  # Avoid duplicate legend entries
                marker=dict(color=colors.get(method, "#000000")),
                line=dict(color=colors.get(method, "#000000"))
            ),
            row=2, col=1
        )

    # Total number of traces is twice the number of methods
    total_traces = total_methods * 2

    # Build interactive buttons
    buttons = [
        {
            "label": "Show All",
            "method": "update",
            "args": [{"visible": [True] * total_traces}]
        },
        {
            "label": "Hide All",
            "method": "update",
            "args": [{"visible": [False] * total_traces}]
        }
    ]

    # For each method, create a button to show its corresponding swap and depth traces only
    for i, method in enumerate(methods):
        visibility = [False] * total_traces
        visibility[i] = True                   # Swap trace in row 1
        visibility[i + total_methods] = True   # Depth trace in row 2
        buttons.append({
            "label": method,
            "method": "update",
            "args": [{"visible": visibility}]
        })

    # Update layout with interactive buttons and axis titles
    fig.update_layout(
        
        title={
            "text": "Sherbrooke Benchmark on ankaa 3 Backend",
            "x": 0.5,  # Center the title
            "xanchor": "center",
            "y": 0.97,  # Position at the top
            "yanchor": "top"
        },
        yaxis_title=yaxis_title_swaps,
        yaxis2_title=yaxis_title_depth
    )
    
    # Update the x-axis of the bottom subplot (row 2) to include the title at the bottom
    fig.update_xaxes(
        title_text="qops",
        title_standoff=20,  # Add some space between axis and title
        row=2, col=1  # Apply to the bottom subplot
    )
    
    return fig


def compute_confusion_matrix(results):
    """
    Compute pairwise comparisons for a given metric (swap count or depth).
    
    Parameters:
        results (dict): Dictionary where keys are method names and values are lists of values.
        
    Returns:
        matrix (np.ndarray): A 2D array (percentage) where matrix[i,j] shows the percentage 
                             of times method i is better than method j.
        methods (list): List of method names (order corresponding to the matrix indices).
    """
    methods = list(results.keys())
    num_methods = len(methods)
    matrix = np.zeros((num_methods, num_methods))
    
    for i, method_i in enumerate(methods):
        for j, method_j in enumerate(methods):
            if i != j:
                # Compare each trial value: count how often method_i is better than method_j.
                count_better = sum(np.array(results[method_i]) <= np.array(results[method_j]))
                matrix[i, j] = (count_better / len(results[method_i])) * 100  # Percentage
    return matrix, methods

def plot_confusion_matrices(benchmark,results_swaps, results_depth):
    """
    Plot confusion matrices for swap counts and circuit depth using Matplotlib.
    
    Parameters:
        results_swaps (dict): Dictionary with swap count results for each method.
        results_depth (dict): Dictionary with circuit depth results for each method.
        
    Returns:
        fig (matplotlib.figure.Figure): Matplotlib figure containing the two confusion matrices.
    """
    # Compute confusion matrices and get the method order.
    conf_matrix_swaps, methods = compute_confusion_matrix(results_swaps)
    conf_matrix_depth, _ = compute_confusion_matrix(results_depth)
    num_methods = len(methods)
    
    # Create a figure with two subplots side by side.
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # Plot the confusion matrix for swap counts.
    im1 = axes[0].imshow(conf_matrix_swaps, cmap="Blues", vmin=0, vmax=100)
    axes[0].set_title("Confusion Matrix - Swap Counts")
    axes[0].set_xticks(np.arange(num_methods))
    axes[0].set_xticklabels(methods)
    axes[0].set_yticks(np.arange(num_methods))
    axes[0].set_yticklabels(methods)
    axes[0].set_xlabel("Compared Against")
    axes[0].set_ylabel("Method")
    
    # Add text annotations to the swap confusion matrix.
    for i in range(num_methods):
        for j in range(num_methods):
            if i != j:
                text_color = "black" if conf_matrix_swaps[i, j] < 50 else "white"
                axes[0].text(j, i, f"{conf_matrix_swaps[i, j]:.1f}%", ha="center", va="center", color=text_color)
    
    fig.colorbar(im1, ax=axes[0])
    
    # Plot the confusion matrix for circuit depths.
    im2 = axes[1].imshow(conf_matrix_depth, cmap="Reds", vmin=0, vmax=100)
    axes[1].set_title("Confusion Matrix - Circuit Depth")
    axes[1].set_xticks(np.arange(num_methods))
    axes[1].set_xticklabels(methods)
    axes[1].set_yticks(np.arange(num_methods))
    axes[1].set_yticklabels(methods)
    axes[1].set_xlabel("Compared Against")
    # The y-axis label is already on the left subplot.
    
    # Add text annotations to the depth confusion matrix.
    for i in range(num_methods):
        for j in range(num_methods):
            if i != j:
                text_color = "black" if conf_matrix_depth[i, j] < 50 else "white"
                axes[1].text(j, i, f"{conf_matrix_depth[i, j]:.1f}%", ha="center", va="center", color=text_color)
    
    fig.colorbar(im2, ax=axes[1])
    
    fig.suptitle("benchmark", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    return fig

def plot_scatter_with_mean_band(x_data, y_data_dict, xlabel, ylabel, title):
    """
    Scatter plot with a mean line and standard deviation band for each method.
    """
    plt.figure(figsize=(10, 6))
    markers = ['o', 's', '^', 'x', '+', '*', 'D', 'v']
    colors = plt.cm.tab10.colors

    for i, (method, y_vals) in enumerate(y_data_dict.items()):
        color = colors[i % len(colors)]
        marker = markers[i % len(markers)]
        x_vals = x_data[method]

        # Scatter points
        plt.scatter(x_vals, y_vals, alpha=0.5, label=method, color=color, marker=marker)

        # Mean line across unique depths
        unique_depths = sorted(set(x_vals))
        means = []
        stds = []

        for d in unique_depths:
            group_vals = [y for x, y in zip(x_vals, y_vals) if x == d]
            means.append(np.mean(group_vals))
            stds.append(np.std(group_vals))

        plt.plot(unique_depths, means, color=color, linewidth=2)
        plt.fill_between(unique_depths,
                         np.array(means) - np.array(stds),
                         np.array(means) + np.array(stds),
                         color=color, alpha=0.2)

    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.title(title, fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.show()
    
def print_depth_improvemnt(depth_grouped):
    # Plot swaps vs original depth
    categories = {
        'medium (≤500 gates)': [100, 200, 300,400,500],
        'large (600–900 gates)': [600,700,800,900],
    }

    # Define optimal depths equal to size
    optimal = {size: size for size in [100, 200, 300, 400, 500, 600, 700, 800, 900]}

    # Precompute optimal means per category
    optimal_cat_means = {
        cat: sum(optimal[s] for s in sizes) / len(sizes)
        for cat, sizes in categories.items()
    }

    # Compute depth-factors
    rows = []
    for mapper, depths in depth_grouped.items():
        row = {'Mapper': mapper}
        for category, sizes in categories.items():
            all_vals = [val for size in sizes for val in depths[size]]
            avg_mapped = sum(all_vals) / len(all_vals)
            row[category] = avg_mapped / optimal_cat_means[category]
        rows.append(row)
    for row in rows :
        print(row)
  
  
def print_top_depth_improvements(depth_grouped, top_n=10):
    """
    Compute and display the top mappers by depth improvement (mapped/original depth ratio).

    Args:
        depth_grouped (dict): Mapping from mapper names to dicts of sizes and their mapped depths.
                              Expected format: {mapper: {size: [depths...]}}.
        top_n (int): Number of top mappers to display.
    """
    # Define categories and associated sizes
    categories = {
        'medium (≤500 gates)': [100, 200, 300, 400, 500],
        'large (600–900 gates)': [600, 700, 800, 900],
    }

    # Optimal depths equal to size
    optimal = {size: size for size in range(100, 1000, 100)}

    # Compute mean optimal depth per category
    optimal_cat_means = {
        cat: sum(optimal[size] for size in sizes) / len(sizes)
        for cat, sizes in categories.items()
    }

    # Aggregate improvement ratios per mapper
    metrics = []
    for mapper, depth_dict in depth_grouped.items():
        # Compute ratio per category
        ratios = {}
        for cat, sizes in categories.items():
            # Collect all mapped depths for this mapper and category sizes
            values = []
            for size in sizes:
                values.extend(depth_dict.get(size, []))
            if not values:
                ratios[cat] = float('nan')
            else:
                avg_mapped = sum(values) / len(values)
                # Ratio: mapped depth / optimal category mean
                ratios[cat] = avg_mapped / optimal_cat_means[cat]
        # Overall ratio: average across categories
        valid_ratios = [rat for rat in ratios.values() if not isnan(rat)]
        overall = sum(valid_ratios) / len(valid_ratios) if valid_ratios else float('nan')
        metrics.append({
            'Mapper': mapper,
            **ratios,
            'Overall Ratio': overall
        })

    # Sort by Overall Ratio ascending (best improvement first)
    sorted_metrics = sorted(metrics, key=lambda x: x['Overall Ratio'])

    # Display top N
    for entry in sorted_metrics[:top_n]:
        print(f"{entry['Mapper']}: Overall Ratio = {entry['Overall Ratio']:.3f}")
        for cat in categories:
            print(f"  {cat}: {entry[cat]:.3f}")
        print() 
def print_swaps_improvemnt(swaps_grouped):
    categories = {
        'medium (≤500 gates)': [100, 200, 300,400,500],
        'large (600–900 gates)': [600,700,800,900],
    }
    qclosure_data = swaps_grouped['Qlosure']

    # Compute ratios per mapper and category
    rows = []
    for mapper, data in swaps_grouped.items():
        row = {'Mapper': mapper}
        for category, sizes in categories.items():
            swaps = [v for size in sizes for v in data[size]]
            qclosure_swaps = [v for size in sizes for v in qclosure_data[size]]
            avg_swap = sum(swaps) / len(swaps)
            avg_qclosure = sum(qclosure_swaps) / len(qclosure_swaps)
            row[category] = avg_swap / avg_qclosure
        rows.append(row)
    for row in rows :
        print(row)
 
def compute_improvement(results, our_method, methods):
    """
    Compute percentage improvement for each method relative to our_method.
    returns dict: {method: [swap_improvements], ...} in percent.
    """
    improvements = {'swaps': {}, 'depth': {}}

    n = len(results['swaps'][our_method])
    for method in methods:
        if method == our_method:
            continue
        imp_sw = []
        imp_dp = []
        for i in range(n):
            base_sw = results['swaps'][method][i]
            our_sw = results['swaps'][our_method][i]
            # reduction: (base - ours)/base * 100
            sw_imp = (base_sw - our_sw) / base_sw * 100 if base_sw != 0 else 0
            imp_sw.append(sw_imp)

            base_dp = results['depth'][method][i]
            our_dp = results['depth'][our_method][i]
            dp_imp = (base_dp - our_dp) / base_dp * 100 if base_dp != 0 else 0
            imp_dp.append(dp_imp)

        improvements['swaps'][method] = imp_sw
        improvements['depth'][method] = imp_dp

    return

import seaborn as sns


def plots(benchmark, methods,layout,backends,our_method= "Qlosure"):

    
    results_swaps, results_depth,qops_values,depth_ratio = get_data(benchmark,methods,layout,backends)
    swaps_grouped, depth_grouped = get_data_grouped_by_circuit_depth(benchmark, methods, layout, backends)


    
    
    # Plot compiled depth vs original depth
    plot_grouped_scatter_with_noise(swaps_grouped, ylabel="Swaps", title=f"")

    plot_grouped_scatter_with_noise(depth_grouped, ylabel="Compiled Depth", title="")
    return
    performance_fig = plot_performance(benchmark,results_swaps, results_depth,qops_values)
    performance_fig.show()
    
    performance_fig = plot_performance(benchmark,results_swaps, results_depth,qops_values,relative=True)
    #performance_fig.show()
    
    # Generate and display the Matplotlib confusion matrices
    confusion_fig = plot_confusion_matrices(benchmark,results_swaps, results_depth)
     

    calculate_improvements(benchmark,results_swaps,"Swaps",our_method=our_method)
    calculate_improvements(benchmark,results_depth,"Depth",our_method=our_method)
    
    #bars = plot_depth_ratios(benchmark, methods, layout,backend)
    plt.show()


    

def compute_average_depth_ratio(benchmark, methods, layout,backend):
    folder_path = "../experiment_results/" + benchmark 
    avg_ratios = {}
    
    for method in methods:
        file_path = os.path.join(folder_path, f"{method}.csv")
        ratios = []
        rows = []
        try:
            with open(file_path, newline='') as csvfile:
                reader = csv.DictReader(csvfile)
                
                # Check if the required "depth" column is present.
                if "depth" not in reader.fieldnames:
                    return None
                
                for row in reader:
                    # Skip any row that is None or empty.
                    if not row:
                        continue
                    rows.append(row)
            
            if not rows:
                print(f"No valid data found in {file_path}.")
                avg_ratios[method] = None
                continue
            
            # First pass: determine the maximum value in depth_{layout} (ignoring "timeout" and invalid entries)
            numeric_values = []
            for row in rows:
                val_str = row.get(f"{backend}_depth_{layout}", "")
                if val_str != "timeout" and val_str != "error" and val_str != "":
                    try:
                        numeric_values.append(float(val_str))
                    except Exception as e:
                        print(f"Error converting value in {file_path} to float: {e}")
            
            max_val = max(numeric_values) if numeric_values else 0
            
            # Second pass: compute ratios, replacing "timeout" with max_val
            for row in rows:
                try:
                    baseline_str = row.get("depth", "")
                    if baseline_str in ["timeout", ""]:
                        continue  # skip if baseline is not valid
                    baseline = float(baseline_str)
                    if baseline == 0:
                        continue  # Avoid division by zero
                        
                    depth_str = row.get(f"depth_{layout}", "")
                    if depth_str == "timeout" or depth_str == "":
                        depth_val = max_val
                    else:
                        depth_val = float(depth_str)
                    
                    ratios.append(depth_val / baseline)
                except Exception as row_err:
                    print(f"Error processing row in {file_path}: {row_err}")
            
            avg_ratios[method] = sum(ratios) / len(ratios) if ratios else None
        except FileNotFoundError:
            print(f"File {file_path} not found.")
            avg_ratios[method] = None
        except Exception as e:
            print(f"An error occurred processing {file_path}: {e}")
            avg_ratios[method] = None
            
    return avg_ratios

def plot_depth_ratios(benchmark, methods, layout,backend):
    colors = {
        "qmap": "#8c564b",     # blue
        "sabre": "#ff7f0e",    # orange
        "pytket": "#2ca02c",   # green 
        "pytket2": "#2ca02c",
        "cirq": "#9467bd",     # purple
        "Qlosure": "#d62728",  # red
        "no_read": "#1f77b4",  # brown
    }
    
    
    avg_ratios = compute_average_depth_ratio(benchmark, methods, layout,backend)
    if not avg_ratios:
        return 
    methods = list(avg_ratios.keys())
    values = list(avg_ratios.values())
    bar_colors = [colors.get(method, "#000000") for method in methods]
    
    # Create a bar chart
    plt.figure(figsize=(10, 6))
    bars = plt.bar(methods, values, color=bar_colors)

    # Set labels and title
    plt.xlabel('Method')
    plt.ylabel('Average Depth Ratio')
    plt.title(f'Average Depth Ratio  {benchmark}')
    plt.ylim(0, max(values) + 1)

    # Annotate each bar with its value
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, height + 0.05, f'{height:.2f}', 
                ha='center', va='bottom')
        
    return bars

def plot_grouped_scatter_with_noise(data_grouped, ylabel, title,
                                   font_size=21,
                                   marker_size=100):
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.ticker import FuncFormatter
    
    plt.figure(figsize=(10, 6))
    # bump up the global font size
    plt.rcParams.update({'font.size': font_size})
    colors = plt.cm.tab10.colors
    markers = ['o', 's', '^', '*', 'D']
    
    for i, (method, depth_dict) in enumerate(data_grouped.items()):
        color = colors[i % len(colors)]
        marker = markers[i % len(markers)]
        all_depths = sorted(depth_dict.keys())
        all_vals = [depth_dict[d] for d in all_depths]
        
        # Flatten all scatter points
        x = []
        y = []
        for d, values in zip(all_depths, all_vals):
            x.extend([d] * len(values))
            y.extend(values)
        
        # scatter with larger markers
        plt.scatter(x, y,
                   s=marker_size,
                   alpha=0.5,
                   color=color,
                   marker=marker,
                   label=method)
        
        # Plot mean line with std band
        means = [np.mean(v) for v in all_vals]
        stds = [np.std(v) for v in all_vals]
        plt.plot(all_depths, means,
                color=color,
                linewidth=2)
        plt.fill_between(all_depths,
                        np.array(means) - np.array(stds),
                        np.array(means) + np.array(stds),
                        color=color,
                        alpha=0)
    
    # Custom formatter for y-axis to show values as "5k", "10k", etc.
    def thousands_formatter(x, pos):
        if x >= 1000:
            return f'{int(x/1000)}k'
        else:
            return f'{int(x)}'
    
    formatter = FuncFormatter(thousands_formatter)
    plt.gca().yaxis.set_major_formatter(formatter)
    
    # axis labels and title at larger font
    plt.xlabel("Initial depth", fontsize=25)
    plt.ylabel(ylabel, fontsize=25)
    plt.title(title, fontsize=font_size + 2)
    
    # grid and legend
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=font_size)
    
    # bump up tick label size
    plt.xticks(fontsize=font_size)
    plt.yticks(fontsize=font_size)
    plt.tight_layout()
    plt.show() 

def plot_grouped_scatter_with_noise(data_grouped, ylabel, title,
                                   font_size=21,
                                   marker_size=100):
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.ticker import FuncFormatter
    
    plt.figure(figsize=(10, 6))
    plt.rcParams.update({'font.size': font_size})
    
    # Fixed color and marker per method
    method_colors = {
        "sabre": "#1f77b4",    # blue
        "cirq": "#ff7f0e",     # orange
        "qmap": "#2ca02c",     # green
        "pytket": "#d62728",   # red
        "Qlosure": "#9467bd",  # purple
    }

    method_markers = {
        "sabre": "o",
        "cirq": "s",
        "qmap": "^",
        "pytket": "*",
        "Qlosure": "D",
    }

    for i, (method, depth_dict) in enumerate(data_grouped.items()):
        color = method_colors.get(method, "#333333")  # fallback to gray
        marker = method_markers.get(method, "o")      # fallback to circle
        
        all_depths = sorted(depth_dict.keys())
        all_vals = [depth_dict[d] for d in all_depths]
        
        x = []
        y = []
        for d, values in zip(all_depths, all_vals):
            x.extend([d] * len(values))
            y.extend(values)
        
        plt.scatter(x, y,
                    s=marker_size,
                    alpha=0.5,
                    color=color,
                    marker=marker,
                    label=method)
        
        # Mean and std band
        means = [np.mean(v) for v in all_vals]
        stds = [np.std(v) for v in all_vals]
        plt.plot(all_depths, means,
                 color=color,
                 linewidth=2)
        plt.fill_between(all_depths,
                         np.array(means) - np.array(stds),
                         np.array(means) + np.array(stds),
                         color=color,
                         alpha=0)

    # Formatter for y-axis (e.g., 1000 -> 1k)
    def thousands_formatter(x, pos):
        return f'{int(x/1000)}k' if x >= 1000 else f'{int(x)}'
    
    formatter = FuncFormatter(thousands_formatter)
    plt.gca().yaxis.set_major_formatter(formatter)

    # Axis labels and title
    plt.xlabel("Initial depth", fontsize=25)
    plt.ylabel(ylabel, fontsize=25)
    plt.title(title, fontsize=font_size + 2)

    # Styling
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=font_size)
    plt.xticks(fontsize=font_size)
    plt.yticks(fontsize=font_size)
    plt.tight_layout()
    plt.show()

def get_data_grouped_by_circuit_depth(benchmarks, methods, layout, backends):
    # Will return: {method: {original_depth: [values...]}}
    swaps_grouped = {method: defaultdict(list) for method in methods}
    depth_grouped = {method: defaultdict(list) for method in methods}
    
    for bench in benchmarks:
        folder = os.path.join("../experiment_results", bench)
        for backend in backends:
            for method in methods:
                path = os.path.join(folder, f"{method}.csv")
                if not os.path.isfile(path):
                    print(f"Warning: {path} not found.")
                    continue

                df = pd.read_csv(path)
                if 'depth' not in df.columns:
                    print(f"Missing 'depth' column in {path}")
                    continue

                swap_col = f"{backend}_swaps_{layout}"
                depth_col = f"{backend}_depth_{layout}"
                if swap_col not in df.columns or depth_col not in df.columns:
                    print(f"Missing swap/depth cols in {path}")
                    continue

                for _, row in df.iterrows():
                    try:
                        orig_depth = int(row['depth'])
                        swap_val = int(row[swap_col]) 
                        depth_val = int(row[depth_col]) 

                        if swap_val is not None:
                            swaps_grouped[method][orig_depth].append(swap_val)
                        if depth_val is not None:
                            depth_grouped[method][orig_depth].append(depth_val)
                    except:
                        continue

    return swaps_grouped, depth_grouped

def get_data_900(benchmarks, methods, layout, backend, target_depth=900):
    # Initialize dictionaries to store aggregated results
    raw_swaps = {method: [] for method in methods}
    raw_depth = {method: [] for method in methods}

    # Iterate over each benchmark
    for benchmark in benchmarks:
        folder_path = os.path.join("../experiment_results", benchmark)

        for method in methods:
            file_path = os.path.join(folder_path, f"{method}.csv")
            try:
                with open(file_path, newline='') as csvfile:
                    reader = csv.DictReader(csvfile)
                    swap_key  = f"{backend}_swaps_{layout}"
                    depth_key = f"{backend}_depth_{layout}"

                    for row in reader:
                        depth = row.get("depth", "")
                        # only keep rows where depth is a digit and equals target_depth
                        if depth.isdigit() and int(depth) == target_depth:
                            raw_swaps[method].append(row.get(swap_key, "error"))
                            raw_depth[method].append(row.get(depth_key, "error"))
            except FileNotFoundError:
                print(f"File {file_path} not found.")
            except Exception as e:
                print(f"An error occurred processing {file_path}: {e}")

    # Now post-process: replace non-digits with the max seen
    results_swaps = {}
    results_depth = {}
    for method in methods:
        # swaps
        numeric_swaps = [int(v) for v in raw_swaps[method] if v.isdigit()]
        max_swaps = max(numeric_swaps, default=0)
        results_swaps[method] = [
            int(v) if v.isdigit() else max_swaps
            for v in raw_swaps[method]
        ]

        # depths (though they should all be == target_depth)
        numeric_depth = [int(v) for v in raw_depth[method] if v.isdigit()]
        max_depth = max(numeric_depth, default=target_depth)
        results_depth[method] = [
            int(v) if v.isdigit() else max_depth
            for v in raw_depth[method]
        ]

    return results_swaps, results_depth,None


def plot_swaps(results_swaps, methods, title="Swaps Comparison"):
    """
    Plot swaps results using scatter + line plot style.
    
    Parameters:
    - results_swaps: dict of lists (method -> swap counts) from get_data function
    - methods: list of method names
    - title: plot title
    """
    
    # Convert results_swaps dict to DataFrame for easier plotting
    df_swaps = pd.DataFrame(results_swaps)
    
    plt.figure(figsize=(10, 6))
    colors = plt.cm.tab10.colors
    markers = ['o', 's', '^', 'x', '+', '*', 'D', 'v']
    
    for i, m in enumerate(methods):
        x = df_swaps.index  # default 0,1,2,...
        y = df_swaps[m]     # swaps for method m
        color = colors[i % len(colors)]
        marker = markers[i % len(markers)]
        
        plt.scatter(x, y, alpha=0.6, color=color, marker=marker, label=m)
        plt.plot(x, y, linestyle='-', linewidth=1.5, color=color)
    
    plt.xlabel("")
    plt.ylabel("swaps")
    plt.title(title)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.xticks([])
    plt.tight_layout()
    plt.show()
    

def get_data(benchmarks, methods, layout, backends):
    """
    Read CSV results for each benchmark and method, compute per-instance swaps, depth, qops,
    display a styled table with methods as columns and instances (rows) showing swaps, depth, qops,
    minimum swaps and depth per row highlighted, and compute depth ratios per row relative to the minimum depth.
    
    CSV files are sorted by depth column before processing.
    
    Returns:
    - results_swaps: dict of lists (method -> swap counts)
    - results_depth: dict of lists (method -> depths)
    - results_qops: dict of lists (method -> qops)
    - results_depth_ratio: dict of lists (method -> depth ratio = depth / min_depth)
    """
    # Initialize raw data containers
    raw_swaps = {method: [] for method in methods}
    raw_depth = {method: [] for method in methods}
    raw_qops = {method: [] for method in methods}
    
    # Iterate over each benchmark and method
    for benchmark in benchmarks:
        folder = os.path.join("../experiment_results", benchmark)
        for method in methods:
            for backend in backends:
                path = os.path.join(folder, f"{method}.csv")
                try:
                    with open(path, newline='') as csvfile:
                        reader = csv.DictReader(csvfile)
                        rows = list(reader)
                        
                        # Sort rows by depth column (convert to int for proper sorting)
                        rows.sort(key=lambda row: int(row.get('depth', '0')) if row.get('depth', '0').isdigit() else 0)
                        
                        # Process sorted rows
                        for row in rows:
                            raw_swaps[method].append(row.get(f"{backend}_swaps_{layout}", "0"))
                            raw_depth[method].append(row.get(f"{backend}_depth_{layout}", "0"))
                            raw_qops[method].append(row.get("qops", "0"))
                            
                except FileNotFoundError:
                    print(f"File not found: {path}")
                except Exception as e:
                    print(f"Error reading {path}: {e}")
        
    # Process raw data into numeric results
    results_swaps = {}
    results_depth = {}
    results_qops = {}
    
    for method in methods:
        # Swaps
        nums = [int(v) for v in raw_swaps[method] if v.isdigit()]
        max_sw = max(nums, default=0)
        results_swaps[method] = [int(v) if v.isdigit() else max_sw for v in raw_swaps[method]]
        
        # Depth
        nums = [int(v) for v in raw_depth[method] if v.isdigit()]
        max_dp = max(nums, default=0)
        results_depth[method] = [int(v) if v.isdigit() else max_dp for v in raw_depth[method]]
        
        # QOPs
        results_qops[method] = [int(v) if v.isdigit() else 0 for v in raw_qops[method]]
    
    # Compute depth ratios relative to minimum depth per instance
    n_instances = max(len(results_depth[m]) for m in methods)
    results_depth_ratio = {method: [] for method in methods}
    
    for idx in range(n_instances):
        # gather depths for this instance across methods
        depths = [results_depth[m][idx] for m in methods if idx < len(results_depth[m])]
        min_depth = min(depths) if depths else 1
        
        for m in methods:
            if idx < len(results_depth[m]):
                ratio = results_depth[m][idx] / min_depth if min_depth > 0 else float('inf')
                results_depth_ratio[m].append(ratio)
            else:
                results_depth_ratio[m].append(float('inf'))
    
    # Build DataFrames
    df_swaps = pd.DataFrame(results_swaps)
    df_depth = pd.DataFrame(results_depth)
    df_qops = pd.DataFrame(results_qops)
    df_ratio = pd.DataFrame(results_depth_ratio)
    
    # Combine into multi-index DataFrame
    df_all = pd.concat({
        'swaps': df_swaps,
        'depth': df_depth,
        'qops': df_qops,
        'depth_ratio': df_ratio
    }, axis=1)
    
    df_all = df_all.swaplevel(0,1, axis=1)
    df_all = df_all.reindex(columns=pd.MultiIndex.from_product(
        [methods, ['swaps','depth','qops','depth_ratio']]
    ))
    
    # Styling: highlight minima for swaps and depth
    swap_cols = [(m,'swaps') for m in methods]
    depth_cols = [(m,'depth') for m in methods]
    
    return results_swaps, results_depth, results_qops, results_depth_ratio

def calculate_improvements(benchmark,results_swaps,metric, our_method="Qlosure"):
    # Calculate our average value
    our_values = results_swaps[our_method]
    our_avg = sum(our_values) / len(our_values)
    
    improvements = {}
    for method, values in results_swaps.items():
        if not values:
            continue  
        method_avg = sum(values) / len(values)
        if method == our_method:
            continue  
        improvement = ((method_avg - our_avg) / method_avg) * 100
        improvements[method] = improvement

    # Prepare data for the bar chart
    methods = list(improvements.keys())
    improvement_values = list(improvements.values())
    colors = {
            "qmap": "#8c564b",     # blue
            "sabre": "#8c564b",    # orange
            "pytket": "#2ca02c",   # green 
            "pytket2": "#2ca02c",   # green
            "cirq": "#ff7f0e",     # purple
            "Qlosure": "#d62728",  # red
            "no_read": "#1f77b4",  # brown

        }
    # Create the bar chart
    plt.figure(figsize=(8, 6))
    bar_colors = [colors.get(method, "#000000") for method in methods]
    bars = plt.bar(methods, improvement_values, color=bar_colors)
    plt.xlabel("Method")
    plt.ylabel("Improvement (%)")
    plt.title(f"{metric} improvement ")
    plt.axhline(0, color='gray', linewidth=0.8)  # horizontal line at zero
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Annotate each bar with its improvement value
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval,
                 f"{yval:.2f}", ha='center', va='bottom')
    
    plt.show()



import pandas as pd
from qiskit import QuantumCircuit
import json
from collections import defaultdict
from qiskit.providers.fake_provider import Fake127QPulseV1
import matplotlib.pyplot as plt
import math

edges = Fake127QPulseV1().configuration().coupling_map
def compute_connectivity_metrics(original_circuit, edges,all_metrics = False):
    width = set()
    for instr in original_circuit.data:
        for q in instr.qubits:
            width.add(q._index)
    if all_metrics:
        logical_connections = defaultdict(lambda: defaultdict(int))
        for instr in original_circuit.data:
            if instr[0].name == "cx":
                q1 = instr[1][0]._index  
                q2 = instr[1][1]._index  
                logical_connections[q1][q2] += 1
                logical_connections[q2][q1] += 1
                
        required_connectivity = {q: set(neighbors.keys()) 
                                for q, neighbors in logical_connections.items()}
        
        physical_conn = defaultdict(set)
        for a, b in edges:
            physical_conn[a].add(b)
        
        physical_conn = {k: set(v) for k, v in physical_conn.items()}
        
        weight = {}
        for logical, required in required_connectivity.items():
            physical_available = physical_conn.get(logical, set())
            missing_connections = required - physical_available
            weight[logical] = max(0, len(missing_connections))
        
        total_weight = sum(weight.values())
        max_weight = max(weight.values()) if weight else 0
        average_weight = total_weight / len(weight) if weight else 0

        return total_weight, max_weight, average_weight,original_circuit.depth(),original_circuit.num_qubits,len(width)
    return 0, 0, 0,original_circuit.depth(),original_circuit.num_qubits,len(width)


def read_and_sort(csv_file):
    df = pd.read_csv(csv_file)
    return df.sort_values(by='qops').reset_index(drop=True)

def load_circuit_from_json(file_path):
    with open("../"+file_path, 'r') as f:
        data = json.load(f)
    return QuantumCircuit.from_qasm_str(data["qasm_code"])

def collect_connectivity_data(qlosure_csv, pytket_csv, edges):
    df_q = read_and_sort(qlosure_csv)
    df_p = read_and_sort(pytket_csv)

    results = []

    for i in range(len(df_q)):
        path = df_q.loc[i, 'file_path']
        circ = load_circuit_from_json(path)
        total, max_w, avg,depth,width,real_width = compute_connectivity_metrics(circ, edges)

        q_swaps = df_q.loc[i, 'sherbrooke_swaps_trivial']
        q_depth = df_q.loc[i, 'sherbrooke_depth_trivial']
        p_swaps = df_p.loc[i, 'sherbrooke_swaps_trivial']
        p_depth = df_p.loc[i, 'sherbrooke_depth_trivial']

        def safe_int(value):
            try:
                return int(value)
            except Exception:
                return float("inf")


        results.append({
            "file": path.split("/")[-1],
            "total": total,
            "max": max_w,
            "avg": avg,
            "depth": depth,
            "width": width,
            "surface": width * depth,
            "real_width":real_width,
            "q_swaps": safe_int(q_swaps),
            "q_depth": safe_int(q_depth),
            "p_swaps": safe_int(p_swaps),
            "p_depth": safe_int(p_depth)
        })

    return results




def plot_q_vs_p_by_metric(json_path="results.json", our_method='Qlosure', method='pytket', metric='real_width'):
    # Step 1: Load results from JSON
    with open(json_path, 'r') as f:
        results = json.load(f)

    # Step 2: Filter out entries that do not have both methods' data
    filtered_results = [
        r for r in results
        if f"{our_method}_swaps" in r and f"{method}_swaps" in r and
           f"{our_method}_depth" in r and f"{method}_depth" in r
    ]

    # Step 3: Collect and bucket metric values
    metric_values = [int(math.floor(r[metric])) for r in filtered_results]
    unique_metric_values = sorted(set(metric_values))
    
    min_val, max_val = min(unique_metric_values), max(unique_metric_values)
    #print(f"{metric} ∈ [{min_val}, {max_val}]")
    
    buckets = defaultdict(lambda: {'swaps': 0, 'depth': 0, 'count': 0})

    for r in filtered_results:
        level = int(math.floor(r[metric]))
        if level in unique_metric_values:
            if r[f"{our_method}_swaps"] < r[f"{method}_swaps"]:
                buckets[level]['swaps'] += 1
            elif r[f"{our_method}_swaps"] > r[f"{method}_swaps"]:
                buckets[level]['swaps'] -= 1

            if r[f"{our_method}_depth"] < r[f"{method}_depth"]:
                buckets[level]['depth'] += 1
            elif r[f"{our_method}_depth"] > r[f"{method}_depth"]:
                buckets[level]['depth'] -= 1

            buckets[level]['count'] += 1

    
    # Step 3: prepare data for plotting
    x = unique_metric_values
    swap_diff = [buckets[i]['swaps'] for i in x]
    depth_diff = [buckets[i]['depth'] for i in x]

    # Step 4: plot
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(14, 6), sharey=True)

    # Plot for Swaps (Q - P) on the left
    ax1.bar(x, swap_diff, color='blue', width=0.6)
    ax1.axhline(0, color='black', linewidth=0.8)
    ax1.set_xlabel(f"{metric.capitalize()} value (binned)")
    ax1.set_ylabel("Qlosure better (+) or worse (−)")
    ax1.set_title("Swaps (Q - P)")
    ax1.grid(True, linestyle='--', alpha=0.3)

    # Plot for Depth (Q - P) on the right
    ax2.bar(x, depth_diff, color='red', width=0.6)
    ax2.axhline(0, color='black', linewidth=0.8)
    ax2.set_xlabel(f"{metric.capitalize()} value (binned)")
    ax2.set_title("Depth (Q - P)")
    ax2.grid(True, linestyle='--', alpha=0.3)
    
    fig.suptitle(f"Comparison of {our_method} and {method}", fontsize=15)
    # Adjust layout to prevent overlap
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()



import json

def safe_int(value):
    try:
        return int(value)
    except Exception:
        return float("inf")

def collect_connectivity_data_for_multi_methods(benchmarks, methods, edges, output_path):

    results = []

    for benchmark in benchmarks:
        method_dfs = {}
        for method in methods:
            csv_path = f"../experiment_results/{benchmark}/{method}.csv"
            method_dfs[method] = read_and_sort(csv_path)
        
        baseline_method = methods[0]
        baseline_df = method_dfs[baseline_method]
        
        for i in range(len(baseline_df)):
            path = baseline_df.loc[i, 'file_path']
            circ = load_circuit_from_json(path)
            total, max_w, avg, depth, width, real_width = compute_connectivity_metrics(circ, edges)
            
            result_entry = {
                "benchmark": benchmark,
                "file": path.split("/")[-1],
                "total": total,
                "max": max_w,
                "avg": avg,
                "depth": depth,
                "width": width,
                "surface": width * depth,
                "real_width": real_width,
            }
            
            for method in methods:
                df = method_dfs[method]
                q_swaps = df.loc[i, 'sherbrooke_swaps_trivial']
                q_depth = df.loc[i, 'sherbrooke_depth_trivial']
                result_entry[f"{method}_swaps"] = safe_int(q_swaps)
                result_entry[f"{method}_depth"] = safe_int(q_depth)
            
            results.append(result_entry)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    return results





def improvment(benchmarks,methods,layout,backends,our_method="Qlosure"):
    raw_swaps = {method: [] for method in methods}
    raw_depth = {method: [] for method in methods}

    for benchmark in benchmarks:
        for backend in backends:
            folder = os.path.join("../experiment_results", benchmark)
            for method in methods:
                path = os.path.join(folder, f"{method}.csv")
                try:
                    with open(path, newline='') as f:
                        reader = csv.DictReader(f)
                        for row in reader:
                            raw_swaps[method].append(row.get(f"{backend}_swaps_{layout}", "0"))
                            raw_depth[method].append(row.get(f"{backend}_depth_{layout}", "0"))
                except FileNotFoundError:
                    print(f"Warning: {path} not found.")

    results_swaps = {}
    results_depth = {}
    results_qops = {}

    for method in methods:
        # fill non-digit with max
        sw = [int(v) if v.isdigit() else None for v in raw_swaps[method]]
        sw_max = max([v for v in sw if v is not None], default=0)
        results_swaps[method] = [v if v is not None else sw_max for v in sw]

        dp = [int(v) if v.isdigit() else None for v in raw_depth[method]]
        dp_max = max([v for v in dp if v is not None], default=0)
        results_depth[method] = [v if v is not None else dp_max for v in dp]
        

    def compute_improvement(results, our_method, methods):
        """
        Compute percentage improvement for each method relative to our_method.
        returns dict: {method: [swap_improvements], ...} in percent.
        """
        improvements = {}

        n = len(results[our_method])
        for method in methods:
            if method == our_method:
                continue
            imp_sw = []
            imp_dp = []
            for i in range(n):
                base_sw = results[method][i]
                our_sw = results[our_method][i]
                # reduction: (base - ours)/base * 100
                sw_imp = (base_sw - our_sw) / base_sw * 100 if base_sw != 0 else 0
                imp_sw.append(sw_imp)

                

            improvements[method] = imp_sw

        df_swaps = pd.DataFrame(improvements)

        print("\nAverage Swap Improvement (%) by Method\n")
        print(df_swaps.mean().round(2))
        return 

    compute_improvement(results_swaps, our_method, methods)
    
    print("*" * 50)
    compute_improvement(results_depth, our_method, methods)
    
    
def time_plot(banchmark="queko_54"):
    if banchmark != "queko_54":
        print("benchmark doesn't exsist")
        return
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.ticker import FuncFormatter


    df = pd.read_csv('../experiment_results/queko-bss-54qbt/excution_time.csv')
    data_grouped = {
        'Sherbrooke': {},
        'Ankaa 3': {},
        'Sherbrooke-2X': {}
    }

    for _, row in df.iterrows():
        x = row['qops']

        # append Sherbrooke time
        data_grouped['Sherbrooke'].setdefault(x, []).append(
            float(row['time_sherbrooke'])
        )
        # append Ankaa time
        data_grouped['Ankaa 3'].setdefault(x, []).append(
            float(row['time_ankaa'])
        )
        
        data_grouped['Sherbrooke-2X'].setdefault(x, []).append(
            float(row['qlosure_256'])
        )

    # --- 3) Updated plotting function ---
    def plot_grouped_scatter_with_noise(data_grouped, xlabel, ylabel, title):
        plt.figure(figsize=(10, 4))
        colors = plt.cm.tab10.colors
        plt.rcParams.update({'font.size': 21})
        markers = ['o', 's', '^', 'x', '+', '*', 'D', 'v']

        for i, (method, grouping) in enumerate(data_grouped.items()):
            color  = colors[i % len(colors)]
            marker = markers[i % len(markers)]

            xs = sorted(grouping.keys())
            ys = [grouping[x] for x in xs]

            # scatter points
            x_pts, y_pts = [], []
            for xv, vals in zip(xs, ys):
                x_pts.extend([xv] * len(vals))
                y_pts.extend(vals)
            plt.scatter(x_pts, y_pts, alpha=0.5, color=color, marker=marker, label=method)

            # mean ± std line
            means = [np.mean(vals) for vals in ys]
            stds  = [np.std(vals)  for vals in ys]
            plt.plot(xs, means, color=color, linewidth=2)
            plt.fill_between(xs,
                            np.array(means) - np.array(stds),
                            np.array(means) + np.array(stds),
                            color=color, alpha=0)
        def thousands_formatter(x, pos):
            if x >= 1000:
                return f'{int(x/1000)}k'
            else:
                return f'{int(x)}'
        
        formatter = FuncFormatter(thousands_formatter)
        plt.gca().xaxis.set_major_formatter(formatter)
        plt.gca().tick_params(axis='x', labelsize=16)
        plt.gca().tick_params(axis='y', labelsize=16)
        plt.xlabel(xlabel,fontsize =18)
        plt.ylabel(ylabel,fontsize =18)
        plt.title(title,fontsize =0)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend(fontsize=17)
        plt.tight_layout()
        plt.show()

    # --- 4) Call it with QOPs vs Time ---
    plot_grouped_scatter_with_noise(
        data_grouped,
        xlabel="QOPs",
        ylabel="Time (s)",
        title=""
    )
    
    
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import os
from matplotlib.ticker import FuncFormatter

def plot_ablation_results(methods,benchmark="queko-bss-81qbt",backend="sherbrooke",baseline = 'distance_only'):
    # 1) Read your CSV
    if benchmark not in ["queko-bss-81qbt"]:
        print("Unsupported benchmark. Use 'queko-bss-81qbt'.")
        return
        
    if backend =="sherbrooke":
        df = pd.read_csv("../experiment_results/queko-bss-81qbt/abl_study_sherbrooke.csv")
    elif backend == "ankaa":
        df = pd.read_csv("../experiment_results/queko-bss-81qbt/abl_study_ankaa.csv")
    else:
        print("Unsupported backend. Use 'sherbrooke' or 'ankaa'.")
        return
    df = df.sort_values(by='cycle').reset_index(drop=True)

    # 2) Detect all methods if no explicit order given
    swaps_cols = [c for c in df.columns if c.startswith('swaps_')]
    auto_methods = [c.replace('swaps_', '') for c in swaps_cols]

    # 3) Decide on final methods list
    if methods is not None:
        # keep only those that actually exist in the CSV
        methods = [m for m in methods if m in auto_methods]
        # warn if any in order aren’t present
        missing = set(methods) - set(methods)
        if missing:
            print(f"Warning: these methods not found in CSV and will be skipped: {missing}")
    else:
        methods = sorted(auto_methods)
        
    display_names = {
        'distance_only':       'Distance-only',
        'layer_adjusted':      'Layer-adjusted',
        'qlosure':             'Dependency-weighted',
        'our_layout_qlosure':  'Bidirectional-passes',
    }

    # Prepare grouped storage
    data_grouped = {
        'swaps': {m: defaultdict(list) for m in methods},
        'depth': {m: defaultdict(list) for m in methods}
    }

    # 4) Group data by cycle (x-axis)
    for _, row in df.iterrows():
        x = row['cycle']
        for m in methods:
            data_grouped['swaps'][m][x].append(row[f'swaps_{m}'])
            data_grouped['depth'][m][x].append(row[f'depth_{m}'])

    # 5) Generic plot function
    def _plot(data_dict, ylabel, title):
        plt.figure(figsize=(10, 6))
        plt.rcParams.update({'font.size': 21})
        colors = plt.cm.tab10.colors
        markers = ['o', 's', '^',  'D', 'v']

        for i, method in enumerate(methods):
            label = display_names.get(method, method)
            grouped = data_dict[method]
            col = colors[i % len(colors)]
            mk  = markers[i % len(markers)]

            xs = sorted(grouped.keys())
            ys_lists = [grouped[x] for x in xs]

            # scatter
            xs_sc, ys_sc = [], []
            for x_val, y_vals in zip(xs, ys_lists):
                xs_sc.extend([x_val] * len(y_vals))
                ys_sc.extend(y_vals)
            plt.scatter(xs_sc, ys_sc,s=100, alpha=0.5, color=col, marker=mk, label=label)

            # mean ± std
            means = np.array([np.mean(y) for y in ys_lists])
            stds  = np.array([np.std(y)  for y in ys_lists])
            plt.plot(xs, means, color=col, lw=2)
            plt.fill_between(xs, means - stds, means + stds, color=col, alpha=0)

        def thousands_formatter(x, pos):
            if x >= 1000:
                return f'{int(x/1000)}k'
            else:
                return f'{int(x)}'
        
        formatter = FuncFormatter(thousands_formatter)
        plt.gca().yaxis.set_major_formatter(formatter)
        plt.xlabel("Initial depth",fontsize=25)
        plt.ylabel(ylabel,fontsize=25)
        plt.title(title,fontsize=27)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend(fontsize=21)
        plt.tight_layout()
        plt.show()

    # 6) Draw both plots
    _plot(data_grouped['swaps'], ylabel="Swaps", title="")
    _plot(data_grouped['depth'], ylabel="Compiled Depth", title="")
    
    


    others     = [m for m in methods if m != baseline]

    for m in others:
        # ratio of baseline over m
        df[f'ratio_swaps_{m}'] = df[f'swaps_{baseline}'] / df[f'swaps_{m}']
        df[f'ratio_depth_{m}'] = df[f'depth_{baseline}'] / df[f'depth_{m}']

        # percentage improvement = (ratio − 1) × 100
        df[f'perc_improv_swaps_{m}'] = (df[f'ratio_swaps_{m}'] - 1) * 100
        df[f'perc_improv_depth_{m}'] = (df[f'ratio_depth_{m}'] - 1) * 100

    for m in others:
        mean_sw  = df[f'ratio_swaps_{m}'].mean()
        mean_dp  = df[f'ratio_depth_{m}'].mean()
        mean_psw = df[f'perc_improv_swaps_{m}'].mean()
        mean_pdp = df[f'perc_improv_depth_{m}'].mean()
        print(
            f"{m:10s} — "
            f"avg swaps ratio: {mean_sw:.3f}, "
            f"avg depth ratio: {mean_dp:.3f}, "
            f"avg swaps : {mean_psw:.1f}%, "
            f"avg depth : {mean_pdp:.1f}%"
        )

