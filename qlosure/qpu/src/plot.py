import matplotlib.pyplot as plt
from matplotlib import patches



def plot_topology(backend="ibm_sherbrooke", color="#25b6f5"):
    if backend == "ibm_sherbrooke":
        sherbrooke(color)
    elif backend == "ibm_sherbrooke2x":
        sherbrooke2x(color)
    elif backend == "sycamore":
        sycamore(color)
    elif backend == "aspen_4":
        aspen_4(color)
    elif backend == "anka3":
        ankaa3(color)
    else:
        print(f"Unknown backend: {backend}. Please choose from 'ibm_sherbrooke', 'ibm_sherbrooke2x', 'sycamore', 'aspen_4', or 'anka3'.")

def sycamore(color = '#25b6f5'):
    positions = {
        i: (((i % 6)*2 + 1) if i//6 % 2 == 1 else (i % 6)*2, i // 6)
        for i in range(54)
    }

    edges = []

    for i, (x, y) in positions.items():
        if y % 2 == 1: 
            for j, (x2, y2) in positions.items():
                if (abs(y2 - y )== 1) and (abs(x2 - x) == 1):  
                    edges.append((i, j))


    r_normal   = 0.25
    edge_color = color
    edge_width = 2
    node_fc    = 'white'
    node_ec    = color
    node_lw    = 2

    fig, ax = plt.subplots(figsize=(8,6))

    for u,v in edges:
        x1,y1 = positions[u]
        x2,y2 = positions[v]
        ax.plot([x1,x2], [y1,y2],
                color=edge_color, linewidth=edge_width, zorder=2)

    for node, (x,y) in positions.items():
        circ = patches.Circle((x,y), radius=r_normal,
                            facecolor=node_fc,
                            edgecolor=node_ec,
                            lw=node_lw,
                            zorder=2)
        ax.add_patch(circ)

    all_x = [x for x,y in positions.values()]
    all_y = [y for x,y in positions.values()]
    xmin, xmax = min(all_x)-1, max(all_x)+1
    ymin, ymax = min(all_y)-1, max(all_y)+1

    ax.set_aspect('equal')
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.axis('off')
    plt.tight_layout()
    plt.show()


def aspen_4(color = '#25b6f5'): 
    positions = {
        0: (1, 0), 1: (2, 0), 2: (5, 0), 3: (6, 0),
        4: (0, 1), 5: (3, 1), 6: (4, 1), 7: (7, 1),
        8: (0, 2), 9: (3, 2), 10: (4, 2), 11: (7, 2),
        12: (1, 3), 13: (2, 3), 14: (5, 3), 15: (6, 3)
    }


    edges = [
        (0,1),
        (2,3),
        (0,4),
        (1,5),
        (5,6),
        (6,2),
        (3,7),
        (4,8),
        (5,9),
        (6,10),
        (7,11),
        (8,12),
        (9,13),
        (9,10),
        (10,14),
        (11,15),
        (12,13),
        (14,15)
    ]

    r_normal   = 0.25
    edge_color = color
    edge_width = 2
    node_fc    = 'white'
    node_ec    = color
    node_lw    = 2

    fig, ax = plt.subplots(figsize=(8,6))

    for u,v in edges:
        x1,y1 = positions[u]
        x2,y2 = positions[v]
        ax.plot([x1,x2], [y1,y2],
                color=edge_color, linewidth=edge_width, zorder=2)

    for node, (x,y) in positions.items():
        circ = patches.Circle((x,y), radius=r_normal,
                            facecolor=node_fc,
                            edgecolor=node_ec,
                            lw=node_lw,
                            zorder=2)
        ax.add_patch(circ)

    # finalize plot
    all_x = [x for x,y in positions.values()]
    all_y = [y for x,y in positions.values()]
    xmin, xmax = min(all_x)-1, max(all_x)+1
    ymin, ymax = min(all_y)-1, max(all_y)+1

    ax.set_aspect('equal')
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.axis('off')
    plt.tight_layout()
    plt.show()



def sherbrooke(color = "#25b6f5"):
    n_rows, n_cols = 13, 15 
    r_normal = 0.25
    edge_color = color
    edge_width = 2
    node_fc = 'white'
    node_ec = color
    node_lw = 2
    label_fontsize = 10


    positions = {i: (i % n_cols, i // n_cols) for i in range(n_rows * n_cols)}


    edges = []
    for i in range(n_rows * n_cols):
        x, y = positions[i]
        neighbors = [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)]
        for nx, ny in neighbors:
            if 0 <= nx < n_cols and 0 <= ny < n_rows:
                j = ny * n_cols + nx
                if j > i:
                    edges.append((i, j))


    remove_nodes = [0,16,15,20,18,19,24,22,23,26,27,28,46,47,48,50,51,52,54,55,56,58,59,75,76,78,79,80,82,83,84,86,87,88,
                    106,107,108,110,111,112,114,115,116,118,119,135,136,138,139,140,142,143,144,146,147,148,
                    166,167,168,170,171,172,174,175,176,178,179,194]
    filtered_edges = [(u, v) for u, v in edges if u not in remove_nodes and v not in remove_nodes]

    fig, ax = plt.subplots(figsize=(6, 8))

    for u, v in filtered_edges:
        x1, y1 = positions[u]
        x2, y2 = positions[v]
        ax.plot([x1, x2], [y1, y2],
                color=edge_color, linewidth=edge_width, zorder=2)

    for node, (x, y) in positions.items():
        if node in remove_nodes:
            continue
        else:
            fc, ec, lw = node_fc, node_ec, node_lw

        circ = patches.Circle((x, y), radius=r_normal,
                            facecolor=fc,
                            edgecolor=ec,
                            lw=lw,
                            zorder=2)
        ax.add_patch(circ)
        

    ax.set_aspect('equal')
    all_x = [x for x,y in positions.values()]
    all_y = [y for x,y in positions.values()]
    ax.set_xlim(min(all_x)-1, max(all_x)+1)
    ax.set_ylim(min(all_y)-1, max(all_y)+1)
    ax.axis('off')
    plt.tight_layout()
    plt.show()


def sherbrooke2x(color = '#25b6f5'):
        
    n_rows, n_cols = 13, 15
    r = 0.25
    edge_color = color
    edge_width = 2
    node_fc = 'white'
    node_ec = color
    node_lw = 2
    fontsize = 8

    positions = {i: (i % n_cols, i // n_cols) for i in range(n_rows * n_cols)}

    edges = []
    for i, (x, y) in positions.items():
        for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]:
            nx, ny = x+dx, y+dy
            if 0 <= nx < n_cols and 0 <= ny < n_rows:
                j = ny * n_cols + nx
                if j > i:
                    edges.append((i, j))

    removed = {
        16,15,20,18,19,24,22,23,26,27,28,46,47,48,50,51,52,54,55,56,58,59,
        75,76,78,79,80,82,83,84,86,87,88,106,107,108,110,111,112,114,115,116,
        118,119,135,136,138,139,140,142,143,144,146,147,148,166,167,168,170,
        171,172,174,175,176,178,179
    }
    filtered = [(u, v) for u, v in edges if u not in removed and v not in removed]

    label_offset = max(positions.keys()) + 1 
    axis_offset = (15, 0)

    all_positions = positions.copy()
    for orig, (x, y) in positions.items():
        if orig in removed:
            continue
        new_id = orig + label_offset
        all_positions[new_id] = (x + axis_offset[0], y + axis_offset[1])

    cross_edges = [(194, 375), (164, 345),(134,315),(104,285),(74,255),(44,225),(14,195)]
    nodes_to_remove = [0,389]
    filtered = [(u, v) for u, v in filtered if u not in nodes_to_remove and v not in nodes_to_remove]
    new_edges = [(u + label_offset, v + label_offset) for u, v in filtered]
    new_edges = [(u, v) for u, v in new_edges if u not in nodes_to_remove and v not in nodes_to_remove]
    new_edges.append((195, 196))

    fig = plt.figure(figsize=(16, 16))
    ax = fig.add_axes([0, 0, 1, 1]) 
    
    for u, v in filtered:
        x1, y1 = positions[u]; x2, y2 = positions[v]
        ax.plot([x1, x2], [y1, y2], color=edge_color, linewidth=edge_width)

    for u, v in new_edges:
        x1, y1 = all_positions[u]; x2, y2 = all_positions[v]
        ax.plot([x1, x2], [y1, y2], color=edge_color, linewidth=edge_width)

    for u, v in cross_edges:
        if u in positions and v in all_positions:
            x1, y1 = positions[u]; x2, y2 = all_positions[v]
            ax.plot([x1, x2], [y1, y2], color=edge_color, linewidth=edge_width)

    for node, (x, y) in all_positions.items():
        if node < label_offset and node in removed or node in nodes_to_remove:
            continue
        circ = patches.Circle((x, y), r, facecolor=node_fc, edgecolor=node_ec, lw=node_lw, zorder=2)
        ax.add_patch(circ)
        #ax.text(x, y, str(node), ha='center', va='center', fontsize=fontsize, zorder=3)

    ax.set_aspect('equal')
    ax.set_xlim(-0.5, n_cols*2 + axis_offset[0] - 0.5)
    ax.set_ylim(-0.5, n_rows - 0.5)
    ax.axis('off')
    plt.margins(0, 0)
    ax.margins(0, 0)

    original_nodes = set(positions.keys()) - removed - set(nodes_to_remove)
    offset_nodes = set(range(label_offset, label_offset + len(positions))) - set(nodes_to_remove) - {n + label_offset for n in removed}
    total_nodes = len(original_nodes) + len(offset_nodes)

    original_edges = len(filtered)
    new_graph_edges = len(new_edges) 
    cross_graph_edges = len(cross_edges)
    total_edges = original_edges + new_graph_edges + cross_graph_edges


    plt.show()


def ankaa3(color="#256fb5"):

    n_rows, n_cols = 7, 12  
    r_normal = 0.25
    edge_color = color
    edge_width = 2
    node_fc = 'white'
    node_ec = color
    node_lw = 2
    label_fontsize = 10

    positions = {i: (i % n_cols, i // n_cols) for i in range(n_rows * n_cols)}

    edges = []
    for i in range(n_rows * n_cols):
        x, y = positions[i]
        neighbors = [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)]
        for nx, ny in neighbors:
            if 0 <= nx < n_cols and 0 <= ny < n_rows:
                j = ny * n_cols + nx
                if j > i:
                    edges.append((i, j))
                    
    remove_nodes = [5,77]
    filtered_edges = [(u, v) for u, v in edges if u not in remove_nodes and v not in remove_nodes]

    fig, ax = plt.subplots(figsize=(8, 8))
    removed_edges = [(26,38),(7,8),(31,43),(45,46),(53,65)]
    for u, v in filtered_edges:
        if (u,v) in removed_edges:
            continue
        x1, y1 = positions[u]
        x2, y2 = positions[v]
        ax.plot([x1, x2], [y1, y2],
                color=edge_color, linewidth=edge_width, zorder=2)


    for node, (x, y) in positions.items():
        if node in remove_nodes:
            continue
        else:
            fc, ec, lw = node_fc, node_ec, node_lw

        circ = patches.Circle((x, y), radius=r_normal,
                            facecolor=fc,
                            edgecolor=ec,
                            lw=lw,
                            zorder=2)
        ax.add_patch(circ)
        #ax.text(x, y, str(node), ha='center', va='center',
        #        fontsize=label_fontsize, zorder=3)

    ax.set_aspect('equal')
    all_x = [x for x,y in positions.values()]
    all_y = [y for x,y in positions.values()]
    ax.set_xlim(min(all_x)-1, max(all_x)+1)
    ax.set_ylim(min(all_y)-1, max(all_y)+1)
    ax.axis('off')
    plt.tight_layout()
    plt.show()
