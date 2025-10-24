from collections import deque, defaultdict
import random


def create_leveled_extended_successor_set(front_points, dag, extended_set_size=40, node_data=None):
    visited = []
    layer_index = {}
    queue = deque()

    for point in front_points:
        queue.append((point, 1))

    while queue and len(visited) < extended_set_size:
        current, current_layer = queue.popleft()

        if current in dag:
            for succ in dag[current]:
                if succ not in layer_index:
                    if node_data and node_data[succ]["is_control_flow"]:
                        continue
                    visited.append(succ)
                    layer_index[succ] = current_layer + 1

                    queue.append((succ, current_layer + 1))

                    if len(visited) >= extended_set_size:
                        break

    return visited, layer_index


def qlosure_poly_heuristic(front_layer, extended_layer, mapping, distance_matrix, node_data, decay_parameter, deps_count, extended_layer_index, gate, with_closure_depth=True, with_layer_factor=True):
    W = 1
    # 1) max decay
    max_decay = max(decay_parameter[gate[0]],
                    decay_parameter[gate[1]])
    front_layer_size = 0
    extended_layer_size = 0

    # 2) front-layer normalization
    f_distance = 0
    for g in front_layer:
        tmp_block_distance = 0
        if node_data[g]['is_control_flow']:
            continue
        for q1, q2 in node_data[g]['coupled_qubits_accessed']:
            front_layer_size += 1
            Q1, Q2 = mapping[q1], mapping[q2]
            tmp_block_distance += distance_matrix[Q1][Q2]
        deps = deps_count[g] if with_closure_depth else 0
        f_distance += (deps+1) * tmp_block_distance

    e_distance = 0
    for g in extended_layer:
        tmp_block_distance = 0
        if node_data[g]['is_control_flow']:
            continue
        for q1, q2 in node_data[g]['coupled_qubits_accessed']:
            extended_layer_size += 1
            Q1, Q2 = mapping[q1], mapping[q2]
            tmp_block_distance += distance_matrix[Q1][Q2]
        deps = deps_count[g] if with_closure_depth else 0
        layer_factor = (extended_layer_index.get(
            g, 0) + 1) if with_layer_factor else 1

        e_distance += (deps+1) * tmp_block_distance / layer_factor

    H = max_decay * ((f_distance / front_layer_size if front_layer_size else 0) +
                     W * (e_distance / extended_layer_size if extended_layer_size else 0))
    return H


def find_min_score_swap_gate(heuristic_score, epsilon=1e-10):
    random.seed(21)
    min_score = float('inf')
    best_swaps = []

    for gate, score in heuristic_score.items():

        if score - min_score < -epsilon:
            min_score = score
            best_swaps = [gate]
        elif abs(score - min_score) <= epsilon:
            best_swaps.append(gate)

    best_swaps.sort()

    return random.choice(best_swaps)
    # return best_swaps[0] if best_swaps else None


def evaluate_mapping_quality(front_layer, extended_layer, mapping, distance_matrix, node_data, decay_parameter, deps_count, extended_layer_index):
    W = 1

    # 2) front-layer normalization
    f_distance = 0
    for g in front_layer:
        for q1, q2 in node_data[g]['coupled_qubits_accessed']:
            # q1, q2 = node_data[g]['qubits_accessed']
            Q1, Q2 = mapping[q1], mapping[q2]
            deps = deps_count[g]
            # print(f"     for {g} , dep : {deps},Qops {Q1,Q2}, dist :{distance_matrix[Q1][Q2]}")
            f_distance += (deps+1) * distance_matrix[Q1][Q2]
    f_norm = f_distance / len(front_layer) if front_layer else 0

    # 3) bucket extended_layer by layer
    layer_sums = defaultdict(float)
    layer_counts = defaultdict(int)
    for g in extended_layer:
        idx = extended_layer_index.get(g, 0)
        for q1, q2 in node_data[g]['coupled_qubits_accessed']:
            # q1, q2 = node_data[g]['qubits_accessed']
            Q1, Q2 = mapping[q1], mapping[q2]
            deps = deps_count[g]
            # print(f"     for {g} , dep : {deps},Qops {Q1,Q2}, dist :{distance_matrix[Q1][Q2]}, index {idx}")
            weight = (deps+1) * distance_matrix[Q1][Q2]
            layer_sums[idx] += weight
            # print("layer_sums :",layer_sums)
            layer_counts[idx] += 1
    # print("f nor :",f_norm)

    # 4) normalize each bucket, then average
    if layer_counts:
        layer_decay = {i: i for i in layer_counts}
        e_norm = sum(
            layer_sums[i] / (layer_counts[i] * (layer_decay[i]+1))
            for i in layer_counts
        )
    else:
        e_norm = 0
    # 5) final heuristic

    H = (f_norm + W * e_norm)
    return H
