from src.utils.python_to_isl import list_to_isl_set
from collections import deque
import random
import time
import math
from collections import defaultdict


def qlosure_poly_heuristic(front_layer, extended_layer, mapping, distance_matrix, access, decay_parameter, deps_count, extended_layer_index, gate):
    max_decay = max(decay_parameter[gate[0]],
                    decay_parameter[gate[1]])

    f_distance = 0
    for g in front_layer:
        q1, q2 = access[g]
        Q1, Q2 = mapping[q1], mapping[q2]
        deps = deps_count[g]
        f_distance += (deps+1) * distance_matrix[Q1][Q2]
    f_norm = f_distance / len(front_layer) if front_layer else 0

    layer_sums = defaultdict(float)
    layer_counts = defaultdict(int)
    for g in extended_layer:
        idx = extended_layer_index.get(g, 0)
        q1, q2 = access[g]
        Q1, Q2 = mapping[q1], mapping[q2]
        deps = deps_count[g]
        weight = (deps+1) * distance_matrix[Q1][Q2]
        layer_sums[idx] += weight
        layer_counts[idx] += 1

    if layer_counts:
        layer_decay = {i: i for i in layer_counts}
        e_norm = sum(
            layer_sums[i] / (layer_counts[i] * (layer_decay[i]))
            for i in layer_counts
        )
    else:
        e_norm = 0

    H = max_decay * (f_norm + e_norm)
    return H


def distance_only_poly_heuristic(front_layer, extended_layer, mapping, distance_matrix, access, decay_parameter, deps_count, extended_layer_index, gate):
    W = 1
    front_layer_size = len(front_layer)
    extended_layer_size = len(extended_layer)

    max_decay = max(decay_parameter[gate[0]], decay_parameter[gate[1]])

    f_distance = 0

    for g in front_layer:
        q1, q2 = access[g]
        Q1, Q2 = mapping[q1], mapping[q2]

        f_distance += distance_matrix[Q1][Q2]

    e_distance = 0
    for g in extended_layer:
        q1, q2 = access[g]
        Q1, Q2 = mapping[q1], mapping[q2]

        e_distance += distance_matrix[Q1][Q2]

    H = max_decay * (f_distance / front_layer_size + W *
                     ((e_distance / extended_layer_size) if extended_layer_size else 0))

    return H


def layered_poly_qlosure_heuristic(front_layer, extended_layer,
                                   mapping, distance_matrix, access,
                                   decay_parameter, deps_count,
                                   extended_layer_index, gate):
    W = 1
    # 1) max decay
    max_decay = max(decay_parameter[gate[0]],
                    decay_parameter[gate[1]])

    # 2) front-layer normalization
    f_distance = 0
    for g in front_layer:
        q1, q2 = access[g]
        Q1, Q2 = mapping[q1], mapping[q2]
        deps = deps_count[g]
        # print(f"     for {g} , dep : {deps},Qops {Q1,Q2}, dist :{distance_matrix[Q1][Q2]}")
        f_distance += distance_matrix[Q1][Q2]
    f_norm = f_distance / len(front_layer) if front_layer else 0

    # 3) bucket extended_layer by layer
    layer_sums = defaultdict(float)
    layer_counts = defaultdict(int)
    for g in extended_layer:
        idx = extended_layer_index.get(g, 0)
        q1, q2 = access[g]
        Q1, Q2 = mapping[q1], mapping[q2]
        deps = deps_count[g]
        # print(f"     for {g} , dep : {deps},Qops {Q1,Q2}, dist :{distance_matrix[Q1][Q2]}, index {idx}")
        weight = distance_matrix[Q1][Q2]
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

    H = max_decay * (f_norm + W * e_norm)
    return H


def create_extended_successor_set(front_points, dag, access, extended_set_size=40):
    extended_set_size = extended_set_size * 5

    visited = set()
    queue = deque(front_points)

    while queue and len(visited) < extended_set_size:
        current = queue.popleft()

        if current in dag:
            if len(access.get(current, [])) > 1:
                visited.add(current)

            for succ in dag[current]:
                if succ not in visited:
                    queue.append(succ)

                    if len(visited) >= extended_set_size:
                        break

    return list(visited)


def create_leveled_extended_successor_set(front_points, dag, access, extended_set_size=40):
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
                    visited.append(succ)
                    layer_index[succ] = current_layer + 1

                    queue.append((succ, current_layer + 1))

                    if len(visited) >= extended_set_size:
                        break

    return visited, layer_index


def get_all_predecessors(node, predecessors, visited=None):
    if visited is None:
        visited = set()
    if node in visited:
        return set()
    visited.add(node)
    preds = set(predecessors.get(node, []))
    for p in predecessors.get(node, []):
        preds |= get_all_predecessors(p, predecessors, visited)
    return preds


def create_lookahead_path_set(front_points, dag, predecessors, lookahead_path_size=20):

    def dfs_paths(node, depth):

        if depth >= lookahead_path_size:
            return [[node]]
        successors = list(dag[node])

        valid_successors = [succ for succ in successors if len(
            predecessors.get(succ, [])) == 1]
        if not valid_successors:
            return [[node]]
        paths = []
        for succ in valid_successors:
            for sub_path in dfs_paths(succ, depth + 1):
                paths.append([node] + sub_path)
        return paths

    all_paths = []

    for front in front_points:
        paths_from_front = dfs_paths(front, 0)
        for path in paths_from_front:

            extended_nodes = set(path)

            for node in path:
                extended_nodes |= get_all_predecessors(node, predecessors)

            extended_path = list(extended_nodes)
            if front not in extended_path:
                extended_path.insert(0, front)
            all_paths.append(extended_path)
    return all_paths


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


def order_extended_layer_from_successors(extended_layer, successors):

    extended_set = set(extended_layer)

    in_degree = {node: 0 for node in extended_layer}
    for node in extended_layer:
        for succ in successors.get(node, []):
            if succ in extended_set:
                in_degree[succ] += 1

    layers = []
    current_layer = [node for node in extended_layer if in_degree[node] == 0]

    while current_layer:
        layers.append(current_layer)
        next_layer = []

        for node in current_layer:
            for succ in successors.get(node, []):
                if succ in extended_set:
                    in_degree[succ] -= 1
                    if in_degree[succ] == 0:
                        next_layer.append(succ)
        current_layer = next_layer

    return layers
