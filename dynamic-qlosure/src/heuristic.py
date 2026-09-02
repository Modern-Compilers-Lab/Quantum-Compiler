from collections import deque, defaultdict,Counter
from typing import Dict, Optional
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



def distance_poly_heuristic(front_layer, extended_layer, mapping, distance_matrix, node_data, decay_parameter, deps_count, extended_layer_index, gate, with_closure_depth=True, with_layer_factor=True):
    W = 0.5
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
        f_distance +=  tmp_block_distance

    e_distance = 0
    for g in extended_layer:
        tmp_block_distance = 0
        if node_data[g]['is_control_flow']:
            continue
        for q1, q2 in node_data[g]['coupled_qubits_accessed']:
            extended_layer_size += 1
            Q1, Q2 = mapping[q1], mapping[q2]
            tmp_block_distance += distance_matrix[Q1][Q2]
        layer_factor = (extended_layer_index.get(
            g, 0) + 1) if with_layer_factor else 1

        e_distance +=  tmp_block_distance 

    H = max_decay * ((f_distance / front_layer_size if front_layer_size else 0) +
                     W * (e_distance / extended_layer_size if extended_layer_size else 0) ) 
    return H



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
                     W * (e_distance / extended_layer_size if extended_layer_size else 0) ) 
    return H



def new_qlosure_poly_heuristic(
    front_layer, extended_layer, mapping, distance_matrix, node_data,
    decay_parameter, deps_count, extended_layer_index, gate,
    qubit_props,  # add here so we can fetch two-qubit error
    with_closure_depth=True, with_layer_factor=True,
    lambda_hot=0.5, alpha_usage=1.0, beta_error=1.0,
):
    from collections import Counter, defaultdict
    from typing import Optional

    def _scale01(d):
        if not d:
            return defaultdict(float)
        lo, hi = min(d.values()), max(d.values())
        if hi == lo:
            return defaultdict(float)
        return defaultdict(float, {k: (v - lo) / (hi - lo) for k, v in d.items()})

    # --- 1) Usage density to avoid hotspots ---
    usage = Counter()
    for layer in (front_layer, extended_layer):
        for g in layer:
            nd = node_data[g]
            if nd.get("is_control_flow"):
                continue
            for q1, q2 in nd["coupled_qubits_accessed"]:
                usage[mapping[q1]] += 1
                usage[mapping[q2]] += 1
    usage01 = _scale01(usage)

    # --- 2) Heuristic computation ---
    W = 1
    max_decay = max(decay_parameter[gate[0]], decay_parameter[gate[1]])
    front_layer_size = 0
    extended_layer_size = 0
    f_distance = 0.0
    e_distance = 0.0

    def _pair_error(q0, q1):
        """Return two-qubit error for physical pair (q0,q1)."""
        e01 = qubit_props[str(q0)]["two_qubit_err"].get(str(q1))
        e10 = qubit_props[str(q1)]["two_qubit_err"].get(str(q0))
        if e01 is not None and e10 is not None:
            return 0.5 * (e01 + e10)
        return e01 or e10 or 0.0

    # --- 3) Front layer ---
    for g in front_layer:
        nd = node_data[g]
        if nd.get("is_control_flow"):
            continue
        tmp = 0.0
        for q1, q2 in nd["coupled_qubits_accessed"]:
            front_layer_size += 1
            Q1, Q2 = mapping[q1], mapping[q2]
            base_dist = distance_matrix[Q1][Q2]
            # combine physical error + hotspot penalty
            err = _pair_error(Q1, Q2)
            hot_mult = 1.0 + lambda_hot * (usage01[Q1] + usage01[Q2]) * 0.5
            err_mult = 1.0 + beta_error * err
            # tmp += base_dist * hot_mult * err_mult
            tmp += base_dist   * err_mult * hot_mult
        deps = deps_count[g] if with_closure_depth else 0
        f_distance += (deps + 1) * tmp

    # --- 4) Extended layer ---
    for g in extended_layer:
        nd = node_data[g]
        if nd.get("is_control_flow"):
            continue
        tmp = 0.0
        for q1, q2 in nd["coupled_qubits_accessed"]:
            extended_layer_size += 1
            Q1, Q2 = mapping[q1], mapping[q2]
            base_dist = distance_matrix[Q1][Q2]
            err = _pair_error(Q1, Q2)
            hot_mult = 1.0 + lambda_hot * (usage01[Q1] + usage01[Q2]) * 0.5
            err_mult = 1.0 + beta_error * err
            # tmp += base_dist * hot_mult * err_mult
            tmp += base_dist  * err_mult * hot_mult
        deps = deps_count[g] if with_closure_depth else 0
        layer_factor = (extended_layer_index.get(g, 0) + 1) if with_layer_factor else 1
        e_distance += (deps + 1) * tmp / layer_factor

    H = max_decay * (
        (f_distance / front_layer_size if front_layer_size else 0.0)
        + W * (e_distance / extended_layer_size if extended_layer_size else 0.0)
    )
    return H



def depth_poly_heuristic(
    front_layer, extended_layer, mapping, distance_matrix, node_data,
    decay_parameter, deps_count, extended_layer_index, gate,
    qubit_props,  # kept for API compatibility; not used in this heuristic
    with_closure_depth=True, with_layer_factor=True,  # kept; not used
    lambda_hot=0.5, alpha_usage=1.0, beta_error=1.0,  # kept; not used
):
    """
    New heuristic using layered, dependency-weighted distances and a depth-based tie-breaker,
    while keeping the original function signature.
    """
    # ------------------------------
    # 0) Build a "layers" structure from the existing inputs
    # ------------------------------
    # Layer 0 is the front layer. Extended layers follow according to extended_layer_index.
    # extended_layer_index[g] is assumed to start at 0 for the first extended layer.
    max_ext_idx = max(extended_layer_index.values()) if extended_layer_index else -1
    num_ext_layers = max_ext_idx + 1 if max_ext_idx >= 0 else 0

    # layers[0] = front_layer; layers[1 + i] = extended layer i
    layers = []
    layers.append(list(front_layer))  # layer 0
    for _ in range(num_ext_layers):
        layers.append([])

    for g in extended_layer:
        idx = extended_layer_index.get(g, 0)
        target_layer = 1 + idx  # shift because 0 is front_layer
        # grow if needed (defensive)
        while target_layer >= len(layers):
            layers.append([])
        layers[target_layer].append(g)

    # ------------------------------
    # 1) Build `access` and `depths` from node_data
    # ------------------------------
    # access[g] -> (logical_q1, logical_q2)
    # depths[q] -> max layer index in which logical qubit q appears
    access = {}
    depths = {}

    for layer_index, layer in enumerate(layers):
        for g in layer:
            nd = node_data[g]
            if nd.get("is_control_flow"):
                continue
            # assume each gate has at least one 2-qubit pair in coupled_qubits_accessed
            # if there are multiple, we take the first; adapt if your data is different
            if not nd["coupled_qubits_accessed"]:
                continue
            q1, q2 = nd["coupled_qubits_accessed"][0]
            access[g] = (q1, q2)

            # update depths for logical qubits
            depths[q1] = max(depths.get(q1, 0), layer_index)
            depths[q2] = max(depths.get(q2, 0), layer_index)

    # ------------------------------
    # 2) Safety: handle empty layers
    # ------------------------------
    max_layers = len(layers)
    if max_layers == 0:
        return 0.0

    # ------------------------------
    # 3) Decay & depth-rate (as in your new snippet)
    # ------------------------------
    # decay for the gate under consideration (keeps original behavior)
    max_decay = max(decay_parameter[gate[0]], decay_parameter[gate[1]])

    # depth_rate: normalize depth of the qubits in this gate w.r.t. max depth
    if depths:
        max_depth_val = max(depths.values())
        if max_depth_val > 0:
            depth_rate = max(depths.get(gate[0], 0), depths.get(gate[1], 0)) / max_depth_val
        else:
            depth_rate = 0.0
    else:
        depth_rate = 0.0

    # Weights for distance and depth tie-breaker (tunable constants)
    W_distance = 1.0
    tie_breaker_weight = 0.01

    # ------------------------------
    # 4) Main loop: layered, dependency-weighted distances
    # ------------------------------
    total_distance_score = 0.0
    total_layer_weight = 0.0

    for i in range(max_layers):
        layer = layers[i]
        layer_size = len(layer)
        layer_weight = 1.0 / (i + 1)  # earlier layers matter more
        total_layer_weight += layer_weight

        if layer_size == 0:
            # contribute nothing for empty layers
            continue

        weighted_distances = []
        per_gate_weights = []

        for g in layer:
            if g not in access:
                # e.g. control flow or something with no qubit access
                continue

            q1, q2 = access[g]
            P1, P2 = mapping[q1], mapping[q2]
            deps = deps_count[g] if with_closure_depth else 0
            gate_weight = deps + 1

            d = distance_matrix[P1][P2]
            weighted_distances.append(gate_weight * d)
            per_gate_weights.append(gate_weight)

        # normalized layer distance: average criticality-weighted distance (safe division)
        denom = sum(per_gate_weights)
        if denom > 0:
            layer_distance = sum(weighted_distances) / denom
        else:
            layer_distance = 0.0

        # accumulate with layer weight
        total_distance_score += layer_weight * layer_distance

    # normalize by total layer weight (so heuristic is independent of lookahead length)
    if total_layer_weight > 0:
        total_distance_score /= total_layer_weight

    # ------------------------------
    # 5) Final heuristic
    # ------------------------------
    # combine distance and depth-based tie-breaker, scaled by max_decay
    H = max_decay * (W_distance * total_distance_score + tie_breaker_weight * depth_rate)
    return H


def depth_poly_heuristic(
    front_layer, extended_layer, mapping, distance_matrix, node_data,
    decay_parameter, deps_count, extended_layer_index, gate,
    qubit_props,
    with_closure_depth=True, with_layer_factor=True,  # kept; not used
    lambda_hot=0.5, alpha_usage=1.0, beta_error=1.0,  # alpha_usage kept; not used
):
    """
    Layered, dependency-weighted distances with a depth-based tie-breaker.
    Pair distances are scaled by an error multiplier and a hotspot multiplier;
    set beta_error or lambda_hot to 0 to disable either.
    """
    from collections import Counter

    def _pair_error(q0, q1):
        """Return two-qubit error for physical pair (q0,q1)."""
        e01 = qubit_props[str(q0)]["two_qubit_err"].get(str(q1))
        e10 = qubit_props[str(q1)]["two_qubit_err"].get(str(q0))
        if e01 is not None and e10 is not None:
            return 0.5 * (e01 + e10)
        return e01 or e10 or 0.0

    # hotspot usage density over the lookahead window
    usage = Counter()
    if lambda_hot > 0.0:
        for layer_ in (front_layer, extended_layer):
            for g in layer_:
                nd = node_data[g]
                if nd.get("is_control_flow"):
                    continue
                for q1_, q2_ in nd["coupled_qubits_accessed"]:
                    usage[mapping[q1_]] += 1
                    usage[mapping[q2_]] += 1
    usage01 = _scale01(usage)

    # ------------------------------
    # 0) Build a "layers" structure from the existing inputs
    # ------------------------------
    # Layer 0 is the front layer. Extended layers follow according to extended_layer_index.
    # extended_layer_index[g] is assumed to start at 0 for the first extended layer.
    max_ext_idx = max(extended_layer_index.values()) if extended_layer_index else -1
    num_ext_layers = max_ext_idx + 1 if max_ext_idx >= 0 else 0

    # layers[0] = front_layer; layers[1 + i] = extended layer i
    layers = []
    layers.append(list(front_layer))  # layer 0
    for _ in range(num_ext_layers):
        layers.append([])

    for g in extended_layer:
        idx = extended_layer_index.get(g, 0)
        target_layer = 1 + idx  # shift because 0 is front_layer
        # grow if needed (defensive)
        while target_layer >= len(layers):
            layers.append([])
        layers[target_layer].append(g)

    # ------------------------------
    # 1) Build `access` and `depths` from node_data
    # ------------------------------
    # access[g] -> (logical_q1, logical_q2)
    # depths[q] -> max layer index in which logical qubit q appears
    access = {}
    depths = {}

    for layer_index, layer in enumerate(layers):
        for g in layer:
            nd = node_data[g]
            if nd.get("is_control_flow"):
                continue
            # assume each gate has at least one 2-qubit pair in coupled_qubits_accessed
            # if there are multiple, we take the first; adapt if your data is different
            if not nd["coupled_qubits_accessed"]:
                continue
            q1, q2 = nd["coupled_qubits_accessed"][0]
            access[g] = (q1, q2)

            # update depths for logical qubits
            depths[q1] = max(depths.get(q1, 0), layer_index)
            depths[q2] = max(depths.get(q2, 0), layer_index)

    # ------------------------------
    # 2) Safety: handle empty layers
    # ------------------------------
    max_layers = len(layers)
    if max_layers == 0:
        return 0.0

    # ------------------------------
    # 3) Decay & depth-rate (as in your new snippet)
    # ------------------------------
    # decay for the gate under consideration (keeps original behavior)
    max_decay = max(decay_parameter[gate[0]], decay_parameter[gate[1]])

    # depth_rate: normalize depth of the qubits in this gate w.r.t. max depth
    if depths:
        max_depth_val = max(depths.values())
        if max_depth_val > 0:
            depth_rate = max(depths.get(gate[0], 0), depths.get(gate[1], 0)) / max_depth_val
        else:
            depth_rate = 0.0
    else:
        depth_rate = 0.0

    # Weights for distance and depth tie-breaker (tunable constants)
    W_distance = 1.0
    tie_breaker_weight = 0.01

    # ------------------------------
    # 4) Main loop: layered, dependency-weighted distances
    # ------------------------------
    total_distance_score = 0.0
    total_layer_weight = 0.0

    for i in range(max_layers):
        layer = layers[i]
        layer_size = len(layer)
        layer_weight = 1.0 / (i + 1)  # earlier layers matter more
        total_layer_weight += layer_weight

        if layer_size == 0:
            # contribute nothing for empty layers
            continue

        weighted_distances = []
        per_gate_weights = []

        for g in layer:
            if g not in access:
                # e.g. control flow or something with no qubit access
                continue

            q1, q2 = access[g]
            P1, P2 = mapping[q1], mapping[q2]
            deps = deps_count[g] if with_closure_depth else 0
            gate_weight = deps + 1

            d = distance_matrix[P1][P2]
            err = _pair_error(P1, P2)
            hot = 1.0 + lambda_hot * (usage01[P1] + usage01[P2]) * 0.5
            d *= (1.0 + beta_error * err) * hot

            weighted_distances.append(gate_weight * d)
            per_gate_weights.append(gate_weight)

        # normalized layer distance: average criticality-weighted distance (safe division)
        denom = sum(per_gate_weights)
        if denom > 0:
            layer_distance = sum(weighted_distances) / denom
        else:
            layer_distance = 0.0

        # accumulate with layer weight
        total_distance_score += layer_weight * layer_distance

    # normalize by total layer weight (so heuristic is independent of lookahead length)
    if total_layer_weight > 0:
        total_distance_score /= total_layer_weight

    # ------------------------------
    # 5) Final heuristic
    # ------------------------------
    # combine distance and depth-based tie-breaker, scaled by max_decay
    H = max_decay * (W_distance * total_distance_score + tie_breaker_weight * depth_rate)
    return H



def find_min_score_swap_gate(heuristic_score, epsilon=1e-4, seed=42):
    random.seed(seed)
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




# ── helpers ───────────────────────────────────────────────────────────────────
 
def _scale01(counter):
    """Min-max normalise Counter values to [0, 1]. Returns defaultdict(float)."""
    if not counter:
        return defaultdict(float)
    lo = min(counter.values())
    hi = max(counter.values())
    if hi == lo:
        return defaultdict(float)
    return defaultdict(float, {k: (v - lo) / (hi - lo) for k, v in counter.items()})
 
 
def _pair_error(qubit_props, p0, p1):
    """
    Return symmetric two-qubit gate error for physical pair (p0, p1).
    Falls back to 0.0 if qubit_props is None or key is missing.
 
    Expected qubit_props shape:
        {
          "0": {"two_qubit_err": {"1": 0.012, "2": 0.034}},
          "1": {"two_qubit_err": {"0": 0.012}},
          ...
        }
    """
    if qubit_props is None:
        return 0.0
    try:
        e01 = qubit_props[str(p0)]["two_qubit_err"].get(str(p1))
    except (KeyError, AttributeError, TypeError):
        e01 = None
    try:
        e10 = qubit_props[str(p1)]["two_qubit_err"].get(str(p0))
    except (KeyError, AttributeError, TypeError):
        e10 = None
    if e01 is not None and e10 is not None:
        return 0.5 * (e01 + e10)
    return e01 if e01 is not None else (e10 if e10 is not None else 0.0)
 
 
# ── main heuristic ────────────────────────────────────────────────────────────
 
def dynamiq_poly_heuristic(
    front_layer,
    extended_layer,
    mapping,
    distance_matrix,
    access,               # = self.node_data  {gate_id -> node_dict}
    decay_parameter,
    deps_count,
    extended_layer_index,
    gate,                 # candidate SWAP: (physical_q1, physical_q2)
    alpha=1.0,
    W_distance=1.0,
    tie_breaker_weight=0.2,
    c_g=None,
    depths=None,
    ablation_variant="default",
    # ── new noise-aware parameters (all default to neutral / off) ──────
    qubit_props=None,     # calibration dict for two-qubit errors
    lambda_hot=0.5,       # hotspot-penalty weight  (set 0 to disable)
    beta_error=1.0,       # error-penalty weight    (set 0 to disable)
):
    """
    Noise-aware, hotspot-penalising SWAP heuristic.
 
    Returns a scalar float H >= 0.
    The router selects the SWAP candidate with *minimum* H.
 
    access[g] must be a node dict with the key:
        'coupled_qubits_accessed': [(logical_q1, logical_q2), ...]
    e.g.
        {'node_id': 1, 'qubits_accessed': [31, 38],
         'coupled_qubits_accessed': [(31, 38)],
         'operation_type': 'cz', 'is_control_flow': False}
    """
 
    # ── 0. build layer buckets ────────────────────────────────────────
    layers = defaultdict(list)
    for g in front_layer:
        layers[0].append(g)
    for g in extended_layer:
        layers[extended_layer_index.get(g, 1)].append(g)
 
    if not layers:
        return 0.0
 
    if c_g is None:
        c_g = {}
 
    # ── 1. hotspot usage density ──────────────────────────────────────
    usage = Counter()
    if lambda_hot > 0.0:
        for g in (*front_layer, *extended_layer):
            for q1, q2 in access[g]['coupled_qubits_accessed']:
                usage[mapping[q1]] += 1
                usage[mapping[q2]] += 1
    usage01 = _scale01(usage)
 
    # ── 2. decay / depth for the candidate SWAP ───────────────────────
    max_decay = max(decay_parameter[gate[0]], decay_parameter[gate[1]])
 
    if depths:
        max_d = max(depths.values())
        depth_rate = (
            max(depths.get(gate[0], 0), depths.get(gate[1], 0)) / max_d
            if max_d > 0 else 0.0
        )
    else:
        depth_rate = 0.0
 
    # ── 3. ablation-variant overrides (local copies, no mutation) ─────
    _alpha      = alpha
    _tie_w      = tie_breaker_weight
    _depth_rate = depth_rate
    _lambda_hot = lambda_hot
    _beta_error = beta_error
 
    if ablation_variant == "no_depth":
        _depth_rate = 0.0
    elif ablation_variant in ("no_layer_weight", "uniform_layer"):
        _alpha = 0.0
    elif ablation_variant == "aggressive_decay":
        _alpha = 2.0
    elif ablation_variant == "depth_priority":
        _tie_w = 1.0
    elif ablation_variant == "distance_only":
        _depth_rate = 0.0
        _lambda_hot = 0.0
        _beta_error = 0.0
 
    def gate_crit(g):
        return max(0.01, c_g.get(g, 1.0))
 
    def deps_w(g):
        return deps_count[g] + 1
 
    if ablation_variant == "no_crit_deps":
        def gate_crit(g): return 1.0   # noqa: E306
        def deps_w(g):    return 1      # noqa: E306
 
    # ── 4. layer-weighted distance accumulation ───────────────────────
    total_score = 0.0
    total_lw    = 0.0
 
    for i in range(len(layers)):
        layer = layers[i]
        if not layer:
            continue
 
        lw       = 1.0 / ((i + 1) ** _alpha)
        total_lw += lw
 
        w_dists = []
        w_gates = []
 
        for g in layer:
            pairs = access[g]['coupled_qubits_accessed']
            if not pairs:
                continue
 
            # average effective distance over all qubit pairs in this gate
            pair_d = []
            for q1, q2 in pairs:
                P1 = mapping[q1]
                P2 = mapping[q2]
                d  = float(distance_matrix[P1][P2])
 
                hot = (
                    1.0 + _lambda_hot * (usage01[P1] + usage01[P2]) * 0.5
                    if _lambda_hot > 0.0 else 1.0
                )
                err  = _pair_error(qubit_props, P1, P2)
                pair_d.append(d * hot * (1.0 + _beta_error * err))
 
            eff_d = sum(pair_d) / len(pair_d)   # explicit float, not generator
            gw    = gate_crit(g) * deps_w(g)
            w_dists.append(gw * eff_d)
            w_gates.append(gw)
 
        denom = sum(w_gates)
        layer_dist = sum(w_dists) / denom if denom > 0.0 else 0.0
        total_score += lw * layer_dist
 
    if total_lw > 0.0:
        total_score /= total_lw
 
    # ── 5. final value ────────────────────────────────────────────────
    H = float(max_decay * (W_distance * total_score + _tie_w * _depth_rate))
    return H

