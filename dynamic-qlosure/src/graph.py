from collections import defaultdict, deque
from typing import DefaultDict, Set, List, Tuple, TypeVar


def build_backend_graph(edges: List[Tuple[int, int]]):
    graph = defaultdict(set)
    for node1, node2 in edges:
        graph[node1].add(node2)
        graph[node2].add(node1)

    return graph


def compute_distance_matrix(graph: DefaultDict[int, Set[int]]):

    nodes = sorted(graph.keys())
    n = nodes[-1] + 1

    dist_matrix = [[float('inf')] * n for _ in range(n)]

    # For each node, run a BFS to compute distances to all other nodes.
    for start_node in nodes:
        dist_matrix[start_node][start_node] = 0
        queue = deque([start_node])

        while queue:
            current = queue.popleft()
            current_idx = current
            current_dist = dist_matrix[start_node][current_idx]

            for neighbor in graph[current]:
                neighbor_idx = neighbor
                # If we haven't visited this neighbor yet, update distance and queue it.
                if dist_matrix[start_node][neighbor_idx] == float('inf'):
                    dist_matrix[start_node][neighbor_idx] = current_dist + 1
                    queue.append(neighbor)

    return dist_matrix


# generate swap candidates based on the active qubits

def generate_swap_candidates(active_qubits, backend):
    candidates = []
    # for qubit, neighbors in backend.items():
    #    for neighbor in neighbors:
    #        candidates.append((qubit, neighbor))

    # return candidates
    for qubit in active_qubits:
        for neighbor in backend[qubit]:
            candidates.append((qubit, neighbor))

    return candidates


def compute_dependencies_length_old(graph):
    memo = {}

    def dfs(node):
        if node in memo:
            return memo[node]
        closure = set()
        for neighbor in graph.get(node, []):
            closure.add(neighbor)
            closure |= dfs(neighbor)
        memo[node] = closure
        return closure

    dependencies_length = {}
    for node in graph:
        dependencies_length[node] = len(dfs(node))
    return dependencies_length


def compute_dependencies_length(graph, predecessors,):
    out_degree = {}
    for node in graph:
        out_degree[node] = len(graph.get(node, []))

    queue = deque(node for node, deg in out_degree.items() if deg == 0)

    transitive_dependents = {node: set() for node in graph}

    while queue:
        x = queue.popleft()

        for p in predecessors.get(x, []):
            transitive_dependents[p].update(transitive_dependents[x])
            transitive_dependents[p].add(x)

            out_degree[p] -= 1
            if out_degree[p] == 0:
                queue.append(p)
    dependents_length = defaultdict(int)
    for node in graph:
        dependents_length[node] = len(transitive_dependents[node])

    return dependents_length


def compute_transitive_closure_bitset(graph, predecessors, max_reach=1000):

    all_nodes = list([node for node in graph.keys()])
    if not all_nodes:
        return {}
    index_of = {}
    for i, node in enumerate(all_nodes):
        index_of[node] = i

    n = len(all_nodes)

    out_degree = {}
    for node in all_nodes:
        out_degree[node] = len(graph.get(node, []))
    queue = deque([node for node in all_nodes if out_degree[node] == 0])

    bit_reach = [0] * n

    current_reach = 0
    while queue:
        for _ in range(len(queue)):
            x = queue.popleft()
            x_i = index_of[x]

            for p in predecessors.get(x, []):

                p_i = index_of[p]

                before = bit_reach[p_i]
                bit_reach[p_i] |= bit_reach[x_i]
                bit_reach[p_i] |= (1 << x_i)

                if bit_reach[p_i] != before:
                    out_degree[p] -= 1
                    if out_degree[p] == 0:
                        queue.append(p)
                else:
                    out_degree[p] -= 1
                    if out_degree[p] == 0:
                        queue.append(p)
        current_reach += 1

    dependents_length = [0]*(max(all_nodes)+1)
    for node in all_nodes:
        i = index_of[node]

        dependents_length[node] = bit_reach[i].bit_count()

    return dependents_length


def _flatten_loops_in_dag(graph, predecessors, node_data, default_while_iterations=10):
    """
    Walk the DAG and unroll every for_loop / while_loop control-flow node.

    For each loop node found:
      - The loop body sub-DAG (from node_data[node]['control_flow_blocks'][0])
        is replicated `iterations` times.
      - Edges are rewired so that:
            predecessors(loop_node) → entry nodes of iteration-0 body
            exit nodes of iteration-i → entry nodes of iteration-(i+1) body
            exit nodes of last iteration → successors(loop_node) in outer DAG
      - The original loop node is removed.

    Returns (flat_graph, flat_predecessors, first_iter_node_ids)
      where first_iter_node_ids is the set of new node IDs corresponding to
      the first iteration's body (useful for interpreting results).

    NOTE: This mutates nothing — all structures are freshly built.
    """
    import copy as _copy

    # Deep-copy so we don't mutate the caller's data
    flat_graph = _copy.deepcopy(dict(graph))
    flat_preds = _copy.deepcopy(dict(predecessors))
    flat_node_data = _copy.deepcopy(dict(node_data)) if node_data else {}

    # Collect loop nodes (iterate over a snapshot of keys)
    loop_nodes = [
        n for n in list(flat_graph.keys())
        if flat_node_data.get(n, {}).get('is_control_flow', False)
        and flat_node_data.get(n, {}).get('operation_type', '') in ('for_loop', 'while_loop')
    ]

    # We need a fresh ID generator that won't collide with existing node IDs
    if flat_graph:
        next_id = max(max(flat_graph.keys()), max(flat_preds.keys(), default=0)) + 1
    else:
        next_id = 0

    # Track which new node IDs belong to the first iteration
    first_iter_node_ids = set()

    for loop_node in loop_nodes:
        # --- Determine iteration count ---
        op_type = flat_node_data[loop_node].get('operation_type', '')
        if op_type == 'for_loop':
            # The iteration count may not be stored in node_data;
            # we fall back to default_while_iterations if not found
            iterations = flat_node_data[loop_node].get('iterations', default_while_iterations)
        else:  # while_loop
            iterations = default_while_iterations

        # Clamp iterations to at least 1
        iterations = max(1, iterations)

        # --- Extract the body sub-DAG ---
        blocks = flat_node_data[loop_node].get('control_flow_blocks')
        if not blocks or len(blocks) == 0:
            continue
        body_dag = blocks[0]  # first (and usually only) block for for/while

        body_nodes = sorted(body_dag['nodes'])
        body_successors = body_dag['successors']
        body_predecessors = body_dag['predecessors']

        if not body_nodes:
            # Empty body — just remove the loop node
            _remove_node(flat_graph, flat_preds, loop_node)
            continue

        # Identify entry nodes (no predecessors in the body) and exit nodes (no successors)
        body_entries = [n for n in body_nodes if len(body_predecessors.get(n, set())) == 0]
        body_exits = [n for n in body_nodes if len(body_successors.get(n, set())) == 0]

        # --- Collect outer connectivity of the loop node ---
        outer_preds_of_loop = set(flat_preds.get(loop_node, set()))
        outer_succs_of_loop = set(flat_graph.get(loop_node, set()))

        # --- Replicate the body `iterations` times ---
        # For each iteration i we create a mapping: original_body_node -> new_id
        iter_id_maps = []   # list of dicts {orig_body_id: new_id}

        for i in range(iterations):
            id_map = {}
            for orig_id in body_nodes:
                id_map[orig_id] = next_id
                # Register in flat graph
                flat_graph[next_id] = set()
                flat_preds[next_id] = set()
                # Copy node data (as plain gate, not control-flow)
                if orig_id in body_dag.get('node_data', {}):
                    flat_node_data[next_id] = _copy.deepcopy(body_dag['node_data'][orig_id])
                    flat_node_data[next_id]['node_id'] = next_id
                next_id += 1

                if i == 0:
                    first_iter_node_ids.add(id_map[orig_id])

            iter_id_maps.append(id_map)

        # --- Wire internal edges within each iteration ---
        for i in range(iterations):
            id_map = iter_id_maps[i]
            for orig_id in body_nodes:
                new_id = id_map[orig_id]
                for orig_succ in body_successors.get(orig_id, set()):
                    new_succ = id_map[orig_succ]
                    flat_graph[new_id].add(new_succ)
                    flat_preds[new_succ].add(new_id)

        # --- Wire iteration boundaries (iteration i exits → iteration i+1 entries) ---
        for i in range(iterations - 1):
            curr_map = iter_id_maps[i]
            next_map = iter_id_maps[i + 1]
            for exit_node in body_exits:
                for entry_node in body_entries:
                    new_exit = curr_map[exit_node]
                    new_entry = next_map[entry_node]
                    flat_graph[new_exit].add(new_entry)
                    flat_preds[new_entry].add(new_exit)

        # --- Wire outer predecessors → iteration-0 entries ---
        first_map = iter_id_maps[0]
        for p in outer_preds_of_loop:
            # Remove the old edge p → loop_node
            flat_graph[p].discard(loop_node)
            for entry_node in body_entries:
                new_entry = first_map[entry_node]
                flat_graph[p].add(new_entry)
                flat_preds[new_entry].add(p)

        # --- Wire last-iteration exits → outer successors ---
        last_map = iter_id_maps[-1]
        for s in outer_succs_of_loop:
            # Remove the old edge loop_node → s
            flat_preds[s].discard(loop_node)
            for exit_node in body_exits:
                new_exit = last_map[exit_node]
                flat_graph[new_exit].add(s)
                flat_preds[s].add(new_exit)

        # --- Remove the original loop node ---
        flat_graph.pop(loop_node, None)
        flat_preds.pop(loop_node, None)
        flat_node_data.pop(loop_node, None)

    return flat_graph, flat_preds, first_iter_node_ids, flat_node_data


def _remove_node(graph, predecessors, node):
    """Remove a node from graph/predecessors, rewiring pred→succ edges."""
    preds = set(predecessors.get(node, set()))
    succs = set(graph.get(node, set()))

    for p in preds:
        graph[p].discard(node)
        for s in succs:
            graph[p].add(s)
            predecessors[s].add(p)

    for s in succs:
        predecessors[s].discard(node)

    graph.pop(node, None)
    predecessors.pop(node, None)


def compute_transitive_closure_bitset_unrolled(
    graph, predecessors, node_data=None,
    max_reach=1000, default_while_iterations=10
):
    """
    Alternative transitive-closure computation that first **unrolls / flattens**
    every for_loop and while_loop control-flow node, then runs the standard
    bitset-based transitive closure on the expanded (flat) DAG.

    Parameters
    ----------
    graph : dict[int, set[int] | list[int]]
        Successor adjacency list of the 2-qubit DAG.
    predecessors : dict[int, set[int] | list[int]]
        Predecessor adjacency list of the 2-qubit DAG.
    node_data : dict[int, dict] | None
        The DAG's node_data mapping, needed to identify control-flow nodes and
        access their ``control_flow_blocks``.  If None, behaviour is identical
        to ``compute_transitive_closure_bitset``.
    max_reach : int
        Upper bound on the node ids used for the result array sizing.
    default_while_iterations : int
        Number of iterations assumed for while_loop nodes (which don't carry a
        static iteration count).  For for_loop nodes the count stored in
        ``node_data`` is used when available, otherwise this default.

    Returns
    -------
    list[int]
        ``dependents_length`` indexed by original node id.  For nodes that
        existed before flattening (including the body gates of the *first*
        unrolled iteration) the value is the popcount of its transitive-reach
        bitmask in the expanded graph.  Original loop-node ids are mapped to 0.
    """

    # --- If there is no node_data we cannot detect loops; fall back. ---
    if node_data is None:
        return compute_transitive_closure_bitset(graph, predecessors, max_reach)

    # --- Check whether there are any loops to flatten ---
    has_loops = any(
        node_data.get(n, {}).get('is_control_flow', False)
        and node_data.get(n, {}).get('operation_type', '') in ('for_loop', 'while_loop')
        for n in graph
    )
    if not has_loops:
        # Nothing to unroll — use the standard algorithm directly
        return compute_transitive_closure_bitset(graph, predecessors, max_reach)

    # --- Flatten ---
    flat_graph, flat_preds, first_iter_ids, flat_node_data = _flatten_loops_in_dag(
        graph, predecessors, node_data,
        default_while_iterations=default_while_iterations,
    )

    # --- Run the standard bitset closure on the flat graph ---
    flat_deps = compute_transitive_closure_bitset(flat_graph, flat_preds, max_reach=max_reach)

    # --- Build the result array keyed by *original* node ids ---
    # Original non-loop nodes keep their id and their computed value.
    # First-iteration body nodes are new ids; we include their values too.
    # Original loop-node ids get 0 (they no longer exist in the flat graph).
    all_original_ids = list(graph.keys())
    if all_original_ids:
        result_size = max(max(all_original_ids), max_reach, len(flat_deps))
    else:
        result_size = max_reach

    # Start from the flat result (which may be larger because of new ids).
    # Extend it if needed so the original ids are covered.
    if len(flat_deps) < result_size + 1:
        flat_deps.extend([0] * (result_size + 1 - len(flat_deps)))

    return flat_deps


# ---------------------------------------------------------------------------
# Unified entry point
# ---------------------------------------------------------------------------

def compute_transitive_closure(
    graph, predecessors, *,
    method="default",
    node_data=None,
    max_reach=1000,
    default_while_iterations=10,
):
    """
    Unified entry point for transitive-closure computation.

    Parameters
    ----------
    method : str
        ``"default"``  — uses the original ``compute_transitive_closure_bitset``.
        ``"unrolled"`` — first flattens for/while loops, then computes the
                         closure on the expanded graph.
    node_data : dict | None
        Required when *method="unrolled"*; ignored otherwise.
    default_while_iterations : int
        Iteration count assumed for while-loops when *method="unrolled"*.

    All other parameters are forwarded verbatim to the underlying function.
    """
    if method == "default":
        return compute_transitive_closure_bitset(graph, predecessors, max_reach)
    elif method == "unrolled":
        return compute_transitive_closure_bitset_unrolled(
            graph, predecessors,
            node_data=node_data,
            max_reach=max_reach,
            default_while_iterations=default_while_iterations,
        )
    else:
        raise ValueError(
            f"Unknown transitive-closure method '{method}'. "
            f"Supported: 'default', 'unrolled'."
        )
