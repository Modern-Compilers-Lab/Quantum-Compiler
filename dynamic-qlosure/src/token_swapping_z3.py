from z3 import *
import networkx as nx
# ------------------------------------------------------------

# logical qubit lq on physical qubit pq at depth d


def loc_name(lq, pq, d): return f"loc_l{lq}_p{pq}_d{d}"

# swap physical qubits pu and pv at depth d


def swap_name(pu, pv, d): return f"swap_{pu}_{pv}_d{d}"

# exactly one of the Boolean variables in `bs` is True


def exactly_one(bs): return PbEq([(b, 1) for b in bs], 1)


def solve_token_swapping(coupling_edges,
                         initial_mapping,          # {logical → physical}
                         target_mapping,           # {logical → physical}
                         verbose=False):

    # check if initial == target
    if initial_mapping == target_mapping:
        return [], []

    coupling_edges, initial_mapping, target_mapping, longest_path_length = relax_problem(
        coupling_edges, initial_mapping, target_mapping)

    print(initial_mapping)
    print(target_mapping)
    print(f"Relaxed graph contains nodes {len(initial_mapping)}")
    print(coupling_edges)

    phys_qubits = sorted({q for e in coupling_edges for q in e})
    logical_qubits = sorted(set(initial_mapping.keys()))
    n_phys = len(phys_qubits)

    depth_lower_bound = longest_path_length
    # make edges (u,v) with u < v
    E = [(min(u, v), max(u, v)) for u, v in coupling_edges]

    # for each depth we try to generate a solution
    for depth in range(depth_lower_bound, n_phys*n_phys):
        if verbose:
            print(f"--- Trying depth = {depth} ---")

        s = Optimize()

        # Bool vars --------------------------------------------------
        Loc = {(lq, pq, d): Bool(loc_name(lq, pq, d))
               for lq in logical_qubits
               for pq in phys_qubits
               for d in range(depth + 1)}

        Swap = {(pu, pv, d): Bool(swap_name(pu, pv, d))
                for (pu, pv) in E
                for d in range(depth)}

        # (1) initial placement -------------------------------------
        for lq, pq in initial_mapping.items():
            for p in phys_qubits:
                s.add(Loc[(lq, p, 0)] == (p == pq))

        # (2) target placement --------------------------------------
        for lq, pq in target_mapping.items():
            for p in phys_qubits:
                s.add(Loc[(lq, p, depth)] == (p == pq))

        # (3) exclusivity constraints -------------------------------
        for d in range(depth + 1):
            # each logical is on exactly one physical
            for lq in logical_qubits:
                s.add(exactly_one([Loc[(lq, p, d)] for p in phys_qubits]))
            # each physical hosts exactly one logical
            for pq in phys_qubits:
                s.add(exactly_one([Loc[(lq, pq, d)] for lq in logical_qubits]))

        # (4) transition after applying SWAPs ---------------------------------
        for d in range(depth):
            # 4a) swap qubits after applying a swap
            for (pu, pv) in E:
                sw = Swap[(pu, pv, d)]
                for lq in logical_qubits:
                    s.add(Implies(sw, Loc[(lq, pu, d+1)] == Loc[(lq, pv, d)]))
                    s.add(Implies(sw, Loc[(lq, pv, d+1)] == Loc[(lq, pu, d)]))

            # 4b) qubits don't change position if we don't apply a swap that touches them
            for lq in logical_qubits:
                for pq in phys_qubits:
                    none = And(*[Not(Swap[(u, v, d)])
                                 for (u, v) in E if pq in (u, v)])
                    s.add(
                        Implies(none, Loc[(lq, pq, d+1)] == Loc[(lq, pq, d)]))

            # 4c) no overlapping swaps in the same depth
            for pq in phys_qubits:
                incident = [Swap[(u, v, d)] for (u, v) in E if pq in (u, v)]
                s.add(AtMost(*incident, 1))

        total = Int('total_swaps')
        s.add(total == Sum([If(sw, 1, 0) for sw in Swap.values()]))
        s.minimize(total)

        if s.check() == sat:
            min_swaps = s.model()[total].as_long()
            # if verbose:
            #     print(f"depth {depth}: optimum = {min_swaps} swaps")

            # ========== 2)  enumerate *all* optimal schedules ==========
            s2 = Solver()
            s2.add(s.assertions())     # reuse every constraint
            s2.add(total == min_swaps)   # …plus the optimum requirement
            print("HERE")
            # (5) SAT check ---------------------------------------------
            flattened_swaps = []
            if s2.check() == sat:
                m = s2.model()
                schedule = []
                for d in range(depth):
                    layer = [(u, v) for (u, v) in E
                             if m.evaluate(Swap[(u, v, d)], model_completion=True)]
                    if layer:
                        schedule.append(layer)
                        flattened_swaps.extend(layer)
                return schedule, flattened_swaps

    return None                   # no route within max_depth


def relax_problem(coupling_map, initial_mapping, target_mapping):
    H = nx.Graph()

    H.add_edges_from(coupling_map)
    # make undirected

    H = H.to_undirected()

    # Handle case where mapping is a list or dictionary
    if isinstance(initial_mapping, list):
        physical_to_logical_initial = {
            pq: lq for lq, pq in enumerate(initial_mapping)}
    else:
        physical_to_logical_initial = {
            v: k for k, v in initial_mapping.items()}

    if isinstance(target_mapping, list):
        physical_to_logical_target = {
            pq: lq for lq, pq in enumerate(target_mapping)}
    else:
        physical_to_logical_target = {v: k for k, v in target_mapping.items()}

    nx.set_node_attributes(H, physical_to_logical_initial, "L1")
    nx.set_node_attributes(H, physical_to_logical_target, "L2")

    moved = [v for v in H.nodes if H.nodes[v]["L1"] != H.nodes[v]["L2"]]

    steiner = nx.algorithms.approximation.steiner_tree(H, moved)

    updated_initial_mapping = {}
    updated_target_mapping = {}

    for v in steiner.nodes:
        updated_initial_mapping[v] = physical_to_logical_initial[v]
        updated_target_mapping[v] = physical_to_logical_target[v]

    reverse_updated_initial_mapping = {
        v: k for k, v in updated_initial_mapping.items()}
    reverse_updated_target_mapping = {
        v: k for k, v in updated_target_mapping.items()}

    # Find the longest path (diameter) in the steiner tree
    if len(steiner.nodes) > 1:
        # Calculate all shortest paths between all pairs of nodes
        all_pairs_shortest = dict(nx.all_pairs_shortest_path_length(steiner))

        # Find the maximum shortest path length (diameter)
        longest_path_length = max(
            length for node_distances in all_pairs_shortest.values()
            for length in node_distances.values()
        )
    else:
        longest_path_length = 0

    return steiner.edges, reverse_updated_initial_mapping, reverse_updated_target_mapping, longest_path_length


def verify_schedule(coupling_edges,
                    init_map_log2phys,
                    goal_map_log2phys,
                    schedule_layers,                 # list[list[(u,v)]]
                    verbose=False):
    """
    Return True iff  applying `schedule_layers` starting from `init_map`
    reaches exactly `goal_map` and every layer is legal.

    Layers: schedule_layers[depth]  == list of edges swapped *in parallel*
    """

    # ---------- canonical data -------------------------------------
    Eset = {tuple(sorted(e)) for e in coupling_edges}

    #  Convert to phys→logical for easier mutation
    phys2log = {pq: None for pq in {q for e in coupling_edges for q in e}}
    for lq, pq in init_map_log2phys.items():
        phys2log[pq] = lq

    # ---------- simulate layer by layer ----------------------------
    for depth, layer in enumerate(schedule_layers):
        if verbose:
            print(f"Executing layer {depth}: {layer}")

        # a) legality checks
        seen = set()
        for u, v in layer:
            edge = tuple(sorted((u, v)))
            if edge not in Eset:
                if verbose:
                    print(f"Illegal edge {edge} not in coupling map")
                return False
            if u in seen or v in seen:
                if verbose:
                    print("Vertex-disjointness violated in this layer")
                return False
            seen.update((u, v))

        # b) perform swaps
        for u, v in layer:
            phys2log[u], phys2log[v] = phys2log[v], phys2log[u]

    # ---------- compare with goal mapping --------------------------
    for lq, pq_goal in goal_map_log2phys.items():
        if phys2log[pq_goal] != lq:
            if verbose:
                print("Final placement mismatch: "
                      f"logical {lq} expected on phys {pq_goal}, "
                      f"but found {phys2log[pq_goal]}")
            return False

    # also ensure no stray logical sits on a physical qubit that goal
    # says should be empty (covers idle logicals if any)
    for pq, lq in phys2log.items():
        if lq is not None and goal_map_log2phys.get(lq, pq) != pq:
            if verbose:
                print(f"Logical {lq} ended on phys {pq}, "
                      "which is not its goal location.")
            return False

    return True
