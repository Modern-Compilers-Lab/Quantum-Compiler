from collections import deque, defaultdict
from typing import Dict, List, Tuple, Iterable, Union, Set
import json
import itertools
import os
import csv
import time
import argparse

Node = int
Edge = Tuple[Node, Node]
MapDict = Dict[Node, Node]


def total_distance(current, goal_pos, dist):
    return sum(dist[p][goal_pos[tok]] for p, tok in current.items())


def to_int(x: Union[int, str]) -> int:
    if isinstance(x, int):
        return x
    val = int(str(x).strip())
    return val


def convert_mapping(d) -> MapDict:
    return {q_physical: q_logical for q_logical, q_physical in enumerate(d)}


def normalize_edges(edges: Iterable[Tuple[Union[int, str], Union[int, str]]]) -> List[Edge]:
    out = []
    for u, v in edges:
        u = to_int(u)
        v = to_int(v)
        if u == v:
            continue
        out.append((min(u, v), max(u, v)))
    return sorted(set(out))


def build_adj(nodes: List[Node], edges: List[Edge]) -> Dict[Node, List[Node]]:
    adj = defaultdict(list)
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    for k in nodes:
        adj[k] = sorted(set(adj[k]))
    return adj


def solve_token_swapping(
    edges_in: Iterable[Tuple[Union[int, str], Union[int, str]]],
    init_map_in: Dict[Union[int, str], Union[int, str]],
    target_map_in: Dict[Union[int, str], Union[int, str]],
    max_layers: int = 1000,
    COOL: int = 3,
    stuck: int = 2,
    fallback2_version: bool = True
) -> List[List[Edge]]:
    """
    Returns a list of swap layers L, where each layer is a set of edges
    that can be executed in parallel (i.e., no two edges share a common node).
    Procedure for each round:
      1) Select a maximal set of parallelizable edges with positive gain 
         (i.e., swaps that reduce the distance to the targets).
      2) If no such edges exist, select an edge connected to the farthest node 
         to ensure progress toward the target mapping.
      3) If no such edges exist, select an edge connected to the farthest node 
         with the lowest cooldown value.
    """
    init = convert_mapping(init_map_in)  # physical : logical
    target = convert_mapping(target_map_in)  # physical : logical
    edges: List[Edge] = normalize_edges(edges_in)
    nodes = sorted(init.keys())

    if set(nodes) != set(target.keys()):
        raise ValueError("init and goal must have same set of physical ids")
    if sorted(init.values()) != sorted(target.values()):
        raise ValueError("init and goal must contain the same logical tokens")

    adj = build_adj(nodes, edges)  # build dict of list of adjacent nodes

    # All Path Shortest Path
    INF = 10**9
    dist = {s: {t: INF for t in nodes} for s in nodes}
    for s in nodes:
        dist[s][s] = 0
        q = deque([s])
        while q:
            u = q.popleft()
            for v in adj[u]:
                if dist[s][v] == INF:
                    dist[s][v] = dist[s][u] + 1
                    q.append(v)

    goal_pos = {v: k for k, v in target.items()}  # logical : physical

    for tok in sorted(init.values()):
        s = next(p for p, t in init.items() if t == tok)
        t = goal_pos[tok]
        if dist[s][t] >= INF:
            raise ValueError(
                f"Token {tok} cannot reach {t} (disconnected graph).")

    current = init.copy()  # physical : logical
    layers: List[List[Edge]] = []
    layer_count = 0
    edge_cooldown: Dict[Edge, int] = {}  # edge -> remaining blocked iterations
    PATIENCE = 8                      # iterations allowed with no improvement
    best_cost = total_distance(init, goal_pos, dist)
    stagnant = 0
    last_layer: Tuple[Edge, ...] | None = None

    while current != target and layer_count < max_layers:
        layer_count += 1

        edge_gains = []

        for (u, v) in edges:  # calculate every edge how much it can make closer after applying swap
            a, b = current[u], current[v]  # what's in current physical qubit
            # distance before applying swap
            before = dist[u][goal_pos[a]] + dist[v][goal_pos[b]]
            # distance after applying swap
            after = dist[v][goal_pos[a]] + dist[u][goal_pos[b]]
            gain = before - after
            edge_gains.append(((u, v), gain))

        # max gain first, lexicographic tie-break
        edge_gains.sort(key=lambda kv: (-kv[1], kv[0][0], kv[0][1]))

        used: Set[Node] = set()
        layer: List[Edge] = []

        for (u, v), g in edge_gains:
            if g <= 0:
                continue
            if u in used or v in used:
                continue
            e = (u, v)
            if edge_cooldown.get(e, 0) > 0:
                continue
            layer.append(e)
            used.update([u, v])

        ######### First Fallback #########################################################
        if not layer:
            MAX_D = -1  # Finding max distance
            for p in nodes:
                if MAX_D < dist[p][goal_pos[current[p]]]:
                    MAX_D = dist[p][goal_pos[current[p]]]

            candidates = [
                (p, current[p], dist[p][goal_pos[current[p]]])
                for p in nodes
                if current[p] != target[p] and dist[p][goal_pos[current[p]]] >= MAX_D
            ]

            # farthest-first; tie-break by smaller physical id
            candidates.sort(key=lambda x: (-x[2], x[0]))

            used2: Set[Node] = set()
            layer = []  # collect multiple parallel swaps

            for p, tok, d in candidates:
                if d <= 0 or p in used2:
                    continue

                t = goal_pos[tok]
                closer = [nb for nb in adj[p]
                          if nb not in used2 and dist[nb][t] == d - 1]
                picked = None

                for nb in sorted(closer):
                    e = (min(p, nb), max(p, nb))
                    if edge_cooldown.get(e, 0) > 0:
                        continue
                    picked = e
                    break

                if picked is not None:
                    layer.append(picked)
                    used2.update(picked)

            ######### Second Fallback #########################################################
            if not layer:
                ok_edge = None
                last_edge = None

                if fallback2_version:  # First version of second fallback
                    p, tok, d = candidates[0]  # farthest (by sorting above)
                    t = goal_pos[tok]
                    closer = [nb for nb in adj[p] if dist[nb][t] == d - 1]

                    closer.sort(key=lambda nb: (
                        edge_cooldown.get((min(p, nb), max(p, nb)), 0),
                        min(p, nb), max(p, nb)
                    ))

                    nb = closer[0]
                    ok_edge = (min(p, nb), max(p, nb))

                else:  # Second version of second fallback
                    candidates = [
                        (p, current[p], dist[p][goal_pos[current[p]]])
                        for p in nodes
                        if current[p] != target[p]
                    ]

                    candidates.sort(key=lambda x: (x[2], x[0]))

                    for p, tok, d in candidates:
                        t = goal_pos[tok]

                        closer = [nb for nb in adj[p] if nb not in used2]
                        closer.sort(key=lambda nb: (
                            edge_cooldown.get((min(p, nb), max(p, nb)), 0),
                            min(p, nb), max(p, nb)
                        ))

                        nb = closer[0]

                        e = (min(p, nb), max(p, nb))
                        if edge_cooldown.get(e, 0) == 0:
                            ok_edge = e
                        else:
                            last_edge = e  # Farthest node's edge that has smallest cooldown value

                    if ok_edge == None:
                        ok_edge = last_edge

                layer = [ok_edge]

        layer.sort()
        new_current = current.copy()
        for u, v in layer:  # Apply swaps
            new_current[u], new_current[v] = new_current[v], new_current[u]
        current = new_current
        layers.append(layer)

        for e in list(edge_cooldown):
            if edge_cooldown[e] > 0:
                edge_cooldown[e] -= 1
        for e in layer:
            edge_cooldown[e] = COOL

        # stall detection (no cost improvement or identical layer)
        cur_cost = total_distance(current, goal_pos, dist)
        same_as_last = (tuple(layer) == last_layer)
        last_layer = tuple(layer)

        if cur_cost < best_cost:
            best_cost = cur_cost
            stagnant = 0
        else:
            stagnant += 1

        # force diversification if we’re stuck
        if stagnant >= PATIENCE or same_as_last:
            for e in layer:
                # extend cooldown for this whole layer to force different choices next round
                edge_cooldown[e] = max(edge_cooldown.get(e, 0), COOL + stuck)
            stagnant = 0
    swaps = []
    for layer in layers:
        swaps.extend(layer)

    if current != target:
        # print("Exceeded max layers without reaching goal.")
        return False, layers, swaps
        # raise RuntimeError("Exceeded max layers without reaching goal.")
    return True, layers, swaps
