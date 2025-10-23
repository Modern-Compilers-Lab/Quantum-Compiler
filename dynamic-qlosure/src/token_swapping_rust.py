from __future__ import annotations
from typing import Iterable, Tuple, Union, Dict, Hashable, List
import numpy as np
import rustworkx as rx

Phy = Union[int, str]
Log = Hashable
Edge = Tuple[Phy, Phy]


def convert_mapping(d):
    return {q_physical: q_logical for q_logical, q_physical in enumerate(d)}


def solve_token_swapping(
    edges_in: Iterable[Edge],
    init_map: Dict[Phy, Log],
    target_map: Dict[Phy, Log],
    trials: int = 4,
    seed: int | np.random.Generator | None = None,
    parallel_threshold: int = 50,
) -> List[Tuple[Phy, Phy]]:
    """
    Compute swaps (on physical nodes) that move each logical token from init_map to target_map.

    Returns:
        List of swaps as (physical_u, physical_v) in your original physical labels.
    """

    init_map = convert_mapping(init_map)
    target_map = convert_mapping(target_map)

    # ---- 1) Collect all physical nodes we might reference
    phys_nodes: set[Phy] = set()
    for u, v in edges_in:
        phys_nodes.add(u)
        phys_nodes.add(v)
    phys_nodes.update(init_map.keys())
    phys_nodes.update(target_map.keys())

    # Stable ordering for index assignment
    labels: List[Phy] = list(phys_nodes)
    label_to_idx = {lab: i for i, lab in enumerate(labels)}
    idx_to_label = {i: lab for lab, i in label_to_idx.items()}

    # ---- 2) Build the undirected graph
    g = rx.PyGraph(multigraph=False)
    # node index = position in this list; label stored as weight
    g.add_nodes_from(labels)
    g.add_edges_from([(label_to_idx[u], label_to_idx[v], None)
                     for (u, v) in edges_in])

    # ---- 3) Build the partial mapping: current physical -> target physical (as indices)
    # For each logical in BOTH maps, find its current and desired physical location.
    # init_map: {phys: log}  target_map: {phys: log}
    log_to_phys_init: Dict[Log, Phy] = {}
    for p, l in init_map.items():
        log_to_phys_init[l] = p
    log_to_phys_tgt: Dict[Log, Phy] = {}
    for p, l in target_map.items():
        log_to_phys_tgt[l] = p

    overlap_logs = set(log_to_phys_init).intersection(log_to_phys_tgt)

    mapping_indices: Dict[int, int] = {}
    for l in overlap_logs:
        p_src = log_to_phys_init[l]
        p_dst = log_to_phys_tgt[l]
        # skip if already in place
        if p_src == p_dst:
            continue
        if p_src not in label_to_idx or p_dst not in label_to_idx:
            raise ValueError(
                f"Physical node for logical {l} not present in the graph nodes.")
        mapping_indices[label_to_idx[p_src]] = label_to_idx[p_dst]

    # If nothing to move, return empty
    if not mapping_indices:
        return 0, 0, []

    # ---- 4) Seed handling (rx expects an int seed)
    if isinstance(seed, np.random.Generator):
        seed_int = int(seed.integers(1, 2**31 - 1))
    else:
        rng = np.random.default_rng(seed)
        seed_int = int(rng.integers(1, 2**31 - 1))
    # ---- 5) Run token swapper
    swaps_idx: List[Tuple[int, int]] = rx.graph_token_swapper(
        g, mapping_indices, trials, seed_int, parallel_threshold
    )

    # ---- 6) Translate swaps back to your physical labels
    swaps_labels = [(idx_to_label[u], idx_to_label[v]) for (u, v) in swaps_idx]
    return None, None, swaps_labels


# ---------- Example ----------
if __name__ == "__main__":
    # Physical connectivity (e.g., a simple line 0-1-2-3)
    edges = [(0, 1), (1, 2), (2, 3)]

    # init_map/target_map are {physical: logical}
    init_map = {0: "a", 1: "b", 2: "c", 3: "d"}
    target_map = {0: "b", 1: "a", 2: "d", 3: "c"}

    swaps = solve_token_swapping(
        edges, init_map, target_map, trials=8, seed=1234)
    print("Swaps:", swaps)
    # Example output (may vary with seed): Swaps: [(0, 1), (2, 3)]
