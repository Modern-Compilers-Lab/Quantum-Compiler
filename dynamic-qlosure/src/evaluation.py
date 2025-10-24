from typing import Dict, List, Any, Tuple, Set
from copy import deepcopy
from collections import defaultdict

# ---- Public API -------------------------------------------------------------


def compute_max_swaps_count(trace: List[Dict[str, Any]], loop_iterations: int = 100) -> int:
    """
    Max swaps count:
      - Count 'swap' nodes.
      - For 'for' blocks, multiply the inner count by loop_iterations.
      - For 'if' blocks, take the max swaps among branches.
    """
    return _max_swaps_count_list(trace, loop_iterations)


def compute_structural_depth(trace: List[Dict[str, Any]], loop_iterations: int = 100) -> int:
    """
    Structural depth of the longest branch (counts both gates and swaps):
      - Each 'gate' and 'swap' adds 1.
      - For 'for' blocks, inner depth * loop_iterations.
      - For 'if' blocks, take max depth among branches.
    """
    return _structural_depth_list(trace, loop_iterations)


def compute_quantum_depth(
    trace: List[Dict[str, Any]],
    loop_iterations: int = 100,
    use_physical_qubits: bool = True
) -> int:
    """
    Quantum depth (minimum number of parallel layers) under qubit constraints:
      - 1-qubit ops occupy their qubit for the layer.
      - 2-qubit ops (incl. swaps) occupy both qubits for the layer.
      - For 'for' blocks, the body depth is multiplied by loop_iterations, and
        only the qubits touched by the body are delayed accordingly.
      - For 'if' blocks, we take the max depth branch (worst-case).
    Returns the total depth (an integer).
    """
    _, depth = _quantum_depth_list(
        trace,
        qubit_time=defaultdict(int),  # availability per qubit
        loop_iterations=loop_iterations,
        use_physical_qubits=use_physical_qubits
    )
    return depth


# ---- Internal helpers: Max swaps -------------------------------------------

def _max_swaps_count_list(items: List[Dict[str, Any]], loop_iterations: int) -> int:
    total = 0
    for it in items:
        t = it.get("type")
        if t not in ("swap", "gate", "for", "while", "if"):
            raise ValueError(f"Unknown item type: {t}")
            exit(1)
        if t == "swap":
            total += 1
        elif t == "gate":
            # gates don't count towards swap metric
            pass
        elif t == "for" or t == "while":
            body = it.get("body", [])
            total += loop_iterations * \
                _max_swaps_count_list(body, loop_iterations)
        elif t == "if":
            branches = it.get("branches", [])
            # Take max across branches
            branch_counts = [
                _max_swaps_count_list(b.get("body", []), loop_iterations) for b in branches
            ]
            total += max(branch_counts) if branch_counts else 0
        else:
            # Unknown node type; ignore
            pass
    return total


# ---- Internal helpers: Structural depth ------------------------------------

def _structural_depth_list(items: List[Dict[str, Any]], loop_iterations: int) -> int:
    """
    Depth of the longest branch counting both gates and swaps.
    """
    depth = 0
    for it in items:
        t = it.get("type")
        if t in ("gate", "swap"):
            depth += 1
        elif t == "for":
            body = it.get("body", [])
            inner = _structural_depth_list(body, loop_iterations)
            depth += loop_iterations * inner
        elif t == "if":
            branches = it.get("branches", [])
            branch_depths = [
                _structural_depth_list(b.get("body", []), loop_iterations) for b in branches
            ]
            depth += max(branch_depths) if branch_depths else 0
        else:
            # Unknown node type; ignore
            pass
    return depth


# ---- Internal helpers: Quantum depth ---------------------------------------

def _get_qubits_for_item(
    item: Dict[str, Any],
    use_physical_qubits: bool
) -> List[Tuple[int, ...]]:
    """
    For a single 'gate' or 'swap' item, return a list with one tuple of qubits it touches.
    For control-flow nodes, return empty; their qubits are determined recursively.
    """
    t = item.get("type")
    key = "physical_qubits" if use_physical_qubits else "logical_qubits"

    if t == "gate":
        qs = item.get(key, []) or item.get("logical_qubits", [])
        return [tuple(qs)]
    elif t == "swap":
        qs = item.get(key, []) or item.get("logical_qubits", [])
        return [tuple(qs)]
    # for/if handled at higher level
    return []


def _collect_qubits_used(
    items: List[Dict[str, Any]],
    use_physical_qubits: bool
) -> Set[int]:
    """
    Collect all qubits touched within a (possibly nested) list.
    Used to advance availability only for impacted qubits after 'for' blocks.
    """
    used: Set[int] = set()
    for it in items:
        t = it.get("type")
        if t in ("gate", "swap"):
            key = "physical_qubits" if use_physical_qubits else "logical_qubits"
            qs = it.get(key, []) or it.get("logical_qubits", [])
            used.update(qs)
        elif t == "for":
            used.update(_collect_qubits_used(
                it.get("body", []), use_physical_qubits))
        elif t == "if":
            for br in it.get("branches", []):
                used.update(_collect_qubits_used(
                    br.get("body", []), use_physical_qubits))
    return used


def _schedule_op(
    qubits: Tuple[int, ...],
    qubit_time: Dict[int, int]
) -> int:
    """
    ASAP schedule for a single op on given qubits.
    Returns the layer used by this op, and updates qubit_time.
    """
    start = 0
    for q in qubits:
        start = max(start, qubit_time[q])
    layer = start + 1
    for q in qubits:
        qubit_time[q] = layer
    return layer


def _quantum_depth_list(
    items: List[Dict[str, Any]],
    qubit_time: Dict[int, int],
    loop_iterations: int,
    use_physical_qubits: bool
) -> Tuple[Dict[int, int], int]:
    """
    Schedule a sequence of items (basic block) and return:
      - updated qubit_time
      - total depth (max layer) achieved so far
    """
    # Work on a mutable dict (caller passes one)
    max_layer = max(qubit_time.values(), default=0)

    for it in items:
        t = it.get("type")

        if t in ("gate", "swap"):
            ops = _get_qubits_for_item(it, use_physical_qubits)
            for qs in ops:
                layer = _schedule_op(qs, qubit_time)
                if layer > max_layer:
                    max_layer = layer

        elif t == "for" or t == "while":
            body = it.get("body", [])
            # Compute body depth starting from current state, on a copy
            body_qubit_time = deepcopy(qubit_time)
            body_qubit_time_after, body_depth = _quantum_depth_list(
                body, body_qubit_time, loop_iterations, use_physical_qubits
            )
            # Effective added depth = body_depth * iterations
            added = body_depth * loop_iterations
            # Only qubits touched by the body are delayed
            touched = _collect_qubits_used(body, use_physical_qubits)
            for q in touched:
                qubit_time[q] = qubit_time[q] + added
            # Update max layer
            max_layer = max(max_layer, max(qubit_time.values(), default=0))

        elif t == "if":
            branches = it.get("branches", [])

            # Evaluate each branch from the *same* incoming state, pick worst depth
            best_depth = max_layer
            best_state = None

            for br in branches:
                br_time = deepcopy(qubit_time)
                br_time_after, br_depth = _quantum_depth_list(
                    br.get("body", []), br_time, loop_iterations, use_physical_qubits
                )
                if br_depth > best_depth:
                    best_depth = br_depth
                    best_state = br_time_after

            if best_state is not None:
                qubit_time.clear()
                qubit_time.update(best_state)
                max_layer = best_depth
            # If no branches, nothing changes

        else:
            # Unknown type; ignore gracefully
            pass

    return qubit_time, max_layer
