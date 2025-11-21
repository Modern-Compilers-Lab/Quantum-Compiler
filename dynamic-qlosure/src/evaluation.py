from typing import Dict, List, Any, Tuple, Set,Iterable,Union
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



# ---- Internal helpers: Quantum depth ---------------------------------------

def _get_qubits_for_item(
    item: Dict[str, Any],
    use_physical_qubits: bool = True
) -> List[Tuple[int, ...]]:
    """
    For a single 'gate' or 'swap' item, return a list with one tuple of qubits it touches.
    For control-flow nodes, return empty; their qubits are determined recursively.
    """
    t = item.get("type")
    key = "physical_qubits" if use_physical_qubits else "logical_qubits"

    if t == "gate":
        qs = item.get(key, []) or item.get("physical_qubits", [])
        return [tuple(qs)]
    elif t == "swap":
        qs = item.get(key, []) or item.get("physical_qubits", [])
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
        elif t == "for" or t == "while":
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
    layer = max(qubit_time[q] for q in qubits) + 1
    for q in qubits:
        qubit_time[q] = layer
    return layer


def _quantum_depth_list(
    items: List[Dict[str, Any]],
    qubit_time: Dict[int, int],
    loop_iterations: int,
    use_physical_qubits: bool = True
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
                body, defaultdict(int), loop_iterations, use_physical_qubits
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
            best_depth = float('-inf')
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



### error 


# def estimate_circuit_duration(qc, qubit_props):
#     qubit_props = {int(k): v for k, v in qubit_props.items()}
#     finish_times = [0.0] * qc.num_qubits
#     error_sums = [0.0] * qc.num_qubits   # track error accumulation per qubit

#     for instr, qargs, _ in qc.data:
        
#         # single-qubit gates
#         if len(qargs) == 1 and instr.name not in ["measure", "reset"]:
#             q = qargs[0]._index
#             duration = qubit_props[q]["single_qubit_len"]
#             finish_times[q] += duration

#             # add single-qubit error
#             error_sums[q] += qubit_props[q]["single_qubit_err"]

#         # two-qubit gates
#         elif len(qargs) == 2:
#             q0, q1 = [q._index for q in qargs]

#             # duration
#             duration = (qubit_props[q0]["two_qubit_len"].get(str(q1))
#                         or qubit_props[q1]["two_qubit_len"].get(str(q0)))

#             start = max(finish_times[q0], finish_times[q1])
#             end = start + duration
#             finish_times[q0] = finish_times[q1] = end

#             # error (check both directions, like in duration)
#             err = (qubit_props[q0]["two_qubit_err"].get(str(q1))
#                    or qubit_props[q1]["two_qubit_err"].get(str(q0)))
#             if err:
#                 error_sums[q0] += err
#                 error_sums[q1] += err


#     # Compute decoherence probabilities for each qubit
#     probs = {}
#     for q in range(qc.num_qubits):
#         probs[q] = {
#             "time_us": finish_times[q],
#             "sum_error": error_sums[q]
#         }


#     max_time = max(p["time_us"] for p in probs.values())
#     max_error = max(p["sum_error"] for p in probs.values())

#     return {
#         "max_time": max_time,
#         "max_error": max_error,
#     }


# ---------------------------
# Timing & error application
# ---------------------------



def _apply_op(
    op_qubits: Tuple[int, ...],
    finish_times: Dict[int, float],
    error_sums: Dict[int, float],
    qubit_props: Dict[int, Dict[str, Any]],
    gate_multiplier: float = 1.0
) -> None:
    """
    Apply one operation on op_qubits with an ASAP start time determined by the
    current finish_times. Updates finish_times and error_sums in-place.
    """
    if len(op_qubits) == 0:
        raise ValueError("Operation must act on at least one qubit")

    if len(op_qubits) == 1:
        q = op_qubits[0]
        props = qubit_props[q]
        duration = float(props["single_qubit_len"])
        start = float(finish_times[q])
        end = start + duration
        finish_times[q] = end
        error_sums[q] += float(props.get("single_qubit_err", 0.0)) * gate_multiplier
        # error_sums[q] += .5 * gate_multiplier
        return

    if len(op_qubits) == 2:
        q0, q1 = op_qubits
        # duration lookup is symmetric with per-edge tables
        d01 = qubit_props[q0]["two_qubit_len"].get(str(q1))
        d10 = qubit_props[q1]["two_qubit_len"].get(str(q0))
        if d01 is None and d10 is None:
            raise KeyError(f"No two_qubit_len entry for edge ({q0},{q1})")
        duration = float(d01 if d01 is not None else d10)

        start = max(float(finish_times[q0]), float(finish_times[q1]))
        end = start + duration
        finish_times[q0] = end
        finish_times[q1] = end

        e01 = qubit_props[q0]["two_qubit_err"].get(str(q1)) 
        e10 = qubit_props[q1]["two_qubit_err"].get(str(q0)) 
        err = e01 if e01 is not None else e10
        if err is not None:
            err = float(err)
            # err = 1

            error_sums[q0] += err * gate_multiplier
            error_sums[q1] += err * gate_multiplier

        return

    # (Optional) extend here for 3+ qubit ops if your IR includes them.
    raise NotImplementedError(f"Arity-{len(op_qubits)} ops not supported")

# ---------------------------
# Core recursive scheduler
# ---------------------------

def _simulate_items(
    items: List[Dict[str, Any]],
    qubit_props: Dict[int, Dict[str, Any]],
    loop_iterations: int,
    use_physical_qubits: bool,
    # mutable state
    finish_times: Dict[int, float],
    error_sums: Dict[int, float],
) -> None:
    """
    Recursively simulate 'items' into finish_times & error_sums (in-place).
    Item types supported: 'gate', 'swap', 'for', 'while', 'if'.
    - 'for'/'while' use 'body' and multiply the single-iteration delta by 'loop_iterations'.
    - 'if' evaluates each branch from the SAME incoming state and commits the
      worst-case branch (by (max_time, then max_error)).
    """
    for it in items:
        t = it.get("type")

        if t in ( "swap"):
            for op in _get_qubits_for_item(it, use_physical_qubits):
                _apply_op(op, finish_times, error_sums, qubit_props, gate_multiplier=3)
        elif t == "gate":
            for op in _get_qubits_for_item(it, use_physical_qubits):
                _apply_op(op, finish_times, error_sums, qubit_props, gate_multiplier=1)

        elif t == "for" or t == "while":
            body = it.get("body", [])
            iters = int(loop_iterations)
            if iters <= 0 or not body:
                print("SKIPED")
                continue

            # simulate one iteration starting from zero to get the per-iteration delta
            tmp_finish = defaultdict(float)
            tmp_error = defaultdict(float)
            _simulate_items(body, qubit_props, loop_iterations, use_physical_qubits, tmp_finish, tmp_error)

            touched = _collect_qubits_used(body, use_physical_qubits)
            # max_error_per_iter = max(tmp_error[q] for q in touched) if touched else 0.0
            max_time_per_iter  = max(tmp_finish[q] for q in touched) if touched else 0.0
            max_error_per_iter = max(tmp_error[q] for q in touched) if touched else 0.0
            for q in touched:
                finish_times[q] += max_time_per_iter * iters
                error_sums[q]  += tmp_error[q] * iters

            
        elif t == "if":
            branches = it.get("branches", [])
            if not branches:
                continue

            # Evaluate each branch from the same incoming state
            best = None  # (max_time, max_error, finish, error)
            for br in branches:
                br_finish = deepcopy(finish_times)
                br_error  = deepcopy(error_sums)
                _simulate_items(br.get("body", []), qubit_props, loop_iterations, use_physical_qubits, br_finish, br_error)
                br_max_time = max(br_finish.values(), default=0.0)
                br_max_err  = max(br_error.values(), default=0.0)
                score = (br_max_time, br_max_err)
                if best is None or score > best[:2]:
                    best = (br_max_time, br_max_err, br_finish, br_error)

            # Commit worst-case branch
            _, _, best_finish, best_error = best
            finish_times.clear()
            finish_times.update(best_finish)
            error_sums.clear()
            error_sums.update(best_error)

        else:
            # Unknown item type -> ignore gracefully
            continue

# ---------------------------
# Public API
# ---------------------------

def estimate_dynamic_circuit(
    items: List[Dict[str, Any]],
    qubit_props: Dict[int, Dict[str, Any]],
    loop_iterations: int = 1,
    use_physical_qubits: bool = True,
) -> Dict[str, Any]:
    """
    Schedule a dynamic block list with ASAP semantics and accumulate:
      - per-qubit finish time (µs) and error sum
      - overall max_time / max_error

    'items' is your structured IR (list of dicts) with nodes of type:
      - {"type":"gate","qubits":[q]} or {"type":"gate","qubits":[q0,q1]}
      - {"type":"swap","qubits":[q0,q1]} or {"type":"swap","ops":[[q0,q1],[q1,q2],...]}
      - {"type":"for","iterations":K,"body":[...]}
      - {"type":"while","cond":..., "body":[...]}   # simulated with 'loop_iterations' bound
      - {"type":"if","branches":[{"cond":..., "body":[...]}, ...]}

    'qubit_props' matches your static function schema:
      qubit_props[q] = {
         "single_qubit_len": float,
         "single_qubit_err": float,
         "two_qubit_len": { str(other_q): float },
         "two_qubit_err": { str(other_q): float },
      }
    """
    # Normalize keys to int (like your static version)
    qubit_props = {int(k): v for k, v in qubit_props.items()}

    finish_times: Dict[int, float] = defaultdict(float)
    error_sums: Dict[int, float]   = defaultdict(float)

    _simulate_items(items, qubit_props, loop_iterations, use_physical_qubits, finish_times, error_sums)

    # Build result (mirrors your original return)
    max_time  = max(finish_times.values(), default=0.0)
    max_error = sum(error_sums.values())/len(error_sums) if error_sums else 0.0
    
    # for i in range(len(error_sums)):
    #     print(f"{i:.4f}:{error_sums[i]:.6f}", end=", ")
    # print()

    # for i in range(len(nb_single_gate)):
    #     print(f"Qubit {i}: single gates={nb_single_gate[i]}, two-qubit gates={nb_two_gate[i]}")
    # print("----------------------")
    
    return {
        "max_time": float(max_time),
        "max_error": float(max_error),
    }