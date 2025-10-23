from typing import Any, Dict, List, Optional, Tuple
from copy import deepcopy

try:
    # Qiskit Terra >= 0.24 style control-flow ops
    from qiskit.circuit.controlflow import ForLoopOp, IfElseOp, WhileLoopOp
except Exception:
    # Older Terra fallback (names existed under qiskit.circuit.library.control_flow)
    from qiskit.circuit.library.control_flow import ForLoopOp, IfElseOp, WhileLoopOp

from qiskit import QuantumCircuit
from qiskit.circuit import Instruction, Gate
from qiskit.circuit.library import SwapGate


# =============== Public API =================

def build_structured_trace_from_circuit(
    circuit: QuantumCircuit,
    logical_to_physical: Optional[Dict[int, int]] = None,
    *,
    decompose: bool = False
) -> List[Dict[str, Any]]:
    """
    Convert a Qiskit QuantumCircuit (with possible control-flow) into a structured trace.

    Args:
        circuit: Qiskit circuit to trace.
        logical_to_physical: Optional mapping {logical_index -> physical_index}.
            If None, physical_qubits == logical_qubits in the output.
        decompose: If True, call circuit = circuit.decompose(reps=10) before tracing
            so composite gates are expanded.

    Returns:
        A list of structured trace entries (dicts) with schema:
          - {"type":"gate","op":<str>,"logical_qubits":[...],"physical_qubits":[...]}
          - {"type":"swap","logical_qubits":[lq0,lq1],"physical_qubits":[pq0,pq1]}
          - {"type":"for","iterations":<int>,"body":[...]}
          - {"type":"if","branches":[{"label":"then","body":[...]},{"label":"else","body":[...]}]}
          - {"type":"while","body":[...]}   # included if while-loops exist
    """
    circ = circuit
    # Pre-compute logical qubit index lookup (Qubit -> int)
    q_index = _make_qubit_index(circ)

    # Default physical == logical if not provided
    if logical_to_physical is None:
        logical_to_physical = {i: i for i in range(len(circ.qubits))}

    trace = _trace_block(circ, q_index, logical_to_physical)
    return trace


# =============== Internals ==================

IGNORED_OP_NAMES = {"barrier", "delay", "snapshot"}  # extend if desired


def _make_qubit_index(circ: QuantumCircuit) -> Dict[Any, int]:
    """Map each Qubit object to its logical index in circuit.qubits."""
    return {q: i for i, q in enumerate(circ.qubits)}


def _logical_and_physical(
    q_index: Dict[Any, int],
    logical_to_physical: Dict[int, int],
    qargs: Tuple[Any, ...],
) -> Tuple[List[int], List[int]]:
    """Return (logical_indices, physical_indices) for the given qargs."""
    logical = [q_index[q] for q in qargs]
    physical = [logical_to_physical.get(i, i) for i in logical]
    return logical, physical


def _is_swap(op: Instruction) -> bool:
    """Detect swap gate robustly."""
    return isinstance(op, SwapGate) or op.name.lower() == "swap"


def _is_control_flow(op: Instruction) -> bool:
    return isinstance(op, (ForLoopOp, IfElseOp, WhileLoopOp))


def _trace_block(
    circ: QuantumCircuit,
    q_index: Dict[Any, int],
    logical_to_physical: Dict[int, int],
) -> List[Dict[str, Any]]:
    """Trace a single circuit block (no wrapping CF) into a list of entries."""
    entries: List[Dict[str, Any]] = []

    for instr, qargs, cargs in circ.data:
        op: Instruction = instr

        # Control-flow ops → recurse
        if _is_control_flow(op):
            if isinstance(op, ForLoopOp):
                # ForLoopOp has .blocks = [body], and .params[0] is the index set (e.g., range(...))
                body_circ: QuantumCircuit = op.blocks[0]
                indexset = op.params[0]
                iterations = _len_indexset(indexset)
                body_entries = _trace_block(
                    body_circ, q_index, logical_to_physical)
                entries.append({
                    "type": "for",
                    "iterations": iterations,
                    "body": body_entries
                })

            elif isinstance(op, IfElseOp):
                # IfElseOp has .blocks = [true_body, false_body(or None)]
                true_circ: QuantumCircuit = op.blocks[0]
                false_circ: Optional[QuantumCircuit] = op.blocks[1] if len(
                    op.blocks) > 1 else None
                true_entries = _trace_block(
                    true_circ, q_index, logical_to_physical)
                branches = [{"label": "then", "body": true_entries}]
                if false_circ is not None:
                    false_entries = _trace_block(
                        false_circ, q_index, logical_to_physical)
                    branches.append({"label": "else", "body": false_entries})
                entries.append({
                    "type": "if",
                    "branches": branches
                })

            elif isinstance(op, WhileLoopOp):
                # While loops don't have a fixed iteration count statically.
                # We represent as a while block for completeness.
                body_circ: QuantumCircuit = op.blocks[0]
                body_entries = _trace_block(
                    body_circ, q_index, logical_to_physical)
                entries.append({
                    "type": "while",
                    "body": body_entries
                })

            continue  # handled

        # Non-control-flow ops
        name = op.name.lower()

        # Ignore non-semantic ops
        if name in IGNORED_OP_NAMES:
            continue

        # Qubit indices
        lqs, pqs = _logical_and_physical(
            q_index, logical_to_physical, tuple(qargs))

        if _is_swap(op):
            # Expect 2-qubit swap
            entries.append({
                "type": "swap",
                "logical_qubits": lqs,
                "physical_qubits": pqs
            })
        else:
            # Generic gate (keep op name)
            # NOTE: multi-qubit gates are supported (len(lqs) can be 1 or 2+)
            entries.append({
                "type": "gate",
                "op": name,
                "logical_qubits": lqs,
                "physical_qubits": pqs
            })

    return entries


def _len_indexset(indexset) -> int:
    """
    Robustly get the iteration count from ForLoopOp.indexset.
    Handles range(...), list/tuple, and ints.
    """
    try:
        return len(indexset)
    except TypeError:
        # range-like object without __len__? fall back:
        try:
            # If it's a qiskit-like RangeParameter or similar iterable
            return sum(1 for _ in indexset)
        except Exception:
            # Worst case: treat as 1 iteration
            return 1


def format_structured_trace(trace: List[Dict[str, Any]], indent: int = 0, gate_counter: Optional[List[int]] = None) -> str:
    """
    Pretty print a structured trace with global numbering and indentation.

    Args:
        trace: The structured trace to format
        indent: Current indentation level
        gate_counter: Mutable list [count] to track global gate numbering across recursive calls

    Returns:
        Formatted string representation of the trace
    """
    if gate_counter is None:
        gate_counter = [0]

    lines = []
    pad = "  " * indent

    def qlabel(qs: List[int]) -> str:
        # Logical qubit labels for display (q0, q1, ...)
        return ", ".join(f"q{q}" for q in qs)

    for item in trace:
        if item["type"] == "gate":
            gate_counter[0] += 1
            op = item["op"]
            lqs = item["logical_qubits"]
            lines.append(
                f"{pad}gate {gate_counter[0]} ({op}, {qlabel(lqs)})")

        elif item["type"] == "swap":
            gate_counter[0] += 1
            lqs = item["logical_qubits"]
            lines.append(
                f"{pad}gate {gate_counter[0]} (swap, {qlabel(lqs)})")

        elif item["type"] == "for":
            lines.append(f"{pad}for (iterations={item['iterations']}) {{")
            lines.append(format_structured_trace(
                item["body"], indent + 1, gate_counter))
            lines.append(f"{pad}}}")

        elif item["type"] == "if":
            lines.append(f"{pad}if {{")
            for idx, br in enumerate(item["branches"]):
                label = br.get("label", f"branch{idx}")
                lines.append(f"{pad}  // {label}")
                lines.append(format_structured_trace(
                    br["body"], indent + 1, gate_counter))
            lines.append(f"{pad}}}")
            # Optionally show reconciliation swaps info (commented style)
            rec = item.get("reconciliation_swaps") or {}
            for k, swaps in rec.items():
                if swaps:
                    formatted = ", ".join(
                        [f"(q{a}, q{b})" for a, b in swaps])
                    lines.append(
                        f"{pad}// reconciliation_swaps[{k}]: {formatted}")

        elif item["type"] == "while":
            lines.append(f"{pad}while {{")
            lines.append(format_structured_trace(
                item["body"], indent + 1, gate_counter))
            lines.append(f"{pad}}}")

    return "\n".join(lines)
