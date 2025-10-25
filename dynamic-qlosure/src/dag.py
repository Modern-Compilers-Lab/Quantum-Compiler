from typing import Dict, Set, Callable
from itertools import count

from qiskit.circuit.controlflow import condition_resources, node_resources
from qiskit.converters import circuit_to_dag
from qiskit.circuit import SwitchCaseOp, Clbit, ClassicalRegister

from qiskit.dagcircuit.dagnode import DAGOpNode
from qiskit.circuit.controlflow import ControlFlowOp
from qiskit.converters import circuit_to_dag


def dag_simplifier(dagcircuit, _id_counter=None):
    """
    Convert a Qiskit DAGCircuit into a simple Python DAG structure,
    including only DAGOpNode nodes and handling control flow blocks recursively.
    Node IDs are assigned as 0, 1, 2, ...
    """
    simple_dag = {
        'nodes': set(),
        'successors': {},
        'predecessors': {},
        'node_data': {}
    }

    if _id_counter is None:
        _id_counter = count()

    qubit_list = list(dagcircuit.qubits)

    # Only consider DAGOpNode nodes
    op_nodes = [node for node in dagcircuit.nodes()
                if isinstance(node, DAGOpNode)]

    # Assign new sequential IDs
    node_id_map = {node: next(_id_counter)
                   for node in op_nodes}

    for node in op_nodes:

        # print(node._node_id)
        # exit(1)
        node_id = node_id_map[node]
        simple_dag['nodes'].add(node_id)
        simple_dag['successors'][node_id] = set()
        simple_dag['predecessors'][node_id] = set()

        qubits_accessed = [q._index for q in node.qargs]
        # coupled_qubits = [tuple(qubits_accessed)] if len(
        #     qubits_accessed) == 2 else []
        coupled_qubits = (
            [tuple(qubits_accessed)]
            if getattr(node.op, "num_qubits", 0) == 2
            else []
        )

        control_flow_blocks = None

        if isinstance(node.op, ControlFlowOp):
            control_flow_blocks = []
            qubits_in_blocks = set()
            coupled_qubits: set[tuple[int, int]] = set()
            for block in node.op.blocks:
                block_dag = circuit_to_dag(block)
                block_simple_dag = dag_simplifier(
                    block_dag, _id_counter)  # recursive call
                control_flow_blocks.append(block_simple_dag)
                # Collect all qubits accessed in this block
                for ninfo in block_simple_dag['node_data'].values():
                    qubits = ninfo['qubits_accessed']
                    qubits_in_blocks.update(qubits)
                    coupled_qubits.update(ninfo['coupled_qubits_accessed'])
            qubits_accessed = sorted(qubits_in_blocks)

        node_info = {
            'node_id': node_id,
            'qubits_accessed': qubits_accessed,
            'coupled_qubits_accessed': coupled_qubits,
            'operation_type': getattr(node.op, 'name', type(node.op).__name__),
            'control_flow_blocks': control_flow_blocks,
            'is_control_flow': isinstance(node.op, ControlFlowOp),
        }

        simple_dag['node_data'][node_id] = node_info

    # Only connect DAGOpNode nodes, using new IDs
    for node in op_nodes:
        node_id = node_id_map[node]
        for succ in dagcircuit.successors(node):
            if isinstance(succ, DAGOpNode):
                succ_id = node_id_map[succ]
                simple_dag['successors'][node_id].add(succ_id)
                simple_dag['predecessors'][succ_id].add(node_id)

    return simple_dag


def build_dag(qc):
    return dag_simplifier(circuit_to_dag(qc))


# ---------------------------------------------------------------
# helper: follow successors through nodes that are *not* kept
# ---------------------------------------------------------------

def _reachable_kept_successors(node: int,
                               simple_dag: Dict,
                               kept: Set[int]) -> Set[int]:
    """
    Return the set of kept-nodes that are reachable from `node`
    without passing through another kept-node (i.e. first kept
    nodes you hit along each path).
    """
    out, stack, seen = set(), list(simple_dag['successors'][node]), set()
    while stack:
        cur = stack.pop()
        if cur in seen:
            continue
        seen.add(cur)
        if cur in kept:
            out.add(cur)             # first kept node on that branch
        else:
            stack.extend(simple_dag['successors'][cur])
    return out


def _compress(simple_dag: Dict,
              keep_predicate: Callable[[Dict], bool]) -> Dict:
    """
    Generic graph compression.
    Nodes that satisfy `keep_predicate(node_info)` survive;
    everything else is removed, but reachability between surviving
    nodes is preserved.
    """
    # 1. decide which nodes we keep
    kept = {n for n, info in simple_dag['node_data'].items()
            if keep_predicate(info)}

    # 2. build new container
    new_dag = {
        'nodes': set(kept),
        'successors': {n: set() for n in kept},
        'predecessors': {n: set() for n in kept},
        'node_data': {}
    }

    # 3. copy node data (and recurse into control-flow blocks)
    for n in kept:
        info = simple_dag['node_data'][n].copy()
        if info['control_flow_blocks'] is not None:
            info['control_flow_blocks'] = [
                _compress(bl, keep_predicate)
                for bl in info['control_flow_blocks']
            ]
        new_dag['node_data'][n] = info

    # 4. connect nodes, preserving indirect reachability
    for n in kept:
        for ksucc in _reachable_kept_successors(n, simple_dag, kept):
            new_dag['successors'][n].add(ksucc)
            new_dag['predecessors'][ksucc].add(n)

    return new_dag


def extract_multi_qubit_dag(simple_dag: Dict) -> Dict:
    """
    Compress `simple_dag` to contain only operations that touch
    more than one qubit, while *preserving* dependency edges via
    paths that pass through dropped single-qubit nodes.
    """
    return _compress(
        simple_dag,
        keep_predicate=lambda info: len(info['qubits_accessed']) > 1
    )
