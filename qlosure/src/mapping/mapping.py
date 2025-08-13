
import random
import islpy as isl
import networkx as nx

from qiskit import QuantumCircuit
from qiskit.transpiler import CouplingMap
from qiskit.transpiler.passes import SabreLayout
from qiskit.converters import circuit_to_dag


def generate_random_initial_mapping(num_qubits: int):
    """
    Generate a random mapping from logical qubits to physical qubits, as arrays.
    - mapping[logical] = physical
    - reverse_mapping[physical] = logical
    """
    logical_qubits = list(range(num_qubits))
    physical_qubits = list(range(num_qubits))
    random.shuffle(physical_qubits)

    # Initialize arrays
    mapping = [-1] * num_qubits
    reverse_mapping = [-1] * num_qubits

    for logical_qubit, physical_qubit in zip(logical_qubits, physical_qubits):
        mapping[logical_qubit] = physical_qubit
        reverse_mapping[physical_qubit] = logical_qubit

    return mapping, reverse_mapping


def generate_trivial_initial_mapping(num_qubits: int):
    """
    Generate a trivial mapping from logical qubits to physical qubits (arrays).
    - mapping[logical] = logical
    - reverse_mapping[logical] = logical
    """
    mapping = list(range(num_qubits))          # mapping[i] = i
    reverse_mapping = list(range(num_qubits))  # reverse_mapping[i] = i
    return mapping, reverse_mapping


def generate_sabre_initial_mapping(qasm_code, backend_edges, num_qubits):
    """
    Use Qiskit's SabreLayout to generate an initial layout, returned as arrays.
    - mapping[logical] = physical
    - reverse_mapping[physical] = logical
    """
    circuit = QuantumCircuit.from_qasm_str(qasm_code)
    dag_circuit = circuit_to_dag(circuit)
    coupling_map = CouplingMap(backend_edges)
    sabre_layout = SabreLayout(coupling_map, seed=21)
    sabre_layout.run(dag_circuit)

    layout = sabre_layout.property_set["layout"]

    # Figure out how many qubits are in use (excluding ancillas).
    # You could also just assume 'num_qubits = circuit.num_qubits'.
    # For safety, we go by the max qubit index found in the layout.
    max_index = -1
    for v in layout._v2p:
        if v._register._name != "ancilla":
            if v._index > max_index:
                max_index = v._index
            if layout._v2p[v] > max_index:
                max_index = layout._v2p[v]

    # Initialize array-based mappings
    mapping = [-1] * num_qubits
    reverse_mapping = [-1] * num_qubits

    for v in layout._v2p:
        # Skip ancilla qubits
        if v._register._name == "ancilla":
            continue

        logical_idx = v._index
        physical_idx = layout._v2p[v]
        if logical_idx < num_qubits and physical_idx < num_qubits:
            mapping[logical_idx] = physical_idx
            reverse_mapping[physical_idx] = logical_idx

    return mapping, reverse_mapping


def swap_logical_physical_mappings(logical_to_physical, physical_to_logical, swap_pair, inplace=False):
    updated_mapping = logical_to_physical if inplace else logical_to_physical[:]
    physical_1, physical_2 = swap_pair

    logical_1 = physical_to_logical[physical_1]
    logical_2 = physical_to_logical[physical_2]

    if logical_1 != -1:
        updated_mapping[logical_1] = physical_2

    if logical_2 != -1:
        updated_mapping[logical_2] = physical_1

    if inplace:
        physical_to_logical[physical_1] = logical_2
        physical_to_logical[physical_2] = logical_1

    return updated_mapping


def swap_logical_physical_isl_mapping(isl_mapping, swap_pair):
    q1, q2 = swap_pair

    swap_domain = isl.Set(f"{{[{q1}];[{q2}]}}")
    swap_map = isl.Map(f"{{[{q1}] -> [{q2}]; [{q2}] -> [{q1}]}}")

    other_mapping = isl_mapping.subtract_range(swap_domain)
    return isl_mapping.apply_range(swap_map).union(other_mapping)


def swap_logical_physical_isl_mapping_path(isl_mapping, swap_path_map):
    if swap_path_map.is_empty():
        return isl_mapping
    other_mapping = isl_mapping.subtract_range(swap_path_map.domain())
    return isl_mapping.apply_range(swap_path_map).union(other_mapping)
