
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
    """
    Swap the mappings between two physical qubits and update the corresponding logical mappings.
    This function performs a swap operation between two physical qubits by updating both the
    logical-to-physical and physical-to-logical mapping dictionaries/lists. The swap updates
    which logical qubits are mapped to which physical qubits after the swap operation.
    Args:
        logical_to_physical (list): A list where index represents logical qubit and value 
                                   represents the physical qubit it's mapped to
        physical_to_logical (list): A list where index represents physical qubit and value 
                                   represents the logical qubit mapped to it (-1 if unused)
        swap_pair (tuple): A tuple of two integers representing the physical qubits to swap
        inplace (bool, optional): If True, modifies the original mappings in place. 
                                 If False, returns a copy. Defaults to False.
    Returns:
        list: Updated logical_to_physical mapping after the swap operation. If inplace=True,
              this is the same object as the input; otherwise, it's a copy.
    Note:
        When inplace=True, the physical_to_logical mapping is also updated in place.
        The function handles cases where physical qubits may not have logical qubits
        mapped to them (indicated by -1 in physical_to_logical).
    """
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
    """
    Swap the mappings between two physical qubits using ISL (Integer Set Library) operations.

    This function performs a swap operation on an ISL mapping by creating a swap transformation
    that exchanges the mappings for two specified physical qubits.

    Args:
        isl_mapping: An ISL Map object representing the current logical-to-physical qubit mapping
        swap_pair (tuple): A tuple of two integers representing the physical qubits to swap

    Returns:
        An updated ISL Map object with the swap operation applied

    Example:
        If the original mapping has logical qubit 0 -> physical qubit 1 and logical qubit 1 -> physical qubit 2,
        and swap_pair is (1, 2), the result will have logical qubit 0 -> physical qubit 2 and 
        logical qubit 1 -> physical qubit 1.
    """
    q1, q2 = swap_pair

    # Create a set containing the two physical qubits to be swapped
    swap_domain = isl.Set(f"{{[{q1}];[{q2}]}}")

    # Create a mapping that swaps q1 -> q2 and q2 -> q1
    swap_map = isl.Map(f"{{[{q1}] -> [{q2}]; [{q2}] -> [{q1}]}}")

    # Extract mappings that don't involve the qubits being swapped
    other_mapping = isl_mapping.subtract_range(swap_domain)

    # Apply the swap transformation and combine with unchanged mappings
    return isl_mapping.apply_range(swap_map).union(other_mapping)


def swap_logical_physical_isl_mapping_path(isl_mapping, swap_path_map):
    """
    Apply a swap path mapping to transform logical-physical qubit mappings.

    This function updates the ISL mapping by applying swap operations defined in the
    swap_path_map. It handles the transformation of qubit mappings when swaps are
    performed during quantum circuit execution.

    Args:
        isl_mapping: The current ISL (Integer Set Library) mapping representing
                    the logical to physical qubit relationships.
        swap_path_map: A mapping that defines the swap operations to be applied.
                      If empty, no transformation is performed.

    Returns:
        The updated ISL mapping after applying the swap path transformations.
        If swap_path_map is empty, returns the original isl_mapping unchanged.

    Note:
        The function preserves mappings outside the swap domain by subtracting
        the swap domain from the original mapping and then unioning it with
        the transformed mapping.
    """
    if swap_path_map.is_empty():
        return isl_mapping
    other_mapping = isl_mapping.subtract_range(swap_path_map.domain())
    return isl_mapping.apply_range(swap_path_map).union(other_mapping)
