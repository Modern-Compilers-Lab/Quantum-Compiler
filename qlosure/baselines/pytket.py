from pytket import Circuit,OpType
from pytket.qasm import circuit_from_qasm_str
from pytket.architecture import Architecture
from pytket.placement import place_with_map
from pytket.passes import RoutingPass
from pytket._tket.unit_id import Node 
from pytket.utils.stats import gate_counts
from pytket.transform import Transform










def run_pytket(data,edges,initial_mapping=None):
    circuit = circuit_from_qasm_str(data["qasm_code"])
    architecture = Architecture(edges)

    if initial_mapping == "trivial":
        mapping = {q: Node(i) for i, q in enumerate(circuit.qubits)}
        place_with_map(circuit, mapping)


    routing_pass = RoutingPass(architecture)
    routing_pass.apply(circuit)

    circuit = custom_decompose_bridge(circuit)
  
    swap_count = sum(1 for gate in circuit.get_commands() if gate.op.type == OpType.SWAP) 


    return {
        "swaps": swap_count,
        "depth": circuit.depth(),
        "circ":circuit
    }
    
    

def custom_decompose_bridge(circuit):
    new_circuit = Circuit(len(circuit.qubits))
    qubit_map = {q: new_circuit.qubits[i] for i, q in enumerate(circuit.qubits)}
    
    for cmd in circuit.get_commands():
        if cmd.op.type == OpType.BRIDGE:
            q0, q1, q2 = cmd.qubits  
            new_q0, new_q1, new_q2 = qubit_map[q0], qubit_map[q1], qubit_map[q2]
            new_circuit.SWAP(new_q0, new_q1)
            new_circuit.CX(new_q1, new_q2)
            new_circuit.SWAP(new_q0, new_q1)
        else:
            new_qubits = [qubit_map[q] for q in cmd.qubits]
            new_circuit.add_gate(cmd.op, new_qubits)
    return new_circuit