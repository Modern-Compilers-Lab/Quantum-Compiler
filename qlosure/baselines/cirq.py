import cirq
import cirq_google as cg
from cirq.contrib.qasm_import import circuit_from_qasm
import networkx as nx
import re

def Sycamore():
    sycamore_device = cg.Sycamore
    device_graph = sycamore_device.metadata.nx_graph

    edges = list(device_graph.edges())

    all_qubits = sorted({q for edge in edges for q in edge}, key=lambda q: (q.row, q.col))

    qubit_to_index = {qubit: idx for idx, qubit in enumerate(all_qubits)}

    integer_edges = [[qubit_to_index[q1], qubit_to_index[q2]] for q1, q2 in edges]

    return integer_edges



def run_cirq(data,edges=None,initial_mapping="abstract"):
    
    qasm_str = data["qasm_code"].replace("u(", "u3(")
    qasm_str = re.sub(r'cp\(pi/[0-9.]+\)', 'cx', qasm_str)
    qasm_str = qasm_str.replace("cp(0)", "cx")
    cleaned_qasm = "\n".join(line for line in qasm_str.splitlines() if "reset" not in line)
    if edges:
        swap_count,depth,cx_count = run_on_ibm(cleaned_qasm,edges,initial_mapping=initial_mapping)
    else:
        swap_count,depth,cx_count = run_on_google(cleaned_qasm,initial_mapping=initial_mapping)
        

    return {
        "swaps": swap_count,
        "depth": depth,
        "cx_count": cx_count
    }
    
def run_on_google(qasm_str,initial_mapping="abstract"):
    circuit = circuit_from_qasm(qasm_str)
    sycamore_device = cg.Sycamore
    device_graph = sycamore_device.metadata.nx_graph

    router = cirq.RouteCQC(device_graph)

    if initial_mapping == "trivial":
        initial_mapper = get_trivial_mapping(circuit,device_graph)
        routed_circuit, _, _ = router.route_circuit(
            circuit,initial_mapper=initial_mapper, tag_inserted_swaps=True
        )
    else:
        routed_circuit, _, _ = router.route_circuit(
            circuit, tag_inserted_swaps=True
        )
    
    swap_count = sum(
    1 for op in routed_circuit.all_operations()
        if isinstance(op, cirq.TaggedOperation) and cirq.RoutingSwapTag() in op.tags
    )
    cx_count = sum(
        1 for op in routed_circuit.all_operations()
        if isinstance(op, cirq.GateOperation)
            and isinstance(op.gate, cirq.CNotPowGate)
            and op.gate.exponent == 1
    )

    depth = len(routed_circuit)
    
    return swap_count,depth,cx_count
    
    
def get_trivial_mapping(circuit,device_graph):
    class TrivialInitialMapper:
        def __init__(self, mapping):
            self._mapping = mapping

        def initial_mapping(self, circuit):
            return self._mapping

    # Now create your trivial mapping dictionary as before.
    logical_qubits_sorted = sorted(circuit.all_qubits(), key=lambda q: q.name)
    physical_qubits_sorted = sorted(device_graph.nodes(), key=lambda q: (q.row, q.col))
    trivial_mapping = {lq: pq for lq, pq in zip(logical_qubits_sorted, physical_qubits_sorted)}

    # Wrap the mapping in the helper class.
    initial_mapper = TrivialInitialMapper(trivial_mapping)
    
    return initial_mapper
    
    
def edges_to_device(edge_list):

    g = nx.Graph()

    nodes = set()
    for edge in edge_list:
        nodes.update(edge)
    
    qubit_map = {node: cirq.NamedQubit(str(node)) for node in nodes}
    
    for q in qubit_map.values():
        g.add_node(q)
    
    for edge in edge_list:
        q1 = qubit_map[edge[0]]
        q2 = qubit_map[edge[1]]
        g.add_edge(q1, q2)
    
    return g

def run_on_ibm(qasm_str,edges=None,initial_mapping="abstract"):
    cirq_circuit = circuit_from_qasm(qasm_str)
    device_graph = nx.Graph()
    for edge in edges:
        device_graph.add_edge(edge[0], edge[1])

    # Convert the NetworkX graph to a Cirq device graph
    cirq_device_graph = nx.Graph()
    for node in device_graph.nodes():
        cirq_device_graph.add_node(cirq.LineQubit(node))
    for edge in device_graph.edges():
        cirq_device_graph.add_edge(cirq.LineQubit(edge[0]), cirq.LineQubit(edge[1]))

    if initial_mapping == "abstract":
        router = cirq.RouteCQC(cirq_device_graph)

        routed_circuit, _, _ = router.route_circuit(
            cirq_circuit,
            tag_inserted_swaps=True
        )
    else:
        # Create trivial initial layout
        sorted_qubits = sorted(cirq_circuit.all_qubits())
        initial_mapping = {q: cirq.LineQubit(i) for i, q in enumerate(sorted_qubits)}
        
        class CustomInitialMapper:
            def __init__(self, mapping):
                self.mapping = mapping
            
            def initial_mapping(self, circuit):
                return self.mapping

        # Create the router
        router = cirq.RouteCQC(cirq_device_graph)
        
        initial_mapper = CustomInitialMapper(initial_mapping)

        # Route the circuit
        routed_circuit, _, _ = router.route_circuit(
            cirq_circuit,
            initial_mapper=initial_mapper,
            tag_inserted_swaps=True
        )
    
    swap_count = sum(
        1 for op in routed_circuit.all_operations()
            if isinstance(op, cirq.TaggedOperation) and cirq.RoutingSwapTag() in op.tags
        )
    cx_count = sum(
        1 for op in routed_circuit.all_operations()
        if isinstance(op, cirq.GateOperation)
            and isinstance(op.gate, cirq.CNotPowGate)
            and op.gate.exponent == 1
    )

    depth = len(routed_circuit)
    
    return swap_count,depth,cx_count