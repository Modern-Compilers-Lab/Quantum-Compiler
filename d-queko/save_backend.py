import networkx as nx
import random


DENSITY = .3
SEED = 42


def generate_dense_backend(n: int, density: float = 0.8, seed: int = 42) -> nx.Graph:
    G = nx.Graph()
    G.add_nodes_from(range(n))

    random.seed(seed)

    for i in range(n):
        for j in range(i+1, n):
            if random.random() < density:
                G.add_edge(i, j)

    return G


# Generate backends for different qubit counts
# backend_16 = generate_dense_backend(16, DENSITY, SEED)
# backend_54 = generate_dense_backend(54, DENSITY, SEED)
# backend_81 = generate_dense_backend(81, DENSITY, SEED)
# backend_127 = generate_dense_backend(127, DENSITY, SEED)
backend_121 = generate_dense_backend(121, DENSITY, SEED)

# Save backends to files
# nx.write_gml(
#     backend_16, f"backend_16_qubits_seed_{SEED}_density_{DENSITY}.gml")
# nx.write_gml(
#     backend_54, f"backend_54_qubits_seed_{SEED}_density_{DENSITY}.gml")
# nx.write_gml(
#     backend_81, f"backend_81_qubits_seed_{SEED}_density_{DENSITY}.gml")
# nx.write_gml(
#     backend_127, f"backend_127_qubits_seed_{SEED}_density_{DENSITY}.gml")

nx.write_gml(
    backend_121, f"backend_121_qubits_seed_{SEED}_density_{DENSITY}.gml")
print("Generated and saved backends for 16, 54, 81, 127, and 121 qubits")
