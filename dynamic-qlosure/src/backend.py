from src.graph import *


class QuantumBackend:
    """Handles backend topology and precomputed data"""

    def __init__(self, edges, qubit_props=None):
        self.edges = edges
        self.connections = set(tuple(edge) for edge in edges)
        self.graph = build_backend_graph(edges)
        self.distance_matrix = compute_distance_matrix(self.graph)
        self.num_qubits = len(self.distance_matrix) + 1
        self.qubit_props = qubit_props if qubit_props is not None else {}

    @classmethod
    def from_edges(cls, edges):
        """Create backend from edge list"""
        return cls(edges)

    def __eq__(self, other):
        """Check if two backends are equivalent"""
        if not isinstance(other, QuantumBackend):
            return False
        return self.edges == other.edges

    def __hash__(self):
        """Make backend hashable for caching"""
        return hash(tuple(sorted(self.edges)))
