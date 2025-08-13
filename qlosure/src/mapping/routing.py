from src.qlosure.python.utils.circuit_utils import *
from src.qlosure.python.mapping.heuristic import *
from src.qlosure.python.utils.isl_data_loader import *
from src.qlosure.python.utils.isl_to_python import *
from src.qlosure.python.utils.python_to_isl import *
from src.qlosure.python.graph.graph import *
from src.qlosure.python.mapping.mapping import *

import islpy as isl
import random
from tqdm import tqdm
from time import time

import copy


class Qlosure():
    def __init__(self, edges, data, use_isl=False, with_circuit=True) -> None:

        self.backend_connections = set(tuple(edge) for edge in edges)
        self.backend = build_backend_graph(edges)

        self.use_isl = use_isl
        self.with_circuit = with_circuit

        self.data = data

        self.distance_matrix = compute_distance_matrix(self.backend)
        self.num_qubits = len(self.distance_matrix) + 1

        self.access, self.write_dict = self.data["read"], self.data["write"]
        self.macro_gates = self.data.get("macro_gates", {})
        self.access2q = None

        self.decay_parameter = [1 for _ in range(self.num_qubits)]
        self.qubit_depth = {q: 0 for q in range(self.num_qubits)}

        self.reset = 5
        self.isl_mapping = None
        self.mapping_dict = None
        self.reverse_mapping_dict = None
        self.front_layer = None
        self.extended_layer = None

        if with_circuit:
            self.circuit = QuantumCircuit(self.num_qubits - 1)

        self.results = {}
        self.instruction_times = defaultdict(int)

    def run(self, heuristic_method="Qlosure", enforce_read_after_read=True, transitive_reduction=True, initial_mapping_method="sabre", dag_mode="default", num_iter=1, param=5, verbose=0):
        self.init_mapping(method=initial_mapping_method)
        self.results = {}
        min_swaps = float('inf')
        min_depth = float('inf')

        successors2q, dag_predecessors2q, successors_full, dag_predecessors_full, self.access2q = generate_dag(
            self.access, self.write_dict, self.num_qubits, enforce_read_after_read, transitive_reduction)

        if not enforce_read_after_read and dag_mode == "hybrid":
            successors2q_rar_included, dag_predecessors2q_rar_included, _, _, _ = generate_dag(
                self.access, self.write_dict, self.num_qubits, enforce_read_after_read=True, transitive_reduction=transitive_reduction)
        else:
            successors2q_rar_included = successors2q
            dag_predecessors2q_rar_included = dag_predecessors2q

        dag_forward_dependencies_count = compute_dependencies_length_bitset(
            successors2q_rar_included, dag_predecessors2q_rar_included)

        if num_iter > 1:
            dag_backward_dependencies_count = compute_dependencies_length_bitset(
                dag_predecessors2q_rar_included, successors2q_rar_included)
        else:
            dag_backward_dependencies_count = dag_forward_dependencies_count

        for i in range(2*(num_iter-1)+1):
            if i % 2 == 0:
                self.dag_dependencies_count = dag_forward_dependencies_count
                self.dag2q = successors2q
                self.dag_predecessors2q = dag_predecessors2q
                self.dag2q_restricted = successors2q_rar_included
                self.dag_predecessors2q_restricted = dag_predecessors2q_rar_included
                self.dag_full = successors_full
                self.dag_predecessors_full = copy.deepcopy(
                    dag_predecessors_full) if num_iter > 1 else dag_predecessors_full
            else:
                self.dag_dependencies_count = dag_backward_dependencies_count
                self.dag2q = dag_predecessors2q
                self.dag_predecessors2q = successors2q
                self.dag2q_restricted = dag_predecessors2q_rar_included
                self.dag_predecessors2q_restricted = successors2q_rar_included
                self.dag_full = dag_predecessors_full
                self.dag_predecessors_full = copy.deepcopy(
                    successors_full) if num_iter > 1 else successors_full

            self.init_front_layer()
            self.qubit_depth = {q: 0 for q in range(self.num_qubits)}
            start = time()
            swap_count = self.execute_algorithm(
                heuristic_method, param, verbose)
            exec_time = time()-start
            if i % 2 == 0:
                if swap_count < min_swaps:
                    min_swaps = min(min_swaps, swap_count)
                    min_depth = min(min_depth, self.get_circuit_depth())
                elif swap_count == min_swaps:
                    min_depth = min(min_depth, self.get_circuit_depth())

        return min_swaps, min_depth, exec_time

    def init_mapping(self, method="trivial"):
        if method == "random":
            self.mapping_dict, self.reverse_mapping_dict = generate_random_initial_mapping(
                self.num_qubits)

        elif method == "trivial":
            self.mapping_dict, self.reverse_mapping_dict = generate_trivial_initial_mapping(
                self.num_qubits)
        elif method == "sabre":
            self.mapping_dict, self.reverse_mapping_dict = generate_sabre_initial_mapping(
                self.data["qasm_code"], self.backend_connections, self.num_qubits)
        elif method == "cirq":
            self.mapping_dict, self.reverse_mapping_dict = generate_cirq_initial_mapping(
                self.data["qasm_code"])
        else:
            raise ValueError(
                f"Unknown mapping initialization method: {method}")

        if self.use_isl:
            self.isl_mapping = dict_to_isl_map(self.mapping_dict)

    def init_front_layer(self):
        self.front_layer = set()
        for gate in self.dag_full:
            if len(self.dag_predecessors_full[gate]) == 0:
                self.front_layer.add(gate)

    def execute_algorithm(self, huristic_method, param, verbose):
        swap_count = 0
        total_gates = len(self.access)
        self.decay_parameter = [1 for _ in range(self.num_qubits)]

        with tqdm(total=total_gates, desc="Running Qlosure", mininterval=0.1, disable=(verbose == 0), leave=True) as pbar:
            while len(self.front_layer) > 0:

                ready_to_execute_gates = self.extract_ready_to_execute_gate_list()
                if len(ready_to_execute_gates) > 0:

                    self.update_front_layer(
                        ready_to_execute_gates)

                    self.decay_parameter = [1 for _ in range(self.num_qubits)]
                    pbar.update(len(ready_to_execute_gates))

                else:

                    local_swap_count = self.apply_heuristic(
                        huristic_method, param, verbose=verbose)

                    swap_count += local_swap_count

        return swap_count

    def extract_ready_to_execute_gate_list(self,):
        ready_to_execute_gates_list = []

        for gate in self.front_layer:
            if self.is_gate_executable(gate):
                ready_to_execute_gates_list.append(gate)

        return ready_to_execute_gates_list

    def is_gate_executable(self, gate) -> bool:
        if len(self.access[gate]) == 1:
            q = self.access[gate][0]
            phys_q = self.mapping_dict[q]
            new_depth = self.qubit_depth.get(phys_q, 0) + 1
            self.qubit_depth[phys_q] = new_depth
            if self.with_circuit:
                self.circuit.x(phys_q)
            return True

        q1, q2 = self.access[gate]
        phys_q1, phys_q2 = self.mapping_dict[q1], self.mapping_dict[q2]

        if (phys_q1, phys_q2) in self.backend_connections or (phys_q2, phys_q1) in self.backend_connections:
            current_depth_q1 = self.qubit_depth.get(phys_q1, 0)
            current_depth_q2 = self.qubit_depth.get(phys_q2, 0)
            new_depth = max(current_depth_q1, current_depth_q2) + 1

            self.qubit_depth[phys_q1] = new_depth
            self.qubit_depth[phys_q2] = new_depth

            if self.with_circuit:
                self.circuit.cx(min(q1, q2), max(q1, q2))

            return True
        return False

    def update_front_layer(self, executable_gates):
        for gate in executable_gates:
            for successor_gate in self.dag_full[gate]:
                self.dag_predecessors_full[successor_gate].discard(gate)
                if len(self.dag_predecessors_full[successor_gate]) == 0:
                    self.front_layer.add(successor_gate)

            self.front_layer.discard(gate)

    def apply_heuristic(self, huristic_method, param, verbose=0):
        if huristic_method not in ["decay",  "max_focus", "more_excuted", "Qlosure", "dep_weighted", "layer_adjusted", "distance_only", "fixed_extended_layer", "with_depth", "macro_gates"]:
            raise ValueError(
                f"Invalid heuristic method provided {huristic_method}. ")

        if huristic_method == "decay":
            return self._apply_decay_heuristic()

        if huristic_method == "max_focus":
            return self._apply_max_focus_heuristic()

        if huristic_method == "more_excuted":
            return self._apply_more_excuted_heuristic()

        if huristic_method == "Qlosure":
            return self._apply_closure_score_heuristic(param)

        if huristic_method == "dep_weighted":
            return self._apply_dep_weight_closure_heuristic(param)

        if huristic_method == "layer_adjusted":
            return self._apply_layered_closure_heuristic(param)

        if huristic_method == "distance_only":
            return self._apply_distance_closure_heuristic()

        if huristic_method == "fixed_extended_layer":
            return self._apply_fixed_extended_layer_heuristic()

        if huristic_method == "with_depth":
            return self._apply_with_depth_heuristic(param)
        if huristic_method == "macro_gates":
            return self._apply_macro_gates_heuristic(param)

    def _apply_decay_heuristic(self):

        logical_qubits = [
            q for gate in self.front_layer for q in self.access[gate]]
        physical_qubits = set(self.mapping_dict[q] for q in logical_qubits)

        self.extended_layer = create_extended_successor_set(
            self.front_layer, self.dag2q, self.access, extended_set_size=len(
                physical_qubits)
        )

        candidate_swaps = generate_swap_candidates(
            physical_qubits, self.backend)

        heuristic_score = {}
        for swap_gate in candidate_swaps:
            temp_mapping_dict = swap_logical_physical_mappings(
                self.mapping_dict, self.reverse_mapping_dict, swap_gate
            )
            score = decay_poly_heuristic(
                self.front_layer,
                self.extended_layer,
                temp_mapping_dict,
                self.distance_matrix,
                self.access,
                self.decay_parameter,
                swap_gate
            )
            heuristic_score[swap_gate] = score

        best_swap_gate = find_min_score_swap_gate(heuristic_score)

        if self.use_isl:
            self.isl_mapping = swap_logical_physical_isl_mapping(
                self.isl_mapping, best_swap_gate)

        swap_logical_physical_mappings(
            self.mapping_dict, self.reverse_mapping_dict, best_swap_gate, inplace=True
        )

        self.decay_parameter[best_swap_gate[0]] += 0.001
        self.decay_parameter[best_swap_gate[1]] += 0.001

        self.update_depth(best_swap_gate[0], best_swap_gate[1])

        return 1

    def _apply_max_focus_heuristic(self):

        logical_qubits = [
            q for gate in self.front_layer for q in self.access[gate]]
        physical_qubits = set(self.mapping_dict[q] for q in logical_qubits)

        self.extended_layer = create_extended_successor_set(
            self.front_layer, self.dag2q, self.access, len(physical_qubits)
        )

        candidate_swaps = generate_swap_candidates(
            physical_qubits, self.backend)

        heuristic_score = {}
        for swap_gate in candidate_swaps:
            temp_mapping_dict = swap_logical_physical_mappings(
                self.mapping_dict, self.reverse_mapping_dict, swap_gate
            )
            score = max_focus_poly_heuristic(
                self.front_layer,
                self.extended_layer,
                temp_mapping_dict,
                self.distance_matrix,
                self.access,
                self.decay_parameter,
                swap_gate
            )
            heuristic_score[swap_gate] = score

        best_swap_gate = find_min_score_swap_gate(heuristic_score)

        if self.use_isl:
            self.isl_mapping = swap_logical_physical_isl_mapping(
                self.isl_mapping, best_swap_gate)
        swap_logical_physical_mappings(
            self.mapping_dict, self.reverse_mapping_dict, best_swap_gate, inplace=True
        )

        self.decay_parameter[best_swap_gate[0]] += 0.001
        self.decay_parameter[best_swap_gate[1]] += 0.001
        self.update_depth(best_swap_gate[0], best_swap_gate[1])

        return 1

    def _apply_more_excuted_heuristic(self):

        logical_qubits = [
            q for gate in self.front_layer for q in self.access[gate]]
        physical_qubits = set(self.mapping_dict[q] for q in logical_qubits)
        self.extended_layer = create_extended_successor_set(
            self.front_layer, self.dag2q, self.access, len(physical_qubits)
        )
        candidate_swaps = generate_swap_candidates(
            physical_qubits, self.backend)

        heuristic_score = {}
        for swap_gate in candidate_swaps:
            temp_mapping_dict = swap_logical_physical_mappings(
                self.mapping_dict, self.reverse_mapping_dict, swap_gate
            )
            score = more_excuted_heuristic(
                self.front_layer,
                self.extended_layer,
                temp_mapping_dict,
                self.distance_matrix,
                self.access,
                self.decay_parameter,
                swap_gate
            )
            heuristic_score[swap_gate] = score

        best_swap_gate = find_min_score_swap_gate(heuristic_score)

        if self.use_isl:
            self.isl_mapping = swap_logical_physical_isl_mapping(
                self.isl_mapping, best_swap_gate)
        swap_logical_physical_mappings(
            self.mapping_dict, self.reverse_mapping_dict, best_swap_gate, inplace=True
        )

        self.decay_parameter[best_swap_gate[0]] += 0.001
        self.decay_parameter[best_swap_gate[1]] += 0.001
        self.update_depth(best_swap_gate[0], best_swap_gate[1])

        return 1

    def _apply_with_depth_heuristic(self, param):

        logical_qubits = [
            q for gate in self.front_layer for q in self.access2q[gate]]
        physical_qubits = set(self.mapping_dict[q] for q in logical_qubits)

        self.extended_layer, extended_layer_index = create_leveled_extended_successor_set(
            self.front_layer, self.dag2q_restricted, self.access2q, len(
                physical_qubits)*param
        )

        candidate_swaps = generate_swap_candidates(
            physical_qubits, self.backend)

        heuristic_score = {}
        for swap_gate in candidate_swaps:
            temp_mapping_dict = swap_logical_physical_mappings(
                self.mapping_dict, self.reverse_mapping_dict, swap_gate
            )

            score = with_depth_poly_heuristic(self.front_layer, self.extended_layer, temp_mapping_dict,
                                              self.distance_matrix, self.access2q, self.decay_parameter, self.dag_dependencies_count, extended_layer_index, swap_gate, self.qubit_depth)
            heuristic_score[swap_gate] = score

        best_swap_gate = find_min_score_swap_gate(heuristic_score)

        if self.use_isl:
            self.isl_mapping = swap_logical_physical_isl_mapping(
                self.isl_mapping, best_swap_gate)

        swap_logical_physical_mappings(
            self.mapping_dict, self.reverse_mapping_dict, best_swap_gate, inplace=True
        )

        self.decay_parameter[best_swap_gate[0]] += 0.001
        self.decay_parameter[best_swap_gate[1]] += 0.001

        self.update_depth(best_swap_gate[0], best_swap_gate[1])

        return 1

    def _apply_macro_gates_heuristic(self, param):

        logical_qubits = [
            q for gate in self.front_layer for q in self.access2q[gate]]
        physical_qubits = set(self.mapping_dict[q] for q in logical_qubits)

        self.extended_layer, extended_layer_index = create_leveled_extended_successor_set(
            self.front_layer, self.dag2q_restricted, self.access2q, len(
                physical_qubits)*param
        )

        candidate_swaps = generate_swap_candidates(
            physical_qubits, self.backend)

        heuristic_score = {}
        for swap_gate in candidate_swaps:
            temp_mapping_dict = swap_logical_physical_mappings(
                self.mapping_dict, self.reverse_mapping_dict, swap_gate
            )

            score = macro_gates_poly_heuristic(self.front_layer, self.extended_layer, temp_mapping_dict,
                                               self.distance_matrix, self.access2q, self.decay_parameter, self.dag_dependencies_count, extended_layer_index, swap_gate, self.qubit_depth, self.macro_gates)
            heuristic_score[swap_gate] = score

        best_swap_gate = find_min_score_swap_gate(heuristic_score)

        if self.use_isl:
            self.isl_mapping = swap_logical_physical_isl_mapping(
                self.isl_mapping, best_swap_gate)

        swap_logical_physical_mappings(
            self.mapping_dict, self.reverse_mapping_dict, best_swap_gate, inplace=True
        )

        self.decay_parameter[best_swap_gate[0]] += 0.001
        self.decay_parameter[best_swap_gate[1]] += 0.001

        self.update_depth(best_swap_gate[0], best_swap_gate[1])

        return 1

    def _apply_closure_score_heuristic(self, param):

        logical_qubits = [
            q for gate in self.front_layer for q in self.access2q[gate]]
        physical_qubits = set(self.mapping_dict[q] for q in logical_qubits)

        self.extended_layer, extended_layer_index = create_leveled_extended_successor_set(
            self.front_layer, self.dag2q_restricted, self.access2q, len(
                physical_qubits)*param
        )

        candidate_swaps = generate_swap_candidates(
            physical_qubits, self.backend)

        heuristic_score = {}
        for swap_gate in candidate_swaps:
            temp_mapping_dict = swap_logical_physical_mappings(
                self.mapping_dict, self.reverse_mapping_dict, swap_gate
            )

            score = closure_poly_heuristic(self.front_layer, self.extended_layer, temp_mapping_dict,
                                           self.distance_matrix, self.access2q, self.decay_parameter, self.dag_dependencies_count, extended_layer_index, swap_gate)
            heuristic_score[swap_gate] = score

        best_swap_gate = find_min_score_swap_gate(heuristic_score)

        if self.use_isl:
            self.isl_mapping = swap_logical_physical_isl_mapping(
                self.isl_mapping, best_swap_gate)

        swap_logical_physical_mappings(
            self.mapping_dict, self.reverse_mapping_dict, best_swap_gate, inplace=True
        )

        self.decay_parameter[best_swap_gate[0]] += 0.001
        self.decay_parameter[best_swap_gate[1]] += 0.001

        self.update_depth(best_swap_gate[0], best_swap_gate[1])

        return 1

    def _apply_distance_closure_heuristic(self):
        # print("dag : ",self.dag_dependencies_count)
        # print("distance matrix : ",self.distance_matrix)
        logical_qubits = [
            q for gate in self.front_layer for q in self.access2q[gate]]
        physical_qubits = set(self.mapping_dict[q] for q in logical_qubits)

        self.extended_layer, extended_layer_index = create_leveled_extended_successor_set(
            self.front_layer, self.dag2q_restricted, self.access2q, len(
                physical_qubits)*5
        )

        candidate_swaps = generate_swap_candidates(
            physical_qubits, self.backend)
        # print("swaps candidates", candidate_swaps)
        heuristic_score = {}
        for swap_gate in candidate_swaps:
            temp_mapping_dict = swap_logical_physical_mappings(
                self.mapping_dict, self.reverse_mapping_dict, swap_gate
            )
            # print("for swap gate", swap_gate)
            score = distance_only_poly_heuristic(self.front_layer, self.extended_layer, temp_mapping_dict,
                                                 self.distance_matrix, self.access2q, self.decay_parameter, self.dag_dependencies_count, extended_layer_index, swap_gate)
            heuristic_score[swap_gate] = score

            # print("score", score)

        best_swap_gate = find_min_score_swap_gate(heuristic_score)
        # print("best swap gate", best_swap_gate)
        # print("------------------")
        if self.use_isl:
            self.isl_mapping = swap_logical_physical_isl_mapping(
                self.isl_mapping, best_swap_gate)

        swap_logical_physical_mappings(
            self.mapping_dict, self.reverse_mapping_dict, best_swap_gate, inplace=True
        )

        self.decay_parameter[best_swap_gate[0]] += 0.001
        self.decay_parameter[best_swap_gate[1]] += 0.001

        self.update_depth(best_swap_gate[0], best_swap_gate[1])

        return 1

    def _apply_dep_weight_closure_heuristic(self, param):
        # print("dag : ",self.dag_dependencies_count)
        # print("distance matrix : ",self.distance_matrix)
        logical_qubits = [
            q for gate in self.front_layer for q in self.access2q[gate]]
        physical_qubits = set(self.mapping_dict[q] for q in logical_qubits)

        self.extended_layer, extended_layer_index = create_leveled_extended_successor_set(
            self.front_layer, self.dag2q_restricted, self.access2q, len(
                physical_qubits)*param
        )

        candidate_swaps = generate_swap_candidates(
            physical_qubits, self.backend)
        # print("swaps candidates", candidate_swaps)
        heuristic_score = {}
        for swap_gate in candidate_swaps:
            temp_mapping_dict = swap_logical_physical_mappings(
                self.mapping_dict, self.reverse_mapping_dict, swap_gate
            )
            # print("for swap gate", swap_gate)
            score = dep_weight_poly_heuristic(self.front_layer, self.extended_layer, temp_mapping_dict,
                                              self.distance_matrix, self.access2q, self.decay_parameter, self.dag_dependencies_count, extended_layer_index, swap_gate)
            heuristic_score[swap_gate] = score

            # print("score", score)

        best_swap_gate = find_min_score_swap_gate(heuristic_score)
        # print("best swap gate", best_swap_gate)
        # print("------------------")
        if self.use_isl:
            self.isl_mapping = swap_logical_physical_isl_mapping(
                self.isl_mapping, best_swap_gate)

        swap_logical_physical_mappings(
            self.mapping_dict, self.reverse_mapping_dict, best_swap_gate, inplace=True
        )

        self.decay_parameter[best_swap_gate[0]] += 0.001
        self.decay_parameter[best_swap_gate[1]] += 0.001

        self.update_depth(best_swap_gate[0], best_swap_gate[1])

        return 1

    def _apply_layered_closure_heuristic(self, param):
        # print("dag : ",self.dag_dependencies_count)
        # print("distance matrix : ",self.distance_matrix)
        logical_qubits = [
            q for gate in self.front_layer for q in self.access2q[gate]]
        physical_qubits = set(self.mapping_dict[q] for q in logical_qubits)

        self.extended_layer, extended_layer_index = create_leveled_extended_successor_set(
            self.front_layer, self.dag2q_restricted, self.access2q, len(
                physical_qubits)*param
        )

        candidate_swaps = generate_swap_candidates(
            physical_qubits, self.backend)
        # print("swaps candidates", candidate_swaps)
        heuristic_score = {}
        for swap_gate in candidate_swaps:
            temp_mapping_dict = swap_logical_physical_mappings(
                self.mapping_dict, self.reverse_mapping_dict, swap_gate
            )
            # print("for swap gate", swap_gate)
            score = layered_poly_closure_heuristic(self.front_layer, self.extended_layer, temp_mapping_dict,
                                                   self.distance_matrix, self.access2q, self.decay_parameter, self.dag_dependencies_count, extended_layer_index, swap_gate)
            heuristic_score[swap_gate] = score

            # print("score", score)

        best_swap_gate = find_min_score_swap_gate(heuristic_score)
        # print("best swap gate", best_swap_gate)
        # print("------------------")
        if self.use_isl:
            self.isl_mapping = swap_logical_physical_isl_mapping(
                self.isl_mapping, best_swap_gate)

        swap_logical_physical_mappings(
            self.mapping_dict, self.reverse_mapping_dict, best_swap_gate, inplace=True
        )

        self.decay_parameter[best_swap_gate[0]] += 0.001
        self.decay_parameter[best_swap_gate[1]] += 0.001

        self.update_depth(best_swap_gate[0], best_swap_gate[1])

        return 1

    def _apply_fixed_extended_layer_heuristic(self):
        # print("dag : ",self.dag_dependencies_count)
        # print("distance matrix : ",self.distance_matrix)
        logical_qubits = [
            q for gate in self.front_layer for q in self.access2q[gate]]
        physical_qubits = set(self.mapping_dict[q] for q in logical_qubits)

        self.extended_layer, extended_layer_index = create_leveled_extended_successor_set(
            self.front_layer, self.dag2q_restricted, self.access2q, 20
        )

        candidate_swaps = generate_swap_candidates(
            physical_qubits, self.backend)
        # print("swaps candidates", candidate_swaps)
        heuristic_score = {}
        for swap_gate in candidate_swaps:
            temp_mapping_dict = swap_logical_physical_mappings(
                self.mapping_dict, self.reverse_mapping_dict, swap_gate
            )
            # print("for swap gate", swap_gate)
            score = distance_only_poly_heuristic(self.front_layer, self.extended_layer, temp_mapping_dict,
                                                 self.distance_matrix, self.access2q, self.decay_parameter, self.dag_dependencies_count, extended_layer_index, swap_gate)
            heuristic_score[swap_gate] = score

            # print("score", score)

        best_swap_gate = find_min_score_swap_gate(heuristic_score)
        # print("best swap gate", best_swap_gate)
        # print("------------------")
        if self.use_isl:
            self.isl_mapping = swap_logical_physical_isl_mapping(
                self.isl_mapping, best_swap_gate)

        swap_logical_physical_mappings(
            self.mapping_dict, self.reverse_mapping_dict, best_swap_gate, inplace=True
        )

        self.decay_parameter[best_swap_gate[0]] += 0.001
        self.decay_parameter[best_swap_gate[1]] += 0.001

        self.update_depth(best_swap_gate[0], best_swap_gate[1])

        return 1

    def update_depth(self, q1, q2):

        current_depth_q1 = self.qubit_depth.get(q1, 0)
        current_depth_q2 = self.qubit_depth.get(q2, 0)
        new_depth = max(current_depth_q1, current_depth_q2) + 1

        self.qubit_depth[q1] = new_depth
        self.qubit_depth[q2] = new_depth

        if self.with_circuit:
            self.circuit.swap(q1, q2)

    def get_circuit_depth(self):
        return max(self.qubit_depth.values())
