import copy
from platform import node
from tqdm import tqdm

from src.graph import *
from src.mapping import *
from src.heuristic import *
from src.backend import QuantumBackend
from src.dag import extract_multi_qubit_dag
from src.token_swapping_rust import solve_token_swapping

from typing import List, Tuple, Dict, Any, Optional


class Qlosure():
    def __init__(self, backend: QuantumBackend, with_circuit=True) -> None:
        self.backend_config = backend
        self.backend_connections = backend.connections
        self.backend = backend.graph
        self.distance_matrix = backend.distance_matrix
        self.num_qubits = backend.num_qubits

        self.with_circuit = with_circuit

        self.decay_parameter = [1 for _ in range(self.num_qubits)]
        self.qubit_depth = {q: 0 for q in range(self.num_qubits)}

        self.reset = 5
        if with_circuit:
            self.circuit = QuantumCircuit(self.num_qubits - 1)

        self.results = {}
        self.node_data = None

        # --- Trace state ---
        # Stack of lists; top ([-1]) is the current block we append into.
        self._trace_stack: List[List[Dict[str, Any]]] = [[]]
        # Single global counter across the whole program for pretty numbering
        self._global_gate_counter: int = 0

    def run(self, dag, dag2q, heuristic_method="Qlosure", initial_mapping_method="trivial", initial_mapping=None, num_iter=1, param=5, verbose=0):

        self.init_mapping(method=initial_mapping_method,
                          initial_mapping=initial_mapping)
        self.results = {}
        self.swap_history = []
        min_swaps = float('inf')
        min_depth = float('inf')

        self.dag = dag
        self.dag2q = dag2q

        successors2q = dag2q['successors']
        predecessors2q = dag2q['predecessors']
        successors_full = dag['successors']
        predecessors_full = dag['predecessors']

        self.node_data = dag['node_data']

        dag_forward_dependencies_count = compute_transitive_closure_bitset(
            successors2q, predecessors2q)

        if num_iter > 1:
            dag_backward_dependencies_count = compute_transitive_closure_bitset(
                predecessors2q, successors2q)
        else:
            dag_backward_dependencies_count = dag_forward_dependencies_count

        initial_mapping = None
        for i in range(2*(num_iter-1)+1):

            if i % 2 == 0:
                self.dag_dependencies_count = dag_forward_dependencies_count
                self.dag_successors2q = successors2q
                self.dag_predecessors2q = predecessors2q
                self.dag_successors_full = successors_full
                self.dag_predecessors_full = copy.deepcopy(
                    predecessors_full) if num_iter > 1 else predecessors_full
                initial_mapping = copy.deepcopy(self.mapping_dict)
            else:
                self.dag_dependencies_count = dag_backward_dependencies_count
                self.dag_successors2q = predecessors2q
                self.dag_predecessors2q = successors2q
                self.dag_successors_full = predecessors_full
                self.dag_predecessors_full = copy.deepcopy(
                    successors_full) if num_iter > 1 else successors_full

            self.init_front_layer()
            self.qubit_depth = {q: 0 for q in range(self.num_qubits)}
            swap_count = self.execute_algorithm(
                heuristic_method, param, verbose)

            if i % 2 == 0:
                if swap_count < min_swaps:
                    min_swaps = min(min_swaps, swap_count)
                    min_depth = min(min_depth, self.get_circuit_depth())
                elif swap_count == min_swaps:
                    min_depth = min(min_depth, self.get_circuit_depth())
        return min_swaps, min_depth, initial_mapping

    def init_mapping(self, method="trivial", initial_mapping=None):

        if method not in ["random", "trivial", "sabre", "custom"]:
            raise ValueError(
                f"Unknown mapping initialization method: {method}")

        if method == "random":
            self.mapping_dict, self.reverse_mapping_dict = generate_random_initial_mapping(
                self.num_qubits)

        elif method == "trivial":
            self.mapping_dict, self.reverse_mapping_dict = generate_trivial_initial_mapping(
                self.num_qubits)
        elif method == "custom":
            if initial_mapping is None:
                raise ValueError(
                    "Initial mapping must be provided for 'custom' method")
            self.mapping_dict, self.reverse_mapping_dict = generate_custom_initial_mapping(
                initial_mapping, self.num_qubits)
        # elif method == "sabre":
        #     self.mapping_dict, self.reverse_mapping_dict = generate_sabre_initial_mapping(
        #         self.data["qasm_code"], self.backend_connections, self.num_qubits)

        else:
            raise ValueError(
                f"Unknown mapping initialization method: {method}")

    def init_front_layer(self):
        self.front_layer = set()
        for gate in self.dag_successors_full:
            if len(self.dag_predecessors_full[gate]) == 0:
                self.front_layer.add(gate)

    def execute_algorithm(self, huristic_method, param, verbose):
        swap_count = 0
        total_gates = len(self.dag_successors_full)
        self.decay_parameter = [1 for _ in range(self.num_qubits)]
        with tqdm(total=total_gates, desc="Running Qlosure", mininterval=0.1, disable=(verbose == 0), leave=True) as pbar:
            while len(self.front_layer) > 0:

                ready_to_execute_gates = self.extract_ready_to_execute_gate_list()

                if len(ready_to_execute_gates) > 0:

                    self.update_front_layer(
                        ready_to_execute_gates)

                    self.decay_parameter = [1 for _ in range(self.num_qubits)]
                    pbar.update(len(ready_to_execute_gates))
                elif all(self.node_data[gate]['is_control_flow'] for gate in self.front_layer):
                    node_id = next(
                        gate for gate in self.front_layer if self.node_data[gate]['is_control_flow'])
                    operation = self.node_data[node_id]['operation_type']
                    if operation == "for_loop" or operation == "while_loop":
                        # handle for loop
                        local_swap_count, local_depth = self.handle_for_loop(
                            node_id, self.dag, self.dag2q, operation)
                    elif operation == "if_else":
                        # handle if-else
                        local_swap_count, local_depth = self.handle_if_else(
                            node_id, self.dag, self.dag2q)

                    else:
                        raise ValueError(
                            f"Unknown control flow operation: {operation}")
                    self.decay_parameter = [
                        1 for _ in range(self.num_qubits)]
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
        if len(self.node_data[gate]['qubits_accessed']) == 1:
            q = self.node_data[gate]['qubits_accessed'][0]
            phys_q = self.mapping_dict[q]
            new_depth = self.qubit_depth.get(phys_q, 0) + 1
            self.qubit_depth[phys_q] = new_depth
            if self.with_circuit:
                self.circuit.x(phys_q)

            node = self.node_data[gate]
            op = node.get('operation_type', None)
            op_label = op if op else "x"
            self._append_gate_entry(op_label, [q], [phys_q], node_id=gate)

            return True

        elif len(self.node_data[gate]['qubits_accessed']) == 2:

            q1, q2 = self.node_data[gate]['qubits_accessed']
            phys_q1, phys_q2 = self.mapping_dict[q1], self.mapping_dict[q2]

            if (phys_q1, phys_q2) in self.backend_connections or (phys_q2, phys_q1) in self.backend_connections:
                current_depth_q1 = self.qubit_depth.get(phys_q1, 0)
                current_depth_q2 = self.qubit_depth.get(phys_q2, 0)
                new_depth = max(current_depth_q1, current_depth_q2) + 1

                self.qubit_depth[phys_q1] = new_depth
                self.qubit_depth[phys_q2] = new_depth

                if self.with_circuit:
                    self.circuit.cx(min(q1, q2), max(q1, q2))

                op = self.node_data[gate].get('operation_type', None)
                op_label = op if op else "cx"
                self._append_gate_entry(op_label, [q1, q2], [
                    phys_q1, phys_q2], node_id=gate)
                return True
            return False
        else:
            return False
            for q1, q2 in self.node_data[gate]['coupled_qubits_accessed']:
                phys_q1, phys_q2 = self.mapping_dict[q1], self.mapping_dict[q2]
                if (phys_q1, phys_q2) not in self.backend_connections and (phys_q2, phys_q1) not in self.backend_connections:
                    return False

                current_depth_q1 = self.qubit_depth.get(phys_q1, 0)
                current_depth_q2 = self.qubit_depth.get(phys_q2, 0)
                new_depth = max(current_depth_q1, current_depth_q2) + 1

                self.qubit_depth[phys_q1] = new_depth
                self.qubit_depth[phys_q2] = new_depth

                if self.with_circuit:
                    self.circuit.cx(min(q1, q2), max(q1, q2))

            return True

    def update_front_layer(self, executable_gates):
        for gate in executable_gates:
            for successor_gate in self.dag_successors_full[gate]:
                self.dag_predecessors_full[successor_gate].discard(gate)
                if len(self.dag_predecessors_full[successor_gate]) == 0:
                    self.front_layer.add(successor_gate)

            self.front_layer.discard(gate)

    def apply_heuristic(self, huristic_method, param, verbose=0):
        if huristic_method not in ["Qlosure"]:
            raise ValueError(
                f"Invalid heuristic method provided {huristic_method}. ")

        if huristic_method == "Qlosure":
            return self._apply_qlosure_score_heuristic(param)

    def _apply_qlosure_score_heuristic(self, param):

        logical_qubits = [
            q for gate in self.front_layer for q in self.node_data[gate]['qubits_accessed'] if len(self.node_data[gate]['qubits_accessed']) == 2 and not self.node_data[gate]['is_control_flow']]
        physical_qubits = set(self.mapping_dict[q] for q in logical_qubits)

        self.extended_layer, extended_layer_index = create_leveled_extended_successor_set(
            self.front_layer, self.dag_successors2q, len(
                physical_qubits)*param, self.node_data
        )

        candidate_swaps = generate_swap_candidates(
            physical_qubits, self.backend)

        heuristic_score = {}
        for swap_gate in candidate_swaps:
            temp_mapping_dict = swap_logical_physical_mappings(
                self.mapping_dict, self.reverse_mapping_dict, swap_gate
            )

            score = qlosure_poly_heuristic(self.front_layer, self.extended_layer, temp_mapping_dict,
                                           self.distance_matrix, self.node_data, self.decay_parameter, self.dag_dependencies_count, extended_layer_index, swap_gate)
            heuristic_score[swap_gate] = score

        best_swap_gate = find_min_score_swap_gate(heuristic_score)

        swap_logical_physical_mappings(
            self.mapping_dict, self.reverse_mapping_dict, best_swap_gate, inplace=True
        )

        if self.with_circuit:
            self.circuit.swap(self.reverse_mapping_dict[best_swap_gate[0]],
                              self.reverse_mapping_dict[best_swap_gate[1]])

        pq0, pq1 = best_swap_gate
        lq0 = self.reverse_mapping_dict[pq0]
        lq1 = self.reverse_mapping_dict[pq1]
        self._append_swap_entry([lq0, lq1], [pq0, pq1])

        self.decay_parameter[best_swap_gate[0]] += 0.001
        self.decay_parameter[best_swap_gate[1]] += 0.001

        self.swap_history.append(best_swap_gate)

        self.update_depth(best_swap_gate[0], best_swap_gate[1])

        return 1

    def update_depth(self, q1, q2):

        current_depth_q1 = self.qubit_depth.get(q1, 0)
        current_depth_q2 = self.qubit_depth.get(q2, 0)
        new_depth = max(current_depth_q1, current_depth_q2) + 1

        self.qubit_depth[q1] = new_depth
        self.qubit_depth[q2] = new_depth

    def get_circuit_depth(self):
        return max(self.qubit_depth.values())

    def handle_for_loop(self, loop_node_id, dag, dag2q, operation):
        loop_dag = copy.deepcopy(
            dag['node_data'][loop_node_id]["control_flow_blocks"][0])

        loop_dag2q = extract_multi_qubit_dag(loop_dag)

        # Merge node_data
        loop_dag2q['node_data'] = loop_dag2q['node_data'] | dag2q['node_data']
        loop_dag['node_data'] = loop_dag['node_data'] | dag['node_data']

        # ---- Pass 1: find a good entry mapping for the loop body
        qlosure1 = Qlosure(self.backend_config, with_circuit=self.with_circuit)

        # first pass to generate a good initial mapping
        swap_count, depth, good_initial_mapping = qlosure1.run(loop_dag, loop_dag2q, heuristic_method="Qlosure",
                                                               initial_mapping_method="custom", initial_mapping=self.mapping_dict, num_iter=3)

        _, _, before_loop_swaps = solve_token_swapping(
            self.backend_connections,
            self.mapping_dict,
            good_initial_mapping,
        )

        # ---- BEFORE-LOOP reconciliation: current -> good_initial_mapping
        for (p0, p1) in before_loop_swaps:
            # translate to logical using reverse mapping at this point
            l0 = self.reverse_mapping_dict[p0]
            l1 = self.reverse_mapping_dict[p1]

            if self.with_circuit:
                self.circuit.swap(
                    self.reverse_mapping_dict[p0], self.reverse_mapping_dict[p1])

            self._append_swap_entry([l0, l1], [p0, p1])

        # ---- Pass 2: compile the loop body from good_initial_mapping
        qlosure2 = Qlosure(self.backend_config, with_circuit=self.with_circuit)
        swap_count, depth, good_initial_mapping = qlosure2.run(loop_dag, loop_dag2q, heuristic_method="Qlosure",
                                                               initial_mapping_method="custom", initial_mapping=good_initial_mapping)

        # Compute per-iteration END → START reconciliation (to make mapping invariant per iter)
        _, _, after_loop_swaps = solve_token_swapping(
            self.backend_connections,
            qlosure2.mapping_dict,
            good_initial_mapping,
        )

        # Append those reconciliation swaps to the LOOP BODY itself (so they occur every iteration)
        # Use the branch/body's own reverse mapping to label logical qubits correctly
        for (p0, p1) in after_loop_swaps:
            l0 = qlosure2.reverse_mapping_dict[p0]
            l1 = qlosure2.reverse_mapping_dict[p1]
            if self.with_circuit:
                # body circuit uses physical indices
                qlosure2.circuit.swap(p0, p1)
            qlosure2._append_swap_entry([l0, l1], [p0, p1])

        self.update_front_layer([loop_node_id])

        iterations = 100
        if self.with_circuit:
            for_body = qlosure2.circuit
            self.circuit.for_loop(
                range(
                    iterations), None, for_body, self.circuit.qubits, self.circuit.clbits
            )

            # Structured trace: body is per-iteration content (includes reconciliation swaps)
        inner_body = qlosure2.get_structured_trace()
        if operation == "for_loop":
            self._append_for_block(iterations=iterations, body=inner_body)
        elif operation == "while_loop":
            self._append_while_block(
                condition="condition_placeholder", body=inner_body)

        return swap_count, depth

    def handle_if_else(self, if_else_node_id, dag, dag2q):
        blocks = dag['node_data'][if_else_node_id]["control_flow_blocks"]

        initial_mapping = copy.deepcopy(self.mapping_dict)
        mapping_possibilities = {-1: initial_mapping}
        branch_traces = {}

        for block_idx, block in enumerate(blocks):

            dag_block = copy.deepcopy(block)
            dag_block2q = extract_multi_qubit_dag(dag_block)

            # for leaf_node in dag_block2q['nodes']:
            #     if len(dag_block2q['successors'][leaf_node]) == 0:
            #         # attach it to the successors of if_else_node_id
            #         for succ in dag2q['successors'][if_else_node_id]:
            #             dag_block2q['successors'][leaf_node].add(succ)
            #             dag_block2q['predecessors'][succ].add(leaf_node)

            dag_block2q['node_data'] = dag_block2q['node_data'] | dag2q['node_data']
            dag_block['node_data'] = dag_block['node_data'] | dag['node_data']

            qlosure = Qlosure(self.backend_config,
                              with_circuit=self.with_circuit)
            swap_count, depth, good_initial_mapping = qlosure.run(
                dag_block, dag_block2q, heuristic_method="Qlosure",
                initial_mapping_method="custom", initial_mapping=initial_mapping)

            mapping_possibilities[block_idx] = qlosure.mapping_dict
            branch_traces[block_idx] = qlosure.get_structured_trace()

        self.update_front_layer([if_else_node_id])

        logical_qubits = [q for gate in self.front_layer for q in self.node_data[gate]['qubits_accessed'] if len(
            self.node_data[gate]['qubits_accessed']) == 2 and not self.node_data[gate]['is_control_flow']]
        physical_qubits = set(self.mapping_dict[q] for q in logical_qubits)

        extended_layer, extended_layer_index = create_leveled_extended_successor_set(
            self.front_layer, self.dag_successors2q, len(
                physical_qubits)*5
        )

        mapping_scores = {}
        for block_idx, mapping in mapping_possibilities.items():

            mapping_scores[block_idx] = evaluate_mapping_quality(self.front_layer, extended_layer, mapping, self.distance_matrix,
                                                                 self.node_data, self.decay_parameter, self.dag_dependencies_count, extended_layer_index)

        best_block_idx = min(mapping_scores, key=mapping_scores.get)

        best_mapping = mapping_possibilities[best_block_idx]
        swaps_sequences_for_each_possibility = {}
        for block_idx, mapping in mapping_possibilities.items():
            if block_idx == best_block_idx:
                swaps_sequences_for_each_possibility[block_idx] = []
                continue

            _, _, swaps = solve_token_swapping(
                self.backend_connections,
                mapping,
                best_mapping,
            )

            swaps_sequences_for_each_possibility[block_idx] = swaps

            # Build a reverse map for this branch to recover logical labels
            branch_rev = {p: l for l, p in enumerate(mapping)}

            # Append swap trace entries at the END of the branch body
            body = branch_traces.get(block_idx, [])
            for (p0, p1) in swaps:
                l0 = branch_rev[p0]
                l1 = branch_rev[p1]
                body.append({
                    "type": "swap",
                    "logical_qubits": [l0, l1],
                    "physical_qubits": [p0, p1],
                })

        labels = []
        for idx, block_idx in enumerate(sorted(k for k in branch_traces.keys() if k != -1)):
            if idx == 0:
                labels.append(("then", block_idx))
            elif idx == 1:
                labels.append(("else", block_idx))
            else:
                labels.append((f"else{idx}", block_idx))

        branches_payload = []
        for label, idx in labels:
            branches_payload.append({
                "label": label,
                "body": branch_traces[idx]
            })

        # Map reconciliation swaps to logical pairs for readability
        rec_swaps_pretty = {}
        for label, idx in labels:
            pairs = []
            for (p0, p1) in swaps_sequences_for_each_possibility.get(idx, []):
                l0 = self.reverse_mapping_dict[p0]
                l1 = self.reverse_mapping_dict[p1]
                pairs.append([l0, l1])
            rec_swaps_pretty[label] = pairs

        self._append_if_block(branches=branches_payload,
                              reconciliation_swaps={})

        self.mapping_dict = copy.deepcopy(best_mapping)
        self.reverse_mapping_dict = {
            p: l for l, p in enumerate(self.mapping_dict)}

        return 0, 0

    # ---------- TRACE UTILITIES ----------
    def _current_trace(self) -> List[Dict[str, Any]]:
        return self._trace_stack[-1]

    def _push_block(self):
        self._trace_stack.append([])

    def _pop_block(self) -> List[Dict[str, Any]]:
        return self._trace_stack.pop()

    def _append_gate_entry(self, op: str, logical_qubits: List[int], physical_qubits: List[int], node_id: Optional[int] = None):
        self._current_trace().append({
            "type": "gate",
            "op": op,
            "logical_qubits": logical_qubits,
            "physical_qubits": physical_qubits,
            "node_id": node_id
        })

    def _append_swap_entry(self, logical_pair: List[int], physical_pair: List[int]):
        self._current_trace().append({
            "type": "swap",
            "logical_qubits": logical_pair,
            "physical_qubits": physical_pair
        })

    def _append_for_block(self, iterations: int, body: List[Dict[str, Any]]):
        self._current_trace().append({
            "type": "for",
            "iterations": iterations,
            "body": body
        })

    def _append_while_block(self, condition: str, body: List[Dict[str, Any]]):
        self._current_trace().append({
            "type": "while",
            "condition": condition,
            "body": body
        })

    def _append_if_block(self, branches: List[Dict[str, Any]], reconciliation_swaps: Dict[str, List[List[int]]] = None):
        self._current_trace().append({
            "type": "if",
            "branches": branches,
            "reconciliation_swaps": reconciliation_swaps or {}
        })

    def get_structured_trace(self) -> List[Dict[str, Any]]:
        """Return the top-level structured trace."""
        return self._trace_stack[0]

    # Pretty printing with global numbering and indentation
    def format_structured_trace(self, trace: Optional[List[Dict[str, Any]]] = None, indent: int = 0) -> str:
        if trace is None:
            trace = self.get_structured_trace()

        lines = []
        pad = "  " * indent

        def qlabel(qs: List[int]) -> str:
            # Logical qubit labels for display (q0, q1, ...)
            return ", ".join(f"q{q}" for q in qs)

        for item in trace:
            if item["type"] == "gate":
                self._global_gate_counter += 1
                op = item["op"]
                pqs = item["physical_qubits"]
                lines.append(
                    f"{pad}gate {self._global_gate_counter} ({op}, {qlabel(pqs)})")

            elif item["type"] == "swap":
                self._global_gate_counter += 1
                pqs = item["physical_qubits"]
                lines.append(
                    f"{pad}gate {self._global_gate_counter} (swap, {qlabel(pqs)})")

            elif item["type"] == "for":
                lines.append(f"{pad}for (iterations={item['iterations']}) {{")
                lines.append(self.format_structured_trace(
                    item["body"], indent + 1))
                lines.append(f"{pad}}}")
            elif item["type"] == "while":
                lines.append(f"{pad}while (condition={item['condition']}) {{")
                lines.append(self.format_structured_trace(
                    item["body"], indent + 1))
                lines.append(f"{pad}}}")

            elif item["type"] == "if":
                lines.append(f"{pad}if {{")
                for idx, br in enumerate(item["branches"]):
                    label = br.get("label", f"branch{idx}")
                    lines.append(f"{pad}  // {label}")
                    lines.append(self.format_structured_trace(
                        br["body"], indent + 1))
                lines.append(f"{pad}}}")
                # Optionally show reconciliation swaps info (commented style)
                rec = item.get("reconciliation_swaps") or {}
                for k, swaps in rec.items():
                    if swaps:
                        formatted = ", ".join(
                            [f"(q{a}, q{b})" for a, b in swaps])
                        lines.append(
                            f"{pad}// reconciliation_swaps[{k}]: {formatted}")
        return "\n".join(lines)
