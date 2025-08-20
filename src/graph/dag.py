from collections import defaultdict
from typing import Dict, List, Set, DefaultDict, Optional
from tqdm import tqdm
import copy
import numpy as np
import time


class DAG:
    def __init__(
        self,
        num_qubits: int,
        read_dependencies: Dict[int, List[int]],
        write_dependencies: Dict[int, List[int]],
        enforce_read_after_read: Optional[bool] = True,
        transitive_reduction: bool = False,
    ) -> None:

        self.num_qubits = num_qubits
        self.read_dependencies = read_dependencies
        self.schedule = sorted(read_dependencies.keys())

        self.access2q = []

        self.predecessors_full: DefaultDict[int, Set[int]] = defaultdict(set)
        self.successors_full: DefaultDict[int, Set[int]] = defaultdict(set)

        self.enforce_read_after_read = enforce_read_after_read
        self.write_dependencies = write_dependencies

        self._build_edges_full()

        self.predecessors_2q: DefaultDict[int, Set[int]] = defaultdict(set)
        self.successors_2q: DefaultDict[int, Set[int]] = defaultdict(set)

        self._build_edges_2q()

        if transitive_reduction:
            # self._transitive_reduction_2q()
            self.transitive_reduction_2q_bitset()

    def _build_edges_full(self) -> None:

        latest_writer_for_qubit = {}
        active_readers_for_qubit = {}
        read_since_writer_for_qubit = {}

        for node in self.schedule:
            write_qubits = self.write_dependencies[node]
            read_qubits = [
                qubit for qubit in self.read_dependencies[node] if qubit not in write_qubits]

            for q in read_qubits:
                if q in latest_writer_for_qubit:
                    writer_node = latest_writer_for_qubit[q]
                    if writer_node is not None and writer_node != node:
                        self.successors_full[writer_node].add(node)
                        self.predecessors_full[node].add(writer_node)

                # Also handle READ-AFTER-READ (RAR) if enforce_read_after_read=True
                if self.enforce_read_after_read and q in active_readers_for_qubit:
                    # All existing readers of q must precede this new read
                    for old_reader_node in active_readers_for_qubit[q]:
                        if old_reader_node != node:
                            self.successors_full[old_reader_node].add(node)
                            self.predecessors_full[node].add(old_reader_node)

                    # If we do NOT allow parallel reads, then once we add edges
                    # from old readers, we can clear them because the new read
                    # becomes the "latest" read. This ensures sequential reading:
                    active_readers_for_qubit[q].clear()

                # Now record that this node is actively reading q
                if q not in active_readers_for_qubit:
                    active_readers_for_qubit[q] = set()
                active_readers_for_qubit[q].add(node)
                read_since_writer_for_qubit[q] = True

            #
            # 2) WRITE-AFTER-WRITE (WAW): If node writes qubit q, it depends on the latest writer of q.
            #
            # 3) WRITE-AFTER-READ (WAR): If node writes qubit q, it depends on all *active readers* of q.
            #
            for q in write_qubits:
                # (a) WAW
                if q in latest_writer_for_qubit:
                    old_writer = latest_writer_for_qubit[q]
                    if old_writer is not None and old_writer != node:
                        if not read_since_writer_for_qubit.get(q, False):
                            self.successors_full[old_writer].add(node)
                            self.predecessors_full[node].add(old_writer)

                # (b) WAR
                if q in active_readers_for_qubit:
                    for old_reader_node in active_readers_for_qubit[q]:
                        if old_reader_node != node:
                            self.successors_full[old_reader_node].add(node)
                            self.predecessors_full[node].add(old_reader_node)

                    # Once we write, we effectively overwrite the old data,
                    # so any active readers of q are now outdated:
                    active_readers_for_qubit[q].clear()

                # (c) This node becomes the latest writer of q
                latest_writer_for_qubit[q] = node
                read_since_writer_for_qubit[q] = False

    def _build_edges_2q(self) -> None:
        """Build the 2-qubit DAG by collapsing out single-qubit nodes 
        from the full DAG structure."""
        two_qubit_nodes = [node for node in self.schedule if len(
            self.read_dependencies[node]) == 2]
        max_node = max(two_qubit_nodes) if two_qubit_nodes else 1
        self.access2q = [[]]*(max_node+1)
        for node in two_qubit_nodes:
            q1, q2 = self.read_dependencies[node]
            self.access2q[node] = [q1, q2]

        two_qubit_set = set(two_qubit_nodes)

        self.successors_2q = defaultdict(set)
        self.predecessors_2q = defaultdict(set)

        for n in two_qubit_nodes:

            queue = list(self.successors_full[n])
            visited = set()

            while queue:
                x = queue.pop(0)
                if x in visited:
                    continue
                visited.add(x)

                if x in two_qubit_set:
                    self.successors_2q[n].add(x)
                    self.predecessors_2q[x].add(n)
                else:
                    queue.extend(self.successors_full[x])

    def _transitive_reduction_2q(self) -> None:
        order = self.schedule  # topological order
        reachable = {node: set() for node in order}
        cpt = 0
        for u in reversed(order):
            if u not in self.predecessors_2q and u not in self.successors_2q:
                continue

            new_succ = set()
            for v in self.successors_2q[u]:
                if v in reachable[u]:
                    cpt += 1
                    continue
                else:
                    new_succ.add(v)
                    reachable[u].update(reachable[v])
                    reachable[u].add(v)

            self.successors_2q[u] = new_succ

        self.predecessors_2q = defaultdict(set)
        for u, sucs in self.successors_2q.items():
            for v in sucs:
                self.predecessors_2q[v].add(u)

    def transitive_reduction_2q_bitset(self):

        order = self.schedule
        index_of = {}
        for i, node in enumerate(order):
            index_of[node] = i

        n = len(order)

        successors_indexed = [[] for _ in range(n)]
        for u, sucset in self.successors_2q.items():
            u_i = index_of[u]
            successors_indexed[u_i] = [index_of[v] for v in sucset]

        bit_reach = [0] * n

        skipped_edges = 0

        for u_i in reversed(range(n)):
            new_succ = []
            for v_i in successors_indexed[u_i]:
                if bit_reach[u_i] & (1 << v_i):
                    skipped_edges += 1
                    continue

                new_succ.append(v_i)
                bit_reach[u_i] |= bit_reach[v_i]
                bit_reach[u_i] |= (1 << v_i)

            successors_indexed[u_i] = new_succ

        new_successors = defaultdict(set)
        for node, u_i in index_of.items():
            new_successors[node] = {order[v_i]
                                    for v_i in successors_indexed[u_i]}

        self.successors_2q = new_successors

        # STEP 5: Rebuild predecessors_2q from successors_2q
        new_predecessors = defaultdict(set)
        for u, sucs in self.successors_2q.items():
            for v in sucs:
                new_predecessors[v].add(u)
        self.predecessors_2q = new_predecessors

    def print_dag_full(self) -> None:
        print("=== FULL DAG ===")
        for node in self.schedule:
            succ = self.successors_full[node]
            pred = self.predecessors_full[node]
            print(f"Node {node}: successors={succ}, predecessors={pred}")

    def print_dag_2q(self) -> None:
        print("=== 2Q DAG ===")
        all_2q_nodes = set(self.predecessors_2q.keys()) | set(
            self.successors_2q.keys())
        for node in self.schedule:
            if node not in all_2q_nodes:
                continue
            succ = self.successors_2q[node]
            pred = self.predecessors_2q[node]
            print(f"Node {node}: successors={succ}, predecessors={pred}")
