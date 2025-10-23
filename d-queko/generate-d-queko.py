#!/usr/bin/env python3
"""
d-QUEKO benchmark generator (structure-fixed, leafs-vary-per-replicate)
========================================================================

What this does
--------------
Builds OpenQASM 3 programs with structured control flow (IF / FOR / WHILE)
composed of QUEKO-style "leaf" blocks. For each *benchmark*:

  • The control-flow *structure* (which nodes appear where) and *all arguments*
    (loop iteration counts, condition qubits, each leaf's depth & subgraph size,
    α, β, allowed gate sets) are fixed and shared across all replicates.

  • The 10 circuits (replicates) differ by **re-sampling the leaves only**:
    each leaf’s subgraph patch and cycle matchings are drawn from a replicate-
    specific RNG, while arguments remain unchanged. Gates can also differ.

Benchmark mode
--------------
In one run:
  • 9 benchmarks by default (top-level lengths 1..9).
  • For each benchmark, 10 circuits are emitted (same structure/args; leafs differ).

Quick start
-----------
# 9 benchmarks (top-level 1..9), 10 circuits each, on an 8x8 grid, using only 28 logical qubits:
python generate_dqueko_bench.py \
  --device grid_8x8 \
  --n-qubits 28 \
  --outdir out/ \
  --seed 7 \
  --emit-metadata

# Single benchmark with 11 top-level blocks, deeper nesting and larger leaves (32 logical qubits from a 10x10 grid):
python generate_dqueko_bench.py \
  --device grid_10x10 \
  --n-qubits 32 \
  --outdir out11/ \
  --seed 42 \
  --bench-depths 11 \
  --nest-depth 3 \
  --child-len 3..5 \
  --leaf-depth 8..12 \
  --leaf-subgraph-size 22..30 \
  --emit-metadata

Outputs
-------
out/
  bench_toplen_01/
    bench.json        # skeleton + args + sample counts (from replicate 0)
    circ_00.qasm      # 10 circuits with same structure/args; leafs differ
    ...
    circ_09.qasm
  ...
  bench_toplen_09/

Notes
-----
• OpenQASM 3 kept simple/portable; includes stdgates.inc.
• If you want the original QUEKO gate sequence semantics, replace the inside
  of LeafSpec.realize()/emit() with your own.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple, Union
from tqdm import tqdm

import networkx as nx
from networkx.algorithms.matching import maximal_matching
import random

# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

GRID_RE = re.compile(r"grid_(\d+)x(\d+)$", re.IGNORECASE)


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def parse_range(s: str, kind=int) -> Tuple[Union[int, float], Union[int, float]]:
    """
    Parse a single value 'x' or a closed range 'a..b' into (lo, hi).
    """
    s = str(s).strip()
    if ".." in s:
        a, b = s.split("..", 1)
        return kind(a), kind(b)
    v = kind(s)
    return v, v


def choose_from_range_rng(rng: random.Random, lo: Union[int, float], hi: Union[int, float], is_int=True):
    if is_int:
        return rng.randint(int(lo), int(hi))
    return rng.uniform(float(lo), float(hi))


def parse_density_pair(s: str) -> Tuple[float, float]:
    """
    Parse 'alpha,beta' -> (alpha, beta). alpha=1q density, beta=2q density per cycle.
    """
    parts = [x.strip() for x in str(s).split(",")]
    if len(parts) != 2:
        raise ValueError("leaf-density must be 'alpha,beta'")
    return float(parts[0]), float(parts[1])


def parse_mix(s: str) -> Dict[str, float]:
    """
    Parse a probability mix like 'leaf:0.5,if:0.2,for:0.2,while:0.1'.
    Missing keys default to 0. Values are normalized to sum to 1.
    """
    base = {"leaf": 0.3, "for": 0.25, "while": 0.25, "if": 0.2, }
    if not s:
        return base
    out = {k: 0.0 for k in base}
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        k, v = part.split(":")
        k = k.strip().lower()
        v = float(v.strip())
        if k not in out:
            raise ValueError(f"unknown mix key '{k}', use leaf/if/for/while")
        out[k] = v
    total = sum(out.values())
    if total <= 0:
        return base
    return {k: out[k]/total for k in out}


def parse_int_list_or_range(s: str) -> List[int]:
    """
    Accepts '1,3,7' or '1..9' and returns a list of ints.
    """
    s = str(s).strip()
    if ".." in s:
        lo, hi = parse_range(s, int)
        if lo > hi:
            lo, hi = hi, lo
        return list(range(int(lo), int(hi) + 1))
    return [int(x.strip()) for x in s.split(",") if x.strip()]

# ---------------------------------------------------------------------------
# Device graphs
# ---------------------------------------------------------------------------


def _make_grid_graph(m: int, n: int) -> nx.Graph:
    """
    Create an m x n 4-neighborhood grid as an undirected graph,
    nodes labeled 0..m*n-1.
    """
    G = nx.Graph()
    def idx(r, c): return r * n + c
    for r in range(m):
        for c in range(n):
            u = idx(r, c)
            if r + 1 < m:
                G.add_edge(u, idx(r + 1, c))
            if c + 1 < n:
                G.add_edge(u, idx(r, c + 1))
    return G


def load_device(backend_name: str) -> nx.Graph:
    """
    Load a device graph:
      • 'grid_MxN' to generate a grid on the fly
      • JSON path with {"coupling_map": [[u,v], ...]}
      • Known names mapped under qpu/topologies/*.json
    """
    # grid_MxN
    m = GRID_RE.match(backend_name)
    if m:
        R, C = int(m.group(1)), int(m.group(2))
        return _make_grid_graph(R, C)

    # direct JSON file path
    if os.path.exists(backend_name) and backend_name.lower().endswith(".json"):
        with open(backend_name, "r") as f:
            data = json.load(f)
        if "coupling_map" not in data:
            raise KeyError("JSON must contain key 'coupling_map'")
        G = nx.Graph()
        for u, v in data["coupling_map"]:
            G.add_edge(int(u), int(v))
        return G

    # known map in qpu/topologies
    TOPOLOGIES_DIR = "qpu/topologies"
    BACKEND_FILE_MAP = {
        "fake_5q_v1": "fake_5q_v1.json",
        "fake_20q_v1": "fake_20q_v1.json",
        "fake_27q_pulse_v1": "fake_27q_pulse_v1.json",
        "fake_127q_pulse_v1": "fake_127q_pulse_v1.json",
        "ibm_brisbane": "ibm_brisbane.json",
        "ibm_kyiv": "ibm_kyiv.json",
        "ibm_sherbrooke": "ibm_sherbrooke.json",
        "ankaa": "Ankaa-3.json",
        "imb_sherbrooke2X": "IBM_sherbrooke2x.json",
    }
    if backend_name in BACKEND_FILE_MAP:
        file_path = os.path.join(
            TOPOLOGIES_DIR, BACKEND_FILE_MAP[backend_name])
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File '{file_path}' not found.")
        with open(file_path, "r") as f:
            data = json.load(f)
        if "coupling_map" not in data:
            raise KeyError("JSON must contain key 'coupling_map'")
        G = nx.Graph()
        for u, v in data["coupling_map"]:
            G.add_edge(int(u), int(v))
        return G

    raise KeyError(
        f"Unknown device '{backend_name}'. Use grid_MxN, a JSON path, or a known name in qpu/topologies/.")


def generate_dense_backend(n: int, density: float = 0.8, seed: int = 42) -> nx.Graph:

    G = nx.Graph()
    G.add_nodes_from(range(n))

    for i in range(n):
        for j in range(i+1, n):
            if random.random() < density:
                G.add_edge(i, j)

    print(G.edges())

    return G


def relabel_to_zero_based(G: nx.Graph) -> nx.Graph:
    """Relabel graph nodes to 0..n-1 in sorted order for clean QASM indices."""
    nodes_sorted = sorted(G.nodes())
    mapping = {old: i for i, old in enumerate(nodes_sorted)}
    return nx.relabel_nodes(G, mapping, copy=True)


def make_working_device(G_full: nx.Graph, n_qubits: int, rng: random.Random) -> nx.Graph:
    """
    Return a connected, relabeled working device of size n_qubits (<= |G_full|).
    If n_qubits == |G_full|, just relabel to 0..n-1 for cleanliness.
    """
    N = G_full.number_of_nodes()
    if n_qubits <= 0:
        raise ValueError("--n-qubits must be positive.")
    if n_qubits > N:
        raise ValueError(f"--n-qubits ({n_qubits}) exceeds device size ({N}).")
    if n_qubits == N:
        return relabel_to_zero_based(G_full)

    # pick a connected BFS patch then relabel
    patch = bfs_patch(G_full, size=n_qubits, rng=rng)
    H = G_full.subgraph(patch).copy()
    return relabel_to_zero_based(H)

# ---------------------------------------------------------------------------
# Subgraph picking (BFS patch), RNG-aware
# ---------------------------------------------------------------------------


def bfs_patch(
    G: nx.Graph,
    size: int,
    rng: random.Random,
    avoid: Optional[set] = None,
    min_dist: int = 0,
    tries: int = 200
) -> List[int]:
    """
    Pick a connected 'patch' (list of nodes) of the requested size via BFS,
    optionally far from an 'avoid' set by graph distance >= min_dist.
    Uses the provided rng for reproducibility.
    """
    avoid = avoid or set()
    all_nodes = list(G.nodes())
    if len(all_nodes) < size:
        raise ValueError("Requested subgraph size exceeds device size.")

    def ok_distance(patch: List[int]) -> bool:
        if not avoid or min_dist <= 0:
            return True
        for u in patch:
            for v in avoid:
                try:
                    d = nx.shortest_path_length(G, u, v)
                    if d < min_dist:
                        return False
                except nx.NetworkXNoPath:
                    continue
        return True

    for _ in range(tries):
        seed = rng.choice(all_nodes)
        if seed in avoid:
            continue
        visited = {seed}
        q = [seed]
        order = [seed]
        while q and len(order) < size:
            cur = q.pop(0)
            nbrs = list(G.neighbors(cur))
            rng.shuffle(nbrs)
            for nei in nbrs:
                if nei not in visited and nei not in avoid:
                    visited.add(nei)
                    q.append(nei)
                    order.append(nei)
                    if len(order) >= size:
                        break
        if len(order) == size and ok_distance(order):
            return order

    # Fallback (relax distance)
    seed = rng.choice(all_nodes)
    visited = {seed}
    q = [seed]
    order = [seed]
    while q and len(order) < size:
        cur = q.pop(0)
        nbrs = list(G.neighbors(cur))
        rng.shuffle(nbrs)
        for nei in nbrs:
            if nei not in visited:
                visited.add(nei)
                q.append(nei)
                order.append(nei)
                if len(order) >= size:
                    break
    if len(order) != size:
        raise RuntimeError(
            "Failed to construct a BFS patch of requested size.")
    return order

# ---------------------------------------------------------------------------
# Gate sampling helpers (RNG-aware)
# ---------------------------------------------------------------------------


PARAMETRIC_1Q = {"rx", "ry", "rz", "p"}
PARAMETRIC_2Q = {"rzx"}


def format_angle(theta: float) -> str:
    return f"{theta:.8f}"


def sample_angle(rng: random.Random) -> float:
    # small discrete set for readability/reproducibility
    choices = [math.pi/8, 3*math.pi/8, 5*math.pi/8, 7*math.pi/8]
    return rng.choice(choices)


def emit_1q(gate: str, q: int, rng: random.Random) -> str:
    if gate in PARAMETRIC_1Q:
        return f"{gate}({format_angle(sample_angle(rng))}) q[{q}];"
    return f"{gate} q[{q}];"


def emit_2q(gate: str, u: int, v: int, rng: random.Random) -> str:
    if gate in PARAMETRIC_2Q:
        return f"{gate}({format_angle(sample_angle(rng))}) q[{u}], q[{v}];"
    return f"{gate} q[{u}], q[{v}];"

# ---------------------------------------------------------------------------
# IR nodes and context
# ---------------------------------------------------------------------------


@dataclass
class EmitContext:
    n_qubits: int
    bit_alloc: int = 0
    lines: List[str] = field(default_factory=list)
    indent: int = 0
    count_1q: int = 0  # accumulated during realize
    count_2q: int = 0
    # structure+realization metadata
    blocks: List[Dict] = field(default_factory=list)

    def w(self, s: str):
        self.lines.append(("  " * self.indent) + s)

    def alloc_bit(self) -> int:
        b = self.bit_alloc
        self.bit_alloc += 1
        return b


class Node:
    def realize(self, ctx: EmitContext, leaf_rng: random.Random, avoid: Optional[set] = None) -> set:
        """Build per-replicate plans (e.g., leaf patches). Returns set of used device nodes."""
        raise NotImplementedError

    def emit(self, ctx: EmitContext, gates_rng: random.Random):
        """Emit code using the already-realized plan."""
        raise NotImplementedError

# ---------------------------------------------------------------------------
# Leaf specification (arguments fixed by structure RNG; instances vary per replicate)
# ---------------------------------------------------------------------------


class LeafSpec(Node):
    """
    Leaf with *fixed arguments* (depth, alpha, beta, subgraph size, gates sets),
    but *variable instances* per replicate (subgraph patch + matchings).

    depth / size are chosen by the structure RNG and therefore identical across
    all 10 circuits in a benchmark. During `realize` (per replicate) we draw
    a fresh BFS patch and cycle matchings using the replicate's leaf RNG.
    """

    def __init__(
        self,
        device: nx.Graph,
        size: int,
        depth: int,
        alpha: float,
        beta: float,
        gates_1q: Sequence[str],
        gates_2q: Sequence[str],
        conflict_level: int,
        name: str = "leaf",
    ):
        self.device = device
        self.size = int(size)
        self.depth = int(depth)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.gates_1q = list(gates_1q) or ["x", "h", "z"]
        self.gates_2q = list(gates_2q) or ["cx"]
        self.conflict_level = int(conflict_level)
        self.name = name
        if self.depth <= 0:
            raise ValueError("Leaf depth must be positive.")
        if self.size <= 0:
            raise ValueError("Leaf subgraph size must be positive.")
        # Per-replicate plan (rebuilt every realize):
        self._plan_by_cycle: List[List[Tuple[str, int, Optional[int]]]] = []
        self._last_nodes: List[int] = []  # last chosen patch (for metadata)

    def random_maximal_matching(self, G: nx.Graph, nodes, rng: random.Random):
        # Build candidate edges inside the patch and randomize their order
        edges = [(u, v) for u, v in G.subgraph(nodes).edges()]
        rng.shuffle(edges)

        used = set()
        chosen = []
        for u, v in edges:
            if u in used or v in used:
                continue
            chosen.append((u, v))
            used.add(u)
            used.add(v)
        return chosen  # list of disjoint edges (a randomized maximal matching)

    def realize(self, ctx: EmitContext, leaf_rng: random.Random, avoid: Optional[set] = None) -> set:
        # Reset plan for this replicate
        self._plan_by_cycle = []
        self._last_nodes = bfs_patch(
            self.device, size=self.size, rng=leaf_rng,
            avoid=(avoid or set()), min_dist=self.conflict_level
        )
        patchG = self.device.subgraph(self._last_nodes).copy()

        n = len(self._last_nodes)
        target_2q = max(0, int(round(self.beta * n / 2.0)))
        target_1q = max(0, int(round(self.alpha * n)))
        if target_1q == 0 and target_2q == 0:
            target_1q = 1

        for _ in range(self.depth):
            cycle_ops: List[Tuple[str, int, Optional[int]]] = []
            temp = nx.Graph(patchG)
            matching = list(self.random_maximal_matching(
                temp, self._last_nodes, leaf_rng))
            leaf_rng.shuffle(matching)
            chosen_2q = matching[:target_2q]

            used = set()
            for (u, v) in chosen_2q:
                used.add(u)
                used.add(v)
                cycle_ops.append(("2q", int(u), int(v)))
            ctx.count_2q += len(chosen_2q)

            free_nodes = [q for q in self._last_nodes if q not in used]
            leaf_rng.shuffle(free_nodes)
            k = min(target_1q, len(free_nodes))
            if len(chosen_2q) == 0 and k == 0 and free_nodes:
                k = 1
            for idx in range(k):
                cycle_ops.append(("1q", int(free_nodes[idx]), None))
            ctx.count_1q += k

            self._plan_by_cycle.append(cycle_ops)

        # Record per-replicate metadata
        ctx.blocks.append({
            "type": "leaf",
            "name": self.name,
            "n_qubits": n,
            "subgraph_nodes": self._last_nodes,
            "depth": self.depth,
            "alpha": self.alpha,
            "beta": self.beta,
            "gates_1q": self.gates_1q,
            "gates_2q": self.gates_2q,
        })
        return set(self._last_nodes)

    def emit(self, ctx: EmitContext, gates_rng: random.Random):
        ctx.w(
            f"// --- BEGIN LEAF: {self.name} (depth={self.depth}, size={self.size})")
        for ops in self._plan_by_cycle:
            for op in ops:
                if op[0] == "2q":
                    u, v = op[1], op[2]
                    gate2 = gates_rng.choice(self.gates_2q)
                    ctx.w(emit_2q(gate2, u, v, gates_rng))
                else:
                    q = op[1]
                    gate1 = gates_rng.choice(self.gates_1q)
                    ctx.w(emit_1q(gate1, q, gates_rng))
        ctx.w(f"// --- END LEAF: {self.name}")

# ---------------------------------------------------------------------------
# Composite nodes
# ---------------------------------------------------------------------------


class Seq(Node):
    def __init__(self, *children: Node):
        self.children = list(children)

    def realize(self, ctx: EmitContext, leaf_rng: random.Random, avoid: Optional[set] = None) -> set:
        used_total: set = set()
        for ch in self.children:
            # no cross-segment avoidance
            used = ch.realize(ctx, leaf_rng, avoid=None)
            used_total |= (used or set())
        return used_total

    def emit(self, ctx: EmitContext, gates_rng: random.Random):
        for ch in self.children:
            ch.emit(ctx, gates_rng)


class IfElse(Node):
    def __init__(self, cond_qubit: int, then_blk: Node, else_blk: Node, conflict_level: int, name: str = "ifelse"):
        self.cond_qubit = int(cond_qubit)
        self.then_blk = then_blk
        self.else_blk = else_blk
        self.name = name
        self.cond_bit: Optional[int] = None
        self.conflict_level = int(conflict_level)

    def realize(self, ctx: EmitContext, leaf_rng: random.Random, avoid: Optional[set] = None) -> set:
        self.cond_bit = ctx.alloc_bit()
        used_then = self.then_blk.realize(ctx, leaf_rng, avoid=avoid)
        avoid_else = (avoid or set()).union(used_then or set())
        used_else = self.else_blk.realize(ctx, leaf_rng, avoid=avoid_else)
        ctx.blocks.append({
            "type": "ifelse",
            "name": self.name,
            "cond_qubit": self.cond_qubit,
            "cond_bit": self.cond_bit,
        })
        return (used_then or set()) | (used_else or set())

    def emit(self, ctx: EmitContext, gates_rng: random.Random):
        assert self.cond_bit is not None
        ctx.w(f"// --- IF/ELSE: {self.name}")
        ctx.w(f"measure q[{self.cond_qubit}] -> c[{self.cond_bit}];")
        ctx.w(f"if (c[{self.cond_bit}] == true) {{")
        ctx.indent += 1
        self.then_blk.emit(ctx, gates_rng)
        ctx.indent -= 1
        ctx.w("} else {")
        ctx.indent += 1
        self.else_blk.emit(ctx, gates_rng)
        ctx.indent -= 1
        ctx.w("}")
        ctx.w(f"// --- END IF/ELSE: {self.name}")


class ForLoop(Node):
    def __init__(self, K: int, body: Node, name: str = "forloop"):
        self.K = int(K)
        self.body = body
        self.name = name

    def realize(self, ctx: EmitContext, leaf_rng: random.Random, avoid: Optional[set] = None) -> set:
        used = self.body.realize(ctx, leaf_rng, avoid=avoid)
        ctx.blocks.append({"type": "for", "name": self.name, "iters": self.K})
        return used or set()

    def emit(self, ctx: EmitContext, gates_rng: random.Random):
        ctx.w(f"// --- FOR: {self.name}")
        ctx.w(f"for int i in [0:{self.K}] {{")
        ctx.indent += 1
        self.body.emit(ctx, gates_rng)
        ctx.indent -= 1
        ctx.w("}")
        ctx.w(f"// --- END FOR: {self.name}")


class WhileLoop(Node):
    """
    A while loop that repeats a body until a measurement on `cond_qubit`
    writes a 1 to classical bit c[cond_bit].
    """

    def __init__(self, cond_qubit: int, body: Node, name: str = "whileloop"):
        self.cond_qubit = int(cond_qubit)
        self.body = body
        self.name = name
        self.cond_bit: Optional[int] = None

    def realize(self, ctx: EmitContext, leaf_rng: random.Random, avoid: Optional[set] = None) -> set:
        self.cond_bit = ctx.alloc_bit()
        used = self.body.realize(ctx, leaf_rng, avoid=avoid)
        ctx.blocks.append({
            "type": "while",
            "name": self.name,
            "cond_qubit": self.cond_qubit,
            "cond_bit": self.cond_bit
        })
        return used or set()

    def emit(self, ctx: EmitContext, gates_rng: random.Random):
        assert self.cond_bit is not None
        ctx.w(f"// --- WHILE: {self.name}")
        ctx.w(f"measure q[{self.cond_qubit}] -> c[{self.cond_bit}];")
        ctx.w(f"while (c[{self.cond_bit}] == false) {{")
        ctx.indent += 1
        self.body.emit(ctx, gates_rng)
        ctx.w(f"measure q[{self.cond_qubit}] -> c[{self.cond_bit}];")
        ctx.indent -= 1
        ctx.w("}")
        ctx.w(f"// --- END WHILE: {self.name}")

# ---------------------------------------------------------------------------
# Program builder (top-level path + nesting)
# ---------------------------------------------------------------------------


@dataclass
class GenParams:
    nbQubits: int
    device: nx.Graph
    seed: int
    # Leaf defaults
    leaf_depth_rng: Tuple[int, int]
    leaf_density: Tuple[float, float]
    gates_1q: List[str]
    gates_2q: List[str]
    # Structure knobs
    # number of top-level blocks (your "very high level" nodes)
    top_len: int
    nest_depth: int                 # max nesting levels below top level
    child_len_rng: Tuple[int, int]  # number of blocks in each nested Seq
    # probability mix among {leaf, if, for, while}
    mix: Dict[str, float]
    for_iters_rng: Tuple[int, int]  # iteration count range for FOR
    conflict_level: int             # min graph distance between IF branches
    emit_metadata: bool


class ProgramBuilder:
    def __init__(self, params: GenParams):
        self.P = params
        self.G = params.device
        # One RNG for structure only (fixed across replicates)
        self.rng_shape = random.Random(self.P.seed)

    def _rand_cond_qubit(self) -> int:
        return self.rng_shape.randrange(0, self.G.number_of_nodes())

    def _choose_type(self, parent_type: Optional[str]) -> str:
        r = self.rng_shape.random()

        node_types = ["leaf", "if", "for", "while"]

        acc = 0.0
        probs = self.P.mix
        if parent_type == "if":
            probs = probs.copy()
            for node_type in node_types:
                if node_type == "if":
                    probs[node_type] = 0.0
                else:
                    probs[node_type] /= (1.0 - self.P.mix.get("if", 0.0))
        elif parent_type == "while":
            probs = probs.copy()
            for node_type in node_types:
                if node_type == "while":
                    probs[node_type] = 0.0
                else:
                    probs[node_type] /= (1.0 - self.P.mix.get("while", 0.0))
        elif parent_type == "for":
            probs = probs.copy()
            for node_type in node_types:
                if node_type == "for":
                    probs[node_type] = 0.0
                else:
                    probs[node_type] /= (1.0 - self.P.mix.get("for", 0.0))
        for k in node_types:
            acc += probs.get(k, 0.0)
            if r <= acc:
                if k == "if":
                    print("choosing IF")
                return k
        return "leaf"

    def _make_leaf_spec(self, name: str) -> LeafSpec:
        size = self.P.nbQubits
        depth = choose_from_range_rng(
            self.rng_shape, *self.P.leaf_depth_rng, True)
        alpha, beta = self.P.leaf_density
        return LeafSpec(
            device=self.G,
            size=size,
            depth=depth,
            alpha=alpha,
            beta=beta,
            gates_1q=self.P.gates_1q,
            gates_2q=self.P.gates_2q,
            conflict_level=self.P.conflict_level,
            name=name
        )

    def _build_nested_seq(self, level: int, tag: str, parent_type: Optional[str]) -> Node:
        """
        Build a Seq of child_len blocks at nesting 'level' (1..nest_depth).
        Structure & all arguments are frozen by rng_shape here.
        """
        if level > self.P.nest_depth:
            return self._make_leaf_spec(f"Leaf_L{level}_{tag}")
        m = choose_from_range_rng(self.rng_shape, *self.P.child_len_rng, True)
        if m <= 0:
            m = 1
        kids: List[Node] = []
        for j in range(m):
            t = self._choose_type(parent_type=parent_type)
            if t == "leaf":
                kids.append(self._make_leaf_spec(f"Leaf_L{level}_{tag}_{j}"))
            elif t == "if":
                cq = self._rand_cond_qubit()
                then_blk = self._build_nested_seq(
                    level + 1, f"{tag}_{j}T", parent_type=t)
                else_blk = self._build_nested_seq(
                    level + 1, f"{tag}_{j}E", parent_type=t)
                kids.append(IfElse(cq, then_blk, else_blk, conflict_level=self.P.conflict_level,
                                   name=f"If_L{level}_{tag}_{j}"))
            elif t == "for":
                K = choose_from_range_rng(
                    self.rng_shape, *self.P.for_iters_rng, True)
                body = self._build_nested_seq(
                    level + 1, f"{tag}_{j}F", parent_type=t)
                kids.append(ForLoop(K, body, name=f"For_L{level}_{tag}_{j}"))
            else:  # while
                cq = self._rand_cond_qubit()
                body = self._build_nested_seq(
                    level + 1, f"{tag}_{j}W", parent_type=t)
                kids.append(
                    WhileLoop(cq, body, name=f"While_L{level}_{tag}_{j}"))
        return Seq(*kids)

    def build_top(self) -> Node:
        """
        Build exactly `top_len` top-level blocks (this is your "very high level"
        longest path). Nested content does not count toward this.
        """
        kids: List[Node] = []
        for i in range(self.P.top_len):
            t = self._choose_type(parent_type=None)
            if t == "leaf":
                kids.append(self._make_leaf_spec(f"TopLeaf_{i}"))
            elif t == "if":
                cq = self._rand_cond_qubit()
                then_blk = self._build_nested_seq(1, f"Top{i}T", parent_type=t)
                else_blk = self._build_nested_seq(1, f"Top{i}E", parent_type=t)
                kids.append(IfElse(cq, then_blk, else_blk, conflict_level=self.P.conflict_level,
                                   name=f"TopIf_{i}"))
            elif t == "for":
                K = choose_from_range_rng(
                    self.rng_shape, *self.P.for_iters_rng, True)
                body = self._build_nested_seq(1, f"Top{i}F", parent_type=t)
                kids.append(ForLoop(K, body, name=f"TopFor_{i}"))
            elif t == "while":
                cq = self._rand_cond_qubit()
                body = self._build_nested_seq(1, f"Top{i}W", parent_type=t)
                kids.append(WhileLoop(cq, body, name=f"TopWhile_{i}"))
        return Seq(*kids)

# ---------------------------------------------------------------------------
# Emission helpers
# ---------------------------------------------------------------------------


def emit_program(root: Node, n_qubits: int, ctx: EmitContext, gates_rng: random.Random) -> str:
    """
    Emit OpenQASM 3 text assuming `root.realize(...)` has already been called.
    """
    header = []
    header.append("OPENQASM 3;")
    header.append('include "stdgates.inc";')
    header.append(f"qubit[{n_qubits}] q;")
    header.append(f"bit[{max(1, ctx.bit_alloc)}] c;")
    header.append("")
    root.emit(ctx, gates_rng)
    return "\n".join(header + ctx.lines)


def count_nodes(root: Node) -> Dict[str, int]:
    """
    Count node types in the skeleton for metadata (structure-only).
    """
    cnt = {"leaf": 0, "if": 0, "for": 0, "while": 0}

    def visit(n: Node):
        nonlocal cnt
        if isinstance(n, LeafSpec):
            cnt["leaf"] += 1
        elif isinstance(n, IfElse):
            cnt["if"] += 1
            visit(n.then_blk)
            visit(n.else_blk)
        elif isinstance(n, ForLoop):
            cnt["for"] += 1
            visit(n.body)
        elif isinstance(n, WhileLoop):
            cnt["while"] += 1
            visit(n.body)
        elif isinstance(n, Seq):
            for ch in n.children:
                visit(ch)
    visit(root)
    return cnt


def count_bits_from_skeleton(root: Node) -> int:
    """
    Classical bits needed (one per IF/ELSE and WHILE).
    """
    c = count_nodes(root)
    return c["if"] + c["while"]

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="d-QUEKO: structure-fixed benchmark generator; leafs vary per replicate")

    # I/O & reproducibility
    # p.add_argument("--device", type=str, required=True,
    #                help="grid_MxN, a JSON path, or known name in qpu/topologies/")
    p.add_argument("--outdir", type=str, default="benchmarks",
                   help="Output directory to place benchmarks")
    p.add_argument("--seed", type=int, default=1,
                   help="Seed for the STRUCTURE RNG (arguments and shape are frozen per benchmark)")
    p.add_argument("--emit-metadata", action="store_true",
                   help="Write bench.json with structure, args and sample counts")

    # NEW: logical qubit count used for this benchmark
    p.add_argument("--n-qubits", type=int, default=None,
                   help="Number of qubits to use in the generated circuits. "
                        "If smaller than the device, a connected subgraph of this size is selected and relabeled to 0..n-1. "
                        "If omitted, uses the full device size.")

    # Structure (top level + nesting)

    p.add_argument("--replicates", type=int, default=10,
                   help="Circuits per benchmark that share structure/args (leafs differ)")
    p.add_argument("--top-len", type=int, default=10,
                   help=" generate a benchmark with this top-level length ")
    p.add_argument("--nest-depth", type=int, default=2,
                   help="Max nesting levels below the top level (0 = no nesting)")
    p.add_argument("--child-len", type=str, default="2",
                   help="INT or A..B: number of blocks inside each nested Seq")
    p.add_argument("--mix", type=str, default="leaf:0.6,if:0.2,for:0.15,while:0.05",
                   help="Probability mix among node types; normalized if needed")

    # FOR loop iterations
    p.add_argument("--for-iters", type=str, default="4",
                   help="INT or A..B for the number of FOR iterations")

    # Leaf params (arguments are frozen per leaf across replicates)
    p.add_argument("--leaf-depth", type=str, default="8",
                   help="INT or A..B cycles per leaf (frozen per leaf)")
    p.add_argument("--leaf-density", type=str, default="0.5,0.5",
                   help="alpha,beta per cycle (alpha=1q/node; beta=2q/node)")
    p.add_argument("--gates-1q", type=str, default="x,h,z,rx,rz",
                   help="Comma list of allowed 1q gates (rx/ry/rz/p are parametric)")
    p.add_argument("--gates-2q", type=str, default="cx,cz",
                   help="Comma list of allowed 2q gates (rzx is parametric)")

    # Optional geometric separation for IF branches
    p.add_argument("--conflict-level", type=int, default=2,
                   help="Min graph distance between sibling branch patches")

    return p

# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def main():
    args = build_argparser().parse_args()

    # Parse structure knobs
    child_len_rng = parse_range(args.child_len, int)
    for_iters_rng = parse_range(args.for_iters, int)
    mix = parse_mix(args.mix)

    # Parse leaf knobs
    leaf_depth_rng = parse_range(args.leaf_depth, int)
    leaf_density = parse_density_pair(args.leaf_density)
    gates_1q = [g.strip() for g in args.gates_1q.split(",") if g.strip()]
    gates_2q = [g.strip() for g in args.gates_2q.split(",") if g.strip()]

    # Prepare output
    ensure_dir(args.outdir)

    # Deterministic per-benchmark selection of a connected subgraph (if needed)
    nQ_target = int(args.n_qubits)
    G = generate_dense_backend(nQ_target, density=.6)
    nQ = G.number_of_nodes()

    top_len = int(args.top_len)

    bench_dir = os.path.join(
        args.outdir, f"queko-{int(nQ):03d}qbt_nest_{int(args.nest_depth):02d}_nodes{int(top_len):03d}_leaf-depth-{leaf_depth_rng[0]}")
    ensure_dir(bench_dir)

    # Build skeleton once (structure RNG decides everything structural + leaf arguments)
    P = GenParams(
        nbQubits=nQ,
        device=G,
        # vary by top_len but deterministic
        seed=int(args.seed) + 101 * int(top_len),
        leaf_depth_rng=leaf_depth_rng,
        leaf_density=leaf_density,
        gates_1q=gates_1q,
        gates_2q=gates_2q,
        top_len=int(top_len),
        nest_depth=int(args.nest_depth),
        child_len_rng=child_len_rng,
        mix=mix,
        for_iters_rng=for_iters_rng,
        conflict_level=int(args.conflict_level),
        emit_metadata=bool(args.emit_metadata),
    )

    builder = ProgramBuilder(P)
    root = builder.build_top()

    # Metadata: skeleton-only counts and bit count (fixed across replicates)
    skeleton_counts = count_nodes(root)
    bit_count = count_bits_from_skeleton(root)

    circ_paths: List[str] = []

    # Produce replicates: each has different leaf instances (leaf RNG), optional different gates RNG
    for rep in tqdm(range(int(args.replicates))):
        leaf_seed = (int(args.seed) * 1000003) ^ (int(top_len)
                                                  * 911) ^ (rep * 9721) ^ 0xA53
        gates_seed = (int(args.seed) * 1000003) ^ (int(top_len)
                                                   * 911) ^ (rep * 9721) ^ 0x5A3
        leaf_rng = random.Random(leaf_seed)
        gates_rng = random.Random(gates_seed)

        ctx = EmitContext(n_qubits=nQ)
        # Allocate bits deterministically by traversing the skeleton;
        # leaf realizations use replicate-specific leaf_rng.
        root.realize(ctx, leaf_rng)

        # Sanity: ensure bit count matches skeleton expectation
        if ctx.bit_alloc != bit_count:
            raise RuntimeError(
                f"bit_alloc mismatch: skeleton={bit_count}, realized={ctx.bit_alloc}"
            )

        qasm = emit_program(root, n_qubits=nQ, ctx=ctx,
                            gates_rng=gates_rng)
        circ_path = os.path.join(bench_dir, f"circ_{rep:02d}.qasm")
        with open(circ_path, "w") as f:
            f.write(qasm)
        circ_paths.append(circ_path)

        # For the first replicate, keep a snapshot of counts for metadata
        if rep == 0:
            sample_counts = {
                "n_bits": max(1, ctx.bit_alloc),
                "total_1q": ctx.count_1q,
                "total_2q": ctx.count_2q,
                "n_leaves": sum(1 for b in ctx.blocks if b.get("type") == "leaf"),
            }

    # Metadata per benchmark (structure and args only + sample counts from rep 0)
    if P.emit_metadata:
        meta = {
            "n_qubits": nQ,                     # logical/working qubits used
            "top_len": int(top_len),
            "nest_depth": int(args.nest_depth),
            "child_len_rng": [int(child_len_rng[0]), int(child_len_rng[1])],
            "for_iters_rng": [int(for_iters_rng[0]), int(for_iters_rng[1])],
            "mix": P.mix,
            "leaf_args": {
                "depth_rng": [int(leaf_depth_rng[0]), int(leaf_depth_rng[1])],
                "density": {"alpha_1q": leaf_density[0], "beta_2q": leaf_density[1]},
                "gates_1q": gates_1q,
                "gates_2q": gates_2q,
            },
            "skeleton_counts": skeleton_counts,
            "bit_count": bit_count,
            "replicates": int(args.replicates),
            "circuits": [os.path.basename(p) for p in circ_paths],
            "args": {
                "seed": int(args.seed),
                "device": args.device,
                "n_qubits": int(nQ),
                "conflict_level": int(args.conflict_level),
            },
            "sample_counts_rep0": sample_counts,
        }
        with open(os.path.join(bench_dir, "bench.json"), "w") as f:
            json.dump(meta, f, indent=2)

        print(
            f"[OK] bench toplen={int(top_len):2d} (n_qubits={nQ}) -> {len(circ_paths)} circuits in {bench_dir}")


if __name__ == "__main__":
    main()
