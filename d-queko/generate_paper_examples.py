#!/usr/bin/env python3
"""
Generate two small D-QUEKO-style example circuits for paper figures:

1. A single WHILE loop containing 5 IF/ELSE blocks.
2. Two nested WHILE loops.

Each example is emitted in a separate output directory.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import random
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
GENERATOR_PATH = SCRIPT_DIR / "generate-d-queko.py"


def load_generator_module():
    spec = importlib.util.spec_from_file_location("d_queko_generator", GENERATOR_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load generator module from {GENERATOR_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def build_params(gen, args, device, seed):
    return gen.GenParams(
        nbQubits=args.n_qubits,
        device=device,
        seed=seed,
        leaf_depth=args.leaf_depth,
        leaf_density=(args.alpha, args.beta),
        gates_1q=[g.strip() for g in args.gates_1q.split(",") if g.strip()],
        gates_2q=[g.strip() for g in args.gates_2q.split(",") if g.strip()],
        top_len=1,
        nest_depth=2,
        child_len_rng=(1, 1),
        mix={"leaf": 1.0, "if": 0.0, "for": 0.0, "while": 0.0},
        for_iters_rng=(1, 1),
        conflict_level=args.conflict_level,
        emit_metadata=args.emit_metadata,
    )


def make_tiny_leaf(gen, builder, name):
    return gen.LeafSpec(
        device=builder.G,
        size=argsafe_int(builder, "leaf_size"),
        depth=argsafe_int(builder, "leaf_block_depth"),
        alpha=builder.P.leaf_density[0],
        beta=builder.P.leaf_density[1],
        gates_1q=builder.P.gates_1q,
        gates_2q=builder.P.gates_2q,
        conflict_level=builder.P.conflict_level,
        name=name,
    )


def argsafe_int(builder, attr_name):
    return int(getattr(builder.P, attr_name))


def build_single_while_with_ifs(gen, builder, num_ifs):
    if_blocks = []
    for idx in range(num_ifs):
        cond_qubit = builder._rand_cond_qubit()
        then_blk = make_tiny_leaf(gen, builder, f"PaperA_If{idx}_Then")
        else_blk = make_tiny_leaf(gen, builder, f"PaperA_If{idx}_Else")
        if_blocks.append(
            gen.IfElse(
                cond_qubit,
                then_blk,
                else_blk,
                conflict_level=builder.P.conflict_level,
                name=f"PaperA_If{idx}",
            )
        )
    while_cond = builder._rand_cond_qubit()
    return gen.WhileLoop(while_cond, gen.Seq(*if_blocks), name="PaperA_While")


def build_two_nested_whiles(gen, builder):
    inner_body = gen.Seq(
        make_tiny_leaf(gen, builder, "PaperB_Inner_Pre"),
        make_tiny_leaf(gen, builder, "PaperB_Inner_Core"),
        make_tiny_leaf(gen, builder, "PaperB_Inner_Post"),
    )
    inner_while = gen.WhileLoop(
        builder._rand_cond_qubit(),
        inner_body,
        name="PaperB_InnerWhile",
    )
    outer_body = gen.Seq(
        make_tiny_leaf(gen, builder, "PaperB_Outer_Pre"),
        inner_while,
        make_tiny_leaf(gen, builder, "PaperB_Outer_Post"),
    )
    return gen.WhileLoop(
        builder._rand_cond_qubit(),
        outer_body,
        name="PaperB_OuterWhile",
    )


def emit_example(gen, root, example_name, outdir, args, seed):
    example_dir = outdir / example_name
    example_dir.mkdir(parents=True, exist_ok=True)

    skeleton_counts = gen.count_nodes(root)
    bit_count = gen.count_bits_from_skeleton(root)
    circuit_files = []
    sample_counts = None

    for rep in range(args.replicates):
        leaf_seed = (seed * 1000003) ^ (rep * 9721) ^ 0xA53
        gates_seed = (seed * 1000003) ^ (rep * 9721) ^ 0x5A3
        leaf_rng = random.Random(leaf_seed)
        gates_rng = random.Random(gates_seed)

        ctx = gen.EmitContext(n_qubits=args.n_qubits)
        root.realize(ctx, leaf_rng)
        qasm = gen.emit_program(root, n_qubits=args.n_qubits, ctx=ctx, gates_rng=gates_rng)

        qasm_path = example_dir / f"circ_{rep:02d}.qasm"
        qasm_path.write_text(qasm, encoding="utf-8")
        circuit_files.append(qasm_path.name)

        if sample_counts is None:
            sample_counts = {
                "n_bits": max(1, ctx.bit_alloc),
                "total_1q": ctx.count_1q,
                "total_2q": ctx.count_2q,
                "n_leaves": sum(1 for block in ctx.blocks if block.get("type") == "leaf"),
            }

    if args.emit_metadata:
        metadata = {
            "example": example_name,
            "description": {
                "single_while_5ifs": "One WHILE loop containing 5 IF/ELSE blocks.",
                "two_nested_whiles": "Two nested WHILE loops.",
            }[example_name],
            "args": {
                "seed": seed,
                "n_qubits": args.n_qubits,
                "leaf_depth": args.leaf_depth,
                "leaf_block_depth": args.leaf_block_depth,
                "leaf_size": args.leaf_size,
                "leaf_density": {"alpha": args.alpha, "beta": args.beta},
                "gates_1q": [g.strip() for g in args.gates_1q.split(",") if g.strip()],
                "gates_2q": [g.strip() for g in args.gates_2q.split(",") if g.strip()],
                "replicates": args.replicates,
                "conflict_level": args.conflict_level,
                "device": str(args.device),
            },
            "skeleton_counts": skeleton_counts,
            "bit_count": bit_count,
            "circuits": circuit_files,
            "sample_counts_rep0": sample_counts,
        }
        (example_dir / "bench.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate two small paper example circuits in separate directories."
    )
    parser.add_argument("--n-qubits", type=int, default=16, help="Logical qubits used in both examples.")
    parser.add_argument("--leaf-depth", type=int, default=2, help="Leaf depth recorded in the metadata.")
    parser.add_argument("--leaf-block-depth", type=int, default=2, help="Actual depth used by each tiny leaf block.")
    parser.add_argument("--leaf-size", type=int, default=4, help="Number of qubits used by each tiny leaf block.")
    parser.add_argument("--alpha", type=float, default=0.25, help="1-qubit gate density per leaf cycle.")
    parser.add_argument("--beta", type=float, default=1.0, help="2-qubit gate density per leaf cycle.")
    parser.add_argument("--gates-1q", type=str, default="x,h,rz", help="Comma-separated 1-qubit gates.")
    parser.add_argument("--gates-2q", type=str, default="cx", help="Comma-separated 2-qubit gates.")
    parser.add_argument("--replicates", type=int, default=1, help="How many circuits to emit per example.")
    parser.add_argument("--num-ifs", type=int, default=5, help="Number of IF/ELSE blocks inside the single-WHILE example.")
    parser.add_argument("--conflict-level", type=int, default=0, help="Minimum graph distance between sibling IF branches.")
    parser.add_argument(
        "--device",
        type=Path,
        default=Path("../qpu/dqueko/backend_16_qubits_seed_42_density_0.3.gml"),
        help="Path to the backend GML file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmarks/paper_examples"),
        help="Base output directory.",
    )
    parser.add_argument("--seed", type=int, default=7, help="Base seed.")
    parser.add_argument("--emit-metadata", action="store_true", help="Write a small bench.json beside each example.")
    return parser.parse_args()


def main():
    args = parse_args()
    gen = load_generator_module()
    device = gen.load_gml_device(str((SCRIPT_DIR / args.device).resolve()))

    example_specs = [
        ("single_while_5ifs", build_single_while_with_ifs, args.seed + 101),
        ("two_nested_whiles", build_two_nested_whiles, args.seed + 202),
    ]

    outdir = (SCRIPT_DIR / args.output_dir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    for example_name, builder_fn, seed in example_specs:
        params = build_params(gen, args, device, seed)
        params.leaf_size = args.leaf_size
        params.leaf_block_depth = args.leaf_block_depth
        builder = gen.ProgramBuilder(params)
        if builder_fn is build_single_while_with_ifs:
            root = builder_fn(gen, builder, args.num_ifs)
        else:
            root = builder_fn(gen, builder)
        emit_example(gen, root, example_name, outdir, args, seed)

    print(f"Generated examples in: {outdir}")


if __name__ == "__main__":
    main()
