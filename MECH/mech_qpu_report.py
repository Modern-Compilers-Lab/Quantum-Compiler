import argparse
import builtins
from statistics import mean

import Chiplet
import Circuit
import HighwayOccupancy
import Router
from Chiplet import (
    gen_chiplet_array,
    gen_highway_layout,
    custom_highway_layout,
    get_highway_qubits,
    get_local_coupling_graph,
    get_highway_coupling_graph,
    get_distance_to_highway,
)
from MECHBenchmarks import MECH_experiments

Chiplet.min = builtins.min
Chiplet.max = builtins.max
Chiplet.sum = builtins.sum
Router.min = builtins.min
Router.max = builtins.max
Router.sum = builtins.sum
HighwayOccupancy.min = builtins.min
HighwayOccupancy.max = builtins.max
HighwayOccupancy.sum = builtins.sum
Circuit.min = builtins.min
Circuit.max = builtins.max
Circuit.sum = builtins.sum


def parse_pair(text):
    parts = [p.strip() for p in text.split(",")]
    if len(parts) != 2:
        raise ValueError(f"Expected 'a,b' pair, got: {text}")
    return int(parts[0]), int(parts[1])


def degree_stats(graph):
    degrees = [d for _, d in graph.degree()]
    if not degrees:
        return {"min": 0, "max": 0, "avg": 0.0}
    return {
        "min": min(degrees),
        "max": max(degrees),
        "avg": mean(degrees),
    }


def print_section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def summarize_qpu(args):
    array_x, array_y = parse_pair(args.array)
    chip_x, chip_y = parse_pair(args.chiplet)

    chip = gen_chiplet_array(
        args.structure,
        array_x,
        array_y,
        chip_x,
        chip_y,
        cross_link_sparsity=args.sparsity,
    )

    if args.custom_highway:
        local_positions = [parse_pair(p) for p in args.custom_highway]
        custom_highway_layout(chip, local_positions)
    else:
        gen_highway_layout(chip)

    total_qubits = len(chip.nodes)
    highway_qubits = get_highway_qubits(chip)
    highway_num = len(highway_qubits)
    data_num = total_qubits - highway_num
    highway_ratio = 100.0 * highway_num / total_qubits if total_qubits else 0.0

    on_chip_edges = 0
    cross_chip_edges = 0
    for e in chip.edges:
        if chip.edges[e].get("type") == "cross_chip":
            cross_chip_edges += 1
        else:
            on_chip_edges += 1

    print_section("MECH QPU SPECIFICATION")
    print(f"structure            : {args.structure}")
    print(f"array dim            : ({array_x}, {array_y})")
    print(f"chiplet size         : ({chip_x}, {chip_y})")
    print(f"cross-link sparsity  : {args.sparsity}")
    print(f"total qubits         : {total_qubits}")
    print(f"data qubits          : {data_num}")
    print(f"highway qubits       : {highway_num}")
    print(f"highway percentage   : {highway_ratio:.2f}%")

    print_section("PHYSICAL CONNECTIVITY")
    print(f"on-chip edges        : {on_chip_edges}")
    print(f"cross-chip edges     : {cross_chip_edges}")
    if on_chip_edges + cross_chip_edges > 0:
        print(
            f"cross-chip edge ratio: {100.0 * cross_chip_edges / (on_chip_edges + cross_chip_edges):.2f}%"
        )

    local_graph = get_local_coupling_graph(chip).graph
    highway_graph = get_highway_coupling_graph(chip).graph
    local_deg = degree_stats(local_graph)
    highway_deg = degree_stats(highway_graph)

    print_section("COUPLING GRAPH STATS")
    print(
        f"local graph          : nodes={local_graph.number_of_nodes()}, edges={local_graph.number_of_edges()}, degree(min/avg/max)=({local_deg['min']}/{local_deg['avg']:.2f}/{local_deg['max']})"
    )
    print(
        f"highway graph        : nodes={highway_graph.number_of_nodes()}, edges={highway_graph.number_of_edges()}, degree(min/avg/max)=({highway_deg['min']}/{highway_deg['avg']:.2f}/{highway_deg['max']})"
    )

    distance_vals = [get_distance_to_highway(chip, node) for node in chip.nodes if node not in highway_qubits]
    if distance_vals:
        print(
            f"data->highway dist   : min={min(distance_vals)}, avg={mean(distance_vals):.2f}, max={max(distance_vals)}"
        )

    benchmarks = [b.strip() for b in args.benchmarks.split(",") if b.strip()]
    if not benchmarks:
        benchmarks = ["bv"]

    result = MECH_experiments(
        structure=args.structure,
        chiplet_array_dim=(array_x, array_y),
        chiplet_size=(chip_x, chip_y),
        benchmarks=benchmarks,
        cross_link_sparsity=args.sparsity,
        prep_period=args.prep_period,
        meas_period=args.meas_period,
        cross_chip_gate_weight=args.cross_chip_weight,
        meas_weight=args.meas_weight,
    )

    print_section("ESTIMATED LATENCY / COST (MECH)")
    print(f"weights              : cross-chip={args.cross_chip_weight}, measurement={args.meas_weight}")
    print(f"periods              : prep={args.prep_period if args.prep_period is not None else 'auto'}, meas={args.meas_period}")
    for name in benchmarks:
        if name not in result:
            continue
        r = result[name]
        print(
            f"{name:20s} depth={r['depth']}, eff_gate_num={r['eff_gate_num']}, on-chip={r['on-chip']}, cross-chip={r['cross-chip']}, meas={r['meas_num']}, shuttles={r['shuttle_num']}"
        )


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Generate and analyze a MECH QPU configuration.")
    parser.add_argument("--structure", default="square", choices=["square", "hexagon", "heavy_square", "heavy_hexagon"])
    parser.add_argument("--array", default="3,3", help="Chiplet array dimensions as 'x,y'.")
    parser.add_argument("--chiplet", default="7,7", help="Chiplet size as 'x,y'.")
    parser.add_argument("--sparsity", type=int, default=1, help="Cross-chip link sparsity parameter.")
    parser.add_argument("--benchmarks", default="bv", help="Comma-separated benchmarks: vqe,qft,qaoa,bv")
    parser.add_argument("--cross-chip-weight", type=float, default=7.4, dest="cross_chip_weight")
    parser.add_argument("--meas-weight", type=float, default=2.2, dest="meas_weight")
    parser.add_argument("--prep-period", type=int, default=None, dest="prep_period")
    parser.add_argument("--meas-period", type=int, default=2, dest="meas_period")
    parser.add_argument(
        "--custom-highway",
        nargs="*",
        default=None,
        help="Optional local chiplet highway positions like: --custom-highway 1,0 1,2 3,2",
    )
    return parser


if __name__ == "__main__":
    summarize_qpu(build_arg_parser().parse_args())
