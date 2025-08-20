
import json
import islpy as isl
from src.utils.isl_to_python import *
import os
import ast
from time import time


def min_schedule(domain, read, schedule, C=100):
    ctx = isl.Context()

    deps = read.apply_range(read.reverse()).apply_domain(
        schedule).apply_range(schedule).intersect(isl.Map("{[i]->[j] : i<j}"))
    sc = isl.ScheduleConstraints.on_domain(domain)
    sc = sc.set_validity(deps)
    sc = sc.set_proximity(deps)

    sched_min = sc.compute_schedule()

    sched_um = sched_min.get_map()
    d_out = sched_um.sample().dim(isl.dim_type.out)

    parts = []
    for k in range(d_out):
        coeff = C**(d_out - 1 - k)
        parts.append(f"{coeff}*t{k}")
    expr = " + ".join(parts)

    map_str = "{ [" + ",".join(f"t{k}" for k in range(d_out)) + "]" \
        + " -> [" + expr + "] }"

    flatten_map = isl.Map.read_from_str(ctx, map_str)
    new_schedule = sched_um.apply_range(flatten_map)
    return new_schedule.intersect_domain(domain)


def max_schedule(domain, read, schedule, C=100):
    ctx = isl.Context()

    deps = read.apply_range(read.reverse()).apply_domain(
        schedule).apply_range(schedule).intersect(isl.Map("{[i]->[j] : i<j}"))
    sc = isl.ScheduleConstraints.on_domain(domain)
    sc = sc.set_validity(deps)
    sc = sc.set_proximity(deps)

    ctx.set_schedule_maximize_coincidence(True)
    sched_max = sc.compute_schedule()

    sched_um = sched_max.get_map()
    d_out = sched_um.sample().dim(isl.dim_type.out)

    parts = []
    for k in range(d_out):
        coeff = C**(d_out - 1 - k)
        parts.append(f"{coeff}*t{k}")
    expr = " + ".join(parts)

    map_str = "{ [" + ",".join(f"t{k}" for k in range(d_out)) + "]" \
        + " -> [" + expr + "] }"

    flatten_map = isl.Map.read_from_str(ctx, map_str)
    new_schedule = sched_um.apply_range(flatten_map)
    return new_schedule.intersect_domain(domain)


def json_file_to_isl(file_path: str, with_single: bool = True, reschedule: str = "default"):
    with open(file_path) as f:
        data = json.load(f)
    domain = isl.UnionSet(data["Domain"])
    read = isl.UnionMap(data["Read"])
    write = isl.UnionMap(data["Write"])
    schedule = isl.UnionMap(data["RecoveredSchedule"])

    if reschedule == "min":
        schedule = min_schedule(domain, read, schedule)
    elif reschedule == "max":
        schedule = max_schedule(domain, read, schedule)

    access_read = read2access(read, schedule)
    access_write = read2access(write, schedule)
    qasm_code = data["qasm_code"]

    group_dict = group_schedule(schedule.reverse())

    if not with_single:
        multi_keys = {key for key, vals in access_read.items()
                      if len(vals) > 1}
        access_read = {key: vals for key,
                       vals in access_read.items() if key in multi_keys}
        access_write = {key: vals for key,
                        vals in access_write.items() if key in multi_keys}

    result = {
        "qasm_code": qasm_code,
        "read": access_read,
        "write": access_write,
        "macro_gates": group_dict,
    }

    return result


def load_qasm(file_path: str):
    with open(file_path) as f:
        data = json.load(f)

    qasm_code = data["qasm_code"]

    result = {
        "qasm_code": qasm_code,
    }

    return result


def load_backend(file_path: str):
    with open(file_path) as f:
        data = json.load(f)

    return {
        "backend_name": data["backend_name"],
        "coupling_map": data["coupling_map"]

    }


def extract_multi_qubit_gates(access_map):
    return access_map.subtract(access_map.lexmin().intersect(access_map.lexmax())).domain()


def access_to_gates(read_dependencies_map, schedule_map):
    if schedule_map.is_empty():
        return None
    return schedule_map.reverse().apply_range(read_dependencies_map).as_map()


def filter_multi_qubit_gates(domain, read_dependencies, schedule):
    new_domain = extract_multi_qubit_gates(read_dependencies).coalesce()
    filtered_schedule = schedule.intersect_domain(new_domain)

    if filtered_schedule is None:
        return None, None, None

    new_read_dependicies = read_dependencies.intersect_domain(
        new_domain).coalesce()

    # new_schedule = rescheduling(filtered_schedule)
    new_schedule = filtered_schedule

    return new_domain, new_read_dependicies, new_schedule


def rescheduling(schedule):

    schedule_points_set = schedule.range()
    schedule_points_list = isl_set_to_python_list(schedule_points_set)

    schedule_points_list.sort()

    nb_points = len(schedule_points_list)

    compact_schedule_points_list = list(range(nb_points))

    dispersed_to_compact_schedule_map = isl.Map(
        "{" + ";".join(f"[{x}]->[{y}]" for x, y in zip(schedule_points_list, compact_schedule_points_list)) + "}")

    return schedule.apply_range(dispersed_to_compact_schedule_map)
