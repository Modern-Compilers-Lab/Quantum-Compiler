import islpy as isl
from collections import defaultdict
from collections import defaultdict


def isl_map_to_python_dict(_map):
    domain_point = isl_set_to_python_list(_map.domain())

    map_dict = {}
    for point in domain_point:
        qubits_list = isl_set_to_python_list(
            _map.intersect_domain(isl.Set(f"{{[{point}]}}")).range().as_set())
        qubits_list.sort()

        map_dict[point] = qubits_list

    return map_dict


def isl_map_to_dict_optimized(m: isl.Map):
    result = defaultdict(list)
    dim_set = isl.dim_type.set
    to_py = isl.Val.to_python  # Cache method lookup

    def callback(p: isl.Point) -> None:
        domain = to_py(p.get_coordinate_val(dim_set, 0))
        range_val = to_py(p.get_coordinate_val(dim_set, 1))
        result[domain].append(range_val)

    m.wrap().foreach_point(callback)

    return result


def isl_map_to_dict_optimized2(_map):
    result = defaultdict(list)

    def map_to_dict(b):
        dim_set = isl.dim_type.set
        to_py = isl.Val.to_python

        def callback(p) -> None:
            domain = to_py(p.get_coordinate_val(dim_set, 0))
            range_val = to_py(p.get_coordinate_val(dim_set, 1))
            result[domain].append(range_val)

        b.foreach_point(callback)

    for b in _map.wrap().get_basic_sets():
        map_to_dict(b)

    return result


def isl_map_to_dict_optimized3(_map):
    result = defaultdict(int)

    def map_to_dict(b):
        dim_set = isl.dim_type.set
        to_py = isl.Val.to_python

        def callback(p) -> None:
            domain = to_py(p.get_coordinate_val(dim_set, 0))
            range_val = to_py(p.get_coordinate_val(dim_set, 1))
            result[domain] = range_val

        b.foreach_point(callback)

    for b in _map.wrap().get_basic_sets():
        map_to_dict(b)

    return result


def isl_count_dependencies(_map):
    result = defaultdict(int)

    def map_to_dict(b):
        dim_set = isl.dim_type.set
        to_py = isl.Val.to_python

        def callback(p) -> None:
            domain = to_py(p.get_coordinate_val(dim_set, 0))
            result[domain] = result.get(domain, 0) + 1

        b.foreach_point(callback)

    _map = _map.transitive_closure()[0].wrap().as_set()
    for b in _map.get_basic_sets():
        map_to_dict(b)

    return result


def count_dependencies(dag):
    memo = {}

    def reachable(node):
        if node in memo:
            return memo[node]
        # Set to hold nodes reachable from 'node'
        reached = set()
        for neighbor in dag.get(node, []):
            reached.add(neighbor)
            reached |= reachable(neighbor)
        memo[node] = reached
        return reached

    # Compute the transitive closure for each node in the DAG
    closure = {node: reachable(node) for node in dag}

    return {node: len(reachable_nodes) for node, reachable_nodes in closure.items()}


def isl_set_to_python_list(_set):
    points = []

    def point_to_int(point):
        points.append(point.to_set().dim_min_val(0).to_python())

    _set.foreach_point(point_to_int)

    return points


def isl_set_to_python_set(_set):
    points = set()

    def point_to_int(point):
        points.add(point.to_set().dim_min_val(0).to_python())

    _set.foreach_point(point_to_int)
    return points


def isl_set_to_list_points(_set):
    points = []

    def point_to_int(point):
        points.append(point.to_set())

    _set.foreach_point(point_to_int)

    return points


def shedule_to_qubits(_schedule):

    result = defaultdict(list)

    dim_set = isl.dim_type.set
    to_str = isl.Val.to_str
    to_py =  isl.Val.to_python

    def bs_to_key(bs):
        name = bs.get_tuple_name().lower()
        n_dims = bs.dim(dim_set)
        coords = [to_str(bs.sample_point().get_coordinate_val(dim_set, i)) for i in range(n_dims)]
        return name + "_" + "_".join(coords)

    def map_callback(_map):
        def map_to_dict(b):

            def callback(_point) -> None:
                _m = _point.to_set().unwrap()
                domain = bs_to_key(_m.domain())
                range_val = to_py(_m.range().sample_point().get_coordinate_val(dim_set, 0))
                result[domain].append(range_val)

            b.foreach_point(callback)

        for b in _map.get_basic_sets():
            map_to_dict(b)


    _schedule.wrap().foreach_set(map_callback)
    
    return result


def time_to_schedule(_schedule):

    result = defaultdict(list)

    dim_set = isl.dim_type.set
    to_str = isl.Val.to_str
    to_py =  isl.Val.to_python

    def bs_to_key(bs):
        name = bs.get_tuple_name().lower()
        n_dims = bs.dim(dim_set)
        coords = [to_str(bs.sample_point().get_coordinate_val(dim_set, i)) for i in range(n_dims)]
        return name + "_" + "_".join(coords)

            
    def map_callback(_map):
        def map_to_dict(b):

            def callback(_point) -> None:
                _m = _point.to_set().unwrap()
                domain = to_py(_m.domain().sample_point().get_coordinate_val(dim_set, 0))
                range = bs_to_key(_m.range())
                result[domain].append(range)

            b.foreach_point(callback)

        for b in _map.get_basic_sets():
            map_to_dict(b)


    _schedule.wrap().foreach_set(map_callback)
    
    return result



def group_schedule(_schedule):

    result = defaultdict(list)

    dim_set = isl.dim_type.set
    to_str = isl.Val.to_str
    to_py =  isl.Val.to_python

    def bs_to_key(bs):
        name = bs.get_tuple_name().lower()
        n_dims = bs.dim(dim_set)
        coords = [to_str(bs.sample_point().get_coordinate_val(dim_set, i)) for i in range(n_dims)]
        return name

            
    def map_callback(_map):
        def map_to_dict(b):

            def callback(_point) -> None:
                _m = _point.to_set().unwrap()
                domain = to_py(_m.domain().sample_point().get_coordinate_val(dim_set, 0))
                range = bs_to_key(_m.range())
                result[domain].append(range)

            b.foreach_point(callback)

        for b in _map.get_basic_sets():
            map_to_dict(b)


    _schedule.wrap().foreach_set(map_callback)
    
    return result


def read2access(read,schedule):
    t2s = time_to_schedule(schedule.reverse())
    s2q= shedule_to_qubits(read)
    time_to_qubits = defaultdict(list)
    for time, schedules in t2s.items():
        for schedule in schedules:
            qubits = s2q.get(schedule, [])
            time_to_qubits[time].extend(qubits)
            
    return time_to_qubits

    