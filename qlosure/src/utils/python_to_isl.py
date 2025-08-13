
import islpy as isl


def dict_to_isl_map(input_dict: dict, chunk_size: int = 50) -> isl.UnionMap:
    """
    Convert a dictionary of {key: [values]} to an ISL UnionMap string format,
    grouping the entries into chunks of `chunk_size`.
    """
    entries = []
    for key, values in input_dict.items():
        if isinstance(values, (list, tuple, set)):
            for val in values:
                entries.append(f"[{key}]->[{val}]")
        else:
            entries.append(f"[{key}]->[{values}]")

    if not entries:
        return isl.UnionMap("{}")

    chunks = [entries[i:i + chunk_size]
              for i in range(0, len(entries), chunk_size)]

    union_map = isl.UnionMap("{}")
    for chunk in chunks:
        isl_map = isl.Map("{" + ";".join(chunk) + "}")
        union_map = union_map.union(isl_map)

    return union_map


def list_to_isl_set(input_list):
    if not input_list:
        return isl.UnionSet("{}")

    point_strings = []
    for item in input_list:
        point_strings.append(f"[{item}]")

    set_str = "{" + ";".join(point_strings) + "}"

    return isl.Set(set_str)


def int_to_isl_set(input):
    return isl.Set("{" + f"[{input}]" + "}")
