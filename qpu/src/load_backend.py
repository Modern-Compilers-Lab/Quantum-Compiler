import json
import os

TOPOLOGIES_DIR = os.path.join(os.path.dirname(__file__), "..", "topologies")

BACKEND_FILE_MAP = {
    "fake_5q_v1": "fake_5q_v1.json",
    "fake_20q_v1": "fake_20q_v1.json",
    "fake_27q_pulse_v1": "fake_27q_pulse_v1.json",
    "fake_127q_pulse_v1": "fake_127q_pulse_v1.json",
    "ibm_brisbane": "ibm_brisbane.json",
    "ibm_kyiv": "ibm_kyiv.json",
    "ibm_sherbrooke": "ibm_sherbrooke.json",
    "ankaa": "Ankaa-3.json",
    "ibm_sherbrooke2X": "IBM_sherbrooke2x.json",
    "ibm_kingston": "ibm_kingston.json",
    "heavy_hexagon": "backend_heavy_hexagon_2x2_8x8.json",
    "heavy_square": "backend_heavy_square_2x2_8x8.json",
    "ibm_brisbane_old": "ibm_brisbane_old.json",
    "ibm_flamingo": "ibm_flamingo.json",
    "mech_heavy_hex": "mech_heavy_hex_3x4_8x8.json",
    "mech_heavy_square": "mech_heavy_square_3x3_8x8.json",
    "mech_hex": "mech_hex_3x3_8x8.json",
    "mech_square": "mech_square_3x3_7x7.json",
}


def _resolve_backend_path(backend_name):
    """Return the resolved file path for a backend name."""
    if backend_name not in BACKEND_FILE_MAP:
        raise KeyError(
            f"Backend '{backend_name}' not found in the file mapping.")

    file_path = os.path.join(TOPOLOGIES_DIR, BACKEND_FILE_MAP[backend_name])

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File '{file_path}' does not exist.")

    return file_path


def load_backend_data(backend_name):
    """Load and return the full topology JSON dict for *backend_name*.

    The returned dict is guaranteed to contain at least ``"coupling_map"``.
    It may also contain ``"qubits"`` (qubit properties) depending on the
    topology file.
    """
    file_path = _resolve_backend_path(backend_name)

    with open(file_path, 'r') as f:
        data = json.load(f)

    if "coupling_map" not in data:
        raise KeyError(f"Key 'coupling_map' not found in '{file_path}'.")

    return data


def load_backend_edges(backend_name):
    """Return only the coupling-map edge list for *backend_name*."""
    return load_backend_data(backend_name)["coupling_map"]
