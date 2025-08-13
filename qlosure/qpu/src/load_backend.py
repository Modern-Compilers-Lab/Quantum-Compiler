import json
import os

TOPLOGIES_DIR = "hardware/topologies"

BACKEND_FILE_MAP = {
    "fake_5q_v1": "fake_5q_v1.json",
    "fake_20q_v1": "fake_20q_v1.json",
    "fake_27q_pulse_v1": "fake_27q_pulse_v1.json",
    "fake_127q_pulse_v1": "fake_127q_pulse_v1.json",
    "ibm_brisbane": "ibm_brisbane.json",
    "ibm_kyiv": "ibm_kyiv.json",
    "ibm_sherbrooke": "ibm_sherbrooke.json"
}


def load_backend_edges(backend_name):
    if backend_name not in BACKEND_FILE_MAP:
        raise KeyError(
            f"Backend '{backend_name}' not found in the file mapping.")

    file_path = f"{TOPLOGIES_DIR}/{BACKEND_FILE_MAP[backend_name]}"

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File '{file_path}' does not exist.")

    with open(file_path, 'r') as f:
        data = json.load(f)

    if "coupling_map" not in data:
        raise KeyError(f"Key 'coupling_map' not found in '{file_path}'.")

    return data["coupling_map"]
