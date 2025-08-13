import json
import os
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit.providers.fake_provider import (
    Fake5QV1,
    Fake20QV1,
    Fake27QPulseV1,
    Fake127QPulseV1,
)


def save_backend_coupling_map(backend, backend_name, output_dir):
    config = backend.configuration()
    if hasattr(config, "coupling_map") and config.coupling_map:
        filename = os.path.join(output_dir, f"{backend_name}.json")
        with open(filename, "w") as f:
            json.dump(
                {"backend_name": backend_name, "coupling_map": config.coupling_map},
                f,
                indent=2,
            )
        print(f"Saved coupling map for {backend_name}")


def save_coupling_maps():
    output_dir = "topologies"
    os.makedirs(output_dir, exist_ok=True)

    print("\nProcessing real backends...")
    real_backends_names = [
        'ibm_sherbrooke',
        'ibm_kyiv',
        'ibm_brisbane'
    ]
    try:
        service = QiskitRuntimeService()
        for backend_name in real_backends_names:
            try:
                backend = service.backend(backend_name)
                save_backend_coupling_map(backend, backend_name, output_dir)
            except Exception as e:
                print(f"Error processing {backend_name}: {e}")
    except Exception as e:
        print(f"Error accessing IBM Quantum backends: {e}")

    print("\nProcessing fake backends...")
    fake_backends = [
        Fake5QV1(),
        Fake20QV1(),
        Fake27QPulseV1(),
        Fake127QPulseV1(),
    ]
    for backend in fake_backends:
        try:
            backend_name = backend.name()
            save_backend_coupling_map(backend, backend_name, output_dir)
        except Exception as e:
            print(
                f"Error processing {backend.name() if hasattr(backend, 'name') else 'unknown'}: {e}")


if __name__ == "__main__":
    print("Getting coupling maps for all Qiskit backends...")
    save_coupling_maps()
    print("\nFinished saving coupling maps to the 'topologies' directory.")
