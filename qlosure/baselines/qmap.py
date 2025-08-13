from qiskit import QuantumCircuit
from mqt import qmap

def run_qmap(data,edges,initial_mapping=None):
    qc = QuantumCircuit.from_qasm_str(data["qasm_code"])
    edges_arch = {(u, v) for u, v in edges}
    qubits = {u for u, v in edges_arch} | {v for u, v in edges_arch}
    num_qubits = len(qubits)
    arch = qmap.Architecture(num_qubits, edges_arch)
    new_qc = QuantumCircuit(qc.num_qubits, qc.num_clbits)

    for instr, qargs, cargs in qc.data:
        if instr.name != 'reset':
            new_qc.append(instr, qargs, cargs)
        

    # Map the circuit using QMAP (choose method: "exact" for optimal mapping or "heuristic" for faster results)
    
    if initial_mapping == "trivial": 
        qc_mapped, res = qmap.compile(new_qc, arch, method="heuristic",initial_layout=qmap.InitialLayout.identity,pre_mapping_optimizations=False,  post_mapping_optimizations=False)
    else:
        qc_mapped, res = qmap.compile(new_qc, arch, method="heuristic",initial_layout=qmap.InitialLayout.dynamic, pre_mapping_optimizations=False,post_mapping_optimizations=False)


    return {
        "swaps":res.output.swaps,
        "depth":qc_mapped.depth(),
        "cx_count":qc_mapped.count_ops().get("cx", 0),
        "results": res,
        "circuit": qc_mapped,
    }