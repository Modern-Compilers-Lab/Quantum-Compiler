#!/usr/bin/env python3
import subprocess
import tempfile
import textwrap
from pathlib import Path
import shlex

# ==== CONFIG ====
OUT_DIR = "/scratch/mb10324/Quantum-Compiler/dynamic-qlosure/out"
ERR_DIR = "/scratch/mb10324/Quantum-Compiler/dynamic-qlosure/err"
EMAIL  = "mb10324@nyu.edu"
PART   = "compute"
NODES  = 1
CPUS   = 4
MEM    = "64G"
TIME   = "2-0"  # days-hours
RESERVATION = "c2"  # uncomment if needed
# ==============

COMMANDS = [
    "python /scratch/mb10324/Quantum-Compiler/dynamic-qlosure/run_benchmark_seeds_sabre.py  --bench=54qbt --backend=ibm_brisbane_old --template=one_loop",
    "python /scratch/mb10324/Quantum-Compiler/dynamic-qlosure/run_benchmark_seeds_sabre.py  --bench=81qbt --backend=ibm_brisbane_old --template=one_loop",
    "python /scratch/mb10324/Quantum-Compiler/dynamic-qlosure/run_benchmark_seeds_sabre.py  --bench=121qbt --backend=ibm_brisbane_old --template=one_loop",
    "python /scratch/mb10324/Quantum-Compiler/dynamic-qlosure/run_benchmark_seeds.py  --bench=54qbt --backend=ibm_brisbane_old --template=one_loop",
    "python /scratch/mb10324/Quantum-Compiler/dynamic-qlosure/run_benchmark_seeds.py  --bench=81qbt --backend=ibm_brisbane_old --template=one_loop",
    "python /scratch/mb10324/Quantum-Compiler/dynamic-qlosure/run_benchmark_seeds.py  --bench=121qbt --backend=ibm_brisbane_old --template=one_loop",
    "python /scratch/mb10324/Quantum-Compiler/dynamic-qlosure/run_benchmark_seeds_sabre.py  --bench=54qbt --backend=ibm_kingston --template=one_loop",
    "python /scratch/mb10324/Quantum-Compiler/dynamic-qlosure/run_benchmark_seeds_sabre.py  --bench=81qbt --backend=ibm_kingston --template=one_loop",
    "python /scratch/mb10324/Quantum-Compiler/dynamic-qlosure/run_benchmark_seeds_sabre.py  --bench=121qbt --backend=ibm_kingston --template=one_loop",
    "python /scratch/mb10324/Quantum-Compiler/dynamic-qlosure/run_benchmark_seeds.py  --bench=54qbt --backend=ibm_kingston --template=one_loop",
    "python /scratch/mb10324/Quantum-Compiler/dynamic-qlosure/run_benchmark_seeds.py  --bench=81qbt --backend=ibm_kingston --template=one_loop",
    "python /scratch/mb10324/Quantum-Compiler/dynamic-qlosure/run_benchmark_seeds.py  --bench=121qbt --backend=ibm_kingston --template=one_loop",
]

def make_job_script(cmd: str, job_name: str) -> str:
    """Return full SLURM script content for a given command."""
    nb_cpu = CPUS if "sabre" in cmd else 2
    
    return textwrap.dedent(f"""\
        #!/bin/bash
        #SBATCH -p {PART}
        #SBATCH --nodes={NODES}
        #SBATCH -c {nb_cpu}
        #SBATCH --mem={MEM}
        #SBATCH -t {TIME}
        #SBATCH -o {OUT_DIR}/%x_%j.out
        #SBATCH -e {ERR_DIR}/%x_%j.err
        #SBATCH --mail-type=ALL
        #SBATCH --mail-user={EMAIL}
        ##SBATCH --reservation=c2
        #SBATCH -J {job_name}

        module load miniconda-nobashrc
        eval "$(conda shell.bash hook)"
        module load gcc
        conda activate main

        echo "Running: {shlex.quote(cmd)}"
        {cmd}
    """)

def submit_job(script_text: str, job_name: str) -> str:
    """Write a temporary job script and submit with sbatch, returning the job ID."""
    with tempfile.TemporaryDirectory() as td:
        script_path = Path(td) / f"{job_name}.sh"
        script_path.write_text(script_text)
        script_path.chmod(0o755)

        # pass job name explicitly to sbatch (overrides header if needed)
      
        result = subprocess.run(
            ["sbatch", "--parsable", f"--job-name={job_name}", str(script_path)],
            check=True,
            capture_output=True,
            text=True,
        )

        return result.stdout.strip()

def main():
    job_ids = []
    for i, cmd in enumerate(COMMANDS, 1):
        method  = "SABRE" if "sabre" in cmd else "qroqi"
        job_name =  method + "_".join(cmd.split()[2:6]).replace("--", "").replace("=", "_")
        script_text = make_job_script(cmd, job_name)
        print(job_name)
        print(script_text)
        jid = submit_job(script_text, job_name)
        job_ids.append((job_name, jid))
        print(f"Submitted {job_name}  ->  Job ID {jid}")

    print("\nAll jobs submitted successfully:")
    for name, jid in job_ids:
        print(f"{name:12s} | {jid}")

if __name__ == "__main__":
    main()
