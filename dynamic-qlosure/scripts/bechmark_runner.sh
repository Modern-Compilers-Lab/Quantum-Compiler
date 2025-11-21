#!/bin/bash
#SBATCH -p compute
#SBATCH --nodes=1
#SBATCH -c 8
#SBATCH --mem=128G
##SBATCH --reservation=c2
#SBATCH -t 5-0
#SBATCH -o /scratch/mb10324/Quantum-Compiler/dynamic-qlosure/out/%x_%j.out
#SBATCH -e /scratch/mb10324/Quantum-Compiler/dynamic-qlosure/err/%x_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mb10324@nyu.edu
#SBATCH -J ablation-depth-rate_with_error-no-error-brisbane-81qbt


# Load modules and activate conda environment
module load miniconda-nobashrc
eval "$(conda shell.bash hook)"
module load gcc
conda activate main


# python /scratch/mb10324/Quantum-Compiler/dynamic-qlosure/run_benchmark_seeds_sabre.py  --bench=54qbt --backend=ibm_brisbane_old --template=one_loop
# python /scratch/mb10324/Quantum-Compiler/dynamic-qlosure/run_benchmark_seeds_sabre.py  --bench=81qbt --backend=ibm_brisbane_old --template=one_loop
# python /scratch/mb10324/Quantum-Compiler/dynamic-qlosure/run_benchmark_seeds_sabre.py  --bench=121qbt --backend=ibm_brisbane_old --template=one_loop

# python /scratch/mb10324/Quantum-Compiler/dynamic-qlosure/run_benchmark_seeds.py  --bench=54qbt --backend=ibm_brisbane_old --template=one_loop
# python /scratch/mb10324/Quantum-Compiler/dynamic-qlosure/run_benchmark_seeds.py  --bench=81qbt --backend=ibm_brisbane_old --template=one_loop
# python /scratch/mb10324/Quantum-Compiler/dynamic-qlosure/run_benchmark_seeds.py  --bench=121qbt --backend=ibm_brisbane_old --template=one_loop


# python /scratch/mb10324/Quantum-Compiler/dynamic-qlosure/run_benchmark_seeds_sabre.py  --bench=54qbt --backend=ibm_kingston --template=one_loop
# python /scratch/mb10324/Quantum-Compiler/dynamic-qlosure/run_benchmark_seeds_sabre.py  --bench=81qbt --backend=ibm_kingston --template=one_loop
# python /scratch/mb10324/Quantum-Compiler/dynamic-qlosure/run_benchmark_seeds_sabre.py  --bench=121qbt --backend=ibm_kingston --template=one_loop

# python /scratch/mb10324/Quantum-Compiler/dynamic-qlosure/run_benchmark_seeds.py  --bench=54qbt --backend=ibm_kingston --template=one_loop
# python /scratch/mb10324/Quantum-Compiler/dynamic-qlosure/run_benchmark_seeds.py  --bench=81qbt --backend=ibm_kingston --template=one_loop
# python /scratch/mb10324/Quantum-Compiler/dynamic-qlosure/run_benchmark_seeds.py  --bench=121qbt --backend=ibm_kingston --template=one_loop


# python /scratch/mb10324/Quantum-Compiler/dynamic-qlosure/run_benchmark_seeds.py  --bench=121qbt --backend=ibm_brisbane_old --template=nest0
# python /scratch/mb10324/Quantum-Compiler/dynamic-qlosure/run_benchmark_seeds.py  --bench=121qbt --backend=ibm_brisbane_old --template=nest0


# python /scratch/mb10324/Quantum-Compiler/dynamic-qlosure/run_benchmarks_nested.py  --bench=121qbt --backend=ibm_kingston --leaf_depth=10
# python /scratch/mb10324/Quantum-Compiler/dynamic-qlosure/run_benchmarks_nested_sabre.py  --bench=121qbt --backend=ibm_kingston --leaf_depth=10

# python /scratch/mb10324/Quantum-Compiler/dynamic-qlosure/run_benchmarks_nested.py  --bench=121qbt --backend=ibm_kingston --leaf_depth=10
# python /scratch/mb10324/Quantum-Compiler/dynamic-qlosure/run_benchmarks_nested_sabre.py  --bench=121qbt --backend=ibm_kingston --leaf_depth=10

# python /scratch/mb10324/Quantum-Compiler/dynamic-qlosure/run_benchmark_seeds.py  --bench=121qbt 
# python /scratch/mb10324/Quantum-Compiler/dynamic-qlosure/run_benchmark_seeds.py  --bench=121qbt --backend=ibm_brisbane_old
python /scratch/mb10324/Quantum-Compiler/dynamic-qlosure/run_benchmark_seeds.py  --bench=81qbt --backend=ibm_brisbane_old
