#!/bin/bash
#SBATCH --job-name=soay-abc-full
#SBATCH --clusters=srf_cpu_01
#SBATCH --partition=standard-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=36
#SBATCH --mem=128G
#SBATCH --time=14-00:00:00
#SBATCH --output=soay_abc_full_%j.out

set -euo pipefail

source "/bitbucket/$USER/miniforge3/etc/profile.d/conda.sh"
conda activate "/bitbucket/$USER/conda-envs/soay-abc"

cd "$SLURM_SUBMIT_DIR"

echo "Host: $(hostname)"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Temporary directory: $TMPDIR"

python SOAYSHEEP/run_soay_abc_hpc.py \
    SOAYSHEEP/gp_ibm_dataset.csv \
    SOAYSHEEP/gpdf_gp_samples.npz \
    SOAYSHEEP/hpc_full_output
