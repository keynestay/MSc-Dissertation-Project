#!/bin/bash
#SBATCH --job-name=gpdf-inference
#SBATCH --clusters=srf_gpu_01
#SBATCH --partition=standard-gpu
#SBATCH --nodelist=cholgpu01
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --output=gpdf_inference_%j.out
#SBATCH --error=gpdf_inference_%j.err

set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/dl4bi}"
RESULTS_ROOT="/bitbucket/$USER/dl4bi-results/gpdf-inference"
RUN_DIR="${OUTPUT_DIR:-$REPO_DIR/generated/soay/size_only/inference}"

if [[ -e "$RUN_DIR" ]]; then
    echo "Refusing to overwrite existing directory: $RUN_DIR"
    exit 1
fi
mkdir -p "$RUN_DIR"

source "/bitbucket/$USER/miniforge3/etc/profile.d/conda.sh"
conda activate "/bitbucket/$USER/envs/soay-deeprv"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"

cd "$REPO_DIR"

echo "Job ID: $SLURM_JOB_ID"
echo "Host: $(hostname)"
echo "Output directory: $RUN_DIR"
nvidia-smi -L

srun --cpu-bind=none \
    --ntasks=1 \
    --cpus-per-task="$SLURM_CPUS_PER_TASK" \
    python -u -m experiments.soay.size_only.inference.run_gp_df_inference \
    --output-dir "$RUN_DIR"
