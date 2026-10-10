#!/bin/bash
#SBATCH --job-name=soay-smooth-control
#SBATCH --clusters=srf_gpu_01
#SBATCH --partition=standard-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=14-00:00:00
#SBATCH --output=logs/soay_transformer_smooth_control_%j.out

set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/dl4bi}"
cd "$REPO_DIR"

OUTPUT_DIR="${OUTPUT_DIR:-$REPO_DIR/generated/soay/size_only/validation/ks/smooth_control}"
if [[ -e "$OUTPUT_DIR" ]]; then
    echo "Refusing to overwrite existing directory: $OUTPUT_DIR"
    exit 1
fi
mkdir -p "$OUTPUT_DIR"

source /bitbucket/wolf7754/miniforge3/etc/profile.d/conda.sh
conda activate /bitbucket/wolf7754/envs/soay-deeprv

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"

echo "Job ID: $SLURM_JOB_ID"
echo "Host: $(hostname)"
echo "Output Directory: $OUTPUT_DIR"
nvidia-smi

srun --cpu-bind=none \
    --ntasks=1 \
    --cpus-per-task="$SLURM_CPUS_PER_TASK" \
    python -m experiments.soay.size_only.validation.ks.evaluate_transformer_smooth_ks \
    --checkpoint "$REPO_DIR/generated/soay/size_only/training/transformer/model.ckpt" \
    --data "$REPO_DIR/generated/soay/size_only/data/gp_ibm_dataset.csv" \
    --samples "$REPO_DIR/generated/soay/size_only/inference/gpdf_gp_samples.npz" \
    --output-dir "$OUTPUT_DIR" \
    --realizations 50 \
    --seed 20260908
