#!/bin/bash
#SBATCH --job-name=soay-age-size-smooth
#SBATCH --clusters=srf_gpu_01
#SBATCH --partition=standard-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=14-00:00:00
#SBATCH --output=logs/soay_transformer_smooth_control_age_size_%j.out

set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/dl4bi}"
cd "$REPO_DIR"

OUTPUT_DIR="${OUTPUT_DIR:-$REPO_DIR/generated/soay/age_size/validation/ks/smooth_control}"
if [[ -e "$OUTPUT_DIR" ]]; then
    echo "Refusing to overwrite existing directory: $OUTPUT_DIR"
    exit 1
fi
mkdir -p "$OUTPUT_DIR"

CHECKPOINT_PATH="$REPO_DIR/generated/soay/age_size/training/transformer/model.ckpt"
if [[ ! -d "$CHECKPOINT_PATH" ]]; then
    echo "Checkpoint not found: $CHECKPOINT_PATH"
    exit 1
fi

source /bitbucket/wolf7754/miniforge3/etc/profile.d/conda.sh
conda activate /bitbucket/wolf7754/envs/soay-deeprv

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"

EXACT_CONTROL_ARGUMENTS=()
if [[ "${SKIP_EXACT_CONTROL:-0}" == "1" ]]; then
    EXACT_CONTROL_ARGUMENTS+=(--skip-exact-control)
fi

echo "Job ID: $SLURM_JOB_ID"
echo "Host: $(hostname)"
echo "Output Directory: $OUTPUT_DIR"
echo "Checkpoint: $CHECKPOINT_PATH"
echo "Exact GP Control: $([[ ${#EXACT_CONTROL_ARGUMENTS[@]} -eq 0 ]] && echo Enabled || echo Skipped)"
nvidia-smi

srun --cpu-bind=none \
    --ntasks=1 \
    --cpus-per-task="$SLURM_CPUS_PER_TASK" \
    python -m experiments.soay.age_size.validation.ks.evaluate_transformer_smooth_ks_age_size \
    --checkpoint "$CHECKPOINT_PATH" \
    --data "$REPO_DIR/generated/soay/age_size/data/gp_ibm_dataset_agesize.csv" \
    --samples "$REPO_DIR/generated/soay/age_size/inference/gpdf_agesize_gp_samples.npz" \
    --output-dir "$OUTPUT_DIR" \
    --realizations 50 \
    --seed 20260909 \
    "${EXACT_CONTROL_ARGUMENTS[@]}"
