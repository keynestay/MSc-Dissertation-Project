#!/bin/bash
#SBATCH --job-name=soay-age-size-dl
#SBATCH --clusters=srf_gpu_01
#SBATCH --partition=standard-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=14-00:00:00
#SBATCH --output=/bitbucket/wolf7754/soay_age_size_dowson_landau_%j.out

set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/dl4bi}"
cd "$REPO_DIR"

SOURCE_DIR="$REPO_DIR/experiments/soay/age_size/validation/covariance"
DATA_PATH="$REPO_DIR/generated/soay/age_size/data/gp_ibm_dataset_agesize.csv"
SAMPLES_PATH="$REPO_DIR/generated/soay/age_size/inference/gpdf_agesize_gp_samples.npz"
CHECKPOINT_PATH="$REPO_DIR/generated/soay/age_size/training/transformer/model.ckpt"

OUTPUT_DIR="${OUTPUT_DIR:-$REPO_DIR/generated/soay/age_size/validation/covariance/year2}"
LOG_PATH="/bitbucket/wolf7754/soay_age_size_dowson_landau_${SLURM_JOB_ID}.out"
RESUME="${RESUME:-0}"

for path in "$SOURCE_DIR/evaluate_transformer_dowson_landau_age_size.py" "$SOURCE_DIR/run_transformer_dowson_landau_age_size.sh" "$CHECKPOINT_PATH" "$DATA_PATH" "$SAMPLES_PATH"; do
    if [[ ! -e "$path" ]]; then
        echo "Missing Required Input: $path"
        exit 1
    fi
done

if [[ -e "$OUTPUT_DIR" && "$RESUME" != "1" ]]; then
    echo "Refusing To Overwrite Existing Directory: $OUTPUT_DIR"
    exit 1
fi
if [[ ! -e "$OUTPUT_DIR" && "$RESUME" == "1" ]]; then
    echo "Cannot Resume Because The Output Directory Does Not Exist: $OUTPUT_DIR"
    exit 1
fi

mkdir -p "$OUTPUT_DIR"/{inputs,progress,results,figures,logs,provenance}
PROVENANCE_DIR="$OUTPUT_DIR/provenance/job_${SLURM_JOB_ID}"
mkdir -p "$PROVENANCE_DIR"
ln -s "$LOG_PATH" "$OUTPUT_DIR/logs/slurm_${SLURM_JOB_ID}.out"

cp "$SOURCE_DIR/evaluate_transformer_dowson_landau_age_size.py" "$PROVENANCE_DIR/"
cp "$SOURCE_DIR/run_transformer_dowson_landau_age_size.sh" "$PROVENANCE_DIR/"
cp "$REPO_DIR/experiments/soay/age_size/abc/deeprv/deeprv_decoder_agesize.py" "$PROVENANCE_DIR/"
cp "$REPO_DIR/experiments/soay/age_size/training/train.py" "$PROVENANCE_DIR/train_age_size.py"

if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    git rev-parse HEAD > "$PROVENANCE_DIR/git_commit.txt"
    git status --short > "$PROVENANCE_DIR/git_status.txt"
else
    echo "Not Available: Source Directory Is Not A Git Worktree" > "$PROVENANCE_DIR/git_commit.txt"
    : > "$PROVENANCE_DIR/git_status.txt"
fi

source /bitbucket/wolf7754/miniforge3/etc/profile.d/conda.sh
conda activate /bitbucket/wolf7754/envs/soay-deeprv

conda list --explicit > "$PROVENANCE_DIR/conda_explicit.txt"
python -m pip freeze > "$PROVENANCE_DIR/pip_freeze.txt"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export XLA_PYTHON_CLIENT_PREALLOCATE=false

echo "Job ID: $SLURM_JOB_ID"
echo "Host: $(hostname)"
echo "Output Directory: $OUTPUT_DIR"
echo "Checkpoint: $CHECKPOINT_PATH"
echo "Resume: $RESUME"
nvidia-smi
nvidia-smi -q > "$PROVENANCE_DIR/nvidia_smi.txt"
python -c "import jax; print('JAX Devices:', jax.devices())"

RESUME_ARGUMENTS=()
if [[ "$RESUME" == "1" ]]; then
    RESUME_ARGUMENTS+=(--resume)
fi

srun --cpu-bind=none \
    --ntasks=1 \
    --cpus-per-task="$SLURM_CPUS_PER_TASK" \
    python -m experiments.soay.age_size.validation.covariance.evaluate_transformer_dowson_landau_age_size \
    --checkpoint "$CHECKPOINT_PATH" \
    --data "$DATA_PATH" \
    --samples "$SAMPLES_PATH" \
    --output-dir "$OUTPUT_DIR" \
    --years 2 \
    --realizations 500 \
    --decoder-batch-size 50 \
    --save-every 25 \
    --seed 20260909 \
    "${RESUME_ARGUMENTS[@]}"

echo "Completed Successfully"
echo "Results: $OUTPUT_DIR"
