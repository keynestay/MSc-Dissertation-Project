#!/bin/bash
#SBATCH --job-name=soay-dl-counts
#SBATCH --clusters=srf_gpu_01
#SBATCH --partition=standard-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=14-00:00:00
#SBATCH --output=/bitbucket/wolf7754/soay_dowson_landau_draw_counts_%j.out

set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/dl4bi}"
cd "$REPO_DIR"

SOURCE_DIR="$REPO_DIR/experiments/soay/size_only/validation/covariance"
CHECKPOINT_DIR="$REPO_DIR/generated/soay/size_only/training/transformer"
DATA_PATH="$REPO_DIR/generated/soay/size_only/data/gp_ibm_dataset.csv"
SAMPLES_PATH="$REPO_DIR/generated/soay/size_only/inference/gpdf_gp_samples.npz"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_DIR/generated/soay/size_only/validation/covariance/draw_counts}"
LOG_PATH="/bitbucket/wolf7754/soay_dowson_landau_draw_counts_${SLURM_JOB_ID}.out"

if [[ -e "$OUTPUT_DIR" ]]; then
    echo "Refusing to overwrite existing directory: $OUTPUT_DIR"
    exit 1
fi

for path in "$SOURCE_DIR/evaluate_transformer_dowson_landau_draw_counts.py" "$SOURCE_DIR/evaluate_transformer_dowson_landau.py" "$CHECKPOINT_DIR/model.ckpt" "$CHECKPOINT_DIR/metadata.json" "$DATA_PATH" "$SAMPLES_PATH"; do
    if [[ ! -e "$path" ]]; then
        echo "Missing required input: $path"
        exit 1
    fi
done

mkdir -p "$OUTPUT_DIR"/{inputs,intermediates,results,figures,progress,logs}
ln -s "$LOG_PATH" "$OUTPUT_DIR/logs/slurm.out"

cp "$SOURCE_DIR/evaluate_transformer_dowson_landau_draw_counts.py" "$OUTPUT_DIR/inputs/"
cp "$SOURCE_DIR/evaluate_transformer_dowson_landau.py" "$OUTPUT_DIR/inputs/"
cp "$SOURCE_DIR/run_transformer_dowson_landau_draw_counts.sh" "$OUTPUT_DIR/inputs/"
cp "$DATA_PATH" "$OUTPUT_DIR/inputs/"
cp "$SAMPLES_PATH" "$OUTPUT_DIR/inputs/"
cp "$CHECKPOINT_DIR/metadata.json" "$OUTPUT_DIR/inputs/checkpoint_metadata.json"
cp "$REPO_DIR/experiments/soay/size_only/training/deeprv_decoder.py" "$OUTPUT_DIR/inputs/"
cp "$REPO_DIR/experiments/soay/size_only/training/train_transformer_deeprv.py" "$OUTPUT_DIR/inputs/"
cp -a "$CHECKPOINT_DIR/model.ckpt" "$OUTPUT_DIR/inputs/"

if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    git rev-parse HEAD > "$OUTPUT_DIR/inputs/git_commit.txt"
    git status --short > "$OUTPUT_DIR/inputs/git_status.txt"
else
    echo "Not Available: Source Directory Is Not A Git Worktree" > "$OUTPUT_DIR/inputs/git_commit.txt"
    : > "$OUTPUT_DIR/inputs/git_status.txt"
fi

source /bitbucket/wolf7754/miniforge3/etc/profile.d/conda.sh
conda activate /bitbucket/wolf7754/envs/soay-deeprv

conda list --explicit > "$OUTPUT_DIR/inputs/conda_explicit.txt"
python -m pip freeze > "$OUTPUT_DIR/inputs/pip_freeze.txt"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export XLA_PYTHON_CLIENT_PREALLOCATE=false

echo "Job ID: $SLURM_JOB_ID"
echo "Host: $(hostname)"
echo "Output Directory: $OUTPUT_DIR"
echo "Log: $LOG_PATH"
nvidia-smi
nvidia-smi -q > "$OUTPUT_DIR/inputs/nvidia_smi.txt"
python -c "import jax; print('JAX Devices:', jax.devices())"

srun --cpu-bind=none \
    --ntasks=1 \
    --cpus-per-task="$SLURM_CPUS_PER_TASK" \
    python -m experiments.soay.size_only.validation.covariance.evaluate_transformer_dowson_landau_draw_counts \
    --checkpoint "$OUTPUT_DIR/inputs/model.ckpt" \
    --data "$OUTPUT_DIR/inputs/gp_ibm_dataset.csv" \
    --samples "$OUTPUT_DIR/inputs/gpdf_gp_samples.npz" \
    --output-dir "$OUTPUT_DIR" \
    --year 1 \
    --realization-counts 50 100 250 500 \
    --decoder-batch-size 10 \
    --save-every 25 \
    --seed 20260908

echo "Completed Successfully"
echo "Results: $OUTPUT_DIR"
