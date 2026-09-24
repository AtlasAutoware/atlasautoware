#!/bin/bash
#SBATCH --job-name=atlas-prod
#SBATCH --partition=gpu
#SBATCH --gres=gpu:turing:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=slurm_logs/09_prod_%j.out
# One stage of the production run (ml/run_production.sh): STAGE=prep | seed (with SEEDS) | select
#   STAGE=prep sbatch ml/slurm/09_prod.sbatch
set -e
cd "$SLURM_SUBMIT_DIR"
source $HOME/atlas_venv/bin/activate
python3 -c "import onnxruntime" 2>/dev/null || pip install -q onnxruntime onnx
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
export PY=$(which python3) WORKERS=${SLURM_CPUS_PER_TASK:-16}
mkdir -p data logs runs
echo "stage ${STAGE:-?} seeds ${SEEDS:-}"
bash ml/run_production.sh "${STAGE:-prep}"
