#!/bin/bash
#SBATCH --job-name=atlas-clean
#SBATCH --partition=gpu
#SBATCH --gres=gpu:turing:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=slurm_logs/08_clean_%j.out
# One seed of the follow-up experiment (ml/run_experiments_clean.sh): plain BC, then DAgger
# rounds 1-3 from that model without augmentation. Needs data/demos from 06_prep.
#   SEEDS=3 sbatch ml/slurm/08_clean_pipeline.sbatch
set -e
cd "$SLURM_SUBMIT_DIR"
source $HOME/atlas_venv/bin/activate
python3 -c "import onnxruntime" 2>/dev/null || pip install -q onnxruntime onnx
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
export PY=$(which python3) WORKERS=${SLURM_CPUS_PER_TASK:-16}
mkdir -p data logs runs
[ -f data/demos/summary.json ] || { echo "data/demos missing: run 06_prep first"; exit 1; }
SEEDS="${SEEDS:-0}" bash ml/run_experiments_clean.sh
