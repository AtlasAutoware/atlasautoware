#!/bin/bash
#SBATCH --job-name=atlas-ablate
#SBATCH --partition=gpu
#SBATCH --gres=gpu:turing:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=slurm_logs/07_ablation_%j.out
# Milestone T24 (sim part): lidar-only vs camera-only students (see ml/run_ablation.sh).
set -e
cd "$SLURM_SUBMIT_DIR"
source $HOME/atlas_venv/bin/activate
python3 -c "import onnxruntime" 2>/dev/null || pip install -q onnxruntime onnx
PY=$(which python3) bash ml/run_ablation.sh
