#!/bin/bash
#SBATCH --job-name=atlas-teacher
#SBATCH --partition=gpu
#SBATCH --gres=gpu:turing:2
# gpu partition MaxMemPerNode=64000 MB; keep --mem under that
#SBATCH --cpus-per-task=16
#SBATCH --mem=60G
#SBATCH --time=06:00:00
#SBATCH --output=slurm_logs/03_teacher_%j.out
# Qwen-Drive-1.0-4B teacher: download (13.8 GB, cached on NFS) and extract vision features
# on the sim frames. fp16 across 2 Turing GPUs via device_map=auto. Best-effort: if the
# custom architecture will not load, falls back to Qwen3-VL-2B; if that fails the student
# trains BC-only (04 uses afterany).
set -e
cd "$SLURM_SUBMIT_DIR"
source $HOME/atlas_venv/bin/activate
export HF_HOME=$HOME/.cache/huggingface
export TOKENIZERS_PARALLELISM=false
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
python3 ml/extract_teacher.py --data runs/lerobot_sim --out runs/teacher --every 5
cat runs/teacher/meta.json
