#!/bin/bash
#SBATCH --job-name=atlas-env
#SBATCH --partition=compute
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=slurm_logs/01_env_%j.out
# One-time Python environment on NFS (no modules/conda on this cluster; system python 3.12).
set -e
cd "$SLURM_SUBMIT_DIR"
VENV=$HOME/atlas_venv
if [ ! -x "$VENV/bin/python" ]; then python3 -m venv "$VENV"; fi
source "$VENV/bin/activate"
pip install -q --upgrade pip
# Turing GPUs -> CUDA 12.x wheels
pip install -q torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -q numpy scipy pillow opencv-python-headless pandas pyarrow onnx \
    "transformers>=4.45" accelerate safetensors huggingface_hub sentencepiece einops timm qwen-vl-utils pyyaml
python - <<'PY'
import torch, transformers, cv2, pandas, pyarrow
print('torch', torch.__version__, 'cuda build', torch.version.cuda, '| transformers', transformers.__version__)
PY
echo "env ready: $VENV"
