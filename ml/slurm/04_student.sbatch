#!/bin/bash
#SBATCH --job-name=atlas-student
#SBATCH --partition=gpu
#SBATCH --gres=gpu:turing:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=slurm_logs/04_student_%j.out
# Train the small goal-conditioned student (BC on the expert + distillation from the
# teacher features when available) and export ONNX for the Jetson.
set -e
cd "$SLURM_SUBMIT_DIR"
source $HOME/atlas_venv/bin/activate
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
TF=runs/teacher/feats.npz
[ -f "$TF" ] && ARG="--teacher-feats $TF" || ARG=""
python3 ml/train_student.py --data runs/lerobot_sim --out runs/student --epochs ${EPOCHS:-40} --workers 8 $ARG
ls -la runs/student/
echo "student: runs/student/student.onnx  (copy to the car: models/student.onnx)"
