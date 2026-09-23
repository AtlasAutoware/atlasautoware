#!/bin/bash
#SBATCH --job-name=atlas-prep
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=slurm_logs/06_prep_%j.out
# Shared inputs for the 9/23 experiments: the 400 expert demos (deterministic, seed 0) and the
# closed-loop evaluation of the pre-9/23 student (models/student.onnx) on the same task set.
set -e
cd "$SLURM_SUBMIT_DIR"
source $HOME/atlas_venv/bin/activate
python3 -c "import onnxruntime" 2>/dev/null || pip install -q onnxruntime onnx
W=${SLURM_CPUS_PER_TASK:-16}; M=levine,Spielberg_map,comp_track,my_track
mkdir -p data runs/baseline
[ -f data/demos/summary.json ] || python3 ml/sim_rollout.py collect --policy expert --beta 1 --maps $M --n 100 --seed 0 --workers $W --out data/demos
for pert in none lidar camera nocam; do
  P=""; [ $pert != none ] && P="--perturb $pert"
  python3 ml/sim_rollout.py eval --policy models/student.onnx --maps $M,uploadtest --n 30 --seed 1000 --workers $W $P --out runs/baseline/eval_$pert
done
echo PREP_DONE
