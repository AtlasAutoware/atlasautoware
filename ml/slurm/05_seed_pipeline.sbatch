#!/bin/bash
#SBATCH --job-name=atlas-seed
#SBATCH --partition=gpu
#SBATCH --gres=gpu:turing:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=slurm_logs/05_seed_%j.out
# One full seed of the 9/23 experiment on the cluster: expert demos (deterministic, seed 0,
# identical to the laptop's), BC without/with augmentation, 3 DAgger rounds, closed-loop evals.
#   SEEDS=2 sbatch ml/slurm/05_seed_pipeline.sbatch
set -e
cd "$SLURM_SUBMIT_DIR"
source $HOME/atlas_venv/bin/activate
python3 -c "import onnxruntime" 2>/dev/null || pip install -q onnxruntime onnx
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
export PY=$(which python3) WORKERS=${SLURM_CPUS_PER_TASK:-16}
mkdir -p data logs runs
[ -f data/demos/summary.json ] || python3 ml/sim_rollout.py collect --policy expert --beta 1 \
    --maps levine,Spielberg_map,comp_track,my_track --n 100 --seed 0 --workers $WORKERS --out data/demos
SEEDS="${SEEDS:-2}" bash ml/run_experiments.sh
