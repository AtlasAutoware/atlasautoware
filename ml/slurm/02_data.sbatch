#!/bin/bash
#SBATCH --job-name=atlas-data
#SBATCH --partition=compute
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=slurm_logs/02_data_%j.out
# Regenerate the goal-conditioned sim dataset on the cluster (pure python, ~1 episode/s)
# and package it as LeRobot. No upload needed: the maps are in the repo.
set -e
cd "$SLURM_SUBMIT_DIR"
source $HOME/atlas_venv/bin/activate
EPS=${EPISODES_PER_MAP:-150}
rm -rf runs/sim_episodes
python3 tools/sim_generate.py --maps levine,Spielberg_map,comp_track,my_track \
    --per-map $EPS --out runs/sim_episodes --seed 0
python3 tools/episodes_to_lerobot.py runs/sim_episodes --out runs/lerobot_sim --only good
python3 - <<'PY'
import json; i=json.load(open('runs/lerobot_sim/meta/info.json'))
print('dataset:', i['total_episodes'], 'episodes', i['total_frames'], 'frames', i['total_tasks'], 'instructions')
PY
