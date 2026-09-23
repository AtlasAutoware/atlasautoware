#!/bin/bash
# Submit the follow-up experiment: one job per seed (0-4), each plain BC + 3 clean DAgger
# rounds + evals. Run from the experiment directory (the one holding data/demos and runs/):
#   bash ml/slurm/submit_clean.sh
set -e
mkdir -p slurm_logs
for s in 0 1 2 3 4; do
  j=$(sbatch --parsable --export=ALL,SEEDS=$s ml/slurm/08_clean_pipeline.sbatch)
  echo "clean seed $s -> $j"
done
