#!/bin/bash
# Production run: prep (extra expert demos) -> 5 seeds in parallel -> select the best.
# Run from the experiment directory (the one holding data/ and runs/):  bash ml/slurm/submit_prod.sh
set -e
mkdir -p slurm_logs
prep=$(sbatch --parsable --export=ALL,STAGE=prep ml/slurm/09_prod.sbatch); echo "prod prep -> $prep"
deps=""
for s in 0 1 2 3 4; do
  j=$(sbatch --parsable --dependency=afterok:$prep --export=ALL,STAGE=seed,SEEDS=$s ml/slurm/09_prod.sbatch)
  echo "prod seed $s -> $j"; deps="$deps:$j"
done
# afterany: a seed that crashes should not block choosing among the others
sel=$(sbatch --parsable --dependency=afterany$deps --time=00:10:00 --export=ALL,STAGE=select ml/slurm/09_prod.sbatch)
echo "prod select -> $sel"
