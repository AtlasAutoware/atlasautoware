#!/bin/bash
# Route-hint run: prep (re-collect 1200 expert demos with the hint) -> 5 seeds -> select.
# Run from the experiment directory:  bash ml/slurm/submit_route.sh
set -e
mkdir -p slurm_logs
prep=$(sbatch --parsable --export=ALL,STAGE=prep ml/slurm/10_route.sbatch); echo "route prep -> $prep"
deps=""
for s in 0 1 2 3 4; do
  j=$(sbatch --parsable --dependency=afterok:$prep --export=ALL,STAGE=seed,SEEDS=$s ml/slurm/10_route.sbatch)
  echo "route seed $s -> $j"; deps="$deps:$j"
done
sel=$(sbatch --parsable --dependency=afterany$deps --time=00:10:00 --export=ALL,STAGE=select ml/slurm/10_route.sbatch)
echo "route select -> $sel"
