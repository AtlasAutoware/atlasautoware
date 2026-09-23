#!/bin/bash
# Submit the whole 9/23 experiment on the TJ cluster: prep (demos + baseline eval), then one
# job per training seed (BC, BC+aug, 3 DAgger rounds, evals) and the input ablation, all in
# parallel once prep finishes. Run from the repo root on the login node:
#   bash ml/slurm/submit_0923.sh
# then: squeue -u $USER ; tail -f slurm_logs/*.out ; python3 ml/summarize_results.py
set -e
mkdir -p slurm_logs
prep=$(sbatch --parsable ml/slurm/06_prep.sbatch)
echo "prep $prep"
for s in 0 1 2; do
  j=$(sbatch --parsable --dependency=afterok:$prep --export=ALL,SEEDS=$s ml/slurm/05_seed_pipeline.sbatch)
  echo "seed $s -> $j"
done
j=$(sbatch --parsable --dependency=afterok:$prep ml/slurm/07_ablation.sbatch); echo "ablation -> $j"
