#!/usr/bin/env bash
# Whole distillation pipeline on the TJ cluster, one command from the login node:
#   git clone https://github.com/AtlasAutoware/atlasautoware.git ~/atlasautoware
#   bash ~/atlasautoware/ml/slurm/pipeline.sh
# Submits four dependent Slurm jobs: env -> data -> teacher (GPU) -> student (GPU).
set -e
cd "$(dirname "$0")/../.."
REPO=$PWD; mkdir -p "$REPO/runs" "$REPO/slurm_logs"
j1=$(sbatch --parsable "$REPO/ml/slurm/01_env.sbatch")
j2=$(sbatch --parsable --dependency=afterok:$j1 "$REPO/ml/slurm/02_data.sbatch")
j3=$(sbatch --parsable --dependency=afterok:$j2 "$REPO/ml/slurm/03_teacher.sbatch")
j4=$(sbatch --parsable --dependency=afterany:$j3 "$REPO/ml/slurm/04_student.sbatch")
echo "submitted: env=$j1 data=$j2 teacher=$j3 student=$j4"
echo "watch:  squeue -u \$USER    logs: $REPO/slurm_logs/"
