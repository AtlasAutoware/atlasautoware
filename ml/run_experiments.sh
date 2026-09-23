#!/usr/bin/env bash
# 2026-09-23 experiment pipeline (laptop RTX 5070): BC without/with augmentation, then DAgger,
# 3 seeds each, every checkpoint evaluated closed-loop in the sim. GPU jobs run one at a time.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-$HOME/f1tenth_train/venv/bin/python}
TRAIN_MAPS=levine,Spielberg_map,comp_track,my_track
EP=${EPOCHS:-25}; N=${NEVAL:-40}; W=${WORKERS:-16}
ev() {  # ev <onnx> <outdir>
  for pert in none lidar camera nocam; do
    P=""; [ "$pert" != none ] && P="--perturb $pert"
    $PY ml/sim_rollout.py eval --policy "$1" --maps $TRAIN_MAPS,uploadtest --n $N --seed 1000 --workers $W $P \
        --out "$2/eval_$pert" > "$2/eval_$pert.log" 2>&1
    echo "  $(basename $2) $pert: $(head -c 220 $2/eval_$pert/summary.json | tr -d '\n ')"
  done
}
mkdir -p runs/baseline
echo "== baseline: models/student.onnx (trained before 9/23 on the LeRobot video dataset)"
ev models/student.onnx runs/baseline
for seed in 0 1 2; do
  echo "== seed $seed: BC, no augmentation"
  $PY ml/train_policy.py --data data/demos --out runs/bc_noaug_s$seed --epochs $EP --seed $seed \
      --cam-aug 0 --cam-drop 0 --beam-drop 0 > runs/bc_noaug_s$seed.log 2>&1
  ev runs/bc_noaug_s$seed/student.onnx runs/bc_noaug_s$seed
  echo "== seed $seed: BC + augmentation"
  $PY ml/train_policy.py --data data/demos --out runs/bc_aug_s$seed --epochs $EP --seed $seed > runs/bc_aug_s$seed.log 2>&1
  ev runs/bc_aug_s$seed/student.onnx runs/bc_aug_s$seed
  prev=runs/bc_aug_s$seed; dat="data/demos"
  for r in 1 2 3; do
    echo "== seed $seed: DAgger round $r"
    # beta: probability the expert drives a tick; decays so later rounds see the student's own mistakes
    beta=$(python3 -c "print({1:0.5,2:0.25,3:0.1}[$r])")
    $PY ml/sim_rollout.py collect --policy $prev/student.onnx --beta $beta --maps $TRAIN_MAPS --n 50 \
        --seed $((100 + 10 * seed + r)) --workers $W --out data/dagger_s${seed}_r$r > data/dagger_s${seed}_r$r.log 2>&1
    dat="$dat data/dagger_s${seed}_r$r"
    $PY ml/train_policy.py --data $dat --out runs/dagger_s${seed}_r$r --epochs $EP --seed $seed \
        --init $prev/best.pt > runs/dagger_s${seed}_r$r.log 2>&1
    prev=runs/dagger_s${seed}_r$r
    [ $r = 3 ] && ev $prev/student.onnx $prev || { $PY ml/sim_rollout.py eval --policy $prev/student.onnx \
        --maps $TRAIN_MAPS,uploadtest --n $N --seed 1000 --workers $W --out $prev/eval_none > $prev/eval_none.log 2>&1;
        echo "  $(basename $prev) none: $(head -c 220 $prev/eval_none/summary.json | tr -d '\n ')"; }
  done
done
echo ALL_DONE
