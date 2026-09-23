#!/usr/bin/env bash
# 2026-09-23 experiment pipeline (laptop RTX 5070): BC without/with augmentation, then 3 rounds
# of DAgger, 3 seeds each; every checkpoint is evaluated closed-loop in the sim on the same
# seeded task set (30 tasks x 5 maps; uploadtest is never trained on). GPU jobs run one at a time.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-$HOME/f1tenth_train/venv/bin/python}
TRAIN_MAPS=levine,Spielberg_map,comp_track,my_track
EP=${EPOCHS:-25}; N=${NEVAL:-30}; W=${WORKERS:-18}
ev() {  # ev <onnx> <outdir> <perturbations...>
  local onnx=$1 out=$2; shift 2
  for pert in "$@"; do
    P=""; [ "$pert" != none ] && P="--perturb $pert"
    $PY ml/sim_rollout.py eval --policy "$onnx" --maps $TRAIN_MAPS,uploadtest --n $N --seed 1000 --workers $W $P \
        --out "$out/eval_$pert" > "$out/eval_$pert.log" 2>&1
    echo "  $(basename $out) $pert: $(head -c 200 $out/eval_$pert/summary.json | tr -d '\n ')"
  done
}
for seed in ${SEEDS:-0 1 2}; do
  echo "== seed $seed: BC, no augmentation"
  [ -f runs/bc_noaug_s$seed/eval_none/summary.json ] || { $PY ml/train_policy.py --data data/demos --out runs/bc_noaug_s$seed --epochs $EP --seed $seed \
      --cam-aug 0 --cam-drop 0 --beam-drop 0 > runs/bc_noaug_s$seed.log 2>&1
  ev runs/bc_noaug_s$seed/student.onnx runs/bc_noaug_s$seed none; }
  echo "== seed $seed: BC + augmentation"
  [ -f runs/bc_aug_s$seed/eval_nocam/summary.json ] || { $PY ml/train_policy.py --data data/demos --out runs/bc_aug_s$seed --epochs $EP --seed $seed > runs/bc_aug_s$seed.log 2>&1
  ev runs/bc_aug_s$seed/student.onnx runs/bc_aug_s$seed none nocam; }
  prev=runs/bc_aug_s$seed; dat="data/demos"
  for r in 1 2 3; do
    echo "== seed $seed: DAgger round $r"
    if [ -f runs/dagger_s${seed}_r$r/eval_none/summary.json ]; then prev=runs/dagger_s${seed}_r$r; dat="$dat data/dagger_s${seed}_r$r"; continue; fi
    # beta = probability the expert drives a tick; it decays so later rounds visit the
    # states the student itself gets into. Labels always come from the expert.
    beta=$(python3 -c "print({1:0.5,2:0.25,3:0.1}[$r])")
    $PY ml/sim_rollout.py collect --policy $prev/student.onnx --beta $beta --maps $TRAIN_MAPS --n 50 \
        --seed $((100 + 10 * seed + r)) --workers $W --out data/dagger_s${seed}_r$r > data/dagger_s${seed}_r$r.log 2>&1
    dat="$dat data/dagger_s${seed}_r$r"
    $PY ml/train_policy.py --data $dat --out runs/dagger_s${seed}_r$r --epochs $EP --seed $seed \
        --init $prev/best.pt > runs/dagger_s${seed}_r$r.log 2>&1
    prev=runs/dagger_s${seed}_r$r
    if [ $r = 3 ]; then ev $prev/student.onnx $prev none lidar camera nocam; else ev $prev/student.onnx $prev none; fi
  done
done
echo ALL_DONE
