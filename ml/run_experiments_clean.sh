#!/usr/bin/env bash
# Follow-up experiment (9/23 evening, "improved run"): isolate DAgger's own effect.
#   * plain behaviour cloning (no augmentation) for 5 seeds (0-2 are reused if already run)
#   * DAgger rounds 1-3 that start from that plain-BC model and also train without
#     augmentation, so the only change from BC is the DAgger data
#   * every checkpoint on the same 150 seeded tasks; plain BC and DAgger round 3 are also
#     evaluated under the lidar / camera / no-camera perturbations
# Runs are named dcl_s<seed>_r<round>; data goes to data/dcl_s<seed>_r<round>.
#   SEEDS="3" bash ml/run_experiments_clean.sh
set -uo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-$HOME/f1tenth_train/venv/bin/python}
TRAIN_MAPS=levine,Spielberg_map,comp_track,my_track
EP=${EPOCHS:-25}; N=${NEVAL:-30}; W=${WORKERS:-18}
NOAUG="--cam-aug 0 --cam-drop 0 --beam-drop 0"
ev() {  # ev <onnx> <outdir> <perturbations...>   (skips evaluations that already exist)
  local onnx=$1 out=$2; shift 2
  for pert in "$@"; do
    [ -f "$out/eval_$pert/summary.json" ] && continue
    P=""; [ "$pert" != none ] && P="--perturb $pert"
    $PY ml/sim_rollout.py eval --policy "$onnx" --maps $TRAIN_MAPS,uploadtest --n $N --seed 1000 --workers $W $P \
        --out "$out/eval_$pert" > "$out/eval_$pert.log" 2>&1
    echo "  $(basename $out) $pert: $(head -c 200 $out/eval_$pert/summary.json | tr -d '\n ')"
  done
}
for seed in ${SEEDS:-0 1 2 3 4}; do
  echo "== seed $seed: BC, no augmentation"
  [ -f runs/bc_noaug_s$seed/student.onnx ] || $PY ml/train_policy.py --data data/demos --out runs/bc_noaug_s$seed \
      --epochs $EP --seed $seed $NOAUG > runs/bc_noaug_s$seed.log 2>&1
  ev runs/bc_noaug_s$seed/student.onnx runs/bc_noaug_s$seed none lidar camera nocam
  prev=runs/bc_noaug_s$seed; dat="data/demos"
  for r in 1 2 3; do
    echo "== seed $seed: DAgger (from plain BC, no augmentation) round $r"
    run=runs/dcl_s${seed}_r$r; d=data/dcl_s${seed}_r$r
    if [ ! -f $run/student.onnx ]; then
      beta=$(python3 -c "print({1:0.5,2:0.25,3:0.1}[$r])")
      [ -f $d/summary.json ] || $PY ml/sim_rollout.py collect --policy $prev/student.onnx --beta $beta --maps $TRAIN_MAPS \
          --n 50 --seed $((200 + 10 * seed + r)) --workers $W --out $d > $d.log 2>&1
      $PY ml/train_policy.py --data $dat $d --out $run --epochs $EP --seed $seed $NOAUG \
          --init $prev/best.pt > $run.log 2>&1
    fi
    dat="$dat $d"; prev=$run
    if [ $r = 3 ]; then ev $run/student.onnx $run none lidar camera nocam; else ev $run/student.onnx $run none; fi
  done
done
echo ALL_DONE
