#!/usr/bin/env bash
# T24 (sim part): which image input does the student use? BC on clean demos, seed 0, same
# settings as bc_noaug_s0, with the other input zeroed inside the network.
cd "$(dirname "$0")/.."; PY=${PY:-$HOME/f1tenth_train/venv/bin/python}
for inp in lidar camera; do
  [ -f runs/abl_${inp}_s0/eval_none/summary.json ] && continue
  $PY ml/train_policy.py --data data/demos --out runs/abl_${inp}_s0 --epochs 25 --seed 0       --cam-aug 0 --cam-drop 0 --beam-drop 0 --inputs $inp > runs/abl_${inp}_s0.log 2>&1
  $PY ml/sim_rollout.py eval --policy runs/abl_${inp}_s0/student.onnx --maps levine,Spielberg_map,comp_track,my_track,uploadtest       --n 30 --seed 1000 --workers 8 --out runs/abl_${inp}_s0/eval_none > runs/abl_${inp}_s0/eval.log 2>&1
  echo "$inp: $(head -c 220 runs/abl_${inp}_s0/eval_none/summary.json | tr -d "\n ")"
done
