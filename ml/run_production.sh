#!/usr/bin/env bash
# Production model for the car (9/24). Recipe = the best one from the 9/23 experiments:
# plain behaviour cloning, no augmentation, no DAgger. What changes: 3x the expert data
# (data/demos + data/demos_extra, 1200 episodes) and several seeds, with the final model
# chosen on a SEPARATE selection task set (seed 3000) so the reported numbers on the
# standard set (seed 1000) are not inflated by picking the luckiest seed.
#   bash ml/run_production.sh prep            # expert demos (once)
#   SEEDS="0" bash ml/run_production.sh seed  # train + evaluate one seed
#   bash ml/run_production.sh select          # pick the best seed -> runs/prod/student.onnx
set -uo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-$HOME/f1tenth_train/venv/bin/python}
TRAIN_MAPS=levine,Spielberg_map,comp_track,my_track
EP=${EPOCHS:-25}; N=${NEVAL:-30}; W=${WORKERS:-18}
NOAUG="--cam-aug 0 --cam-drop 0 --beam-drop 0"
ev() {  # ev <onnx> <outdir> <taskseed> <name> <perturbation>
  local onnx=$1 out=$2 tseed=$3 name=$4 pert=$5
  [ -f "$out/$name/summary.json" ] && return
  P=""; [ "$pert" != none ] && P="--perturb $pert"
  $PY ml/sim_rollout.py eval --policy "$onnx" --maps $TRAIN_MAPS,uploadtest --n $N --seed $tseed --workers $W $P \
      --out "$out/$name" > "$out/$name.log" 2>&1
  echo "  $(basename $out) $name: $(head -c 200 $out/$name/summary.json | tr -d '\n ')"
}
case "${1:-}" in
prep)
  [ -f data/demos/summary.json ] || $PY ml/sim_rollout.py collect --policy expert --beta 1 --maps $TRAIN_MAPS \
      --n 100 --seed 0 --workers $W --out data/demos
  [ -f data/demos_extra/summary.json ] || $PY ml/sim_rollout.py collect --policy expert --beta 1 --maps $TRAIN_MAPS \
      --n 200 --seed 5000 --workers $W --out data/demos_extra
  echo PREP_DONE ;;
seed)
  for seed in ${SEEDS:-0}; do
    run=runs/prod_s$seed; mkdir -p $run
    echo "== production seed $seed"
    [ -f $run/student.onnx ] || $PY ml/train_policy.py --data data/demos data/demos_extra --out $run \
        --epochs $EP --seed $seed $NOAUG > $run.log 2>&1
    ev $run/student.onnx $run 3000 eval_sel none           # selection set, used only to pick a seed
    for p in none lidar camera nocam; do ev $run/student.onnx $run 1000 eval_$p $p; done
  done
  echo SEED_DONE ;;
select)
  python3 - <<'EOF'
import glob, json, os, shutil
rows = []
for d in sorted(glob.glob('runs/prod_s*')):
    f = os.path.join(d, 'eval_sel', 'summary.json')
    if os.path.isfile(f) and os.path.isfile(os.path.join(d, 'student.onnx')):
        s = json.load(open(f)); rows.append((s['success_rate'], -s['collision_rate'], s['mean_progress'], d))
rows.sort(reverse=True)
for r in rows: print(f"{r[3]}: selection success {100*r[0]:.1f}%  collisions {-100*r[1]:.1f}%  progress {100*r[2]:.1f}%")
best = rows[0][3]; os.makedirs('runs/prod', exist_ok=True)
shutil.copy(os.path.join(best, 'student.onnx'), 'runs/prod/student.onnx')
std = json.load(open(os.path.join(best, 'eval_none', 'summary.json')))
json.dump({'chosen': best, 'selection_ranking': [[r[3], r[0], -r[1], r[2]] for r in rows], 'standard_eval': std},
          open('runs/prod/choice.json', 'w'), indent=1)
print(f"CHOSEN {best}: standard-set success {100*std['success_rate']:.1f}%  collisions {100*std['collision_rate']:.1f}%")
EOF
  ;;
*) echo "usage: $0 prep|seed|select"; exit 1 ;;
esac
