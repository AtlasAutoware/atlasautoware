#!/usr/bin/env bash
# Route-hint model (9/25). Same recipe as the production run (plain BC, no augmentation,
# 1200 expert episodes, 5 seeds, seed chosen on the selection task set) with one change:
# the network also gets a point 2 m ahead on the planned route, in the car frame
# (state[2:4], see policy_io.route_hint). The demos have to be re-collected because the
# old ones recorded zeros in those slots.
#   bash ml/run_route.sh prep | SEEDS="0" bash ml/run_route.sh seed | bash ml/run_route.sh select
set -uo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-$HOME/f1tenth_train/venv/bin/python}
TRAIN_MAPS=levine,Spielberg_map,comp_track,my_track
EP=${EPOCHS:-25}; N=${NEVAL:-30}; W=${WORKERS:-18}
FLAGS="--cam-aug 0 --cam-drop 0 --beam-drop 0 --state-mask 0,0,1,1,0"
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
  [ -f data/rdemos/summary.json ] || $PY ml/sim_rollout.py collect --policy expert --beta 1 --maps $TRAIN_MAPS \
      --n 100 --seed 0 --workers $W --out data/rdemos
  [ -f data/rdemos_extra/summary.json ] || $PY ml/sim_rollout.py collect --policy expert --beta 1 --maps $TRAIN_MAPS \
      --n 200 --seed 5000 --workers $W --out data/rdemos_extra
  echo PREP_DONE ;;
seed)
  for seed in ${SEEDS:-0}; do
    run=runs/route_s$seed; mkdir -p $run
    echo "== route-hint seed $seed"
    [ -f $run/student.onnx ] || $PY ml/train_policy.py --data data/rdemos data/rdemos_extra --out $run \
        --epochs $EP --seed $seed $FLAGS > $run.log 2>&1
    ev $run/student.onnx $run 3000 eval_sel none
    for p in none lidar camera nocam; do ev $run/student.onnx $run 1000 eval_$p $p; done
  done
  echo SEED_DONE ;;
select)
  python3 - <<'EOF'
import glob, json, os, shutil
rows = []
for d in sorted(glob.glob('runs/route_s*')):
    f = os.path.join(d, 'eval_sel', 'summary.json')
    if os.path.isfile(f) and os.path.isfile(os.path.join(d, 'student.onnx')):
        s = json.load(open(f)); rows.append((s['success_rate'], -s['collision_rate'], s['mean_progress'], d))
rows.sort(reverse=True)
for r in rows: print(f"{r[3]}: selection success {100*r[0]:.1f}%  collisions {-100*r[1]:.1f}%  progress {100*r[2]:.1f}%")
best = rows[0][3]; os.makedirs('runs/route', exist_ok=True)
shutil.copy(os.path.join(best, 'student.onnx'), 'runs/route/student.onnx')
std = json.load(open(os.path.join(best, 'eval_none', 'summary.json')))
json.dump({'chosen': best, 'selection_ranking': [[r[3], r[0], -r[1], r[2]] for r in rows], 'standard_eval': std},
          open('runs/route/choice.json', 'w'), indent=1)
print(f"CHOSEN {best}: standard-set success {100*std['success_rate']:.1f}%  collisions {100*std['collision_rate']:.1f}%")
EOF
  ;;
*) echo "usage: $0 prep|seed|select"; exit 1 ;;
esac
