#!/usr/bin/env bash
# Finish a SLAM mapping session: save the map, then generate the raceline.
#
#   tools/finish_mapping.sh <track_name> [seed_x seed_y]
#
# Run after driving clean mapping laps with slam_mapping_launch.py.
set -e

NAME="${1:-track}"
SEED_X="${2:-0.0}"
SEED_Y="${3:-0.0}"
MAPS="maps"
RACELINES="racelines"

echo "[1/2] saving map -> ${MAPS}/${NAME}.{yaml,pgm}"
ros2 run nav2_map_server map_saver_cli -f "${MAPS}/${NAME}" --ros-args -p save_map_timeout:=10.0

echo "[2/2] generating raceline -> ${RACELINES}/best_raceline.csv"
python3 f1tenth_gym_ros/raceline_optimizer.py \
    --map "${MAPS}/${NAME}.yaml" \
    --output "${RACELINES}/best_raceline.csv" \
    --seed "${SEED_X}" "${SEED_Y}" \
    --margin 0.30 --apex-bias 1.0

echo "Done. Switch to localization against ${MAPS}/${NAME}.yaml and run the race agent."
