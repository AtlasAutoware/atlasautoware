#!/usr/bin/env bash
# Milestones T12/T19 on the bench, with NO actuation: drive_node is never started, the policy
# publishes on /drive_bench and raceline_mpc on /nav_drive, and nothing subscribes to either.
#  A) microbenchmarks: student on ONNX Runtime (CPU) and TensorRT FP16, alone and while the
#     YOLOv8-640 TensorRT engine runs flat out on the GPU
#  B) the real nodes together: lidar + OAK-D + policy_bridge + camera_perception (TensorRT)
#     + raceline_mpc; control-period jitter from `ros2 topic hz`, load from tegrastats
#   bash hardware/scripts/bench_concurrency.sh <student.onnx> <student_fp16.engine>
set -o pipefail
source /opt/ros/${ROS_DISTRO:-jazzy}/setup.bash
source "$HOME/f1tenth_ws/install/setup.bash"; source "$HOME/atlas_ws/install/setup.bash"
REPO="$HOME/atlas_ws/src/atlasautoware"
ONNX="${1:-$REPO/models/student.onnx}"; SENG="${2:-/tmp/student_fp16.engine}"
YENG="${XDG_CACHE_HOME:-$HOME/.cache}/atlasautoware/car_yolov8_640.engine"
S=front:1x3x96x128,bev:1x1x96x96,state:1x5,ids:1x24
lat() { grep -E "^\[.*\] \[I\] Latency:" | sed -E 's/.*mean = ([0-9.]+) ms.*percentile\(99%\) = ([0-9.]+) ms.*/mean \1 ms, p99 \2 ms/'; }
echo "== A1 student TensorRT FP16 alone:   $(trtexec --loadEngine=$SENG --shapes=$S --duration=10 2>&1 | lat)"
echo "== A2 student ORT CPU alone:         $(python3 $REPO/ml/bench_onboard.py $ONNX --n 300 | python3 -c 'import json,sys;d=json.load(sys.stdin);print(d["ort_cpu_4thr_ms"])')"
trtexec --loadEngine=$YENG --duration=40 > /tmp/yolo_load.log 2>&1 & Y=$!
sleep 4
echo "== A3 student TensorRT FP16 + YOLO:   $(trtexec --loadEngine=$SENG --shapes=$S --duration=10 2>&1 | lat)"
echo "== A4 student ORT CPU + YOLO:         $(python3 $REPO/ml/bench_onboard.py $ONNX --n 300 | python3 -c 'import json,sys;d=json.load(sys.stdin);print(d["ort_cpu_4thr_ms"])')"
wait $Y; echo "== A5 YOLO while sharing:             $(lat < /tmp/yolo_load.log)"

pkill -f "[c]ar_bringup_launch|[p]olicy_bridge|[o]akd_camera|[r]plidar|[c]amera_perception|[r]aceline_mpc" >/dev/null 2>&1; sleep 1
ros2 launch f1tenth_gym_ros car_bringup_launch.py use_drive:=false use_racing:=false camera_backend:=oakd > /tmp/cc_bringup.log 2>&1 &
sleep 12
ros2 run f1tenth_gym_ros camera_perception --ros-args -p backend:=tensorrt -p model_path:=$YENG -p image_topic:=/oakd/rgb > /tmp/cc_perc.log 2>&1 &
ros2 run f1tenth_gym_ros raceline_mpc --ros-args --params-file $REPO/config/hardware.yaml -r /drive:=/nav_drive > /tmp/cc_mpc.log 2>&1 &
ros2 run f1tenth_gym_ros policy_bridge --ros-args -p model:="$ONNX" -p drive_topic:=/drive_bench -p image_topic:=/oakd/rgb \
    -p max_speed:=0.6 -p instruction:="go straight to the end and stop" > /tmp/cc_policy.log 2>&1 &
sleep 10
echo "== B  publishers on /drive: $(ros2 topic info /drive 2>/dev/null | grep -i 'publisher count' || echo 'topic absent')"
(timeout 22 tegrastats --interval 1000 > /tmp/cc_tegra.log 2>&1 &)
for t in /drive_bench /scan /oakd/rgb /camera_opponents_poses; do
  r=$(timeout 12 ros2 topic hz $t 2>/dev/null | grep -A1 -m1 "average rate" | tr '\n' ' '); echo "== B  $t ${r:-no data}"
done
sleep 2
echo "== B  tegrastats (last line): $(tail -1 /tmp/cc_tegra.log | cut -c1-260)"
pkill -f "[c]ar_bringup_launch|[p]olicy_bridge|[o]akd_camera|[r]plidar|[c]amera_perception|[r]aceline_mpc" >/dev/null 2>&1
echo "== policy log: $(tail -1 /tmp/cc_policy.log | cut -c1-160)"; echo "== perception log: $(grep -iE 'backend|engine|error' /tmp/cc_perc.log | head -2 | cut -c1-160)"
