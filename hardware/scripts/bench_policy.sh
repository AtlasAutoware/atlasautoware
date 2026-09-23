#!/usr/bin/env bash
# Wheels-safe bench check of the learned policy's input/output plumbing on the car.
# Starts ONLY the lidar and the camera (no drive_node, no VESC, no racing stack), runs
# policy_bridge with its output on /drive_bench (nothing subscribes to it), and reports
# topic rates and what the policy would command. Nothing can move the car.
#   bash hardware/scripts/bench_policy.sh [model.onnx] [seconds]
set -o pipefail   # not -u: ROS setup.bash reads unset variables
source /opt/ros/${ROS_DISTRO:-jazzy}/setup.bash
source "$HOME/f1tenth_ws/install/setup.bash"   # rplidar_ros, vesc
source "$HOME/atlas_ws/install/setup.bash"
MODEL="${1:-$HOME/atlas_ws/src/atlasautoware/models/student.onnx}"; SECS="${2:-25}"
pkill -f "[c]ar_bringup_launch|[p]olicy_bridge|[o]akd_camera|[r]plidar" >/dev/null 2>&1 || true
sleep 1
ros2 launch f1tenth_gym_ros car_bringup_launch.py use_drive:=false use_racing:=false \
    camera_backend:=oakd > /tmp/bench_bringup.log 2>&1 &
BR=$!
sleep 12
ros2 run f1tenth_gym_ros policy_bridge --ros-args -p model:="$MODEL" -p drive_topic:=/drive_bench \
    -p image_topic:=/oakd/rgb -p odom_topic:=/odom -p max_speed:=0.6 \
    -p instruction:="go straight to the end and stop" > /tmp/bench_policy.log 2>&1 &
PB=$!
sleep 6
echo "== publishers on /drive (must be none from this test):"; ros2 topic info /drive 2>/dev/null | grep -i "publisher count" || echo "  /drive not present"
for t in /scan /oakd/rgb /oakd/imu /drive_bench; do
  r=$(timeout 8 ros2 topic hz $t 2>/dev/null | grep -m1 "average rate"); echo "== $t ${r:-no data}"
done
echo "== /policy/status (3 samples)"
timeout 8 ros2 topic echo /policy/status --once --field data 2>/dev/null | head -3
timeout 8 ros2 topic echo /drive_bench --once 2>/dev/null | grep -A3 "drive:" | head -4
sleep $(( SECS > 20 ? SECS - 20 : 1 ))
kill $PB 2>/dev/null; kill $BR 2>/dev/null; sleep 2
pkill -f "[c]ar_bringup_launch|[p]olicy_bridge|[o]akd_camera|[r]plidar" >/dev/null 2>&1 || true
echo "== policy_bridge log"; tail -4 /tmp/bench_policy.log
