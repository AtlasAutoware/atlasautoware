#!/usr/bin/env bash
# Simulator data-collection + policy-eval environment. Runs the sim world, the teleop
# chain, the web pilot (FPV + WASD + predicted path), and the episode logger -- no
# hardware touched. Drive it in the browser, or run a policy against it (raceline_mpc,
# or a policy server) to test the driving policy in closed loop.
#
#   ./run_sim.sh [maps/<map>.yaml] [start_x start_y start_theta]
# Then open http://<this-host>:8080/ . Isolated on ROS_DOMAIN_ID=7 so it never collides
# with a real car on the network.
export ROS_DOMAIN_ID="${SIM_DOMAIN:-7}"
source /opt/ros/humble/setup.bash
source "$HOME/f1tenth_ws/install/setup.bash" 2>/dev/null
source "$HOME/atlas_ws/install/setup.bash"
MAP="${1:-$HOME/atlas_ws/src/atlasautoware/maps/levine.yaml}"
[ -f "$MAP" ] || MAP="$HOME/atlas_ws/src/atlasautoware/$1"
SX="${2:-0.0}"; SY="${3:-0.0}"; STH="${4:-0.0}"
echo '{"steer_trim":0.0}' > "$HOME/.atlascar_sim_trim.json"   # sim has no mechanical bias
pkill -f "f1tenth_gym_ros/sim_env" 2>/dev/null
trap 'echo; echo stopping sim; kill 0' INT TERM

ros2 run f1tenth_gym_ros sim_env --ros-args -p map_yaml:="$MAP" \
    -p start_x:=$SX -p start_y:=$SY -p start_theta:=$STH -p camera:=true &
sleep 2
# teleop chain so the browser (via /joy) drives the sim, exactly like the car
export F1TENTH_CONTROLLER=f310
ros2 launch f1tenth_stack joy_teleop_launch.py 2>/dev/null &
ros2 run f1tenth_gym_ros web_pilot --ros-args -p trim_file:="$HOME/.atlascar_sim_trim.json" \
    -p default_trim:=0.0 -p image_topic:=/camera/color/image_raw &
ros2 run f1tenth_gym_ros episode_logger --ros-args -p root:=${EPISODE_ROOT:-$HOME/sim_episodes} \
    -p image_topic:=/camera/color/image_raw -p odom_topic:=/odom -p imu_topic:=/none &
echo "──────────────────────────────────────────────"
echo " SIM ready (ROS_DOMAIN_ID=$ROS_DOMAIN_ID).  http://$(hostname -I | awk '{print $1}'):8080/"
echo " Drive with WASD, or run a policy:  ROS_DOMAIN_ID=$ROS_DOMAIN_ID ros2 run f1tenth_gym_ros raceline_mpc \\"
echo "   --ros-args -p raceline:=<csv> -p odom_topic:=/pf/pose/odom -p v_scale:=0.5"
echo "──────────────────────────────────────────────"
wait
