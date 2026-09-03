#!/usr/bin/env bash
# Localize against a known map with the particle filter (publishes /pf/pose/odom).
#   ./run_localize.sh maps/my_track.yaml [init_x init_y init_theta]
# Run this alongside remote mode. Then the pilot page's pose check goes green and you can
# engage raceline mode with pose /pf/pose/odom. For an UNKNOWN space, build a map first:
#   ros2 launch f1tenth_gym_ros slam_online.launch.py     (drive around, then map_saver_cli)
source /opt/ros/humble/setup.bash
source "$HOME/atlas_ws/install/setup.bash"
MAP="${1:-$HOME/atlas_ws/src/atlasautoware/maps/my_track.yaml}"
[ -f "$MAP" ] || MAP="$HOME/atlas_ws/src/atlasautoware/$1"
IX="${2:-0.0}"; IY="${3:-0.0}"; ITH="${4:-0.0}"
echo "localizing against $MAP  (init $IX,$IY,$ITH)"
exec ros2 run f1tenth_gym_ros particle_filter --ros-args \
    -p map_yaml:="$MAP" -p scan_topic:=/scan -p odom_topic:=/odom \
    -p initial_x:=$IX -p initial_y:=$IY -p initial_theta:=$ITH
