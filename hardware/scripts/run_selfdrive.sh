#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# SELF-DRIVING MODE  (autonomous racing, direct current/torque control)
#
#   lidar + camera ─► raceline_mpc ─► /drive ─► drive_node (CURRENT mode) ─► VESC
#
#   drive_node converts the planner's speed to MOTOR CURRENT (config/hardware.yaml
#   control_mode: current) so the car launches like the manual path instead of
#   stalling in the VESC RPM loop's low-speed dead zone.
#
# Usage:
#   ./run_selfdrive.sh            # full autonomy (MPC drives the car)
#   ./run_selfdrive.sh bench      # sensors + actuation ONLY, no autonomy (safe)
#   ./run_selfdrive.sh use_camera:=false   # pass any launch arg through
# ─────────────────────────────────────────────────────────────────────────────
# NOTE: no `set -u` — ROS setup.bash references unset vars (AMENT_TRACE_SETUP_FILES)
source /opt/ros/humble/setup.bash
source "$HOME/atlas_ws/install/setup.bash"

# The manual stack talks to the same VESC over the same UART — free the port.
pkill -f vesc_driver_node     >/dev/null 2>&1 || true
pkill -f ackermann_to_vesc    >/dev/null 2>&1 || true
pkill -f "bringup_launch.py"  >/dev/null 2>&1 || true
sleep 1

if [ "${1:-}" = "diag" ]; then
    echo "──────────────────────────────────────────────────────────────"
    echo " SELF-DRIVE • HARDWARE DIAGNOSTICS  (VESC, RPLidar, OAK-D, servo)"
    echo " Add 'servo' to actively sweep the steering — WHEELS OFF GROUND,"
    echo " main drive battery connected (the servo 5V comes from the VESC BEC)."
    echo "──────────────────────────────────────────────────────────────"
    DIAG="$HOME/atlas_ws/src/atlasautoware/f1tenth_gym_ros/hw_diag.py"
    if [ "${2:-}" = "servo" ]; then
        exec python3 "$DIAG" --servo-sweep
    fi
    exec python3 "$DIAG"
fi

if [ "${1:-}" = "bench" ]; then
    echo "──────────────────────────────────────────────────────────────"
    echo " SELF-DRIVE • BENCH MODE   (lidar + actuation, NO autonomy)"
    echo " RPLidar publishes /scan; drive_node up; nothing auto-commands /drive."
    echo " (camera off; check scans:  ros2 topic hz /scan)"
    echo "──────────────────────────────────────────────────────────────"
    exec ros2 launch f1tenth_gym_ros car_bringup_launch.py \
        use_racing:=false use_lidar:=true use_camera:=false
fi

echo "──────────────────────────────────────────────────────────────"
echo " SELF-DRIVE MODE   (current-control, MPC autonomous)"
echo " ⚠⚠  WHEELS OFF THE GROUND — the MPC starts commanding /drive"
echo "      ~2 s after launch (arm_time).  Be ready on the kill switch."
echo " Needs your localization (/pf/pose/odom) running for sane motion."
echo " Sensors-only first?   ./run_selfdrive.sh bench"
echo "──────────────────────────────────────────────────────────────"

exec ros2 launch f1tenth_gym_ros car_bringup_launch.py "$@"
