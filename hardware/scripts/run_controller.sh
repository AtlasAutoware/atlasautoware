#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# MANUAL CONTROLLER MODE  (PS5 / Xbox joystick, direct current/torque control)
#
#   joystick ─► joy_teleop ─► mux ─► ackermann_to_vesc (CURRENT mode) ─► VESC
#
#   • Hold the DEADMAN to move:  Xbox = LB,  PS5 = L1
#   • Left stick  = throttle (torque),  steering = right stick
#   • Controller is auto-detected by name; override below.
#
# Usage:
#   ./run_controller.sh            # auto-detect the pad
#   ./run_controller.sh xbox       # force Xbox profile
#   ./run_controller.sh ps5        # force PS5 profile
# ─────────────────────────────────────────────────────────────────────────────
# NOTE: no `set -u` — ROS setup.bash references unset vars (AMENT_TRACE_SETUP_FILES)
source /opt/ros/humble/setup.bash
source "$HOME/f1tenth_ws/install/setup.bash"

# The self-driving stack talks to the same VESC over the same UART — make sure
# none of its nodes are holding the port before we open it.
pkill -f drive_node       >/dev/null 2>&1 || true
pkill -f raceline_mpc     >/dev/null 2>&1 || true
sleep 1

if [ "${1:-}" = "diag" ]; then
    echo "──────────────────────────────────────────────────────────────"
    echo " CONTROLLER • HARDWARE DIAGNOSTICS  (VESC, RPLidar, OAK-D, servo)"
    echo " Add 'servo' to actively sweep the steering — WHEELS OFF GROUND,"
    echo " main drive battery connected (the servo 5V comes from the VESC BEC)."
    echo "──────────────────────────────────────────────────────────────"
    DIAG="$HOME/atlas_ws/src/atlasautoware/f1tenth_gym_ros/hw_diag.py"
    if [ "${2:-}" = "servo" ]; then
        exec python3 "$DIAG" --servo-sweep
    fi
    exec python3 "$DIAG"
fi

export F1TENTH_CONTROLLER="${1:-auto}"

echo "──────────────────────────────────────────────────────────────"
echo " CONTROLLER MODE   profile = ${F1TENTH_CONTROLLER}   (current-control)"
echo " Hold the deadman (Xbox LB / PS5 L1) and push the left stick."
echo " ⚠  WHEELS OFF THE GROUND for the first run — this is direct torque."
echo "──────────────────────────────────────────────────────────────"

exec ros2 launch f1tenth_stack bringup_launch.py
