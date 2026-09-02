#!/bin/bash
# Stop any running remote-pilot session and start a fresh one (detached, log in /tmp/remote.log).
for pat in "run_remote.sh" "bringup_launch.py" "car_bringup_launch.py" "vesc_driver_node" "drive_node" \
           "orbbec_camera_node" "component_container" "remote_joy_bridge" "mjpeg_server" "web_pilot" "rplidar"; do
    for pid in $(pgrep -f "$pat"); do [ "$pid" != "$$" ] && kill "$pid" 2>/dev/null; done
done
sleep 2
nohup "$HOME/run_remote.sh" ${1:-} > /tmp/remote.log 2>&1 &
echo "remote mode restarting (pid $!) — see /tmp/remote.log"
