#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# REMOTE PILOT MODE  —  drive the car from any browser on the car's WiFi.
#
#   browser (WASD or gamepad) ──HTTP POST /cmd──► web_pilot ──/joy──► joy_teleop ──► mux
#   browser  ◄──── MJPEG /stream ─────────────── web_pilot ◄── orbbec_camera   ──► ackermann_to_vesc ──► VESC
#
#   1. The car is its own hotspot: SSID AtlasCar, car = 10.42.0.1 (NetworkManager,
#      autoconnect). Join it from the laptop/phone.
#   2. On the car:   ./run_remote.sh            (drive + FPV)
#                    ./run_remote.sh novideo    (drive only)
#                    PILOT_TIMEOUT=0.6 ./run_remote.sh lowbw   (cellular / Tailscale: small video, 600 ms watchdog)
#      Other networks (mesh client, 2.4 GHz hotspot, phone tether): hardware/scripts/carnet.sh
#   3. Open  http://10.42.0.1:8080/   — W/S throttle, A/D steer, or hold LB on a pad
#      plugged into the laptop (browser Gamepad API). Release everything = car stops.
#   Over the USB-C link the same page is at http://192.168.55.1:8080/ .
#   tools/remote_pilot.py (python + pygame, UDP 5005) still works as an alternative.
# ─────────────────────────────────────────────────────────────────────────────
source /opt/ros/humble/setup.bash
source "$HOME/f1tenth_ws/install/setup.bash"
source "$HOME/atlas_ws/install/setup.bash"

# One stack owns the VESC: stop whatever else is running.
pkill -f "car_bringup_launch.py"  >/dev/null 2>&1 || true
pkill -f "bringup_launch.py"      >/dev/null 2>&1 || true
pkill -f "vesc_driver_node|drive_node|orbbec_camera_node|component_container|remote_joy_bridge|mjpeg_server|web_pilot" >/dev/null 2>&1 || true
sleep 1

export F1TENTH_CONTROLLER=f310        # joy_teleop uses the F310 profile even with no pad on the Jetson
trap 'echo; echo "stopping remote mode"; kill 0' INT TERM

ros2 launch f1tenth_stack bringup_launch.py &
sleep 4
if [ "${1:-}" != "novideo" ]; then
    # Depth on: the fusion node folds it into /scan_fused so the brake and planner see
    # obstacles above/below the lidar plane. DEPTH=0 ./run_remote.sh turns it off
    # (e.g. if the USB-2 cable cannot carry both streams; 640x480@15 should fit).
    if [ "${DEPTH:-1}" = "1" ]; then DEPTH_ARGS="enable_depth:=true depth_width:=${DEPTH_W:-640} depth_height:=${DEPTH_H:-480} depth_fps:=15";
    else DEPTH_ARGS="enable_depth:=false"; fi
    ros2 launch orbbec_camera gemini_330_series.launch.py \
        enable_color:=true color_width:=640 color_height:=480 color_fps:=15 \
        $DEPTH_ARGS enable_point_cloud:=false enable_ir:=false \
        enable_accel:=true enable_gyro:=true enable_sync_output_accel_gyro:=true \
        accel_rate:=200hz gyro_rate:=200hz \
        log_level:=warn > /tmp/remote_camera.log 2>&1 &
    sleep 6
    # IMU into ROS body axes (proprioception for the episode logger; the racing stack does the same)
    ros2 run f1tenth_gym_ros imu_optical_to_body --ros-args -p in_topic:=/camera/gyro_accel/sample \
        -p out_topic:=/oakd/imu -p frame_id:=camera_link > /tmp/imu_relay.log 2>&1 &
    if [ "${DEPTH:-1}" = "1" ]; then
        ros2 run f1tenth_gym_ros depth_fusion --ros-args -p pitch_deg:=${CAM_PITCH_DEG:-0.0} \
            > /tmp/depth_fusion.log 2>&1 &
    fi
fi
# ── Prime the odometry ────────────────────────────────────────────────────────
# vesc_to_odom runs with use_servo_cmd_to_calc_angular_velocity, and its VESC-state
# callback returns early until it has seen ONE servo command. So without this, odometry
# (and therefore the odom->base_link transform, SLAM, the particle filter and the
# map-frame pose the racing stack needs) only starts existing after somebody drives.
# One neutral command unblocks it for good; the mux drops it after its 0.2 s timeout, so
# it cannot mask autonomy.
ros2 topic pub -1 /teleop ackermann_msgs/msg/AckermannDriveStamped "{}" >/dev/null 2>&1 || true

# ── Localization, on by default ───────────────────────────────────────────────
# Something has to publish a map-frame pose or self-driving can never engage. SLAM is the
# default because it needs no prior map, so it works in a room we have never mapped; it
# publishes map->odom and pose_relay turns that into /pf/pose/odom, the topic the racing
# stack reads. With a map of the space, the particle filter is cheaper and drift-free:
#   MAP=maps/my_track.yaml ./run_remote.sh      localize against that map instead
#   LOCALIZE=off ./run_remote.sh                no localization (manual driving only)
if [ "${LOCALIZE:-on}" != "off" ]; then
    if [ -n "${MAP:-}" ]; then
        [ -f "$MAP" ] || MAP="$HOME/atlas_ws/src/atlasautoware/$MAP"
        ros2 run f1tenth_gym_ros particle_filter --ros-args -p map_yaml:="$MAP" \
            -p scan_topic:=/scan -p odom_topic:=/odom > /tmp/localize.log 2>&1 &
        echo "localization: particle filter on $(basename "$MAP")"
    else
        SLAM_CFG="$HOME/atlas_ws/src/atlasautoware/config/slam_toolbox.yaml"
        ros2 run slam_toolbox async_slam_toolbox_node --ros-args --params-file "$SLAM_CFG" \
            > /tmp/slam.log 2>&1 &
        sleep 3
        ros2 run f1tenth_gym_ros pose_relay > /tmp/pose_relay.log 2>&1 &
        echo "localization: SLAM (building a map as you drive) -> /pf/pose/odom"
    fi
fi

# lowbw: cellular / Tailscale — smaller video and a longer command watchdog (PILOT_TIMEOUT, s)
if [ "${1:-}" = "lowbw" ]; then VID="-p width:=320 -p quality:=45 -p fps:=10.0"; else VID=""; fi
ros2 run f1tenth_gym_ros web_pilot --ros-args -p timeout:=${PILOT_TIMEOUT:-0.25} $VID &
# demonstration recorder: idle until the page (or /episode/cmd) starts an episode
ros2 run f1tenth_gym_ros episode_logger --ros-args -p root:=${EPISODE_ROOT:-$HOME/episodes} \
    -p image_topic:=/camera/color/image_raw -p odom_topic:=/vesc/odom -p imu_topic:=/oakd/imu \
    > /tmp/episode_logger.log 2>&1 &

echo "──────────────────────────────────────────────────────────────"
echo " REMOTE PILOT MODE ready.  Car IPs: $(hostname -I)"
echo " Browser:  http://10.42.0.1:8080/   (on the AtlasCar hotspot)"
echo "           http://192.168.55.1:8080/ (over the USB-C link)"
echo " W/S throttle, A/D steer, or hold LB on a gamepad. Ctrl-C here stops everything."
echo "──────────────────────────────────────────────────────────────"
wait
