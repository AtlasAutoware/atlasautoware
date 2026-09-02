# RoboRacer car runbook (updated 2026-09-02)

## Bring up everything
    ros2 launch f1tenth_stack bringup_launch.py
Starts: joy, joy_teleop, ackermann_to_vesc, vesc_to_odom, vesc_driver, ackermann_mux,
static base_link->laser TF, and the RPLIDAR C1 (rplidar_node). No separate lidar launch needed.

Quick health check (second shell):
    ros2 topic hz /scan            # ~10 Hz  (RPLIDAR C1, 460800 baud, /dev/sensors/rplidar)
    ros2 topic hz /sensors/core    # ~50 Hz  (VESC FW 6.6, /dev/sensors/vesc)
    ros2 topic echo /sensors/core --once | grep -E "voltage_input|fault_code"
    ros2 topic hz /odom            # only after the first joystick/servo command

## Hardware facts
- Lidar: RPLIDAR C1 (S/N 9B2F...), NOT an A2. A2/A3/S1 launches time out on it.
- VESC: Flipsky FSESC 6.7 Pro, HW 60, FW 6.06. udev: 0483:5740 -> /dev/sensors/vesc.
  VESC only enumerates on USB when the battery is connected.
- Motor: Castle 1415 2400Kv sensored, 4-pole (2 pole pairs). FDR 10.2, tire 0.109 m.
- Servo: Hitec D625MW.

## vesc.yaml (src/f1tenth_system/f1tenth_stack/config/vesc.yaml)
- speed_to_erpm_gain = 3575 (was 4614, the reference-car value). Used by odometry and by
  control_mode "speed"; NOT used by "erpm" mode. Verify: drive a measured 5 m, compare /odom.
- control_mode "erpm": stick fraction mapped between min_erpm 3000 and max_erpm 10000
  (about 0.8 to 2.8 m/s with this gearing). Written to avoid the sensorless stall/smoke.
  With hall sensors that stall zone is gone, so "speed" mode (true m/s closed loop, what
  autonomy nodes expect) is viable again - bench test on a stand before switching.
- STILL TO CALIBRATE with the D625MW: steering_angle_to_servo_offset (straight-ahead value),
  servo_min / servo_max (lock-to-lock), steering_angle_to_servo_gain. Current values are
  reference-car defaults.
- wheelbase (vesc_to_odom_node) is 0.25 by default: measure axle-to-axle on this chassis.
- static TF base_link->laser is x=0.27 z=0.11: measure the C1's real position.

## Config edits need a rebuild (install is a copy, not a symlink)
    cd ~/f1tenth_ws && colcon build --packages-select f1tenth_stack

## Backups
    config/vesc.yaml.bak, launch/bringup_launch.py.bak (pre-2026-09-02 versions)

## VESC gotchas learned the hard way (2026-09-02)
- The PPM header and the servo output are the SAME pin. "Enable Servo Output" (App Settings >
  General) flips it from RC input to servo output. A fresh Flipsky board ships with servo output
  OFF and App = PPM+UART, so the servo wire gets decoded as an RC input and the PPM app zeroes
  every USB command: motor commands do nothing, no fault code, R/L detection returns 0/0.
  Fix = Enable Servo Output + App to Use = UART (or No App), then REBOOT the VESC.
- After any VESC config change that touches the servo/PPM pin, reboot the VESC before testing.
- Limits set on this board: Motor 60/-60 A, Abs 120 A, Slow ABS Current Limit ON, Battery 99/-60 A.
- Config backup: ~/f1tenth_ws/vesc_backup/{mcconf,appconf}_good_2026-09-02.bin plus the python
  tools used to read/write them over USB without VESC Tool (vesc_mcconf3.py decodes the motor
  config; vesc_fix_app.py / vesc_fix_limits.py show how to write). Also save an XML from VESC Tool.
- Pad: Logitech F310, switch on X. LB = deadman, left stick = throttle, right stick = steer.
  Profile auto-selected (joy_teleop_f310.yaml). RB = autonomy deadman.
- Speed cap in erpm mode: max_erpm 10000 = ~2.8 m/s with this gearing. Raise max_erpm in vesc.yaml
  for faster laps (20000 = ~5.6 m/s), rebuild, test on a stand first.

## Camera perception (2026-09-02)
- YOLOv8n car detector: models/car_yolov8.onnx (416, CPU/onnxruntime 17 fps) + _640 for TensorRT later.
- Enable with use_perception:=true; feeds /camera_opponents_poses to race_agent only.
- pip3 --user onnxruntime on the Jetson; if it installs NumPy 2, pip3 uninstall numpy (system 1.21 must win).

## Remote pilot mode (2026-09-02)
- The car is its own hotspot: SSID AtlasCar (NetworkManager connection, autoconnect, 5 GHz ch 36), car = 10.42.0.1.
  Internet on the Jetson instead: sudo nmcli con up iPhone ; back: sudo nmcli con up AtlasCar.
- On the car: ~/run_remote.sh (or ~/restart_remote.sh to bounce it). Then open http://10.42.0.1:8080/
  (http://192.168.55.1:8080/ over USB-C): FPV stream, W/S throttle, A/D steer, max-throttle slider,
  gamepad via the browser (hold LB), lidar plot. Release everything = neutral; 250 ms watchdog on the car.
- Only the tab that is driving sends commands; other tabs are viewers. Python client: tools/remote_pilot.py (UDP 5005).
- Never pkill by a pattern that appears in your own command line (it kills the shell); use the scripts.
