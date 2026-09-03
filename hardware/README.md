# hardware/ — the physical AtlasAutoware car (as of 2026-09-02)

Everything here is what lives on the Jetson *outside* this ROS package, collected so
the car can be rebuilt from GitHub. See `RUNBOOK.md` for day-of-race commands.

## What is on the car
| Part | Detail |
|---|---|
| Compute | Jetson (JetPack 6 / L4T R36.5, Ubuntu 22.04, ROS 2 Humble), user `atlas`, reachable at `192.168.55.1` over the USB-C gadget link |
| Chassis | Traxxas Slash running gear on a custom deck, FDR 10.2, 0.109 m tires |
| Motor | Castle Creations 1415 2400 Kv **sensored**, 4-pole |
| ESC | Flipsky FSESC 6.7 Pro, HW 60, FW 6.06; udev `0483:5740 -> /dev/sensors/vesc` |
| Steering | Hitec D625MW on the VESC's PPM/servo header (Servo Output enabled) |
| LiDAR | Slamtec **RPLIDAR C1**, 460800 baud, CP210x `10c4:ea60 -> /dev/sensors/rplidar` |
| Camera | **Orbbec Gemini 335** (`2bc5:0800`), driven by `orbbec_camera` (OrbbecSDK_ROS2 v2-main) |
| Pad | Logitech F310 (switch on X). LB deadman, left stick throttle, right stick steer |
| Battery | 4S 9000 mAh, XT90 |

## Two stacks share the VESC
* `~/f1tenth_ws` — upstream `f1tenth_system` + `rplidar_ros`, with the overrides in
  `f1tenth_stack/` here (C1 lidar in bringup, F310 pad profile, tuned `vesc.yaml`) and the
  custom `vesc_ackermann` control modes in `f1tenth_system_patches/vesc_ackermann.diff`.
  Manual driving: `ros2 launch f1tenth_stack bringup_launch.py`.
* `~/atlas_ws` — this repo (`f1tenth_gym_ros`) + `OrbbecSDK_ROS2`. Autonomy:
  `scripts/run_selfdrive.sh` (kills the manual stack first; both talk to the same VESC).

## Folders
* `f1tenth_stack/` — drop-in replacements for `f1tenth_system/f1tenth_stack/{config,launch}`.
  After copying: `cd ~/f1tenth_ws && colcon build --packages-select f1tenth_stack`.
* `f1tenth_system_patches/` — `git diff` of the `vesc` submodule (erpm/current/speed
  control modes in `ackermann_to_vesc.cpp`). Apply with `git apply` inside the submodule.
* `udev/` — rules for VESC, RPLIDAR, Logitech pads (ROS joy needs r/w on `/dev/input/event*`),
  Orbbec. Copy to `/etc/udev/rules.d/`, then `udevadm control --reload-rules && udevadm trigger`.
  Also add the user to `input`: `sudo usermod -aG input atlas`.
* `vesc/` — the known-good motor + app config blobs and the USB tools that read/write them
  without VESC Tool (`python3 vesc/tools/vesc_mcconf3.py` decodes the motor config).
  Restore = send the blob with `COMM_SET_MCCONF`(13) / `COMM_SET_APPCONF`(16); see
  `vesc_fix_app.py` for the framing. Also keep an XML export from VESC Tool 6.06.
* `scripts/` — launcher, network, and TensorRT install/build scripts used on the Jetson.
* `RUNBOOK.md` — bring-up, health checks, calibration to-dos, gotchas.

## Gotchas that cost hours
* VESC ignores every USB command with **no fault** if `App to Use` is a PPM mode and
  Servo Output is off: the servo wire on the shared PPM pin gets decoded as an RC input.
  Fix: Servo Output ON, App = UART (or No App), then reboot the VESC.
* Reversed battery leads kill the VESC instantly (no protection). Mark the XT90.
* Slow ABS Current Limit must be ON or current ripple trips ABS_OVER_CURRENT.
* The RPLIDAR is a C1: A2/A3/S1 launch files time out on it.
* Jetson WiFi is behind a captive portal: `apt`/`git` fail there. Fetch on a laptop and
  copy over the USB link (`scp ... atlas@192.168.55.1:`).
