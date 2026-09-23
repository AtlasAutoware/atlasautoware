# Remote piloting: the web UI, networks, range, and steering trim

## The web pilot (what runs on the car)
`~/run_remote.sh` starts the manual F1TENTH stack (VESC, lidar, teleop chain), the Orbbec in
colour-only mode, and `web_pilot` (`f1tenth_gym_ros/web_pilot.py`), which serves one page:

| URL | What |
|---|---|
| `http://<car>:8080/` | FPV stream, W/S throttle, A/D steer, max-throttle slider, Q/E steering trim, video quality, gamepad via the browser (hold LB), lidar plot, link latency and battery |
| `/stream` | MJPEG only (VLC, a phone, a second screen) |
| `/status`, `/scan`, `/trim` | JSON: link/battery, downsampled lidar, current trim |

`<car>` is `10.42.0.1` on the car's own hotspot, `192.168.55.1` over the USB-C cable, its DHCP
address on a mesh, or `atlascar` over Tailscale. `ubuntu.local` works wherever mDNS does.

Safety: only the tab that is driving sends commands (at 30 Hz); releasing every key or LB sends
neutral; if the car hears nothing for the watchdog period (250 ms on WiFi; set
`PILOT_TIMEOUT=0.6` before `run_remote.sh` for cellular) it publishes neutral and stops; the VESC's
own 1 s timeout sits behind that. `~/restart_remote.sh` restarts everything cleanly.

## Networks — `hardware/scripts/carnet.sh` (on the car, needs sudo)
| Mode | Command | Range / notes |
|---|---|---|
| Hotspot 5 GHz (default) | `carnet.sh hotspot` | ~30–50 m open air, best bandwidth. Car = 10.42.0.1. Autoconnects at boot. |
| Hotspot 2.4 GHz | `carnet.sh hotspot24` | roughly 2× the range, fewer frames; use outdoors / big rooms |
| **Mesh / any WiFi as client** | `carnet.sh client <SSID> [pw]` | the car roams between mesh nodes (eero, Orbi, Deco, a school WLAN without client isolation), so range = the mesh footprint. Power-save is disabled for low latency. Laptop joins the same network; find the car via `carnet.sh status` or `ubuntu.local`. |
| Phone tether (uplink) | `carnet.sh tether` | phone on USB with USB tethering on; the car gets internet; pair with Tailscale below |
| Status | `carnet.sh status` | interface, SSID, signal, IPs |

The first `hotspot`/`hotspot24` call needs `ATLASCAR_PSK=<password>` in the environment to create the
profile; afterwards it is stored by NetworkManager. One radio: hotspot and client modes exclude
each other. A note from bring-up: `nmcli device wifi hotspot ... band a` picks channel 7 (a 2.4 GHz
channel) and then fails with "supplicant-timeout"; the script sets channel 36 explicitly.

## Beyond WiFi: cellular / anywhere (Tailscale)
1. Give the car internet: `carnet.sh tether` (phone on USB), or a USB LTE modem (Quectel EC25 /
   Huawei E3372-class sticks show up as a USB-Ethernet device and work the same way), or any WiFi.
2. Once: `hardware/scripts/install_tailscale.sh`, then `sudo tailscale up --hostname atlascar` and
   log in from the laptop. Install Tailscale on the laptop with the same account.
3. From then on, from any network: `http://atlascar:8080/`. Set video to **low** (320 px, q45,
   10 fps ≈ 0.5–1 Mbit/s) and start `PILOT_TIMEOUT=0.6 ~/run_remote.sh lowbw`.

Expect 60–150 ms control latency on LTE (WiFi is 1–5 ms). That is fine for driving at walking pace
and awkward above ~3 m/s; drop `max` on the page accordingly. MJPEG over HTTP is the simplest
transport and tolerates loss badly; if you go this route seriously, the next step is WebRTC
(e.g. `aiortc` or a GStreamer `webrtcbin` pipeline) for adaptive, low-latency video.

Other range options, in order of effort: a directional 2.4 GHz antenna on the pit-side laptop
(a $30 panel antenna on a USB WiFi adapter gives several hundred metres line of sight to the car's
2.4 GHz hotspot); a pair of Ubiquiti/Mikrotik 2.4 GHz radios as a long-range bridge; and, for
control only, an ExpressLRS/LoRa link into the VESC's PPM input (no video, kilometres of range).

## Steering trim (the car pulls right)
The page's **Q/E keys** (or the ‹ › buttons) nudge a trim that the car applies to the steering axis
of every source (keyboard, gamepad, UDP client) and stores in `~/.atlascar_trim.json`, so it
survives restarts. Drive straight on a flat floor, tap Q until it tracks straight, done. Hover the
trim value to see the equivalent change to `steering_angle_to_servo_offset` in
`f1tenth_stack/config/vesc.yaml` (Δoffset ≈ −0.4126 × trim); apply that there and in
`config/hardware.yaml` (`steer_trim_us` for `drive_node`) so the **autonomy** stack drives straight
too — the web trim only affects the web pilot path.

## Robustness
`vesc_driver_node`, `ackermann_to_vesc_node` and `joy` respawn automatically (2 s) in the manual
bring-up: the F1TENTH VESC driver aborts with `std::system_error` when the VESC's USB re-enumerates
(battery blip, cable bump), which used to leave the page up and the car dead until a restart.

## Self-driving from the browser

The panel on the left of the pilot page (link: "panel" in the help bar) engages
`raceline_mpc`, the same racing node `run_selfdrive.sh` uses.

**How it is wired.** The autonomy node publishes AckermannDrive on `/drive`. The manual
bring-up's `ackermann_mux` already accepts that as its `navigation` input at priority 10,
while human teleop on `/teleop` is priority 100. So the policy is an extra publisher, not
a mode switch: holding a key or the gamepad dead-man outranks it instantly, and letting go
hands control back. There is nothing to unwind and no state machine to get stuck in.

**Stopping it.** Four independent things stop the car: the STOP button, Space or Escape,
the page going quiet (it heartbeats twice a second; two seconds of silence kills the node,
so closing the tab or losing WiFi stops the car), and `raceline_mpc` exiting. Underneath
those sit the node's own AEB, the mux timeout, and the VESC's 1 s command timeout.

**Preflight.** Engage is refused unless the lidar, the VESC telemetry and the chosen pose
topic are all publishing and the selected raceline file exists; the panel shows which check
failed. The pose topic is selectable because `raceline_mpc` needs a map-frame pose:
`/pf/pose/odom` is the localization output, `/ekf/odom` is dead reckoning (drifts, fine for
a short straight test), `/vesc/odom` is wheel odometry only. Start the speed cap low (30 %)
and raise it a lap at a time.

## Track pictures

The same panel turns a picture of a track into a raceline, using the offline tools:

    picture -> track_from_image.py -> maps/<name>.{pgm,yaml} -> build_raceline.py
            -> racelines/<name>_auto.csv  (+ overlay PNG + feasibility report)

Two kinds of picture work. **photo** is a phone photo of a drawn track or a whiteboard
sketch: dark ink is treated as wall, the lighting is flattened, the image is thresholded,
and the drivable area is the free region the loop encloses. **map** is an occupancy grid
that already exists (a SLAM `.pgm`). Everything outside the lane is written as *occupied*
rather than "unknown", because map_server's trinary thresholds classify the usual 205-grey
as free, which lets the optimizer drive off the paper and report zero wall clearance.

Scale comes from the **lane width**: say how wide the driving lane is in metres, and the
converter measures it in pixels along the distance-transform ridge (the lane centreline)
and derives the map resolution. Measuring the ridge matters; averaging over all free pixels
underestimates the width because most of them sit off-centre.

Upload and convert takes a second or two and shows a preview (green drivable, dark wall).
"Build raceline" then runs the minimum-curvature optimizer and the closed-loop validator,
which takes about a minute and prints a feasibility report: wall clearance, planned lateral
acceleration against the grip budget, and whether a simulated lap completes. Read it before
engaging. A REVIEW verdict usually means a corner is too tight for the lane width, and the
optimizer has slowed that corner to the creep speed rather than failing.

Verified end to end on a synthetic photo of a hand-drawn loop: measured lane 1.00 m against
a 1.0 m target, 21.4 m of track, simulated lap 5.4 s. The engage path itself has been
exercised only against stubbed ROS topics; it has not yet driven the real car.

## Depth fusion (Gemini 335 depth into the scan)

The lidar sees one plane about 11 cm off the floor. `depth_fusion` back-projects the
Gemini 335 depth image into `base_link`, keeps points between `z_min` and `z_max` above
the floor, collapses them to a per-bearing minimum range, and publishes `/scan_fused`:
the lidar scan with, inside the cameras

## Depth fusion (Gemini 335 depth into the scan)

The lidar sees one plane about 11 cm off the floor. `depth_fusion` back-projects the
Gemini 335 depth image into `base_link`, keeps points between `z_min` and `z_max` above
the floor, collapses them to a per-bearing minimum range, and publishes `/scan_fused`:
the lidar scan with, inside the camera's field of view, the closer of lidar and depth per
bearing. Anything that reads `/scan` can read `/scan_fused` unchanged; the pilot page
engages `raceline_mpc` with `scan_topic:=/scan_fused` automatically when fusion is live,
and draws the depth-only virtual scan in orange on the lidar plot.

Remote mode starts depth at 640x480@15 (`DEPTH=0 ./run_remote.sh` to disable, `DEPTH_W` and
`DEPTH_H` to shrink it if the USB-2 cable cannot carry it). The floor filter depends on the
camera mount height and pitch (`cam_z`, `pitch_deg` in hardware.yaml, or `CAM_PITCH_DEG` for
remote mode): if orange points appear on open floor a metre or two ahead, the pitch is off.
Verified offline on a synthetic scene (floor, a 12 cm box, a wall): the box is absent from
the lidar and present in the fused scan at 1.20 m. Not yet run on the car.

## Recording demonstrations

`episode_logger` runs with remote mode and records nothing until an episode is started from
the panel (instruction text, REC, then stop as good or bad, or discard). Each episode is a
directory under `~/episodes` with the colour frames at 10 Hz, per-frame snapshots of lidar,
odometry, IMU, the Ackermann command that actually drove the car (human or policy) and the
raw joystick axes, plus the IMU and the commands at full rate. On the laptop,
`tools/episodes_to_lerobot.py ~/episodes --out <dataset> --only good` repackages them into a
LeRobot v2.1 dataset with the lidar rasterised as a bird's-eye video, which is the input
the open VLA fine-tuning scripts expect.

## Simulator (sim_env) — policy test bench + data collection

`sim_env` is a closed-loop 1/10 simulator that speaks the car's exact interface, so the
same nodes drive it as drive the real car:

    publishes  /scan  /odom  /pf/pose/odom (ground truth)  /camera/color/image_raw (synthetic)  /sim/state
    consumes   /teleop, /drive   (teleop wins when fresh, like the real mux)

Physics is a kinematic bicycle on any occupancy map (maps/*.yaml); the lidar is ray-cast
against the map; the camera is a distance-shaded raycast render (a placeholder, not
photorealistic — a vision policy trained on it will have a sim-to-real gap).

**Run it (no hardware, isolated on ROS_DOMAIN_ID=7):**

    ./run_sim.sh maps/levine.yaml            # drive with WASD in the browser at :8080
    # or point a policy at it:
    ROS_DOMAIN_ID=7 ros2 run f1tenth_gym_ros raceline_mpc --ros-args \
        -p raceline:=racelines/levine_auto.csv -p odom_topic:=/pf/pose/odom -p v_scale:=0.6

**Verified:** the raceline MPC policy read the sim's /scan + /pf/pose/odom and drove the
sim car ~29 m along a generated levine raceline (up to 4.2 m/s) in closed loop. The
episode logger records sim drives in the car's format, so `run_sim.sh` is also a
data-collection environment. `/sim/state` reports distance, collisions, and (with a goal
set) distance-to-goal and a `reached` flag — a success signal for policy evaluation.

**Perf note:** on the Jetson's ARM CPU the synthetic camera render caps the sim at ~5 Hz;
run with `-p camera:=false` for full-rate closed-loop policy eval, or run the sim on a
desktop (with ROS 2) for faster data collection. A Qwen-Drive-style policy would attach
here exactly like raceline_mpc — consume /scan + /camera + /odom, publish /drive.

## Home router (AtlasNet), the range fix

A dedicated router beats the Jetson's own hotspot for FPV range. `carnet.sh home <SSID> <password>`
creates a NetworkManager profile with a fixed address (default 192.168.0.250), autoconnect
priority 20 (the AtlasCar hotspot is 10), and WiFi power-save off, and tries WPA3 then WPA2.
The car joins the router whenever it is in range and falls back to being its own AP when it is
not, with no manual switching. Verified 3 Sept 2026: joined AtlasNet (WPA3, 2.4 GHz, -34 dBm,
162 Mbit/s); from a laptop wired to the router the pilot page answers in about 25 ms and ping is
about 4 ms. Pilot page: http://192.168.0.250:8080/ . Range is now the router's coverage.

## Localization is on by default

Self-driving cannot engage without a map-frame pose, so remote mode now starts localization
itself. The default is SLAM, because it needs no prior map and therefore works in a room
nobody has mapped: slam_toolbox publishes `map -> odom`, and `pose_relay` turns that into
`/pf/pose/odom`, the topic the racing stack reads. Options:

    ./run_remote.sh                      SLAM (default), builds a map as you drive
    MAP=maps/my_track.yaml ./run_remote.sh   particle filter against a known map instead
    LOCALIZE=off ./run_remote.sh         no localization

**The odometry trap.** `vesc_to_odom` runs with `use_servo_cmd_to_calc_angular_velocity`,
and its VESC-state callback returns early until it has received one servo command. Until
somebody drives, it therefore publishes no odometry at all, so there is no
`odom -> base_link` transform, and SLAM discards every scan with "queue is full" while the
particle filter sits waiting for a motion model. This is why `/odom` looked healthy at
37 Hz during a driving session and was silent after a reboot. Remote mode now publishes one
neutral `/teleop` message at startup to prime it; the mux drops that after its 0.2 s
timeout, so it cannot mask autonomy. Verified 3 Sept 2026: `/odom` 43 Hz, `/tf` carrying
both `map->odom` and `odom->base_link`, `/pf/pose/odom` 20 Hz in the map frame, and the
pilot page's pose preflight green without any manual step.

## Fixing the steering permanently

Trim works but it is the wrong home for a fix: it is applied per command by web_pilot, so a
right command plus a left trim can never reach full right lock, and it lives outside the
config, so autonomy and a bare manual bring-up never see it. The centre belongs in
`steering_angle_to_servo_offset`.

**Convert a trim you already dialled in.** Since `servo = gain * angle + offset`, adding a
trim `t` to every command is identical to shifting the centre by `gain * t`:

    ./bake_trim.sh --dry     # show the numbers
    ./bake_trim.sh           # write vesc.yaml, zero the trim, rebuild, restart

Applied 3 Sept 2026: trim -0.150 (= -0.0510 rad) with gain -1.2135 moved the centre
0.5304 -> 0.5923, and the trim went back to 0 with full throw available again.

**Measure it instead of feeling for it.** `calibrate_steering` drives straight at a low
speed with zero commanded steering and reads the yaw rate; on a centred car that is zero, so
anything else is the bias, `delta = yaw_rate * wheelbase / speed`, and the centre follows as
`offset_true = offset - gain * delta`. It runs several passes, drops the acceleration
transient, takes the median, brakes on the lidar, and always publishes zero speed on exit.

    ros2 run f1tenth_gym_ros calibrate_steering --ros-args -p speed:=0.6 -p passes:=3
    #   ... add -p apply:=true to write the result to vesc.yaml

**The real root cause is mechanical.** A centre of 0.5923 sits well away from the middle of
the servo range (0.525 for the current 0.15-0.90 limits), which is why the throw is now
lopsided: about 0.364 rad of left lock against 0.254 rad of right. Software can put the car
straight, but it cannot give back the right lock the misalignment ate. The fix that does is
at the linkage: centre the servo horn and adjust the steering turnbuckles so the wheels
point straight when the servo sits near 0.5, then re-run the calibration. Both sides then
get roughly 0.31 rad and the controller's `max_steer` is honest in both directions.
