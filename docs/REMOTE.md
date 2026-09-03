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
