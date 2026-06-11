# Race Control dashboard

A custom web UI to run the whole pipeline and watch it live — no RViz, no CLI.

## Run it
```
python ui/server.py            # on the host (Windows), stdlib only — no installs
# open http://localhost:8000
```
The server shells into the sim container (`f1tenth_gym_ros-sim-1`) for the heavy
work and reads result images + live telemetry straight from the mounted repo, so
the browser sees everything without a ROS bridge.

## What it does
**Raceline studio** (left)
- Sliders for **wall clearance** (`--margin`), **late-apex/overtaking bias**
  (`--apex-bias`), **lateral grip** (`--a-lat`), **top speed**.
- *Generate* re-runs the optimizer + annotator in the container and shows the
  annotated overlay (corners, apex speeds, overtake zones) + stats (length, lap
  estimate, speed range).

**Race control** (right)
- *Start / Stop* the 2-car opponent demo.
- Live **telemetry** polled 5×/s: mode badge (CRUISE/ATTACK/DEFEND/EVADE), the
  agent's reasoning text, speed, lap, opponent count.
- A live **mini-map**: the speed-colored raceline with the ego (white triangle)
  and opponents (red=lidar, blue=camera, green=fused) moving in real time.

## How it's wired
- `race_agent.py` writes `runtime/race_state.json` 10×/s (mounted → host reads it).
- `ui/server.py` endpoints: `/api/generate`, `/api/race/{start,stop}`,
  `/api/state`, `/api/raceline`, `/api/image/<name>`.
- For the **real car**, run the server where it can reach the car's ROS graph and
  point `CONTAINER`/commands at the car instead of the sim.
