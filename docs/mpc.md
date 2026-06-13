# Model Predictive Control (raceline tracking)

An opt-in MPC lateral+longitudinal controller for `race_agent`, as the
closed-loop upgrade over the geometric controllers (pure pursuit / Stanley) at
the tightest corners.

## What it is
`f1tenth_gym_ros/mpc_controller.py` — a **kinematic-bicycle LTV-MPC**. Each tick
it linearizes the bicycle model about the reference trajectory over a short
horizon and solves a sparse QP (via **OSQP**) for the steering + acceleration
sequence that best tracks the optimized raceline subject to the car's real
limits (`|steer|`, accel/brake, speed). Only the first command is applied;
receding-horizon repeats next tick.

- state `z = [x, y, yaw, v]`, input `u = [steer, accel]`
- cost: track reference pose + speed, penalize input and **input-rate** (smooth
  steering), box-constrained to limits
- reference is **time-parametrized** (marched along the line by `v·dt`) so the
  horizon lands where the car will actually be — sparse on straights, dense in
  corners

## How to enable
```
# default is Stanley; opt in to MPC:
ros2 run f1tenth_gym_ros race_agent --ros-args -p controller:=mpc
```
**Dependency:** `pip3 install osqp==0.6.3` (the 0.6.x line has prebuilt py310
wheels; newer osqp needs a build toolchain the humble image lacks). If osqp is
missing, the agent logs a warning and **falls back to Stanley** — it never
hard-breaks. A runtime solve failure also falls back, per-tick, silently.

## Validated (closed-loop, `tests/test_mpc.py`)
Simulates the true kinematic plant under MPC on the real `comp_raceline.csv`,
starting ~0.5 m + 14° off to force a recovery:
- completes the lap, **0 solver failures**
- steady-state cross-track: **mean 0.13 m, worst-corner 0.40 m**
- solve time: **mean 0.8 ms, max 3.7 ms** (x86) vs the 20 ms @50 Hz budget — huge
  margin even after ARM slowdown on the car

Caveat: the test plant is the same kinematic model the MPC plans with, so it
validates the formulation/signs/real-time cost — not tire slip or actuator lag.
Those show up only on hardware; tune `q_*` / `r_*` / `rd_*` weights and `horizon`
in `KinematicMPC.__init__` if the real car chatters or runs wide.

## Tuning knobs (`KinematicMPC.__init__`)
- `horizon`, `dt` — preview length (default 12 × 0.08 s ≈ 1 s)
- `q_pos`, `q_yaw`, `q_v` — tracking weights (raise `q_pos` to hug the line)
- `r_steer`, `r_accel` — input effort
- `rd_steer`, `rd_accel` — input-rate (raise to smooth steering, lower for sharper
  corner response)

## Actuator-delay compensation (`predict_state` + `raceline_mpc`)
Sensor→actuator latency (serial links, servo lag, ESC ramp; 50–150 ms on a real
car) makes every solve plan from a stale state. The fix is to integrate the
kinematic bicycle forward by the measured delay and solve from the *predicted*
state, with the nearest raceline index recomputed from the predicted pose
(`raceline_mpc._loop` does both; set `-p actuation_delay:=<s>`).

**What the prediction must integrate — the in-flight command pipeline, not the
last command.** During the next `delay` seconds the actuator executes the
commands issued over the *previous* `delay` seconds; the newest command hasn't
reached the wheels yet. `predict_state` therefore accepts `steer`/`v_cmd`
either as scalars (one command held over the window — fine for delays under
~2 control periods) or as equal-length sequences of the in-flight commands,
**oldest first**, each holding an equal slice of the window. `raceline_mpc`
keeps that buffer (`_cmd_buf`, one entry per control tick over the delay
window) and feeds it to `predict_state` each loop. Feeding only the newest
command over-rotates the prediction whenever the steer is changing (every
corner entry/exit) and measurably destabilizes the loop for delays ≥ 0.08 s —
at 0.12 s the car cut corners with xte_mean 0.51 m.

Validated ROS-free with `python3 tools/benchmark_delay.py` (sweeps the delay,
naive vs the node's compensated flow, on `closed_loop.run_lap` /
`comp_raceline.csv`; `--json` for machine-readable output):

| delay (s) | naive lap / xte_mean / xte_max | compensated lap / xte_mean / xte_max |
|-----------|-------------------------------|--------------------------------------|
| 0.00 | 41.48 s / 0.123 / 0.406 | 41.48 s / 0.123 / 0.406 |
| 0.04 | 40.40 s / 0.111 / 0.354 | 41.52 s / 0.122 / 0.405 |
| 0.08 | 43.74 s / 0.195 / 0.538 | 41.54 s / 0.122 / 0.405 |
| 0.10 | 50.48 s / 0.322 / 0.804 | 41.56 s / 0.121 / 0.404 |
| 0.12 | 59.86 s / 0.461 / 1.108 | 41.58 s / 0.121 / 0.404 |
| 0.14 | 69.52 s / 0.588 / 1.381 | 41.60 s / 0.121 / 0.404 |

i.e. pipeline-aware compensation holds zero-delay tracking across the whole
sweep (the compensator is given the true delay). Regression-tested in
`tests/test_mpc.py::test_delay_compensated_lap_tracks_tightly` (0.10 s delay,
must lap with xte_mean < 0.2 m).

## Rejected: curvature-aware speed-tracking weight
Scaling the per-step `q_v` by `1 + k·|kappa_ref|` (enforce the speed profile
harder into tight corners, via per-tick OSQP `Px` updates) was implemented and
benchmarked at zero delay: every gain tried (0.5–4.0) *regressed* tracking
(xte_mean 0.125–0.134 vs 0.123 baseline, no xte_max improvement, lap time
+0.0–0.1 s). On the kinematic plant there is no grip limit for the profile to
protect, so the extra speed tracking only trades away position tracking. The
change was reverted; revisit only with a tire-slip plant or on hardware.
