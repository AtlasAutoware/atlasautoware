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
**Dependency:** `pip3 install osqp==0.6.3` (the 0.6.x line has prebuilt py38
wheels; newer osqp needs a build toolchain the foxy image lacks). If osqp is
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
