# Model Predictive Contouring Control (MPCC) with curvature integration

The lap-time controller: where `mpc_controller.KinematicMPC` *tracks* a fixed
raceline + speed profile, `mpcc_controller.MPCC` *races* — it maximizes
progress along the track, may place the car anywhere inside the physical track
corridor (track width minus a 0.30 m wall-clearance margin), and self-enforces
the physics budget the rest of the stack assumes. The formulation follows the
MPCC family (Liniger et al. 2015; ForzaETH's F1Tenth MPCC) with the
curvature-integration ideas of CiMPCC (arXiv:2502.03695): upcoming path
curvature bounds the admissible speed per stage and shapes the contouring
weight so the solution hugs the reference at apexes and uses the track width
everywhere else.

## What it is

`f1tenth_gym_ros/mpcc_controller.py` — a **kinematic-bicycle LTV-MPCC**, pure
python/numpy/osqp, no ROS.

### Model and linearization
- state `z = [x, y, yaw, v]`, input `u = [steer, accel]`; forward-Euler
  kinematic bicycle (identical to `KinematicMPC`).
- **Real-time-iteration linearization**: each tick the previous solution's
  input sequence is resampled at `+ctrl_dt` (a *fractional* shift — control
  runs at 50 Hz but horizon stages are `dt = 0.08 s`, so a full one-stage
  shift would consume the plan 4x too fast and forever postpone planned
  braking) and rolled out through the nonlinear model from the measured
  state; the QP linearizes about that rollout. Reference stations `s_k`
  along the line are marched by the rollout speeds, so lag error is ~0 at
  the linearization point and Cartesian states suffice — no Frenet model.
- Persistent OSQP exactly like `KinematicMPC`: cost/constraint **sparsity is
  fixed once**, every tick is `update(q, Px, Ax, l, u)` + a warm-started
  solve. (`Px` is updated too because the contour/lag weights rotate with
  the local tangent/normal.)

### Cost (per stage k; t/n = unit tangent / left normal at station k)
| term | meaning |
|---|---|
| `q_lag * (t . (p - p_k))^2` | lag error — keeps the station parametrization honest |
| `q_c(kappa_k) * (n . (p - p_k) - c_tgt,k)^2` | contour error; `q_c` ramps from 4 to 24 as curvature approaches `kappa_knee` (CiMPCC weight shaping: hug the apex, roam the straights); `c_tgt` re-centres the target into the legal band where the raw line is too close to a wall |
| `q_yaw, q_v` | small heading / speed-cap regularization |
| `-gamma * v_k` | **progress reward** — replaces speed-profile tracking |
| `u' R u + du' Rd du` | input magnitude + rate smoothing |

### Hard constraints (the physics budget, self-enforced)
The closed-loop harness plant has **no grip limit**, so a controller that does
not self-enforce the budget can "win" by cornering at 20+ m/s². MPCC enforces
`|a_lat| <= 6.5 m/s²`, `v <= 7`, accel/brake `4/8` with the friction-ellipse
coupling in exactly `velocity_profiler.py`'s convention
(`ax_avail = ax_max * (1-(a_lat/a_lat_max)^p)^(1/p)`, `p = 2`):

- **corridor** (per stage, k = 1..N): `band_lo,k - sl_k <= n_k.(p - p_k) <=
  band_hi,k + sl_k`, `sl_k >= 0` with an exact L1+L2 penalty (`200*sl +
  50*sl²`) — the standard MPCC soft-corridor: identical to a hard constraint
  whenever dynamically reachable, but the QP never deadlocks infeasible
  during transients.
- **curvature-integrated speed caps**: `v_k <= sqrt(0.95 * 6.5 /
  max(|kappa_plan,k|, |kappa_ref,k|))`, then a backward brake-feasibility
  pass along the horizon (ellipse-limited decel), then `v_k <=
  vprof(s_k)` — the profiler's global budget-true profile anchors **every**
  stage, so a slow corner can never enter the horizon too late to brake for.
- **steering**: `|steer_k| <= atan(6.5 * L / v_lin,k²)` — the lateral budget
  expressed on the curvature the car actually drives.
- **longitudinal**: `-8*e_k <= a_k <= 4*e_k`, `e_k` the ellipse share at the
  stage's (budget-clamped) lateral demand.
- **output governor**: the applied command is re-clamped against the
  *measured* speed (steer to the lateral budget, accel to the ellipse), so
  the driven `|v² kappa|` respects the budget even when plan and plant
  disagree.

Two design constants matter and are worth stating for the paper:
- `plan_lat_frac = 0.95`: speed caps plan at 95% of the lateral budget; the
  steering bound and governor keep the full budget. The difference is the
  closed loop's correction authority — a plan that books 100% of the grip
  leaves no steering to remove tracking error at the apex and washes wide.
- Corridor bands threshold the **EDT** (omnidirectional wall distance >=
  0.30 m), not free space along the normal: at corner apexes and chicane
  juts the nearest wall is diagonal to the probe and a 1-D free-space probe
  systematically overestimates the room.

### Track corridor and reference repair
`TrackCorridor` loads `maps/comp_track.yaml/.png` (PIL), builds the
wall-distance field with `scipy.ndimage.distance_transform_edt` scaled by the
map resolution (row 0 = max world y — the ROS map y-flip), and probes the
legal interval along each raceline normal. `build_reference` then *repairs*
the raceline into the corridor: clip into the legal band, smooth the offsets,
**resample to 0.35 m spacing** (a wall jutting into the track *between* the
CSV's ~0.75 m-spaced points is invisible to per-point probes), and recompute
heading/curvature/bands. Every speed cap is therefore computed from the
geometry of a drivable line — this is what makes the controller survive the
hand-drawn `comp_raceline_unrefined.csv`, which runs *through* walls in
places.

- Fallback on solver failure: command (0 steer, decelerate), count the
  failure, restart the linearization from the raceline feed-forward.

## Benchmark (tools/benchmark_mpcc.py)

Shared harness (`tests/closed_loop.py`, kinematic bicycle @ 50 Hz, plant
accel/brake 4/8), same raceline, same budget (a_lat 6.5, v_max 7). All
metrics measured on the **driven trajectory**. `tracking` =
`KinematicMPC` with speeds re-profiled by `velocity_profiler` at the budget;
`+gov` = behind the same output governor MPCC uses; `+gov.80` = governed with
the profile at 0.80 of the budget — the most aggressive fraction at which the
governed tracker still holds its line on this map (its best *valid* lap).

`racelines/comp_raceline.csv` (optimizer-refined, 325 pts; numbers from one
representative run — the CSV may be regenerated, rerun the tool):

| controller | lap_s | clr_min (m) | alat_max | alat_p99.5 | fails | solve ms mean/p95/max |
|---|---|---|---|---|---|---|
| tracking | 43.22 | **0.000** | **22.03** | 16.33 | 0 | 0.50 / 0.60 / 1.7 |
| tracking+gov | 46.52 | **0.000** | 6.50 | 6.50 | 0 | 0.50 / 0.60 / 1.9 |
| tracking+gov.80 | 47.28 | **0.000** | 6.50 | 6.50 | 0 | 0.51 / 0.63 / 1.3 |
| **mpcc** | **47.24** | **0.281** | 6.44 | 6.07 | 0 | 1.42 / 1.79 / 2.8 |

`racelines/comp_raceline_unrefined.csv` (hand-drawn, passes through walls):

| controller | lap_s | clr_min (m) | alat_max | alat_p99.5 | fails | solve ms mean/p95/max |
|---|---|---|---|---|---|---|
| tracking | 44.72 | **0.000** | **27.19** | 17.19 | 0 | 0.48 / 0.56 / 1.3 |
| tracking+gov | 46.88 | **0.000** | 6.50 | 6.50 | 0 | 0.48 / 0.55 / 1.3 |
| tracking+gov.80 | 49.32 | **0.000** | 6.50 | 6.50 | 0 | 0.50 / 0.60 / 1.3 |
| **mpcc** | **50.26** | **0.307** | 6.50 | 6.15 | 0 | 1.38 / 1.60 / 3.0 |

### Honest reading

- The raw tracking MPC's 43.2 s is **not a real lap**: the harness plant has
  no grip limit and no walls, and that lap corners at up to 22-27 m/s²
  (3-4x the budget) while touching walls (clearance 0.000). Under the stated
  physics it does not exist.
- Once the budget is actually enforced (`+gov`), the full-budget profile is
  undrivable: the governor leaves no steering authority for feedback and the
  tracker leaves the track (xte_mean ~1 m on the refined line). The
  tracker's best valid configuration (`+gov.80`) does 47.28 s — and *still*
  has zero wall clearance, because the refined raceline itself passes within
  0.21 m of walls and tracking error eats the rest.
- **MPCC matches the best valid tracking lap on the refined raceline (47.24
  vs 47.28 s, a tie within noise) and is the only controller that never
  comes closer than 0.28 m to a wall**, with zero solver failures and p95
  solve time of 1.8 ms against the 20 ms budget. On the broken unrefined
  line it concedes ~1 s to a "tracker" that is literally driving through
  walls, while autonomously repairing the line into the corridor.
- The CiMPCC-style headline gains (11-12%) do **not** materialize here on
  lap time, and the reason is structural: those gains come from replacing a
  centerline (or otherwise conservative) reference with corridor-wide
  optimization, whereas this stack's offline raceline is already a
  minimum-curvature line optimized over the same corridor — there is little
  width left to exploit at equal budget. What the MPCC machinery buys
  instead, at equal budget and equal lap time, is (1) hard wall-clearance
  guarantees on the driven trajectory, (2) budget compliance enforced by
  the controller rather than assumed of the reference, and (3) robustness
  to a bad reference line. Where it would also buy lap time is any track
  whose stored raceline is *not* near-optimal — as the unrefined line shows
  qualitatively: tracking it faithfully is simply not an option.

## Validated (closed-loop, `tests/test_mpcc.py`)

One full lap on the live-loaded competition raceline + map corridor:
- completes the lap, never leaves the corridor (min EDT wall clearance of the
  driven trajectory >= 0.25 m asserted; measured 0.28 m),
- driven `|v * yaw_rate| <= 6.5 * 1.05` (measured peak 6.44),
- solver failure rate < 5% (measured 0),
- solve-time p95 < 20 ms (measured ~1.8 ms),
plus unit checks that the friction-ellipse helper matches
`velocity_profiler`'s convention and that corridor bands are legal positions.
Suite runtime ~5 s.

## Usage

```python
from mpcc_controller import MPCC, TrackCorridor, build_reference
from closed_loop import load_raceline

x, y, h, c, v = load_raceline('racelines/comp_raceline.csv')
corr = TrackCorridor('maps/comp_track.yaml')
gx, gy, gh, gk, blo, bhi = build_reference(corr, x, y)   # corridor-legal guide
mpcc = MPCC()                                            # budget defaults: 6.5/4/8, v<=7
mpcc.set_raceline(gx, gy, gh, gk, band_lo=blo, band_hi=bhi)
steer, v_target = mpcc.control(px, py, yaw, v, nearest_idx)
```

Dependency: `osqp` (same as the tracking MPC; `available == False` and the
caller falls back if it is missing).
