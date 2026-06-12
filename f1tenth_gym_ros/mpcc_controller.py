"""
Kinematic LTV Model Predictive Contouring Control (MPCC) with curvature
integration — the lap-time controller.
=======================================================================

Where `mpc_controller.KinematicMPC` *tracks* a fixed raceline + speed profile,
this controller *races*: it maximizes progress along the track and is free to
place the car anywhere inside the physical track corridor (track width minus a
safety margin), hard-constrained in the QP.  This is the MPCC family
(Liniger et al. 2015; ForzaETH's F1Tenth MPCC) with the curvature-integration
ideas of CiMPCC (arXiv:2502.03695): upcoming path curvature both bounds the
admissible speed per stage and shapes the contouring weight so the solution
hugs the reference at apexes while exploiting track width everywhere else.

Formulation (kept deliberately tractable — a sparse QP at 50 Hz):

  state  z = [x, y, yaw, v]      input u = [steer, accel]   (Cartesian, same
  forward-Euler kinematic bicycle and the same persistent-OSQP machinery as
  KinematicMPC: fixed sparsity, per-tick update(q, Px, Ax, l, u), warm start).

  Linearization: real-time-iteration style — each tick the previous solution's
  input sequence is shifted one stage and rolled out through the *nonlinear*
  model from the measured state; the QP is linearized about that rollout.
  Reference stations s_k along the raceline are marched by the rollout speeds,
  so lag error is ~0 at the linearization point and a Frenet model is not
  needed: lag/contour errors are the tangential/normal projections of the
  Cartesian position error at each station.

  cost   sum_k  q_lag * e_lag,k^2  +  q_c(kappa_k) * (e_con,k - c_tgt,k)^2
              + q_yaw * (yaw - hdg_k)^2 + q_v * (v - v_ub,k)^2 - gamma * v_k
              + u^T R u + du^T Rd du
    e_lag = t_k . (p - p_k),  e_con = n_k . (p - p_k)   (t/n = unit tangent /
    left normal at station k; p_k the raceline point).  c_tgt re-centres the
    contour target into the *legal* corridor band where the raw raceline sits
    closer than the margin to a wall (or inside one).  -gamma*v is the progress
    reward: with lag soft and the corridor hard, faster == more progress.
    q_c(kappa) = q_contour + (q_contour_apex - q_contour)*min(1,|kappa|/knee)
    is the CiMPCC-style curvature shaping (hug the line at apexes, roam the
    corridor elsewhere).

  hard constraints, per stage (the physics budget — self-enforced because the
  sim plant has no grip limit):
    corridor    band_lo,k + margin - sl_k <= n_k.(p - p_k) <= band_hi,k
                - margin + sl_k,  sl_k >= 0 penalized with an exact L1+L2
                penalty (w_lin*sl + w_quad*sl^2, w_lin far above any progress
                gain) — the standard MPCC soft-corridor: identical to a hard
                constraint whenever the band is dynamically reachable, but the
                QP stays feasible during transients (start error, corridor
                discontinuities of a rough raceline) instead of deadlocking;
    lateral     v_k <= sqrt(a_lat_max / kappa_cap,k)   with kappa_cap from the
                *planned* curvature (tan(steer_lin)/L) blended with the
                reference curvature, backward-pass brake-feasible along the
                horizon (friction-ellipse decel, velocity_profiler convention)
                and anchored beyond the horizon at the profiler's global
                budget-true speed at the terminal station;
    steering    |steer_k| <= atan(a_lat_max * L / v_lin,k^2)   (same budget,
                enforced on the curvature the car will actually drive);
    long.       -a_brake*e_k <= a_k <= a_accel*e_k,
                e_k = (1-(a_lat,k/a_lat_max)^p)^(1/p)  — the friction ellipse,
                exactly velocity_profiler.ax_avail's convention;
    plus the applied command is clamped once more against the *measured*
    speed (output governor), so the driven |v^2 * kappa| respects the budget
    even when plan and plant disagree.

  fallback   on solver failure: (0 steer, decelerate) and count the failure;
             the linearization restarts from the raceline feed-forward.

Corridor geometry comes from the occupancy map: `TrackCorridor` builds a
wall-distance field (PIL + scipy.ndimage.distance_transform_edt, scaled by the
map resolution, y-axis flipped per ROS map_server convention) and probes, along
each raceline normal, the interval where the wall clearance is at least the
0.30 m safety margin -> per-point [band_lo, band_hi].  `build_reference` then
repairs the raceline into that legal corridor (clip + smooth + densify), so the
geometry every speed cap is computed from is the geometry of a drivable line.

Pure python/numpy/osqp — no ROS.  See tests/test_mpcc.py (closed-loop safety +
budget validation) and tools/benchmark_mpcc.py (honest lap-time comparison
against the tracking MPC under the identical physics budget).
"""

import math
import os

import numpy as np

try:
    import osqp
    import scipy.sparse as sp
    _HAVE_OSQP = True
except Exception:                              # pragma: no cover
    _HAVE_OSQP = False

try:                                           # flat import (tests/tools path)
    from velocity_profiler import velocity_profile, segment_lengths
except ImportError:                            # package import
    from f1tenth_gym_ros.velocity_profiler import (velocity_profile,
                                                   segment_lengths)


def ellipse_ax_avail(a_lat_used, a_lat_max, ax_max, p=2.0):
    """Friction-ellipse share left for longitudinal accel.

    Identical convention to velocity_profiler.velocity_profile's ax_avail:
    ax = ax_max * (1 - (a_lat/a_lat_max)^p)^(1/p), a_lat clamped to the budget.
    """
    a_lat_used = min(abs(float(a_lat_used)), a_lat_max)
    radicand = max(0.0, 1.0 - (a_lat_used / a_lat_max) ** p)
    return ax_max * radicand ** (1.0 / p)


def path_normals(x, y):
    """Unit left normals of a closed polyline (central differences)."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    tx = np.roll(x, -1) - np.roll(x, 1)
    ty = np.roll(y, -1) - np.roll(y, 1)
    tn = np.hypot(tx, ty) + 1e-12
    return -ty / tn, tx / tn


def path_heading_curvature(x, y):
    """Heading + signed curvature of a closed polyline (central differences)."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    tx = np.roll(x, -1) - np.roll(x, 1)
    ty = np.roll(y, -1) - np.roll(y, 1)
    hdg = np.arctan2(ty, tx)
    seg = np.hypot(np.diff(x, append=x[0]), np.diff(y, append=y[0]))
    dang = np.roll(hdg, -1) - np.roll(hdg, 1)
    dang = (dang + math.pi) % (2.0 * math.pi) - math.pi
    kappa = dang / np.maximum(seg + np.roll(seg, 1), 1e-9)
    return hdg, kappa


def build_reference(corridor, x, y, margin=0.30, smooth_pts=3, iters=2,
                    resample_step=0.35):
    """Repair a raceline into a corridor-legal *guide line* for the MPCC.

    The MPCC's speed caps integrate the curvature of the path the car will
    actually drive.  A raw raceline can run closer to a wall than the safety
    margin (or, for a hand-drawn line, through one) — then the legal path has
    swerves the raceline's curvature column knows nothing about, and a
    curvature-based speed cap would carry corner-entry speed into a jink it
    cannot make.  This helper clips the line into the legal corridor band,
    smooths the resulting lateral offsets (circular moving average), and
    recomputes heading/curvature/bands on the repaired line, so every
    downstream consumer (velocity profiler, per-stage caps, contour target)
    sees the geometry of a drivable, in-corridor reference.

    The repaired line is then resampled at `resample_step` (m) arc spacing
    before the final corridor probe: raceline CSVs are typically spaced
    ~0.75 m, and a wall corner that juts into the track *between* two points
    is invisible to per-point probes at that spacing.

    Returns (x, y, heading, curvature, band_lo, band_hi).
    """
    x = np.asarray(x, float).copy(); y = np.asarray(y, float).copy()
    w = np.ones(2 * smooth_pts + 1) / (2 * smooth_pts + 1)
    for _ in range(max(1, int(iters))):
        nx, ny = path_normals(x, y)
        blo, bhi = corridor.corridor_bands(x, y, nx, ny,
                                           clear_min=margin + 0.05)
        lo = blo + 0.05
        hi = bhi - 0.05
        bad = hi < lo
        mid = 0.5 * (lo + hi)
        lo[bad] = mid[bad]; hi[bad] = mid[bad]
        c = np.clip(0.0, lo, hi)
        if smooth_pts > 0:
            cpad = np.concatenate([c[-smooth_pts:], c, c[:smooth_pts]])
            c = np.clip(np.convolve(cpad, w, 'valid'), lo, hi)
        x = x + c * nx
        y = y + c * ny
    if resample_step:                          # densify (see docstring)
        from scipy.interpolate import splev, splprep
        seg = np.hypot(np.diff(x, append=x[0]), np.diff(y, append=y[0]))
        n_new = max(len(x), int(round(seg.sum() / float(resample_step))))
        tck, _ = splprep([x, y], s=len(x) * 0.02 ** 2, per=True, k=3)
        u = np.linspace(0.0, 1.0, n_new, endpoint=False)
        xs, ys = splev(u, tck)
        x, y = np.asarray(xs), np.asarray(ys)
    nx, ny = path_normals(x, y)
    blo, bhi = corridor.corridor_bands(x, y, nx, ny, clear_min=margin)
    hdg, kappa = path_heading_curvature(x, y)
    return x, y, hdg, kappa, blo, bhi


# ─────────────────────────────────────────────────────────────────────────────
# Track corridor from the occupancy map
# ─────────────────────────────────────────────────────────────────────────────

class TrackCorridor:
    """Wall-distance field + per-point free corridor from a ROS map (PNG+YAML).

    Pixel convention matches the rest of the stack (raceline_optimizer.GridMap):
    image row 0 is the top = max world y, so row = (H-1) - (wy - oy)/res.
    """

    def __init__(self, yaml_path):
        import yaml
        from PIL import Image
        from scipy import ndimage
        with open(yaml_path) as f:
            meta = yaml.safe_load(f)
        img_path = meta['image']
        if not os.path.isabs(img_path):
            img_path = os.path.join(
                os.path.dirname(os.path.abspath(yaml_path)), img_path)
        img = np.array(Image.open(img_path).convert('L'))
        self.res = float(meta['resolution'])
        self.ox, self.oy = float(meta['origin'][0]), float(meta['origin'][1])
        free_th = float(meta.get('free_thresh', 0.196))
        occ_prob = img / 255.0 if int(meta.get('negate', 0)) \
            else (255 - img) / 255.0
        self.free = occ_prob < free_th
        self.H, self.W = self.free.shape
        # Euclidean distance (m) from each free pixel to the nearest wall pixel
        self.edt_m = ndimage.distance_transform_edt(self.free) * self.res

    # world -> float pixel coords
    def _rc(self, x, y):
        c = (np.asarray(x, float) - self.ox) / self.res
        r = (self.H - 1) - (np.asarray(y, float) - self.oy) / self.res
        return r, c

    def is_free(self, x, y):
        """Nearest-pixel free test (vectorized).  Out-of-map counts occupied."""
        r, c = self._rc(x, y)
        ri = np.round(r).astype(int); ci = np.round(c).astype(int)
        ok = (ri >= 0) & (ri < self.H) & (ci >= 0) & (ci < self.W)
        out = np.zeros(np.shape(ri), bool)
        out[ok] = self.free[ri[ok], ci[ok]]
        return out

    def clearance(self, x, y):
        """Bilinear wall distance (m) at world point(s); 0 outside the map."""
        scalar = np.isscalar(x)
        r, c = self._rc(np.atleast_1d(x), np.atleast_1d(y))
        inside = (r >= 0) & (r <= self.H - 1) & (c >= 0) & (c <= self.W - 1)
        r0 = np.clip(np.floor(r).astype(int), 0, self.H - 2)
        c0 = np.clip(np.floor(c).astype(int), 0, self.W - 2)
        fr = np.clip(r - r0, 0.0, 1.0); fc = np.clip(c - c0, 0.0, 1.0)
        e = self.edt_m
        val = (e[r0, c0] * (1 - fr) * (1 - fc) + e[r0 + 1, c0] * fr * (1 - fc)
               + e[r0, c0 + 1] * (1 - fr) * fc + e[r0 + 1, c0 + 1] * fr * fc)
        val[~inside] = 0.0
        return float(val[0]) if scalar else val

    def corridor_bands(self, x, y, nx, ny, max_offset=3.0, min_run=0.5,
                       smooth=1, clear_min=0.30):
        """Legal corridor interval along each point's normal.

        Returns (band_lo, band_hi): signed offsets along the *left* normal of
        the interval the car's center may occupy at each raceline point.  The
        probe thresholds the omnidirectional wall-distance field (EDT) at
        `clear_min`, not mere free space — so a position inside the band is
        guaranteed `clear_min` of wall clearance even where the nearest wall
        is diagonal to the normal (corner apexes, chicane juts), which a 1-D
        free-space probe systematically overestimates.  Robust to raceline
        points that sit on/inside a wall: among the contiguous legal runs
        along the normal, the closest run at least `min_run` wide is chosen
        (anti-aliasing slivers ignored), so a bad point gets a band fully on
        the drivable side.  `smooth` applies a min/max filter over +-smooth
        neighbours so the band never widens faster than the physical corridor
        (and single-point spikes vanish).
        """
        x = np.asarray(x, float); y = np.asarray(y, float)
        n = len(x)
        step = self.res * 0.5
        offs = np.arange(-max_offset, max_offset + 1e-9, step)
        zero = int(np.argmin(np.abs(offs)))
        lo = np.full(n, -0.05); hi = np.full(n, 0.05)
        for i in range(n):
            fr = self.clearance(x[i] + offs * nx[i],
                                y[i] + offs * ny[i]) >= max(clear_min, 1e-6)
            runs, s0 = [], None
            for j, f in enumerate(fr):
                if f and s0 is None:
                    s0 = j
                elif not f and s0 is not None:
                    runs.append((s0, j - 1)); s0 = None
            if s0 is not None:
                runs.append((s0, len(fr) - 1))
            if not runs:
                continue
            big = [r for r in runs if offs[r[1]] - offs[r[0]] >= min_run]
            if not big:                        # degenerate: take the widest
                big = [max(runs, key=lambda r: r[1] - r[0])]

            def dist0(r):
                if r[0] <= zero <= r[1]:
                    return 0.0
                return min(abs(offs[r[0]]), abs(offs[r[1]]))
            r = min(big, key=dist0)
            lo[i], hi[i] = offs[r[0]], offs[r[1]]
        if smooth:
            from scipy import ndimage
            size = 2 * int(smooth) + 1
            lo = ndimage.maximum_filter1d(lo, size, mode='wrap')
            hi = ndimage.minimum_filter1d(hi, size, mode='wrap')
        bad = hi < lo + 0.1                    # keep the band well-posed
        mid = 0.5 * (lo + hi)
        lo[bad] = mid[bad] - 0.05; hi[bad] = mid[bad] + 0.05
        return lo, hi


# ─────────────────────────────────────────────────────────────────────────────
# The MPCC controller
# ─────────────────────────────────────────────────────────────────────────────

class MPCC:
    NZ = 4                                     # state dim  [x, y, yaw, v]
    NU = 2                                     # input dim  [steer, accel]

    def __init__(self, wheelbase=0.33, horizon=20, dt=0.08,
                 max_steer=0.41, a_accel=4.0, a_brake=8.0,
                 v_min=0.0, v_max=7.0, a_lat_max=6.5, ellipse_p=2.0,
                 q_lag=8.0, q_contour=4.0, q_contour_apex=24.0,
                 kappa_knee=0.4, q_yaw=2.0, q_v=0.5, gamma_progress=2.0,
                 r_steer=0.6, r_accel=0.05, rd_steer=8.0, rd_accel=0.3,
                 margin=0.0, ctrl_dt=0.02, kappa_ref_blend=1.0,
                 plan_lat_frac=0.95,
                 w_corridor_lin=200.0, w_corridor_quad=50.0):
        self.L = float(wheelbase)
        self.N = int(horizon)
        self.dt = float(dt)
        self.max_steer = float(max_steer)
        self.a_accel = float(a_accel)
        self.a_brake = float(a_brake)
        self.v_min = float(v_min)
        self.v_max = float(v_max)
        self.a_lat_max = float(a_lat_max)
        self.ellipse_p = float(ellipse_p)
        self.q_lag = float(q_lag)
        self.q_contour = float(q_contour)
        self.q_contour_apex = float(q_contour_apex)
        self.kappa_knee = float(kappa_knee)
        self.q_yaw = float(q_yaw)
        self.q_v = float(q_v)
        self.gamma = float(gamma_progress)
        self.R = np.diag([r_steer, r_accel]).astype(float)
        self.Rd = np.diag([rd_steer, rd_accel]).astype(float)
        self.margin = float(margin)
        self.ctrl_dt = float(ctrl_dt)
        self.kappa_ref_blend = float(kappa_ref_blend)
        # Speed caps plan at plan_lat_frac of the lateral budget; the steering
        # bound + output governor keep the full budget.  The difference is the
        # closed loop's correction authority: a controller whose plan uses
        # 100% of the grip has no steering left to remove tracking error at
        # the apex, and washes wide instead.
        self.plan_lat_frac = float(plan_lat_frac)
        self.w_corr_lin = float(w_corridor_lin)
        self.w_corr_quad = float(w_corridor_quad)

        self.available = _HAVE_OSQP
        self._rl = None
        self._qp = None
        self._sol_u = None                     # previous solution inputs (N,2)
        self._u_prev = np.zeros(self.NU)       # last applied [steer, accel]
        # diagnostics
        self.fail_count = 0
        self.solve_ms = []
        if self.available:
            self._setup_qp_structure()

    # ── raceline + corridor geometry ───────────────────────────────────────────
    def set_raceline(self, x, y, hdg, curv, speed=None,
                     band_lo=None, band_hi=None):
        """Reference line (same CSV columns as KinematicMPC) + corridor bands.

        `band_lo/band_hi` are the legal-corridor offsets along the left normal
        per point (from TrackCorridor.corridor_bands, which already embeds the
        0.30 m wall-clearance margin via the EDT threshold; `self.margin` adds
        an extra along-normal buffer on top, default 0).  Defaults to a
        +-0.45 m tube (≈ pure tracking) when no corridor is supplied.  The
        speed column is NOT trusted: the budget-true profile is recomputed
        with the stack's velocity profiler at (plan_lat_frac * a_lat_max,
        a_accel, a_brake, v_max), so the horizon's terminal speed anchor
        always respects the physics budget.
        """
        x = np.asarray(x, float); y = np.asarray(y, float)
        curv = np.asarray(curv, float)
        n = len(x)
        seg = np.hypot(np.diff(x, append=x[0]), np.diff(y, append=y[0]))
        s = np.concatenate([[0.0], np.cumsum(seg)[:-1]])
        total = float(seg.sum())
        tx = np.roll(x, -1) - np.roll(x, 1)
        ty = np.roll(y, -1) - np.roll(y, 1)
        tn = np.hypot(tx, ty) + 1e-12
        tx, ty = tx / tn, ty / tn
        vprof = velocity_profile(curv, seg,
                                 a_lat_max=self.plan_lat_frac * self.a_lat_max,
                                 a_accel_max=self.a_accel,
                                 a_brake_max=self.a_brake,
                                 v_max=self.v_max, v_min=self.v_min,
                                 p=self.ellipse_p, closed=True)
        if band_lo is None or band_hi is None:
            band_lo = np.full(n, -(0.45 + self.margin))
            band_hi = np.full(n, +(0.45 + self.margin))
        rl = dict(x=x, y=y, tx=tx, ty=ty, kappa=curv, vprof=vprof,
                  blo=np.asarray(band_lo, float),
                  bhi=np.asarray(band_hi, float),
                  s=s, total=total, n=n)
        # wrap-extended copies for arc-length interpolation
        rl['s_e'] = np.append(s, total)
        for k in ('x', 'y', 'tx', 'ty', 'kappa', 'vprof', 'blo', 'bhi'):
            rl[k + '_e'] = np.append(rl[k], rl[k][0])
        self._rl = rl
        self._sol_u = None                     # geometry changed: cold restart

    def _ref_at(self, s_arr):
        """Interpolate reference quantities at arc positions (wrapping)."""
        rl = self._rl
        a = np.asarray(s_arr, float) % rl['total']
        out = {}
        for k in ('x', 'y', 'tx', 'ty', 'kappa', 'vprof', 'blo', 'bhi'):
            out[k] = np.interp(a, rl['s_e'], rl[k + '_e'])
        tn = np.hypot(out['tx'], out['ty']) + 1e-12
        out['tx'] /= tn; out['ty'] /= tn
        out['nx'], out['ny'] = -out['ty'], out['tx']
        return out

    # ── linearized discrete dynamics about an operating point ──────────────────
    def _f(self, z, u):
        L, dt = self.L, self.dt
        return np.array([z[0] + dt * z[3] * math.cos(z[2]),
                         z[1] + dt * z[3] * math.sin(z[2]),
                         z[2] + dt * z[3] * math.tan(u[0]) / L,
                         z[3] + dt * u[1]])

    def _linearize(self, z, u):
        L, dt = self.L, self.dt
        _, _, psi, v = z
        A = np.eye(4)
        A[0, 2] = -dt * v * math.sin(psi); A[0, 3] = dt * math.cos(psi)
        A[1, 2] = dt * v * math.cos(psi);  A[1, 3] = dt * math.sin(psi)
        A[2, 3] = dt * math.tan(u[0]) / L
        B = np.zeros((4, 2))
        B[2, 0] = dt * v / (L * math.cos(u[0]) ** 2)
        B[3, 1] = dt
        f = self._f(z, u)
        return A, B, f - A @ z - B @ u

    # ── persistent QP structure (fixed sparsity; values updated per tick) ──────
    # Variables: [z_0..z_N | u_0..u_{N-1} | sl_1..sl_N]; sl_k is the corridor
    # slack at stage k (exact L1+L2 penalty -> soft-hard corridor).
    def _setup_qp_structure(self):
        N, nz, nu = self.N, self.NZ, self.NU
        nZ = nz * (N + 1)
        self._nZ = nZ
        self._nS = nZ + nu * N                 # slack offset
        nv = self._nS + N
        n_eq = nz * (N + 1)
        self._row_input = n_eq
        self._row_speed = n_eq + nu * N
        self._row_corr = self._row_speed + (N + 1)
        # corridor: 2 rows (upper/lower with slack) per stage k=1..N,
        # then N slack box rows
        n_rows = self._row_corr + 3 * N

        # P: constant input/rate/slack blocks + per-tick lag/contour xy blocks
        self._Pd = np.zeros((nv, nv))
        mark = np.zeros((nv, nv), bool)
        for k in range(N + 1):
            i = k * nz
            mark[i:i + 2, i:i + 2] = True      # xy block rewritten per tick
            mark[i + 2, i + 2] = mark[i + 3, i + 3] = True
            self._Pd[i + 2, i + 2] = 2.0 * self.q_yaw
            self._Pd[i + 3, i + 3] = 2.0 * self.q_v
        for k in range(N):
            j = nZ + k * nu
            self._Pd[j:j + nu, j:j + nu] += 2.0 * self.R + 2.0 * self.Rd
            mark[j:j + nu, j:j + nu] = True
            if k > 0:
                jp = nZ + (k - 1) * nu
                self._Pd[jp:jp + nu, jp:jp + nu] += 2.0 * self.Rd
                self._Pd[j:j + nu, jp:jp + nu] += -2.0 * self.Rd
                self._Pd[jp:jp + nu, j:j + nu] += -2.0 * self.Rd
                mark[j:j + nu, jp:jp + nu] |= np.eye(nu, dtype=bool)
                mark[jp:jp + nu, j:j + nu] |= np.eye(nu, dtype=bool)
        for k in range(N):                                     # slack quad
            j = self._nS + k
            self._Pd[j, j] = 2.0 * self.w_corr_quad
            mark[j, j] = True
        Tp = sp.csc_matrix(np.triu(mark).astype(float))
        self._P_indices = Tp.indices.copy()
        self._P_indptr = Tp.indptr.copy()
        self._P_coords = (Tp.indices,
                          np.repeat(np.arange(nv), np.diff(Tp.indptr)))

        # A: dense buffer; constant blocks filled once
        self._Ad = np.zeros((n_rows, nv))
        self._Ad[:nz, :nz] = np.eye(nz)                        # z0 pin
        row = nz
        for k in range(N):                                     # dynamics
            ik1 = (k + 1) * nz
            self._Ad[row:row + nz, ik1:ik1 + nz] = np.eye(nz)
            row += nz
        for k in range(N):                                     # input box
            j = nZ + k * nu
            self._Ad[row, j] = 1.0
            self._Ad[row + 1, j + 1] = 1.0
            row += nu
        for k in range(N + 1):                                 # speed box
            self._Ad[row, k * nz + 3] = 1.0
            row += 1
        for k in range(1, N + 1):                              # corridor rows
            js = self._nS + k - 1
            self._Ad[row, js] = -1.0           # n.p - sl <= c0 + bhi
            self._Ad[row + 1, js] = 1.0        # n.p + sl >= c0 + blo
            row += 2
        for k in range(N):                                     # slack box
            self._Ad[row, self._nS + k] = 1.0
            row += 1
        # structural template (every entry any tick can touch)
        Sa = np.eye(nz)
        Sa[0, 2] = Sa[0, 3] = Sa[1, 2] = Sa[1, 3] = Sa[2, 3] = 1.0
        Sb = np.zeros((nz, nu)); Sb[2, 0] = Sb[3, 1] = 1.0
        T = (self._Ad != 0.0).astype(float)
        row = nz
        for k in range(N):
            T[row:row + nz, k * nz:k * nz + nz] = Sa
            T[row:row + nz, nZ + k * nu:nZ + k * nu + nu] = Sb
            row += nz
        row = self._row_corr
        for k in range(1, N + 1):                              # normal entries
            T[row, k * nz] = T[row, k * nz + 1] = 1.0
            T[row + 1, k * nz] = T[row + 1, k * nz + 1] = 1.0
            row += 2
        Tc = sp.csc_matrix(T)
        self._A_indices = Tc.indices.copy()
        self._A_indptr = Tc.indptr.copy()
        self._A_coords = (Tc.indices,
                          np.repeat(np.arange(nv), np.diff(Tc.indptr)))
        self._lo = np.zeros(n_rows)
        self._hi = np.zeros(n_rows)
        # constant rows: slack boxes
        self._lo[self._row_corr + 2 * N:] = 0.0
        self._hi[self._row_corr + 2 * N:] = 3.0
        self._q = np.zeros(nv)
        self._q[self._nS:] = self.w_corr_lin   # constant L1 slack penalty

    # ── one MPC tick ───────────────────────────────────────────────────────────
    def _rollout(self, state):
        """Linearization rollout + stations (RTI shift, or feed-forward)."""
        N, dt, L = self.N, self.dt, self.L
        rl = self._rl
        px, py, yaw0, v0 = state
        z_lin = np.zeros((N + 1, 4)); z_lin[0] = state
        u_lin = np.zeros((N, 2))
        s_st = np.zeros(N + 1)
        warm = self._sol_u is not None
        if warm:
            # fractional RTI shift: only ctrl_dt (not a full stage dt) elapsed
            # since the previous solution, so resample it at +ctrl_dt — a
            # full-stage shift would consume the plan dt/ctrl_dt times too
            # fast and keep sliding planned braking into the future.
            f = min(max(self.ctrl_dt / self.dt, 0.0), 1.0)
            ext = np.vstack([self._sol_u, self._sol_u[-1]])
            u_lin = (1.0 - f) * ext[:-1] + f * ext[1:]
            u_lin[:, 0] = np.clip(u_lin[:, 0], -self.max_steer, self.max_steer)
            u_lin[:, 1] = np.clip(u_lin[:, 1], -self.a_brake, self.a_accel)
        # arc position of the projection of the current pose
        j = 0 if rl is None else int(np.argmin(
            (rl['x'] - px) ** 2 + (rl['y'] - py) ** 2))
        dproj = ((px - rl['x'][j]) * rl['tx'][j]
                 + (py - rl['y'][j]) * rl['ty'][j])
        s_st[0] = rl['s'][j] + float(np.clip(dproj, -1.0, 1.0))
        for k in range(N):
            vk = max(z_lin[k, 3], 0.0)
            if not warm:                       # raceline feed-forward
                a = s_st[k] % rl['total']
                kap = np.interp(a, rl['s_e'], rl['kappa_e'])
                vp = np.interp(a, rl['s_e'], rl['vprof_e'])
                u_lin[k, 0] = np.clip(math.atan(L * kap),
                                      -self.max_steer, self.max_steer)
                u_lin[k, 1] = np.clip((vp - vk) / dt,
                                      -self.a_brake, self.a_accel)
            z_lin[k + 1] = self._f(z_lin[k], u_lin[k])
            z_lin[k + 1, 3] = max(0.0, z_lin[k + 1, 3])
            s_st[k + 1] = s_st[k] + max(vk, 0.3) * dt
        return z_lin, u_lin, s_st

    def solve(self, state, nearest=None):
        """state=(x,y,yaw,v) -> (steer, v_target) or None on failure.

        `nearest` is accepted for interface parity with KinematicMPC but the
        arc position is recomputed from the pose (the MPCC deviates from the
        line by design, so the caller's nearest index is only advisory).
        """
        if not self.available or self._rl is None:
            return None
        N, nz, nu = self.N, self.NZ, self.NU
        nZ = self._nZ
        px, py, yaw0, v0 = (float(v) for v in state)
        z_lin, u_lin, s_st = self._rollout((px, py, yaw0, v0))
        ref = self._ref_at(s_st)

        # headings from reference tangents, unwrapped onto the yaw branch
        hdg = np.arctan2(ref['ty'], ref['tx'])
        hdg[0] += 2.0 * math.pi * round((yaw0 - hdg[0]) / (2.0 * math.pi))
        for k in range(1, N + 1):
            hdg[k] += 2.0 * math.pi * round((hdg[k - 1] - hdg[k])
                                            / (2.0 * math.pi))

        # ── curvature integration: per-stage speed caps + ellipse accel ───────
        kap_plan = np.tan(u_lin[:, 0]) / self.L
        kap_plan = np.append(kap_plan, kap_plan[-1])           # (N+1,)
        kap_cap = np.maximum(np.abs(kap_plan),
                             self.kappa_ref_blend * np.abs(ref['kappa']))
        a_lat_plan = self.plan_lat_frac * self.a_lat_max
        v_ub = np.minimum(self.v_max,
                          np.sqrt(a_lat_plan / np.maximum(kap_cap, 1e-6)))
        # anchor every stage to the profiler's global budget-true profile: it
        # already encodes accel/brake feasibility through the friction
        # ellipse *beyond* the horizon, so a slow corner cannot "appear too
        # late" for an in-horizon braking plan to be feasible.
        v_ub = np.minimum(v_ub, ref['vprof'])
        v_lin = z_lin[:, 3]
        ds_st = np.maximum(v_lin[:N], 0.3) * self.dt
        for k in range(N - 1, -1, -1):                         # brake-feasible
            brake_av = ellipse_ax_avail(
                v_ub[k + 1] ** 2 * kap_cap[k + 1], self.a_lat_max,
                self.a_brake, self.ellipse_p)
            v_ub[k] = min(v_ub[k], math.sqrt(
                v_ub[k + 1] ** 2 + 2.0 * brake_av * ds_st[k]))
        # friction-ellipse accel bounds.  The lateral demand used here is
        # clamped to the *planned* lateral budget: the rollout can transiently
        # demand more than the budget (so the raw share would be 0, locking
        # out braking exactly when the plan must slow down) — but steering is
        # a decision variable, so the new plan can always trade steer for the
        # planned-budget share.  The output governor still applies the exact
        # ellipse at the measured state.
        a_lat_lin = np.minimum(v_lin[:N] ** 2 * np.abs(kap_plan[:N]),
                               self.plan_lat_frac * self.a_lat_max)
        ax_hi = np.array([ellipse_ax_avail(a, self.a_lat_max, self.a_accel,
                                           self.ellipse_p) for a in a_lat_lin])
        ax_lo = -np.array([ellipse_ax_avail(a, self.a_lat_max, self.a_brake,
                                            self.ellipse_p) for a in a_lat_lin])
        # reachability relax: the braking-only trajectory must stay feasible
        v_ub[0] = max(v_ub[0], v0 + 1e-3)
        vfloor = v0
        for k in range(N):
            vfloor = max(0.0, vfloor + self.dt * ax_lo[k])
            v_ub[k + 1] = max(v_ub[k + 1], vfloor + 1e-3)
        v_ub = np.maximum(v_ub, 0.05)
        d_max = np.minimum(self.max_steer, np.arctan(
            self.a_lat_max * self.L / np.maximum(v_lin[:N] ** 2, 0.25)))

        # ── corridor bands (lateral-offset space, after the safety margin) ────
        blo = ref['blo'] + self.margin
        bhi = ref['bhi'] - self.margin
        narrow = bhi - blo < 0.1
        mid = 0.5 * (blo + bhi)
        blo[narrow] = mid[narrow] - 0.05; bhi[narrow] = mid[narrow] + 0.05
        c_tgt = np.clip(0.0, blo + 0.05, bhi - 0.05)   # legal contour target

        # ── cost values ────────────────────────────────────────────────────────
        q = self._q
        kfrac = np.minimum(1.0, np.abs(ref['kappa']) / self.kappa_knee)
        q_c = self.q_contour + (self.q_contour_apex - self.q_contour) * kfrac
        for k in range(N + 1):
            i = k * nz
            t = np.array([ref['tx'][k], ref['ty'][k]])
            nrm = np.array([ref['nx'][k], ref['ny'][k]])
            W = self.q_lag * np.outer(t, t) + q_c[k] * np.outer(nrm, nrm)
            self._Pd[i:i + 2, i:i + 2] = 2.0 * W
            tgt = np.array([ref['x'][k] + c_tgt[k] * nrm[0],
                            ref['y'][k] + c_tgt[k] * nrm[1]])
            q[i:i + 2] = -2.0 * (W @ tgt)
            q[i + 2] = -2.0 * self.q_yaw * hdg[k]
            q[i + 3] = -2.0 * self.q_v * v_ub[k] - self.gamma
        q[nZ:self._nS] = 0.0
        q[nZ:nZ + nu] = -2.0 * self.Rd @ self._u_prev          # rate continuity

        # ── constraint values ──────────────────────────────────────────────────
        lo, hi = self._lo, self._hi
        z_init = np.array([px, py, yaw0, v0])
        lo[:nz] = hi[:nz] = z_init
        row = nz
        for k in range(N):
            Ak, Bk, gk = self._linearize(z_lin[k], u_lin[k])
            self._Ad[row:row + nz, k * nz:k * nz + nz] = -Ak
            self._Ad[row:row + nz, nZ + k * nu:nZ + k * nu + nu] = -Bk
            lo[row:row + nz] = hi[row:row + nz] = gk
            row += nz
        for k in range(N):                                     # input boxes
            lo[row], hi[row] = -d_max[k], d_max[k]
            lo[row + 1], hi[row + 1] = ax_lo[k], ax_hi[k]
            row += nu
        for k in range(N + 1):                                 # speed boxes
            lo[row], hi[row] = self.v_min, v_ub[k]
            row += 1
        for k in range(1, N + 1):                              # soft corridor
            self._Ad[row, k * nz] = ref['nx'][k]
            self._Ad[row, k * nz + 1] = ref['ny'][k]
            self._Ad[row + 1, k * nz] = ref['nx'][k]
            self._Ad[row + 1, k * nz + 1] = ref['ny'][k]
            c0 = ref['nx'][k] * ref['x'][k] + ref['ny'][k] * ref['y'][k]
            lo[row], hi[row] = -np.inf, c0 + bhi[k]     # n.p - sl <= c0+bhi
            lo[row + 1], hi[row + 1] = c0 + blo[k], np.inf  # n.p + sl >= ...
            row += 2

        A_data = self._Ad[self._A_coords]
        P_data = self._Pd[self._P_coords]
        try:
            if self._qp is None:
                P_csc = sp.csc_matrix(
                    (P_data, self._P_indices, self._P_indptr),
                    shape=(len(q), len(q)))
                A_csc = sp.csc_matrix(
                    (A_data, self._A_indices, self._A_indptr),
                    shape=self._Ad.shape)
                self._qp = osqp.OSQP()
                self._qp.setup(P=P_csc, q=q, A=A_csc, l=lo, u=hi,
                               verbose=False, warm_start=True, polish=True,
                               max_iter=4000, eps_abs=1e-4, eps_rel=1e-4)
            else:
                self._qp.update(q=q, Px=P_data, Ax=A_data, l=lo, u=hi)
            res = self._qp.solve()
        except Exception:
            self._qp = None                    # rebuild from scratch next tick
            self._sol_u = None
            return None
        if res.info.status_val not in (1, 2) or res.x is None \
                or not np.all(np.isfinite(res.x)):
            self._sol_u = None                 # cold restart the linearization
            return None

        U = res.x[nZ:self._nS].reshape(N, nu)
        self._sol_u = U.copy()
        steer, accel = float(U[0, 0]), float(U[0, 1])
        # ── output governor: budget enforced against the *measured* speed ─────
        d_cap = min(self.max_steer, math.atan(
            self.a_lat_max * self.L / max(v0 * v0, 1e-6)))
        steer = float(np.clip(steer, -d_cap, d_cap))
        a_lat_now = v0 * v0 * abs(math.tan(steer)) / self.L
        accel = float(np.clip(
            accel,
            -ellipse_ax_avail(a_lat_now, self.a_lat_max, self.a_brake,
                              self.ellipse_p),
            ellipse_ax_avail(a_lat_now, self.a_lat_max, self.a_accel,
                             self.ellipse_p)))
        v_target = float(np.clip(v0 + accel * self.ctrl_dt,
                                 self.v_min, self.v_max))
        self._u_prev = np.array([steer, accel])
        return steer, v_target

    # ── run_lap-compatible interface with fallback + diagnostics ───────────────
    def control(self, px, py, yaw, v, nearest=None):
        """control_fn(px, py, yaw, v, nearest) -> (steer, v_target)."""
        import time
        t0 = time.perf_counter()
        out = self.solve((px, py, yaw, v), nearest)
        self.solve_ms.append((time.perf_counter() - t0) * 1e3)
        if out is None:                        # fallback: 0 steer, decelerate
            self.fail_count += 1
            return 0.0, max(0.0, v - self.a_brake * self.ctrl_dt)
        return out
