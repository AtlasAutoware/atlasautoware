"""
Kinematic-bicycle LTV-MPC for raceline tracking.
================================================

A receding-horizon controller that, every tick, rolls the kinematic bicycle
model forward over a short horizon and solves for the steering + acceleration
sequence that best tracks the optimized raceline subject to the car's actual
limits (|steer|, accel/brake, speed) — then applies only the first command.

This is the closed-loop optimal-control upgrade over the geometric controllers
(pure pursuit / Stanley): instead of reacting to the current cross-track + a
fixed feed-forward, it *plans* through the corner using a model, which is what
lets it carry the line cleanly through the tightest hairpins where the geometric
controllers run wide.

Formulation (standard LTV-MPC):
  state  z = [x, y, yaw, v]          input u = [steer, accel]
  model  x⁺ = x + dt·v·cosψ          (forward-Euler kinematic bicycle, rear axle)
         y⁺ = y + dt·v·sinψ
         ψ⁺ = ψ + dt·v·tanδ / L
         v⁺ = v + dt·a
  Linearized about the reference trajectory each tick → a sparse QP in
  [z₀..z_N, u₀..u_{N-1}], solved with OSQP.  Cost tracks the reference pose +
  speed, penalizes input and input-rate (smooth steer), box-constrained to the
  car's limits.

Dependency: `osqp` (pip install osqp).  If it is not importable the controller
reports `available == False` so the agent can fall back to its geometric
controller — deployment never hard-breaks on a missing solver.

Pure (no ROS); unit-/sim-testable.  See tests/test_mpc.py for the closed-loop
kinematic validation.
"""

import math

import numpy as np

try:
    import osqp
    import scipy.sparse as sp
    _HAVE_OSQP = True
except Exception:                              # pragma: no cover
    _HAVE_OSQP = False


class KinematicMPC:
    NZ = 4                                      # state dim  [x, y, yaw, v]
    NU = 2                                      # input dim  [steer, accel]

    def __init__(self, wheelbase=0.33, horizon=12, dt=0.08,
                 max_steer=0.41, max_accel=4.0, max_brake=8.0,
                 v_min=0.0, v_max=8.0,
                 q_pos=14.0, q_yaw=6.0, q_v=2.5,
                 r_steer=1.0, r_accel=0.2,
                 rd_steer=12.0, rd_accel=0.5):
        self.L = float(wheelbase)
        self.N = int(horizon)
        self.dt = float(dt)
        self.max_steer = float(max_steer)
        self.max_accel = float(max_accel)
        self.max_brake = float(max_brake)
        self.v_min = float(v_min)
        self.v_max = float(v_max)
        # cost weights (diagonal)
        self.Q = np.diag([q_pos, q_pos, q_yaw, q_v]).astype(float)
        self.Qf = self.Q * 2.0                  # heavier terminal tracking
        self.R = np.diag([r_steer, r_accel]).astype(float)
        self.Rd = np.diag([rd_steer, rd_accel]).astype(float)
        self.available = _HAVE_OSQP
        self._u_prev = np.zeros(self.NU)        # last applied [steer, accel]
        # raceline (set via set_raceline)
        self._rl = None

    # ── raceline geometry ──────────────────────────────────────────────────────
    def set_raceline(self, x, y, hdg, curv, speed):
        x = np.asarray(x, float); y = np.asarray(y, float)
        hdg = np.asarray(hdg, float)
        speed = np.asarray(speed, float)
        n = len(x)
        # cumulative arc length around the closed loop (for time-parametrized ref)
        dx = np.diff(x, append=x[0]); dy = np.diff(y, append=y[0])
        seg = np.hypot(dx, dy)
        s = np.concatenate([[0.0], np.cumsum(seg)[:-1]])
        total = float(seg.sum())
        # left normals (for applying a strategic lateral offset to the reference)
        tx = np.roll(x, -1) - np.roll(x, 1)
        ty = np.roll(y, -1) - np.roll(y, 1)
        tn = np.hypot(tx, ty) + 1e-9
        nx, ny = -ty / tn, tx / tn
        self._rl = dict(x=x, y=y, hdg=hdg, v=np.clip(speed, self.v_min, self.v_max),
                        s=s, total=total, n=n, nx=nx, ny=ny)

    def _idx_at_arc(self, arc):
        rl = self._rl
        a = arc % rl['total']
        j = int(np.searchsorted(rl['s'], a))
        return j % rl['n']

    # ── reference trajectory over the horizon ──────────────────────────────────
    def _reference(self, nearest, offset):
        """Build zr (N+1, 4) and ur (N, 2), marching along the line by v·dt.

        The reference is time-parametrized (advance arc length by the reference
        speed each step) so the MPC's temporal horizon lines up with where the
        car will actually be — short on the straights, dense in the corners.
        `offset` shifts the reference laterally (strategic line) along the normal.
        """
        rl = self._rl
        N, dt = self.N, self.dt
        idxs = []
        cur_s = rl['s'][nearest % rl['n']]
        for _ in range(N + 1):
            j = self._idx_at_arc(cur_s)
            idxs.append(j)
            cur_s += max(rl['v'][j], 0.5) * dt
        zr = np.zeros((N + 1, 4))
        prev_h = rl['hdg'][idxs[0]]
        for k, j in enumerate(idxs):
            h = rl['hdg'][j]
            h += 2.0 * math.pi * round((prev_h - h) / (2.0 * math.pi))  # unwrap
            prev_h = h
            zr[k, 0] = rl['x'][j] + offset * rl['nx'][j]
            zr[k, 1] = rl['y'][j] + offset * rl['ny'][j]
            zr[k, 2] = h
            zr[k, 3] = rl['v'][j]
        ur = np.zeros((N, 2))
        for k in range(N):
            ds = max(rl['v'][idxs[k]] * dt, 1e-3)
            kappa = (zr[k + 1, 2] - zr[k, 2]) / ds          # signed path curvature
            ur[k, 0] = np.clip(math.atan(self.L * kappa),
                               -self.max_steer, self.max_steer)
            ur[k, 1] = (zr[k + 1, 3] - zr[k, 3]) / dt        # ref accel
        return zr, ur

    # ── linearized discrete dynamics about an operating point ──────────────────
    def _linearize(self, z, u):
        L, dt = self.L, self.dt
        x, y, psi, v = z
        delta, a = u
        A = np.eye(4)
        A[0, 2] = -dt * v * math.sin(psi); A[0, 3] = dt * math.cos(psi)
        A[1, 2] = dt * v * math.cos(psi);  A[1, 3] = dt * math.sin(psi)
        A[2, 3] = dt * math.tan(delta) / L
        B = np.zeros((4, 2))
        B[2, 0] = dt * v / (L * math.cos(delta) ** 2)
        B[3, 1] = dt
        f = np.array([x + dt * v * math.cos(psi),
                      y + dt * v * math.sin(psi),
                      psi + dt * v * math.tan(delta) / L,
                      v + dt * a])
        g = f - A @ z - B @ u                                # affine offset
        return A, B, g

    # ── solve one MPC step ─────────────────────────────────────────────────────
    def solve(self, state, nearest, offset=0.0):
        """state=(x,y,yaw,v). Returns (steer, v_target) or None on failure."""
        if not self.available or self._rl is None:
            return None
        N, nz, nu = self.N, self.NZ, self.NU
        zr, ur = self._reference(nearest, offset)

        x0, y0, yaw0, v0 = state
        yaw0 += 2.0 * math.pi * round((zr[0, 2] - yaw0) / (2.0 * math.pi))  # branch
        z_init = np.array([x0, y0, yaw0, v0])

        nZ = nz * (N + 1)
        nU = nu * N
        nv = nZ + nU

        # ── cost  0.5 xᵀP x + qᵀx ──────────────────────────────────────────────
        P = np.zeros((nv, nv))
        q = np.zeros(nv)
        for k in range(N + 1):
            Qk = self.Qf if k == N else self.Q
            i = k * nz
            P[i:i + nz, i:i + nz] += 2.0 * Qk
            q[i:i + nz] += -2.0 * Qk @ zr[k]
        for k in range(N):
            j = nZ + k * nu
            P[j:j + nu, j:j + nu] += 2.0 * self.R
            q[j:j + nu] += -2.0 * self.R @ ur[k]
        # input-rate penalty (incl. continuity with the last applied command)
        for k in range(N):
            j = nZ + k * nu
            P[j:j + nu, j:j + nu] += 2.0 * self.Rd
            if k == 0:
                q[j:j + nu] += -2.0 * self.Rd @ self._u_prev
            else:
                jp = nZ + (k - 1) * nu
                P[jp:jp + nu, jp:jp + nu] += 2.0 * self.Rd
                P[j:j + nu, jp:jp + nu] += -2.0 * self.Rd
                P[jp:jp + nu, j:j + nu] += -2.0 * self.Rd

        # ── constraints ────────────────────────────────────────────────────────
        # equalities: initial state + dynamics;  inequalities: input + speed box
        n_eq = nz * (N + 1)
        n_in = nu * N + (N + 1)
        A = np.zeros((n_eq + n_in, nv))
        lo = np.zeros(n_eq + n_in)
        hi = np.zeros(n_eq + n_in)
        # z0 = z_init
        A[:nz, :nz] = np.eye(nz)
        lo[:nz] = hi[:nz] = z_init
        # z_{k+1} = A_k z_k + B_k u_k + g_k
        row = nz
        for k in range(N):
            Ak, Bk, gk = self._linearize(zr[k], ur[k])
            ik, ik1 = k * nz, (k + 1) * nz
            jk = nZ + k * nu
            A[row:row + nz, ik1:ik1 + nz] = np.eye(nz)
            A[row:row + nz, ik:ik + nz] = -Ak
            A[row:row + nz, jk:jk + nu] = -Bk
            lo[row:row + nz] = hi[row:row + nz] = gk
            row += nz
        # input box:  -max_steer ≤ δ ≤ max_steer ;  -max_brake ≤ a ≤ max_accel
        for k in range(N):
            j = nZ + k * nu
            A[row, j] = 1.0;     lo[row] = -self.max_steer; hi[row] = self.max_steer
            A[row + 1, j + 1] = 1.0; lo[row + 1] = -self.max_brake; hi[row + 1] = self.max_accel
            row += nu
        # speed box on each state:  v_min ≤ v_k ≤ v_max
        for k in range(N + 1):
            A[row, k * nz + 3] = 1.0
            lo[row] = self.v_min; hi[row] = self.v_max
            row += 1

        try:
            prob = osqp.OSQP()
            prob.setup(P=sp.csc_matrix(np.triu(P)), q=q, A=sp.csc_matrix(A),
                       l=lo, u=hi, verbose=False, warm_start=True,
                       max_iter=4000, eps_abs=1e-3, eps_rel=1e-3)
            res = prob.solve()
        except Exception:
            return None
        if res.info.status_val not in (1, 2):   # 1 solved, 2 solved_inaccurate
            return None

        sol = res.x
        steer = float(np.clip(sol[nZ], -self.max_steer, self.max_steer))
        accel = float(sol[nZ + 1])
        v_pred = float(sol[nz + 3])              # predicted v one step ahead
        v_target = float(np.clip(v_pred, self.v_min, self.v_max))
        self._u_prev = np.array([steer, accel])
        return steer, v_target
