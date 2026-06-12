"""
Minimum-curvature raceline refinement — corridor-bounded QP (pure, no ROS).
===========================================================================

The raceline CSVs were fit from driven laps, so they follow where the car
*went*, not the geometrically fastest line.  This module refines such a
closed line the way the TUMFTM global planner does: shift every point
laterally by d_i along its left normal n_i so the squared curvature of the
shifted path is minimal, while no point strays more than `corridor` meters
from where it started — the corridor is the safety knob, kept conservative
because the true wall clearance is unknown.

Linearization (Heilmeier): new points p_i' = p_i + d_i * n_i, and curvature
is approximated by the periodic second arc-length difference of positions,

  kappa_i ~ | (D (p + N d))_i |,   D row i:  2/(ds_{i-1}+ds_i) *
            [ 1/ds_{i-1},  -(1/ds_{i-1} + 1/ds_i),  1/ds_i ]  (wrapped),

so  J(d) = ||D (p + N d)||^2 + ridge ||d||^2 + smooth ||L d||^2  is a convex
QP in d (L = periodic first difference of d per arc length, which keeps the
offset profile — and hence the line — smooth).  Solved with OSQP under the
box constraint |d_i| <= corridor (scipy lsq_linear fallback), then the
geometry is re-linearized and re-solved `iterations` times, recomputing the
normals each pass; the box shrinks by the displacement already spent, so the
total excursion from the ORIGINAL line never exceeds `corridor`.

Heading and curvature of the refined line are recomputed with the same
periodic finite differences the raceline CSVs use (central-difference
heading, unsigned Menger curvature — see raceline_optimizer.curvature_heading;
pursuit_agent.fit_raceline uses the same stencil but omits the factor 2).

Reference: Heilmeier et al., "Minimum curvature trajectory planning and
control for an autonomous race car", Vehicle System Dynamics 2020;
github.com/TUMFTM/global_racetrajectory_optimization.
"""

import numpy as np

try:
    import osqp
    import scipy.sparse as _sp
    _HAVE_OSQP = True
except ImportError:                                   # pragma: no cover
    _HAVE_OSQP = False


def segment_lengths(x, y):
    """ds[i] = |p_{i+1} - p_i| with periodic wrap."""
    return np.hypot(np.diff(x, append=x[0]), np.diff(y, append=y[0]))


def heading_curvature(x, y):
    """Periodic heading + unsigned curvature, matching the CSV conventions.

    heading[i] = atan2 of the central difference p_{i+1} - p_{i-1};
    curvature[i] = Menger curvature of (p_{i-1}, p_i, p_{i+1}) = 4*Area/(abc).
    """
    x = np.asarray(x, float); y = np.asarray(y, float)
    xa, ya = np.roll(x, 1), np.roll(y, 1)             # p_{i-1}
    xc, yc = np.roll(x, -1), np.roll(y, -1)           # p_{i+1}
    hdg = np.arctan2(yc - ya, xc - xa)
    area2 = np.abs((x - xa) * (yc - ya) - (xc - xa) * (y - ya))
    d1 = np.hypot(x - xa, y - ya)
    d2 = np.hypot(xc - x, yc - y)
    d3 = np.hypot(xc - xa, yc - ya)
    denom = d1 * d2 * d3
    curv = np.where(denom > 1e-9, 2.0 * area2 / np.maximum(denom, 1e-12), 0.0)
    return hdg, curv


def left_normals(x, y):
    """Unit left normals from central-difference tangents (closed loop)."""
    tx = np.roll(x, -1) - np.roll(x, 1)
    ty = np.roll(y, -1) - np.roll(y, 1)
    norm = np.maximum(np.hypot(tx, ty), 1e-12)
    return -ty / norm, tx / norm


def _second_diff_matrix(ds):
    """Dense periodic second-difference operator scaled by arc length.

    (D p)_i ~ d^2 p / ds^2 at i, so |(D p)_i| approximates curvature.
    """
    n = len(ds)
    dsm = np.roll(ds, 1)                              # ds_{i-1}
    w = 2.0 / (dsm + ds)
    D = np.zeros((n, n))
    idx = np.arange(n)
    D[idx, (idx - 1) % n] = w / dsm
    D[idx, (idx + 1) % n] = w / ds
    D[idx, idx] = -(w / dsm + w / ds)
    return D


def _first_diff_matrix(ds):
    """Dense periodic first difference per arc length: (L d)_i = (d_{i+1}-d_i)/ds_i."""
    n = len(ds)
    L = np.zeros((n, n))
    idx = np.arange(n)
    L[idx, idx] = -1.0 / ds
    L[idx, (idx + 1) % n] = 1.0 / ds
    return L


def _solve_box_qp(H, g, lo, hi):
    """min 0.5 d^T H d + g^T d  s.t.  lo <= d <= hi   (H symmetric PD)."""
    n = len(g)
    if _HAVE_OSQP:
        prob = osqp.OSQP()
        prob.setup(P=_sp.csc_matrix(np.triu(H)), q=g,
                   A=_sp.identity(n, format='csc'), l=lo, u=hi,
                   eps_abs=1e-8, eps_rel=1e-8, max_iter=20000,
                   polish=True, verbose=False)
        d = prob.solve().x
        if d is None or not np.all(np.isfinite(d)):   # pragma: no cover
            d = np.zeros(n)
    else:                                             # pragma: no cover
        from scipy.optimize import lsq_linear
        # H = M^T M up to scaling is not available here; fall back to the
        # generic bounded least-squares on the Cholesky factor of H.
        R = np.linalg.cholesky(H + 1e-10 * np.eye(n)).T
        rhs = -np.linalg.solve(R.T, g)
        d = lsq_linear(R, rhs, bounds=(lo, hi)).x
    return np.clip(d, lo, hi)


def refine_raceline(x, y, corridor=0.25, iterations=3, smooth=0.1, ridge=1e-3):
    """Minimum-curvature refinement of a closed raceline.

    Parameters
    ----------
    x, y       : closed-loop waypoints (no repeated endpoint).
    corridor   : max lateral excursion from the ORIGINAL points (m).  Keep
                 conservative — wall clearance of the input line is unknown.
    iterations : linearize-solve-shift passes (normals recomputed each pass).
    smooth     : weight on ||(d_{i+1}-d_i)/ds||^2 — smoothness of the offset.
    ridge      : Tikhonov weight on ||d||^2 — keeps the QP well-posed.

    Returns (x_new, y_new, heading, curvature); heading/curvature use the
    same periodic finite differences as the raceline CSVs.
    """
    x0 = np.asarray(x, float)
    y0 = np.asarray(y, float)
    xc, yc = x0.copy(), y0.copy()
    for _ in range(int(iterations)):
        ds = np.maximum(segment_lengths(xc, yc), 1e-9)
        nx, ny = left_normals(xc, yc)
        D = _second_diff_matrix(ds)
        L = _first_diff_matrix(ds)
        Ax = D * nx[None, :]                          # D @ diag(nx)
        Ay = D * ny[None, :]
        H = 2.0 * (Ax.T @ Ax + Ay.T @ Ay + smooth * (L.T @ L)
                   + ridge * np.eye(len(xc)))
        g = 2.0 * (Ax.T @ (D @ xc) + Ay.T @ (D @ yc))
        # remaining corridor budget relative to the original line (triangle
        # inequality => total displacement stays <= corridor)
        budget = np.maximum(corridor - np.hypot(xc - x0, yc - y0), 0.0)
        d = _solve_box_qp(H, g, -budget, budget)
        xc = xc + d * nx
        yc = yc + d * ny
        if np.abs(d).max() < 1e-4:                    # converged early
            break
    hdg, curv = heading_curvature(xc, yc)
    return xc, yc, hdg, curv
