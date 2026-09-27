"""Collocation DGSEM for the Laplace / heat equation on a curvilinear annulus mesh.

Element geometry
----------------
Kr x Kt quadrilaterals covering r1 <= r <= r2 (xi direction), 0 <= theta < 2 pi (eta,
periodic). Two mappings:

* ``curved``   isoparametric: nodes placed on the exact polar map, so the element
               boundaries approximate the circles with spectral accuracy
* ``straight`` bilinear map between the four corner points (polygonal walls)

Metric terms are computed by differentiating the interpolated mapping; in 2D this
satisfies the discrete metric identities exactly.

Discretisation of  lap(u)
-------------------------
Mixed form q = grad(u), lap(u) = div(q), both in strong DGSEM form on LGL nodes
(SBP property, surface terms lifted with 1 / w_end):

    J q      = Ja^1 u_xi + Ja^2 u_eta + sum_faces (u* - u) nS / w
    J lap(u) = d_xi(Ja^1 . q) + d_eta(Ja^2 . q) + sum_faces (q* . nS - q . nS) / w

with BR1 central fluxes plus an interior-penalty term (LDG/IP-type stabilisation):
    u* = {u},            q* = {q} - eta [[u]]      (interior faces)
    u* = g,              q* = q - eta (u - g) n    (Dirichlet walls)
    eta = (N+1)^2 / h_n,  h_n = 2 J / |nS|  (element height normal to the face)

The operator is linear and affine in the wall data g; it is applied to batches of
vectors and the global sparse matrix is assembled by probing.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu

from .basis import diff_matrix, lgl


@dataclass
class Couette:
    """Circular Couette flow between an inner and an outer rotating cylinder."""

    r1: float = 1.0
    r2: float = 2.0
    omega1: float = 1.0
    omega2: float = 0.0
    mu: float = 1.0

    @property
    def A(self):
        return (self.omega2 * self.r2**2 - self.omega1 * self.r1**2) / (self.r2**2 - self.r1**2)

    @property
    def B(self):
        return (self.omega1 - self.omega2) * self.r1**2 * self.r2**2 / (self.r2**2 - self.r1**2)

    def u_theta(self, r):
        return self.A * r + self.B / r

    def velocity(self, x, y):
        r = np.hypot(x, y)
        ut = self.u_theta(r)
        return -ut * y / r, ut * x / r

    def torque(self):
        """Magnitude of the viscous torque per unit length on either cylinder: 4 pi mu |B|."""
        return 4 * np.pi * self.mu * abs(self.B)


@dataclass
class Mesh:
    N: int
    Kr: int
    Kt: int
    r1: float = 1.0
    r2: float = 2.0
    curved: bool = True
    x: np.ndarray = field(init=False, repr=False)
    y: np.ndarray = field(init=False, repr=False)

    def __post_init__(self):
        n = self.N
        self.xi, self.w = lgl(n)
        self.D = diff_matrix(self.xi)
        K = self.Kr * self.Kt
        ir, it = np.divmod(np.arange(K), self.Kt)
        ra = self.r1 + (self.r2 - self.r1) * ir / self.Kr
        rb = self.r1 + (self.r2 - self.r1) * (ir + 1) / self.Kr
        ta = 2 * np.pi * it / self.Kt
        tb = 2 * np.pi * (it + 1) / self.Kt
        s = (self.xi[None, :, None] + 1) / 2  # along xi (radial)
        t = (self.xi[None, None, :] + 1) / 2  # along eta (azimuthal)
        R = lambda a: a[:, None, None]  # noqa: E731
        if self.curved:
            r = R(ra) + s * R(rb - ra)
            th = R(ta) + t * R(tb - ta)
            self.x, self.y = r * np.cos(th), r * np.sin(th)
        else:
            corners = [(R(ra), R(ta)), (R(rb), R(ta)), (R(ra), R(tb)), (R(rb), R(tb))]
            (x00, y00), (x10, y10), (x01, y01), (x11, y11) = [(rr * np.cos(tt), rr * np.sin(tt)) for rr, tt in corners]
            self.x = (1 - s) * (1 - t) * x00 + s * (1 - t) * x10 + (1 - s) * t * x01 + s * t * x11
            self.y = (1 - s) * (1 - t) * y00 + s * (1 - t) * y10 + (1 - s) * t * y01 + s * t * y11
        D = self.D
        x_xi = np.einsum("ik,ekj->eij", D, self.x)
        x_eta = np.einsum("jk,eik->eij", D, self.x)
        y_xi = np.einsum("ik,ekj->eij", D, self.y)
        y_eta = np.einsum("jk,eik->eij", D, self.y)
        self.J = x_xi * y_eta - x_eta * y_xi
        self.Ja1 = np.stack([y_eta, -x_eta])  # (2, K, n, n)
        self.Ja2 = np.stack([-y_xi, x_xi])
        self.K = K
        self.shape = (K, n + 1, n + 1)
        self.ndof = K * (n + 1) ** 2
        # neighbour tables (structured, periodic in theta)
        self.east = np.where(ir < self.Kr - 1, (ir + 1) * self.Kt + it, -1)  # across xi = +1
        self.north = ir * self.Kt + (it + 1) % self.Kt  # across eta = +1

    # -------------------------------------------------------------- faces
    def face_normals(self):
        """Outward scaled normals nS for faces xi-, xi+, eta-, eta+ ; each (2, K, n+1)."""
        return (
            -self.Ja1[:, :, 0, :],
            self.Ja1[:, :, -1, :],
            -self.Ja2[:, :, :, 0],
            self.Ja2[:, :, :, -1],
        )


def _faces(a):
    """Traces of a (..., K, n, n, B) on faces xi-, xi+, eta-, eta+ -> (..., K, n, B)."""
    return a[..., :, 0, :, :], a[..., :, -1, :, :], a[..., :, :, 0, :], a[..., :, :, -1, :]


def _add_face(a, face, val):
    """a[..., K, i, j, B] += val on one face (same face order as _faces)."""
    s = slice(None)
    idx = [(s, 0, s, s), (s, -1, s, s), (s, s, 0, s), (s, s, -1, s)][face]
    a[(Ellipsis,) + idx] += val


class Laplacian:
    """Affine DGSEM operator  u -> lap_h(u; g_inner, g_outer)."""

    def __init__(self, mesh: Mesh, penalty: float = 1.0):
        self.m = mesh
        n = mesh.N
        self.nS = mesh.face_normals()
        self.norm = [np.linalg.norm(v, axis=0) for v in self.nS]
        self.eta = [penalty * (n + 1) ** 2 / (2 * mesh.J[f_idx] / nrm)
                    for f_idx, nrm in zip(self._face_index(), self.norm)]
        self.w_end = (mesh.w[0], mesh.w[-1], mesh.w[0], mesh.w[-1])
        self.inner = np.arange(mesh.Kt)  # elements touching r = r1 (face xi-)
        self.outer = (mesh.Kr - 1) * mesh.Kt + np.arange(mesh.Kt)  # r = r2 (face xi+)
        self._matrix = None

    @staticmethod
    def _face_index():
        return [(slice(None), 0), (slice(None), -1), (slice(None), slice(None), 0), (slice(None), slice(None), -1)]

    def _neighbour_trace(self, tr, face):
        """Trace on the other side of each face (NaN-free; boundary rows filled with own trace)."""
        m = self.m
        if face == 0:  # xi-: neighbour is the element whose east is me
            out = tr[1].copy()  # placeholder shape
            west = np.full(m.K, -1)
            has = m.east >= 0
            west[m.east[has]] = np.nonzero(has)[0]
            ok = west >= 0
            out[ok] = tr[1][west[ok]]
            out[~ok] = tr[0][~ok]
            return out
        if face == 1:
            out = tr[1].copy()
            ok = m.east >= 0
            out[ok] = tr[0][m.east[ok]]
            return out
        if face == 2:
            south = np.empty(m.K, dtype=int)
            south[m.north] = np.arange(m.K)
            return tr[3][south]
        return tr[2][m.north]

    def gradient(self, u, g_in, g_out):
        """q = grad_h(u); u (K, n, n, B), wall data g_* (Kt, n, B). Returns q (2, K, n, n, B)."""
        m = self.m
        D = m.D
        u_xi = np.einsum("ik,ekjb->eijb", D, u)
        u_eta = np.einsum("jk,eikb->eijb", D, u)
        J = m.J[..., None]
        q = (m.Ja1[..., None] * u_xi + m.Ja2[..., None] * u_eta) / J
        tr = _faces(u)
        for f in range(4):
            other = self._neighbour_trace(tr, f)
            ustar = 0.5 * (tr[f] + other)
            if f == 0:
                ustar[self.inner] = g_in
            if f == 1:
                ustar[self.outer] = g_out
            fi = self._face_index()[f]
            Jf = m.J[fi][..., None]
            _add_face(q, f, (ustar - tr[f])[None] * self.nS[f][..., None] / (self.w_end[f] * Jf[None]))
        return q

    def apply(self, u, g_in, g_out):
        m = self.m
        D = m.D
        q = self.gradient(u, g_in, g_out)
        F1 = (m.Ja1[..., None] * q).sum(axis=0)
        F2 = (m.Ja2[..., None] * q).sum(axis=0)
        lap = np.einsum("ik,ekjb->eijb", D, F1) + np.einsum("jk,eikb->eijb", D, F2)
        tr_u = _faces(u)
        tr_q = _faces(q)
        for f in range(4):
            nS = self.nS[f][..., None]
            qn = (tr_q[f] * nS).sum(axis=0)
            qo = np.stack([self._neighbour_trace([t[c] for t in tr_q], f) for c in range(2)])
            uo = self._neighbour_trace(tr_u, f)
            eta = np.maximum(self.eta[f], self._neighbour_trace([e for e in self.eta], f))[..., None]
            qstar_n = 0.5 * (qn + (qo * nS).sum(axis=0)) - eta * (tr_u[f] - uo) * self.norm[f][..., None]
            if f == 0:
                b = self.inner
                qstar_n[b] = qn[b] - eta[b] * (tr_u[f][b] - g_in) * self.norm[f][b][..., None]
            if f == 1:
                b = self.outer
                qstar_n[b] = qn[b] - eta[b] * (tr_u[f][b] - g_out) * self.norm[f][b][..., None]
            _add_face(lap, f, (qstar_n - qn) / self.w_end[f])
        return lap / m.J[..., None]

    # -------------------------------------------------------------- global matrix
    def matrix(self, batch=400):
        if self._matrix is None:
            m = self.m
            n1 = m.N + 1
            z = np.zeros((m.Kt, n1, 1))
            rows, cols, vals = [], [], []
            for start in range(0, m.ndof, batch):
                idx = np.arange(start, min(start + batch, m.ndof))
                E = np.zeros((m.ndof, len(idx)))
                E[idx, np.arange(len(idx))] = 1.0
                zb = np.broadcast_to(z, (m.Kt, n1, len(idx)))
                col = self.apply(E.reshape(m.shape + (len(idx),)), zb, zb).reshape(m.ndof, len(idx))
                r, c = np.nonzero(np.abs(col) > 1e-13 * np.abs(col).max())
                rows.append(r)
                cols.append(idx[c])
                vals.append(col[r, c])
            self._matrix = sp.csc_matrix(
                (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(m.ndof, m.ndof)
            )
        return self._matrix

    def affine(self, g_in, g_out):
        u0 = np.zeros(self.m.shape + (g_in.shape[-1],))
        return self.apply(u0, g_in, g_out).reshape(self.m.ndof, -1)


def wall_data(mesh: Mesh, flow: Couette):
    """Wall velocity (Omega x x) at the boundary nodes of the (possibly polygonal) walls."""
    xi_in = mesh.x[: mesh.Kt, 0, :], mesh.y[: mesh.Kt, 0, :]
    xo = mesh.x[-mesh.Kt :, -1, :], mesh.y[-mesh.Kt :, -1, :]
    g_in = np.stack([-flow.omega1 * xi_in[1], flow.omega1 * xi_in[0]], axis=-1)  # (Kt, n, 2): (u_x, u_y)
    g_out = np.stack([-flow.omega2 * xo[1], flow.omega2 * xo[0]], axis=-1)
    return g_in, g_out


def solve_steady(mesh: Mesh, flow: Couette, penalty=1.0):
    """Steady Couette flow: lap(u_x) = lap(u_y) = 0 with rotating-wall data. Returns (u (K,n,n,2), operator)."""
    op = Laplacian(mesh, penalty)
    A = op.matrix()
    g_in, g_out = wall_data(mesh, flow)
    rhs = -op.affine(g_in, g_out)
    u = splu(A).solve(rhs).reshape(mesh.shape + (2,))
    return u, op, (g_in, g_out)


def l2_error(mesh: Mesh, flow: Couette, u):
    ux, uy = flow.velocity(mesh.x, mesh.y)
    W = mesh.w[:, None] * mesh.w[None, :]
    err2 = (W * mesh.J * ((u[..., 0] - ux) ** 2 + (u[..., 1] - uy) ** 2)).sum()
    ref2 = (W * mesh.J * (ux**2 + uy**2)).sum()
    return np.sqrt(err2 / ref2)


def inner_torque(mesh: Mesh, flow: Couette, u, op: Laplacian, g):
    """Viscous torque on the inner cylinder from the DG velocity gradient."""
    q = op.gradient(u, g[0], g[1])  # (2 [d/dx, d/dy], K, n, n, 2 [u_x, u_y])
    b = op.inner
    Gx = q[0][b, 0]  # d/dx of (u_x, u_y) on the inner wall: (Kt, n, 2)
    Gy = q[1][b, 0]
    xs, ys = mesh.x[b, 0], mesh.y[b, 0]
    nS = op.nS[0][:, b]  # outward from the fluid element = into the cylinder
    ds = np.linalg.norm(nS, axis=0)
    nx, ny = nS[0] / ds, nS[1] / ds
    txx = 2 * flow.mu * Gx[..., 0]
    tyy = 2 * flow.mu * Gy[..., 1]
    txy = flow.mu * (Gy[..., 0] + Gx[..., 1])
    # traction exerted by the fluid on the cylinder: -tau . n_out(fluid)
    tx = -(txx * nx + txy * ny)
    ty = -(txy * nx + tyy * ny)
    torque = ((xs * ty - ys * tx) * ds * mesh.w[None, :]).sum()
    return abs(torque)
