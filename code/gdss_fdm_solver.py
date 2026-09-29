"""
Conservative Crank--Nicolson--Picard finite-difference solver for the
two-dimensional generalized Davey--Stewartson system (GDSS), integer order,
homogeneous Dirichlet boundary conditions.

    i u_t + alpha u_xx + beta u_yy = gamma |u|^2 u + xi u (w_x + v_y)
    psi w_xx + eta w_yy + theta v_xy = d_x |u|^2
    phi v_xx + chi v_yy + theta w_xy = d_y |u|^2

Short-wave field advanced by a time-centered discrete-gradient Crank--Nicolson
step solved by Picard iteration; long-wave potentials (w,v) recovered at each
iteration from the discrete elliptic subsystem via one reusable sparse LU.

Structure-preserving properties proved for the scheme (see manuscript Sec. 5):
  * discrete mass and discrete Hamiltonian conserved exactly by the scheme
    (in floating point: to round-off and the Picard stopping tolerance),
  * discrete momentum balance  d_t J = R_disp + R_cub + R_lw  (no exact
    conservation under Dirichlet BC).
"""
from __future__ import annotations
from dataclasses import dataclass, replace
import time
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import cg, splu


# --------------------------------------------------------------------------
#  Parameters and grid
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class GDSSParams:
    alpha: float = 1.0
    beta: float = 1.0
    gamma: float = 1.0
    xi: float = 1.0
    psi: float = 1.0
    eta: float = 2.0
    phi: float = 2.0
    chi: float = 1.0
    theta: float = 1.0  # theta^2 = (phi-psi)(eta-chi)


@dataclass
class FDGrid:
    xL: float
    xR: float
    yL: float
    yR: float
    Nx: int           # interior points in x
    Ny: int           # interior points in y

    def __post_init__(self):
        self.hx = (self.xR - self.xL) / (self.Nx + 1)
        self.hy = (self.yR - self.yL) / (self.Ny + 1)
        self.x = self.xL + self.hx * np.arange(1, self.Nx + 1)
        self.y = self.yL + self.hy * np.arange(1, self.Ny + 1)
        # meshgrid with y slow (row), x fast (col): shape (Ny, Nx)
        self.X, self.Y = np.meshgrid(self.x, self.y)
        self.N = self.Nx * self.Ny

    def flat(self, F2d):
        return F2d.reshape(-1)          # C-order: idx = j*Nx + i

    def grid(self, Fflat):
        return Fflat.reshape(self.Ny, self.Nx)


def _d1(N, h):
    """Centered first difference, Dirichlet (skew-symmetric)."""
    e = np.ones(N)
    return sp.diags([e[:-1], -e[:-1]], [1, -1]) * (1.0 / (2.0 * h))


def _d2(N, h):
    """Centered second difference, Dirichlet (sym. neg. def.)."""
    e = np.ones(N)
    return sp.diags([-2 * e, e[:-1], e[:-1]], [0, 1, -1]) * (1.0 / (h * h))


# --------------------------------------------------------------------------
#  Discrete operators (Kronecker; y slow, x fast)
# --------------------------------------------------------------------------
class Operators:
    def __init__(self, g: FDGrid):
        Ix = sp.identity(g.Nx)
        Iy = sp.identity(g.Ny)
        D1x, D2x = _d1(g.Nx, g.hx), _d2(g.Nx, g.hx)
        D1y, D2y = _d1(g.Ny, g.hy), _d2(g.Ny, g.hy)
        self.dx = sp.kron(Iy, D1x, format="csr")
        self.dy = sp.kron(D1y, Ix, format="csr")
        self.dxx = sp.kron(Iy, D2x, format="csr")
        self.dyy = sp.kron(D2y, Ix, format="csr")
        self.dxy = sp.kron(D1y, D1x, format="csr")     # = dx @ dy (symmetric)
        self.I = sp.identity(g.N, format="csr")
        self.g = g

    def Lh(self, p: GDSSParams):
        return (p.alpha * self.dxx + p.beta * self.dyy).tocsr()


# --------------------------------------------------------------------------
#  Discrete inner products / norms  (with the h_x h_y weight)
# --------------------------------------------------------------------------
class Inner:
    def __init__(self, g: FDGrid):
        self.w = g.hx * g.hy

    def ip(self, F, G):                      # <F,G>_h = hx hy sum F conj(G)
        return self.w * np.vdot(G, F)        # vdot conjugates first arg -> sum F conj(G)

    def l2(self, F):
        return np.sqrt(np.real(self.w * np.vdot(F, F)))

    def l4_4(self, F):
        return self.w * np.sum(np.abs(F) ** 4)


# --------------------------------------------------------------------------
#  Elliptic long-wave recovery  A_h [W;V] = [dx rho; dy rho] (+ sources)
# --------------------------------------------------------------------------
class Elliptic:
    def __init__(self, ops: Operators, p: GDSSParams, method="lu",
                 cg_rtol=1e-10, cg_maxiter=None):
        A11 = p.psi * ops.dxx + p.eta * ops.dyy
        A12 = p.theta * ops.dxy
        A21 = p.theta * ops.dxy
        A22 = p.phi * ops.dxx + p.chi * ops.dyy
        A = sp.bmat([[A11, A12], [A21, A22]], format="csc")
        if method not in {"lu", "cg"}:
            raise ValueError("elliptic method must be 'lu' or 'cg'")
        self.method = method
        self.cg_rtol = cg_rtol
        self.cg_maxiter = cg_maxiter
        t0 = time.perf_counter()
        if method == "lu":
            self.lu = splu(A)
            self.A_spd = None
        else:
            # In the EEE regime A is negative definite, so CG is applied to
            # the equivalent SPD system (-A) z = -rhs.
            self.lu = None
            self.A_spd = (-A).tocsr()
        self.setup_time = time.perf_counter() - t0
        self.ops = ops
        self.N = ops.g.N
        self.reset_stats()

    def reset_stats(self):
        self.solve_time = 0.0
        self.calls = 0
        self.cg_iterations = []

    def solve_wv(self, rho, Sw=None, Sv=None):
        dx, dy = self.ops.dx, self.ops.dy
        rhs1 = dx @ rho
        rhs2 = dy @ rho
        if Sw is not None:
            rhs1 = rhs1 + Sw
        if Sv is not None:
            rhs2 = rhs2 + Sv
        rhs = np.concatenate([rhs1, rhs2])
        t0 = time.perf_counter()
        if self.method == "lu":
            z = self.lu.solve(rhs)
        else:
            nit = 0

            def count_iteration(_):
                nonlocal nit
                nit += 1

            z, info = cg(self.A_spd, -rhs, rtol=self.cg_rtol, atol=0.0,
                         maxiter=self.cg_maxiter, callback=count_iteration)
            if info != 0:
                raise RuntimeError(
                    f"CG failed for the coupled long-wave system: info={info}, "
                    f"iterations={nit}"
                )
            self.cg_iterations.append(nit)
        self.solve_time += time.perf_counter() - t0
        self.calls += 1
        W, V = z[:self.N], z[self.N:]
        Q = dx @ W + dy @ V
        return W, V, Q


# --------------------------------------------------------------------------
#  Solver
# --------------------------------------------------------------------------
class GDSSSolver:
    def __init__(self, g: FDGrid, p: GDSSParams, elliptic_method="lu",
                 cg_rtol=1e-10, cg_maxiter=None):
        self.g = g
        self.p = p
        self.ops = Operators(g)
        self.inner = Inner(g)
        self.ell = Elliptic(self.ops, p, method=elliptic_method,
                            cg_rtol=cg_rtol, cg_maxiter=cg_maxiter)
        self.Lh = self.ops.Lh(p)
        self._dt = None
        self._cn_lu = None

    # -- factor the constant CN left-hand side once per dt --
    def _prep_dt(self, dt):
        if self._dt == dt:
            return
        A = (self.ops.I - (0.5j * dt) * self.Lh).tocsc()
        self._cn_lu = splu(A)
        self._dt = dt

    # -- one conservative CN--Picard step --------------------------------
    def step(self, Un, dt, tol=1e-12, kmax=100, Sbar_u=None, Sw=None, Sv=None,
             Sw_old=None, Sv_old=None):
        # Sw, Sv: long-wave sources at t^{n+1}; Sw_old, Sv_old: at t^n
        # (default: time-independent sources).
        p = self.p
        self._prep_dt(dt)
        if Sw_old is None:
            Sw_old = Sw
        if Sv_old is None:
            Sv_old = Sv
        rhoN = np.abs(Un) ** 2
        _, _, QN = self.ell.solve_wv(rhoN, Sw_old, Sv_old)
        rhs_lin = Un + (0.5j * dt) * (self.Lh @ Un)
        Uk = Un.copy()
        iters = 0
        for k in range(kmax):
            iters = k + 1
            rhok = np.abs(Uk) ** 2
            _, _, Qk = self.ell.solve_wv(rhok, Sw, Sv)
            S = Uk + Un
            c = 0.25 * p.gamma * (rhok + rhoN) + 0.25 * p.xi * (Qk + QN)
            Nhat = c * S
            rhs = rhs_lin - 1j * dt * Nhat
            if Sbar_u is not None:
                rhs = rhs - 1j * dt * Sbar_u
            Uk1 = self._cn_lu.solve(rhs)
            if self.inner.l2(Uk1 - Uk) <= tol * self.inner.l2(Uk1):
                Uk = Uk1
                break
            Uk = Uk1
        return Uk, iters

    # -- nonlinear force and RHS (for explicit / non-conservative schemes)
    def N_force(self, U):
        p = self.p
        _, _, Q = self.ell.solve_wv(np.abs(U) ** 2)
        return p.gamma * (np.abs(U) ** 2) * U + p.xi * U * Q

    def rhs(self, U):
        return 1j * (self.Lh @ U - self.N_force(U))

    # -- NON-conservative CN (plain average of N) via Picard -------------
    #    baseline for the comparison experiment; does NOT preserve invariants
    def step_avg(self, Un, dt, tol=1e-12, kmax=100):
        self._prep_dt(dt)
        Nn = self.N_force(Un)
        rhs_lin = Un + (0.5j * dt) * (self.Lh @ Un)
        Uk = Un.copy()
        for k in range(kmax):
            Nk = self.N_force(Uk)
            rhs = rhs_lin - 0.5j * dt * (Nk + Nn)
            Uk1 = self._cn_lu.solve(rhs)
            if self.inner.l2(Uk1 - Uk) <= tol * self.inner.l2(Uk1):
                return Uk1, k + 1
            Uk = Uk1
        return Uk, kmax

    # -- classical explicit RK4 -----------------------------------------
    def step_rk4(self, Un, dt):
        k1 = self.rhs(Un)
        k2 = self.rhs(Un + 0.5 * dt * k1)
        k3 = self.rhs(Un + 0.5 * dt * k2)
        k4 = self.rhs(Un + dt * k3)
        return Un + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4), 4

    # -- diagnostics -----------------------------------------------------
    def mass(self, U):
        return np.real(self.inner.ip(U, U))

    def energy(self, U):
        p = self.p
        kin = np.real(self.inner.ip(-(self.Lh @ U), U))      # <-Lh U, U>
        quartic = 0.5 * p.gamma * self.inner.l4_4(U)
        _, _, Q = self.ell.solve_wv(np.abs(U) ** 2)
        coupl = 0.5 * p.xi * np.real(self.inner.ip(Q, np.abs(U) ** 2))
        return kin + quartic + coupl

    def momentum(self, U):
        ax = self.inner.ip(self.ops.dx @ U, U)
        ay = self.inner.ip(self.ops.dy @ U, U)
        Jx = np.real(1j * (ax - np.conj(ax)))               # = -2 Im(ax)
        Jy = np.real(1j * (ay - np.conj(ay)))
        return Jx, Jy

    # -- momentum-balance residual terms for a step (Un -> Un1) ----------
    def momentum_residuals(self, Un, Un1, dt):
        p = self.p
        ops, inner = self.ops, self.inner
        S = Un1 + Un
        rhoN = np.abs(Un) ** 2
        rho1 = np.abs(Un1) ** 2
        _, _, QN = self.ell.solve_wv(rhoN)
        _, _, Q1 = self.ell.solve_wv(rho1)
        c_cub = 0.25 * p.gamma * (rho1 + rhoN)
        c_lw = 0.25 * p.xi * (Q1 + QN)
        out = {}
        for lab, D in (("x", ops.dx), ("y", ops.dy)):
            dS = D @ S
            R_disp = np.real(inner.ip(dS, self.Lh @ S))
            R_cub = -2.0 * np.real(inner.ip(dS, c_cub * S))
            R_lw = -2.0 * np.real(inner.ip(dS, c_lw * S))
            Jn = self.momentum(Un)[0 if lab == "x" else 1]
            J1 = self.momentum(Un1)[0 if lab == "x" else 1]
            dJ = (J1 - Jn) / dt
            out[lab] = dict(dJ=dJ, R_disp=R_disp, R_cub=R_cub, R_lw=R_lw,
                            defect=dJ - (R_disp + R_cub + R_lw))
        return out
