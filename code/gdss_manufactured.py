"""
Method-of-manufactured-solutions (MMS) source terms for the GDSS.

We prescribe a smooth, doubly-localized triple (u*, w*, v*) that vanishes on the
domain boundary to round-off, and derive -- symbolically, via sympy -- the
analytic sources (S_u, S_w, S_v) that render (u*, w*, v*) an exact solution of
the forced GDSS

    i u_t + a u_xx + b u_yy - g|u|^2 u - xi u (w_x+v_y) = S_u
    psi w_xx + eta w_yy + th v_xy - (|u|^2)_x           = S_w
    phi v_xx + chi v_yy + th w_xy - (|u|^2)_y           = S_v

with u* = A (1 + eps sin(nu t)) exp(-(x^2+y^2)/(2 s^2)) exp(i(kx x + ky y - om t)),
     w* = d_x Psi, v* = d_y Psi,  Psi = B (1 + eps cos(nu t)) exp(-(x^2+y^2)/s^2).

With eps = 0 (default) the intensity |u*|^2 and the long-wave fields are
time-independent; eps > 0 makes rho*, w*, v*, Q* and the long-wave sources
time-dependent, so the time-centred coupling of the scheme is exercised.

The discrete scheme (augmented with the time-centered sources) then has u* as
its exact solution, so ||U_h - u*|| is the true discretization error.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import sympy as smp


@dataclass(frozen=True)
class MMSParams:
    A: float = 0.4
    s: float = 1.5
    kx: float = 0.6
    ky: float = -0.4
    om: float = 0.5
    B: float = 0.3
    eps: float = 0.0
    nu: float = 2.0


def build_mms(p, mp: MMSParams):
    """Return callables u_exact(g,t), Su_of_t(g,t), Sw_grid(g,t), Sv_grid(g,t)
    and wvq_exact(g,t) evaluated on grids (t defaults to 0 where optional)."""
    x, y, t = smp.symbols("x y t", real=True)
    a, b, g, xi = p.alpha, p.beta, p.gamma, p.xi
    psi, eta, phi, chi, th = p.psi, p.eta, p.phi, p.chi, p.theta
    A, s, kx, ky, om, B = mp.A, mp.s, mp.kx, mp.ky, mp.om, mp.B
    eps, nu = mp.eps, mp.nu

    env = A * (1 + eps * smp.sin(nu * t)) * smp.exp(-(x**2 + y**2) / (2 * s**2))
    phase = kx * x + ky * y - om * t
    u = env * smp.exp(smp.I * phase)
    Psi = B * (1 + eps * smp.cos(nu * t)) * smp.exp(-(x**2 + y**2) / s**2)
    w = smp.diff(Psi, x)
    v = smp.diff(Psi, y)

    rho = smp.simplify(u * smp.conjugate(u))          # |u|^2 (real)
    Q = smp.diff(w, x) + smp.diff(v, y)

    Su = (smp.I * smp.diff(u, t) + a * smp.diff(u, x, 2) + b * smp.diff(u, y, 2)
          - g * rho * u - xi * u * Q)
    Sw = (psi * smp.diff(w, x, 2) + eta * smp.diff(w, y, 2) + th * smp.diff(v, x, y)
          - smp.diff(rho, x))
    Sv = (phi * smp.diff(v, x, 2) + chi * smp.diff(v, y, 2) + th * smp.diff(w, x, y)
          - smp.diff(rho, y))

    f_u = smp.lambdify((x, y, t), u, "numpy")
    f_Su = smp.lambdify((x, y, t), Su, "numpy")
    f_Sw = smp.lambdify((x, y, t), Sw, "numpy")
    f_Sv = smp.lambdify((x, y, t), Sv, "numpy")
    f_w = smp.lambdify((x, y, t), w, "numpy")
    f_v = smp.lambdify((x, y, t), v, "numpy")
    f_Q = smp.lambdify((x, y, t), Q, "numpy")

    def wvq_exact(g_, t_=0.0):
        return (g_.flat(np.asarray(f_w(g_.X, g_.Y, t_), dtype=float) + 0 * g_.X),
                g_.flat(np.asarray(f_v(g_.X, g_.Y, t_), dtype=float) + 0 * g_.X),
                g_.flat(np.asarray(f_Q(g_.X, g_.Y, t_), dtype=float) + 0 * g_.X))

    def u_exact(g_, t_):
        return g_.flat(np.asarray(f_u(g_.X, g_.Y, t_), dtype=complex))

    def Su_of_t(g_, t_):
        return g_.flat(np.asarray(f_Su(g_.X, g_.Y, t_), dtype=complex)
                       + 0 * g_.X)                      # broadcast guard

    def Sw_grid(g_, t_=0.0):
        return g_.flat(np.asarray(f_Sw(g_.X, g_.Y, t_), dtype=float) + 0 * g_.X)

    def Sv_grid(g_, t_=0.0):
        return g_.flat(np.asarray(f_Sv(g_.X, g_.Y, t_), dtype=float) + 0 * g_.X)

    return u_exact, Su_of_t, Sw_grid, Sv_grid, wvq_exact
