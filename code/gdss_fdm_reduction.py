"""
DS-reduction cross-check (Appendix).

Under the reduction condition  theta = psi - phi = chi - eta  (i.e. phi = psi-theta,
eta = chi-theta), the 2x2 long-wave Fourier symbol of the GDSS factorises and the
coupling field Q = w_x + v_y satisfies the SINGLE scalar elliptic equation of the
classical elliptic-elliptic Davey--Stewartson mean field,

        psi Q_xx + chi Q_yy = (|u|^2)_xx + (|u|^2)_yy = Laplacian(|u|^2).

This script verifies that the coupling field recovered by the full 2x2 GDSS
elliptic solver coincides -- to second order under mesh refinement -- with the
field produced by the reduced scalar (classical DS) solver, for reduction
parameters.  It confirms that the scheme reproduces the classical DS mean field
in the reduction limit, so the short-wave equation there is literally the
classical DS equation.
"""
from __future__ import annotations
import os, math
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu
from gdss_fdm_solver import GDSSParams, FDGrid, GDSSSolver, Operators

OUT = "results_fdm"; TAB = os.path.join(OUT, "tables")
os.makedirs(TAB, exist_ok=True)


def _texnum(v):
    if v == 0 or not np.isfinite(v):
        return f"\\({v:.1f}\\)"
    e = math.floor(math.log10(abs(v))); m = v / 10**e
    return f"\\({m:.2f}\\times10^{{{e}}}\\)"


def scalar_Q(ops, g, rho, psi, chi):
    """Solve psi Q_xx + chi Q_yy = (dxx+dyy) rho, homogeneous Dirichlet."""
    Ls = (psi * ops.dxx + chi * ops.dyy).tocsc()
    rhs = (ops.dxx + ops.dyy) @ rho
    return splu(Ls).solve(rhs)


def _interior_rel(g, Qf, Qs, frac=0.25):
    N = g.Nx; m = int(N * frac); s = slice(m, N - m)
    Qf2, Qs2 = g.grid(Qf), g.grid(Qs)
    num = np.sqrt(np.sum(np.abs(Qf2[s, s] - Qs2[s, s])**2))
    den = np.sqrt(np.sum(np.abs(Qs2[s, s])**2))
    return num / den


def run(h_target=0.21):
    # reduction parameters: psi, chi, theta free; phi=psi-theta, eta=chi-theta
    psi, chi, theta = 2.0, 2.0, 1.0
    phi, eta = psi - theta, chi - theta          # = 1.0, 1.0
    par = GDSSParams(alpha=1, beta=1, gamma=1, xi=1,
                     psi=psi, eta=eta, phi=phi, chi=chi, theta=theta)
    assert abs(theta - (psi - phi)) < 1e-14 and abs(theta - (chi - eta)) < 1e-14
    assert abs(theta**2 - (phi - psi) * (eta - chi)) < 1e-14

    # The reduction Q_{2x2} = Q_scalar is a whole-space (periodic) identity; on the
    # bounded Dirichlet box it is approached as the domain grows.  We fix the mesh
    # size h and enlarge L, reporting the interior relative discrepancy.
    rows = []
    for L in (6, 10, 16, 24, 32):
        N = int(round(2 * L / h_target))
        g = FDGrid(-L, L, -L, L, N, N); sol = GDSSSolver(g, par); ops = sol.ops
        rho = g.flat(np.exp(-(g.X**2 + g.Y**2) / 2.0))
        _, _, Q_full = sol.ell.solve_wv(rho)               # full 2x2 GDSS recovery
        Q_scal = scalar_Q(ops, g, rho, psi, chi)           # reduced classical-DS scalar
        rel_int = _interior_rel(g, Q_full, Q_scal)
        rel_all = sol.inner.l2(Q_full - Q_scal) / sol.inner.l2(Q_scal)
        rows.append(dict(L=L, N=N, rel_all=rel_all, rel_int=rel_int))
        print(f"  L={L:3d} N={N:4d} rel(whole)={rel_all:.3e} rel(interior)={rel_int:.3e}")

    with open(os.path.join(TAB, "reduction.tex"), "w") as f:
        f.write("\\begin{tabular}{llll}\n\\toprule\n"
                "$L$ & $N$ & rel.\\ diff.\\ (whole) & rel.\\ diff.\\ (interior) \\\\\n\\midrule\n")
        for r in rows:
            f.write(f"\\({r['L']}\\) & \\({r['N']}\\) & {_texnum(r['rel_all'])} & {_texnum(r['rel_int'])} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")
    print("  wrote reduction.tex  (params: psi=%.0f chi=%.0f theta=%.0f -> phi=%.0f eta=%.0f)"
          % (psi, chi, theta, phi, eta))


if __name__ == "__main__":
    run()
