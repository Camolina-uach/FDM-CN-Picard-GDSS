"""Quantify whether the manufactured solution genuinely tests theta coupling.

The script reports the norms of both exact long-wave potentials, the relative
weight of the mixed-derivative terms, and a sensitivity test in which theta is
set to zero only in the recovery matrix while the exact fields and sources are
kept unchanged.
"""
from __future__ import annotations

import csv
import math
import os

import numpy as np
import sympy as sp

from gdss_fdm_solver import FDGrid, GDSSParams, GDSSSolver
from gdss_manufactured import MMSParams, build_mms


OUT = "results_fdm"
TAB = os.path.join(OUT, "tables")
os.makedirs(TAB, exist_ok=True)

EEE = GDSSParams(alpha=1, beta=1, gamma=1, xi=1,
                 psi=1, eta=2, phi=2, chi=1, theta=1)


def _texnum(value):
    if value == 0:
        return r"\(0\)"
    exponent = math.floor(math.log10(abs(value)))
    mantissa = value / 10**exponent
    return rf"\({mantissa:.3f}\times10^{{{exponent}}}\)"


def run(L=12.0, diagnostic_N=384, sizes=(24, 48, 96, 192)):
    mp = MMSParams()
    x, y = sp.symbols("x y", real=True)
    potential = mp.B * sp.exp(-(x**2 + y**2) / mp.s**2)
    w = sp.diff(potential, x)
    v = sp.diff(potential, y)

    expressions = (
        w,
        v,
        EEE.theta * sp.diff(v, x, y),
        EEE.psi * sp.diff(w, x, 2) + EEE.eta * sp.diff(w, y, 2),
        EEE.theta * sp.diff(w, x, y),
        EEE.phi * sp.diff(v, x, 2) + EEE.chi * sp.diff(v, y, 2),
    )
    functions = [sp.lambdify((x, y), expr, "numpy") for expr in expressions]
    grid = FDGrid(-L, L, -L, L, diagnostic_N, diagnostic_N)
    values = [np.asarray(fun(grid.X, grid.Y), dtype=float) for fun in functions]

    def norm(array):
        return math.sqrt(grid.hx * grid.hy * np.sum(array**2))

    metrics = {
        "theta": EEE.theta,
        "v_over_w": norm(values[1]) / norm(values[0]),
        "cross_w_equation": norm(values[2]) / norm(values[3]),
        "cross_v_equation": norm(values[4]) / norm(values[5]),
    }

    mms = build_mms(EEE, mp)
    u_exact, _, Sw_grid, Sv_grid, wvq_exact = mms
    uncoupled = GDSSParams(alpha=1, beta=1, gamma=1, xi=1,
                           psi=1, eta=2, phi=2, chi=1, theta=0)
    rows = []
    for N in sizes:
        g = FDGrid(-L, L, -L, L, N, N)
        solver = GDSSSolver(g, uncoupled)
        rho = np.abs(u_exact(g, 0.0))**2
        W, V, Q = solver.ell.solve_wv(rho, Sw_grid(g), Sv_grid(g))
        w_exact, v_exact, q_exact = wvq_exact(g)
        rel = lambda numerical, exact: solver.inner.l2(numerical - exact) / solver.inner.l2(exact)
        rows.append({
            "N": N,
            "h": 2 * L / (N + 1),
            "relW_theta0": rel(W, w_exact),
            "relV_theta0": rel(V, v_exact),
            "relQ_theta0": rel(Q, q_exact),
        })

    with open(os.path.join(OUT, "coupling_validation.csv"), "w", newline="") as stream:
        fieldnames = list(metrics) + ["N", "h", "relW_theta0", "relV_theta0", "relQ_theta0"]
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for index, row in enumerate(rows):
            writer.writerow({**({key: value if index == 0 else "" for key, value in metrics.items()}), **row})

    with open(os.path.join(TAB, "coupling_validation.tex"), "w") as stream:
        stream.write("\\begin{tabular}{lllll}\n\\toprule\n")
        stream.write("$N$ & $h$ & $\\|W-w^\\star\\|_2/\\|w^\\star\\|_2$ & "
                     "$\\|V-v^\\star\\|_2/\\|v^\\star\\|_2$ & "
                     "$\\|Q-Q^\\star\\|_2/\\|Q^\\star\\|_2$ \\\\\n\\midrule\n")
        for row in rows:
            stream.write(
                f"\\({row['N']}\\) & {row['h']:.3e} & "
                f"{_texnum(row['relW_theta0'])} & {_texnum(row['relV_theta0'])} & "
                f"{_texnum(row['relQ_theta0'])} \\\\\n"
            )
        stream.write("\\bottomrule\n\\end{tabular}\n")

    print("coupling metrics:", metrics)
    for row in rows:
        print(row)
    return metrics, rows


if __name__ == "__main__":
    run()
