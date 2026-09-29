"""Momentum drift under spatial, temporal and domain-size refinement.

Tests the remark after the discrete momentum-balance theorem on the
source-free localized run (Experiments 2-3, T = 1).  The accumulated change of
J_x is split, through the exact balance identity, into the accumulated
dispersive (boundary-supported), cubic and long-wave residuals:

    Delta J_x = dt * sum_n (R_disp + R_cub + R_lw)      (up to round-off).

Blocks:
  space   L = 10, N = 64, 128, 256, fixed dt
  time    L = 10, N = 128, three time steps
  domain  L = 14, N = 90, 180 (same mesh sizes as N = 64, 128 at L = 10)

Writes results_fdm/momentum_refinement.csv and
results_fdm/tables/momentum_refinement.tex.
"""
from __future__ import annotations

import csv
import math
import os
import time

import numpy as np

from gdss_fdm_solver import FDGrid, GDSSParams, GDSSSolver


OUT = "results_fdm"
TAB = os.path.join(OUT, "tables")
os.makedirs(TAB, exist_ok=True)

EEE = GDSSParams(alpha=1, beta=1, gamma=1, xi=1,
                 psi=1, eta=2, phi=2, chi=1, theta=1)
DT = 4.4248e-3


def _texnum(value):
    if value == 0:
        return r"\(0\)"
    exponent = math.floor(math.log10(abs(value)))
    mantissa = value / 10**exponent
    return rf"\({mantissa:.2f}\times10^{{{exponent}}}\)"


def _run(N, dt, L, T=1.0):
    g = FDGrid(-L, L, -L, L, N, N)
    sol = GDSSSolver(g, EEE)
    r = np.sqrt(g.X**2 + g.Y**2)
    U = g.flat((0.2 / np.cosh(r) * np.exp(1j * (g.X + g.Y) / math.sqrt(2))).astype(complex))
    Jx0 = sol.momentum(U)[0]
    sums = dict(sum_disp=0.0, sum_cub=0.0, sum_lw=0.0)
    for _ in range(round(T / dt)):
        Un = U
        U, _ = sol.step(U, dt, tol=1e-12)
        res = sol.momentum_residuals(Un, U, dt)["x"]
        sums["sum_disp"] += dt * res["R_disp"]
        sums["sum_cub"] += dt * res["R_cub"]
        sums["sum_lw"] += dt * res["R_lw"]
    dJ = sol.momentum(U)[0] - Jx0
    defect = dJ - sums["sum_disp"] - sums["sum_cub"] - sums["sum_lw"]
    return dict(L=L, N=N, h=2 * L / (N + 1), dt=dt, dJx=dJ, **sums,
                balance_defect=defect)


def run():
    t0 = time.time()
    cache = {}

    def get(block, N, dt, L):
        key = (N, dt, L)
        if key not in cache:
            cache[key] = _run(N, dt, L)
            row = cache[key]
            print(f"  L={L:g} N={N:4d} dt={dt:.4e} dJx={row['dJx']:+.3e} "
                  f"disp={row['sum_disp']:+.3e} cub={row['sum_cub']:+.3e} "
                  f"lw={row['sum_lw']:+.3e}", flush=True)
        return dict(cache[key], block=block)

    rows = [get("space", N, DT, 10.0) for N in (64, 128, 256)]
    rows += [get("time", 128, dt, 10.0) for dt in (2 * DT, DT, DT / 2)]
    rows += [get("domain", N, DT, 14.0) for N in (90, 180)]

    fields = ["block", "L", "N", "h", "dt", "dJx", "sum_disp", "sum_cub",
              "sum_lw", "balance_defect"]
    with open(os.path.join(OUT, "momentum_refinement.csv"), "w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in fields})

    with open(os.path.join(TAB, "momentum_refinement.tex"), "w") as stream:
        stream.write("\\begin{tabular}{lllllll}\n\\toprule\n")
        stream.write("$L$ & $N$ & $\\Delta t$ & $\\Delta J_x$ & "
                     "$\\Delta t\\sum_n\\mathcal{R}^{\\mathrm{disp}}_x$ & "
                     "$\\Delta t\\sum_n\\mathcal{R}^{\\mathrm{cub}}_x$ & "
                     "$\\Delta t\\sum_n\\mathcal{R}^{\\mathrm{lw}}_x$ \\\\\n\\midrule\n")
        previous = rows[0]["block"]
        for row in rows:
            if row["block"] != previous:
                stream.write("\\midrule\n")
                previous = row["block"]
            stream.write(f"\\({row['L']:g}\\) & \\({row['N']}\\) & {row['dt']:.4e} & "
                         f"{_texnum(row['dJx'])} & {_texnum(row['sum_disp'])} & "
                         f"{_texnum(row['sum_cub'])} & {_texnum(row['sum_lw'])} \\\\\n")
        stream.write("\\bottomrule\n\\end{tabular}\n")
    print(f"  momentum refinement done in {time.time() - t0:.1f} s")
    return rows


if __name__ == "__main__":
    run()
