"""Same-grid comparison of reused sparse LU and repeated conjugate gradients.

Both variants use the same finite-difference operators and CN--Picard update.
Only the solution method for the coupled long-wave subsystem changes.
"""
from __future__ import annotations

import argparse
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


def _initial_data(grid):
    radius = np.sqrt(grid.X**2 + grid.Y**2)
    return grid.flat((0.2 / np.cosh(radius) *
                      np.exp(1j * (grid.X + grid.Y) / math.sqrt(2))).astype(complex))


def benchmark_case(N, steps=4, L=10.0, dt=4.4248e-3, cg_rtol=1e-10):
    grid = FDGrid(-L, L, -L, L, N, N)

    t0 = time.perf_counter()
    lu_solver = GDSSSolver(grid, EEE, elliptic_method="lu")
    lu_total_setup = time.perf_counter() - t0
    t0 = time.perf_counter()
    cg_solver = GDSSSolver(grid, EEE, elliptic_method="cg", cg_rtol=cg_rtol,
                           cg_maxiter=max(1000, 8 * N))
    cg_total_setup = time.perf_counter() - t0

    # The CN matrix is common to both variants and is prepared before timing.
    lu_solver._prep_dt(dt)
    cg_solver._prep_dt(dt)
    initial = _initial_data(grid)

    def run(solver):
        solver.ell.reset_stats()
        state = initial.copy()
        picard_iterations = []
        start = time.perf_counter()
        for _ in range(steps):
            state, iterations = solver.step(state, dt, tol=1e-12)
            picard_iterations.append(iterations)
        elapsed = time.perf_counter() - start
        return state, elapsed, picard_iterations

    U_lu, time_lu, picard_lu = run(lu_solver)
    U_cg, time_cg, picard_cg = run(cg_solver)

    lu_wave_time = lu_solver.ell.solve_time
    cg_wave_time = cg_solver.ell.solve_time
    cg_iterations = np.asarray(cg_solver.ell.cg_iterations, dtype=float)

    rel_u = lu_solver.inner.l2(U_lu - U_cg) / lu_solver.inner.l2(U_lu)
    _, _, Q_lu = lu_solver.ell.solve_wv(np.abs(U_lu)**2)
    _, _, Q_cg = cg_solver.ell.solve_wv(np.abs(U_cg)**2)
    rel_q = lu_solver.inner.l2(Q_lu - Q_cg) / lu_solver.inner.l2(Q_lu)
    return {
        "N": N,
        "unknowns": 2 * N * N,
        "steps": steps,
        "lu_factor_s": lu_solver.ell.setup_time,
        "lu_total_setup_s": lu_total_setup,
        "cg_total_setup_s": cg_total_setup,
        "lu_total_s": time_lu,
        "lu_step_s": time_lu / steps,
        "lu_wave_fraction": lu_wave_time / time_lu,
        "cg_total_s": time_cg,
        "cg_step_s": time_cg / steps,
        "cg_wave_fraction": cg_wave_time / time_cg,
        "cg_iterations_avg": float(np.mean(cg_iterations)),
        "cg_iterations_max": int(np.max(cg_iterations)),
        "picard_lu_avg": float(np.mean(picard_lu)),
        "picard_cg_avg": float(np.mean(picard_cg)),
        "rel_u": rel_u,
        "rel_q": rel_q,
    }


def run(sizes=(64, 128, 256, 512), steps=4, cg_rtol=1e-10):
    rows = []
    for N in sizes:
        print(f"benchmark N={N}", flush=True)
        row = benchmark_case(N, steps=steps, cg_rtol=cg_rtol)
        rows.append(row)
        print(row, flush=True)

    csv_path = os.path.join(OUT, "cost_comparison.csv")
    with open(csv_path, "w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    with open(os.path.join(TAB, "cost_comparison.tex"), "w") as stream:
        stream.write("\\begin{tabular}{rrrrrrr}\n\\toprule\n")
        stream.write("$N$ & LU setup & LU/step & LU wave & CG/step & CG/LU & CG wave \\\\\n")
        stream.write(" & (s) & (s) & (\\%) & (s) & ratio & (\\%) \\\\\n\\midrule\n")
        for row in rows:
            stream.write(
                f"{row['N']} & {row['lu_factor_s']:.3e} & {row['lu_step_s']:.3e} & "
                f"{100 * row['lu_wave_fraction']:.1f} & {row['cg_step_s']:.3e} & "
                f"{row['cg_step_s'] / row['lu_step_s']:.1f} & "
                f"{100 * row['cg_wave_fraction']:.1f} \\\\\n"
            )
        stream.write("\\bottomrule\n\\end{tabular}\n\\par\\medskip\n")
        stream.write("\\begin{tabular}{rrrrr}\n\\toprule\n")
        stream.write("$N$ & CG iter. (avg.) & CG iter. (max) & rel. $U$ diff. & rel. $Q$ diff. \\\\\n\\midrule\n")
        for row in rows:
            stream.write(
                f"{row['N']} & {row['cg_iterations_avg']:.1f} & {row['cg_iterations_max']} & "
                f"{row['rel_u']:.2e} & {row['rel_q']:.2e} \\\\\n"
            )
        stream.write("\\bottomrule\n\\end{tabular}\n")
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", nargs="+", type=int, default=[64, 128, 256, 512])
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--cg-rtol", type=float, default=1e-10)
    args = parser.parse_args()
    run(tuple(args.sizes), steps=args.steps, cg_rtol=args.cg_rtol)
