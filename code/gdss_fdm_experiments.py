"""
Experiments for the conservative CN--Picard GDSS finite-difference solver.

Produces:
  results_fdm/*.csv            raw numbers
  results_fdm/tables/*.tex     LaTeX tables (drop into the manuscript)
  Images/*.png                 figures referenced by the manuscript

Experiments:
  1a  linear Schrodinger standing-wave convergence (joint refinement)
  1b  manufactured-solution convergence (spatial + temporal self-convergence),
      with time-dependent envelope and long-wave fields
  2   source-free conservation (RE_M, RE_E, RE_Jx, RE_Jy vs t)
  3   momentum-balance decomposition (dt J = R_disp + R_cub + R_lw + defect)
  4   representative localized-wave run to T=2 (fields)
"""
from __future__ import annotations
import os, math, csv, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from gdss_fdm_solver import GDSSParams, FDGrid, GDSSSolver
from gdss_manufactured import MMSParams, build_mms

OUT = "results_fdm"; TAB = os.path.join(OUT, "tables"); IMG = "Images"
for d in (OUT, TAB, IMG):
    os.makedirs(d, exist_ok=True)

EEE = GDSSParams(alpha=1, beta=1, gamma=1, xi=1, psi=1, eta=2, phi=2, chi=1, theta=1)


def _order(prev, cur):
    return math.log(prev / cur) / math.log(2) if prev else None


def _fmt(v):
    return "n/a" if v is None else f"{v:.3f}"


def _sci(v):
    return f"\\({v:.4f}\\times10^{{{0}}}\\)" if v == 0 else \
        (lambda m, e: f"\\({m:.4f}\\times10^{{{e}}}\\)")(v / 10**math.floor(math.log10(abs(v))),
                                                          math.floor(math.log10(abs(v))))


# ---------------------------------------------------------------- Exp 1a
def exp1a_linear():
    def sw(g, t, p=1, q=1):
        Lx, Ly = g.xR - g.xL, g.yR - g.yL
        om = EEE.alpha * (p * math.pi / Lx) ** 2 + EEE.beta * (q * math.pi / Ly) ** 2
        u = np.sin(p * math.pi * (g.X - g.xL) / Lx) * np.sin(q * math.pi * (g.Y - g.yL) / Ly) * np.exp(-1j * om * t)
        return g.flat(u.astype(complex))
    par = GDSSParams(alpha=1, beta=1, gamma=0.0, xi=0.0)
    T = 0.1
    rows = []; prev = None
    for N in (16, 32, 64, 128):
        dt = 1.0 / N / 20.0             # refine dt with h so joint order shows
        g = FDGrid(0, 1, 0, 1, N, N); sol = GDSSSolver(g, par)
        U = sw(g, 0.0); nt = round(T / dt)
        for n in range(nt):
            U, _ = sol.step(U, dt, tol=1e-13)
        err = sol.inner.l2(U - sw(g, nt * dt))
        rows.append(dict(N=N, h=1 / (N + 1), dt=dt, err=err, order=_order(prev, err))); prev = err
    _write_csv(rows, "exp1a_linear.csv")
    with open(os.path.join(TAB, "exp1a_linear.tex"), "w") as f:
        f.write("\\begin{tabular}{lllll}\n\\toprule\n$M=N$ & $h$ & $\\Delta t$ & $\\|U_h-u^\\star\\|_2$ & order \\\\\n\\midrule\n")
        for r in rows:
            f.write(f"${r['N']}$ & {r['h']:.4e} & {r['dt']:.4e} & {r['err']:.4e} & {_fmt(r['order'])} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")
    return rows


# ---------------------------------------------------------------- Exp 1b
def _mms_run(N, dt, T, L, par, mms):
    """Forced run: time-centred S_u, and long-wave sources evaluated at t^n
    (for Q^n) and t^{n+1} (for the Picard iterate), as in the scheme."""
    u_exact, Su_of_t, Sw_grid, Sv_grid = mms[0], mms[1], mms[2], mms[3]
    g = FDGrid(-L, L, -L, L, N, N); sol = GDSSSolver(g, par)
    U = u_exact(g, 0.0); nt = round(T / dt)
    Su_old, Sw_old, Sv_old = Su_of_t(g, 0.0), Sw_grid(g, 0.0), Sv_grid(g, 0.0)
    for n in range(nt):
        t1 = (n + 1) * dt
        Su_new, Sw_new, Sv_new = Su_of_t(g, t1), Sw_grid(g, t1), Sv_grid(g, t1)
        Sbar = 0.5 * (Su_old + Su_new)
        U, _ = sol.step(U, dt, tol=1e-12, Sbar_u=Sbar, Sw=Sw_new, Sv=Sv_new,
                        Sw_old=Sw_old, Sv_old=Sv_old)
        Su_old, Sw_old, Sv_old = Su_new, Sw_new, Sv_new
    return g, sol, U


MMS_EPS, MMS_NU = 0.25, 2.0      # time-dependent envelope and long-wave fields
MMS_T_SPACE, MMS_DT_SPACE = 0.5, 1e-3


def exp1b_mms(L=12.0):
    par = EEE
    mms = build_mms(par, MMSParams(eps=MMS_EPS, nu=MMS_NU))
    u_exact = mms[0]
    # spatial (dt small, refine grid)
    srows = []; prev = None
    for N in (24, 48, 96, 192):
        g, sol, U = _mms_run(N, MMS_DT_SPACE, MMS_T_SPACE, L, par, mms)
        err = sol.inner.l2(U - u_exact(g, MMS_T_SPACE))
        srows.append(dict(N=N, h=2 * L / (N + 1), err=err, order=_order(prev, err))); prev = err
        print(f"  MMS space N={N}: err={err:.4e} order={srows[-1]['order']}", flush=True)
    _write_csv(srows, "exp1b_spatial.csv")
    # temporal self-convergence (fixed grid, fast phase; reference dt = T/512)
    mms_t = build_mms(par, MMSParams(om=4.0, eps=MMS_EPS, nu=MMS_NU))
    N = 96; T = 0.4
    gref, solref, Uref = _mms_run(N, T / 512, T, L, par, mms_t)
    trows = []; prev = None
    for dt in (0.05, 0.025, 0.0125, 0.00625):
        g, sol, U = _mms_run(N, dt, T, L, par, mms_t)
        err = sol.inner.l2(U - Uref)
        trows.append(dict(dt=dt, err=err, order=_order(prev, err))); prev = err
        print(f"  MMS time dt={dt}: self-err={err:.4e} order={trows[-1]['order']}", flush=True)
    _write_csv(trows, "exp1b_temporal.csv")
    # combined table
    with open(os.path.join(TAB, "exp1b_mms.tex"), "w") as f:
        f.write("\\begin{tabular}{lll@{\\hskip 2em}lll}\n\\toprule\n")
        f.write("$M=N$ & $h$ & $\\|U_h-u^\\star\\|_2$ (order) & $\\Delta t$ & self-error & order \\\\\n\\midrule\n")
        for i in range(4):
            s = srows[i]; t = trows[i]
            f.write(f"${s['N']}$ & {s['h']:.3e} & {s['err']:.3e} ({_fmt(s['order'])}) & "
                    f"{t['dt']:.3e} & {t['err']:.3e} & {_fmt(t['order'])} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")
    # figure
    plt.figure(figsize=(10, 4))
    ax = plt.subplot(1, 2, 1)
    hs = [r["h"] for r in srows]; es = [r["err"] for r in srows]
    ax.loglog(hs, es, "o-", label="$L_2$ error")
    ax.loglog(hs, [es[0] * (h / hs[0]) ** 2 for h in hs], "k--", label="slope 2")
    ax.set_xlabel("$h$"); ax.set_ylabel("error"); ax.set_title("MMS spatial"); ax.legend(); ax.grid(True, which="both", alpha=.3)
    ax = plt.subplot(1, 2, 2)
    ds = [r["dt"] for r in trows]; et = [r["err"] for r in trows]
    ax.loglog(ds, et, "s-", label="self-error")
    ax.loglog(ds, [et[0] * (d / ds[0]) ** 2 for d in ds], "k--", label="slope 2")
    ax.set_xlabel("$\\Delta t$"); ax.set_title("MMS temporal"); ax.legend(); ax.grid(True, which="both", alpha=.3)
    plt.tight_layout(); plt.savefig(os.path.join(IMG, "convergence_orders.png"), dpi=140); plt.close()
    return srows, trows


# --------------------------------------------------- Exp 2,3,4 (source-free)
def _localized_u0(g, A0=0.2, w0=1.0, kx=1/math.sqrt(2), ky=1/math.sqrt(2)):
    r = np.sqrt(g.X**2 + g.Y**2)
    return g.flat((A0 / np.cosh(r / w0) * np.exp(1j * (kx * g.X + ky * g.Y))).astype(complex))


def exp234(N=128, L=10.0, T=2.0, dt=4.4248e-3):
    par = EEE
    g = FDGrid(-L, L, -L, L, N, N); sol = GDSSSolver(g, par)
    U = _localized_u0(g)
    M0, E0 = sol.mass(U), sol.energy(U); Jx0, Jy0 = sol.momentum(U)
    nt = round(T / dt)
    sample_t = [0.2, 0.4, 0.6, 0.8, 1.0]
    cons_rows = []; mom_rows = []
    hist = dict(t=[], REM=[], REE=[], REJx=[], REJy=[])
    itsum = 0; maxdef = 0.0
    for n in range(nt):
        Un = U.copy()
        U, it = sol.step(U, dt, tol=1e-12); itsum += it
        t = (n + 1) * dt
        M, E = sol.mass(U), sol.energy(U); Jx, Jy = sol.momentum(U)
        REM = abs((M - M0) / M0); REE = abs((E - E0) / E0)
        REJx = abs((Jx - Jx0) / Jx0); REJy = abs((Jy - Jy0) / Jy0)
        hist["t"].append(t); hist["REM"].append(REM); hist["REE"].append(REE)
        hist["REJx"].append(REJx); hist["REJy"].append(REJy)
        for st in sample_t:                      # residuals only at sample times (cost)
            if abs(t - st) < dt / 2:
                res = sol.momentum_residuals(Un, U, dt)
                maxdef = max(maxdef, abs(res["x"]["defect"]), abs(res["y"]["defect"]))
                cons_rows.append(dict(t=st, REM=REM, REE=REE, REJx=REJx, REJy=REJy))
                mom_rows.append(dict(t=st, **res["x"]))
    # store final fields
    absu = np.abs(g.grid(U)); Wf, Vf, Qf = sol.ell.solve_wv(np.abs(U)**2)
    np.savez_compressed(os.path.join(OUT, "exp4_final_fields.npz"),
                        x=g.x, y=g.y, absu=absu, W=g.grid(Wf), V=g.grid(Vf), Q=g.grid(Qf))
    # csv
    _write_csv(cons_rows, "exp2_conservation.csv")
    _write_csv(mom_rows, "exp3_momentum.csv")
    # tables
    with open(os.path.join(TAB, "exp2_conservation.tex"), "w") as f:
        f.write("\\begin{tabular}{lllll}\n\\toprule\n$t$ & $\\mathrm{RE}_M$ & $\\mathrm{RE}_E$ & $\\mathrm{RE}_{J_x}$ & $\\mathrm{RE}_{J_y}$ \\\\\n\\midrule\n")
        for r in cons_rows:
            f.write(f"\\({r['t']:.1f}\\) & {_texnum(r['REM'])} & {_texnum(r['REE'])} & {_texnum(r['REJx'])} & {_texnum(r['REJy'])} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")
    with open(os.path.join(TAB, "exp3_momentum.tex"), "w") as f:
        f.write("\\begin{tabular}{llllll}\n\\toprule\n$t$ & $\\delta_t J_{x}$ & $\\mathcal{R}^{\\mathrm{disp}}_x$ & $\\mathcal{R}^{\\mathrm{cub}}_x$ & $\\mathcal{R}^{\\mathrm{lw}}_x$ & defect \\\\\n\\midrule\n")
        for r in mom_rows:
            f.write(f"\\({r['t']:.1f}\\) & {_texnum(r['dJ'])} & {_texnum(r['R_disp'])} & {_texnum(r['R_cub'])} & {_texnum(r['R_lw'])} & {_texnum(r['defect'])} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")
    # invariant-history figure
    plt.figure(figsize=(9, 4.2))
    ax = plt.subplot(1, 2, 1)
    ax.semilogy(hist["t"], np.maximum(hist["REM"], 1e-17), label="$\\mathrm{RE}_M$")
    ax.semilogy(hist["t"], np.maximum(hist["REE"], 1e-17), label="$\\mathrm{RE}_E$")
    ax.axhline(np.finfo(float).eps, color="grey", ls=":", label="machine $\\epsilon$")
    ax.set_xlabel("$t$"); ax.set_ylabel("relative drift"); ax.set_title("Mass / energy"); ax.legend(); ax.grid(True, alpha=.3)
    ax = plt.subplot(1, 2, 2)
    ax.semilogy(hist["t"], np.maximum(hist["REJx"], 1e-17), label="$\\mathrm{RE}_{J_x}$")
    ax.semilogy(hist["t"], np.maximum(hist["REJy"], 1e-17), label="$\\mathrm{RE}_{J_y}$")
    ax.set_xlabel("$t$"); ax.set_title("Momenta"); ax.legend(); ax.grid(True, alpha=.3)
    plt.tight_layout(); plt.savefig(os.path.join(IMG, "invariant_histories.png"), dpi=140); plt.close()
    # fields figure
    d = np.load(os.path.join(OUT, "exp4_final_fields.npz"))
    ext = [g.xL, g.xR, g.yL, g.yR]
    plt.figure(figsize=(10, 8))
    for k, (name, fld) in enumerate([("$|u|$", d["absu"]), ("$w$", d["W"]), ("$v$", d["V"]), ("$Q$", d["Q"])]):
        ax = plt.subplot(2, 2, k + 1)
        im = ax.imshow(fld, extent=ext, origin="lower", aspect="auto", cmap="viridis")
        ax.set_title(f"{name} at $T={T}$"); plt.colorbar(im, ax=ax)
    plt.tight_layout(); plt.savefig(os.path.join(IMG, "localized_run_fields.png"), dpi=140); plt.close()
    summary = dict(N=N, L=L, T=T, dt=dt, steps=nt, avg_iter=itsum / nt,
                   max_defect=maxdef, M0=M0, E0=E0, Jx0=Jx0, Jy0=Jy0,
                   final_REM=hist["REM"][-1], final_REE=hist["REE"][-1],
                   final_REJx=hist["REJx"][-1], final_REJy=hist["REJy"][-1])
    _write_csv([summary], "exp4_summary.csv")
    return summary


def _texnum(v):
    if v == 0 or not np.isfinite(v):
        return f"\\({v:.1f}\\)"
    e = math.floor(math.log10(abs(v))); m = v / 10**e
    return f"\\({m:.2f}\\times10^{{{e}}}\\)"


def _write_csv(rows, name):
    if not rows:
        return
    with open(os.path.join(OUT, name), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)


if __name__ == "__main__":
    import sys
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    t0 = time.time()
    if which in ("1a", "all"):
        print("Exp1a linear ..."); r = exp1a_linear()
        for x in r: print("  ", x["N"], f"{x['err']:.3e}", _fmt(x["order"]))
    if which in ("1b", "all"):
        print("Exp1b MMS ..."); s, t = exp1b_mms()
        for x in s: print("  spatial N=", x["N"], f"{x['err']:.3e}", _fmt(x["order"]))
        for x in t: print("  temporal dt=", x["dt"], f"{x['err']:.3e}", _fmt(x["order"]))
    if which in ("234", "all"):
        print("Exp2/3/4 ..."); summ = exp234()
        print("  summary:", {k: (f"{v:.3e}" if isinstance(v, float) else v) for k, v in summ.items()})
    print(f"DONE {which} in {time.time()-t0:.1f}s")
