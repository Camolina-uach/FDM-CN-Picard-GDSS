"""
Batch-2 appendix experiments (robustness):

  picard    Picard iteration count vs dt (incl. dt >> CFL, A-stability) and
            vs amplitude; mass/energy stay conserved  -> tables
  boundary  sensitivity to domain size L: boundary tail and momentum drift
            decrease as L grows (defends the Dirichlet approximation)
  regimes   several EEE parameter sets + focusing/defocusing: conservation
            is robust across the regime

Run:  python3 gdss_fdm_experiments3.py {picard|boundary|regimes}
"""
from __future__ import annotations
import os, sys, math, time
import numpy as np
from gdss_fdm_solver import GDSSParams, FDGrid, GDSSSolver

OUT = "results_fdm"; TAB = os.path.join(OUT, "tables")
os.makedirs(TAB, exist_ok=True)
EEE = GDSSParams(alpha=1, beta=1, gamma=1, xi=1, psi=1, eta=2, phi=2, chi=1, theta=1)
CFL = 2.8 / (4 / (20/97)**2 + 4 / (20/97)**2)   # ~ explicit RK4 step at N=96,L=10


def _u0(g, A0=0.2, w0=1.0, kx=1/math.sqrt(2), ky=1/math.sqrt(2)):
    r = np.sqrt(g.X**2 + g.Y**2)
    return g.flat((A0 / np.cosh(r / w0) * np.exp(1j*(kx*g.X + ky*g.Y))).astype(complex))


def _texnum(v):
    if v == 0 or not np.isfinite(v):
        return f"\\({v:.1f}\\)"
    e = math.floor(math.log10(abs(v))); m = v/10**e
    return f"\\({m:.2f}\\times10^{{{e}}}\\)"


def _run(sol, g, U, dt, T, tol=1e-12):
    M0, E0 = sol.mass(U), sol.energy(U); Jx0, Jy0 = sol.momentum(U)
    nt = round(T/dt); its = []; mM = mE = mJ = 0.0
    for n in range(nt):
        U, it = sol.step(U, dt, tol=tol); its.append(it)
        mM = max(mM, abs((sol.mass(U)-M0)/M0)); mE = max(mE, abs((sol.energy(U)-E0)/E0))
        Jx, Jy = sol.momentum(U); mJ = max(mJ, abs(Jx-Jx0), abs(Jy-Jy0))
    return U, dict(avg_it=float(np.mean(its)), max_it=int(np.max(its)),
                   maxREM=mM, maxREE=mE, maxJdrift=mJ, nt=nt)


# ------------------------------------------------------- Picard robustness
def picard():
    g = FDGrid(-10, 10, -10, 10, 96, 96); sol = GDSSSolver(g, EEE)
    rows = []
    for fac in (0.5, 1, 2, 4, 8, 16):
        dt = fac * CFL
        _, r = _run(sol, g, _u0(g), dt, min(1.0, 40*dt))
        rows.append(dict(dt=dt, fac=fac, **r))
        print(f"  dt={dt:.3e} ({fac}xCFL) avg_it={r['avg_it']:.2f} max_it={r['max_it']} REM={r['maxREM']:.2e} REE={r['maxREE']:.2e}")
    arows = []
    for A0 in (0.2, 0.5, 1.0, 1.5):
        _, r = _run(sol, g, _u0(g, A0=A0), 4.4248e-3, 0.5)
        arows.append(dict(A0=A0, **r)); print(f"  A0={A0} avg_it={r['avg_it']:.2f} REE={r['maxREE']:.2e}")
    with open(os.path.join(TAB, "picard_dt.tex"), "w") as f:
        f.write("\\begin{tabular}{lllllll}\n\\toprule\n$\\Delta t$ & $\\Delta t/\\Delta t_{\\mathrm{CFL}}$ & avg.\\ iter & max iter & max $\\mathrm{RE}_M$ & max $\\mathrm{RE}_E$ \\\\\n\\midrule\n")
        for r in rows:
            f.write(f"{r['dt']:.3e} & \\({r['fac']:g}\\) & \\({r['avg_it']:.2f}\\) & \\({r['max_it']}\\) & {_texnum(r['maxREM'])} & {_texnum(r['maxREE'])} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")
    with open(os.path.join(TAB, "picard_amp.tex"), "w") as f:
        f.write("\\begin{tabular}{llll}\n\\toprule\n$A_0$ & avg.\\ iter & max iter & max $\\mathrm{RE}_E$ \\\\\n\\midrule\n")
        for r in arows:
            f.write(f"\\({r['A0']}\\) & \\({r['avg_it']:.2f}\\) & \\({r['max_it']}\\) & {_texnum(r['maxREE'])} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")


# ------------------------------------------------------- boundary / domain
def boundary():
    rows = []
    for L, N in ((8, 52), (10, 64), (14, 90), (20, 128)):
        g = FDGrid(-L, L, -L, L, N, N); sol = GDSSSolver(g, EEE)
        U0 = _u0(g); U, r = _run(sol, g, U0, 4.4248e-3, 2.0)
        Ug = g.grid(U); bmax = max(np.abs(Ug[0, :]).max(), np.abs(Ug[-1, :]).max(),
                                   np.abs(Ug[:, 0]).max(), np.abs(Ug[:, -1]).max())
        tail = bmax / np.abs(Ug).max()
        rows.append(dict(L=L, N=N, h=2*L/(N+1), tail=tail, Jdrift=r['maxJdrift'], REM=r['maxREM'], REE=r['maxREE']))
        print(f"  L={L} N={N} tail={tail:.2e} Jdrift={r['maxJdrift']:.2e} REM={r['maxREM']:.2e}")
    with open(os.path.join(TAB, "boundary.tex"), "w") as f:
        f.write("\\begin{tabular}{llllll}\n\\toprule\n$L$ & $N$ & boundary tail & max $|\\Delta J|$ & max $\\mathrm{RE}_M$ & max $\\mathrm{RE}_E$ \\\\\n\\midrule\n")
        for r in rows:
            f.write(f"\\({r['L']}\\) & \\({r['N']}\\) & {_texnum(r['tail'])} & {_texnum(r['Jdrift'])} & {_texnum(r['REM'])} & {_texnum(r['REE'])} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")


# ------------------------------------------------------- parameter regimes
def regimes():
    def theta(psi, eta, phi, chi):
        return math.sqrt((phi-psi)*(eta-chi))
    cases = [
        ("baseline EEE, defocusing", dict(gamma=1, xi=1, psi=1, eta=2, phi=2, chi=1)),
        ("anisotropic EEE", dict(gamma=1, xi=1, psi=1, eta=3, phi=2, chi=1)),
        ("focusing $\\gamma<0$", dict(gamma=-1, xi=1, psi=1, eta=2, phi=2, chi=1)),
        ("strong coupling $\\xi=3$", dict(gamma=1, xi=3, psi=1, eta=2, phi=2, chi=1)),
    ]
    g = FDGrid(-10, 10, -10, 10, 96, 96)
    rows = []
    for name, kw in cases:
        th = theta(kw['psi'], kw['eta'], kw['phi'], kw['chi'])
        par = GDSSParams(alpha=1, beta=1, theta=th, **kw)
        sol = GDSSSolver(g, par)
        _, r = _run(sol, g, _u0(g), 4.4248e-3, 1.0)
        rows.append(dict(name=name, th=th, **r))
        print(f"  {name}: REM={r['maxREM']:.2e} REE={r['maxREE']:.2e} avg_it={r['avg_it']:.2f}")
    with open(os.path.join(TAB, "regimes.tex"), "w") as f:
        f.write("\\begin{tabular}{lllll}\n\\toprule\nregime & $\\theta$ & max $\\mathrm{RE}_M$ & max $\\mathrm{RE}_E$ & avg.\\ iter \\\\\n\\midrule\n")
        for r in rows:
            f.write(f"{r['name']} & \\({r['th']:.3f}\\) & {_texnum(r['maxREM'])} & {_texnum(r['maxREE'])} & \\({r['avg_it']:.2f}\\) \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    t0 = time.time()
    {"picard": picard, "boundary": boundary, "regimes": regimes}[which]()
    print(f"DONE {which} in {time.time()-t0:.1f}s")
