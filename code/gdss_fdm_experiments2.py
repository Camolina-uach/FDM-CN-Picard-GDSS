"""
Batch-1 additional experiments (publication-strengthening):

  compare_<m>  method m in {cons, avg, rk4}: invariant drift vs time on the
               same localized run  ->  results_fdm/compare_<m>.npz
  compare_plot combine the three  ->  figure + table (value of conservation)
  longtime     conservative scheme, long horizon (T=10)  -> drift stays ~1e-14
  recovery     order-2 convergence of the long-wave recovery (w, v, Q)

Run pieces separately (each < time cap):
  python3 gdss_fdm_experiments2.py compare_cons
  python3 gdss_fdm_experiments2.py compare_avg
  python3 gdss_fdm_experiments2.py compare_rk4
  python3 gdss_fdm_experiments2.py compare_plot
  python3 gdss_fdm_experiments2.py longtime
  python3 gdss_fdm_experiments2.py recovery
"""
from __future__ import annotations
import os, sys, math, csv, time
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


def _localized_u0(g, A0=0.2, w0=1.0, kx=1/math.sqrt(2), ky=1/math.sqrt(2)):
    r = np.sqrt(g.X**2 + g.Y**2)
    return g.flat((A0 / np.cosh(r / w0) * np.exp(1j * (kx * g.X + ky * g.Y))).astype(complex))


def _texnum(v):
    if v == 0 or not np.isfinite(v):
        return f"\\({v:.1f}\\)"
    e = math.floor(math.log10(abs(v))); m = v / 10**e
    return f"\\({m:.2f}\\times10^{{{e}}}\\)"


def _order(prev, cur):
    return math.log(prev / cur) / math.log(2) if prev else None


# ------------------------------------------------ non-conservative comparison
def compare_run(method, N=96, L=10.0, T=6.0, dt=4.4248e-3):
    g = FDGrid(-L, L, -L, L, N, N); sol = GDSSSolver(g, EEE)
    U = _localized_u0(g)
    M0, E0 = sol.mass(U), sol.energy(U)
    step = {"cons": sol.step, "avg": sol.step_avg, "rk4": sol.step_rk4}[method]
    nt = round(T / dt)
    ts, REM, REE = [], [], []
    t0 = time.time(); blew = False
    for n in range(nt):
        if method == "rk4":
            U, _ = sol.step_rk4(U, dt)
        else:
            U, _ = step(U, dt, tol=1e-12)
        if not np.all(np.isfinite(U)) or sol.mass(U) > 1e6 * M0:
            blew = True; break
        if n % 5 == 0 or n == nt - 1:
            ts.append((n + 1) * dt)
            REM.append(abs((sol.mass(U) - M0) / M0))
            REE.append(abs((sol.energy(U) - E0) / E0))
    cpu = time.time() - t0
    np.savez(os.path.join(OUT, f"compare_{method}.npz"),
             t=np.array(ts), REM=np.array(REM), REE=np.array(REE),
             cpu=cpu, blew=blew, N=N, T=T, dt=dt)
    print(f"  {method}: steps={nt} cpu={cpu:.1f}s blew={blew} "
          f"finalREM={REM[-1]:.2e} finalREE={REE[-1]:.2e}")


def compare_plot():
    lab = {"cons": "CN–Picard (conservative)", "avg": "CN, averaged nonlinearity (non-conservative)",
           "rk4": "RK4 (explicit)"}
    col = {"cons": "C0", "avg": "C1", "rk4": "C2"}
    d = {m: np.load(os.path.join(OUT, f"compare_{m}.npz")) for m in ("cons", "avg", "rk4")}
    plt.figure(figsize=(10, 4))
    for k, key in enumerate(("REM", "REE")):
        ax = plt.subplot(1, 2, k + 1)
        for m in ("cons", "avg", "rk4"):
            ax.semilogy(d[m]["t"], np.maximum(d[m][key], 1e-17), col[m], label=lab[m])
        ax.axhline(np.finfo(float).eps, color="grey", ls=":", label="machine $\\epsilon$")
        ax.set_xlabel("$t$"); ax.set_title("Mass" if key == "REM" else "Energy")
        ax.set_ylabel("relative drift"); ax.grid(True, alpha=.3)
        if k == 0:
            ax.legend(fontsize=8)
    plt.tight_layout(); plt.savefig(os.path.join(IMG, "compare_nonconservative.png"), dpi=140); plt.close()
    with open(os.path.join(TAB, "compare_nonconservative.tex"), "w") as f:
        f.write("\\begin{tabular}{lllll}\n\\toprule\n"
                "Scheme & conserves? & max $\\mathrm{RE}_M$ & max $\\mathrm{RE}_E$ & CPU (s) \\\\\n\\midrule\n")
        for m, name in (("cons", "CN--Picard (this work)"), ("avg", "CN, averaged nonlin."), ("rk4", "RK4 (explicit)")):
            x = d[m]
            mm, me = float(np.max(x["REM"])), float(np.max(x["REE"]))
            cons = "yes" if m == "cons" else "no"
            f.write(f"{name} & {cons} & {_texnum(mm)} & {_texnum(me)} & \\({float(x['cpu']):.1f}\\) \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")
    print("  wrote compare figure + table")


# ------------------------------------------------ long-time conservation
def longtime(N=112, L=14.0, T=10.0, dt_values=(0.04, 0.02, 0.01)):
    """Long-window test at several time steps.

    The centered packet keeps the boundary interpretation simple.  The outer
    ten percent of interior grid lines is monitored explicitly (maximum
    amplitude in that band divided by the initial peak amplitude), as are the
    momentum drift and Picard iteration count.
    """
    histories = []
    rows = []
    for dt in dt_values:
        g = FDGrid(-L, L, -L, L, N, N)
        sol = GDSSSolver(g, EEE)
        U = _localized_u0(g, kx=0.0, ky=0.0)
        peak0 = float(np.max(np.abs(U)))
        M0, E0 = sol.mass(U), sol.energy(U)
        Jx0, Jy0 = sol.momentum(U)
        nt = round(T / dt)
        stride = max(1, nt // 250)
        ts, REM, REE, JDRIFT, BOUNDARY = [], [], [], [], []
        iterations = []
        band = max(1, round(0.1 * min(g.Nx, g.Ny)))
        band_mask = np.zeros((g.Ny, g.Nx), dtype=bool)
        band_mask[:band, :] = True; band_mask[-band:, :] = True
        band_mask[:, :band] = True; band_mask[:, -band:] = True
        t0 = time.time()
        for n in range(nt):
            U, it = sol.step(U, dt, tol=1e-14)
            iterations.append(it)
            if n % stride == 0 or n == nt - 1:
                ts.append((n + 1) * dt)
                REM.append(abs((sol.mass(U) - M0) / M0))
                REE.append(abs((sol.energy(U) - E0) / E0))
                Jx, Jy = sol.momentum(U)
                JDRIFT.append(max(abs(Jx - Jx0), abs(Jy - Jy0)))
                amplitude = np.abs(g.grid(U))
                BOUNDARY.append(float(np.max(amplitude[band_mask]) / peak0))
        cpu = time.time() - t0
        history = dict(dt=dt, t=np.asarray(ts), REM=np.asarray(REM), REE=np.asarray(REE),
                       JDRIFT=np.asarray(JDRIFT), BOUNDARY=np.asarray(BOUNDARY))
        histories.append(history)
        rows.append(dict(
            dt=dt,
            steps=nt,
            maxREM=float(np.max(REM)),
            maxREE=float(np.max(REE)),
            maxJdrift=float(np.max(JDRIFT)),
            maxBandRelPeak0=float(np.max(BOUNDARY)),
            finalPeakRelPeak0=float(np.max(np.abs(U)) / peak0),
            avgIter=float(np.mean(iterations)),
            maxIter=int(np.max(iterations)),
            cpu=cpu,
        ))
        print(f"  longtime dt={dt:g}: steps={nt} cpu={cpu:.1f}s "
              f"maxREM={np.max(REM):.2e} maxREE={np.max(REE):.2e} "
              f"maxJdrift={np.max(JDRIFT):.2e} boundary={np.max(BOUNDARY):.2e} "
              f"avgIter={np.mean(iterations):.2f}")

    with open(os.path.join(OUT, "longtime.csv"), "w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)

    plt.figure(figsize=(10, 4))
    for index, key in enumerate(("REM", "REE")):
        ax = plt.subplot(1, 2, index + 1)
        for history in histories:
            ax.semilogy(history["t"], np.maximum(history[key], 1e-17),
                        label=rf"$\Delta t={history['dt']:g}$")
        ax.axhline(np.finfo(float).eps, color="grey", ls=":", label="machine $\\epsilon$")
        ax.set_xlabel("$t$"); ax.set_ylabel("relative drift")
        ax.set_title("Mass" if key == "REM" else "Hamiltonian energy")
        ax.legend(fontsize=8); ax.grid(True, alpha=.3)
    plt.tight_layout(); plt.savefig(os.path.join(IMG, "longtime_invariants.png"), dpi=140); plt.close()

    with open(os.path.join(TAB, "longtime.tex"), "w") as stream:
        stream.write("\\begin{tabular}{rrrrrrrrr}\n\\toprule\n")
        stream.write("$\\Delta t$ & steps & max $\\mathrm{RE}_M$ & max $\\mathrm{RE}_E$ & "
                     "max $|\\Delta J|$ & band/peak$_0$ & peak$_T$/peak$_0$ & avg. iter & max iter \\\\\n\\midrule\n")
        for row in rows:
            stream.write(
                f"{row['dt']:.3f} & {row['steps']} & {_texnum(row['maxREM'])} & "
                f"{_texnum(row['maxREE'])} & {_texnum(row['maxJdrift'])} & "
                f"{row['maxBandRelPeak0']:.3f} & {row['finalPeakRelPeak0']:.3f} & "
                f"{row['avgIter']:.2f} & {row['maxIter']} \\\\\n"
            )
        stream.write("\\bottomrule\n\\end{tabular}\n")
    return rows


# ------------------------------------------------ long-wave recovery order
def recovery(L=12.0):
    mms = build_mms(EEE, MMSParams())
    u_exact, _, Sw_grid, Sv_grid, wvq_exact = mms
    rows = []; pw = pv = pq = None
    for N in (24, 48, 96, 192):
        g = FDGrid(-L, L, -L, L, N, N); sol = GDSSSolver(g, EEE)
        rho = np.abs(u_exact(g, 0.0)) ** 2
        W, V, Q = sol.ell.solve_wv(rho, Sw_grid(g), Sv_grid(g))
        we, ve, qe = wvq_exact(g)
        ew, ev, eq = sol.inner.l2(W - we), sol.inner.l2(V - ve), sol.inner.l2(Q - qe)
        rows.append(dict(N=N, h=2 * L / (N + 1), ew=ew, ev=ev, eq=eq,
                         ow=_order(pw, ew), ov=_order(pv, ev), oq=_order(pq, eq)))
        pw, pv, pq = ew, ev, eq
    def od(o):
        return "n/a" if o is None else f"{o:.2f}"
    with open(os.path.join(TAB, "recovery.tex"), "w") as f:
        f.write("\\begin{tabular}{lllllll}\n\\toprule\n"
                "$N$ & $h$ & $\\|W-w^\\star\\|_2$ & ord & $\\|V-v^\\star\\|_2$ & ord & $\\|Q-Q^\\star\\|_2$ (ord) \\\\\n\\midrule\n")
        for r in rows:
            f.write(f"${r['N']}$ & {r['h']:.3e} & {r['ew']:.3e} & {od(r['ow'])} & "
                    f"{r['ev']:.3e} & {od(r['ov'])} & {r['eq']:.3e} ({od(r['oq'])}) \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")
    for r in rows:
        print(f"  N={r['N']:4d} ew={r['ew']:.3e}({r['ow']}) ev={r['ev']:.3e}({r['ov']}) eq={r['eq']:.3e}({r['oq']})")


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    t0 = time.time()
    if which.startswith("compare_") and which != "compare_plot":
        compare_run(which.split("_")[1])
    elif which == "compare_plot":
        compare_plot()
    elif which == "longtime":
        longtime()
    elif which == "recovery":
        recovery()
    print(f"DONE {which} in {time.time()-t0:.1f}s")
