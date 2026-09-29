"""
Level-2 physical-dynamics experiments:

  collision  head-on collision of two localized packets: nonlinear interaction,
             induced long-wave field Q, invariants conserved to round-off
  focusing   self-focusing run (gamma<0): peak amplitude grows while the
             discrete mass and energy stay conserved
  cost       CPU-time scaling vs grid size N, Picard iteration counts

Run:  python3 gdss_fdm_experiments4.py {collision|focusing|cost}
"""
from __future__ import annotations
import os, sys, math, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from gdss_fdm_solver import GDSSParams, FDGrid, GDSSSolver

OUT = "results_fdm"; TAB = os.path.join(OUT, "tables"); IMG = "Images"
for d in (OUT, TAB, IMG):
    os.makedirs(d, exist_ok=True)
EEE = GDSSParams(alpha=1, beta=1, gamma=1, xi=1, psi=1, eta=2, phi=2, chi=1, theta=1)


def _texnum(v):
    if v == 0 or not np.isfinite(v):
        return f"\\({v:.1f}\\)"
    e = math.floor(math.log10(abs(v))); m = v / 10**e
    return f"\\({m:.2f}\\times10^{{{e}}}\\)"


def _packet(g, x0, y0, A, w0, kx, ky):
    r = np.sqrt((g.X - x0)**2 + (g.Y - y0)**2)
    return A / np.cosh(r / w0) * np.exp(1j * (kx * (g.X - x0) + ky * (g.Y - y0)))


def _collision_grid(snaps, snap_t, ext, field, fname):
    """Print-size 2x2 grid (5 in wide, the IJMPC text width) of |u| (field=0)
    or Q (field=1) at the four snapshot times; Q uses limits symmetric about 0."""
    label, cmap = ("$|u|$", "viridis") if field == 0 else ("$Q$", "RdBu_r")
    with plt.rc_context({"font.size": 8, "axes.titlesize": 9}):
        fig, axes = plt.subplots(2, 2, figsize=(5.0, 4.7), constrained_layout=True)
        for k, (ax, st) in enumerate(zip(axes.flat, snap_t)):
            f = snaps[st][field]
            lim = dict(vmin=0) if field == 0 else dict(vmin=-np.abs(f).max(), vmax=np.abs(f).max())
            im = ax.imshow(f, extent=ext, origin="lower", cmap=cmap, **lim)
            ax.set_title(f"{label}, $t={st:g}$")
            if k % 2 == 0:
                ax.set_ylabel("$y$")
            if k >= 2:
                ax.set_xlabel("$x$")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        fig.savefig(os.path.join(IMG, fname), dpi=300)
        plt.close(fig)


# --------------------------------------------------------- two-packet collision
def collision(N=128, L=16.0, T=6.0, dt=0.02):
    g = FDGrid(-L, L, -L, L, N, N); sol = GDSSSolver(g, EEE)
    u0 = _packet(g, -6, 0, 0.3, 1.0, 1.2, 0.0) + _packet(g, 6, 0, 0.3, 1.0, -1.2, 0.0)
    U = g.flat(u0.astype(complex))
    M0, E0 = sol.mass(U), sol.energy(U); Jx0, Jy0 = sol.momentum(U)
    nt = round(T / dt)
    snap_t = [0.0, 2.5, 4.0, 6.0]; snaps = {}
    ts, REM, REE = [], [], []
    for st in snap_t:
        if st == 0.0:
            _, _, Q = sol.ell.solve_wv(np.abs(U)**2)
            snaps[st] = (np.abs(g.grid(U)).copy(), g.grid(Q).copy())
    t0 = time.time()
    for n in range(nt):
        U, _ = sol.step(U, dt, tol=1e-12)
        t = (n + 1) * dt
        if n % 4 == 0:
            ts.append(t); REM.append(abs((sol.mass(U)-M0)/M0)); REE.append(abs((sol.energy(U)-E0)/E0))
        for st in snap_t:
            if st > 0 and abs(t - st) < dt/2:
                _, _, Q = sol.ell.solve_wv(np.abs(U)**2)
                snaps[st] = (np.abs(g.grid(U)).copy(), g.grid(Q).copy())
    cpu = time.time() - t0
    ext = [g.xL, g.xR, g.yL, g.yR]
    fig, axes = plt.subplots(2, 4, figsize=(15, 7))
    for j, st in enumerate(snap_t):
        au, Q = snaps[st]
        im0 = axes[0, j].imshow(au, extent=ext, origin="lower", cmap="viridis", vmin=0)
        axes[0, j].set_title(f"$|u|$, $t={st:g}$"); plt.colorbar(im0, ax=axes[0, j], fraction=0.046)
        im1 = axes[1, j].imshow(Q, extent=ext, origin="lower", cmap="RdBu_r")
        axes[1, j].set_title(f"$Q$, $t={st:g}$"); plt.colorbar(im1, ax=axes[1, j], fraction=0.046)
    plt.tight_layout(); plt.savefig(os.path.join(IMG, "collision_snapshots.png"), dpi=130); plt.close()
    # separate |u| and Q figures at print size for the IJMPC version
    _collision_grid(snaps, snap_t, ext, 0, "collision_absu.png")
    _collision_grid(snaps, snap_t, ext, 1, "collision_Q.png")
    plt.figure(figsize=(6, 4))
    plt.semilogy(ts, np.maximum(REM, 1e-17), label="$\\mathrm{RE}_M$")
    plt.semilogy(ts, np.maximum(REE, 1e-17), label="$\\mathrm{RE}_E$")
    plt.axhline(np.finfo(float).eps, color="grey", ls=":", label="machine $\\epsilon$")
    plt.xlabel("$t$"); plt.ylabel("relative drift"); plt.title("Collision: invariants")
    plt.legend(); plt.grid(True, alpha=.3); plt.tight_layout()
    plt.savefig(os.path.join(IMG, "collision_invariants.png"), dpi=130); plt.close()
    print(f"  collision: steps={nt} cpu={cpu:.1f}s maxREM={max(REM):.2e} maxREE={max(REE):.2e}")


# --------------------------------------------------------- self-focusing (gamma<0)
def focusing(N=112, L=10.0, T=1.0, dt=0.004):
    par = GDSSParams(alpha=1, beta=1, gamma=-3.0, xi=1.0, psi=1, eta=2, phi=2, chi=1, theta=1)
    g = FDGrid(-L, L, -L, L, N, N); sol = GDSSSolver(g, par)
    u0 = _packet(g, 0, 0, 1.0, 1.2, 0.0, 0.0)
    U = g.flat(u0.astype(complex))
    M0, E0 = sol.mass(U), sol.energy(U)
    nt = round(T / dt)
    ts, peak, REM, REE, iters = [], [], [], [], []
    t0 = time.time()
    for n in range(nt):
        U, it = sol.step(U, dt, tol=1e-12, kmax=60)
        if not np.all(np.isfinite(U)):
            print(f"  non-finite at t={(n+1)*dt:.3f}"); break
        t = (n + 1) * dt
        if n % 3 == 0:
            ts.append(t); peak.append(float(np.max(np.abs(U))))
            REM.append(abs((sol.mass(U)-M0)/M0)); REE.append(abs((sol.energy(U)-E0)/E0)); iters.append(it)
    cpu = time.time() - t0
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(ts, peak); ax[0].set_xlabel("$t$"); ax[0].set_ylabel("$\\max|u|$")
    ax[0].set_title("Self-focusing: peak amplitude"); ax[0].grid(True, alpha=.3)
    ax[1].semilogy(ts, np.maximum(REM, 1e-17), label="$\\mathrm{RE}_M$")
    ax[1].semilogy(ts, np.maximum(REE, 1e-17), label="$\\mathrm{RE}_E$")
    ax[1].axhline(np.finfo(float).eps, color="grey", ls=":", label="machine $\\epsilon$")
    ax[1].set_xlabel("$t$"); ax[1].set_title("Invariants during focusing"); ax[1].legend(); ax[1].grid(True, alpha=.3)
    plt.tight_layout(); plt.savefig(os.path.join(IMG, "focusing.png"), dpi=130); plt.close()
    print(f"  focusing: cpu={cpu:.1f}s peak0={peak[0]:.3f} peakT={peak[-1]:.3f} "
          f"maxREM={max(REM):.2e} maxREE={max(REE):.2e} maxiter={max(iters)}")


# --------------------------------------------------------- cost scaling
def cost():
    rows = []
    for N in (48, 64, 96, 128, 160):
        g = FDGrid(-10, 10, -10, 10, N, N); sol = GDSSSolver(g, EEE)
        r = np.sqrt(g.X**2 + g.Y**2)
        U = g.flat((0.2 / np.cosh(r) * np.exp(1j*(g.X+g.Y)/math.sqrt(2))).astype(complex))
        nt = 50; t0 = time.time(); its = []
        for n in range(nt):
            U, it = sol.step(U, 4.4248e-3, tol=1e-12); its.append(it)
        cpu = time.time() - t0
        rows.append(dict(N=N, dof=2*N*N, cpu_step=cpu/nt, avg_it=float(np.mean(its))))
        print(f"  N={N:4d} dof={2*N*N} cpu/step={cpu/nt*1e3:.1f}ms avg_it={np.mean(its):.2f}")
    Ns = np.array([r["N"] for r in rows]); cs = np.array([r["cpu_step"] for r in rows])
    plt.figure(figsize=(6, 4))
    plt.loglog(Ns, cs, "o-", label="CPU/step")
    plt.loglog(Ns, cs[0]*(Ns/Ns[0])**2, "k--", label="$\\mathcal{O}(N^2)$")
    plt.loglog(Ns, cs[0]*(Ns/Ns[0])**3, "k:", label="$\\mathcal{O}(N^3)$")
    plt.xlabel("$N$ (per direction)"); plt.ylabel("CPU per step (s)")
    plt.title("Cost scaling"); plt.legend(); plt.grid(True, which="both", alpha=.3)
    plt.tight_layout(); plt.savefig(os.path.join(IMG, "cost_scaling.png"), dpi=130); plt.close()
    with open(os.path.join(TAB, "cost.tex"), "w") as f:
        f.write("\\begin{tabular}{llll}\n\\toprule\n$N$ & d.o.f.\\ ($2N^2$) & CPU/step (s) & avg.\\ Picard iter \\\\\n\\midrule\n")
        for r in rows:
            f.write(f"\\({r['N']}\\) & \\({r['dof']}\\) & {r['cpu_step']:.3e} & \\({r['avg_it']:.2f}\\) \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    t0 = time.time()
    {"collision": collision, "focusing": focusing, "cost": cost}[which]()
    print(f"DONE {which} in {time.time()-t0:.1f}s")
