"""Generate the graphical abstract (schematic + energy-drift comparison)."""
from __future__ import annotations
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = "results_fdm"; IMG = "Images"


def make():
    d = {m: np.load(os.path.join(OUT, f"compare_{m}.npz")) for m in ("cons", "avg", "rk4")}
    fig = plt.figure(figsize=(12, 4.2))
    ax = fig.add_subplot(1, 2, 1); ax.axis("off"); ax.set_xlim(0, 10); ax.set_ylim(0, 10)

    def box(x, y, w, h, txt, fc):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1", fc=fc, ec="k", lw=1.2))
        ax.text(x + w / 2, y + h / 2, txt, ha="center", va="center", fontsize=10)

    def arr(x1, y1, x2, y2, txt=""):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=15, lw=1.4, color="k"))
        if txt:
            ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.3, txt, ha="center", fontsize=8, style="italic")

    box(0.3, 6.2, 4.2, 2.2, "short-wave  $u$\n$iu_t+\\alpha u_{xx}+\\beta u_{yy}$\n$=\\gamma|u|^2u+\\xi u\\,Q$", "#cfe8ff")
    box(5.5, 6.2, 4.2, 2.2, "long-wave  $w,v$\nsparse elliptic\n$\\mathcal{A}_h[w;v]=\\nabla|u|^2$", "#ffe6cc")
    arr(4.5, 7.3, 5.5, 7.3, "$|u|^2$"); arr(5.5, 6.6, 4.5, 6.6, "$Q=w_x+v_y$")
    box(2.6, 3.4, 4.8, 1.8, "Crank-Nicolson step\n+ Picard iteration\n(reused sparse LU)", "#e6ffe6")
    arr(4.9, 6.2, 5.0, 5.2)
    box(1.3, 0.6, 7.4, 1.6, "Discrete invariants preserved:\nmass & energy to machine $\\epsilon$;  momentum balance", "#f0e6ff")
    arr(5.0, 3.4, 5.0, 2.2)
    ax.text(5, 9.4, "Conservative CN-Picard scheme for the GDSS", ha="center", fontsize=12, weight="bold")

    ax2 = fig.add_subplot(1, 2, 2)
    lab = {"cons": "CN-Picard (this work)", "avg": "CN, averaged nonlin.", "rk4": "RK4 (explicit)"}
    col = {"cons": "C0", "avg": "C1", "rk4": "C2"}
    for m in ("cons", "avg", "rk4"):
        ax2.semilogy(d[m]["t"], np.maximum(d[m]["REE"], 1e-17), col[m], lw=2, label=lab[m])
    ax2.axhline(np.finfo(float).eps, color="grey", ls=":", label="machine $\\epsilon$")
    ax2.set_xlabel("time $t$"); ax2.set_ylabel("relative energy drift"); ax2.grid(True, alpha=.3)
    ax2.legend(fontsize=8, loc="center right"); ax2.set_title("Energy conservation vs. standard schemes")
    plt.tight_layout(); plt.savefig(os.path.join(IMG, "graphical_abstract.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("wrote Images/graphical_abstract.png")


if __name__ == "__main__":
    make()
