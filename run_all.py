#!/usr/bin/env python3
"""
Reproduce every figure and table of the paper

  "A Conservative Crank-Nicolson-Picard Finite-Difference Scheme for the
   Generalized Davey-Stewartson System: Discrete Invariants and Benchmark
   Validation"

Running this script regenerates all outputs directly into ``paper/Images`` and
``paper/results_fdm/tables`` so that ``paper/main.tex`` compiles with fresh
results.

Usage:
    python run_all.py            # run everything (several minutes)
    python run_all.py --fast     # skip the most expensive runs

The full run takes a few minutes on a laptop; the expensive parts are the
N=192 manufactured-solution convergence and the collision/focusing dynamics.
"""
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
MPLCONFIG = os.path.join(HERE, "tmp", "matplotlib")
os.makedirs(MPLCONFIG, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", MPLCONFIG)
sys.path.insert(0, os.path.join(HERE, "code"))
PAPER = os.path.join(HERE, "paper")
os.makedirs(PAPER, exist_ok=True)
os.chdir(PAPER)                      # experiments write Images/ and results_fdm/ here

FAST = "--fast" in sys.argv


def banner(msg):
    print("\n" + "=" * 70 + f"\n  {msg}\n" + "=" * 70, flush=True)


def main():
    t0 = time.time()

    import gdss_fdm_experiments as E1
    import gdss_fdm_experiments2 as E2
    import gdss_fdm_experiments3 as E3
    import gdss_fdm_experiments4 as E4
    import gdss_fdm_reduction as ER
    import gdss_fdm_coupling_validation as CV
    import gdss_fdm_momentum_refinement as MR
    import gdss_fdm_cost_comparison as CC
    import verify_reference_outputs as VR
    import make_graphical_abstract as GA

    banner("Experiment 1a: linear Schrodinger convergence")
    E1.exp1a_linear()

    banner("Experiment 1b: manufactured-solution convergence (spatial + temporal)")
    E1.exp1b_mms()

    banner("Experiments 2-4 (core): conservation, momentum balance, localized run")
    E1.exp234()

    banner("Momentum drift under spatial and temporal refinement")
    MR.run()

    banner("Long-wave recovery convergence (w, v, Q)")
    E2.recovery()

    banner("Manufactured-solution theta-coupling validation")
    CV.run()

    banner("Non-conservative comparison (conservative vs averaged-CN vs RK4)")
    for m in ("cons", "avg", "rk4"):
        E2.compare_run(m)
    E2.compare_plot()

    banner("Long-time conservation")
    E2.longtime()

    banner("Appendix: Picard robustness, boundary sensitivity, parameter regimes")
    E3.picard()
    E3.boundary()
    E3.regimes()

    banner("DS-reduction cross-check")
    ER.run()

    banner("Computational cost scaling")
    E4.cost()

    banner("Reused sparse LU versus repeated conjugate gradients")
    CC.run((64, 128, 256) if FAST else (64, 128, 256, 512),
           steps=1)

    if not FAST:
        banner("Nonlinear dynamics: two-packet collision")
        E4.collision()
        banner("Nonlinear dynamics: self-focusing")
        E4.focusing()
    else:
        print("[--fast] skipping collision and focusing runs")

    banner("Graphical abstract")
    GA.make()

    banner("Archived-reference comparison")
    VR.verify_if_available({"cost_comparison.csv"} if FAST else set())

    print(f"\nAll done in {time.time() - t0:.1f} s. "
          f"Outputs written under {PAPER}/Images and {PAPER}/results_fdm.")


if __name__ == "__main__":
    main()
