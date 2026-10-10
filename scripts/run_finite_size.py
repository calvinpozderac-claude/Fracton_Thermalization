"""Finite-size scaling of the slowest relaxation time, tau(L) ~ L^z.

For each L we record the equilibrium quadrupole moment Q(t) = sum_x (x-xc)^2 n_x
(slowest mode of a closed chain with fixed N and P), starting from uniform
symmetry-sector states, and measure its autocorrelation time. Run lengths
are set to ~`runs` x tau using an estimate extrapolated from the smaller sizes.

    python scripts/run_finite_size.py --n 1 --Ls 12 16 20 24 32 40 48 64
"""
import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from fracton.autocorr import autocorr, efold_time, integrated_time  # noqa: E402
from fracton.dynamics import quadrupole_series  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=float, required=True)
    ap.add_argument("--Ls", type=int, nargs="+", required=True)
    ap.add_argument("--samples", type=int, default=8)
    ap.add_argument("--runs", type=float, default=200, help="run length in units of tau")
    ap.add_argument("--points", type=int, default=50, help="samples of Q per tau")
    ap.add_argument("--z0", type=float, default=5.0, help="initial guess of z for extrapolation")
    a = ap.parse_args()
    out = os.path.join("results", f"finite_size_n{a.n:g}.npz")
    res = {"L": [], "tau_e": [], "tau_int": [], "A": [], "dt": [], "samples": []}
    taus = []
    for i, L in enumerate(a.Ls):
        N = int(round(a.n * L))
        if (N * (L - 1)) % 2:
            print(f"skip L={L}: P not integer", flush=True)
            continue
        if len(taus) >= 3:
            # robust extrapolation: fit through all previous sizes, z clipped to a sane range
            lL, lt = np.log([u for u, _ in taus]), np.log([v for _, v in taus])
            z, c = np.polyfit(lL, lt, 1)
            z = min(max(z, 3.5), 7.0)
            est = np.exp(np.mean(lt - z * lL)) * L ** z
        elif taus:
            est = taus[-1][1] * (L / taus[-1][0]) ** a.z0
        else:
            est = max(10.0, 0.5 * (L / 4) ** a.z0)
        for attempt in range(4):
            dt = max(1, int(est / a.points))
            nsteps = int(a.runs * a.points)
            burn = int(5 * est)
            t0 = time.time()
            Q = quadrupole_series(N, L, dt, nsteps, burn, a.samples, 100 * L + 7 + 1000 * attempt)
            A = autocorr(Q)
            te, ti = efold_time(A, dt), integrated_time(A, dt)
            # redo with a corrected estimate if the run was too short (or sampled too coarsely)
            if np.isfinite(te) and 0.5 * a.runs <= nsteps * dt / te and te >= 5 * dt:
                break
            print(f"  L={L}: retry (tau_e={te:.4g}, est {est:.3g})", flush=True)
            est = te if np.isfinite(te) else 10 * est
        taus.append((L, te))
        res["L"].append(L)
        res["tau_e"].append(te)
        res["tau_int"].append(ti)
        res["A"].append(A[: 10 * a.points])
        res["dt"].append(dt)
        res["samples"].append(Q)
        print(f"n={a.n} L={L} N={N}: tau_e={te:.4g} tau_int={ti:.4g} (est {est:.3g}, dt {dt}, T/tau={nsteps*dt/te:.0f}) {time.time()-t0:.0f}s", flush=True)
        np.savez(out, L=np.array(res["L"]), tau_e=np.array(res["tau_e"]), tau_int=np.array(res["tau_int"]),
                 dt=np.array(res["dt"]), A=np.array(res["A"]), n=a.n, samples=a.samples)


if __name__ == "__main__":
    main()
