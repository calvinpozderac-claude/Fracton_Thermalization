"""Relative size D = D_LKS / D_sym of the largest Krylov sector (notebook plot
and Fig. 4 of arXiv:2210.02469), using the fast counting in fracton.counting.

    python scripts/lks_fraction.py --Lmax 201 --step 10
"""
import argparse
import math
import os
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from fracton.counting import lks_fraction, lks_fraction_sweep  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--Lmin", type=int, default=11)
    ap.add_argument("--Lmax", type=int, default=201)
    ap.add_argument("--step", type=int, default=10)
    ap.add_argument("--workers", type=int, default=None)
    a = ap.parse_args()
    os.makedirs("results", exist_ok=True)

    # D vs n, same N grid as the notebook: N = L%4+4, ..., < 2L in steps of 4
    Ls = list(range(a.Lmin, a.Lmax + 1, a.step))
    data = {}
    for L in Ls:
        t = time.time()
        ns, Ds = lks_fraction_sweep(L, range(L % 4 + 4, 2 * L, 4), a.workers)
        data[L] = (ns, Ds)
        print(f"L = {L}  time = {time.time() - t:.1f} s", flush=True)

    # D at n = 1 versus L (Fig. 4b)
    Lc = np.unique(np.round(np.logspace(np.log10(5), np.log10(a.Lmax), 30)).astype(int))
    Dc = np.array([lks_fraction(L, L) for L in Lc])

    np.savez("results/lks_fraction.npz",
             L=np.concatenate([[L] * len(data[L][0]) for L in Ls]),
             n=np.concatenate([data[L][0] for L in Ls]),
             D=np.concatenate([data[L][1] for L in Ls]), Lc=Lc, Dc=Dc)

    fig, ax = plt.subplots(1, 3, figsize=(17, 5))
    cols = plt.cm.Blues(np.linspace(0.4, 1, len(Ls)))
    for c, L in zip(cols, Ls):
        ns, Ds = data[L]
        ax[0].plot(ns, Ds, "o-", ms=3, c=c, label=f"L={L}" if L in (Ls[0], Ls[-1]) else None)
        m = ns <= 1
        y = ((ns[m] ** 1.5 / (ns[m] + 1)) * np.sqrt(L) * Ds[m]) ** (1 / L)
        ax[1].plot(ns[m], y, ".", ms=3, c=c)
    ax[0].plot([0, 1, 1, 2], [0, 0, 1, 1], "k--", lw=1)
    ax[0].set_xlabel("n = N/L")
    ax[0].set_ylabel(r"$D = D_{LKS}/D_{sym}$")
    ax[0].legend()
    nn = np.linspace(0.01, 1, 200)
    ax[1].plot(nn, (4 * nn) ** nn / (nn + 1) ** (nn + 1), "r--", label=r"$(4n)^n/(n+1)^{n+1}$")
    ax[1].set_xlabel("n")
    ax[1].set_ylabel(r"$(\frac{n^{3/2}}{n+1}\sqrt{L}D)^{1/L}$")
    ax[1].legend()
    ax[2].loglog(Lc, Dc, "o", ms=4)
    ax[2].loglog(Lc, Dc[-1] * np.sqrt(Lc[-1] / Lc), "k--", label=r"$\sim 1/\sqrt{L}$")
    ax[2].set_xlabel("L")
    ax[2].set_ylabel("D(n=1)")
    ax[2].legend()
    fig.tight_layout()
    fig.savefig("results/lks_fraction.png", dpi=130)
    print("D(n=1):", dict(zip(Lc.tolist(), np.round(Dc, 5).tolist())))


if __name__ == "__main__":
    main()
