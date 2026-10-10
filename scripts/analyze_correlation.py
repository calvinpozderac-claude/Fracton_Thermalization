"""Extract the dynamical exponent from C(x, t) (Sec. V.B of arXiv:2210.02469).

x0(t) is the first zero of C(x, t); x0 ~ t^{1/z}. We
  * subtract the static finite-size offset of C (fixed N and P make
    sum_x C(x, t) = 0 over the whole system, giving C a ~ -n(n+1)/L tail),
    estimated from the equal-time correlation C(x >= 1, 0);
  * locate the zero by a local linear fit of C around the first sign change;
  * estimate errors by bootstrap over independent samples;
  * fit z on the paper's windows and report a running exponent z_eff(t).

    python scripts/analyze_correlation.py results/corr_n1_*.npz results/corr_n2_*.npz ...
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from fracton.dynamics import fit_z  # noqa: E402


def zero_crossing(C):
    """First + to - crossing of C(x), refined with a linear fit over nearby points."""
    for x in range(1, len(C)):
        if C[x] <= 0 < C[x - 1]:
            xr = x - 1 + C[x - 1] / (C[x - 1] - C[x])
            if xr < 8:
                return xr
            w = int(0.25 * xr)
            lo, hi = max(0, x - 1 - w), min(len(C), x + w + 1)
            xs = np.arange(lo, hi)
            a, b = np.polyfit(xs, C[lo:hi], 1)
            xf = -b / a if a < 0 else xr
            return xf if lo <= xf <= hi else xr
    return np.nan


def load(path):
    d = np.load(path)
    Csum, cnt = d["Csum"], d["cnt"]
    return dict(Csum=Csum, cnt=cnt, lags=d["lags"], N=int(d["N"]), L=int(d["L"]), path=path)


def curves(Csum, cnt):
    C = Csum.sum(0) / cnt.sum(0)[:, None]
    static = C[-1]
    off = static[1:].mean()
    return C[:-1] - off, off, static


def x0_of(Csum, cnt):
    C, _, _ = curves(Csum, cnt)
    return np.array([zero_crossing(c) for c in C])


def bootstrap_x0(d, nboot=200, seed=0):
    rng = np.random.default_rng(seed)
    S = d["Csum"].shape[0]
    xs = np.array([x0_of(d["Csum"][idx], d["cnt"][idx])
                   for idx in (rng.integers(0, S, S) for _ in range(nboot))])
    return np.nanstd(xs, 0)


def running_z(t, x0, width=1.0):
    """z_eff from a fit of log x0 vs log t over a window of `width` decades."""
    lt = np.log10(t)
    out = np.full(len(t), np.nan)
    for i in range(len(t)):
        m = np.abs(lt - lt[i]) <= width / 2
        m &= np.isfinite(x0)
        if m.sum() >= 4 and lt[i] - width / 2 >= lt[0] - 1e-9 and lt[i] + width / 2 <= lt[-1] + 1e-9:
            s = np.polyfit(lt[m], np.log10(x0[m]), 1)[0]
            out[i] = 1 / s
    return out


def bootstrap_fit(d, tmin, tmax, nboot=200, seed=1):
    rng = np.random.default_rng(seed)
    S = d["Csum"].shape[0]
    zs = []
    for _ in range(nboot):
        idx = rng.integers(0, S, S)
        x0 = x0_of(d["Csum"][idx], d["cnt"][idx])
        try:
            zs.append(fit_z(d["lags"], x0, tmin, tmax)[0])
        except Exception:
            pass
    return np.nanstd(zs)


def main(paths):
    ds = sorted((load(p) for p in paths), key=lambda d: d["N"] / d["L"])
    fig, ax = plt.subplots(1, 3, figsize=(17, 5))
    report = []
    cmap = {1.0: "C3", 1.5: "C1", 2.0: "C0", 3.0: "C2"}
    for d in ds:
        n = d["N"] / d["L"]
        t = d["lags"]
        C, off, static = curves(d["Csum"], d["cnt"])
        x0 = np.array([zero_crossing(c) for c in C])
        err = bootstrap_x0(d)
        col = cmap.get(round(n, 3), None)
        lab = f"n={n:g} (L={d['L']}, {d['Csum'].shape[0]} samples)"
        ax[1].errorbar(t, x0, err, fmt="o", ms=3, c=col, label=lab)
        zr = running_z(t, x0)
        ax[2].plot(t, zr, "o-", ms=3, c=col, label=f"n={n:g}")
        lines = [f"## n = {n:g}  (N={d['N']}, L={d['L']}, samples={d['Csum'].shape[0]})",
                 f"static offset of C: {off:.5f}  (-n(n+1)/L = {-n*(n+1)/d['L']:.5f}),  C(0,0) = {static[0]:.4f}  (n(n+1) = {n*(n+1):.4f})"]
        windows = [(10, 1e5), (10 ** 5.5, 1e7), (1e2, 1e4), (1e4, 1e6), (1e3, t[-1])]
        for tmin, tmax in windows:
            if tmax > t[-1] * 1.0001 or tmin < t[0]:
                continue
            z, zerr, c0 = fit_z(t, x0, tmin, tmax)
            zb = bootstrap_fit(d, tmin, tmax)
            lines.append(f"z fit on {tmin:.3g} <= t <= {tmax:.3g}: z = {z:.2f} +- {zb:.2f} (bootstrap)")
            if (tmin, tmax) in [(10, 1e5), (10 ** 5.5, 1e7)] or (n != 1 and (tmin, tmax) == (1e4, 1e6)):
                tt = np.logspace(np.log10(tmin), np.log10(tmax), 10)
                ax[1].plot(tt, np.exp(c0) * tt ** (1 / z), "k--", lw=1)
                ax[1].text(tt[5], np.exp(c0) * tt[5] ** (1 / z) * 1.25, f"z={z:.1f}", fontsize=8)
        # raw-C (no offset subtraction) for comparison
        Craw = d["Csum"].sum(0)[:-1] / d["cnt"].sum(0)[:-1, None]
        x0raw = np.array([zero_crossing(c) for c in Craw])
        lines.append("x0(t) [t, x0, err, x0 without offset subtraction, z_eff]:")
        for i in range(len(t)):
            lines.append(f"  {t[i]:>9d}  {x0[i]:7.3f}  {err[i]:6.3f}  {x0raw[i]:7.3f}  {zr[i]:6.2f}")
        report.append("\n".join(lines))
        np.savez(os.path.splitext(d["path"])[0] + "_x0.npz", t=t, x0=x0, err=err, zeff=zr)

        if abs(n - 1) < 1e-9:
            # Fig. 6 of the paper: C(x, t) x0 vs x / x0
            sel = np.where(t >= 100)[0]
            sel = sel[np.linspace(0, len(sel) - 1, min(11, len(sel))).astype(int)]
            cols = plt.cm.Blues(np.linspace(0.3, 1, len(sel)))
            xs = np.arange(C.shape[1])
            for c, i in zip(cols, sel):
                ax[0].plot(xs / x0[i], C[i] * x0[i], c=c, lw=1.2, label=f"t=1e{np.log10(t[i]):.1f}" if i in (sel[0], sel[-1]) else None)
            ax[0].axhline(0, c="k", ls=":", lw=1)
            ax[0].set_xlim(0, 3)
            ax[0].set_xlabel(r"$x/x_0$")
            ax[0].set_ylabel(r"$C(x,t)\,x_0$")
            ax[0].set_title("n = 1 (critical): collapse of C(x,t)")
            ax[0].legend()

    ax[1].set_xscale("log")
    ax[1].set_yscale("log")
    ax[1].set_xlabel("t")
    ax[1].set_ylabel(r"$x_0(t)$ (first zero of C)")
    ax[1].legend(fontsize=8)
    ax[2].axhline(4, c="k", ls=":", lw=1)
    ax[2].set_xscale("log")
    ax[2].set_ylim(2, 12)
    ax[2].set_xlabel("t")
    ax[2].set_ylabel(r"$z_{eff}$ (1-decade sliding fit)")
    ax[2].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig("results/dynamical_exponent.png", dpi=130)
    txt = "\n\n".join(report)
    open("results/dynamical_exponent.txt", "w").write(txt + "\n")
    print(txt)


if __name__ == "__main__":
    main(sys.argv[1:])
