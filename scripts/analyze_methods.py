"""Six estimators of the dynamical exponent z from the simulation data.

Bulk (L = 1000, from results/corr_n*_L1000_*.npz):
  1. first zero x0(t) of C(x,t)             x0 ~ t^{1/z}         (paper, Sec. V.B)
  2. return probability C(0,t)              C(0,t) ~ t^{-1/z}
  3. scaling collapse C = t^{-1/z} F(x/t^{1/z}), z chosen to minimise the spread
  4. dynamic structure factor S(k,t), 1/e time tau(k) ~ k^{-z}
  5. charge-transfer variance across a bond  Var dN(t) ~ t^{1/z}
Finite size (results/finite_size_n*.npz):
  6. quadrupole relaxation time             tau(L) ~ L^z

Errors are bootstrap standard deviations over independent samples (bulk) or a
fit error plus the spread of two tau definitions (finite size).

    python scripts/analyze_methods.py
"""
import glob
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_correlation import zero_crossing  # noqa: E402

RNG = np.random.default_rng(12345)
NBOOT = int(os.environ.get("NBOOT", 100))


def slope_fit(x, y, xmin, xmax):
    m = (x >= xmin) & (x <= xmax) & np.isfinite(y) & (y > 0)
    if m.sum() < 3:
        return np.nan
    return np.polyfit(np.log(x[m]), np.log(y[m]), 1)[0]


# --------------------------------------------------------------- bulk estimators
def mean_C(Csum, cnt):
    C = Csum.sum(0) / cnt.sum(0)[:, None]
    off = C[-1, 1:].mean()
    return C[:-1] - off, C[-1] - off


def est_x0(C, t, w):
    x0 = np.array([zero_crossing(c) for c in C])
    return 1 / slope_fit(t, x0, *w), x0


def est_return(C, t, w):
    c0 = C[:, 0]
    return -1 / slope_fit(t, c0, *w), c0


def est_collapse(C, t, w, zs=np.linspace(3, 9, 121)):
    sel = (t >= w[0]) & (t <= w[1])
    Cs, ts = C[sel], t[sel]
    xmax = C.shape[1] - 1
    xs = np.arange(C.shape[1])
    costs = []
    for z in zs:
        s = ts ** (1 / z)
        # common u-range covered by every time, starting at u where x >= 2 for the earliest time
        u = np.linspace(2 / s[0], 0.8 * xmax / s[-1], 60)
        if u[-1] <= u[0]:
            costs.append(np.inf)
            continue
        Y = np.array([np.interp(u * si, xs, c) * si for c, si in zip(Cs, s)])
        costs.append(np.mean(Y.var(0)) / np.mean(Y.mean(0) ** 2))
    costs = np.array(costs)
    return zs[np.argmin(costs)], (zs, costs)


def structure_factor(C, static, ks):
    x = np.arange(C.shape[1])
    w = np.where(x == 0, 1.0, 2.0)
    S = (C * w) @ np.cos(np.outer(x, ks))
    S0 = (static * w) @ np.cos(np.outer(x, ks))
    return S / S0


def est_sk(C, static, t, w, ks=np.geomspace(0.04, 1.2, 25)):
    R = structure_factor(C, static, ks)
    tau = np.full(len(ks), np.nan)
    for i in range(len(ks)):
        r = R[:, i]
        j = np.argmax(r < np.exp(-1))
        if j > 0 and r[j] < np.exp(-1):
            a, b = np.log(t[j - 1]), np.log(t[j])
            fa, fb = r[j - 1] - np.exp(-1), r[j] - np.exp(-1)
            tau[i] = np.exp(a + (b - a) * fa / (fa - fb))
    m = (tau >= w[0]) & (tau <= w[1])
    if m.sum() < 3:
        return np.nan, (ks, tau)
    return -np.polyfit(np.log(ks[m]), np.log(tau[m]), 1)[0], (ks, tau)


def est_transfer(Vsum, Vcnt, t, w):
    V = Vsum.sum(0)[:-1] / Vcnt.sum(0)[:-1]
    return 1 / slope_fit(t, V, *w), V


def bulk(path, windows):
    d = np.load(path)
    n = int(d["N"]) / int(d["L"])
    t = d["lags"].astype(float)
    Csum, cnt = d["Csum"], d["cnt"]
    has_v = "Vsum" in d and len(d["Vsum"]) > 0
    S = Csum.shape[0]
    out = {"n": n, "samples": S, "transfer_samples": len(d["Vsum"]) if has_v else 0, "windows": {}}

    def all_estimates(idx, vidx, w):
        C, static = mean_C(Csum[idx], cnt[idx])
        r = {"x0": est_x0(C, t, w)[0], "return": est_return(C, t, w)[0],
             "collapse": est_collapse(C, t, w)[0], "sk": est_sk(C, static, t, w)[0]}
        if has_v:
            r["transfer"] = est_transfer(d["Vsum"][vidx], d["Vcnt"][vidx], t, w)[0]
        return r

    for name, w in windows.items():
        if w[1] > t[-1] * 1.0001:
            continue
        best = all_estimates(np.arange(S), np.arange(out["transfer_samples"]), w)
        boots = [all_estimates(RNG.integers(0, S, S),
                               RNG.integers(0, max(1, out["transfer_samples"]), max(1, out["transfer_samples"])), w)
                 for _ in range(NBOOT)]
        out["windows"][name] = {k: (float(v), float(np.nanstd([b[k] for b in boots]))) for k, v in best.items()}

    # curves for the figure (full data)
    C, static = mean_C(Csum, cnt)
    out["curves"] = {"t": t, "x0": est_x0(C, t, (1, 1e9))[1], "c0": C[:, 0]}
    out["collapse_cost"] = est_collapse(C, t, windows["main"])[1]
    out["sk"] = est_sk(C, static, t, windows["main"])[1]
    if has_v:
        out["curves"]["V"] = est_transfer(d["Vsum"], d["Vcnt"], t, (1, 1e9))[1]
    return out


# ---------------------------------------------------------- finite-size estimator
def finite_size(path, Lmin=16):
    d = np.load(path)
    L, te, ti = d["L"].astype(float), d["tau_e"], d["tau_int"]
    m = L >= Lmin
    res = {}
    for name, tau in (("efold", te), ("integrated", ti)):
        (z, c), cov = np.polyfit(np.log(L[m]), np.log(tau[m]), 1, cov=True)
        res[name] = (float(z), float(np.sqrt(cov[0, 0])))
    # largest sizes only (drift towards the asymptotic value)
    m2 = L >= np.sort(L)[-4]
    res["efold_large_L"] = (float(np.polyfit(np.log(L[m2]), np.log(te[m2]), 1)[0]), np.nan)
    loc = np.diff(np.log(te)) / np.diff(np.log(L))
    return {"n": float(d["n"]), "L": L, "tau_e": te, "tau_int": ti, "fits": res,
            "local": (np.sqrt(L[1:] * L[:-1]), loc)}


def main():
    os.makedirs("results", exist_ok=True)
    report = {"bulk": {}, "finite_size": {}}
    bulks, fss = [], []
    for p in sorted(q for q in glob.glob("results/corr_n*_L1000_t*.npz") if not q.endswith("_x0.npz")):
        tmax = float(np.load(p)["lags"][-1])
        windows = {"main": (1e3, tmax)}
        if tmax >= 1e7:
            windows.update({"early": (10, 1e5), "late": (10 ** 5.5, 1e7), "mid": (1e3, 1e6)})
        else:
            windows.update({"early": (10, 1e4), "late": (1e4, 1e6)})
        b = bulk(p, windows)
        bulks.append(b)
        report["bulk"][f"n={b['n']:g}"] = {"samples": b["samples"], "transfer_samples": b["transfer_samples"],
                                           "windows": b["windows"]}
        print(f"n={b['n']:g}", json.dumps(b["windows"], indent=None), flush=True)
    for p in sorted(glob.glob("results/finite_size_n*.npz")):
        f = finite_size(p)
        fss.append(f)
        report["finite_size"][f"n={f['n']:g}"] = {"L": f["L"].tolist(), "tau_efold": f["tau_e"].tolist(),
                                                  "tau_int": f["tau_int"].tolist(), "fits": f["fits"]}
        print(f"finite size n={f['n']:g}", f["fits"], flush=True)
    json.dump(report, open("results/methods_summary.json", "w"), indent=1)
    plot(bulks, fss)


def plot(bulks, fss):
    col = {1.0: "C3", 1.5: "C1", 2.0: "C0", 3.0: "C2"}
    fig, ax = plt.subplots(2, 3, figsize=(17, 10))
    ax = ax.ravel()
    for b in sorted(bulks, key=lambda b: b["n"]):
        c = col.get(b["n"])
        t = b["curves"]["t"]
        lab = f"n={b['n']:g}"
        ax[0].loglog(t, b["curves"]["x0"], "o", ms=3, c=c, label=lab)
        ax[1].loglog(t, b["curves"]["c0"], "o", ms=3, c=c, label=lab)
        zs, cost = b["collapse_cost"]
        ax[2].semilogy(zs, cost, c=c, label=lab)
        ks, tau = b["sk"]
        ax[3].loglog(ks, tau, "o-", ms=3, c=c, label=lab)
        if "V" in b["curves"]:
            ax[4].loglog(t, b["curves"]["V"], "o", ms=3, c=c, label=lab)
    for f in sorted(fss, key=lambda f: f["n"]):
        c = col.get(f["n"])
        ax[5].loglog(f["L"], f["tau_e"], "o-", c=c, label=f"n={f['n']:g}: z={f['fits']['efold'][0]:.2f}")
    titles = [("t", "x0(t), first zero of C", "1. first zero  x0 ~ t^(1/z)"),
              ("t", "C(0,t)", "2. return probability  C(0,t) ~ t^(-1/z)"),
              ("z", "collapse spread", "3. scaling collapse (main window)"),
              ("k", "tau(k)", "4. structure factor  tau(k) ~ k^(-z)"),
              ("t", "Var dN(t)", "5. charge transfer  Var dN ~ t^(1/z)"),
              ("L", "tau(L)", "6. quadrupole relaxation  tau ~ L^z")]
    for a, (xl, yl, ti) in zip(ax, titles):
        a.set_xlabel(xl)
        a.set_ylabel(yl)
        a.set_title(ti)
        a.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig("results/methods.png", dpi=120)


if __name__ == "__main__":
    main()
