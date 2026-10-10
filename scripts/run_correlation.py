"""Simulate C(x, t) (Eq. 8 of arXiv:2210.02469) for the l = 3 model.

Results are accumulated batch by batch into an .npz file, so a run can be
interrupted and resumed (new batches use new seeds).

Example (critical point, Sec. V.B / Fig. 6):
    python scripts/run_correlation.py --n 1 --L 1000 --tmax 1e7 --batches 12
"""
import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from fracton.dynamics import correlation_run_multitau, log_times, multitau_schedule  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=float, required=True, help="filling N/L")
    ap.add_argument("--L", type=int, default=1000)
    ap.add_argument("--tmax", type=float, default=1e6, help="largest lag t")
    ap.add_argument("--ttotal", type=float, default=None, help="run length per sample (default 2 tmax)")
    ap.add_argument("--per-decade", type=int, default=10)
    ap.add_argument("--kmax", type=int, default=1000, help="max t0 windows per lag and sample")
    ap.add_argument("--xmax", type=int, default=150)
    ap.add_argument("--margin", type=int, default=None, help="bulk margin (default L/5)")
    ap.add_argument("--batch", type=int, default=4, help="samples per batch (one per core)")
    ap.add_argument("--batches", type=int, default=1)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    L = a.L
    N = int(round(a.n * L))
    if (N * (L - 1)) % 2:
        L += 1
        N = int(round(a.n * L))
    tmax = int(a.tmax)
    T = int(a.ttotal) if a.ttotal else 2 * tmax
    margin = a.margin if a.margin is not None else L // 5
    assert margin >= a.xmax
    lags = log_times(tmax, a.per_decade)
    ev_t, ev_p, ev_i = multitau_schedule(lags, T, a.kmax)
    out = a.out or os.path.join("results", f"corr_n{a.n:g}_L{L}_t{tmax:.0e}.npz")

    if os.path.exists(out):
        d = np.load(out)
        assert d["N"] == N and d["L"] == L and np.array_equal(d["lags"], lags)
        Csum, cnt, nsamp = list(d["Csum"]), list(d["cnt"]), int(d["nsamples"])
    else:
        Csum, cnt, nsamp = [], [], 0

    print(f"N={N} L={L} n={N/L:.4f} lags 1..{tmax} T={T} margin={margin} -> {out}", flush=True)
    for b in range(a.batches):
        t0 = time.time()
        seed = 1 + nsamp
        Cs, cn = correlation_run_multitau(N, L, len(lags), ev_t, ev_p, ev_i, a.xmax, margin, a.batch, seed)
        Csum += list(Cs)
        cnt += list(cn)
        nsamp += a.batch
        np.savez(out, Csum=np.array(Csum), cnt=np.array(cnt), lags=lags, N=N, L=L,
                 nsamples=nsamp, margin=margin, T=T)
        print(f"batch {b+1}/{a.batches}: {nsamp} samples, {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
