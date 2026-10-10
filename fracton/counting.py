"""Fast exact counting of the symmetry sector and the largest Krylov sector (LKS).

Model (Pozderac et al., PRB 107, 045137 / arXiv:2210.02469): N bosons on L sites
with closed boundaries, evolved by charge- and dipole-conserving 3-site gates
U_{3,+-}. We work in the centred symmetry sector P = N(L-1)/2.

Compared with the original notebook implementation this module

* computes D_sym for a single (N, L, P) from the Gaussian binomial coefficient
  in O(min(N, L) * P) big-integer operations and O(P) memory, instead of
  filling a dense (P_max x N_max x L_max) table of Python ints (~2e8 entries
  for L = 101, which is the real memory bottleneck of the notebook);
* evaluates the App. D recursion for D_LKS with prefix sums along the "last
  particle" axis, removing one factor of L, prunes states that cannot reach
  the target dipole moment, and runs it as a compiled numba kernel with
  per-step rescaling so that counts far beyond the float64 range are fine;
* fixes the N < L, N + L even case. There the prefix-sum constraint of
  Eq. (D5) also admits the Krylov sector of the centred block of N ones
  (a second fully extended state), so the notebook's LKS() returns the sum of
  the two largest sectors. We subtract that sector, which is itself counted
  exactly by the same recursion (it is the N-team tournament sector).
  Verified against brute-force Krylov-sector enumeration (tests/).
"""
import math

import numpy as np
from numba import njit


def mins(N, L):
    """Prefix sums z(N, L, k), k = 0..N, of the sorted particle positions of the
    conjectured fully extended state in the LKS (identical to the notebook)."""
    if N < L:
        if L % 2 == 1 and N % 2 == 0:
            s = [i for i in range((L - N - 1) // 2, (L - 3) // 2 + 1)] + \
                [i for i in range((L + 1) // 2, (L + N - 1) // 2 + 1)]
        elif (L + N) % 2 == 0:
            if N == 1:
                s = [(L - 1) // 2]
            elif N == 2:
                s = [(L - 2) // 2, L // 2]
            else:
                s = [(L - N) // 2 - 1] + [i for i in range((L - N) // 2 + 1, (L + N) // 2 - 1)] + [(L + N) // 2]
        else:
            raise ValueError("N odd and L even: P = N(L-1)/2 is not an integer")
    else:
        if (L + N) % 2 == 0:
            s = [0 for i in range((N - L + 2) // 2)] + [i for i in range(1, L - 1)] + \
                [L - 1 for i in range((N - L + 2) // 2)]
        else:
            s = [0 for i in range((N - (L - 3)) // 2)] + [i + 1 for i in range((L - 3) // 2)] + \
                [i + (L + 1) // 2 for i in range((L - 3) // 2)] + [L - 1 for i in range((N - (L - 3)) // 2)]
    return [sum(s[:n]) for n in range(N + 1)]


def _block_prefix(N, L):
    """Prefix sums for the centred block of N ones (the second fully extended
    state that satisfies Eq. (D5) when N < L and N + L is even)."""
    c = (L - N) // 2
    return [k * c + k * (k - 1) // 2 for k in range(N + 1)]


@njit(cache=True)
def _log_count_prefix(N, L, z, P):
    """log of the number of sorted position lists 0 <= x_1 <= ... <= x_N <= L-1
    with sum_i x_i = P and prefix sums S_k >= z[k] for every k.

    f_k[y, S] counts lists of k particles with x_k = y and S_k = S, and
    f_k[y, S] = sum_{x <= y} f_{k-1}[x, S - y]: a running sum over rows."""
    W = P + 1
    prev = np.zeros((L, W))
    new = np.zeros((L, W))
    cum = np.zeros(W)
    plo = np.zeros(L, np.int64)
    phi = np.full(L, -1, np.int64)
    nlo = np.zeros(L, np.int64)
    nhi = np.full(L, -1, np.int64)
    for y in range(L):
        if y >= z[1] and y + (N - 1) * y <= P:
            prev[y, y] = 1.0
            plo[y] = y
            phi[y] = y
    logscale = 0.0
    for k in range(2, N + 1):
        top = 0
        for y in range(L):
            if phi[y] + 1 > top:
                top = phi[y] + 1
        cum[:top] = 0.0
        mx = 0.0
        for y in range(L):
            for S in range(plo[y], phi[y] + 1):
                cum[S] += prev[y, S]
            # clear what this row of `new` held two steps ago
            for S in range(nlo[y], nhi[y] + 1):
                new[y, S] = 0.0
            lo = max(z[k], y)
            hi = min(k * y, P - (N - k) * y)
            if hi - y >= top:
                hi = top - 1 + y
            nlo[y] = lo
            nhi[y] = hi
            for S in range(lo, hi + 1):
                v = cum[S - y]
                new[y, S] = v
                if v > mx:
                    mx = v
        if mx == 0.0:
            return -np.inf
        inv = 1.0 / mx
        for y in range(L):
            for S in range(nlo[y], nhi[y] + 1):
                new[y, S] *= inv
        logscale += np.log(mx)
        prev, new = new, prev
        plo, nlo = nlo, plo
        phi, nhi = nhi, phi
    tot = 0.0
    for y in range(L):
        if plo[y] <= P <= phi[y]:
            tot += prev[y, P]
    if tot == 0.0:
        return -np.inf
    return np.log(tot) + logscale


def _check(N, L):
    if N < 1 or L < 2:
        raise ValueError("need N >= 1 and L >= 2")
    if N % 2 == 1 and L % 2 == 0:
        raise ValueError("N odd and L even: P = N(L-1)/2 is not an integer")


def log_lks_size(N, L):
    """Natural log of D_LKS(N, L) at P = N(L-1)/2."""
    _check(N, L)
    P = N * (L - 1) // 2
    la = _log_count_prefix(N, L, np.array(mins(N, L), np.int64), P)
    if N < L and (N + L) % 2 == 0 and N >= 3:
        lb = _log_count_prefix(N, L, np.array(_block_prefix(N, L), np.int64), P)
        return la + math.log1p(-math.exp(lb - la))
    return la


def sym_size(N, L, P=None):
    """Exact D_sym(N, L, P): number of occupation strings with N bosons on L
    sites and dipole moment P, i.e. the q^P coefficient of the Gaussian
    binomial [N+L-1 choose N]_q. O(min(N, L) * P) big-int operations."""
    if P is None:
        _check(N, L)
        P = N * (L - 1) // 2
    if P < 0 or P > N * (L - 1):
        return 0
    c = np.zeros(P + 1, dtype=object)
    c[0] = 1
    m = min(N, L - 1)
    M = N + L - 1
    for i in range(1, m + 1):
        # multiply by (1 - q^a) / (1 - q^i); the partial product is always the
        # polynomial [M-m+i choose i]_q, so no cancellation blow-up occurs.
        a = M - m + i
        if a <= P:
            c[a:] = c[a:] - c[:P + 1 - a]
        for r in range(min(i, P + 1)):
            c[r::i] = np.cumsum(c[r::i])
    return int(c[P])


def lks_fraction(N, L):
    """D = D_LKS / D_sym at P = N(L-1)/2."""
    return math.exp(log_lks_size(N, L) - math.log(sym_size(N, L)))


def lks_size_exact(N, L):
    """Exact integer D_LKS via the same recursion with Python ints (small sizes)."""
    _check(N, L)
    P = N * (L - 1) // 2

    def count(z):
        f = {(y, y): 1 for y in range(L) if y >= z[1]}
        for k in range(2, N + 1):
            g = {}
            for (S, x), v in f.items():
                for y in range(x, L):
                    T = S + y
                    if T + (N - k) * y > P:
                        break
                    if T >= z[k]:
                        g[(T, y)] = g.get((T, y), 0) + v
            f = g
        return sum(v for (S, y), v in f.items() if S == P)

    out = count(mins(N, L))
    if N < L and (N + L) % 2 == 0 and N >= 3:
        out -= count(_block_prefix(N, L))
    return out


def _fraction_job(args):
    N, L = args
    return N, lks_fraction(N, L)


def lks_fraction_sweep(L, Ns, workers=None):
    """D(N, L) for several N in parallel processes. Returns (ns, Ds) arrays."""
    from concurrent.futures import ProcessPoolExecutor
    jobs = [(N, L) for N in Ns]
    # largest jobs first for better load balancing
    jobs.sort(key=lambda a: -a[0])
    with ProcessPoolExecutor(max_workers=workers) as ex:
        res = dict(ex.map(_fraction_job, jobs))
    Ns = sorted(res)
    return np.array(Ns) / L, np.array([res[N] for N in Ns])
