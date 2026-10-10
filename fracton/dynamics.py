"""Random-circuit dynamics of the l = 3 dipole-conserving model (Sec. II / V.B).

Each gate picks a random set of 3 contiguous sites (x-1, x, x+1) with closed
boundaries and one of U_{3,+} : (n_{x-1}, n_x, n_{x+1}) -> (+1, -2, +1) or
U_{3,-} : (-1, +2, -1) with probability 1/2; it is skipped if an occupation
would become negative. One unit of time = L gates.

Since both the gate positions and the +/- choice are drawn symmetrically, the
Markov chain is doubly stochastic, so the uniform distribution over the
symmetry sector (a size-weighted mixture of uniform distributions over Krylov
sectors) is stationary. Starting from a uniformly random state of the
symmetry sector, as in the paper, the dynamics is therefore in a stationary
ensemble and averaging over t0 is equivalent to averaging over samples.

Two-point function, Eq. (8):
    C(x, t) = < (n_{x0+x}(t0 + t) - n) (n_{x0}(t0) - n) >,
averaged over bulk positions x0 (to stay away from the closed boundaries),
over +x and -x, and over independent samples.
"""
import numpy as np
from numba import njit, prange


@njit(cache=True)
def seed_numba(seed):
    np.random.seed(seed)


@njit(cache=True)
def random_sector_state(N, L, P):
    """Exact uniform sample from the symmetry sector {sum n = N, sum x n_x = P}.

    A uniformly random "stars and bars" string (N particles, L-1 walls) is a
    uniform sample of all N-particle states; accepting it only if its dipole
    moment equals P gives an exact uniform sample of the symmetry sector.
    The acceptance rate is ~ 1/(sqrt(2 pi) sigma_P) = O(L^{-3/2}) at fixed
    density, so this costs O(L^{5/2}) operations per sample."""
    M = N + L - 1
    slots = np.zeros(M, np.int8)
    slots[:N] = 1
    n = np.zeros(L, np.int64)
    while True:
        # Fisher-Yates shuffle of the stars-and-bars string
        for i in range(M - 1, 0, -1):
            j = np.random.randint(0, i + 1)
            slots[i], slots[j] = slots[j], slots[i]
        x = 0
        p = 0
        for i in range(M):
            if slots[i] == 1:
                p += x
            else:
                x += 1
        if p == P:
            n[:] = 0
            x = 0
            for i in range(M):
                if slots[i] == 1:
                    n[x] += 1
                else:
                    x += 1
            return n


@njit(cache=True)
def _splitmix64(x):
    x = (x + np.uint64(0x9E3779B97F4A7C15))
    z = x
    z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return z ^ (z >> np.uint64(31))


@njit(cache=True)
def apply_gates(n, ngates, rng):
    """Apply `ngates` random U_{3,+-} gates in place (L gates = 1 time unit).

    rng is a length-1 uint64 array holding a xorshift64* state. Each 64-bit
    draw is split into two 32-bit numbers mapped to [0, 2(L-2)) by a
    multiply-shift (relative bias < 2(L-2)/2^32), which is several times faster
    than numba's np.random.randint."""
    L = n.shape[0]
    m = np.uint64(2 * (L - 2))
    s = rng[0]
    mask = np.uint64(0xFFFFFFFF)
    sh = np.uint64(32)
    for g in range(ngates):
        if g & 1 == 0:
            s ^= s >> np.uint64(12)
            s ^= s << np.uint64(25)
            s ^= s >> np.uint64(27)
            w = s * np.uint64(0x2545F4914F6CDD1D)
            r = ((w & mask) * m) >> sh
        else:
            r = ((w >> sh) * m) >> sh
        x = 1 + np.int64(r >> np.uint64(1))
        if r & np.uint64(1):
            # U_{3,+}: two particles at x hop apart
            if n[x] >= 2:
                n[x] -= 2
                n[x - 1] += 1
                n[x + 1] += 1
        else:
            # U_{3,-}: particles at x-1 and x+1 hop together onto x
            if n[x - 1] >= 1 and n[x + 1] >= 1:
                n[x] += 2
                n[x - 1] -= 1
                n[x + 1] -= 1
    rng[0] = s


def make_rng(seed):
    return np.array([_splitmix64(np.uint64(seed)) | np.uint64(1)], dtype=np.uint64)


@njit(cache=True)
def _accumulate(C, n, ref, dens, xs_lo, xs_hi, xmax):
    """C[x] += sum_{x0 in [xs_lo, xs_hi)} (dn(x0+x) dn0(x0) + dn(x0-x) dn0(x0)) / 2."""
    for x in range(xmax + 1):
        acc = 0.0
        for x0 in range(xs_lo, xs_hi):
            d0 = ref[x0] - dens
            acc += d0 * ((n[x0 + x] - dens) + (n[x0 - x] - dens))
        C[x] += 0.5 * acc / (xs_hi - xs_lo)


@njit(cache=True)
def _transfer(n, ref, c_lo, c_hi):
    """Mean over bulk bonds c of (Delta N_c)^2, where Delta N_c = sum_{x >= c} (n_x - ref_x)
    is the charge that crossed the bond (c-1, c) in the window."""
    L = n.shape[0]
    acc = 0.0
    d = 0.0
    for x in range(L - 1, c_lo - 1, -1):
        d += n[x] - ref[x]
        if x < c_hi:
            acc += d * d
    return acc / (c_hi - c_lo)


@njit(parallel=True, cache=True)
def correlation_run(N, L, times, xmax, margin, nsamples, seed):
    """Average C(x, t) over `nsamples` independent circuits, in parallel.

    times : increasing int64 array of output times (in units of L gates)
    returns C[t_index, x] for x = 0..xmax, and per-sample data for error bars.
    """
    P = N * (L - 1) // 2
    dens = N / L
    nt = times.shape[0]
    Cs = np.zeros((nsamples, nt, xmax + 1))
    for s in prange(nsamples):
        np.random.seed(seed + 7919 * s)
        rng = np.empty(1, np.uint64)
        rng[0] = _splitmix64(np.uint64(seed * 1000003 + s)) | np.uint64(1)
        n = random_sector_state(N, L, P)
        ref = n.copy()
        t_now = 0
        for it in range(nt):
            apply_gates(n, (times[it] - t_now) * L, rng)
            t_now = times[it]
            _accumulate(Cs[s, it], n, ref, dens, margin, L - margin, xmax)
    return Cs


def multitau_schedule(lags, T_total, kmax):
    """Event schedule for t0-averaging. For lag t_j we use the back-to-back
    windows [k t_j, (k+1) t_j], k = 0..K_j-1 with K_j = min(kmax, T_total // t_j).
    Returns sorted event times and, in CSR form, the lag indices due at each."""
    ev = {}
    for j, t in enumerate(lags):
        for k in range(1, min(kmax, T_total // t) + 1):
            ev.setdefault(k * t, []).append(j)
    times = np.array(sorted(ev), np.int64)
    ptr = np.zeros(len(times) + 1, np.int64)
    idx = []
    for a, t in enumerate(times):
        idx += ev[t]
        ptr[a + 1] = len(idx)
    return times, ptr, np.array(idx, np.int64)


@njit(parallel=True, cache=True)
def correlation_run_multitau(N, L, nlags, ev_times, ev_ptr, ev_idx, xmax, margin, nsamples, seed):
    """C(x, t) of Eq. (8) averaged over x0 in the bulk and over t0 (multi-tau).

    Returns per-sample sums Csum[s, j, x], window counts cnt[s, j] and the summed
    charge-transfer variance Vsum[s, j] (see _transfer); the last lag slot
    nlags holds the equal-time correlation C(x, 0)."""
    P = N * (L - 1) // 2
    dens = N / L
    Csum = np.zeros((nsamples, nlags + 1, xmax + 1))
    cnt = np.zeros((nsamples, nlags + 1))
    Vsum = np.zeros((nsamples, nlags + 1))
    for s in prange(nsamples):
        np.random.seed(seed + 7919 * s)
        rng = np.empty(1, np.uint64)
        rng[0] = _splitmix64(np.uint64(seed * 1000003 + s)) | np.uint64(1)
        n = random_sector_state(N, L, P)
        refs = np.empty((nlags, L), np.int64)
        for j in range(nlags):
            refs[j] = n
        _accumulate(Csum[s, nlags], n, n, dens, margin, L - margin, xmax)
        cnt[s, nlags] += 1
        t_now = 0
        for a in range(ev_times.shape[0]):
            apply_gates(n, (ev_times[a] - t_now) * L, rng)
            t_now = ev_times[a]
            for b in range(ev_ptr[a], ev_ptr[a + 1]):
                j = ev_idx[b]
                _accumulate(Csum[s, j], n, refs[j], dens, margin, L - margin, xmax)
                Vsum[s, j] += _transfer(n, refs[j], margin, L - margin)
                cnt[s, j] += 1
                refs[j] = n
    return Csum, cnt, Vsum


@njit(parallel=True, cache=True)
def quadrupole_series(N, L, dt, nsteps, burn, nsamples, seed):
    """Equilibrium time series of the quadrupole moment Q = sum_x (x - xc)^2 n_x,
    the slowest mode of a closed chain with fixed N and P. Starts from a uniform
    symmetry-sector state (stationary), records Q every dt time steps after
    `burn` steps. Returns Q[s, k]."""
    P = N * (L - 1) // 2
    xc = (L - 1) / 2.0
    Q = np.zeros((nsamples, nsteps))
    for s in prange(nsamples):
        np.random.seed(seed + 7919 * s)
        rng = np.empty(1, np.uint64)
        rng[0] = _splitmix64(np.uint64(seed * 1000003 + s)) | np.uint64(1)
        n = random_sector_state(N, L, P)
        apply_gates(n, burn * L, rng)
        for k in range(nsteps):
            apply_gates(n, dt * L, rng)
            q = 0.0
            for x in range(L):
                q += (x - xc) * (x - xc) * n[x]
            Q[s, k] = q
    return Q


def log_times(tmax, per_decade=10, tmin=1):
    t = np.unique(np.round(np.logspace(np.log10(tmin), np.log10(tmax),
                                       int(per_decade * np.log10(tmax / tmin)) + 1)).astype(np.int64))
    return t[t >= 1]


def first_zero(C):
    """Linearly interpolated position of the first sign change of C(x) from + to -."""
    for x in range(1, len(C)):
        if C[x] <= 0 < C[x - 1]:
            return x - 1 + C[x - 1] / (C[x - 1] - C[x])
    return np.nan


def fit_z(t, x0, tmin, tmax):
    """Fit x0 ~ t^{1/z} on tmin <= t <= tmax; returns z and its fit uncertainty."""
    m = (t >= tmin) & (t <= tmax) & np.isfinite(x0)
    (slope, icpt), cov = np.polyfit(np.log(t[m]), np.log(x0[m]), 1, cov=True)
    z = 1 / slope
    return z, np.sqrt(cov[0, 0]) / slope ** 2, icpt
