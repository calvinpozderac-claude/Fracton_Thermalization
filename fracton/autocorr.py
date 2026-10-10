"""Autocorrelation helpers for finite-size relaxation times."""
import numpy as np


def autocorr(Q):
    """Normalised autocorrelation A(k) of time series Q[s, t], averaged over
    samples s. Each sample's own time average is subtracted (removes the
    Krylov-sector-dependent mean at n <= n_c). FFT-based, O(T log T)."""
    S, T = Q.shape
    d = Q - Q.mean(1, keepdims=True)
    f = np.fft.rfft(d, 2 * T, axis=1)
    ac = np.fft.irfft(f * np.conj(f), axis=1)[:, :T]
    ac = ac.sum(0) / (T - np.arange(T))
    return ac / ac[0]


def efold_time(A, dt):
    """First time A drops below 1/e (linear interpolation)."""
    k = np.argmax(A < np.exp(-1))
    if k == 0:
        return np.nan
    a0, a1 = A[k - 1], A[k]
    return dt * (k - 1 + (a0 - np.exp(-1)) / (a0 - a1))


def integrated_time(A, dt, c=5.0):
    """Integrated autocorrelation time with Sokal's self-consistent window."""
    tau = 0.5
    for w in range(1, len(A)):
        tau = 0.5 + A[1:w + 1].sum()
        if w >= c * tau:
            break
    return dt * tau
