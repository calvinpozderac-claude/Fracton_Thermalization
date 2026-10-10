"""Checks of fracton.counting against brute-force Krylov-sector enumeration.

Run with:  python -m pytest tests/   (or python tests/test_counting.py)
"""
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fracton.counting import log_lks_size, lks_size_exact, sym_size  # noqa: E402


def sector(N, L, P):
    out = []

    def rec(i, left, p, cur):
        if i == L:
            if left == 0 and p == P:
                out.append(tuple(cur))
            return
        for k in range(left + 1):
            if p + k * i <= P:
                rec(i + 1, left - k, p + k * i, cur + [k])
    rec(0, N, 0, [])
    return out


def neighbours(c):
    L = len(c)
    for i in range(1, L - 1):
        for d in (1, -1):
            n = list(c)
            n[i - 1] += d
            n[i + 1] += d
            n[i] -= 2 * d
            if min(n) >= 0:
                yield tuple(n)


def krylov_sizes(N, L, P):
    seen, sizes = set(), []
    for s in sector(N, L, P):
        if s in seen:
            continue
        seen.add(s)
        stack, size = [s], 1
        while stack:
            for t in neighbours(stack.pop()):
                if t not in seen:
                    seen.add(t)
                    stack.append(t)
                    size += 1
        sizes.append(size)
    return sizes


def cases(Lmax, Nmax_factor=2):
    for L in range(3, Lmax + 1):
        for N in range(1, Nmax_factor * L):
            if not (N % 2 == 1 and L % 2 == 0):
                yield N, L


def test_largest_krylov_sector():
    # full enumeration: the counted sector is the largest one
    for N, L in cases(10):
        if sym_size(N, L) > 3e5:
            continue
        P = N * (L - 1) // 2
        sizes = krylov_sizes(N, L, P)
        assert sum(sizes) == sym_size(N, L)
        assert lks_size_exact(N, L) == max(sizes), (N, L)
        assert math.isclose(math.exp(log_lks_size(N, L)), max(sizes), rel_tol=1e-9)


def test_float_kernel_matches_exact():
    for N, L in cases(16):
        if N > 20:
            continue
        e = lks_size_exact(N, L)
        assert math.isclose(math.exp(log_lks_size(N, L)), e, rel_tol=1e-9), (N, L)


def test_sym_size_small_tables():
    # direct recurrence over sites
    from functools import lru_cache

    @lru_cache(None)
    def c(N, L, P):
        if L == 0:
            return int(N == 0 and P == 0)
        return sum(c(N - k, L - 1, P - k * (L - 1)) for k in range(N + 1) if P - k * (L - 1) >= 0)

    for L in range(1, 9):
        for N in range(0, 10):
            for P in range(0, N * (L - 1) + 1):
                assert sym_size(N, L, P) == c(N, L, P)


if __name__ == "__main__":
    test_sym_size_small_tables()
    test_float_kernel_matches_exact()
    test_largest_krylov_sector()
    print("all tests passed")
