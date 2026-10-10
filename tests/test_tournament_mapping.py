"""Checks of the "ceiling" picture of the LKS / tournament mapping.

Phi(x) = sum_{y<=x} sum_{z<=y} n_z is the dipole moment of the charge to the left of
bond (x, x+1), measured from that bond. Each U_{3,+-} gate at site y changes Phi only
at x = y-1, by +-1, and n >= 0 makes Phi convex. The LKS at N = L is the set of
states whose Phi lies below that of the uniform state, which is Landau's condition
for tournament score sequences. For N < L the region below the fully extended
state's Phi can hold more than one 0/1 state, i.e. more than one Krylov sector.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

from fracton.counting import mins  # noqa: E402
from test_counting import neighbours, sector  # noqa: E402


def phi(n):
    out, h, F = [], 0, 0
    for v in n:
        h += v
        F += h
        out.append(F)
    return out


def landau(n):
    s = 0
    xs = [x for x, v in enumerate(n) for _ in range(v)]
    for k, x in enumerate(xs, 1):
        s += x
        if s < k * (k - 1) // 2:
            return False
    return True


def component(start):
    seen, stack = {start}, [start]
    while stack:
        for t in neighbours(stack.pop()):
            if t not in seen:
                seen.add(t)
                stack.append(t)
    return seen


def below(s, ceiling):
    return all(a <= b for a, b in zip(phi(s), ceiling))


def test_gate_moves_phi_at_one_bond():
    n = (1, 0, 3, 1, 0, 2, 1)
    for y in range(1, len(n) - 1):
        for d in (1, -1):  # d = +1 is U_{3,+}
            m = list(n)
            m[y - 1] += d
            m[y + 1] += d
            m[y] -= 2 * d
            if min(m) >= 0:
                diff = [a - b for a, b in zip(phi(m), phi(n))]
                assert diff[y - 1] == d and sum(map(abs, diff)) == 1


def test_lks_is_below_uniform_ceiling_and_landau():
    for L in range(3, 10):
        P = L * (L - 1) // 2
        lks = component((1,) * L)
        ceiling = phi((1,) * L)
        states = sector(L, L, P)
        assert lks == {s for s in states if below(s, ceiling)}
        assert lks == {s for s in states if landau(s)}


def test_two_terminal_states_when_n_less_than_l_same_parity():
    for L in range(5, 11):
        for N in range(3, L):
            if (N + L) % 2:
                continue
            z = mins(N, L)
            A = [0] * L
            for k in range(1, N + 1):
                A[z[k] - z[k - 1]] += 1
            states = sector(N, L, N * (L - 1) // 2)
            region = {s for s in states if below(s, phi(A))}
            terminals = [s for s in region if max(s) <= 1]
            assert len(terminals) == 2
            assert region == set().union(*(component(t) for t in terminals))


if __name__ == "__main__":
    test_gate_moves_phi_at_one_bond()
    test_lks_is_below_uniform_ceiling_and_landau()
    test_two_terminal_states_when_n_less_than_l_same_parity()
    print("all tests passed")
