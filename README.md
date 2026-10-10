# Fracton_Thermalization
Code used for https://journals.aps.org/prb/abstract/10.1103/PhysRevB.107.045137
(arXiv:2210.02469).

* `Fracton_Thermalization.ipynb` – original notebook (D = D_LKS / D_sym, App. D algorithm).
* `fracton/counting.py` – fast exact counting of the symmetry sector and the largest Krylov sector.
* `fracton/dynamics.py` – random-circuit dynamics (U_{3,±} gates) and the two-point function C(x,t).
* `scripts/lks_fraction.py` – D vs n (notebook plot) and Fig. 4, now up to L = 201 in a few minutes.
* `scripts/run_correlation.py`, `scripts/analyze_correlation.py` – Sec. V.B: dynamical exponent.
* `tests/test_counting.py` – checks against brute-force Krylov-sector enumeration.
* `results/` – figures and data produced by the scripts.

Requirements: `numpy`, `numba`, `matplotlib` (all preinstalled on Colab).

```
python tests/test_counting.py
python scripts/lks_fraction.py --Lmax 201
python scripts/run_correlation.py --n 1 --L 1000 --tmax 1e7 --batches 12   # ~4 min / batch on 4 cores
python scripts/run_correlation.py --n 2 --L 1000 --tmax 1e6 --batches 8
python scripts/analyze_correlation.py results/corr_*_L1000_*[0-9].npz
```

## 1. Scaling the counting to larger systems

| | notebook | `fracton.counting` |
|---|---|---|
| D_sym | dense table `sym_sec(P,N,L)` of Python ints for **all** P ≤ P_max, N ≤ N_max, L ≤ L_max (~2·10⁸ objects for L = 101) | one q-binomial coefficient [N+L−1 choose N]_q at q^P, exact big ints, O(min(N,L)·P) time, O(P) memory |
| D_LKS | App. D recursion in pure Python; the inner `sum(... for k in range(...))` makes it O(N·L²·P) | same recursion with a running (prefix) sum over the last-particle position → O(N·L·P), pruned to states that can still reach dipole P, compiled with numba, rescaled each step so counts ≫ 10³⁰⁸ are fine |
| sweep over N | serial | `lks_fraction_sweep` runs N values in parallel processes |
| full notebook sweep, L = 11…101 | ~4 h (L = 101 alone: 122 min) | **~10 s** (L = 101: 1.8 s) |
| single point N = L = 401 / N = 801, L = 401 | – | 10 s / 37 s |

Memory now scales like L·P (≈ N L²/2 doubles), so L ≈ 400–500 runs on a laptop; the
time scales like N² L² ∝ L⁴ at fixed n.

**Bug fix (N < L, N + L even).** In this case the prefix-sum constraint (D5) built from the
fully extended state A = {…0,1,0,1,1,…,1,0,1,0…} is also satisfied by every state of the
Krylov sector of the centred block of N ones (which is itself a fully extended state, since
no U_{3,+} can act on it). The notebook's `LKS(N,L)` therefore returns the sum of the
**two** largest Krylov sectors, e.g. `LKS(7,9) = 143 = 84 + 59`. The new code subtracts
the block sector (counted with the same recursion) and agrees with brute-force
enumeration of all Krylov sectors (`tests/`). The notebook's sweep (odd L, odd N) lies
entirely in this case for n < 1, where its D values were too large by a factor 1.7–2.1.
D is exponentially small there, so the scaling of Eq. (5), Fig. 4 and n_c = 1 are
unaffected (Fig. 4b uses N = L, which was always correct).

`results/lks_fraction.png` reproduces the notebook plot and Fig. 4(a,b) up to L = 201
(D(n=1, L=201) = 0.101).

## 2. Dynamical exponent (Sec. V.B)

Protocol, as in Sec. V.B: N bosons on L sites with closed boundaries, U_{3,+} or U_{3,-}
chosen with probability 1/2 at a random 3-site window and skipped if invalid; L gates =
one time step. The initial state is drawn **uniformly from the symmetry sector**
(P = N(L−1)/2), the first zero x₀(t) of C(x,t) = ⟨(n_{x₀+x}(t₀+t)−n)(n_{x₀}(t₀)−n)⟩ is
measured, and x₀ ~ t^{1/z} is fitted.

What makes it feasible on 4 cores (≈ 50 min for n = 1 up to t = 10⁷):

* **Exact uniform sampling of the symmetry sector**: draw a uniform stars-and-bars
  string and accept it if its dipole moment is P. This is exact, needs no Markov-chain
  equilibration, and costs ~0.1 s for L = 1000.
* **Compiled gate kernel** with an inline xorshift generator (~10 ns per gate, 3× faster than
  `np.random` inside numba), with samples run in parallel (`prange`).
* **t₀ averaging by a multi-τ schedule**: uniform over the symmetry sector is stationary
  (the circuit is doubly stochastic), so for every lag t each run is cut into up to 1000
  back-to-back windows [kt, (k+1)t]. Short lags get thousands of t₀'s per run at no extra
  gate cost.
* **Large L at no statistical cost**: averaging over bulk x₀ (200 sites from each wall) makes the
  statistics per gate independent of L. L = 1000 keeps the finite-size offset of C small.
  Fixed N and P force Σ_x C = 0 over the whole system, giving C a tail of about −n(n+1)/L. That
  offset is measured from the equal-time C(x ≥ 1, 0) and subtracted. Fits without the
  subtraction are also listed in `results/dynamical_exponent.txt`.
* Errors are bootstrap errors over independent samples.

Results (`results/dynamical_exponent.png`, `.txt`), L = 1000:

| filling | samples | t range | z (fit windows) |
|---|---|---|---|
| **n = 1 = n_c** | 72 | 10 < t < 10⁵ | **5.18 ± 0.10** (paper: 5.2) |
| | | 10^5.5 < t < 10⁷ | **7.7 ± 1.2** (paper: 6.2) |
| | | 10³ < t < 10⁷ | **5.76 ± 0.21** (paper: z_c = 5.7 ± 0.5) |
| n = 1.5 | 64 | 10³ < t < 10⁶ | 4.80 ± 0.12 |
| n = 2 | 64 | 10³ < t < 10⁶ | 4.30 ± 0.13 |
| n = 3 | 64 | 10³ < t < 10⁶ | 4.15 ± 0.11 |

* At the critical filling we reproduce the paper's large exponent, z_c ≈ 5.3 at early times,
  drifting upward at late times. We also see the imperfect collapse of C(x,t)·x₀ against x/x₀
  that the paper reports: the peak C(0,t)·x₀ keeps growing with t. The late-time window is
  limited by statistics (only ~150 windows of length 10⁷). Fits without the offset
  subtraction are listed in `results/dynamical_exponent.txt`; they are about 0.3 higher.
* In the thermalizing phase z approaches the hydrodynamic value z = 4 for dipole-conserving
  systems. The approach is slower closer to n_c, consistent with a crossover once x₀(t)
  exceeds the correlation length ξ ≈ 2n_c(n_c+1)/(n−n_c)² of Sec. V.A (ξ ≈ 16 at n = 1.5,
  4 at n = 2, 1 at n = 3).
* To push further, add batches: `run_correlation.py` resumes from an existing `.npz` file and
  continues with fresh seeds.

## 3. Six estimators of z

`scripts/analyze_methods.py` compares six ways to get z (results in `results/methods.png`,
`results/methods_summary.json`):

1. first zero of C(x,t), x₀ ~ t^{1/z} (paper);
2. return probability C(0,t) ~ t^{−1/z};
3. scaling collapse C = t^{−1/z} F(x/t^{1/z});
4. dynamic structure factor, 1/e time τ(k) ~ k^{−z};
5. variance of the charge transferred across a bulk bond, Var ΔN ~ t^{1/z}
   (measured directly in `correlation_run_multitau`);
6. finite-size relaxation time of the quadrupole moment, τ(L) ~ L^z
   (`scripts/run_finite_size.py`, L = 12–64).

| n | x₀ | C(0,t) | collapse | S(k,t) | ΔN | τ(L) |
|---|---|---|---|---|---|---|
| 1 | 5.76 ± 0.19 | 8.00 ± 0.22 | 6.15 ± 0.26 | 5.83 ± 0.20 | 6.21 ± 0.07 | 6.16 ± 0.04 |
| 1.5 | 4.80 ± 0.13 | 4.44 ± 0.14 | 4.90 ± 0.12 | 4.54 ± 0.09 | 4.58 ± 0.04 | 5.00 ± 0.12 |
| 2 | 4.30 ± 0.11 | 4.16 ± 0.15 | 4.50 ± 0.12 | 4.37 ± 0.07 | 4.26 ± 0.03 | 4.52 ± 0.08 |
| 3 | 4.15 ± 0.11 | 4.20 ± 0.15 | 4.25 ± 0.10 | 4.07 ± 0.06 | 4.10 ± 0.03 | 4.43 ± 0.05 |

Bulk fits use 10³ ≤ t ≤ t_max; finite-size fits use L ≥ 16. Excluding the return probability,
which is dominated by slowly relaxing nearly frozen sites, z_c = 6.0 ± 0.3; the thermal phase
approaches z = 4.

```
python scripts/run_finite_size.py --n 1 --Ls 12 16 20 24 28 32 40 48 56 --samples 32 --runs 100
python scripts/analyze_methods.py
```
