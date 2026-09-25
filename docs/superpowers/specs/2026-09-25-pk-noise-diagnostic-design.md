# P(k) noise diagnostic — Design Spec

**Status:** approved design (2026-09-25). Diagnostic notebook; no CLASS change.

## Problem

In notebook 29, P_on/P_off at m_acc = 1e11 GeV shows ±0.5% jitter with sign flips at
k ≈ 0.05–0.5/Mpc, above the 1e-3 emulator target. The sink runs at 1e16 GeV are clean.

## Hypothesis

k-node mismatch: CLASS builds its k list from `k_min = k_min_tau0/τ0` (`perturbations.c:2101`)
with steps set by `2π/rs_rec`, and with C_l output `k_max_cmb ∝ 1/τ0` (`perturbations.c:2124`). The sink changes τ0 (H0 shifts with θ_s fixed), so on and off runs
sample different k nodes and their spline-interpolation errors do not cancel. Alternatives:
rk integration error (`tol_perturbations_integration`), source time sampling
(`perturbations_sampling_stepsize`), discrete daughter births.

## Notebook `notebooks_test/30_pk_noise_diagnostic.ipynb`

- m_acc = 1e11 GeV (η = 1), f_acc = 0.1, κ = 12.1, a_t = 0.133, FIX = 100θ_s, same outputs as
  nb29 (lensed C_l and P(k, z = 0)); only P(k) is analysed.
- **Metric:** r(k) = ln(P₁/P₂) minus a degree-6 polynomial fit in ln k over k ∈ [0.02, 2]/Mpc;
  report rms and max abs r.
- **Null pair:** sink off vs sink off with `k_step_sub = 0.0505` (default 0.05): same physics,
  nodes shifted by up to half a step. Jitter here is pure grid/interpolation noise.
- **Cases** (each an on/off pair with the same precision settings): default; `k_step_sub` 0.02,
  0.01; `k_per_decade_for_bao` 200; `tol_perturbations_integration` 1e-6, 1e-7;
  `perturbations_sampling_stepsize` 0.03; 101 daughter bins; `accdm_smooth_births = 1`; P(k) output only (no C_l).
- **Output:** table (case, rms, max, runtime of the pair), residual plot per case, and a line naming
  the cheapest case with max abs r < 1e-3.

Out of scope: changing defaults in CLASS or in the emulator config (a follow-up once the source is known).

## Results (2026-09-25)

Residual max abs r of ln(P₁/P₂), k ∈ [0.02, 2]/Mpc, m_acc = 1e11 GeV, θ_s fixed.

| pair | rms | max |
|---|---|---|
| null, ΛCDM (shifted k nodes) | 8.1e-7 | 6.2e-6 |
| null, accDM f_acc = 0.01 | 1.6e-4 | 7.8e-4 |
| null, accDM f_acc = 0.1 | 1.6e-3 | 6.8e-3 |
| null, accDM f_acc = 0.1, smooth births | 1.4e-3 | 4.9e-3 |
| on/off, default | 2.1e-3 | 7.4e-3 |
| on/off, `k_step_sub` 0.02 / 0.01 | 1.3e-3 / 1.2e-3 | 9.9e-3 / 5.6e-3 |
| on/off, `k_per_decade_for_bao` 200 | 2.2e-3 | 7.1e-3 |
| on/off, tol 1e-6 / sampling 0.03 | 2.1e-3 | 7.4e-3 |
| on/off, 101 bins / smooth births | 1.8e-3 | 7.2e-3 / 5.1e-3 |
| on/off, P(k) output only | 1.5e-3 | 5.0e-3 |
| on/off, tol 1e-7 | failed: rk step collapses at τ ≈ 4665 Mpc (a ≈ 0.11, birth peak) | |

Follow-up measurements (nb30 sampling test, nb31):

| test | result |
|---|---|
| null pair vs `k_step_sub` 0.05 / 0.02 / 0.01 / 0.005, 51 bins (max) | 6.8e-3 / 7.6e-3 / 3.1e-3 / 5.2e-3 |
| same, 201 bins | 6.8e-3 / 9.8e-3 / 3.2e-3 / 4.8e-3 |
| ln P jaggedness at exact nodes, k ∈ [0.100, 0.105], N_Q = 51 / 201 / 801 / 1601 | 1.4e-3 / 1.5e-3 / 1.5e-3 / 1.5e-3 |
| δ_daughter / δ_cdm today, same band | 1.6e-2 at every N_Q |
| k ∈ [0.1000, 0.1015], spacing 5e-5, N_Q = 201 | smooth sinusoid in δ_d and ln P, period 1.30e-3/Mpc, amplitude ±8e-4 in ln P |
| birth breakpoints (integration stops at every birth) | null pair unchanged (6.762e-3); tol 1e-7 now completes |

**Conclusion.** The "jitter" is a real, numerically converged oscillation of the daughter density
in k, aliased by CLASS's k sampling.

- Physics: every daughter starts with the same kick momentum, so the distance travelled by today
  has a maximum over birth times, D_max ≈ 4800 Mpc. The stationary phase leaves an undamped
  cos(k D_max) term in δ_d, which enters δ_m with the daughter's matter weight. It is independent of
  the number of daughter bins, the tolerance, time sampling and birth handling.
- Aliasing: the period 2π/D_max ≈ 1.3e-3/Mpc is finer than the k node spacing (~2e-3 in the
  `k_step_sub` range and ~10 nodes per decade above k_max_cmb ≈ 0.3/Mpc), so the spline turns it into
  apparent per-mode scatter that moves with the nodes. The null-pair maximum sits in the coarse
  range, which is why denser `k_step_sub` did not converge.
- Earlier readings in this spec were wrong: the scatter is interpolation of a real oscillation, not
  an integration error, and the birth switch-on is not its cause.
- Observability: a ~0.1% ripple (f_acc = 0.1, m_acc = 1e11 GeV) with period 1.3e-3/Mpc is below the
  k resolution of survey windows and averages out in band powers; a spread of kick momenta would
  damp it. Its amplitude grows with f_acc and depends on η.
- Treatment: smooth or band-average P(k) over Δk of a few × 1.3e-3/Mpc before emulator training,
  or evaluate observables that include the survey window. No CLASS change is needed for the ripple.

## What to feed the emulator (nb30, f_acc = 0.3, k ∈ [0.03, 1.5]/Mpc)

| test | max abs Δ ln P (null pair) | ΛCDM shape distortion |
|---|---|---|
| P_m raw | 1.8e-2 | - |
| P_m, Gaussian σ = 0.05 in ln k (kpd 10 / 30) | 6.4e-3 / 1.0e-2 | 1.2e-2 |
| P_m, Savitzky–Golay 0.15, cubic (kpd 10 / 30) | 1.3e-2 / 1.8e-2 | 7.4e-4 |
| **P_cb** | **1.4e-5** | - |

- Smoothing P_m does not work: the aliased error varies on the node scale, so a window narrow
  enough to keep BAO and the broadband shape cannot average it out; more nodes per decade do not help.
- P_cb (baryons + CDM + accDM parent; `delta_cb` is formed before the ncdm species are added) is
  clean to 1.4e-5, since the ripple lives only in the daughter's own δ. It is also the physical input
  for galaxy clustering, as for massive neutrinos.
- ln(P_m/P_cb) ≈ −0.43 (range −0.45 to −0.41) at f_acc = 0.3, so P_m carries the daughter's matter
  share and its slow clustering plus the ripple. Weak lensing needs P_m; a P_cb-based construction
  with a smooth daughter correction is not yet validated.

Recommended settings for scans and emulator data: daughter `ncdm_quadrature_strategy = 5` with 51
bins, `tol_perturbations_integration = 1e-5`, default k sampling, `evolver = 0`,
`ncdm_fluid_approximation = 3`; use `pk_cb` for clustering. Do not use `pk` (P_m) directly for
P(k)-based likelihoods or emulator targets.
