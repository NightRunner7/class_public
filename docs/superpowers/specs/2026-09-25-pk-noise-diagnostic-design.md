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
