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

- The jitter is a per-k-mode error, not interpolation: denser k sampling does not remove it, and
  shifting nodes with identical physics reproduces it.
- It lives in the daughter sector and scales with f_acc; CLASS alone is at 1e-6.
- It is insensitive to the integration tolerance and time sampling, and slightly reduced by smooth
  births; tightening the tolerance makes rk fail at the birth peak. This points to the discontinuous
  birth switch-on (bins pinned to the parent via writes into y inside derivs, then released) but
  does not isolate it from other daughter-sector per-mode errors.
- Consequence: any accDM run at f_acc = 0.1 carries ~0.16% rms, ~0.7% max per-mode P(k) error,
  above the 1e-3 emulator target for f_acc ≳ 0.01. On/off comparisons cancel it only when both runs
  take identical steps (e.g. m_acc = 1e16 in nb29).
- Candidate fix (separate spec): stop the integrator at each bin's birth time and start the bin from
  the parent state there; test with the nb30 null pair (target max < 1e-3 at f_acc = 0.1).
