# DE sink observables notebook — Design Spec

**Status:** approved design (2026-09-25). Diagnostic notebook; no code change in CLASS.

## Goal

Show what `acc_de_sink` changes: background energy densities, lensed CMB spectra and the linear
matter power spectrum, for a mass scan, with the sink off and on and against ΛCDM.

## Notebook

`notebooks_test/29_de_sink_observables.ipynb`, self-contained, runs in a few minutes in memory.

**Setup.** Planck 2018 base. Toggle `FIX = '100*theta_s'` (1.041783, default) or `'H0'` (67.32).
accDM: `ncdm_quadrature_strategy = '0, 5'`, 51 daughter bins, rk evolver, synchronous gauge,
`ncdm_fluid_approximation = 3`, ω_cdm rescaled to match matter at recombination (as nb21/nb28).
Output: lensed C_ℓ to ℓ = 2500, P(k, z = 0) to k = 5/Mpc.

**Scan.** `MASSES = [1e11, 1e12, 1e13, 1e16]` GeV, η = 1e11/m, f_acc = 0.1, κ = 12.1, a_t = 0.133.
Runs: ΛCDM (same neutrino sector) and, per mass, sink off and on. A failed run is reported and
skipped; figures use the masses with both runs.

**Figure 1 — background** (2 × n_masses, sink on): ρ_i/ρ_crit,0 vs a for CDM, parent, daughter,
Λ, DE sink; below, H_on/H_off − 1.

**Figure 2 — CMB** (4 × 3): rows TT, EE, TE, φφ; columns off/ΛCDM, on/ΛCDM, on/off. Relative
differences, TE as ΔC_ℓ/√(C_ℓ^TT C_ℓ^EE). Cosmic-variance band √(2/(2ℓ+1)) on TT and EE.

**Figure 3 — P(k)** (1 × 3): P_off/P_ΛCDM − 1, P_on/P_ΛCDM − 1, P_on/P_off − 1.

**Summary table.** Per mass: η, max Ω_de_acc, the free parameter (H0 or 100θ_s) off/on, σ8 off/on,
max abs ΔC_ℓ^TT/C_ℓ on vs off.

**Sanity check.** The on run has `(.)rho_de_acc` and the off run does not; at the heaviest mass
(η = 1e-5) on/off differ by < 1e-3 in TT and P(k). The bound is 1e-3, not 1e-4, because θ_s
shooting has its own tolerance.

Out of scope: likelihoods / Δχ², option B.
