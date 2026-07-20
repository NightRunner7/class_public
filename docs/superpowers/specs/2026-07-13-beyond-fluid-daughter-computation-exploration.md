# Beyond the fluid: less-approximate daughter computations (exploration note)

**Status: exploration only — no design approved, nothing scheduled for implementation.**
Written 2026-07-13 after the term-by-term audit of our implementation against
Abellán et al. 2021 (arXiv:2102.12498) found no transcription errors: the fluid's
accuracy ceiling is the *closure*, not bugs.

## Why we can do better than the paper's fluid

Abellán et al. close the hierarchy with an ansatz `c_s²(k,τ) = c_g²[1 + (1−2ε)·0.2√(k/k_fs)]`
because their daughter has a broad, continuously-sourced momentum distribution —
no better information exists. Our model has structure they don't:

1. **Monochromatic injection.** Every daughter is born with the same physical
   momentum `P_acc`, so comoving momentum ↔ birth time is a bijection:
   `a_q = q·T_acc/P_acc`. A q-bin is a *birth cohort*.
2. **Instantaneous per-cohort production.** After `a = a_q` a cohort evolves
   collisionlessly (this is what the pinning-IC scheme already exploits).
3. **Deterministic cooling.** A cohort's velocity history
   `v(q,τ) = q/ε(q,τ)` is a pure background quantity.

The effect that defeats every local-in-time closure — free-streaming phase
mixing, i.e. superpositions of oscillations `~ j_ℓ(k∫v dτ)` with different
phases across cohorts (identified in the param scans as the architectural
blocker; see memory `accdm-fluid-f-boundary`) — is therefore *analytically
solvable* per cohort. Calibrated ceff2 (plateau mode 2) is near the ceiling of
what any closure can do; the options below change the representation instead.

## Option A — Exact cohort response integrals (Gilbert / linear response)

Each cohort's free hierarchy has the closed-form line-of-sight solution: birth
IC propagated by spherical-Bessel kernels of `kΔχ`, plus a metric-source
integral, with `Δχ(τ_q,τ) = ∫_{τ_q}^{τ} v dτ′` background-only. Schematically

```
Ψ_ℓ(q,τ) = Σ_{ℓ'} Ψ_{ℓ'}^birth(τ_q) · K_{ℓℓ'}(kΔχ(τ_q,τ))
         + ∫_{τ_q}^{τ} dτ′ (dlnf0/dlnq)·[ḣ,η̇ kernels](τ′) · K_ℓ(kΔχ(τ′,τ))
```

The birth ICs are exactly our pinning values (δ_dcdm − ḣ/(6aH) at ℓ=0, the
(ḣ+6η̇)/(15aH) shear at ℓ=2, 0 at ℓ=1). Only ℓ=0,1,2 moments feed Einstein's
equations, so δρ, δP, (ρ+p)θ, σ of the daughter become double quadratures
(birth time × source time) — a response functional of δ_dcdm(τ), ḣ(τ), η̇(τ).
**No ℓ-truncation, no closure, no per-bin ODEs.**

Feedback into the potentials makes it integro-differential; at f ≲ 0.1 (MCMC
posterior) with free-streaming suppression on top, a Born scheme should
converge in one iteration: pass 1 with the cheap fluid (or no daughter
perturbations) for potentials, pass 2 quadrature for the daughter, optional
re-iterate. Prior art: Gilbert's equation; Ali-Haïmoud & Bird 2013
(linear-response neutrinos in N-body); free-streaming DR solutions.

- Pros: exact; phase mixing lives inside Bessel functions where it belongs;
  likely the fastest accurate scheme for small f; publishable method.
- Cons: does not fit CLASS's ODE architecture (two-pass or post-processing
  pipeline); careful numerics for large kΔχ (Levin/FFTLog); self-consistency
  at f ~ 0.3 needs iteration whose convergence must be measured.
- First validation: python mirror (like `saturating_cfs`) against nb6/nb16
  exact runs; measure the Born-iteration error vs f.

## Option B — Cold-core + warm-skin hybrid

At any time only cohorts born within roughly the last Hubble time are
dynamically warm; older cohorts have `v ∝ a_q/a → 0`. So:

- **Warm skin:** exact per-q hierarchy only for cohorts with `v > v_split` —
  a narrow moving window near the production front (roughly constant width in
  ln q), instead of the ever-growing full grid.
- **Cold core:** when a cohort crosses `v_split`, hand its (δ, θ) off into a
  single accumulated pressureless fluid obeying CDM-like equations plus a
  hand-off source term — structurally the dcdm-daughter equations, i.e.
  CLASS-native ODEs.

This is not a closure: the pressureless limit is exact as `v_split → 0`, so
`v_split` is a convergence knob. The neglected residual pressure perturbation
of retired cohorts is O((v/v_split)²); the *background* pressure of the core
can be kept exactly (cohort momenta are known). The moving window can be faked
inside CLASS's fixed-size state vector by the same freeze/slave mechanism the
pre-production pinning uses, applied at the cold end (retired bins zeroed,
their content accumulated into 2 extra fluid variables).

- Pros: stays inside CLASS; systematically improvable; replaces both the fluid
  *and* the expensive full hierarchy at late times (the current cost center);
  natural validation target for Option A.
- Cons: bookkeeping (hand-off ICs, window management); interaction with the
  ncdmfa machinery and with the a==aq border conventions (see
  `abellan2021-audit` issue 4) must be nailed down; speedup depends on how
  narrow the skin can be at target accuracy — measure first from exact runs
  (fraction of Σ|dΨ| carried by bins with v > v_split, as a function of a).

## Option C — Closure upgrades (listed for completeness; not recommended)

Evolve Π (CS2DYN, already implemented) and/or dynamical shear with ω_p, ω_θ
*measured* for this model from exact-run diagnostics (they are just
higher-weight integrals; with monochromatic injection they may be
near-universal in k/k_fs and trigger age). Both our CS2DYN experience and the
paper's own warning about their Eq. 43 instability argue this path caps out;
phase mixing is not representable by any local c_s². The one cheap useful
piece: outputting ω_p(k,τ), ω_θ(k,τ) from exact runs would quantitatively
explain the mode-2 plateau's origin.

## Bottom line

Better computations exist because the model's injection is monochromatic and
per-cohort evolution is free — structure the paper's generic closure cannot
use. If this ever moves forward: B first for in-CLASS validity where the fluid
fails, A as the small-f fast pipeline and the more elegant result; C only as a
diagnostic. Next concrete step (cheap, no code changes): from existing exact
runs, measure (i) the warm-skin width needed for 1% P(k) (feeds B) and (ii)
the daughter back-reaction size vs f (feeds A's Born-convergence question).
