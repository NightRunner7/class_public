# Making the `rk` evolver viable for the accDM daughter fluid approximation

**Date:** 2026-06-22
**Status:** Design approved; ready for implementation planning.
**Scope:** `source/perturbations.c`, `include/perturbations.h`, `include/precisions.h`,
`notebooks_test/6_test_fluid_vs_exact.ipynb`.

## Problem

The accDM daughter (last ncdm species, `n_ncdm == N_ncdm-1` when `has_acc`) is produced
from dcdm decay. In the **fluid approximation** its continuity and Euler equations carry
a decay source that relaxes the daughter toward the parent:

- continuity: `δ_acc → δ_dcdm`
- Euler:      `θ_acc → ¾ θ_dcdm`
- shear:      `σ_acc → 0`

with relaxation rate

```
Λ = a·Γ·(1+η)·((1+ca2)/(1+w))·ratio_rho,   ratio_rho = rho_acc_cdm / rho_ncdm.
```

At fluid switch-on the daughter has just begun to be produced, so `rho_ncdm` is tiny,
`ratio_rho ~ 1/trigger` is enormous, and `Λ` vastly exceeds the dynamical/oscillation
rates `aH` and `k·c_eff`. The explicit `rk` (RKCK45) stability region cannot accommodate
that large negative eigenvalue: it drives the adaptive step to ~1e-13 and the run dies
with "step too small". `ndf15` (implicit, would absorb the stiffness) segfaults for
accDM, so it is not an option. The **exact Boltzmann hierarchy is stable** through the
same early phase but is expensive (many q-bins × l-modes), so it gives no speedup.

The switch-on IC conversion (`perturbations.c:5077`) projects the exact hierarchy moments
onto the fluid `δ/θ/σ`; it does **not** place the daughter on the fluid attractor.

## Goal

Make `rk` + the daughter fluid path **stable AND faster than the exact hierarchy, while
matching it within tolerance**. This requires removing the stiff eigenvalue, not merely
surviving it.

## Key insight

While `Λ` is large the daughter is **tightly coupled to the parent dcdm** through the decay
term — structurally identical to baryon–photon tight coupling (TCA) in CLASS. The proven
cure is the TCA pattern: remove the stiff variable from the integrated set and replace it
with an algebraic slaving relation (plus an optional first-order slip), eliminating the
fast eigenvalue from the explicit system.

## Strategy: two phases with a measurement gate

### Phase 1 — Baseline + instrumentation (correctness reference)

Keep the (stable) exact hierarchy through the stiff early phase and switch the daughter to
the fluid only once the relaxation is no longer stiff, using the k-aware trigger below.
No new evolved variables; low risk. Purpose: establish a trustworthy `rk` reference curve
and gather the evidence that sets Phase 2's slaving order.

- Keep the existing exact-moment **projection IC** at switch-on (`perturbations.c:5077`)
  unchanged: at modest `Λ` the projected true state is the most accurate IC.
- Keep the `ca2_ncdm` guards already landed (near-singular denominator → source-free
  fallback `ca2_0`, `[0,1]` causality clamp; precision `ncdm_ca2_den_tol`). They protect
  both phases.
- Instrument `6_test_fluid_vs_exact.ipynb` to record:
  - exact-vs-fluid divergence in P(k) and Cl_TT per scale,
  - wall-time ratio (`rk`+fluid-late vs `rk`+exact),
  - the stiffness ratio `Λ / max(aH, k·c_eff)` at the transition for each k.

### Phase 2 — Daughter–parent tight-coupling regime `acctca` (the speedup), gated on Phase 1

Add an approximation regime in which the daughter is slaved to the parent from birth
(cheap + non-stiff), transitioning to the standalone fluid at moderate `Λ`.

- New approximation flag `index_ap_acctca` (enum `acctca_off/on`) defined alongside
  `ncdmfa` in the approximation-interval machinery.
- **Daughter schedule in production:** slaved (`acctca`) from birth → fluid (`ncdmfa`) at
  the shared trigger below. The **exact hierarchy is never used in the production speedup
  path**; it remains available only as the validation path
  (`ncdm_fluid_approximation = none`).
- **Slaving (0th order):** `δ_acc = δ_dcdm`, `θ_acc = ¾ θ_dcdm`, `σ_acc = 0`.
- **Slaving order (0th vs 0th+first-order velocity slip in 1/Λ):** chosen from the Phase-1
  measurements — how large `Λ` still is at the latest `rk`-stable transition, and how much
  exact and fluid diverge there. The first-order slip mirrors the baryon–photon slip in
  CLASS TCA.
- **IC conversions** at `acctca → ncdmfa` are continuous by construction (slaved values ≈
  fluid attractor), so the transition is smooth.

## Shared trigger criterion (built once, used by both phases)

Replace the current k-independent density-ratio gate (`accDM_ready`,
`perturbations.c:6408`) with a stiffness criterion the explicit evolver actually cares
about:

> Switch the daughter out of exact (Phase 1) / out of `acctca` (Phase 2) into the fluid
> only when
> `Λ < κ_stiff · max(a·H, k·sqrt(ceff2))`,
> where `κ_stiff` is a new precision parameter of order 1–3.

This is per-wavenumber (CLASS approximations already switch per k), which is correct:
small-k modes tolerate the fluid earlier than large-k modes.

## New parameters

- `κ_stiff` — `class_precision_parameter`, default ~1–3, controls the exact/`acctca` →
  fluid transition (stiffness ratio threshold).
- `ncdm_ca2_den_tol` — already added.
- The legacy `ncdm_fluid_trigger_rho_accDM_over_rho_dcdm` is superseded by the
  stiffness criterion for the daughter; decide during planning whether to keep it as a
  fallback or retire it.

## Testing / success criteria

- `rk` + fluid completes with no "step too small" at the chosen trigger.
- P(k) and Cl_TT agree with the exact hierarchy within a tolerance to be fixed in the
  implementation plan.
- Net wall-time speedup of `rk`+Phase-2 over `rk`+exact, demonstrated in
  `6_test_fluid_vs_exact.ipynb`.
- The exact hierarchy (`ncdm_fluid_approximation = none`) remains the unchanged validation
  reference.

## Build note

No compiler in the agent shell; the user builds elsewhere. Each phase must be validated by
the user via the notebook after building.

## Decision gate (Phase 1 → Phase 2)

After Phase 1 produces:
- the `Λ / max(aH, k·c_eff)` distribution at the latest `rk`-stable transition, and
- the exact-vs-fluid divergence near that transition,

choose: (a) `κ_stiff` default, and (b) slaving order for `acctca` (0th vs 0th+slip).

---

## Revision 2026-06-22b — k-aware trigger reverted (monotonicity constraint)

**Found during Phase-1 testing:** the k-aware stiffness *trigger* is incompatible
with CLASS's approximation machinery. `perturbations_find_approximation_switches`
requires every approximation flag to be a **monotonic / irreversible** step
function of tau (its bisection assumes a single non-decreasing crossing per
level; see the comment at `perturbations.c` ~3724). The stiffness ratio
`Λ/max(aH, k·√ca2)` is **non-monotonic** in tau because `Γ_acc/H` rises while
`ratio_rho` falls, so gating `ncdmfa` on it makes the approximation reversible
(on→off→on) and aborts with *"you switch 2 approximations at the same time …
one approx is reversible"* (observed for k=1e-2 at tau=3000).

**Consequence for the design:** a genuinely k-aware stiffness *trigger* cannot be
delivered through the standard switch machinery. Phase 1 therefore keeps the
**monotonic** density-ratio gate (`rho_accDM/rho_acc_cdm`, the original trigger)
and demotes the stiffness ratio to a **diagnostic** logged at switch-on. The
`ca2_ncdm` guards remain the substantive Phase-1 robustness win.

**Implication for Phase 2:** the `acctca → ncdmfa` transition is subject to the
same monotonicity requirement. Its trigger must also be a monotonic function of
tau (e.g. the density ratio), not the stiffness ratio. `kappa_stiff` is retained
as a reserved parameter for Phase 2, where it can shape a monotonic threshold
rather than serve as the raw gate.
