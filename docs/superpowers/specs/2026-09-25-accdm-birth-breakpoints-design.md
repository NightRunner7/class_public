# accDM daughter births as integration breakpoints — Design Spec

**Status:** approved design (2026-09-25). Numerical bug fix, default behaviour.

## Problem

Notebook 30 found a per-k-mode P(k) error of ~0.16% rms, ~0.7% max at f_acc = 0.1, scaling with
f_acc and insensitive to the integration tolerance (tol 1e-7 makes rk fail at the birth peak).
Cause: before its birth a daughter bin is pinned to the parent by writing into `y` inside
`perturbations_derivs` with `dy = 0`. Only the first `derivs` call of an rk step sees the real state
(`dei_rkck.c`); the stages use a scratch vector. A step that straddles a birth therefore mixes pinned
and free equations and releases the bin from a state up to one step old; for instant births the
perturbation sums also jump inside the step. The rk error estimate cannot control either.

## Design

1. **Birth times** (`perturbations_init`, once per run): for each daughter bin, the conformal times
   of `lna_birth_lo/hi` via `background_tau_of_z` (τ0 for a ≥ 1, 0 before the background start),
   stored as `ppt->tau_birth_lo_acc/hi_acc`, plus their sorted unique list `ppt->tau_birth_break`.
2. **Sub-split** (`perturbations_solve`): each approximation interval is integrated in sub-intervals
   cut at every `tau_birth_break` value strictly inside it. The state carries over unchanged; the
   approximation scheme and `perturbations_vector_init` are untouched.
3. **Pinning** (`perturbations_derivs`): a bin is pinned while `tau <= tau_birth_hi_acc[q]`. At a
   breakpoint the first `derivs` call has τ equal to the birth time exactly, so the bin is pinned to
   the parent state at that instant and evolves freely from the next evaluation.
4. **Born weight** (perturbation sums in `perturbations_total_stress_energy` and
   `perturbations_vector_init`): instant births (lo = hi) have weight 0 for τ ≤ τ_birth and 1 after,
   so the jump sits on a breakpoint; smooth ramps keep the ln a ramp. The sums get τ through a new
   workspace field `ppw->tau_acc`, set in `perturbations_einstein`.
5. Default behaviour, no flag. A bin born at a ≥ 1 (the `qm_acc_birth` end node) now has weight 0 at
   τ0 instead of 1; its weight is ~1e-10 of the daughter.

## Validation

- New `notebooks_test/test_birth_breakpoints.py`, m_acc = 1e11 GeV, f_acc = 0.1, P(k) only:
  null pair (`k_step_sub` 0.05 vs 0.0505) max residual < 1e-3 (was ~5e-3); a run with
  `tol_perturbations_integration = 1e-7` completes (was a step-size failure).
- `test_de_sink.py` passes; notebook 28 meas/pred with the sink on stays within 5% of 1.
- Runtime of one accDM P(k) run within +20% at 51 bins; the 501-bin `qm_simpson_log` runtime is
  reported. If it is much slower, merge breakpoints closer than a small Δτ (follow-up).
- Then regenerate the golden regression files and rerun notebook 1.
