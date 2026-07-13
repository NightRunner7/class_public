# ceff2(f, eta) empirical formula — extraction, fit, P(k) validation (notebook 18)

**Date:** 2026-07-13
**Status:** approved (design), implementation in notebook 18
**Predecessors:** nb15 (plateau diagnostic), nb17 (eta-plateau mode-3 P(k) validation at f=0.1),
memory `ceff2-fit-structure`.

## Goal

Deliver an explicit, fitted formula `ceff2(f, eta)` for the daughter's effective sound
speed, valid at **fixed production knobs** kappa = 6, a_t = 0.13, m = 1e16 GeV (the user's
production configuration), and validate it end-to-end in P(k). The model-parameter
dependences must be visible in the notebook — no constants buried in helper defaults or
interpolation tables.

Context: the eta-only law `ceff2 = (1/3)(1 - exp(-3*A_eta*eta))` with A_eta = 0.55 was fit
on an eta-scan at f = 0.1 only. nb15 found the plateau "f-independent to 0.5% up to
f = 0.3" at three eta values, but no fit has quantified this across the (f, eta) grid, and
no P(k) validation exists at the f = 0.3 corner where the formula would actually be used.

## Formula form (approach chosen)

Saturating eta-law with f-corrected amplitude:

```
ceff2(f, eta) = (1/3) * (1 - exp(-3 * A(f) * eta)),   A(f) = A0 * (1 + B*f)
```

- If the fitted B is consistent with 0, the deliverable collapses to the eta-only law and
  f-independence is *demonstrated*, not assumed.
- No C changes needed: `ncdm_ceff2_mode = 3` already takes `ncdm_ceff2_eta_A`, so the
  fitted `A(f)` is passed per run as an effective A.
- Rejected alternatives: 2D table/spline (opaque, needs C-side table support); a new C
  mode with native f-dependence (only justified if B is significantly nonzero — deferred).

## Notebook structure (`notebooks_test/18_test_ceff2_f_eta_formula.ipynb`)

All CLASS results pickle-cached under `accDM_scans/nb18_cache/`; tqdm + bracketing prints
on slow runs; rkck evolver (`evolver: 0`) throughout.

### Setup
Same base cosmology/param builders as nb15/nb17. `f` and `eta` are the only scan arguments
of the builders; kappa, a_t, mass appear once as module constants flagged as the
calibration domain.

### Stage A — plateau extraction
- Grid: f = {0.03, 0.1, 0.2, 0.3} x eta = {0.01, 0.05, 0.1, 0.3, 0.5} (20 exact runs).
- nb15's proven extractor: exact hierarchy with `k_output_values` over
  K_GRID = logspace(-2, 0, 18), q_size = 501 (matches the nb15 v3 cache the A_eta = 0.55
  calibration was measured on).
- Plateau per (f, eta): median pole-masked `cs2_ncdm[1] = delta_p/delta_rho` (masked
  against delta, drop_frac = 0.2) over the flat window x = k/k_fs in [0.05, 30]
  (the 2026-07-13 window; x > 30 phase-mixing/q-grid artifacts excluded).
- Output: plateau table (eta rows x f columns), per-eta f-spread, and a plot of
  c_plateau vs eta with one curve per f.

### Stage B — fit
- De-saturate: y = -ln(1 - 3*c)/3, so the model is linear: y = A(f) * eta.
- Per-f slope A_f by least squares through the origin; then fit A(f) = A0*(1 + B*f).
- Report the eta-only null model (B = 0) vs the f-corrected model side by side with
  per-(f, eta) residual tables; adopt the simplest model within measurement scatter.
- Final cell defines `ceff2_formula(f, eta)` as a plain function with the fitted constants
  printed and visible.
- New pure-numpy helpers in `fluid_closure_helpers.py` (TDD, offline unit tests):
  `desaturate_cfs`, `fit_eta_slope`, `fit_A_of_f`, `A_eff_of_f`, `ceff2_f_eta`.

### Stage C — P(k) validation
- Fail-fast C-vs-python assertion first (nb17 pattern): stored `cs2_ncdm[1]` late-time
  value must equal the python formula to <1%, so a silent mode fallback (the nb16 typo
  lesson) cannot go unnoticed.
- Corners (f, eta) in {0.1, 0.3} x {0.05, 0.5} + center (0.2, 0.1): exact reference vs
  mode-3 fluid with `ncdm_ceff2_eta_A = A(f)`, trigger 0.4.
- Exact-reference q-schedule: q_size = 1001 for f <= 0.1, 5001 for f > 0.1
  (memory `accdm-fluid-f-boundary`); the f = 0.3 rows are the slow part.
- Score `max|P_fluid/P_exact - 1|` over **k <= 1 Mpc^-1 only**; success ~1%.

### Verdict
FILL-slot markdown. Standing rule: if P(k) misses 1% with the correct ceff2, the shear
closure `cvis2` is the named suspect — do not tune ceff2 further.

## Success criteria

1. Plateau table complete over the 4x5 grid; flat-window medians finite and sub-1/3.
2. Fitted (A0, B) with uncertainty/scatter; explicit statement whether B is consistent
   with 0.
3. P(k) residual <~1% over k <= 1 at all five corners, including f = 0.3.
4. C-vs-python plateau assertion passes (<1%).

## Out of scope

- Universality of A(f) in (kappa, a_t) — known open item, separate study.
- k > 1 Mpc^-1 (fluid blows up architecturally there; nb7 lesson).
- cvis2 (shear closure) tuning.
