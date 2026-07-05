# Fluid-closure feasibility diagnostic (does a universal daughter closure exist?) — Design Spec

**Status:** draft (2026-07-05). Prerequisite to any recalibration of the accDM daughter fluid approximation. Follows from the finding that [`7_test_ceff2_calibration.ipynb`](../../../notebooks_test/7_test_ceff2_calibration.ipynb) calibrates the wrong thing (a single `amp` on `P(k)`, confounded by the high-k blow-up) and that its Verdict numbers (RMS 2%→1.2%) do not match its own executed output (RMS 46–1316, dominated by architectural blow-up over k up to 10).

**Goal:** decide — *before building any fit* — whether a universal fluid closure exists for the accDM daughter over **k ≤ 1 Mpc⁻¹, f_acc ≤ 0.3**, at the ~1% P(k) accuracy target. Concretely: can the daughter's effective sound speed `ceff2` and viscosity `cvis2` each be written as **one function of `x = k/k_fs`**, or do they split by `f`/`η` (→ f-dependent coefficients), or fail to collapse at all (→ interpolation tables)?

## Why this is the right first step

The current fit fails for two independent reasons:

1. **Wrong target.** `amp` is tuned by brute-force full CLASS runs read off `P(k)`, which convolves the sound-speed error with growth, transfer, and the high-k fluid instability. The quantity the closure actually needs — `ceff2(k,τ) = δp/δρ` — is already emitted pointwise by one exact run (`cs2_ncdm[1]`).
2. **Wrong knob.** Only `ceff2` is tuned; `cvis2` (shear/viscosity) is left at its default `3·w·ca2`. The high-k instability lives in the fluid θ-equation and is damped by the `k²σ` (shear) term, so viscosity — not the sound-speed amplitude — is the likely stabilizing lever (`memory: paper-fluid-approx-2102-12498`, `memory: fluid-approx-marginal-speedup`).

Both the published `sqrt(k/k_fs)` fit and the "add a `cvis2` fit" idea rest on an **untested assumption**: that a single universal curve in `x = k/k_fs` fits all `(τ, f, η)`. This diagnostic measures that assumption directly. It is cheap (a handful of exact runs, pure Python) and its outcome selects the approach:

- **Approach A** — invert for effective `ceff2` and `cvis2`, fit both with saturating (Padé) forms.
- **Approach B** — two-parameter `ceff2` only, calibrated on the daughter transfer (no `cvis2` work).
- **Approach C** — tabulate the effective coefficients and interpolate in C.

## Feasibility gate (kept in frame throughout)

Even the **exact** hierarchy converges <1% only for f < ~0.1 (needs `q_size` ~5001, not 1001, above that; `memory: accdm-fluid-f-boundary`). At f = 0.3 the 1% *target itself* is expensive. The fluid only earns its place if it reaches 1% over k ≤ 1 **more cheaply** than exact + the q(f) schedule ([`14_test_q_schedule_calibration.ipynb`](../../../notebooks_test/14_test_q_schedule_calibration.ipynb)). A negative diagnostic (no clean closure, or a closure no cheaper than exact) is a valid, useful result that routes to notebook 14, not a failure.

## What the exact run already exposes (no C changes)

Per `(k, τ)`, the exact hierarchy stores, for the daughter (`n_acc`, last ncdm, `has_acc`-gated — `memory: last-species-must-be-has-acc-gated`):

- `delta_ncdm[1]` (δ), `theta_ncdm[1]` (θ), `shear_ncdm[1]` (σ) — [`perturbations.c:9076-9078`](../../../source/perturbations.c:9076)
- `cs2_ncdm[1]` = `δp/δρ`, the effective `ceff2`, already gauge-corrected — [`perturbations.c:8994`](../../../source/perturbations.c:8994)
- `k_fss_acc[1]` = `√(3/2)·aH/√ca2` — [`perturbations.c:9084`](../../../source/perturbations.c:9084); inverts to base `ca2 = (3/2)(aH/k_fs)²`

Existing scaffolding that already probes effective coefficients: `w_theta`, `w_sigma` (momentum-weighted effective coefficients for the θ and σ moments, [`perturbations.c:8836-8837`](../../../source/perturbations.c:8836)) and the `w_trial_1`/`w_trial_2` sound-speed trials (`w_trial_2 = ca2·(1 + 0.25·√(k/k_fs))`, "//works better", [`perturbations.c:8865`](../../../source/perturbations.c:8865)). `w_trial_*` are stored as columns; `w_sigma`/`w_theta` are computed but **not** currently emitted — exposing them is a one-line column add if step 4 needs it.

## Method (new notebook `15_test_fluid_closure_diagnostic.ipynb`, pure Python, no rebuild)

1. **Extractor.** Extend notebook 7's `run_cs2_of_k` to return the full **τ-series** (all stored times, not just z=0) of `{δ, θ, σ, δp/δρ, k_fs, aH}` for the daughter, over a k ≤ 1 grid (`logspace(-2, 0, ~20)`), for `f_acc ∈ {0.05, 0.1, 0.2, 0.3}` and two `η` (e.g. 0.1 production + one warmer) to separate `f`- from `η`-dependence. Recover base `ca2` from `k_fs`.

2. **Two dimensionless response functions.**
   - Sound-speed: `R_c(x) = (δp/δρ) / ca2`, `x = k/k_fs`. The current fit models this as `1 + amp·W·√x`.
   - Anisotropic-stress (the `cvis2` signature): `R_v(x) = k·σ/θ` (with `σ/δ` as a cross-check).
   Both are oscillatory and sign-flipping above `k_fs`, so compare on their **upper envelope** (rolling max of `|·|`); the smooth closure is an envelope, not a pointwise match (the paper's Fig-16 point).

3. **Collapse test + decision rule.** Overlay `R_c(x)` and `R_v(x)` across all `τ`, `f`, `η`; quantify the spread at fixed `x` (e.g. max fractional band width across curves).
   - Both collapse in `x` alone → **B/A with one universal formula.**
   - Collapse only at fixed `f` (curves shift monotonically with `f`) → **A with f-dependent coefficients** (fit the shift).
   - No collapse → **C (interpolation tables).**

4. **Surface the existing scaffolding.** Plot `w_trial_1/2` against the extracted `R_c` envelope; if informative for the θ/σ sectors, expose `w_theta`/`w_sigma` (one-line column add) and check whether they already encode a usable `cvis2(x)`.

5. **Feasibility-gate readout.** Record the converged-exact cost (`q_size` 5001 at f = 0.3) next to a candidate fluid cost, so the "is the fluid actually cheaper than exact + q(f)?" question stays explicit and a negative result routes cleanly to notebook 14.

## Global constraints

- Edit `class_accDM`, never the pristine reference (`memory: apply-fixes-to-working-branch`). This diagnostic needs **no C edit and no rebuild** — config-only against the built `classy`, except the optional one-line `w_sigma`/`w_theta` column expose in step 4 (which would require a rebuild, done on the user's side; `memory: build-environment`).
- Gauge: synchronous with `get_perturbations_in_current_gauge = yes`, matching the frame the C fluid equations use. Restrict to k ≤ 1 and late times so modes are sub-horizon, where the accDM quantities are gauge-robust (`memory: super-horizon-gauge-limitation`).
- Plots: STIX serif + ColorBrewer (`memory: notebook-plot-style`), following notebook 7's conventions.

## Success criteria

The diagnostic succeeds if it returns an unambiguous **A / A-with-f / B / C** verdict, backed by:

- collapse plots of `R_c(x)` and `R_v(x)` across `(τ, f, η)` with a quantified band width at fixed `x`;
- an explicit statement of whether `cvis2` (via `R_v`) shows structure the current `ceff2`-only fit ignores;
- the feasibility-gate comparison (candidate fluid cost vs converged exact at f = 0.3).

It does **not** need to produce the fit itself — that is the follow-up plan selected by the verdict.

## Risks / open points

- **Envelope definition.** The rolling-max envelope is sensitive to the τ/k sampling density; if the collapse is borderline, refine sampling before concluding "no collapse."
- **Gauge subtlety in σ, θ.** Shear and velocity are gauge-dependent; `R_v` must be built from the same-gauge quantities the fluid equations use. Sub-horizon k ≤ 1 keeps this controlled, but flag any large-scale points that drift.
- **`cvis2` inversion is deferred.** This diagnostic tests *achievability* via `R_v`, not the exact `cvis2(k,τ)` (which needs `σ'` and the metric sources — noisy). The exact inversion belongs to Approach A's plan, only if the diagnostic greenlights it.
- **Negative result is real.** If neither response collapses, or the closure is no cheaper than converged exact, the honest outcome is "fluid inadequate at f = 0.3; use exact + q(f) schedule (notebook 14)."

## Decision

Run the diagnostic; read the A/B/C verdict off the collapse plots. Only then write the follow-up plan (the fit) matched to the verdict. Do **not** build any fit — or touch `perturbations_ceff2_ncdm` — before the collapse behavior is known.
