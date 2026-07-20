# P3: Bounded, recalibratable free-streaming sound-speed correction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the accDM daughter's effective-sound-speed (free-streaming) correction physically bounded (`c² ≤ ceff2 ≤ 1/3`), DRY (one helper instead of three copy-pasted fit sites), and recalibratable via precision parameters — while defaulting to behaviour bit-identical to the current Eq-(38) fit so nothing changes until the new mode is explicitly enabled and calibrated.

**Architecture:** The paper's Eq-(38) fit `ceff2 = c²·(1 + 0.2·(1−2ε)·√(k/k_fs))` is calibrated for thermal, ε≤½ daughters and grows like `√x` without bound — for the boosted accDM PSD it can overshoot the causal `1/3` ceiling when the base sound speed `c²` is large (a relativistic daughter at early times) times a large `k/k_fs`. (The weight `1−2·eps_acc` stays in `(0,1]` for all physical η≥0 — `eps_acc` rises monotonically to `1/2`, so `W→0` as η→∞; the overshoot is a base-`c²` effect, not a weight-sign effect.) We consolidate the three duplicated fit sites into a single static helper `perturbations_ceff2_ncdm()`, add a `ncdm_ceff2_mode` flag selecting `fit` (mode 0, exact current behaviour, default) or `bounded` (mode 1, the same fit hard-capped at `1/3`, i.e. `min(fit, 1/3)`), and expose the amplitude as `ncdm_ceff2_fs_amp` (default `0.2`). The bounded amplitude is then calibrated against the exact hierarchy in a notebook.

**Tech Stack:** CLASS (C), built with the user's toolchain (no compiler in the agent shell — the user builds and runs every verification). Validation via `classy` in `notebooks_test/`.

> **Implementation update (2026-06-23, supersedes Task-3 code blocks below).** Mode 1 was simplified from the original "saturating blend" to a plain **`min(fit, 1/3)`** hard cap (see `perturbations_ceff2_ncdm` in `source/perturbations.c`). Reason: the blend deviated ~1% from the fit *below* the ceiling, so mode 1 was not a pure no-op in the safe regime; `min(fit,1/3)` is identical to mode 0 until a real overshoot, simpler, and makes the fit-vs-bounded divergence a clean overshoot detector. The `dmax`/`lin` blend in Task 3's code blocks is therefore historical — implement the `min` form. Calibration finding (`notebooks_test/7_test_ceff2_calibration.ipynb`): the published `amp=0.2` is miscalibrated for the boosted PSD; `amp ≈ 1.0–1.5` roughly **halves** the P(k) residual (RMS 2.0%→1.2%, max 5.6%→3.1%), and the bound only starts to engage once amp is pushed that high. Final amp TBD by a refined sweep.

## Global Constraints

- Edit `class_accDM`, never the pristine `axion_project/class_public` reference.
- No compiler in the agent shell: every build/verify step is performed by the user. The "Run" commands below are for the user; treat their output as the gate.
- Daughter = last ncdm species: `n_acc = pba->N_ncdm-1`, guarded by `pba->has_acc == _TRUE_`. Never special-case `n == N_ncdm-1` without the `has_acc` guard (it corrupts plain ncdm runs).
- The exact hierarchy (`ncdm_fluid_approximation = none`) remains the unchanged validation reference.
- This change touches ONLY the fluid effective sound speed (`ceff2` / `delta_p`). It does not, and must not, alter `cg2_ncdm`/`ca2_ncdm` (the adiabatic sound speed), the decay couplings, or the IC.
- The fluid is only stable/accurate at late, cold switch-on (see `memory: fluid-approx-unusable`). Validate in that window (`ncdm_fluid_trigger_rho_accDM_over_rho_dcdm ~ 0.3`); do NOT expect the high-k blow-up to change — it is a separate, architectural instability.
- Mode 0 must be **bit-identical** to today's code with `ncdm_ceff2_fs_amp = 0.2`. The golden regression must not move until mode 1 is explicitly selected.
- Build command placeholder below is `make`; substitute your actual build invocation.

---

### Task 1: Add the `ncdm_ceff2_mode` and `ncdm_ceff2_fs_amp` precision parameters

**Files:**
- Modify: `include/precisions.h` (insert after `ncdm_ca2_den_tol` at line 420)

**Interfaces:**
- Consumes: nothing.
- Produces: `ppr->ncdm_ceff2_mode` (int, 0=fit, 1=bounded), `ppr->ncdm_ceff2_fs_amp` (double, free-streaming correction amplitude).

- [ ] **Step 1: Add the two parameters with documentation**

In `include/precisions.h`, immediately after the `ncdm_ca2_den_tol` block ending at line 420, insert:

```c
/**
 * accDM daughter free-streaming (effective sound speed) correction mode.
 *   0 = fit:     ceff2 = ca2*(1 + ncdm_ceff2_fs_amp*(1-2*eps_acc)*sqrt(k/k_fs))
 *                (the published Eq-38 form; default, reproduces prior behaviour)
 *   1 = bounded: the same fit hard-capped at 1/3, i.e. min(fit, 1/3). Identical
 *                to mode 0 below 1/3; a pure safety cap (see implementation note
 *                at top of plan - the blend form below is superseded).
 * Has no effect when has_acc is false.
 */
class_precision_parameter(ncdm_ceff2_mode,int,0)
/**
 * Amplitude of the accDM daughter free-streaming sound-speed correction (the
 * coefficient multiplying (1-2*eps_acc)*sqrt(k/k_fs)). Default 0.2 reproduces
 * the published fit. Recalibrated against the exact hierarchy for the boosted
 * PSD (see notebooks_test/7_test_ceff2_calibration.ipynb). Has no effect when
 * has_acc is false.
 */
class_precision_parameter(ncdm_ceff2_fs_amp,double,0.2)
```

- [ ] **Step 2: Build**

Run: `make`
Expected: compiles with no error referencing `ncdm_ceff2_mode` or `ncdm_ceff2_fs_amp`.

- [ ] **Step 3: Verify the parameters parse**

Run a one-off `classy` check:

```python
from classy import Class
c = Class()
c.set({'ncdm_ceff2_mode': 0, 'ncdm_ceff2_fs_amp': 0.2})
c.compute()           # any minimal accDM .ini params you already use
print("parsed OK")
```

Expected: prints `parsed OK`, no "read but not used"/"unknown parameter" warning for either name.

- [ ] **Step 4: Commit**

```bash
git add include/precisions.h
git commit -m "feat(accDM): add ncdm_ceff2_mode and ncdm_ceff2_fs_amp precision params"
```

---

### Task 2: Extract the fit into a single helper (pure refactor, bit-identical)

**Files:**
- Modify: `source/perturbations.c` (add static helper near `perturbations_acc_stiff_ratio` at line 3832; replace the three call sites at 7341, 7359, 10093)

**Interfaces:**
- Consumes: `ppr->ncdm_ceff2_fs_amp`, `pba->eps_acc`.
- Produces: `static double perturbations_ceff2_ncdm(struct precision * ppr, struct background * pba, double cs2_base, double k, double a, double H)` — returns the daughter effective sound speed (delta_p/delta_rho) given the base adiabatic sound speed `cs2_base` (`cg2` or `ca2`).

- [ ] **Step 1: Add the helper implementing mode 0 only (exact current fit)**

In `source/perturbations.c`, immediately before `perturbations_acc_stiff_ratio` (line 3832), insert:

```c
/**
 * accDM daughter effective sound speed ceff2 (= delta_p/delta_rho) from the
 * base adiabatic sound speed cs2_base (cg2 or ca2). Mode 0 is the published
 * Eq-38 fit; mode 1 (added in a later task) is the bounded saturating form.
 * cs2_base <= 0 returns 0 (degenerate/unborn daughter). Has no accDM-specific
 * gating itself: callers invoke it only for n_acc with has_acc true.
 */
static double perturbations_ceff2_ncdm(struct precision * ppr,
                                       struct background * pba,
                                       double cs2_base,
                                       double k, double a, double H) {

  if (cs2_base <= 0.) return 0.;

  double W  = 1.0 - 2.0*pba->eps_acc;
  double xr = sqrt(k*sqrt(2./3.)*sqrt(cs2_base)/(a*H));   /* sqrt(k/k_fs) */
  return cs2_base*(1.0 + ppr->ncdm_ceff2_fs_amp*W*xr);
}
```

- [ ] **Step 2: Replace call site at line 7341 (`perturbations_total_stress_energy`)**

Replace:

```c
                ppw->delta_p_over_delta_rho_ncdm[n_ncdm] = cg2_ncdm*(1.0+0.2*(1.0-2.0*pba->eps_acc)*sqrt(k*sqrt(2./3.)*sqrt(cg2_ncdm)/(a*H)));
```

with:

```c
                ppw->delta_p_over_delta_rho_ncdm[n_ncdm] = perturbations_ceff2_ncdm(ppr,pba,cg2_ncdm,k,a,H);
```

- [ ] **Step 3: Replace call site at line 7359 (`perturbations_total_stress_energy`)**

Replace:

```c
              ppw->delta_p += cg2_ncdm*(1.0+0.2*(1.0-2.0*pba->eps_acc)*sqrt(k*sqrt(2./3.)*sqrt(cg2_ncdm)/(a*H)))*rho_ncdm_bg*y[idx];
```

with:

```c
              ppw->delta_p += perturbations_ceff2_ncdm(ppr,pba,cg2_ncdm,k,a,H)*rho_ncdm_bg*y[idx];
```

- [ ] **Step 4: Replace call site at line 10093 (`perturbations_derivs`)**

Replace:

```c
              ceff2_ncdm = ca2_ncdm*(1.0+0.2*(1.0-2.0*pba->eps_acc)*sqrt(k*sqrt(2./3.)*sqrt(ca2_ncdm)/(a*H)));
```

with:

```c
              ceff2_ncdm = perturbations_ceff2_ncdm(ppr,pba,ca2_ncdm,k,a,H);
```

- [ ] **Step 5: Build**

Run: `make`
Expected: compiles cleanly. (`ppr`, `pba`, `k`, `a`, `H` are already in scope in both `perturbations_total_stress_energy` and `perturbations_derivs`.)

- [ ] **Step 6: Verify bit-identical to pre-refactor (golden regression unchanged)**

Run: `notebooks_test/1_test_regression_golden.ipynb` against the committed golden in `notebooks_test/golden/regression_accDM.*`.
Expected: the accDM `Cl`/`Pk` match the golden to machine precision (max relative diff `< 1e-10`). With `ncdm_ceff2_fs_amp` defaulting to `0.2` and mode 0, the helper reproduces the old expression exactly. If the golden moves at all, the extraction changed a value — diff the three sites.

- [ ] **Step 7: Commit**

```bash
git add source/perturbations.c
git commit -m "refactor(accDM): consolidate ceff2 fit into perturbations_ceff2_ncdm helper"
```

---

### Task 3: Implement the bounded (saturating) mode

**Files:**
- Modify: `source/perturbations.c` (the `perturbations_ceff2_ncdm` helper added in Task 2)

**Interfaces:**
- Consumes: `ppr->ncdm_ceff2_mode`, `ppr->ncdm_ceff2_fs_amp`, `pba->eps_acc`.
- Produces: unchanged signature; mode 1 now returns the bounded value.

- [ ] **Step 1: Replace the helper body with mode-switched logic**

Replace the body of `perturbations_ceff2_ncdm` (everything after the `if (cs2_base <= 0.) return 0.;` guard) with:

```c
  double W   = 1.0 - 2.0*pba->eps_acc;
  double xr  = sqrt(k*sqrt(2./3.)*sqrt(cs2_base)/(a*H));   /* sqrt(k/k_fs) */

  if (ppr->ncdm_ceff2_mode == 0) {
    /* mode 0: published Eq-38 fit, unbounded — keep the exact factored form so
       this is bit-identical to the pre-helper code (golden must not move). */
    return cs2_base*(1.0 + ppr->ncdm_ceff2_fs_amp*W*xr);
  }

  double lin = cs2_base*ppr->ncdm_ceff2_fs_amp*W*xr;       /* additive correction (fit's) */

  /* mode 1: saturating blend with the same small-x slope `lin`, capped at the
     causal ceiling 1/3. For physical eta>=0 the weight W=1-2*eps_acc is in (0,1]
     so lin>=0 and the first branch is taken; the lin<0 branch is a defensive
     guard. Both branches are non-singular: denominators are strictly positive. */
  double ceff2;
  if (lin >= 0.) {
    double dmax = 1./3. - cs2_base;
    if (dmax <= 0.) return (cs2_base < 1./3. ? cs2_base : 1./3.); /* already at/above ceiling */
    ceff2 = cs2_base + dmax*lin/(dmax + lin);                     /* -> 1/3 as lin -> inf */
  }
  else {
    ceff2 = cs2_base + cs2_base*lin/(cs2_base - lin);             /* -> 0   as lin -> -inf */
  }

  if (ceff2 < 0.)    ceff2 = 0.;        /* hard causal guards */
  if (ceff2 > 1./3.) ceff2 = 1./3.;
  return ceff2;
```

- [ ] **Step 2: Build**

Run: `make`
Expected: compiles cleanly.

- [ ] **Step 3: Verify mode 0 still bit-identical**

Run: `notebooks_test/1_test_regression_golden.ipynb` (default params → mode 0).
Expected: still matches golden to `< 1e-10`. (Mode 0 keeps the exact factored `cs2_base*(1+...)` form from Task 2 — bit-identical; confirm no regression.)

- [ ] **Step 4: Verify mode 1 stays within [0, 1/3] and reduces to mode 0 at small k/k_fs**

Run this numeric check that reproduces the helper in Python and asserts the two properties:

```python
import numpy as np
def ceff2(cs2, k, a, H, eps, amp=0.2, mode=0):
    if cs2 <= 0: return 0.0
    W = 1 - 2*eps
    xr = np.sqrt(k*np.sqrt(2/3)*np.sqrt(cs2)/(a*H))
    lin = cs2*amp*W*xr
    if mode == 0: return cs2 + lin
    if lin >= 0:
        dmax = 1/3 - cs2
        if dmax <= 0: return min(cs2, 1/3)
        c = cs2 + dmax*lin/(dmax+lin)
    else:
        c = cs2 + cs2*lin/(cs2-lin)
    return min(max(c, 0.0), 1/3)

# (a) bounded everywhere and k>>k_fs. Physical eps_acc is in [0,1/2); the 0.6 and
#     0.99 values (W<0) are NON-physical and exercise the defensive guard only.
for eps in [0.0, 0.3, 0.6, 0.99]:
    for x in np.logspace(-3, 6, 50):           # k/k_fs from tiny to huge
        cs2, a, H = 1e-3, 1e-3, 1.0
        k = x*np.sqrt(3/2)*a*H/np.sqrt(cs2)     # invert k/k_fs
        c1 = ceff2(cs2, k, a, H, eps, mode=1)
        assert 0.0 <= c1 <= 1/3 + 1e-15, (eps, x, c1)
# (b) small-x: mode 1 ~ mode 0 to O((k/k_fs))
k_small = 1e-3*np.sqrt(3/2)*a*H/np.sqrt(cs2)
c0 = ceff2(cs2, k_small, a, H, 0.3, mode=0)
c1 = ceff2(cs2, k_small, a, H, 0.3, mode=1)
assert abs(c1-c0)/abs(c0-cs2) < 1e-2, (c0, c1)
print("bounded-mode invariants hold")
```

Expected: prints `bounded-mode invariants hold` (no assertion error).

- [ ] **Step 5: Commit**

```bash
git add source/perturbations.c
git commit -m "feat(accDM): bounded saturating ceff2 mode (mode 1), capped at causal 1/3"
```

---

### Task 4: Calibrate the bounded amplitude against the exact hierarchy

**Files:**
- Create: `notebooks_test/7_test_ceff2_calibration.ipynb`

**Interfaces:**
- Consumes: the exact hierarchy (`ncdm_fluid_approximation = none`) as ground truth, and the two precision params from Task 1.
- Produces: a recommended `ncdm_ceff2_fs_amp` for mode 1, and a plot of fit vs bounded vs exact `c_s²(k,τ)` (the paper's Fig. 16 analog).

- [ ] **Step 1: Build the exact `c_s²(k,τ)` envelope**

In the new notebook, run the exact hierarchy (`ncdm_fluid_approximation: 'none'`) for a small grid in η (e.g. `eta_acc` mapping to the paper's `ε = 0.5, 0.1, 0.01`) at fixed late τ (cold-switch window), extract the daughter `delta_p/delta_rho` as a function of `k` spanning `k < k_fs` to `k > k_fs`. Use the STIX serif + ColorBrewer style (see `memory: notebook-plot-style`). This is the empirical `c_s²(k)` to match.

- [ ] **Step 2: Overlay the fit (mode 0) and bounded (mode 1) sound speeds**

For the same grid, evaluate the `ceff2` helper (reuse the Python port from Task 3 Step 4) for mode 0 and mode 1 across `ncdm_ceff2_fs_amp ∈ {0.1, 0.2, 0.3}`. Plot all three curves against the exact envelope, marking `k_fs` and the `1/3` ceiling.

- [ ] **Step 3: Pick the amplitude and confirm the cold limit**

Choose the `ncdm_ceff2_fs_amp` that minimises the residual to the exact envelope in the stable window. Confirm that as η→0 the chosen mode-1 curve collapses onto the mode-0 fit (cold-limit reduction). Record the recommended value in a markdown cell and in `memory: paper-fluid-approx-2102-12498` if it differs materially from 0.2.

- [ ] **Step 4: Commit**

```bash
git add notebooks_test/7_test_ceff2_calibration.ipynb
git commit -m "test(accDM): ceff2 calibration notebook (fit vs bounded vs exact c_s^2)"
```

---

### Task 5: Validate fluid-vs-exact improvement in the stable window

**Files:**
- Modify: `notebooks_test/6_test_fluid_vs_exact.ipynb` (add a mode-0 vs mode-1 comparison cell)

**Interfaces:**
- Consumes: modes 0/1 + the calibrated amplitude from Task 4.
- Produces: evidence that mode 1 residuals (vs exact) are ≤ mode 0 residuals and that no `ceff2 > 1/3` occurs.

- [ ] **Step 1: Add a mode-0/mode-1/exact comparison at cold switch-on**

In `6_test_fluid_vs_exact.ipynb`, at the stable trigger (`ncdm_fluid_trigger_rho_accDM_over_rho_dcdm ~ 0.3`), run three cases: exact (`ncdm_fluid_approximation: none`), fluid mode 0, fluid mode 1 (calibrated amp). Compute `P(k)` and `Cl` residuals of each fluid case vs exact.

- [ ] **Step 2: Assert mode 1 does not regress and is bounded**

```python
import numpy as np
# resid_fit, resid_bounded: |fluid - exact| / |exact| arrays over k (Pk) computed above
assert np.nanmax(resid_bounded) <= np.nanmax(resid_fit) * 1.05, "mode 1 regressed vs fit"
# and (if ceff2 is dumped via a perturbation output) no value exceeds 1/3:
# assert np.nanmax(ceff2_dump) <= 1/3 + 1e-12
print("mode 1 within tolerance and bounded")
```

Expected: prints the message; mode 1 is at least as good as the fit in the stable window (the win shows up at high k / large η where the fit overshoots).

- [ ] **Step 3: Record the verdict**

Add a markdown cell summarising: where mode 1 helps (high k / large η), where it is neutral (cold, k<k_fs → identical to fit), and confirming the high-k architectural blow-up is unchanged (expected — out of scope, see `memory: fluid-approx-unusable`).

- [ ] **Step 4: Commit**

```bash
git add notebooks_test/6_test_fluid_vs_exact.ipynb
git commit -m "test(accDM): validate bounded ceff2 mode vs exact in stable window"
```

---

## Notes for the implementer
- **Do not touch** `cg2_ncdm`/`ca2_ncdm` (adiabatic sound speed), the `ncdm_ca2_den_tol` guards, the decay couplings, or the `a<=aq` IC. This plan changes only the *effective* sound speed used in `delta_p` and `delta_p/delta_rho`.
- The `(1−2·eps_acc)` weight (W) is in `(0, 1]` for all physical η≥0 (`eps_acc` rises monotonically to `1/2`, so `W→0` as η→∞ — it never flips sign). The overshoot the bounded mode caps comes from a large base `c²` (relativistic daughter) × large `k/k_fs`, not from a negative weight. The `lin < 0` branch and the hard `[0, 1/3]` clamp are defensive guards only — keep them, but they are not exercised for physical inputs.
- Mode 0 remains the default and the production path until Task 4/5 justify switching. Recalibration of the **weight** `eps_acc` itself (vs the amplitude) is deliberately out of scope here — it is a separate, riskier change; this plan only recalibrates the amplitude and bounds the shape.
