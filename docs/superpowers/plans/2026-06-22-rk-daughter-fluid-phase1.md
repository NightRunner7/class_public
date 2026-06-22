# Phase 1: k-aware stiffness trigger + instrumentation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `rk` complete the accDM daughter fluid run by switching exact→fluid on a physically-based, k-aware stiffness criterion instead of a k-independent density ratio, and instrument the fluid-vs-exact comparison to gather the evidence that decides Phase 2's slaving order.

**Architecture:** The daughter's fluid relaxation rate `Λ = a·Γ·(1+η)·((1+ca2)/(1+w))·ratio_rho` is compared per-wavenumber to the competing dynamical/oscillation rate `max(aH, k·sqrt(ca2))`. The fluid is allowed on only when this stiffness ratio drops below a precision threshold `kappa_stiff`, so the explicit `rk` evolver never sees the large negative eigenvalue. A scratch workspace field carries the ratio so it can be logged at the actual switch-on for the Phase-2 decision gate.

**Tech Stack:** CLASS (C), built with the user's toolchain (no compiler in the agent shell — the user builds and runs each verification). Validation via `classy` in `notebooks_test/6_test_fluid_vs_exact.ipynb`.

## Global Constraints

- Edit `class_accDM`, never the pristine `axion_project/class_public` reference.
- No compiler in the agent shell: every verification step is built and run by the user.
- Daughter = last ncdm species: `n_acc = pba->N_ncdm-1`, guarded by `pba->has_acc == _TRUE_`. Never special-case `n == N_ncdm-1` without the `has_acc` guard (it corrupts plain ncdm runs).
- The exact hierarchy (`ncdm_fluid_approximation = none`) must remain the unchanged validation reference.
- Keep the `ca2_ncdm` guards already in place (`ncdm_ca2_den_tol`, `ca2_0` fallback, `[0,1]` clamp).
- Build command placeholder below is `make`; substitute your actual build invocation.

---

### Task 1: Add the `kappa_stiff` precision parameter

**Files:**
- Modify: `include/precisions.h:398` (insert after `ncdm_fluid_trigger_rho_accDM_over_rho_dcdm`, before `ncdm_ca2_den_tol` at line 406)

**Interfaces:**
- Consumes: nothing.
- Produces: `ppr->kappa_stiff` (double), the stiffness-ratio threshold for switching the daughter exact→fluid.

- [ ] **Step 1: Add the parameter with documentation**

In `include/precisions.h`, immediately after the existing block ending at line 398
(`class_precision_parameter(ncdm_fluid_trigger_rho_accDM_over_rho_dcdm,double,1.0e-6)`),
insert:

```c
/**
 * Stiffness-ratio threshold for switching the accDM daughter from the exact
 * hierarchy (Phase 1) / tight-coupling (Phase 2) into the fluid approximation.
 * The daughter fluid is allowed on only when
 *   Lambda / max(aH, k*sqrt(ca2)) < kappa_stiff,
 * where Lambda = a*Gamma*(1+eta)*((1+ca2)/(1+w))*ratio_rho is the decay
 * relaxation rate. Below this the relaxation is no longer fast compared to the
 * dynamical/oscillation rates, so the explicit rk evolver is stable. Order 1.
 * Has no effect when has_acc is false. Supersedes the k-independent
 * ncdm_fluid_trigger_rho_accDM_over_rho_dcdm gate for the daughter (that
 * parameter is retained, but unused, so existing .ini files still parse).
 */
class_precision_parameter(kappa_stiff,double,1.0)
```

- [ ] **Step 2: Build**

Run: `make`
Expected: compiles with no error referencing `kappa_stiff`.

- [ ] **Step 3: Verify the parameter is recognized**

Create a one-line check `.ini` or set it in a `classy` dict (`{'kappa_stiff': 1.0}`) and run
a trivial model. Expected: runs without "unknown parameter kappa_stiff".

- [ ] **Step 4: Commit**

```bash
git add include/precisions.h
git commit -m "Add kappa_stiff precision parameter for k-aware daughter fluid trigger"
```

---

### Task 2: Add the `acc_stiff_ratio` workspace diagnostic field

**Files:**
- Modify: `include/perturbations.h:666` (add field next to `ca2_ncdm_bad`)

**Interfaces:**
- Consumes: nothing.
- Produces: `ppw->acc_stiff_ratio` (double), last-computed daughter stiffness ratio
  `Λ / max(aH, k·sqrt(ca2))`, written in `perturbations_approximations`, read at switch-on
  logging in `perturbations_vector_init`.

- [ ] **Step 1: Add the field**

In `include/perturbations.h`, in the `perturbations_workspace` struct, replace the block at
lines 663-666:

```c
  /* AccDM: set inside perturbations_derivs when ca2_ncdm goes negative or
     denominator hits zero; checked after the evolver returns so we abort
     cleanly without overflowing the error_message buffer */
  short ca2_ncdm_bad;
```

with:

```c
  /* AccDM: set inside perturbations_derivs when ca2_ncdm goes negative or
     denominator hits zero; checked after the evolver returns so we abort
     cleanly without overflowing the error_message buffer */
  short ca2_ncdm_bad;

  /* AccDM: last daughter stiffness ratio Lambda/max(aH,k*sqrt(ca2)) computed in
     perturbations_approximations; logged at fluid switch-on for the Phase-2
     slaving-order decision. Large value means "still stiff". */
  double acc_stiff_ratio;
```

- [ ] **Step 2: Build**

Run: `make`
Expected: compiles cleanly.

- [ ] **Step 3: Commit**

```bash
git add include/perturbations.h
git commit -m "Add acc_stiff_ratio workspace field for daughter fluid trigger diagnostic"
```

---

### Task 3: Replace the density-ratio gate with the k-aware stiffness criterion

**Files:**
- Modify: `source/perturbations.c:6403-6420` (the `accDM_ready` block in `perturbations_approximations`)

**Interfaces:**
- Consumes: `ppr->kappa_stiff` (Task 1), `ppw->acc_stiff_ratio` (Task 2), and background
  quantities from `ppw->pvecback`: `index_bg_a`, `index_bg_H`, `index_bg_Gamma_acc`,
  `index_bg_rho_acc_cdm`, `index_bg_rho_ncdm1`, `index_bg_p_ncdm1`,
  `index_bg_pseudo_p_ncdm1`; and `pba->eta_acc`, `pba->N_ncdm`, function parameter `k`.
- Produces: the daughter fluid switch-on decision and a populated `ppw->acc_stiff_ratio`.

- [ ] **Step 1: Establish the failing baseline (red)**

In `notebooks_test/6_test_fluid_vs_exact.ipynb`, run the accDM model with the explicit
evolver and an early fluid switch using the *old* density trigger:

```python
from classy import Class
base = {
    # ... your standard accDM parameters (kappa_acc, eta_acc, lifetime, etc.) ...
    'evolver': 0,                       # 0 = rk per enum evolver_type {rk, ndf15} — confirm in your build
    'ncdm_fluid_approximation': 'CLASS',
    'ncdm_fluid_trigger_rho_accDM_over_rho_dcdm': 1e-6,   # early, large ratio_rho
}
cosmo = Class(); cosmo.set(base)
try:
    cosmo.compute()
    print("completed")
except Exception as e:
    print("FAILED:", e)
cosmo.struct_cleanup()
```

Expected: `FAILED: ... step too small ...` (the documented stiffness crash). Record it.

- [ ] **Step 2: Rewrite the gate**

In `source/perturbations.c`, replace lines 6403-6416 (the comment + `accDM_ready`
computation through the closing brace of the `if (pba->has_acc...)` block):

```c
      /* AG: for accDM models, hold ncdmfa off until accDM has been produced
         from dcdm decay. The analytical sound-speed at perturbations.c:~10171
         goes singular when rho_ncdm is essentially zero: ratio_rho is huge,
         gamma/H is tiny, and their finite product can hit
         3*(1+w_ncdm)/(1-eps_acc), zeroing the denominator. */
      short accDM_ready = _TRUE_;
      if (pba->has_acc == _TRUE_) {
        double rho_acc_cdm_bg  = ppw->pvecback[pba->index_bg_rho_acc_cdm];
        double rho_accDM_bg = ppw->pvecback[pba->index_bg_rho_ncdm1 + pba->N_ncdm-1];
        if (rho_acc_cdm_bg > 0. &&
            rho_accDM_bg/rho_acc_cdm_bg < ppr->ncdm_fluid_trigger_rho_accDM_over_rho_dcdm) {
          accDM_ready = _FALSE_;
        }
      }
```

with:

```c
      /* AG: for accDM models, switch the daughter exact->fluid only when the
         decay relaxation is no longer stiff for the explicit rk evolver, i.e.
         when Lambda/max(aH,k*sqrt(ca2)) < kappa_stiff. This is per-k (small-k
         modes tolerate the fluid earlier). Replaces the k-independent
         density-ratio gate. See docs/superpowers/specs/2026-06-22-*. */
      short accDM_ready = _TRUE_;
      ppw->acc_stiff_ratio = 1.e300; /* "infinitely stiff" until computable */
      if (pba->has_acc == _TRUE_) {
        int    n_acc          = pba->N_ncdm-1;
        double rho_ncdm_bg    = ppw->pvecback[pba->index_bg_rho_ncdm1 + n_acc];
        if (rho_ncdm_bg > 0.) {
          double rho_acc_cdm_bg = ppw->pvecback[pba->index_bg_rho_acc_cdm];
          double p_ncdm_bg      = ppw->pvecback[pba->index_bg_p_ncdm1 + n_acc];
          double pseudo_p_ncdm  = ppw->pvecback[pba->index_bg_pseudo_p_ncdm1 + n_acc];
          double Gamma          = ppw->pvecback[pba->index_bg_Gamma_acc];
          double H              = ppw->pvecback[pba->index_bg_H];
          double a_bg           = ppw->pvecback[pba->index_bg_a];
          double eta            = pba->eta_acc;
          double w_ncdm         = p_ncdm_bg/rho_ncdm_bg;
          double ratio_rho      = rho_acc_cdm_bg/rho_ncdm_bg;
          double ca2_0          = w_ncdm*(5.0 - pseudo_p_ncdm/p_ncdm_bg)/(3.0*(1.0+w_ncdm));
          if (ca2_0 < 0.) ca2_0 = 0.;
          double Lambda = a_bg*Gamma*(1.0+eta)*((1.0+ca2_0)/(1.0+w_ncdm))*ratio_rho;
          double rate   = a_bg*H;
          double k_rate = k*sqrt(ca2_0);
          if (k_rate > rate) rate = k_rate;
          ppw->acc_stiff_ratio = (rate > 0.) ? Lambda/rate : 1.e300;
        }
        if (ppw->acc_stiff_ratio > ppr->kappa_stiff) accDM_ready = _FALSE_;
      }
```

(The subsequent `if ((tau/tau_k > ...) && (... != ncdmfa_none) && (accDM_ready == _TRUE_))`
block at lines 6418-6428 is unchanged — it already gates on `accDM_ready`.)

- [ ] **Step 3: Build**

Run: `make`
Expected: compiles cleanly (no undeclared identifier; `index_bg_a` etc. all exist).

- [ ] **Step 4: Verify the run now completes (green)**

Re-run the Step-1 cell unchanged (still `evolver=0`, `ncdm_fluid_approximation='CLASS'`,
old density trigger present but now unused). Add `'kappa_stiff': 1.0` to `base`.
Expected: prints `completed` — no "step too small". The fluid switches on late (when
`Lambda` has dropped below `max(aH,k c_s)`), so `rk` is stable.

- [ ] **Step 5: Commit**

```bash
git add source/perturbations.c
git commit -m "Switch daughter exact->fluid on k-aware stiffness ratio, not density ratio"
```

---

### Task 4: Log the stiffness ratio at fluid switch-on

**Files:**
- Modify: `source/perturbations.c:4976-4977` (the verbose switch-on message in `perturbations_vector_init`)

**Interfaces:**
- Consumes: `ppw->acc_stiff_ratio` (Task 2/3).
- Produces: a per-k stdout line at the actual fluid switch-on carrying the stiffness ratio —
  the raw data for the Phase-2 decision gate.

- [ ] **Step 1: Extend the message**

In `source/perturbations.c`, replace lines 4976-4977:

```c
          if (ppt->perturbations_verbose>2)
            fprintf(stdout,"Mode k=%e: switch on ncdm fluid approximation at tau=%e\n",k,tau);
```

with:

```c
          if (ppt->perturbations_verbose>2)
            fprintf(stdout,"Mode k=%e: switch on ncdm fluid approximation at tau=%e"
                           " (acc stiffness ratio Lambda/max(aH,k*c_s)=%e)\n",
                           k,tau,ppw->acc_stiff_ratio);
```

- [ ] **Step 2: Build**

Run: `make`
Expected: compiles cleanly.

- [ ] **Step 3: Verify the diagnostic prints**

Run the Task-3 Step-4 model with `'perturbations_verbose': 3`. Expected: for each k a line
like `Mode k=... switch on ncdm fluid approximation at tau=... (acc stiffness ratio ...=...)`,
with the ratio at or just below `kappa_stiff`.

- [ ] **Step 4: Commit**

```bash
git add source/perturbations.c
git commit -m "Log daughter stiffness ratio at fluid switch-on for Phase-2 decision gate"
```

---

### Task 5: Instrument the fluid-vs-exact comparison in the notebook

**Files:**
- Modify: `notebooks_test/6_test_fluid_vs_exact.ipynb`

**Interfaces:**
- Consumes: the built CLASS with Tasks 1-4.
- Produces: three recorded metrics for the decision gate — P(k)/Cl_TT divergence vs scale,
  wall-time ratio, and the stiffness ratio at transition per k (parsed from verbose stdout).

- [ ] **Step 1: Add a reusable runner cell**

```python
import time, numpy as np
from classy import Class

def run_model(extra, want_cl=True):
    p = dict(base); p.update(extra)
    cosmo = Class(); cosmo.set(p)
    t0 = time.time(); cosmo.compute(); dt = time.time() - t0
    kk = np.logspace(-4, 0, 300)
    pk = np.array([cosmo.pk(k, 0.) for k in kk])
    cl = cosmo.lensed_cl(2500)['tt'] if want_cl else None
    cosmo.struct_cleanup()
    return {'k': kk, 'pk': pk, 'cl_tt': cl, 'walltime': dt}
```

- [ ] **Step 2: Run exact vs fluid and compute divergence + speedup**

```python
exact = run_model({'ncdm_fluid_approximation': 'none',  'evolver': 0})
fluid = run_model({'ncdm_fluid_approximation': 'CLASS', 'evolver': 0, 'kappa_stiff': 1.0})

dpk = np.abs(fluid['pk']/exact['pk'] - 1.0)
ell = np.arange(exact['cl_tt'].size)
m   = ell >= 2
dcl = np.abs(fluid['cl_tt'][m]/exact['cl_tt'][m] - 1.0)

print(f"max |dPk/Pk|   = {dpk.max():.3e} at k={fluid['k'][dpk.argmax()]:.3e}")
print(f"max |dClTT/Cl| = {dcl.max():.3e}")
print(f"walltime exact={exact['walltime']:.2f}s fluid={fluid['walltime']:.2f}s "
      f"speedup={exact['walltime']/fluid['walltime']:.2f}x")
```

Expected: both runs complete; prints the three numbers. (Phase 1 likely shows modest or no
speedup — that is expected and motivates Phase 2.)

- [ ] **Step 3: Plot divergence vs scale**

```python
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].loglog(fluid['k'], dpk);  ax[0].set_xlabel('k [1/Mpc]'); ax[0].set_ylabel('|dPk/Pk|')
ax[1].loglog(ell[m], dcl);      ax[1].set_xlabel(r'$\ell$');    ax[1].set_ylabel('|dClTT/Cl|')
plt.tight_layout()
```

Expected: a figure showing where fluid departs from exact (feeds the slaving-order decision).

- [ ] **Step 4: Capture stiffness ratio at transition**

```python
import io, contextlib, re
p = dict(base); p.update({'ncdm_fluid_approximation': 'CLASS', 'evolver': 0,
                          'kappa_stiff': 1.0, 'perturbations_verbose': 3})
buf = io.StringIO()
cosmo = Class(); cosmo.set(p)
with contextlib.redirect_stdout(buf):
    cosmo.compute()
cosmo.struct_cleanup()
rows = re.findall(r"Mode k=(\S+).*stiffness ratio.*=(\S+)\)", buf.getvalue())
ks  = np.array([float(a) for a, _ in rows]); rs = np.array([float(b) for _, b in rows])
print(f"transitions logged: {len(rows)};  stiffness ratio range "
      f"[{rs.min():.2e}, {rs.max():.2e}]")
```

Expected: a list of (k, stiffness-ratio) at switch-on. (If `classy` does not surface CLASS
stdout in your environment, run the CLI on a `.ini` with `perturbations_verbose = 3` and
parse the log file instead — note which you used.)

- [ ] **Step 5: Commit**

```bash
git add notebooks_test/6_test_fluid_vs_exact.ipynb
git commit -m "Instrument fluid-vs-exact: P(k)/Cl divergence, speedup, stiffness-at-transition"
```

---

## Decision gate (output of Phase 1, input to Phase 2)

After Task 5, you have:
- the stiffness ratio at the latest `rk`-stable transition per k (Task 4/5 Step 4), and
- the exact-vs-fluid divergence near that transition (Task 5 Steps 2-3).

Use these to set, in the Phase 2 plan: (a) the production `kappa_stiff` default, and (b) the
`acctca` slaving order — 0th-order only if divergence at the transition is within tolerance,
0th+first-order slip if it is not. Phase 2 (the `acctca` tight-coupling regime) is written as
a separate plan once these numbers exist.

## Self-Review

- **Spec coverage:** Phase 1 of the spec — k-aware trigger (Tasks 1-3), instrumentation
  including the `Λ/max(aH,kc_s)` diagnostic (Tasks 2,4,5), keep projection IC + ca2 guards
  (Global Constraints; no code change needed), exact stays the reference (Task 5). Phase 2 is
  explicitly deferred to its own plan per the spec's decision gate. Covered.
- **Placeholder scan:** none — every code step shows full code; the only deferred values
  (`kappa_stiff` production default, slaving order) are the intended decision-gate outputs,
  not implementation placeholders. The user's standard accDM parameter set in `base` is the
  one genuine fill-in, flagged in Task 3 Step 1.
- **Type consistency:** `ppw->acc_stiff_ratio` (double) declared in Task 2, written in Task 3,
  read in Task 4. `ppr->kappa_stiff` (double) declared in Task 1, read in Task 3. `run_model`
  signature/return keys consistent across Task 5 steps.

---

## Revision 2026-06-22b — Task 3 corrected (monotonic gate)

Task 3's k-aware stiffness gate was implemented, tested, and **reverted**: the
stiffness ratio is non-monotonic in tau, and CLASS requires monotonic/
irreversible approximation flags, so it made `ncdmfa` reversible and aborted
("switch 2 approximations at the same time"). The shipped Phase-1 gate is the
**monotonic density ratio** `rho_accDM/rho_acc_cdm` (the original trigger);
`perturbations_acc_stiff_ratio()` survives as the switch-on **diagnostic** only
(Task 4), and `kappa_stiff` (Task 1) is reserved for Phase 2. Tasks 1, 2 (field
later removed in review), 4, 5 stand as built. See the design spec's
"Revision 2026-06-22b" for the full rationale.
