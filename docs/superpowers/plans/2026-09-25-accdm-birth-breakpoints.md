# accDM Birth Breakpoints Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove the per-k-mode P(k) error from accDM daughter births by making every birth time an integration breakpoint.

**Architecture:** Birth times per daughter bin are converted to conformal time once in `perturbations_init`. `perturbations_solve` integrates each approximation interval in sub-intervals cut at those times. Pinning and born weights switch on τ compared with the same values, so the switch happens exactly at a breakpoint.

**Tech Stack:** C (CLASS v3.3 fork), classy, pytest.

**Spec:** `docs/superpowers/specs/2026-09-25-accdm-birth-breakpoints-design.md`

## Global Constraints

- Default behaviour, no flag.
- rk evolver and synchronous gauge only (already enforced for accDM).
- `tol_perturbations_integration = 1e-7` must complete at m_acc = 1e11 GeV, f_acc = 0.1.
- Update after execution: the null-pair test was dropped. Its residual is a physical oscillation aliased by the k sampling (nb31), which this fix does not and should not change; `test_birth_breakpoints.py` keeps only the tolerance test.
- Comments short, describing what the code does.
- Build and test in the user's `accDM` env: `make -j class && make classy`.

## File map

| file | change |
|---|---|
| `include/perturbations.h` | birth-time arrays in `struct perturbations`, `tau_acc` in the workspace, two prototypes |
| `source/perturbations.c` | birth-time functions, NULL init and free, sub-split loop, τ-based pinning and born weight |
| `notebooks_test/test_birth_breakpoints.py` | new tests |

---

### Task 1: Birth-time breakpoints

**Files:**
- Create: `notebooks_test/test_birth_breakpoints.py`
- Modify: `include/perturbations.h` (after `int tau_size;` ~line 369; after `double * pvecback;` ~line 572; after the `perturbations_total_stress_energy` prototype)
- Modify: `source/perturbations.c` (`perturbations_init` ~692/739, `perturbations_free` ~1056, `perturbations_solve` ~3243-3280, `perturbations_vector_init` ~5224, `perturbations_einstein` ~6834, new functions before `perturbations_total_stress_energy` ~7032, sums ~7461, `perturbations_derivs` ~10351)

**Interfaces:**
- Produces: `int perturbations_acc_birth_times(struct background * pba, struct perturbations * ppt)`; `double perturbations_acc_born(struct background * pba, struct perturbations * ppt, int index_q, double tau, double lna)`; `ppt->tau_birth_lo_acc`, `ppt->tau_birth_hi_acc`, `ppt->tau_birth_break`, `ppt->tau_birth_break_size`; `ppw->tau_acc`.

- [ ] **Step 1: Write the failing tests**

Create `notebooks_test/test_birth_breakpoints.py`:

```python
"""accDM daughter births as integration breakpoints: per-k-mode P(k) error.
Run after building classy:  python -m pytest notebooks_test/test_birth_breakpoints.py -v -s
"""
import time
import numpy as np
from classy import Class

KM = np.logspace(np.log10(0.02), np.log10(2.0), 400)    # 1/Mpc
A_REC = 1.0/1091.0


def params(extra=None, f_acc=0.1, mass=1e11, kappa=12.1, a_t=0.133, n_q=51):
    """accDM run on the qm_acc_birth grid, configured as notebook 30.

    C_l output is needed: without it k_step_sub does not shape the k list, so the null
    pair below would compare identical runs."""
    ocdm = 0.12011/(1 + f_acc*(1 - A_REC**kappa)/(1 + (A_REC/a_t)**kappa))
    p = {'omega_b': 0.022383, 'omega_cdm': ocdm, '100*theta_s': 1.041783,
         'A_s': 2.1005829616811546e-9, 'n_s': 0.96605, 'tau_reio': 0.0543, 'N_ur': 0.00441,
         'output': 'tCl,pCl,lCl,mPk', 'lensing': 'yes', 'l_max_scalars': 2500,
         'P_k_max_1/Mpc': 10.0, 'z_max_pk': 0.0,
         'gauge': 'synchronous', 'evolver': 0, 'ncdm_fluid_approximation': 3,
         'vary_Gamma_acc': 'yes', 'kappa_acc': kappa, 'a_t_acc': a_t,
         'f_acc': f_acc, 'eta_acc': 1e11/mass, 'm_acc_in_GeV': mass, 'm_cdm_in_GeV': mass,
         'N_ncdm': 2, 'deg_ncdm': '3, 1', 'm_ncdm': '0.02, {:.6e}'.format(mass*1e9),
         'T_ncdm': '0.71611, 1', 'ncdm_quadrature_strategy': '0, 5',
         'ncdm_N_momentum_bins': '15, {:d}'.format(n_q)}
    p.update(extra or {})
    return p


def pk(p):
    """P(k, z=0) on KM and the run time in seconds."""
    cosmo = Class()
    cosmo.set(p)
    t0 = time.time()
    try:
        cosmo.compute()
        out = np.array([cosmo.pk(k, 0.0) for k in KM])
    finally:
        cosmo.struct_cleanup()
        cosmo.empty()
    return out, time.time() - t0


def residual_max(p1, p2):
    """max |ln(p1/p2) - smooth degree-6 fit in ln k| (notebook 30 metric)."""
    x, y = np.log(KM), np.log(p1/p2)
    return float(np.max(np.abs(y - np.polyval(np.polyfit(x, y, 6), x))))


def test_null_pair_below_1e3():
    """Same physics, k nodes shifted: any residual is per-mode integration error."""
    base, t_base = pk(params())
    shifted, _ = pk(params({'k_step_sub': 0.0505}))
    assert not np.array_equal(base, shifted), 'k_step_sub did not change the k list'
    mx = residual_max(base, shifted)
    print('null pair max residual {:.1e}, one run {:.0f} s'.format(mx, t_base))
    assert mx < 1e-3


def test_tight_tolerance_completes():
    _, t = pk(params({'tol_perturbations_integration': 1e-7}))
    print('tol 1e-7 run {:.0f} s'.format(t))
```

- [ ] **Step 2: Run them on the current build**

Run: `python -m pytest notebooks_test/test_birth_breakpoints.py -v -s`
Expected: both FAIL. Null pair max ≈ 7e-3; tol 1e-7 raises "Step size too small" (C_l output and fixed θ_s, as in nb30). Note the printed run time: it is the baseline for Task 2.

- [ ] **Step 3: Header**

In `include/perturbations.h`, after `int tau_size;             /**< number of values in this array */`:

```c
  double * tau_birth_lo_acc;  /**< accDM daughter: conformal time where each bin starts to be born */
  double * tau_birth_hi_acc;  /**< accDM daughter: conformal time where each bin is fully born */
  double * tau_birth_break;   /**< sorted unique birth times, used as integration breakpoints */
  int tau_birth_break_size;   /**< number of values in tau_birth_break */
```

In `struct perturbations_workspace`, after `double * pvecback;          /**< background quantities */`:

```c
  double tau_acc;             /**< tau of the current Einstein-equation call, for the accDM born weights */
```

After the `perturbations_total_stress_energy` prototype:

```c
  int perturbations_acc_birth_times(
                                    struct background * pba,
                                    struct perturbations * ppt
                                    );

  double perturbations_acc_born(
                                struct background * pba,
                                struct perturbations * ppt,
                                int index_q,
                                double tau,
                                double lna
                                );
```

- [ ] **Step 4: New functions**

In `source/perturbations.c`, directly before the doc comment of `perturbations_total_stress_energy`:

```c
/**
 * Conformal time at ln a, clamped to [0, tau_0] outside the background table.
 */

static int perturbations_acc_tau_of_lna(
                                        struct background * pba,
                                        double lna,
                                        double * tau
                                        ) {
  double z = exp(-lna) - 1.;

  if (z <= pba->z_table[pba->bt_size-1]) {
    *tau = pba->conformal_age;
    return _SUCCESS_;
  }
  if (z >= pba->z_table[0]) {
    *tau = 0.;
    return _SUCCESS_;
  }
  class_call(background_tau_of_z(pba, z, tau),
             pba->error_message,
             pba->error_message);
  return _SUCCESS_;
}

static int perturbations_compare_doubles(const void * x, const void * y) {
  double dx = *(const double *)x;
  double dy = *(const double *)y;
  return (dx > dy) - (dx < dy);
}

/**
 * Birth times of the accDM daughter bins (ramp start and end) and their sorted unique
 * list, used as integration breakpoints so that no rk step straddles a birth.
 */

int perturbations_acc_birth_times(
                                  struct background * pba,
                                  struct perturbations * ppt
                                  ) {
  int n_acc = pba->N_ncdm-1;
  int nq = pba->q_size_ncdm[n_acc];
  int index_q, i, n;
  double * all;

  class_alloc(ppt->tau_birth_lo_acc, nq*sizeof(double), ppt->error_message);
  class_alloc(ppt->tau_birth_hi_acc, nq*sizeof(double), ppt->error_message);
  class_alloc(all, 2*nq*sizeof(double), ppt->error_message);

  for (index_q=0; index_q<nq; index_q++) {
    class_call(perturbations_acc_tau_of_lna(pba, pba->lna_birth_lo_acc[n_acc][index_q],
                                            &(ppt->tau_birth_lo_acc[index_q])),
               pba->error_message,
               ppt->error_message);
    class_call(perturbations_acc_tau_of_lna(pba, pba->lna_birth_hi_acc[n_acc][index_q],
                                            &(ppt->tau_birth_hi_acc[index_q])),
               pba->error_message,
               ppt->error_message);
    all[2*index_q] = ppt->tau_birth_lo_acc[index_q];
    all[2*index_q+1] = ppt->tau_birth_hi_acc[index_q];
  }

  qsort(all, 2*nq, sizeof(double), perturbations_compare_doubles);
  n = 0;
  for (i=0; i<2*nq; i++) {
    if ((n == 0) || (all[i] > all[n-1]))
      all[n++] = all[i];
  }
  ppt->tau_birth_break = all;
  ppt->tau_birth_break_size = n;

  return _SUCCESS_;
}

/**
 * Born weight of daughter bin index_q. Instant births switch just after tau_birth, which
 * is an integration breakpoint; smooth ramps are linear in ln a as in the background.
 */

double perturbations_acc_born(
                              struct background * pba,
                              struct perturbations * ppt,
                              int index_q,
                              double tau,
                              double lna
                              ) {
  if (ppt->tau_birth_lo_acc[index_q] == ppt->tau_birth_hi_acc[index_q])
    return (tau > ppt->tau_birth_hi_acc[index_q]) ? 1. : 0.;
  return background_acc_born_weight(pba, index_q, lna);
}
```

- [ ] **Step 5: Allocate and free**

In `perturbations_init`, directly before `/** - perform preliminary checks */`:

```c
  ppt->tau_birth_lo_acc = NULL;
  ppt->tau_birth_hi_acc = NULL;
  ppt->tau_birth_break = NULL;
  ppt->tau_birth_break_size = 0;
```

After the `class_test` requiring `evolver = 0` for accDM:

```c
  /* accDM: daughter birth times, used as integration breakpoints */
  if (pba->has_acc == _TRUE_) {
    class_call(perturbations_acc_birth_times(pba, ppt),
               ppt->error_message,
               ppt->error_message);
  }
```

In `perturbations_free`, as the first statements inside `if (ppt->has_perturbations == _TRUE_) {`:

```c
    free(ppt->tau_birth_lo_acc);
    free(ppt->tau_birth_hi_acc);
    free(ppt->tau_birth_break);
```

- [ ] **Step 6: Sub-split the integration**

In `perturbations_solve`, replace step (d) — the comment `/** - --> (d) integrate the perturbations over the current interval. */`, the evolver choice, and the whole `class_call_except(generic_evolver(...), ... );` — with:

```c
    /** - --> (d) integrate the perturbations over the current interval, in sub-intervals
        cut at the accDM daughter birth times so that no rk step straddles a birth. */

    if (ppr->evolver == rk){
      generic_evolver = evolver_rk;
    }
    else {
      generic_evolver = evolver_ndf15;
    }

    double tau_start = interval_limit[index_interval];
    while (tau_start < interval_limit[index_interval+1]) {

      double tau_stop = interval_limit[index_interval+1];
      int index_break;
      for (index_break=0; index_break<ppt->tau_birth_break_size; index_break++) {
        if (ppt->tau_birth_break[index_break] > tau_start) {
          tau_stop = MIN(tau_stop, ppt->tau_birth_break[index_break]);
          break;
        }
      }

      class_call_except(generic_evolver(perturbations_derivs,
                                        tau_start,
                                        tau_stop,
                                        ppw->pv->y,
                                        ppw->pv->used_in_sources,
                                        ppw->pv->pt_size,
                                        &ppaw,
                                        ppr->tol_perturbations_integration,
                                        ppr->smallest_allowed_variation,
                                        perturbations_timescale,
                                        ppr->perturbations_integration_stepsize,
                                        ppt->tau_sampling,
                                        tau_actual_size,
                                        perturbations_sources,
                                        perhaps_print_variables,
                                        ppt->error_message),
                        ppt->error_message,
                        ppt->error_message,
                        {
                          ErrorMsg _saved_err;
                          strncpy(_saved_err, ppt->error_message, _ERRORMSGSIZE_-1);
                          _saved_err[_ERRORMSGSIZE_-1] = '\0';
                          snprintf(ppt->error_message, _ERRORMSGSIZE_,
                                   "%s [ca2_ncdm_bad=%d]",
                                   _saved_err,
                                   ppw->ca2_ncdm_bad == _TRUE_ ? 1 : 0);
                        });

      tau_start = tau_stop;
    }
```

- [ ] **Step 7: τ-based born weight and pinning**

In `perturbations_einstein`, directly before `/** - sum up perturbations from all species */`:

```c
  ppw->tau_acc = tau;
```

In `perturbations_total_stress_energy` (~7461) replace
`double born = background_acc_born_weight(pba, index_q, log(a));` with

```c
              double born = perturbations_acc_born(pba, ppt, index_q, ppw->tau_acc, log(a));
```

In `perturbations_vector_init` (~5224) replace
`double born = background_acc_born_weight(pba, index_q, log(a));` with

```c
                double born = perturbations_acc_born(pba, ppt, index_q, tau, log(a));
```

In `perturbations_derivs` (~10351) replace
`if(log(a) <= pba->lna_birth_hi_acc[n_ncdm][index_q]){` with

```c
              if(tau <= ppt->tau_birth_hi_acc[index_q]){
```

(keep the trailing `// AG: ...` comment).

- [ ] **Step 8: Build and run the tests**

Run: `make -j class && make classy && python -m pytest notebooks_test/test_birth_breakpoints.py -v -s`
Expected: 2 passed; null pair max < 1e-3. Report the printed run times.

- [ ] **Step 9: Commit**

```bash
git add include/perturbations.h source/perturbations.c notebooks_test/test_birth_breakpoints.py
git commit -m "Integrate accDM daughter births at exact breakpoints"
```

---

### Task 2: Regression, runtime, golden files

**Files:**
- Modify: `notebooks_test/golden/regression_accDM.json`, `notebooks_test/golden/regression_accDM.npz` (regenerated)
- Modify: `docs/superpowers/specs/2026-09-25-accdm-birth-breakpoints-design.md` (append `## Results`)

- [ ] **Step 1: Existing tests**

Run: `python -m pytest notebooks_test/test_de_sink.py -v`
Expected: 12 passed.

- [ ] **Step 2: Notebook 28 and 30**

Run All on `28_test_de_sink_perturbations.ipynb`: expect meas/pred with the sink on within 5% of 1 at every k.
Run the last cell of `30_pk_noise_diagnostic.ipynb` (after cell 2): expect every accDM null max < 1e-3.

- [ ] **Step 3: Runtime at 501 bins**

Run in Python:

```python
import sys; sys.path.insert(0, 'notebooks_test')
from test_birth_breakpoints import params, pk
_, t = pk(params({'ncdm_quadrature_strategy': '0, 4'}, n_q=501))
print('501-bin qm_simpson_log run: {:.0f} s'.format(t))
```

Compare with the 120 s recorded in SUMMARY.md for 501 bins (P(k) to k = 10, 4 cores). Record it.

- [ ] **Step 4: Golden regression**

Run notebook 1 with `MODE = 'check'` and record which checks fail (a ~1e-3 P(k) change is expected). Then set `MODE = 'generate'`, Run All, set `MODE` back to `'auto'`, and Run All once more: all checks pass.

- [ ] **Step 5: Record results and commit**

Append to the spec:

```markdown
## Results

| check | before | after |
|---|---|---|
| null pair max residual, f_acc = 0.1 (test) | ≈5e-3 | measured value |
| tol 1e-7 run | step-size failure | completes, measured seconds |
| one 51-bin P(k) run | measured seconds | measured seconds |
| one 501-bin qm_simpson_log run | 120 s (SUMMARY.md) | measured seconds |
| nb28 meas/pred, sink on, k = 0.01 / 0.1 / 1 | 1.03 / 1.05 / 1.04 | measured values |
| golden check before regeneration | - | list of failing checks |
```

Replace each "measured" entry with the number, then:

```bash
git add notebooks_test/golden/regression_accDM.json notebooks_test/golden/regression_accDM.npz docs/superpowers/specs/2026-09-25-accdm-birth-breakpoints-design.md
git commit -m "Regenerate golden files after the accDM birth breakpoint fix"
```
