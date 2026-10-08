# accDM Born-Fraction Daughter Nodes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let strategy 5 place a share of the daughter's momentum nodes by born fraction, so more nodes fall inside the birth window at unchanged cost.

**Architecture:** A precision parameter `accdm_q_log_share` = s (default 1, today's grid) selects a node map even in u(ln a) = s·(ln a − ln a_min)/L + (1 − s)·(F − F_min)/(1 − F_min). A new `background_acc_q_nodes` in `background.c` builds the nodes and Simpson-in-u weights and replaces `get_qsampling_manual` for the daughter when s < 1. Everything downstream already reads per-node positions.

**Tech Stack:** C (CLASS v3.3 fork), classy, plain-assert test scripts (pytest-compatible), Jupyter notebook executed through `jupyter_client`.

**Spec:** `docs/superpowers/specs/2026-10-05-accdm-born-fraction-nodes-design.md`

## Global Constraints

- Default `accdm_q_log_share = 1` must stay bit-identical to the current grid (existing code path untouched).
- 0 < s ≤ 1; s < 1 with `accdm_smooth_births` is an input error.
- Node count unchanged: `accdm_q_bins_per_decade` × log10(1/a_min) (odd), or explicit `ncdm_N_momentum_bins`.
- First node at a_min, last node exactly at q = P_acc/T_acc (a = 1).
- Comments short, describing what the code does.
- Build in the user's `accDM` env: `make -j8 class`, then classy with
  `CC=gcc /opt/homebrew/Caskroom/miniforge/base/envs/accDM/bin/python -m pip install --no-build-isolation .`
  (`make classy` fails: pip's isolated build env has no Cython).
- Python: `/opt/homebrew/Caskroom/miniforge/base/envs/accDM/bin/python` (below: `$PY`).

## File map

| file | change |
|---|---|
| `include/precisions.h` | `accdm_q_log_share` precision parameter |
| `source/input.c` | range and smooth-births checks in the `qm_acc_birth` setup |
| `include/background.h`, `source/background.c` | `background_acc_q_nodes`, called from `background_ncdm_init` |
| `notebooks_test/test_accdm_born_nodes.py` | new tests |
| `notebooks_test/18_born_fraction_nodes.ipynb` | evaluation against nb37's references |

---

### Task 1: Parameter and input checks

**Files:**
- Create: `notebooks_test/test_accdm_born_nodes.py`
- Modify: `include/precisions.h` (after `accdm_smooth_births`, ~line 87)
- Modify: `source/input.c` (inside `if (pba->ncdm_quadrature_strategy[idx_acc] == qm_acc_birth)`, ~line 3006)

**Interfaces:**
- Produces: `ppr->accdm_q_log_share` (double, default 1.0); error texts `'accdm_q_log_share' must lie in (0,1]` and `cannot be combined with 'accdm_smooth_births'`.

- [ ] **Step 1: Write the failing tests**

`notebooks_test/test_accdm_born_nodes.py`:

```python
"""accdm_q_log_share: daughter nodes placed partly by born fraction (strategy 5).
Run after building classy:  python notebooks_test/test_accdm_born_nodes.py
"""
import sys

import numpy as np
from classy import Class

from test_accdm_input_checks import accdm_params, expect_error

# chain settings (connect/new): late, fairly sharp transition and a large daughter share
CHAIN = dict(kappa_acc=12.1, a_t_acc=0.133, f_acc=1.0)


def run(params, level, extract):
    cosmo = Class()
    cosmo.set(params)
    try:
        cosmo.compute(level)
        return extract(cosmo)
    finally:
        cosmo.struct_cleanup()
        cosmo.empty()


def test_share_out_of_range_is_rejected():
    for share in (0.0, -0.5, 1.5):
        expect_error(accdm_params(**CHAIN, accdm_q_log_share=share), "must lie in (0,1]")


def test_share_with_smooth_births_is_rejected():
    expect_error(accdm_params(**CHAIN, accdm_q_log_share=0.4, accdm_smooth_births=1),
                 "cannot be combined with 'accdm_smooth_births'")


def test_share_one_is_bit_identical_to_the_default():
    cl = lambda c: c.lensed_cl(1000)
    p = dict(output="tCl,pCl,lCl", lensing="yes", l_max_scalars=1000)
    a = run(accdm_params(**CHAIN, **p), ["lensing"], cl)
    b = run(accdm_params(**CHAIN, **p, accdm_q_log_share=1.0), ["lensing"], cl)
    for s in ("tt", "ee", "te", "pp"):
        assert np.array_equal(a[s], b[s]), s


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    failed = 0
    for test in tests:
        try:
            test()
            print("PASS", test.__name__)
        except AssertionError as error:
            failed += 1
            print("FAIL", test.__name__, "--", error)
        except Exception as error:
            failed += 1
            print("ERROR", test.__name__, "--", type(error).__name__, error)
    sys.exit(1 if failed else 0)
```

- [ ] **Step 2: Run, verify they fail**

Run: `cd notebooks_test && $PY test_accdm_born_nodes.py`
Expected: all three FAIL/ERROR (CLASS does not read `accdm_q_log_share`).

- [ ] **Step 3: Add the parameter** (`include/precisions.h`, after `accdm_smooth_births`)

```c
/**
 * accDM daughter with ncdm_quadrature_strategy = 5: share of the nodes spread
 * evenly in ln a_q; the rest follow the born fraction F(a_q), which puts more
 * nodes inside the birth window. 1 is the even ln a_q grid.
 */
class_precision_parameter(accdm_q_log_share,double,1.)
```

- [ ] **Step 4: Add the checks** (`source/input.c`, first lines inside the `qm_acc_birth` branch, before `background_acc_a_min`)

```c
          class_test((ppr->accdm_q_log_share <= 0.) || (ppr->accdm_q_log_share > 1.),
                     errmsg,
                     "'accdm_q_log_share' must lie in (0,1], got %g.", ppr->accdm_q_log_share);
          class_test((ppr->accdm_q_log_share < 1.) && (ppr->accdm_smooth_births == _TRUE_),
                     errmsg,
                     "'accdm_q_log_share' < 1 cannot be combined with 'accdm_smooth_births': smooth births need cells even in ln a.");
```

- [ ] **Step 5: Build and run the tests**

Run: `make -j8 class && CC=gcc $PY -m pip install --no-build-isolation . && cd notebooks_test && $PY test_accdm_born_nodes.py`
Expected: three PASS (s < 1 with valid range is accepted but does nothing yet).

- [ ] **Step 6: Commit**

```bash
git add include/precisions.h source/input.c notebooks_test/test_accdm_born_nodes.py
git commit -m "Add accdm_q_log_share precision parameter and its input checks"
```

---

### Task 2: Born-fraction nodes

**Files:**
- Modify: `include/background.h` (prototype after `background_acc_a_min`)
- Modify: `source/background.c` (new function after `background_acc_a_min`, ~line 1460; call in the manual-sampling branch of `background_ncdm_init`, ~line 1811)
- Test: `notebooks_test/test_accdm_born_nodes.py`

**Interfaces:**
- Consumes: `ppr->accdm_q_log_share` (Task 1); `background_acc_born_fraction(pba, a)`, `background_acc_birth_rate(pba, a)`, `background_ncdm_distribution(void *, double q, double *f0)`, `pba->a_min_acc`, `pba->P_acc`, `pba->T_acc_GeV`.
- Produces: `int background_acc_q_nodes(struct background *pba, double share, double *q, double *w, int N, void *params_for_distribution)`.

- [ ] **Step 1: Add the failing tests** (append to `test_accdm_born_nodes.py`, above `if __name__`)

```python
def daughter_omega(params):
    def extract(c):
        bg = c.get_background()
        return bg["(.)rho_ncdm[1]"][-1]/bg["(.)rho_crit"][-1]
    return run(params, ["background"], extract)


def test_born_nodes_keep_the_daughter_abundance():
    ref = daughter_omega(accdm_params(**CHAIN))
    new = daughter_omega(accdm_params(**CHAIN, accdm_q_log_share=0.4))
    assert abs(new/ref - 1) < 1e-5, new/ref - 1


def test_born_nodes_move_the_perturbations_slightly():
    s8 = lambda c: c.sigma8()
    p = dict(output="mPk", **CHAIN)
    ref = run(accdm_params(**p), ["fourier"], s8)
    new = run(accdm_params(**p, accdm_q_log_share=0.4), ["fourier"], s8)
    assert 1e-7 < abs(new/ref - 1) < 1e-2, new/ref - 1
```

- [ ] **Step 2: Run, verify the second fails**

Run: `cd notebooks_test && $PY test_accdm_born_nodes.py`
Expected: `test_born_nodes_move_the_perturbations_slightly` FAILS (s is ignored, difference 0); the abundance test passes trivially.

- [ ] **Step 3: Prototype** (`include/background.h`, after `background_acc_a_min`)

```c
  int background_acc_q_nodes(
                             struct background *pba,
                             double share,
                             double * q,
                             double * w,
                             int N,
                             void * params_for_distribution
                             );
```

- [ ] **Step 4: Implementation** (`source/background.c`, after `background_acc_a_min`)

```c
/**
 * Daughter nodes and weights for ncdm_quadrature_strategy = 5 with
 * accdm_q_log_share = share < 1. Nodes are even in
 * u(ln a) = share (ln a - ln a_min)/L + (1-share) (F(a) - F_min)/(1 - F_min), L = ln(1/a_min),
 * so a share 1-share of them follows the born fraction F; weights are Simpson in u.
 * q = a P_acc/T_acc, with the first node at a_min and the last at a = 1.
 */

int background_acc_q_nodes(
                           struct background *pba,
                           double share,
                           double * q,
                           double * w,
                           int N,
                           void * params_for_distribution
                           ) {
  double lna_min = log(pba->a_min_acc);
  double L = -lna_min;
  double F_min = background_acc_born_fraction(pba, pba->a_min_acc);
  double q_of_a = pba->P_acc/pba->T_acc_GeV;
  double h_u = 1./(N-1);
  double lo, hi, mid, u_mid, a, dlna_du, simpson_weight, f0;
  int i, iter;

  class_test((N < 3) || (N % 2 == 0), pba->error_message,
             "Simpson quadrature needs an odd number of momentum bins >= 3, got %d.", N);

  for (i=0; i<N; i++) {
    /* ln a_i solves u(ln a_i) = i h_u; u is monotone in ln a */
    if (i == 0) {
      a = pba->a_min_acc;
    }
    else if (i == N-1) {
      a = 1.;
    }
    else {
      lo = lna_min;
      hi = 0.;
      for (iter=0; (iter<200) && (hi-lo > 1.e-14); iter++) {
        mid = 0.5*(lo+hi);
        u_mid = share*(mid-lna_min)/L
          + (1.-share)*(background_acc_born_fraction(pba, exp(mid))-F_min)/(1.-F_min);
        if (u_mid < i*h_u)
          lo = mid;
        else
          hi = mid;
      }
      a = exp(0.5*(lo+hi));
    }
    q[i] = (i == N-1) ? q_of_a : a*q_of_a;

    dlna_du = 1./(share/L + (1.-share)*background_acc_birth_rate(pba, a)/(1.-F_min));
    simpson_weight = ((i == 0) || (i == N-1)) ? 1./3. : ((i % 2 == 1) ? 4./3. : 2./3.);
    class_call(background_ncdm_distribution(params_for_distribution, q[i], &f0),
               pba->error_message,
               pba->error_message);
    w[i] = f0*q[i]*dlna_du*h_u*simpson_weight;
  }

  return _SUCCESS_;
}
```

- [ ] **Step 5: Call it** (`source/background.c`, manual-sampling branch of `background_ncdm_init`: wrap the existing `get_qsampling_manual` call)

```c
      if ((pba->ncdm_quadrature_strategy[k] == qm_acc_birth) && (ppr->accdm_q_log_share < 1.)) {
        /* daughter nodes partly placed by born fraction */
        class_call(background_acc_q_nodes(pba,
                                          ppr->accdm_q_log_share,
                                          pba->q_ncdm[k],
                                          pba->w_ncdm[k],
                                          pba->q_size_ncdm[k],
                                          &pbadist),
                   pba->error_message,
                   pba->error_message);
      }
      else {
        class_call(get_qsampling_manual(...existing arguments unchanged...),
                   pba->error_message,
                   pba->error_message);
      }
```

- [ ] **Step 6: Build and run all accDM tests**

Run: `make -j8 class && CC=gcc $PY -m pip install --no-build-isolation . && cd notebooks_test && $PY test_accdm_born_nodes.py && $PY test_accdm_input_checks.py && $PY test_birth_breakpoints.py && $PY test_de_sink.py`
Expected: all PASS.

- [ ] **Step 7: Commit**

```bash
git add include/background.h source/background.c notebooks_test/test_accdm_born_nodes.py
git commit -m "Place accDM daughter nodes partly by born fraction when accdm_q_log_share < 1"
```

---

### Task 3: Notebook 38, evaluation

**Files:**
- Create: `notebooks_test/18_born_fraction_nodes.ipynb` (generated, executed with `jupyter_client`; cache `38_cache.pkl`, not committed)

**Interfaces:**
- Consumes: `accdm_q_log_share` (Tasks 1–2); `notebooks_test/17_cache.pkl` (nb37 runs keyed by `json.dumps(params, sort_keys=True)` with params built exactly as nb37's `params(log10m, f_tilde, **per_decade(n))`).

Cells (code copied from nb37 where marked, so the cache keys match):

1. **Intro (markdown):** the problem (nb37 sections 0–1), the map and s, the success criteria from the spec.
2. **Setup:** nb37's setup cell verbatim (imports, `find_root`, `read_bestfit`, `show`, `CHAINS`, `BF`, `LCDM`, `OMEGA_DM_TOT`, `KAPPA`, `A_T`, `SETTINGS`, `per_decade`, `BINS`, `REF`, `F_NULL`, `chain_f95`, `F95`), plus `SHARES = [0.6, 0.4, 0.25]`.
3. **Section 0, nodes in the window (no CLASS):** for log and each s, q_size and the nodes with 0.01 < F < 0.99 and 0.1 < F < 0.9, from the same map in numpy (`born_fraction`, `a_min` as nb37 section 0; nodes from interpolating u on a 100001-point ln a grid).
4. **Section 1 machinery:** nb37's run/cache/Δχ² cell verbatim with `CACHE = Path('38_cache.pkl')`, plus `cache37 = pickle.loads(Path('37_cache.pkl').read_bytes())` and
   `def lookup(p): key = json.dumps(p, sort_keys=True); return cache37[key] if key in cache37 else cached_run(p)`.
5. **Runs:** for MASSES = [11, 12, 14, 16, 18], F_GRID = [0.01, 0.1, 0.5, 0.9]: references `lookup(params(lm, ft, **per_decade(400)))`, log grids `lookup(params(lm, ft, **per_decade(n)))` for n in (50, 100), and `cached_run(params(lm, ft, accdm_q_log_share=s, **per_decade(n)))` for s in SHARES, n in (50, 100). Null `lookup(params(14, F_NULL))`.
6. **Section 1 table:** per (mass, f̃, n): Δχ² vs reference for log and each s, the improvement factor log/s, max |ΔP_cb/P_cb|, Δσ8_cb, runtime ratio s/log.
7. **Section 1 plot:** Δχ²(n = 50) vs f̃ per mass, log (solid) and each s (dashed), tolerance 0.1, chain 95% f̃ stars as nb37.
8. **Section 2, verdict:** for each s, at n = 50: (a) min improvement factor over points where log Δχ² > 0.1 (criterion ≥ 10), (b) number of points where s is worse than log by more than the 0.04 floor, (c) median runtime ratio (criterion ≤ 1.1). Print PASS/FAIL per criterion.
9. **Reading the result (markdown):** filled in after running, with the recommended default s.

- [ ] **Step 1: Write the generator** `make_nb38.py` (scratchpad) producing the cells above, following `make_nb37.py`.
- [ ] **Step 2: Execute** with the `jupyter_client` runner in the `accDM` env (background; ~30–40 min, 120 new runs).
- [ ] **Step 3: Read the outputs, check figures, write "Reading the result".**
- [ ] **Step 4: Commit**

```bash
git add notebooks_test/18_born_fraction_nodes.ipynb
git commit -m "Add notebook 38: born-fraction daughter nodes against nb37 references"
```
