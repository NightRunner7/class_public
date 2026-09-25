# accDM Dark-Energy Sink Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an opt-in w = −1 component (`acc_de_sink = yes`) that pays the accDM daughters' kick energy, restoring background energy conservation.

**Architecture:** ρ_de_acc(a) = η f_acc ρ_cdm,0 J(a) with J(a) = ∫ₐ¹ F′(a′) a′⁻³ d ln a′. J is tabulated once per `background_init` (it depends only on κ, a_t) and read by cubic Hermite lookup in `background_functions`. It is a separate background component; `rho_lambda` and the Ω_Λ closure are untouched because J(1) = 0. Perturbations are unchanged (δρ_de_acc = 0).

**Tech Stack:** C (CLASS v3.3 fork), classy (Cython), pytest + numpy + scipy.

**Spec:** `docs/superpowers/specs/2026-09-25-accdm-de-sink-design.md`

## Global Constraints

- Flag `acc_de_sink`, yes/no, default **no**. With the flag off every output must be bit-identical to the current code.
- `acc_de_sink = yes` without accDM must fail with an error containing `requires accDM`.
- Sink uses the analytic birth law `background_acc_birth_rate`, never the daughter q-grid.
- No change to `rho_lambda`, `Omega0_lambda`, `rho_m`, `rho_plus_p_tot`, or anything in `source/perturbations.c`.
- Branch `accDM_refactor`. The working tree has unrelated modified notebooks: stage only the files named in each task.
- Build and test on a machine with the `accDM` conda env (cluster or WSL); the Windows host has no compiler or Python.
- Comments: short, describe what the code does; no change history.

## Build and test commands (used by every task)

```bash
conda activate accDM
cd software/class_accDM
make -j class && make classy
python -m pytest notebooks_test/test_de_sink.py -v
```

## File map

| file | change |
|---|---|
| `include/background.h` | struct fields, index, two prototypes, two table constants |
| `source/input.c` | read `acc_de_sink`, guard, defaults |
| `source/background.c` | table init/lookup, density in `background_functions`, index, output column, free |
| `notebooks_test/test_de_sink.py` | new pytest module (all tests) |
| `docs/superpowers/specs/2026-09-25-accdm-de-sink-design.md` | results section (Task 4) |

---

### Task 1: Input flag and guard

**Files:**
- Modify: `include/background.h` (struct, next to `short has_acc;` at line 339)
- Modify: `source/input.c` (after the accDM block closing at line 2712; defaults near line 6289)
- Create: `notebooks_test/test_de_sink.py`

**Interfaces:**
- Produces: `short pba->has_acc_de_sink` (`_TRUE_` only if accDM is on and the user set `acc_de_sink = yes`).
- Produces (tests): helpers `accdm_params(...)`, `run(params)` in `test_de_sink.py`, used by all later tasks.

- [ ] **Step 1: Write the test module with the two Task-1 tests**

Create `notebooks_test/test_de_sink.py`:

```python
"""Background tests for the accDM dark-energy sink (acc_de_sink).
Run after building classy:  python -m pytest notebooks_test/test_de_sink.py -v
"""
import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import beta, betainc
from classy import Class

OMEGA_B = 0.022383
OMEGA_CDM0 = 0.12011
H0 = 67.32
N_LOGA = 10001


def accdm_params(f_acc=0.1, eta=0.1, kappa=12.1, a_t=0.133, mass=1e16,
                 n_q=501, strategy=4, sink=None):
    """Background-only accDM run; sink=None leaves acc_de_sink unset."""
    p = {'omega_b': OMEGA_B, 'omega_cdm': OMEGA_CDM0, 'H0': H0,
         'vary_Gamma_acc': 'yes', 'kappa_acc': kappa, 'a_t_acc': a_t,
         'f_acc': f_acc, 'eta_acc': eta,
         'm_acc_in_GeV': mass, 'm_cdm_in_GeV': mass,
         'N_ncdm': 2, 'deg_ncdm': '3, 1',
         'm_ncdm': '0.02, {:.6e}'.format(mass*1e9),
         'T_ncdm': '0.71611, 1',
         'ncdm_quadrature_strategy': '0, {:d}'.format(strategy),
         'ncdm_N_momentum_bins': '15, {:d}'.format(n_q),
         'N_ur': 0.00441, 'background_Nloga': N_LOGA}
    if sink is not None:
        p['acc_de_sink'] = sink
    return p


def run(params):
    """Background table sorted by increasing a, plus 'a' and 'Omega_Lambda'."""
    cosmo = Class()
    cosmo.set(params)
    try:
        cosmo.compute()
        bg = cosmo.get_background()
        omega_lambda = cosmo.Omega_Lambda()
    finally:
        cosmo.struct_cleanup()
        cosmo.empty()
    a = 1.0/(1.0 + np.asarray(bg['z']))
    order = np.argsort(a)
    out = {k: np.asarray(v)[order] for k, v in bg.items()}
    out['a'] = a[order]
    out['Omega_Lambda'] = omega_lambda
    return out


def test_flag_off_is_default():
    unset = run(accdm_params())
    off = run(accdm_params(sink='no'))
    assert '(.)rho_de_acc' not in unset
    for key in ('H [1/Mpc]', '(.)rho_tot', '(.)p_tot', '(.)p_tot_prime'):
        np.testing.assert_array_equal(unset[key], off[key])


def test_sink_requires_accdm():
    p = {'omega_b': OMEGA_B, 'omega_cdm': OMEGA_CDM0, 'H0': H0, 'acc_de_sink': 'yes'}
    with pytest.raises(Exception, match='requires accDM'):
        run(p)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest notebooks_test/test_de_sink.py -v`
Expected: both FAIL. CLASS rejects the unread parameter `acc_de_sink` (the message says it was not read), so `test_flag_off_is_default` errors and `test_sink_requires_accdm` fails its `match`.

- [ ] **Step 3: Add the struct field**

In `include/background.h`, directly below `short has_acc;      /**< presence of accelerating dark matter? */` (line 339):

```c
  short has_acc_de_sink; /**< accDM daughters' kick energy drained from a w=-1 component? */
```

- [ ] **Step 4: Set defaults**

In `source/input.c`, in the `/* START Accelerating DM */` defaults block (line ~6288, right after `pba->eta_acc = 0.;`):

```c
  pba->has_acc = _FALSE_;
  pba->has_acc_de_sink = _FALSE_;
```

- [ ] **Step 5: Read the flag and guard it**

In `source/input.c`, immediately after the closing `}` of the accDM block (the line after `class_read_flag("switch_off_shear_acc", ppt->switch_off_shear_acc);`, line 2712) and before `/** 5) Non-cold relics (ncdm) */`:

```c
  /* accDM DE sink: w=-1 component that pays the daughters' kick energy */
  class_read_flag("acc_de_sink", pba->has_acc_de_sink);
  class_test((pba->has_acc_de_sink == _TRUE_) && (pba->has_acc == _FALSE_),
             errmsg,
             "'acc_de_sink = yes' requires accDM (set f_acc and m_acc_in_GeV).");
```

- [ ] **Step 6: Build and run the tests**

Run: `make -j class && make classy && python -m pytest notebooks_test/test_de_sink.py -v`
Expected: 2 passed.

- [ ] **Step 7: Commit**

```bash
git add include/background.h source/input.c notebooks_test/test_de_sink.py
git commit -m "Add acc_de_sink input flag for the accDM dark-energy sink"
```

---

### Task 2: J(a) table and the ρ_de_acc background component

**Files:**
- Modify: `include/background.h` (struct fields near line 102; index near line 211; prototypes near line 516; constants near line 711)
- Modify: `source/background.c` (new functions after `background_acc_birth_rate` ~line 1312; `background_functions` after the Lambda block ~line 554; `background_init` ~line 855; `background_free_noinput` ~line 937; `background_indices` line 1149; output titles line 2795; output data line 2873)
- Test: `notebooks_test/test_de_sink.py`

**Interfaces:**
- Consumes: `pba->has_acc_de_sink` (Task 1); `double background_acc_birth_rate(struct background *pba, double a)` (existing, = dF/d ln a).
- Produces: `int background_acc_de_sink_init(struct precision *ppr, struct background *pba)`; `double background_acc_de_sink_J(struct background *pba, double a)` (0 for a ≥ 1); `pba->index_bg_rho_de_acc`; output column `(.)rho_de_acc`.

- [ ] **Step 1: Add the failing tests**

Append to `notebooks_test/test_de_sink.py`:

```python
def birth_rate(a, kappa, a_t):
    """dF/dln a, same law as background_acc_birth_rate."""
    x, y = a**kappa, (a/a_t)**kappa
    return kappa*(x + y)/(1.0 + y)**2


def J_closed(a, kappa, a_t):
    """J(a) = int_a^1 F'(a') a'^-3 dln a' via the incomplete beta function (kappa > 3)."""
    al, be = 1.0 - 3.0/kappa, 1.0 + 3.0/kappa
    t = lambda x: (x/a_t)**kappa/(1.0 + (x/a_t)**kappa)
    B = lambda x: betainc(al, be, t(x))*beta(al, be)
    a = np.asarray(a, dtype=float)
    return np.where(a < 1.0, (1.0 + a_t**kappa)*a_t**-3*(B(1.0) - B(np.minimum(a, 1.0))), 0.0)


def J_quad(a, kappa, a_t):
    """Same J by direct quadrature; valid for any kappa."""
    f = lambda l: birth_rate(np.exp(l), kappa, a_t)*np.exp(-3.0*l)
    out = []
    for ai in a:
        if ai >= 1.0:
            out.append(0.0)
            continue
        pts = [np.log(a_t)] if ai < a_t else None
        out.append(quad(f, np.log(ai), 0.0, points=pts, limit=500, epsabs=0.0, epsrel=1e-11)[0])
    return np.array(out)


@pytest.mark.parametrize('kappa,a_t', [(5.0, 0.05), (5.0, 0.133), (12.1, 0.05),
                                       (12.1, 0.133), (2.0, 0.133)])
def test_rho_de_acc_matches_analytic(kappa, a_t):
    eta, f_acc = 0.1, 0.1
    bg = run(accdm_params(f_acc=f_acc, eta=eta, kappa=kappa, a_t=a_t, sink='yes'))
    a = bg['a'][::50]
    rho = bg['(.)rho_de_acc'][::50]
    norm = eta*f_acc*bg['(.)rho_cdm'][-1]
    J = J_closed(a, kappa, a_t) if kappa > 3.0 else J_quad(a, kappa, a_t)
    np.testing.assert_allclose(rho, norm*J, rtol=1e-6, atol=1e-9*norm)


@pytest.mark.parametrize('mass,f_acc', [(1e16, 0.1), (1e11, 0.3)])
def test_today_unchanged(mass, f_acc):
    eta = 1e11/mass
    off = run(accdm_params(f_acc=f_acc, eta=eta, mass=mass))
    on = run(accdm_params(f_acc=f_acc, eta=eta, mass=mass, sink='yes'))
    assert abs(on['(.)rho_de_acc'][-1]) <= 1e-12*on['(.)rho_crit'][-1]
    assert on['(.)rho_de_acc'][0] > 0.0
    assert on['Omega_Lambda'] == pytest.approx(off['Omega_Lambda'], rel=1e-10)
    assert on['H [1/Mpc]'][-1] == pytest.approx(off['H [1/Mpc]'][-1], rel=1e-10)


def test_p_tot_prime_includes_sink():
    """Other components' p(a) do not depend on H, so on-minus-off isolates the sink."""
    off = run(accdm_params())
    on = run(accdm_params(sink='yes'))
    lna = np.log(on['a'])
    dp_on = on['(.)p_tot_prime']/(on['a']*on['H [1/Mpc]'])
    dp_off = off['(.)p_tot_prime']/(off['a']*off['H [1/Mpc]'])
    fd = np.gradient(-on['(.)rho_de_acc'], lna)
    sel = on['a'] > 1e-3
    scale = np.max(np.abs(fd[sel]))
    np.testing.assert_allclose((dp_on - dp_off)[sel], fd[sel], rtol=0, atol=1e-3*scale)
```

- [ ] **Step 2: Run to verify they fail**

Run: `python -m pytest notebooks_test/test_de_sink.py -v`
Expected: the 8 new cases FAIL with `KeyError: '(.)rho_de_acc'`; the 2 Task-1 tests still pass.

- [ ] **Step 3: Header additions**

In `include/background.h`, after the `cfs_acc` field block (line ~100), add:

```c
  int acc_de_sink_n;          /**< intervals of the acc_de_sink J(a) table */
  double acc_de_sink_lna_min; /**< ln a of the first J(a) node */
  double acc_de_sink_dlna;    /**< uniform ln a step of the J(a) table */
  double * acc_de_sink_J;     /**< J(a) = int_a^1 F'(a') a'^-3 dln a' at the nodes */
  double * acc_de_sink_g;     /**< F'(a) a^-3 at the nodes, = -dJ/dln a */
```

After `int index_bg_rho_acc_cdm;` (line 211):

```c
  int index_bg_rho_de_acc;     /**< accDM dark-energy sink density (w=-1, excess over Lambda) */
```

After the `background_acc_birth_rate` prototype (line ~519):

```c
  int background_acc_de_sink_init(
                                  struct precision *ppr,
                                  struct background *pba
                                  );

  double background_acc_de_sink_J(
                                  struct background *pba,
                                  double a
                                  );
```

Next to the `_PSD_DERIVATIVE_EXP_*` defines (line ~711):

```c
#define _ACC_DE_SINK_N_ 20000        /**< intervals of the acc_de_sink J(a) table */
#define _ACC_DE_SINK_LNA_MIN_ -69.   /**< ln a of the first J(a) node (a ~ 1e-30) */
```

- [ ] **Step 4: Table and lookup functions**

In `source/background.c`, directly after the body of `background_acc_birth_rate` (ends ~line 1312):

```c
/**
 * Tabulate J(a) = int_a^1 F'(a') a'^-3 dln a' on a uniform ln a grid, integrating
 * backward from a = 1 with Simpson per cell. rho_de_acc = eta f_acc rho_cdm,0 J(a).
 */

int background_acc_de_sink_init(
                                struct precision *ppr,
                                struct background *pba
                                ) {
  int n = _ACC_DE_SINK_N_;
  int i;
  double h = -_ACC_DE_SINK_LNA_MIN_/n;
  double a, a_mid, g_mid;

  pba->acc_de_sink_n = n;
  pba->acc_de_sink_lna_min = _ACC_DE_SINK_LNA_MIN_;
  pba->acc_de_sink_dlna = h;

  class_alloc(pba->acc_de_sink_J, (n+1)*sizeof(double), pba->error_message);
  class_alloc(pba->acc_de_sink_g, (n+1)*sizeof(double), pba->error_message);

  for (i=0; i<=n; i++) {
    a = exp(pba->acc_de_sink_lna_min + i*h);
    pba->acc_de_sink_g[i] = background_acc_birth_rate(pba,a)/(a*a*a);
  }

  pba->acc_de_sink_J[n] = 0.;
  for (i=n-1; i>=0; i--) {
    a_mid = exp(pba->acc_de_sink_lna_min + (i+0.5)*h);
    g_mid = background_acc_birth_rate(pba,a_mid)/(a_mid*a_mid*a_mid);
    pba->acc_de_sink_J[i] = pba->acc_de_sink_J[i+1]
      + h/6.*(pba->acc_de_sink_g[i] + 4.*g_mid + pba->acc_de_sink_g[i+1]);
  }

  if (pba->background_verbose > 1) {
    printf(" -> acc_de_sink: Omega_de_acc at a_ini = %e (in units of today's rho_crit)\n",
           pba->eta_acc*pba->f_acc*pba->Omega0_cdm
           *background_acc_de_sink_J(pba, ppr->a_ini_over_a_today_default));
  }

  return _SUCCESS_;
}

/**
 * J(a) by cubic Hermite interpolation in ln a with the exact slope dJ/dln a = -F' a^-3.
 * Zero for a >= 1. The caller must keep ln a >= acc_de_sink_lna_min.
 */

double background_acc_de_sink_J(
                                struct background *pba,
                                double a
                                ) {
  double h = pba->acc_de_sink_dlna;
  double * J = pba->acc_de_sink_J;
  double * g = pba->acc_de_sink_g;
  double x, t, t2, t3;
  int i;

  if (a >= 1.) return 0.;

  x = (log(a) - pba->acc_de_sink_lna_min)/h;
  i = (int)floor(x);
  if (i < 0) i = 0;
  if (i > pba->acc_de_sink_n-1) i = pba->acc_de_sink_n-1;
  t = x - i;
  t2 = t*t;
  t3 = t2*t;

  return (2.*t3-3.*t2+1.)*J[i] - (t3-2.*t2+t)*h*g[i]
    + (-2.*t3+3.*t2)*J[i+1] - (t3-t2)*h*g[i+1];
}
```

- [ ] **Step 5: Build the table in `background_init`**

In `source/background.c` `background_init`, right after the `background_indices` `class_call` and before `background_checks` (line ~858):

```c
  /** - accDM DE sink: tabulate J(a) before anything calls background_functions */
  if (pba->has_acc_de_sink == _TRUE_) {
    class_call(background_acc_de_sink_init(ppr,pba),
               pba->error_message,
               pba->error_message);
  }
```

- [ ] **Step 6: Free the table**

In `background_free_noinput`, before `return _SUCCESS_;`:

```c
  if (pba->has_acc_de_sink == _TRUE_) {
    free(pba->acc_de_sink_J);
    free(pba->acc_de_sink_g);
  }
```

- [ ] **Step 7: Define the index**

In `background_indices`, after `class_define_index(pba->index_bg_rho_acc_cdm,pba->has_acc,index_bg,1);` (line 1149):

```c
  class_define_index(pba->index_bg_rho_de_acc,pba->has_acc_de_sink,index_bg,1);
```

- [ ] **Step 8: Add the component in `background_functions`**

Directly after the `/* Lambda */` block (ends ~line 554):

```c
  /* accDM DE sink: w=-1 component that pays the daughters' kick energy */
  if (pba->has_acc_de_sink == _TRUE_) {
    double norm_de_acc = pba->eta_acc * pba->f_acc * pba->Omega0_cdm * pow(pba->H0,2);
    class_test(log(a) < pba->acc_de_sink_lna_min,
               pba->error_message,
               "a = %e is below the start of the acc_de_sink J(a) table", a);
    pvecback[pba->index_bg_rho_de_acc] = norm_de_acc * background_acc_de_sink_J(pba,a);
    rho_tot += pvecback[pba->index_bg_rho_de_acc];
    p_tot -= pvecback[pba->index_bg_rho_de_acc];
    if (a < 1.)
      dp_dloga += norm_de_acc * background_acc_birth_rate(pba,a)/(a*a*a);
  }
```

- [ ] **Step 9: Output column**

After `class_store_columntitle(titles,"(.)rho_acc_cdm",pba->has_acc);` (line 2795):

```c
  class_store_columntitle(titles,"(.)rho_de_acc",pba->has_acc_de_sink);
```

After `class_store_double(dataptr,pvecback[pba->index_bg_rho_acc_cdm],pba->has_acc,storeidx);` (line 2873):

```c
    class_store_double(dataptr,pvecback[pba->index_bg_rho_de_acc],pba->has_acc_de_sink,storeidx);
```

- [ ] **Step 10: Build and run all tests**

Run: `make -j class && make classy && python -m pytest notebooks_test/test_de_sink.py -v`
Expected: 10 passed. If `test_rho_de_acc_matches_analytic` fails only at κ = 12.1 near a_t, report the max relative error; do not loosen the tolerance without asking.

- [ ] **Step 11: Commit**

```bash
git add include/background.h source/background.c notebooks_test/test_de_sink.py
git commit -m "Add the accDM dark-energy sink as a w=-1 background component"
```

---

### Task 3: Conservation test and flag-off regression

**Files:**
- Test: `notebooks_test/test_de_sink.py`

**Interfaces:**
- Consumes: `accdm_params`, `run` (Task 1); `(.)rho_de_acc` and flag (Task 2).

- [ ] **Step 1: Add the conservation test**

Append to `notebooks_test/test_de_sink.py`:

```python
def cum_integral(y, x):
    """Cumulative trapezoid with Euler-Maclaurin end correction, O(h^4) on a uniform grid."""
    cum = np.concatenate(([0.0], np.cumsum(0.5*(y[1:] + y[:-1])*np.diff(x))))
    h = float(np.median(np.diff(x)))
    dy = np.gradient(y, x)
    return cum - (h*h/12.0)*(dy - dy[0])


def omega_k_eff_today(bg, a_start=1e-4):
    """Spurious curvature today; zero iff rho_tot' = -3H(rho_tot + p_tot) (notebook 21)."""
    sel = bg['a'] >= a_start
    a = bg['a'][sel]
    x = np.log(a)
    rho, p = bg['(.)rho_tot'][sel], bg['(.)p_tot'][sel]
    C = a**2*rho + cum_integral(a**2*(rho + 3.0*p), x)
    return (C[-1] - C[0])/(a[-1]*bg['H [1/Mpc]'][sel][-1])**2


def test_background_conservation_restored():
    off = omega_k_eff_today(run(accdm_params(n_q=51, strategy=5)))
    on51 = omega_k_eff_today(run(accdm_params(n_q=51, strategy=5, sink='yes')))
    on101 = omega_k_eff_today(run(accdm_params(n_q=101, strategy=5, sink='yes')))
    print('Omega_K_eff today: off {:+.3e}  on(51) {:+.3e}  on(101) {:+.3e}'.format(off, on51, on101))
    assert abs(on51) < 0.05*abs(off)
    assert abs(on101) <= abs(on51)
```

- [ ] **Step 2: Run it**

Run: `python -m pytest notebooks_test/test_de_sink.py::test_background_conservation_restored -v -s`
Expected: PASS, with the printed off value ~η·f-sized and both on values much smaller. If it fails, report the three printed numbers and stop; the thresholds encode the spec's claim and must not be tuned silently.

- [ ] **Step 3: Flag-off regression against the golden files**

Run: `python -m pytest --nbmake notebooks_test/1_test_regression_golden.ipynb notebooks_test/3_test_bg_conservation.ipynb`
Expected: same pass/fail status as on commit `d899f3ef` (run it there first if unsure). Any new failure means the flag-off path changed: fix before continuing.

- [ ] **Step 4: Commit**

```bash
git add notebooks_test/test_de_sink.py
git commit -m "Test background energy conservation with the accDM DE sink"
```

---

### Task 4: Audit reruns and results note

**Files:**
- Modify: `docs/superpowers/specs/2026-09-25-accdm-de-sink-design.md` (append `## Results`)

- [ ] **Step 1: Rerun notebook 21 with the sink**

In a scratch copy (do not commit), `notebooks_test/21_test_stress_energy_conservation.ipynb` → set `FIDUCIAL`'s run to `run_background(accdm_params(**FIDUCIAL, extra={'acc_de_sink': 'yes'}))`. Note that `accdm_params` in that notebook takes `eta` via `FIDUCIAL`. Record `max |R_windowed|` and `Omega_K_eff(a=1)` against the flag-off values.

- [ ] **Step 2: Rerun notebook 22 the same way**

Pass `extra={'acc_de_sink': 'yes'}` to every accDM run in the scratch copy. Record max|R̄| per scan point and its decision-table band.

- [ ] **Step 3: Rerun notebook 24 the same way**

Record the perturbation-level drift and meas/pred ratio. Expected: unchanged, since option A leaves δρ_de_acc = 0. This number sizes the option-A residual.

- [ ] **Step 4: Append results to the spec**

Add to the end of the spec:

```markdown
## Results (2026-09-25 implementation)

| check | flag off | flag on |
|---|---|---|
| nb21 max abs R_windowed | <value> | <value> |
| nb21 Omega_K_eff(a=1) | <value> | <value> |
| nb22 max abs R-bar, band | <value> | <value> |
| nb24 perturbation drift, meas/pred | <value> | <value> |

Option B needed: <yes/no, one line of reasoning from the nb24 row>.
```

Replace every `<value>` with the measured number before committing.

- [ ] **Step 5: Commit**

```bash
git add docs/superpowers/specs/2026-09-25-accdm-de-sink-design.md
git commit -m "Record conservation audit results for the accDM DE sink"
```
