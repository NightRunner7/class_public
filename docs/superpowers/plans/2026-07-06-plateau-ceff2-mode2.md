# Plateau ceff2 Closure (`ncdm_ceff2_mode = 2`) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the nb15-measured plateau closure `ceff2(a) = min(max(ca2(a), c_fs), 1/3)` with `c_fs = (1/3)(1 - exp(-3*A*ca2_bg(a=1)))` as `ncdm_ceff2_mode = 2` in the C code, and validate it on P(k) restricted to k <= 1 Mpc^-1 at f = 0.3 (notebook 16).

**Architecture:** One new precision parameter (`ncdm_ceff2_fs_A`), one new background-struct double (`pba->cfs_acc`) computed once in `background_init` from the last background-table row, and one new early-return branch in `perturbations_ceff2_ncdm`. Modes 0/1 stay bit-identical. Python mirror (`saturating_cfs`) lives in the tested helpers module; notebook 16 is the decisive P(k) validation with a fluid-trigger scan.

**Tech Stack:** C (CLASS fork `class_accDM`), Python/numpy/classy/pytest, Jupyter.

**Spec:** `docs/superpowers/specs/2026-07-06-plateau-ceff2-mode2-design.md`.

## Global Constraints

- Edit `class_accDM`, never the pristine reference (`memory: apply-fixes-to-working-branch`).
- No compiler or Python in the agent shell (`memory: build-environment`, `memory: agent-shell-no-python`): the agent authors code; the **user** runs `make`, pytest, and notebooks. Build/run steps below are marked **(user)**.
- Daughter = last ncdm species, `has_acc`-gated (`memory: last-species-must-be-has-acc-gated`).
- Modes 0/1 must remain bit-identical: mode-2 branch returns **before** the mode-0/1 fit is computed.
- The closure covers only fixed `(kappa, a_t)` with completed production; `A` default 13.0 is the kappa=6, a_t=0.13 family value (nb15). Scans varying `(kappa, a_t)` stay on exact + q(f) (nb14).
- Wording: the 1/3 cap is the **relativistic free-gas ceiling** (radiation sound speed c/sqrt(3)), not a strict causality bound — use this phrasing in all new/edited comments.
- Plot style: STIX serif + ColorBrewer (`memory: notebook-plot-style`). `w_sigma`/`w_theta` are zero in exact runs (`memory: w-sigma-zero-in-exact-hierarchy`) — do not use them.

---

### Task 1: Python mirror `saturating_cfs` (TDD, no CLASS)

**Files:**
- Modify: `notebooks_test/fluid_closure_helpers.py`
- Test: `notebooks_test/test_fluid_closure_helpers.py`

**Interfaces:**
- Consumes: numpy only.
- Produces: `saturating_cfs(ca2_today, A=13.0) -> np.ndarray` — the mode-2 plateau `(1/3)(1 - exp(-3*A*ca2_today))`, elementwise, clipped to `ca2_today >= 0`. Notebook 16 uses it to predict the C value.

- [ ] **Step 1: Add the failing tests** (append to `test_fluid_closure_helpers.py`; extend the import line with `saturating_cfs`)

```python
def test_saturating_cfs_small_argument_is_linear():
    # small A*ca2: c_fs ~ A*ca2 (unsaturated regime)
    assert np.isclose(saturating_cfs(1e-5, A=13.0), 13.0e-5, rtol=1e-3)

def test_saturating_cfs_saturates_below_one_third():
    # large A*ca2: c_fs -> 1/3 from below (relativistic free-gas ceiling)
    c = saturating_cfs(1.0, A=13.0)
    assert c < 1./3. and np.isclose(c, 1./3., atol=1e-6)

def test_saturating_cfs_matches_nb15_eta1_break():
    # nb15: eta=1 measured c_fs=0.2277 at ca2_today=3.053e-2; the map with
    # A~12.5 should land near it (within ~15%)
    assert abs(saturating_cfs(3.053e-2, A=12.5)/0.2277 - 1.0) < 0.15

def test_saturating_cfs_zero_and_negative_ca2_give_zero():
    out = saturating_cfs(np.array([0.0, -1e-3]), A=13.0)
    assert np.allclose(out, 0.0)
```

- [ ] **Step 2 (user): Run tests, verify the new ones fail**

Run: `cd notebooks_test && python -m pytest test_fluid_closure_helpers.py -v`
Expected: existing tests PASS; the 4 new ones FAIL with `ImportError`/`NameError: saturating_cfs`.

- [ ] **Step 3: Implement** (append to `fluid_closure_helpers.py`)

```python
def saturating_cfs(ca2_today, A=13.0):
    """Mode-2 plateau: c_fs = (1/3)(1 - exp(-3*A*ca2_today)).

    Saturates at the relativistic free-gas ceiling 1/3 (radiation sound
    speed c/sqrt(3)); linear ~A*ca2 when unsaturated. Mirrors the C
    computation of pba->cfs_acc in background_init - keep in sync."""
    ca2 = np.clip(np.asarray(ca2_today, float), 0.0, None)
    return (1.0/3.0)*(1.0 - np.exp(-3.0*A*ca2))
```

- [ ] **Step 4 (user): Run tests, verify all pass**

Run: `cd notebooks_test && python -m pytest test_fluid_closure_helpers.py -v`
Expected: all PASS (15 tests).

- [ ] **Step 5: Commit**

```bash
git add notebooks_test/fluid_closure_helpers.py notebooks_test/test_fluid_closure_helpers.py
git commit -m "feat: saturating_cfs python mirror of mode-2 plateau (TDD)"
```

---

### Task 2: C input parameter + background hook (`pba->cfs_acc`)

**Files:**
- Modify: `include/precisions.h:438` (after `ncdm_ceff2_fs_amp`)
- Modify: `include/background.h:94` (after `eps_acc`)
- Modify: `source/background.c:876` (after the `background_check_ca2_ncdm` call in `background_init`)

**Interfaces:**
- Consumes: `pba->background_table` last row; `pba->index_bg_rho_ncdm1/index_bg_p_ncdm1/index_bg_pseudo_p_ncdm1`, `pba->N_ncdm`, `pba->has_acc`; `ppr->ncdm_ceff2_fs_A`.
- Produces: `pba->cfs_acc` (double; 0 unless `has_acc`) — read by Task 3's mode-2 branch.

- [ ] **Step 1: Declare the precision parameter.** In `include/precisions.h`, insert after line 438 (`class_precision_parameter(ncdm_ceff2_fs_amp,double,0.2)`):

```c
/**
 * Mode-2 family constant A in c_fs = (1/3)(1 - exp(-3*A*ca2_bg(a=1))).
 * Measured by notebooks_test/15_test_fluid_closure_diagnostic.ipynb for the
 * kappa=6, a_t=0.13 family (A ~ 13); re-measure with nb15 when fixing a
 * different (kappa, a_t) family. Has no effect unless ncdm_ceff2_mode = 2.
 */
class_precision_parameter(ncdm_ceff2_fs_A,double,13.0)
```

- [ ] **Step 2: Extend the mode docstring.** In the same file, replace the mode-list comment block (lines 421-429) so it reads:

```c
/**
 * accDM daughter free-streaming (effective sound speed) correction mode.
 *   0 = fit:     ceff2 = ca2*(1 + ncdm_ceff2_fs_amp*(1-2*eps_acc)*sqrt(k/k_fs))
 *                (the published Eq-38 form; default, reproduces prior behaviour)
 *   1 = bounded: the same fit, hard-capped at 1/3. Identical to mode 0 below
 *                1/3; a pure safety cap that engages only when the fit would
 *                exceed the ceiling (large base ca2 x large k/k_fs).
 *   2 = plateau: ceff2 = min(max(ca2, cfs_acc), 1/3) with the k-independent
 *                free-streaming plateau cfs_acc = (1/3)(1-exp(-3*A*ca2_bg(a=1)))
 *                measured from the exact hierarchy (notebook 15). Valid for
 *                fixed (kappa_acc, a_t_acc) with production completed early.
 * The 1/3 cap is the relativistic free-gas ceiling (radiation sound speed
 * c/sqrt(3)), not a strict causality bound. Has no effect when has_acc is false.
 */
```

- [ ] **Step 3: Declare the background member.** In `include/background.h`, insert after line 94 (`double eps_acc;`):

```c
  double cfs_acc; /**< mode-2 plateau ceff2 of the acc daughter,
                       (1/3)(1-exp(-3*A*ca2_bg(a=1))); set in background_init
                       from the last background-table row; 0 unless has_acc */
```

- [ ] **Step 4: Compute it in `background_init`.** In `source/background.c`, insert after the `background_check_ca2_ncdm` call (after line 876, before the `background_output_budget` call):

```c
  /* AccDM: mode-2 plateau ceff2 (spec 2026-07-06). c_fs = (1/3)(1 -
     exp(-3*A*ca2_bg(a=1))) with ca2_bg today from the last background-table
     row (source-free form: production is complete at a=1 in the covered
     regime). Saturates at the relativistic free-gas ceiling 1/3. */
  pba->cfs_acc = 0.;
  if ((pba->has_acc == _TRUE_) && (pba->N_ncdm > 0)) {
    int n_acc = pba->N_ncdm-1;
    double * last_row = pba->background_table + (pba->bt_size-1)*pba->bg_size;
    double rho_d = last_row[pba->index_bg_rho_ncdm1 + n_acc];
    double p_d   = last_row[pba->index_bg_p_ncdm1 + n_acc];
    double pp_d  = last_row[pba->index_bg_pseudo_p_ncdm1 + n_acc];
    if ((rho_d > 0.) && (p_d > 0.)) {
      double w_d = p_d/rho_d;
      double ca2_today = w_d*(5.0 - pp_d/p_d)/(3.0*(1.0+w_d));
      if (ca2_today < 0.) ca2_today = 0.;
      if (ca2_today > 1.) ca2_today = 1.;
      pba->cfs_acc = (1./3.)*(1.0 - exp(-3.0*ppr->ncdm_ceff2_fs_A*ca2_today));
    }
  }
```

- [ ] **Step 5 (user): Build**

Run: `make -j` (user's build environment)
Expected: clean compile, no new warnings.

- [ ] **Step 6: Commit**

```bash
git add include/precisions.h include/background.h source/background.c
git commit -m "feat: ncdm_ceff2_fs_A param + pba->cfs_acc background hook (mode 2)"
```

---

### Task 3: mode-2 branch in `perturbations_ceff2_ncdm` + ceiling rewording

**Files:**
- Modify: `source/perturbations.c:3823-3849` (the helper and its docstring)

**Interfaces:**
- Consumes: `pba->cfs_acc` (Task 2), `ppr->ncdm_ceff2_mode`.
- Produces: mode-2 return path used automatically by both existing call sites (`perturbations.c:7369/7387` source accumulation with `cg2` base; `perturbations.c:10121` fluid derivatives with `ca2` base) — no call-site edits needed.

- [ ] **Step 1: Replace the helper (docstring + body).** Replace lines 3823-3849 of `source/perturbations.c` with:

```c
/**
 * accDM daughter effective sound speed ceff2 (= delta_p/delta_rho) from the
 * base adiabatic sound speed cs2_base (cg2 or ca2). Mode 0 (ncdm_ceff2_mode)
 * is the published Eq-38 fit; mode 1 is the same fit hard-capped at 1/3;
 * mode 2 is the measured plateau closure min(max(cs2_base, cfs_acc), 1/3)
 * (notebook 15): k-independent, adiabatic below the plateau, free-streaming
 * plateau above. The 1/3 cap is the relativistic free-gas ceiling (radiation
 * sound speed c/sqrt(3)), not a strict causality bound. Kinks are harmless:
 * ceff2 enters the perturbation equations algebraically, never
 * differentiated. cs2_base <= 0 returns 0 (degenerate/unborn daughter).
 * Callers invoke it only for n_acc with has_acc.
 */
static double perturbations_ceff2_ncdm(struct precision * ppr,
                                       struct background * pba,
                                       double cs2_base,
                                       double k, double a, double H) {

  if (cs2_base <= 0.) return 0.;

  /* mode 2: measured plateau closure; returns before the fit so modes 0/1
     stay bit-identical. */
  if (ppr->ncdm_ceff2_mode == 2) {
    double ceff2 = (cs2_base > pba->cfs_acc) ? cs2_base : pba->cfs_acc;
    return (ceff2 < 1./3.) ? ceff2 : (1./3.);
  }

  double W   = 1.0 - 2.0*pba->eps_acc;                     /* in (0,1] for physical eta>=0 */
  double xr  = sqrt(k*sqrt(2./3.)*sqrt(cs2_base)/(a*H));   /* sqrt(k/k_fs) */
  double fit = cs2_base*(1.0 + ppr->ncdm_ceff2_fs_amp*W*xr);

  /* mode 0: published Eq-38 fit (bit-identical to the pre-helper code).
     mode 1: same fit, hard-capped at 1/3. */
  if (ppr->ncdm_ceff2_mode == 0) return fit;
  return (fit < 1./3.) ? fit : (1./3.);
}
```

- [ ] **Step 2 (user): Rebuild**

Run: `make -j`
Expected: clean compile.

- [ ] **Step 3 (user): Regression smoke — modes 0/1 unchanged.** Any quick existing run (e.g. one nb15 `cached_extract` config with the cache cleared for one key, or a single nb7 exact P(k)) must reproduce prior numbers exactly; mode defaults to 0 and no default behavior may shift.

Expected: identical P(k) to pre-change build (bitwise or to ~1e-14).

- [ ] **Step 4: Commit**

```bash
git add source/perturbations.c
git commit -m "feat: ncdm_ceff2_mode=2 plateau branch + free-gas-ceiling rewording"
```

---

### Task 4: Validation notebook 16 (P(k) over k<=1 at f=0.3 + trigger scan)

**Files:**
- Create: `notebooks_test/16_test_plateau_fluid_validation.ipynb`

**Interfaces:**
- Consumes: rebuilt `classy` with mode 2 (Tasks 2-3); `saturating_cfs`, `ca2_from_kfs`, `mask_small_denom` from `fluid_closure_helpers` (Task 1); nb15's cache pattern.
- Produces: the Phase-3 verdict — max/rms `|P_fluid/P_exact - 1|` over k <= 1 per trigger, wall-times, and the adoption call.

- [ ] **Step 1: Title + setup cell.** Markdown title:

```markdown
# 16 - plateau fluid (mode 2) vs exact: P(k) over k<=1 at f=0.3
Decisive Phase-3 test of `ncdm_ceff2_mode = 2` (spec 2026-07-06). Success =
max|P_fluid/P_exact - 1| <~ 1% over k<=1 at f=0.3, plus a wall-time win vs the
exact reference. The reference is the converged exact hierarchy (fine q-grid,
`memory: accdm-fluid-f-boundary`; rkck evolver, `memory: ndf15-oom-high-q`).
Only ceff2 is fixed here - if P(k) misses 1% the shear closure (cvis2, still
default 3*w*ca2) is the named suspect, per nb15's sigma/delta panel.
```

Setup cell: copy nb15's imports/style/`base_params`/`PREC`/`accdm_params` (with `f_acc, kappa, a_t` kwargs) verbatim, then add:

```python
from fluid_closure_helpers import saturating_cfs, ca2_from_kfs, mask_small_denom
import time, os, pickle
K_PK   = np.logspace(-3, 0.0, 40)          # k <= 1 ONLY (nb7 failed on k<=10)
F_TEST, ETA_TEST = 0.3, [0.1, 1.0]
Q_REF, Q_STD = 5001, 501                    # converged reference vs production q-grid
TRIGGERS = [0.4, 0.2, 0.1, 0.05]            # fluid switch-on scan (old wall: 0.4)
CACHE_DIR = os.path.join('accDM_scans', 'nb16_cache'); os.makedirs(CACHE_DIR, exist_ok=True)

def run_pk_timed(params, k_list, tag):
    """P(k) at z=0 with wall-time; pickle-cached by tag."""
    path = os.path.join(CACHE_DIR, tag + '.pkl')
    if os.path.exists(path):
        with open(path, 'rb') as fh:
            return pickle.load(fh)
    M = Class(); M.set(params)
    t0 = time.perf_counter(); M.compute(); dt = time.perf_counter() - t0
    pk = np.array([M.pk(float(k), 0.0) for k in k_list])
    M.struct_cleanup(); M.empty()
    out = (pk, dt)
    with open(path, 'wb') as fh:
        pickle.dump(out, fh)
    return out
```

The `accdm_params` copy must set `'ncdm_N_momentum_bins': '15, {}'.format(q_size)` via a new `q_size=Q_STD` kwarg.

- [ ] **Step 2: Reference cell (user runs; expensive one-off).**

```python
REF = {}
for eta in ETA_TEST:
    p = accdm_params(eta, f_acc=F_TEST, q_size=Q_REF)   # exact, fine q
    REF[eta] = run_pk_timed(p, K_PK, 'ref_eta{:g}_f{:g}_q{}'.format(eta, F_TEST, Q_REF))
    print('exact ref eta={}: {:.0f} s'.format(eta, REF[eta][1]))
```

Expected: two converged exact runs (slow — this is the one-off cost; cached thereafter).

- [ ] **Step 3: Fluid mode-2 runs + trigger scan (user runs).**

```python
def fluid_params(eta, trigger):
    p = accdm_params(eta, f_acc=F_TEST, q_size=Q_STD)
    p['ncdm_fluid_approximation'] = 2
    p['ncdm_fluid_trigger_rho_accDM_over_rho_dcdm'] = trigger
    p['ncdm_ceff2_mode'] = 2                      # plateau closure (A default 13.0)
    return p

RES = {}                                          # RES[(eta, trig)] = (max, rms, dt)
for eta in ETA_TEST:
    pk_ref, _ = REF[eta]
    for trig in tqdm(TRIGGERS, desc='eta={}'.format(eta)):
        try:
            pk, dt = run_pk_timed(fluid_params(eta, trig), K_PK,
                                  'fluid_eta{:g}_trig{:g}'.format(eta, trig))
            r = pk/pk_ref - 1.0
            RES[(eta, trig)] = (float(np.nanmax(np.abs(r))), float(np.sqrt(np.nanmean(r**2))), dt)
        except Exception as e:
            RES[(eta, trig)] = None
            print('CRASH eta={} trig={}: {}'.format(eta, trig, str(e)[:120]))
print('{:>5} {:>6} | {:>9} {:>9} {:>8}'.format('eta', 'trig', 'max', 'rms', 'time[s]'))
for (eta, trig), v in RES.items():
    print('{:>5} {:>6} | {}'.format(eta, trig,
          '{:>9.2e} {:>9.2e} {:>8.0f}'.format(*v) if v else '    CRASH'))
```

Expected: no crashes down to some trigger (the plateau is stable by construction — how far below the old 0.4 wall it survives is itself a result); residual table per (eta, trigger).

- [ ] **Step 4: C-vs-python consistency check (user runs).** From one mode-2 fluid run, extract the late-time `cs2_ncdm[1]` (after switch-on it equals `pba->cfs_acc` wherever `ca2 < cfs`), and compare with `saturating_cfs(ca2_today, 13.0)` using `ca2_today` from the nb15 cache:

```python
p = fluid_params(0.1, 0.4); p['k_output_values'] = '0.1'
M = Class(); M.set(p); M.compute()
d = M.get_perturbations()['scalar'][0]
cfs_C = float(np.asarray(d['cs2_ncdm[1]'], float)[-1])   # late-time = plateau
M.struct_cleanup(); M.empty()
ser = pickle.load(open(os.path.join('accDM_scans', 'nb15_cache',
      'v3_eta0.1_f0.1_kap6_at0.13_nk18.pkl'), 'rb'))
s = ser[list(ser.keys())[0]]
i = np.where((s['k_fs'] > 0) & np.isfinite(s['aH']))[0][-1]
ca2_today = float(ca2_from_kfs(s['k_fs'][i], s['aH'][i]))
cfs_py = float(saturating_cfs(ca2_today, 13.0))
print('C plateau {:.4f} vs python {:.4f} ({:+.1%})'.format(cfs_C, cfs_py, cfs_C/cfs_py - 1))
assert abs(cfs_C/cfs_py - 1) < 0.10, 'C and python plateau disagree > 10%'
```

Expected: agreement within ~10% (the two `ca2_today` recoveries differ slightly: background table vs k_fss inversion).

- [ ] **Step 5: Residual plot + verdict markdown.** Plot `P_fluid/P_exact` vs k per (eta, trigger); verdict cell:

```markdown
## Verdict (fill from output)
- max residual over k<=1 at f=0.3: eta=0.1: [FILL] @ best trigger [FILL]; eta=1.0: [FILL].
- meets ~1%? [YES/NO]. If NO with correct ceff2 -> cvis2 (shear closure) is the
  named suspect; do not tune ceff2 further.
- wall-time: exact ref [FILL] s vs fluid [FILL] s -> speedup [FILL]x; compare against
  the safe q-size reduction (`memory: daughter-qsize-overkill`) before adopting.
- trigger wall: fluid stable down to trigger = [FILL] (old wall 0.4).
```

- [ ] **Step 6: Commit**

```bash
git add notebooks_test/16_test_plateau_fluid_validation.ipynb
git commit -m "feat: nb16 plateau-fluid P(k) validation (k<=1, f=0.3, trigger scan)"
```

---

## Self-Review

**Spec coverage:** input param + default (Task 2 Step 1) ✓; background hook with source-free ca2 at a=1 (Task 2 Step 4) ✓; mode-2 branch before the fit, both call sites via helper (Task 3) ✓; free-gas-ceiling rewording (Tasks 2-3 docstrings) ✓; python mirror kept in sync (Task 1) ✓; Phase-3 matrix eta={0.1,1.0}, f=0.3, k<=1, trigger scan, wall-times, cvis2-suspect note (Task 4) ✓; out-of-scope items (kappa/a_t scans, a_t=0.01 outlier) not implemented, per spec ✓.

**Placeholder scan:** Verdict `[FILL]` fields are runtime numbers the user fills after execution — intentional, not plan gaps. All code steps carry complete code.

**Type consistency:** `saturating_cfs(ca2_today, A)` matches its Task 4 call; `pba->cfs_acc` (double) declared in Task 2, read in Task 3; `accdm_params(..., q_size=...)` kwarg introduced in Task 4 Step 1 and used in Steps 2-3; `run_pk_timed -> (pk, dt)` unpacked consistently.

**Known risk carried forward:** `cs2_ncdm[1]` late-time extraction in Task 4 Step 4 assumes the fluid has switched on by the last output time at trigger 0.4 — if not, the assert fires and the trigger for the check should be lowered, not the tolerance.
