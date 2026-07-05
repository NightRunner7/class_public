# Fluid-Closure Feasibility Diagnostic Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a diagnostic (helpers module + notebook `15_test_fluid_closure_diagnostic.ipynb`) that decides whether a universal fluid closure exists for the accDM daughter over k ≤ 1 Mpc⁻¹, f_acc ≤ 0.3 — returning an A/B/C verdict from collapse plots of the effective `ceff2` and `cvis2` responses vs `x = k/k_fs`.

**Architecture:** Pure-numpy helpers (`notebooks_test/fluid_closure_helpers.py`) hold all testable logic — `ca2` recovery from `k_fs`, the two dimensionless response functions, the log-binned upper envelope, and the collapse-band metric — and are unit-tested with pytest offline. The notebook imports them, runs the exact hierarchy across `(f, η)`, extracts per-`(k,τ)` daughter quantities CLASS already emits, and produces the collapse plots + verdict. No C changes, no rebuild.

**Tech Stack:** Python 3, numpy, matplotlib, `classy` (CLASS Python wrapper), pytest for the helpers, Jupyter.

## Global Constraints

- Edit `class_accDM`, never the pristine reference — `memory: apply-fixes-to-working-branch`.
- Daughter is the last ncdm species (`n_ncdm == N_ncdm-1`), `has_acc`-gated — `memory: last-species-must-be-has-acc-gated`.
- Gauge: synchronous, `get_perturbations_in_current_gauge = yes`; restrict to k ≤ 1 and late times (sub-horizon, gauge-robust) — `memory: super-horizon-gauge-limitation`.
- Plot style: STIX serif + ColorBrewer qualitative palette — `memory: notebook-plot-style` (mirror notebook 7's `plt.rcParams` block and `qual_colors`).
- `k_fs` convention: `k_fss_acc = sqrt(3/2)·aH/sqrt(ca2)` (`source/perturbations.c:8771`), so `ca2 = (3/2)·(aH/k_fs)²`.
- No gcc/make/classy in the agent shell — `memory: build-environment`. The helpers (pure numpy) ARE runnable here; the notebook's CLASS cells are authored here and executed by the user.
- The exact hierarchy is the convergence reference; this is a diagnostic, it produces a verdict, not a fit.

---

### Task 1: Pure-Python helpers module (TDD, no CLASS)

**Files:**
- Create: `notebooks_test/fluid_closure_helpers.py`
- Test: `notebooks_test/test_fluid_closure_helpers.py`

**Interfaces:**
- Consumes: nothing (numpy only).
- Produces:
  - `ca2_from_kfs(k_fs, aH) -> np.ndarray` — base adiabatic sound speed from the free-streaming scale.
  - `sound_speed_response(delta_p_over_delta_rho, ca2) -> np.ndarray` — `R_c = (δp/δρ)/ca2`.
  - `shear_response(k, shear, theta, eps=1e-30) -> np.ndarray` — `R_v = k·σ/θ`.
  - `log_upper_envelope(x, y, n_bins=40) -> (np.ndarray, np.ndarray)` — `(x_centers, env)`, max `|y|` per log-x bin, empty bins dropped.
  - `collapse_band(x_grid, curves) -> (np.ndarray, float)` — `curves` is a list of `(x_i, y_i)`; returns `(band_of_x, max_band)` where at each `x` in `x_grid` `band = (max−min)/|median|` across curves that cover it.

- [ ] **Step 1: Write the failing tests**

```python
# notebooks_test/test_fluid_closure_helpers.py
import numpy as np
import pytest
from fluid_closure_helpers import (
    ca2_from_kfs, sound_speed_response, shear_response,
    log_upper_envelope, collapse_band,
)

def test_ca2_from_kfs_inverts_convention():
    # k_fs = sqrt(3/2)*aH/sqrt(ca2)  =>  ca2 = (3/2)*(aH/k_fs)^2
    aH, ca2_true = 2.0, 0.05
    k_fs = np.sqrt(1.5) * aH / np.sqrt(ca2_true)
    assert np.isclose(ca2_from_kfs(k_fs, aH), ca2_true)

def test_ca2_from_kfs_array_and_zero_guard():
    out = ca2_from_kfs(np.array([1.0, 0.0]), np.array([1.0, 1.0]))
    assert np.isfinite(out[0]) and out[1] == 0.0   # k_fs=0 -> 0, no divide error

def test_sound_speed_response_is_ratio():
    r = sound_speed_response(np.array([0.1, 0.2]), np.array([0.05, 0.05]))
    assert np.allclose(r, [2.0, 4.0])

def test_shear_response_scales_with_k_and_guards_zero_theta():
    r = shear_response(2.0, np.array([0.3, 0.3]), np.array([0.6, 0.0]))
    assert np.isclose(r[0], 2.0 * 0.3 / 0.6)       # = 1.0
    assert np.isfinite(r[1])                        # theta=0 guarded, no inf

def test_log_upper_envelope_takes_abs_max_per_bin():
    x = np.array([1.0, 1.1, 10.0, 11.0])
    y = np.array([-5.0, 2.0, 1.0, -0.5])            # bin1 max|y|=5, bin2 max|y|=1
    xc, env = log_upper_envelope(x, y, n_bins=2)
    assert env[0] == 5.0 and env[-1] == 1.0
    assert np.all(np.diff(xc) > 0)

def test_collapse_band_zero_for_identical_curves():
    x = np.logspace(0, 2, 20)
    curves = [(x, np.sqrt(x)), (x, np.sqrt(x))]
    band, mx = collapse_band(x, curves)
    assert mx < 1e-9

def test_collapse_band_measures_spread():
    x = np.logspace(0, 2, 20)
    curves = [(x, np.ones_like(x)), (x, 2.0 * np.ones_like(x))]
    band, mx = collapse_band(x, curves)
    # (max-min)/median = (2-1)/1.5 = 0.6667 everywhere
    assert np.isclose(mx, (2.0 - 1.0) / 1.5, atol=1e-6)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest notebooks_test/test_fluid_closure_helpers.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'fluid_closure_helpers'` (run from `notebooks_test/`, or add `-c` / conftest; simplest is `cd notebooks_test && python -m pytest test_fluid_closure_helpers.py -v`).

- [ ] **Step 3: Write the implementation**

```python
# notebooks_test/fluid_closure_helpers.py
"""Pure-numpy helpers for the fluid-closure feasibility diagnostic (notebook 15).

No CLASS dependency: everything here is unit-tested offline. The notebook feeds
these the daughter perturbation time-series the exact hierarchy already emits
(delta_ncdm, theta_ncdm, shear_ncdm, cs2_ncdm=delta_p/delta_rho, k_fss_acc).
"""
from __future__ import annotations
import numpy as np

_KFS_FACTOR = 1.5  # k_fs = sqrt(3/2)*aH/sqrt(ca2) -> ca2 = (3/2)*(aH/k_fs)^2

def ca2_from_kfs(k_fs, aH):
    """Recover base adiabatic sound speed ca2 from the free-streaming scale.
    k_fs <= 0 (unborn/degenerate daughter) maps to 0."""
    k_fs = np.asarray(k_fs, float)
    aH = np.asarray(aH, float)
    good = k_fs > 0.0
    out = np.zeros(np.broadcast(k_fs, aH).shape, float)
    ratio = np.divide(aH, k_fs, out=np.zeros_like(out), where=good)
    return np.where(good, _KFS_FACTOR * ratio * ratio, 0.0)

def sound_speed_response(delta_p_over_delta_rho, ca2, eps=1e-30):
    """R_c = (delta_p/delta_rho) / ca2 — the quantity the Eq-38 fit models as
    1 + amp*W*sqrt(k/k_fs)."""
    dpr = np.asarray(delta_p_over_delta_rho, float)
    ca2 = np.asarray(ca2, float)
    return dpr / np.where(np.abs(ca2) < eps, np.nan, ca2)

def shear_response(k, shear, theta, eps=1e-30):
    """R_v = k*sigma/theta — the anisotropic-stress (cvis2) signature."""
    shear = np.asarray(shear, float)
    theta = np.asarray(theta, float)
    denom = np.where(np.abs(theta) < eps, np.nan, theta)
    return k * shear / denom

def log_upper_envelope(x, y, n_bins=40):
    """Upper envelope of |y| over log-spaced x bins. Returns (x_centers, env)
    for non-empty bins only. x must be positive."""
    x = np.asarray(x, float); y = np.abs(np.asarray(y, float))
    m = (x > 0) & np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if x.size == 0:
        return np.array([]), np.array([])
    edges = np.logspace(np.log10(x.min()), np.log10(x.max()), n_bins + 1)
    edges[-1] *= 1.0 + 1e-9  # include the max
    idx = np.digitize(x, edges) - 1
    xc, env = [], []
    for b in range(n_bins):
        sel = idx == b
        if np.any(sel):
            xc.append(np.sqrt(edges[b] * edges[b + 1]))
            env.append(y[sel].max())
    return np.asarray(xc), np.asarray(env)

def collapse_band(x_grid, curves, eps=1e-30):
    """Quantify collapse of several (x_i, y_i) curves onto x_grid (log-interp).
    At each x covered by >=2 curves, band = (max-min)/|median|. Returns
    (band_of_x, max_band). NaN where <2 curves cover x."""
    x_grid = np.asarray(x_grid, float)
    logg = np.log10(x_grid)
    stack = np.full((len(curves), x_grid.size), np.nan)
    for i, (xi, yi) in enumerate(curves):
        xi = np.asarray(xi, float); yi = np.asarray(yi, float)
        m = (xi > 0) & np.isfinite(xi) & np.isfinite(yi)
        if m.sum() < 2:
            continue
        order = np.argsort(xi[m])
        xs, ys = np.log10(xi[m][order]), yi[m][order]
        inside = (logg >= xs[0]) & (logg <= xs[-1])
        stack[i, inside] = np.interp(logg[inside], xs, ys)
    band = np.full(x_grid.size, np.nan)
    for j in range(x_grid.size):
        col = stack[:, j]
        col = col[np.isfinite(col)]
        if col.size >= 2:
            med = np.median(col)
            band[j] = (col.max() - col.min()) / (abs(med) + eps)
    finite = band[np.isfinite(band)]
    return band, (float(finite.max()) if finite.size else float("nan"))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd notebooks_test && python -m pytest test_fluid_closure_helpers.py -v`
Expected: PASS (7 passed).

- [ ] **Step 5: Commit**

```bash
git add notebooks_test/fluid_closure_helpers.py notebooks_test/test_fluid_closure_helpers.py
git commit -m "feat: pure-numpy helpers for fluid-closure diagnostic (TDD)"
```

---

### Task 2: Notebook scaffold + τ-series extractor cell

**Files:**
- Create: `notebooks_test/15_test_fluid_closure_diagnostic.ipynb`

**Interfaces:**
- Consumes: `fluid_closure_helpers` (Task 1); `accdm_params` pattern from `notebooks_test/7_test_ceff2_calibration.ipynb`.
- Produces: `extract_daughter_series(params, k_list) -> dict` returning, per k, arrays over stored τ of `{tau, delta, theta, shear, dpr, k_fs, aH}` for the daughter (ncdm index 1). `aH` per τ is recovered as `k_fs·sqrt(ca2)/sqrt(3/2)` is circular, so instead read background: `aH = a·H` reconstructed from the perturbation dict's `a` and the background `H(a)` — see Step 2 note.

- [ ] **Step 1: Title + setup cell**

Create the notebook with a markdown title cell and a setup code cell. The setup cell copies notebook 7's `plt.rcParams` block, `qual_colors`, base cosmology, `accdm_params(...)` (exact mode only needed), and adds:

```python
import sys; sys.path.insert(0, '.')      # import the helpers module from notebooks_test/
from fluid_closure_helpers import (
    ca2_from_kfs, sound_speed_response, shear_response,
    log_upper_envelope, collapse_band,
)
# k<=1 grid and the (f, eta) matrix the collapse test sweeps
K_GRID  = np.logspace(-2, 0.0, 18)         # 1/Mpc, sub-horizon, below k_fs to above
F_LIST  = [0.05, 0.1, 0.2, 0.3]
ETA_LIST = [0.1, 1.0]                       # production (cold) + one warmer boost
```

Markdown title cell text:
```markdown
# 15 - accDM daughter: does a universal fluid closure exist?
Decide A/B/C (one formula / f-dependent / tables) BEFORE fitting. We read the
exact daughter response pointwise and test whether ceff2 and cvis2 collapse onto
one function of x=k/k_fs across (tau, f, eta). See
`docs/superpowers/specs/2026-07-05-fluid-closure-feasibility-diagnostic.md`.
```

- [ ] **Step 2: Extractor cell**

Adapt notebook 7's `run_cs2_of_k` to keep the full τ-series and the extra columns. `accdm_params` must set `f_acc` and `eta_acc` per call (add `f_acc`/`eta` kwargs mirroring notebook 7). Recover `aH` per τ from the daughter's own `k_fs` and `cs2` where both are positive (`aH = k_fs·sqrt(cs2)/sqrt(3/2)`); this is exact by the `k_fss_acc` definition and avoids a separate background call.

```python
def _find_key(d, want):
    if want in d: return want
    for kk in d:
        if kk.replace(' ', '').startswith(want.replace(' ', '')):
            return kk
    raise KeyError('{!r} not in {}'.format(want, list(d.keys())))

def extract_daughter_series(params, k_list):
    ks = np.sort(np.asarray(k_list, float))
    p = dict(params); p['k_output_values'] = ', '.join('{:.8e}'.format(k) for k in ks)
    M = Class(); M.set(p); M.compute()
    perts = M.get_perturbations()['scalar']
    kd  = _find_key(perts[0], 'delta_ncdm[1]');  kt = _find_key(perts[0], 'theta_ncdm[1]')
    ksh = _find_key(perts[0], 'shear_ncdm[1]');  kc = _find_key(perts[0], 'cs2_ncdm[1]')
    kf  = _find_key(perts[0], 'k_fss_acc[1]')
    out = {}
    for k, d in zip(ks, perts):
        kfs = np.asarray(d[kf], float); cs2 = np.asarray(d[kc], float)
        aH = np.where((kfs > 0) & (cs2 > 0), kfs * np.sqrt(np.abs(cs2)) / np.sqrt(1.5), np.nan)
        out[k] = dict(delta=np.asarray(d[kd], float), theta=np.asarray(d[kt], float),
                      shear=np.asarray(d[ksh], float), dpr=cs2, k_fs=kfs, aH=aH)
    M.struct_cleanup(); M.empty()
    return out
```

- [ ] **Step 3: Structure-check (user runs one small extraction)**

Add a cell that runs ONE cheap extraction and asserts shape/keys, so a bad column name fails fast before the full sweep:

```python
_probe = extract_daughter_series(accdm_params(ETA_LIST[0], 'exact', f_acc=0.1), K_GRID[:3])
assert set(_probe.keys()) == set(K_GRID[:3])
_s = _probe[K_GRID[0]]
assert all(_s[q].shape == _s['delta'].shape for q in ('theta','shear','dpr','k_fs','aH'))
print('extractor OK; tau samples per k =', _s['delta'].size)
```

Run (user, ~1-2 min): execute the cell.
Expected: `extractor OK; tau samples per k = <N>` with no assertion error.

- [ ] **Step 4: Commit**

```bash
git add notebooks_test/15_test_fluid_closure_diagnostic.ipynb
git commit -m "feat: nb15 scaffold + daughter tau-series extractor (pending user run)"
```

---

### Task 3: Response-function assembly cell (R_c, R_v across f, η)

**Files:**
- Modify: `notebooks_test/15_test_fluid_closure_diagnostic.ipynb`

**Interfaces:**
- Consumes: `extract_daughter_series` (Task 2); `ca2_from_kfs`, `sound_speed_response`, `shear_response`, `log_upper_envelope` (Task 1).
- Produces: `RESPONSES` — nested dict `RESPONSES[(f, eta)] = {'Rc': (x, env), 'Rv': (x, env)}`, each an envelope in `x = k/k_fs` pooled over all k and τ.

- [ ] **Step 1: Assembly cell**

For each `(f, η)`, run one exact extraction, pool every `(k, τ)` sample into `(x, R_c)` and `(x, R_v)` point clouds (dropping non-finite / `k_fs<=0` / super-horizon `x<... ` points), then envelope. `x = k/k_fs`; recover `ca2` from `k_fs, aH` and cross-check it against `dpr` sign.

```python
def build_responses(f, eta):
    ser = extract_daughter_series(accdm_params(eta, 'exact', f_acc=f), K_GRID)
    xs, rc, rv = [], [], []
    for k, s in ser.items():
        good = np.isfinite(s['k_fs']) & (s['k_fs'] > 0) & np.isfinite(s['aH'])
        if not np.any(good): continue
        x = k / s['k_fs'][good]
        ca2 = ca2_from_kfs(s['k_fs'][good], s['aH'][good])
        xs.append(x)
        rc.append(sound_speed_response(s['dpr'][good], ca2))
        rv.append(shear_response(k, s['shear'][good], s['theta'][good]))
    x = np.concatenate(xs); rc = np.concatenate(rc); rv = np.concatenate(rv)
    return {'Rc': log_upper_envelope(x, rc), 'Rv': log_upper_envelope(x, rv)}

RESPONSES = {}
for eta in ETA_LIST:
    for f in tqdm(F_LIST, desc='eta={}'.format(eta)):
        RESPONSES[(f, eta)] = build_responses(f, eta)
print('built responses for', list(RESPONSES.keys()))
```

Run (user, ~8-16 min: |F_LIST|×|ETA_LIST| exact runs).
Expected: `built responses for [(0.05,0.1),...]`, 8 entries, each with non-empty `Rc`/`Rv` envelopes.

- [ ] **Step 2: Commit**

```bash
git add notebooks_test/15_test_fluid_closure_diagnostic.ipynb
git commit -m "feat: nb15 R_c/R_v response assembly across (f,eta) (pending user run)"
```

---

### Task 4: Collapse plots + A/B/C decision cell

**Files:**
- Modify: `notebooks_test/15_test_fluid_closure_diagnostic.ipynb`

**Interfaces:**
- Consumes: `RESPONSES` (Task 3); `collapse_band` (Task 1).
- Produces: printed A/B/C verdict + two-panel figure (R_c, R_v overlays).

- [ ] **Step 1: Plot + band-metric cell**

Overlay all `(f, η)` envelopes for `R_c` and `R_v`; color by `f`, linestyle by `η`. Overlay the current fit shape `1 + 0.2·W·√x` on the `R_c` panel for reference. Compute two band metrics: pooled-over-all (does everything collapse?) and within-fixed-f (does it collapse once f is fixed?).

```python
X_EVAL = np.logspace(-1, 3, 60)
def _band_all(which):
    return collapse_band(X_EVAL, [RESPONSES[key][which] for key in RESPONSES])[1]
def _band_fixed_f(which):
    per_f = []
    for f in F_LIST:
        curves = [RESPONSES[(f, e)][which] for e in ETA_LIST]
        per_f.append(collapse_band(X_EVAL, curves)[1])
    return float(np.nanmax(per_f))

fig, axes = plt.subplots(1, 2, figsize=(11, 4.3), constrained_layout=True)
ls_eta = {ETA_LIST[0]: '-', ETA_LIST[1]: '--'}
col_f  = {f: qual_colors[i] for i, f in enumerate(F_LIST)}
for (f, eta), r in RESPONSES.items():
    for ax, which in zip(axes, ('Rc', 'Rv')):
        x, env = r[which]
        ax.loglog(x, env, ls_eta[eta], color=col_f[f], alpha=0.85,
                  label='f={}, eta={}'.format(f, eta))
W0 = 1 - 2*(-0.01 + np.sqrt(0.0001 + 0.04 + 0.5*0.1 + 0.02) - 0.2)  # eps_acc(0.1)~0.294
axes[0].loglog(X_EVAL, 1 + 0.2*0.412*np.sqrt(X_EVAL), 'k:', label='fit 1+0.2W√x')
axes[0].set_title('R_c = (δp/δρ)/ca2'); axes[1].set_title('R_v = kσ/θ')
for ax in axes:
    ax.set_xlabel('x = k/k_fs'); ax.grid(True, which='both', alpha=0.3)
axes[0].legend(fontsize=7, ncol=2); plt.show()

band = {which: (_band_all(which), _band_fixed_f(which)) for which in ('Rc', 'Rv')}
print('band (pooled-all, fixed-f):', {w: tuple(round(v,3) for v in band[w]) for w in band})
```

- [ ] **Step 2: Decision cell (verdict A/B/C)**

Encode the decision rule. Thresholds: "collapsed" if band < 0.15 (15%), "f-separable" if fixed-f band < 0.15 while pooled ≥ 0.15.

```python
TOL = 0.15
def verdict(which):
    pooled, fixedf = band[which]
    if pooled < TOL:  return 'UNIVERSAL (one formula in x)'
    if fixedf < TOL:  return 'F-DEPENDENT (formula with f-varying coefficients)'
    return 'NO COLLAPSE (interpolation tables)'
vc, vv = verdict('Rc'), verdict('Rv')
print('R_c (ceff2 sector):', vc)
print('R_v (cvis2 sector):', vv)
worst = max((band['Rc'][0], 'Rc'), (band['Rv'][0], 'Rv'))
print('\nAPPROACH ->',
      'B/A universal' if 'UNIVERSAL' in vc and 'UNIVERSAL' in vv else
      'A with f-coeffs' if 'NO COLLAPSE' not in (vc+vv) else
      'C tables (worst sector: {})'.format(worst[1]))
```

Run (user): execute both cells.
Expected: two-panel figure + printed verdict naming Approach B/A-universal, A-with-f-coeffs, or C.

- [ ] **Step 3: Commit**

```bash
git add notebooks_test/15_test_fluid_closure_diagnostic.ipynb
git commit -m "feat: nb15 collapse plots + A/B/C decision (pending user run)"
```

---

### Task 5: Existing-scaffolding cross-check + feasibility-gate + verdict markdown

**Files:**
- Modify: `notebooks_test/15_test_fluid_closure_diagnostic.ipynb`

**Interfaces:**
- Consumes: `RESPONSES`, `band` (Tasks 3-4).
- Produces: markdown verdict cell; a note on whether `w_trial_2` (the in-code `ca2·(1+0.25√x)` trial) tracks the measured `R_c` envelope.

- [ ] **Step 1: w_trial cross-check cell**

The exact run already stores `w_trial_1`/`w_trial_2` columns (`source/perturbations.c:3394`). Pull them from a single extraction and overlay on the `R_c` envelope to see whether the in-code `0.25` trial amplitude already tracks the measured response better than the `0.2` paper value.

```python
def pull_wtrials(f, eta):
    ser_params = accdm_params(eta, 'exact', f_acc=f)
    ser_params['k_output_values'] = ', '.join('{:.8e}'.format(k) for k in K_GRID)
    M = Class(); M.set(ser_params); M.compute()
    perts = M.get_perturbations()['scalar']
    kw2 = _find_key(perts[0], 'w_trial_2'); kf = _find_key(perts[0], 'k_fss_acc[1]')
    kc = _find_key(perts[0], 'cs2_ncdm[1]')
    xs, w2 = [], []
    for k, d in zip(K_GRID, perts):
        kfs = np.asarray(d[kf], float); cs2 = np.asarray(d[kc], float)
        g = (kfs > 0) & (cs2 > 0)
        ca2 = ca2_from_kfs(kfs[g], kfs[g]*np.sqrt(np.abs(cs2[g]))/np.sqrt(1.5))
        xs.append(k/kfs[g]); w2.append(np.asarray(d[kw2], float)[g]/ca2)
    M.struct_cleanup(); M.empty()
    return log_upper_envelope(np.concatenate(xs), np.concatenate(w2))

xw, w2env = pull_wtrials(0.1, 0.1)
plt.figure(figsize=(6,4))
xr, rcenv = RESPONSES[(0.1,0.1)]['Rc']
plt.loglog(xr, rcenv, 'o-', color=qual_colors[0], label='measured R_c (f=0.1)')
plt.loglog(xw, w2env, 's--', color=qual_colors[1], label='w_trial_2/ca2 (in-code 0.25)')
plt.loglog(X_EVAL, 1+0.2*0.412*np.sqrt(X_EVAL), 'k:', label='paper 0.2')
plt.xlabel('x=k/k_fs'); plt.legend(); plt.grid(True, which='both', alpha=0.3); plt.show()
```

Run (user): execute.
Expected: a plot showing whether the `0.25` in-code trial or the `0.2` paper value better tracks the measured envelope.

- [ ] **Step 2: Feasibility-gate + verdict markdown cell**

Add a final markdown cell summarizing the numeric verdict for the record. Fill the bracketed numbers from the executed `band` dict and the printed verdict (the user completes these after running):

```markdown
## Verdict

- **R_c (ceff2) collapse:** pooled band = [FILL], fixed-f band = [FILL] -> [UNIVERSAL/F-DEP/NO].
- **R_v (cvis2) collapse:** pooled band = [FILL], fixed-f band = [FILL] -> [UNIVERSAL/F-DEP/NO].
- **Chosen approach:** [B/A universal | A with f-coeffs | C tables].
- **cvis2 matters?** R_v shows [structure/no structure] the current ceff2-only fit ignores -> [fit cvis2 too / ceff2 alone suffices].
- **Feasibility gate:** at f=0.3 the converged exact needs q_size~5001 (`memory: accdm-fluid-f-boundary`); the fluid is worth pursuing only if the chosen approach reaches ~1% P(k) over k<=1 cheaper than exact + the q(f) schedule (notebook 14). If the verdict is C or cvis2 shows no usable structure, route to notebook 14 instead.
- **Next:** write the follow-up plan (the fit) matched to this verdict; do NOT touch `perturbations_ceff2_ncdm` before then.
```

- [ ] **Step 3: Commit**

```bash
git add notebooks_test/15_test_fluid_closure_diagnostic.ipynb
git commit -m "feat: nb15 w_trial cross-check + feasibility-gate verdict (pending user run)"
```

---

## Self-Review

**Spec coverage:**
- Extractor of full τ-series {δ,θ,σ,δp/δρ,k_fs} → Task 2. ✓
- Two response functions R_c, R_v on the envelope → Tasks 1 (fns) + 3 (assembly). ✓
- Collapse test + A/B/C decision rule → Task 4. ✓
- Surface existing scaffolding (w_trial/w_sigma) → Task 5 (w_trial_2 overlay; w_sigma/w_theta expose deferred to the fit plan if greenlit, per spec — it needs a rebuild). ✓
- Feasibility-gate readout routing to notebook 14 → Task 5. ✓
- Pure Python / no rebuild constraint → helpers are numpy-only; only optional column-expose (not done here) would need a build. ✓

**Placeholder scan:** The Verdict markdown (Task 5 Step 2) has `[FILL]` fields — these are intentional, filled by the user from executed output, not plan placeholders (the plan cannot know runtime numbers). All code steps contain complete code. No "TBD"/"handle edge cases" in logic steps.

**Type consistency:** `extract_daughter_series` returns per-k dicts with keys `delta/theta/shear/dpr/k_fs/aH`, consumed unchanged in Tasks 3 and 5. `log_upper_envelope` returns `(x, env)` tuples, stored as `RESPONSES[key]['Rc'|'Rv']` and unpacked consistently. `collapse_band` returns `(band_array, max_float)`, and callers use `[1]` for the scalar. `ca2_from_kfs(k_fs, aH)` signature matches all call sites. Consistent.

**Note on `aH` recovery:** deriving `aH = k_fs·√cs2/√(3/2)` is exact by the `k_fss_acc` definition but degenerate when `cs2≤0` (oscillation troughs) — those samples are dropped by the `good`/`g` masks before use, which is correct (we envelope over the survivors).
