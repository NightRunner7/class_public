# accDM → CDM Indistinguishability Test — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `notebooks_test/13_test_accDM_CDM_indistinguishability.ipynb`, which sweeps the daughter mass (via `eta = 1e11/m`) and locates the mass above which accDM's P(k) and CMB spectra become indistinguishable from CDM under fixed-tolerance and cosmic-variance χ² metrics.

**Architecture:** A single self-contained notebook. Pure-numpy helper functions (metrics, threshold extraction) are unit-tested inline with `assert` cells so `pytest --nbmake` exercises them without CLASS. CLASS-dependent cells (parameter builders, the run/cache layer, the scan) are verified by executing the notebook in the user's `accDM` classy environment. Results are cached in a dict keyed by `(f_acc, m, kind)`, mirroring `notebooks_test/5_test_Pk_freestreaming.ipynb`.

**Tech Stack:** Python 3.12, `classy` (CLASS accDM fork), numpy, matplotlib. Test runner: `pytest --nbmake`.

## Global Constraints

- **Cannot run CLASS in the authoring shell.** The agent shell (Windows `E:\`) has no compiler and no `classy`; the classy env is the user's Linux pyenv `accDM`. Build cells with `NotebookEdit`; CLASS-executing cells are verified by the user running the notebook there. See memory `build-environment`.
- **Daughter sector must be exact, never fluid.** `ncdm_quadrature_strategy = 4`, `ncdm_fluid_approximation = 0`. The daughter fluid closure is stiff at switch-on and crashes (memory `fluid-approx-unusable`).
- **Physical mapping is fixed:** `eta_acc = 1e11 / m`, `m` in GeV; daughter `m_ncdm = m * 1e9` eV.
- **Fiducial decay sector, held fixed:** `KAPPA = 2.0`, `A_T = 0.13`, `vary_Gamma_acc = 'yes'`, `A_REC = 1/(1+1090)`.
- **Base cosmology (Planck 2018), copied verbatim from notebook 5:** `omega_b=0.022383`, `omega_cdm0=0.12011`, `A_s=2.1005829616811546e-9`, `n_s=0.96605`, `tau_reio=0.0543`, `H0=67.32`.
- **Plot style (memory `notebook-plot-style`):** STIX mathtext, serif, ColorBrewer qualitative palette `['#377eb8','#ff7f00','#4daf4a','#f781bf','#984ea3']`.
- **Keep k sub-horizon:** k-grid floor `1e-3` /Mpc (accDM gauge construction is sub-horizon only — memory `super-horizon-gauge-limitation`).
- **Number formatting:** exact strings, e.g. daughter mass `'{:.6e}'.format(m*1e9)`.

---

### Task 1: Notebook scaffold — header, imports, constants, parameter builders

**Files:**
- Create: `notebooks_test/13_test_accDM_CDM_indistinguishability.ipynb`

**Interfaces:**
- Produces (module-level names later cells rely on):
  - Constants: `base_params` (dict), `PREC_PK` (dict), `PREC_CMB` (dict), `K_NODES` (np.ndarray, 60 pts), `L_MAX=2500`, `F_SEQ=[0.01,0.05,0.1]`, `MASS_GRID` (np.ndarray, 12 pts, GeV), `KAPPA=2.0`, `A_T=0.13`, `A_REC`, `ETA_COLD=1e-12`, `V_SURVEY` (float, Mpc^3), `F_SKY=0.7`, `qual_colors` (list).
  - `eta_of(m) -> float` : returns `1e11/m`.
  - `ocdm_rescaled(f_acc) -> float` : decayed CDM density.
  - `lcdm_params() -> dict`
  - `accdm_params(f_acc, m) -> dict`  (warm daughter, eta = eta_of(m))
  - `coldlimit_params(f_acc) -> dict` (same as accdm but eta = ETA_COLD; mass fixed at MASS_GRID[-1])

- [ ] **Step 1: Create the notebook with the title/intro markdown cell**

Use `NotebookEdit` (insert, markdown) with content:

```markdown
# 13 - When does accDM become indistinguishable from CDM?

The accelerated-DM daughter is born with a velocity kick `v = sqrt(eta(eta+2))/(1+eta)`
set by the energy boost `eta_acc`. Adopting the physical relation **`eta = 1e11 / m`**
(`m` = daughter mass in GeV), a heavier daughter has smaller `eta`, free-streams less,
and is colder. This notebook sweeps `m` upward and finds the mass above which accDM's
matter power `P(k)` and lensed CMB spectra match "CDM" within our thresholds.

Two references ("CDM"): (1) the **cold limit** of the same model (`eta -> 0`), isolating
free-streaming; (2) **plain LCDM** (no acc species). Two metric families: (A) a fixed
fractional tolerance on the spectra, and (B) a cosmic-variance-limited chi^2 detectability.

Daughter on exact quadrature (fluid closure is unusable). Hybrid notebook: inline unit
asserts for the pure metrics (nbmake) + a CLASS scan + diagnostic plots. Run in the
`accDM` classy environment: `pytest --nbmake notebooks_test/13_test_accDM_CDM_indistinguishability.ipynb`.
```

- [ ] **Step 2: Add the imports + plot-style + constants code cell**

Insert a code cell:

```python
import numpy as np
import matplotlib.pyplot as plt
from classy import Class

plt.rcParams.update({
    'mathtext.fontset': 'stix', 'font.family': 'serif', 'font.size': 11,
    'axes.labelsize': 12, 'legend.fontsize': 9, 'lines.linewidth': 1.5, 'figure.dpi': 300})
qual_colors = ['#377eb8', '#ff7f00', '#4daf4a', '#f781bf', '#984ea3']

# ---- Planck 2018 base cosmology ----------------------------------
omega_b, omega_cdm0 = 0.022383, 0.12011
A_s, n_s, tau_reio, H0 = 2.1005829616811546e-9, 0.96605, 0.0543, 67.32
base_params = {'omega_b': omega_b, 'omega_cdm': omega_cdm0, 'H0': H0,
               'A_s': A_s, 'n_s': n_s, 'tau_reio': tau_reio}

# ---- fixed accDM decay sector ------------------------------------
KAPPA, A_T = 2.0, 0.13
A_REC = 1.0 / (1.0 + 1090.0)
ETA_COLD = 1e-12                     # "cold limit" eta

# ---- scan axes ---------------------------------------------------
F_SEQ = [0.01, 0.05, 0.1]
MASS_GRID = np.logspace(11, 19, 12)  # GeV  -> eta ~ 1 down to ~1e-8

# ---- observables -------------------------------------------------
K_NODES = np.logspace(-3, 0.0, 60)   # 1/Mpc, sub-horizon at z=0
L_MAX = 2500
Q_BINS = 250                          # daughter momentum bins (accuracy vs speed knob)

PREC_COMMON = {'evolver': 0, 'reionization_z_start_max': 80, 'background_Nloga': 2001}
PREC_PK  = {**PREC_COMMON, 'output': 'mPk', 'P_k_max_1/Mpc': 1.0, 'z_max_pk': 0.0}
PREC_CMB = {**PREC_COMMON, 'output': 'tCl,pCl,lCl,mPk', 'lensing': 'yes',
            'l_max_scalars': L_MAX, 'P_k_max_1/Mpc': 1.0, 'z_max_pk': 0.0}

# ---- survey / detectability assumptions --------------------------
V_SURVEY = 100.0**3                   # (100 Mpc)^3 fiducial; edit to taste
F_SKY = 0.7

def eta_of(m):
    return 1e11 / m
```

- [ ] **Step 3: Add the parameter-builder code cell**

Insert a code cell:

```python
def ocdm_rescaled(f_acc):
    """CDM density rescaled for the decayed daughter, as in notebook 5."""
    return omega_cdm0 * (1 + f_acc*(1 - A_REC**KAPPA)/(1 + (A_REC/A_T)**KAPPA))**(-1)

def lcdm_params():
    """Plain LCDM, no acc species; a massive-nu sector matched to the accDM runs."""
    p = dict(base_params); p.update(PREC_CMB)
    p.update({'N_ncdm': 1, 'deg_ncdm': 3, 'm_ncdm': 0.02, 'T_ncdm': 0.71611,
              'ncdm_quadrature_strategy': 0, 'ncdm_N_momentum_bins': 15, 'N_ur': 0.00441})
    return p

def _accdm_common(f_acc, m, eta):
    p = dict(base_params); p.update(PREC_CMB)
    p.update({'omega_cdm': ocdm_rescaled(f_acc),
              'vary_Gamma_acc': 'yes', 'kappa_acc': KAPPA, 'a_t_acc': A_T,
              'f_acc': f_acc, 'eta_acc': eta,
              'm_acc_in_GeV': m, 'm_cdm_in_GeV': m,
              'N_ncdm': 2, 'deg_ncdm': '3, 1',
              'm_ncdm': '0.02, {:.6e}'.format(m*1e9),
              'T_ncdm': '0.71611, 1', 'ncdm_quadrature_strategy': '0, 4',
              'ncdm_N_momentum_bins': '15, {:d}'.format(Q_BINS), 'N_ur': 0.00441,
              'ncdm_fluid_approximation': 0,
              'ncdm_fluid_trigger_rho_accDM_over_rho_dcdm': 10})
    return p

def accdm_params(f_acc, m):
    return _accdm_common(f_acc, m, eta_of(m))

def coldlimit_params(f_acc):
    return _accdm_common(f_acc, MASS_GRID[-1], ETA_COLD)
```

- [ ] **Step 4: Sanity-check the builders (structural, no CLASS) with an inline assert cell**

Insert a code cell (runs even without CLASS reaching `.compute()`):

```python
# structural sanity: mapping + required keys present, exact quadrature enforced
assert abs(eta_of(1e11) - 1.0) < 1e-12
assert abs(eta_of(1e18) - 1e-7) < 1e-15
_p = accdm_params(0.1, 1e15)
assert _p['ncdm_quadrature_strategy'].endswith('4')      # daughter exact
assert _p['ncdm_fluid_approximation'] == 0               # never fluid
assert _p['m_ncdm'].split(',')[1].strip() == '{:.6e}'.format(1e15*1e9)
assert _p['eta_acc'] == eta_of(1e15)
assert coldlimit_params(0.1)['eta_acc'] == ETA_COLD
print('Task 1 param builders OK')
```

- [ ] **Step 5: Commit**

```bash
git add notebooks_test/13_test_accDM_CDM_indistinguishability.ipynb
git commit -m "nb13: scaffold - constants and accDM/LCDM parameter builders"
```

---

### Task 2: Run/cache layer and spectrum extractors

**Files:**
- Modify: `notebooks_test/13_test_accDM_CDM_indistinguishability.ipynb`

**Interfaces:**
- Consumes: `K_NODES`, `L_MAX`, param builders from Task 1.
- Produces:
  - `run(params, key) -> dict` with keys `'pk'` (ndarray over K_NODES), `'ell'`, `'tt'`, `'te'`, `'ee'` (ndarrays, l=2..L_MAX). Cached in `_cache`.
  - `get_lcdm() -> dict`, `get_cold(f_acc) -> dict`, `get_warm(f_acc, m) -> dict` convenience wrappers.

- [ ] **Step 1: Add the run/cache code cell**

Insert a code cell:

```python
_cache = {}
def run(params, key):
    """Compute a CLASS model once; return P(k) on K_NODES and lensed Cl (l=2..L_MAX)."""
    if key in _cache:
        return _cache[key]
    M = Class(); M.set(params); M.compute()
    pk = np.array([M.pk(float(k), 0.0) for k in K_NODES])
    cl = M.lensed_cl(L_MAX)
    ell = cl['ell']
    sel = ell >= 2
    out = {'pk': pk, 'ell': ell[sel],
           'tt': cl['tt'][sel], 'te': cl['te'][sel], 'ee': cl['ee'][sel]}
    M.struct_cleanup(); M.empty()
    _cache[key] = out
    return out

def get_lcdm():
    return run(lcdm_params(), ('lcdm',))

def get_cold(f_acc):
    return run(coldlimit_params(f_acc), ('cold', f_acc))

def get_warm(f_acc, m):
    return run(accdm_params(f_acc, m), ('warm', f_acc, m))
```

- [ ] **Step 2: Add a single-run smoke-test markdown + code cell**

Insert a markdown cell: `## Smoke test: one warm run + LCDM reference`. Then a code cell:

```python
_warm = get_warm(0.1, 1e13)          # eta = 0.01
_lcdm = get_lcdm()
assert _warm['pk'].shape == K_NODES.shape
assert _warm['tt'].shape == _warm['ell'].shape
assert np.all(np.isfinite(_warm['pk'])) and np.all(np.isfinite(_warm['tt']))
assert _warm['ell'][0] == 2 and _warm['ell'][-1] == L_MAX
print('Task 2 run/cache OK: pk[0]={:.3e}, tt[100]={:.3e}'.format(_warm['pk'][0], _warm['tt'][100]))
```

- [ ] **Step 3: Verify in the classy env**

Run (user, on the accDM pyenv):
`pytest --nbmake notebooks_test/13_test_accDM_CDM_indistinguishability.ipynb -k "cell" -x`
Expected: the smoke-test cell prints finite `pk[0]` / `tt[100]` and no assertion fires.
(Authoring shell cannot run this — mark done once the user confirms.)

- [ ] **Step 4: Commit**

```bash
git add notebooks_test/13_test_accDM_CDM_indistinguishability.ipynb
git commit -m "nb13: run/cache layer + spectrum extractors, single-run smoke test"
```

---

### Task 3: Metric functions with inline unit tests (no CLASS)

**Files:**
- Modify: `notebooks_test/13_test_accDM_CDM_indistinguishability.ipynb`

**Interfaces:**
- Consumes: `K_NODES`, `V_SURVEY`, `F_SKY`.
- Produces (pure numpy, unit-tested):
  - `pk_maxdev(pk_acc, pk_ref) -> float`
  - `pk_significance(pk_acc, pk_ref, k=K_NODES, V=V_SURVEY) -> float`
  - `cl_maxdev(cl_acc, cl_ref) -> float`   (accepts one spectrum array)
  - `cmb_significance(acc, ref, f_sky=F_SKY) -> float`  (acc/ref are run-dicts; TT+EE+TE)

- [ ] **Step 1: Write the failing test cell first**

Insert a markdown cell `## Metrics (pure functions) + unit tests`, then a code cell:

```python
# --- unit tests (defined before the functions on purpose; run after next cell) ---
def _test_metrics():
    k = np.logspace(-3, 0, 60)
    a = np.ones_like(k) * 2.0
    # identical spectra -> zero on every metric
    assert pk_maxdev(a, a) == 0.0
    assert pk_significance(a, a, k, 1e6) == 0.0
    assert cl_maxdev(a, a) == 0.0
    # max fractional deviation is exact
    b = a.copy(); b[10] = a[10] * 1.05
    assert abs(pk_maxdev(b, a) - 0.05) < 1e-12
    assert abs(cl_maxdev(b, a) - 0.05) < 1e-12
    # pk_significance matches the hand formula on a flat 1% offset
    ref = np.ones_like(k); acc = ref * 1.01
    dk = np.gradient(k); Nm = k**2 * dk * 1e6 / (2*np.pi**2)
    want = np.sqrt(np.sum((0.01 / np.sqrt(2.0/Nm))**2))
    assert abs(pk_significance(acc, ref, k, 1e6) - want) < 1e-9
    # cmb_significance is zero for identical run-dicts
    d = {'ell': np.arange(2, 50), 'tt': np.ones(48), 'te': np.zeros(48), 'ee': np.ones(48)}
    assert cmb_significance(d, d) == 0.0
    print('metric unit tests PASSED')
```

- [ ] **Step 2: Run to verify it fails**

Run: execute the cell, then in a scratch call `_test_metrics()`.
Expected: `NameError: name 'pk_maxdev' is not defined` (functions not yet written).

- [ ] **Step 3: Write the metric implementations**

Insert a code cell:

```python
def pk_maxdev(pk_acc, pk_ref):
    return float(np.max(np.abs(pk_acc / pk_ref - 1.0)))

def pk_significance(pk_acc, pk_ref, k=K_NODES, V=V_SURVEY):
    dk = np.gradient(k)
    N_modes = k**2 * dk * V / (2.0 * np.pi**2)
    sigmaP_over_P = np.sqrt(2.0 / N_modes)
    resid = (pk_acc - pk_ref) / pk_ref / sigmaP_over_P
    return float(np.sqrt(np.sum(resid**2)))

def cl_maxdev(cl_acc, cl_ref):
    return float(np.max(np.abs(cl_acc / cl_ref - 1.0)))

def cmb_significance(acc, ref, f_sky=F_SKY):
    """Cosmic-variance-limited Knox chi over TT+EE+TE (cross-covariance neglected)."""
    ell = ref['ell']; norm = (2.0 * ell + 1.0) * f_sky
    tt_r, ee_r, te_r = ref['tt'], ref['ee'], ref['te']
    var_tt = 2.0 * tt_r**2 / norm
    var_ee = 2.0 * ee_r**2 / norm
    var_te = (te_r**2 + tt_r * ee_r) / norm
    sig2 = np.sum((acc['tt'] - tt_r)**2 / var_tt)
    sig2 += np.sum((acc['ee'] - ee_r)**2 / var_ee)
    sig2 += np.sum((acc['te'] - te_r)**2 / var_te)
    return float(np.sqrt(sig2))
```

- [ ] **Step 4: Run the test cell to verify it passes**

Insert a code cell:

```python
_test_metrics()
```

Expected output: `metric unit tests PASSED`. (This cell runs in the authoring shell too if numpy is present — it needs no CLASS.)

- [ ] **Step 5: Commit**

```bash
git add notebooks_test/13_test_accDM_CDM_indistinguishability.ipynb
git commit -m "nb13: P(k)/CMB distinguishability metrics + inline unit tests"
```

---

### Task 4: Threshold-extraction helper with inline unit tests (no CLASS)

**Files:**
- Modify: `notebooks_test/13_test_accDM_CDM_indistinguishability.ipynb`

**Interfaces:**
- Produces: `threshold_mass(masses, metric_values, cut) -> float`
  Returns the interpolated mass (on log10 axis) where a **decreasing** metric first
  drops below `cut`. `masses[0]` if already below at the smallest mass; `np.inf` if
  never below within the grid.

- [ ] **Step 1: Write the failing test cell**

Insert a code cell:

```python
def _test_threshold():
    m = np.logspace(11, 19, 9)
    vals = np.array([1.0, 0.8, 0.6, 0.4, 0.2, 0.1, 0.05, 0.02, 0.01])   # decreasing
    # cut between vals[4]=0.2 and vals[5]=0.1 -> mass between m[4] and m[5]
    t = threshold_mass(m, vals, 0.15)
    assert m[4] < t < m[5]
    # already below everywhere -> smallest mass
    assert threshold_mass(m, vals*1e-3, 0.15) == m[0]
    # never below -> inf
    assert threshold_mass(m, vals*1e3, 0.15) == np.inf
    print('threshold unit tests PASSED')
```

- [ ] **Step 2: Run to verify it fails**

Run the cell then call `_test_threshold()`.
Expected: `NameError: name 'threshold_mass' is not defined`.

- [ ] **Step 3: Implement `threshold_mass`**

Insert a code cell:

```python
def threshold_mass(masses, metric_values, cut):
    masses = np.asarray(masses, float); vals = np.asarray(metric_values, float)
    logm = np.log10(masses)
    below = vals < cut
    if below.all():
        return float(masses[0])
    if not below.any():
        return np.inf
    i = int(np.argmax(below))            # first index below the cut
    if i == 0:
        return float(masses[0])
    x0, x1, y0, y1 = logm[i-1], logm[i], vals[i-1], vals[i]
    xc = x0 + (cut - y0) * (x1 - x0) / (y1 - y0)
    return float(10**xc)
```

- [ ] **Step 4: Run the test cell to verify it passes**

Insert a code cell: `_test_threshold()`.
Expected output: `threshold unit tests PASSED`.

- [ ] **Step 5: Commit**

```bash
git add notebooks_test/13_test_accDM_CDM_indistinguishability.ipynb
git commit -m "nb13: log-interpolated threshold-mass extractor + unit tests"
```

---

### Task 5: Run the scan and assemble the results table

**Files:**
- Modify: `notebooks_test/13_test_accDM_CDM_indistinguishability.ipynb`

**Interfaces:**
- Consumes: everything above.
- Produces: `RESULTS` — nested dict `RESULTS[f_acc][ref_name][metric_name] -> np.ndarray`
  aligned to `MASS_GRID`, with `ref_name in {'cold','lcdm'}` and
  `metric_name in {'pk_dev','pk_sig','cmb_dev','cmb_sig'}`. Also `ETA_GRID = eta_of(MASS_GRID)`.

- [ ] **Step 1: Add the scan markdown + driver cell**

Insert a markdown cell `## Scan: distinguishability vs mass, per f_acc and per reference`, then a code cell:

```python
ETA_GRID = eta_of(MASS_GRID)
_lcdm = get_lcdm()
RESULTS = {}
for f_acc in F_SEQ:
    cold = get_cold(f_acc)
    per_ref = {r: {k: np.empty(len(MASS_GRID))
                   for k in ('pk_dev', 'pk_sig', 'cmb_dev', 'cmb_sig')}
               for r in ('cold', 'lcdm')}
    for j, m in enumerate(MASS_GRID):
        warm = get_warm(f_acc, m)
        for ref_name, ref in (('cold', cold), ('lcdm', _lcdm)):
            per_ref[ref_name]['pk_dev'][j]  = pk_maxdev(warm['pk'], ref['pk'])
            per_ref[ref_name]['pk_sig'][j]  = pk_significance(warm['pk'], ref['pk'])
            per_ref[ref_name]['cmb_dev'][j] = cl_maxdev(warm['tt'], ref['tt'])
            per_ref[ref_name]['cmb_sig'][j] = cmb_significance(warm, ref)
    RESULTS[f_acc] = per_ref
    print('f_acc={:<5} done  (pk_dev vs cold: {:.3e} -> {:.3e})'.format(
        f_acc, per_ref['cold']['pk_dev'][0], per_ref['cold']['pk_dev'][-1]))
```

- [ ] **Step 2: Add a results-integrity assert cell**

Insert a code cell:

```python
for f_acc in F_SEQ:
    for ref_name in ('cold', 'lcdm'):
        for k, arr in RESULTS[f_acc][ref_name].items():
            assert arr.shape == MASS_GRID.shape and np.all(np.isfinite(arr)), (f_acc, ref_name, k)
print('Task 5 RESULTS complete and finite')
```

- [ ] **Step 3: Verify in the classy env**

Run (user): execute the notebook top-to-bottom in the `accDM` env.
Expected: the scan prints one line per `f_acc`; `pk_dev vs cold` decreases from the warm
(small-mass) end to the cold (large-mass) end; the integrity cell prints its confirmation.

- [ ] **Step 4: Commit**

```bash
git add notebooks_test/13_test_accDM_CDM_indistinguishability.ipynb
git commit -m "nb13: run the mass x f_acc scan, assemble RESULTS table"
```

---

### Task 6: Threshold table + diagnostic and summary plots

**Files:**
- Modify: `notebooks_test/13_test_accDM_CDM_indistinguishability.ipynb`

**Interfaces:**
- Consumes: `RESULTS`, `MASS_GRID`, `ETA_GRID`, `threshold_mass`, plot constants.
- Produces: `CUTS` dict; `THRESHOLDS[f_acc][ref_name][metric_name] -> mass`; printed table; figures.

- [ ] **Step 1: Add the cuts + threshold-table cell**

Insert a markdown cell `## Threshold masses` then a code cell:

```python
CUTS = {'pk_dev': 0.01, 'pk_sig': 1.0, 'cmb_dev': 0.01, 'cmb_sig': 1.0}  # indistinguishability cuts
THRESHOLDS = {}
print('{:>6} {:>6} {:>10} {:>10} {:>10} {:>10}'.format(
    'f_acc', 'ref', 'pk_dev', 'pk_sig', 'cmb_dev', 'cmb_sig'))
for f_acc in F_SEQ:
    THRESHOLDS[f_acc] = {}
    for ref_name in ('cold', 'lcdm'):
        row = {mk: threshold_mass(MASS_GRID, RESULTS[f_acc][ref_name][mk], CUTS[mk])
               for mk in CUTS}
        THRESHOLDS[f_acc][ref_name] = row
        fmt = lambda x: '{:.2e}'.format(x) if np.isfinite(x) else '  >grid'
        print('{:>6} {:>6} {:>10} {:>10} {:>10} {:>10}'.format(
            f_acc, ref_name, fmt(row['pk_dev']), fmt(row['pk_sig']),
            fmt(row['cmb_dev']), fmt(row['cmb_sig'])))
print('\nThreshold = daughter mass [GeV] above which accDM is indistinguishable from that reference.')
```

- [ ] **Step 2: Add the distinguishability-vs-mass figure cell**

Insert a code cell (2x2: P(k) dev, P(k) sig, CMB dev, CMB sig; one line per f_acc; cold=solid, lcdm=dashed):

```python
metrics = [('pk_dev', r'$\max_k|P_{\rm acc}/P_{\rm ref}-1|$', CUTS['pk_dev'], True),
           ('pk_sig', r'$P(k)$ significance $\sqrt{\chi^2}$', CUTS['pk_sig'], True),
           ('cmb_dev', r'$\max_\ell|C^{TT}_{\rm acc}/C^{TT}_{\rm ref}-1|$', CUTS['cmb_dev'], True),
           ('cmb_sig', r'CMB significance $\sqrt{\chi^2}$', CUTS['cmb_sig'], True)]
fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
for ax, (mk, ylab, cut, logy) in zip(axes.ravel(), metrics):
    for f_acc, c in zip(F_SEQ, qual_colors):
        ax.plot(MASS_GRID, RESULTS[f_acc]['cold'][mk], '-',  color=c,
                label=r'$f_{\rm acc}=%g$ (cold)' % f_acc)
        ax.plot(MASS_GRID, RESULTS[f_acc]['lcdm'][mk], '--', color=c, alpha=0.7)
    ax.axhline(cut, color='k', ls=':', lw=1.2)
    ax.set_xscale('log');  ax.set_xlabel(r'$m\ [\mathrm{GeV}]$');  ax.set_ylabel(ylab)
    if logy: ax.set_yscale('log')
    ax.grid(True, which='both', alpha=0.3)
axes[0, 0].legend(loc='best', fontsize=8)
fig.suptitle(r'accDM $\to$ CDM: solid = vs cold limit, dashed = vs $\Lambda$CDM  '
             r'(dotted = indistinguishability cut)')
plt.show()
```

- [ ] **Step 3: Add the diagnostic-panels cell (spectra ratios at three masses)**

Insert a code cell (fixed `f_acc=0.1`, three masses spanning warm/threshold/cold vs cold ref):

```python
f_show = 0.1
m_show = [MASS_GRID[0], MASS_GRID[len(MASS_GRID)//2], MASS_GRID[-1]]
cold = get_cold(f_show)
fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
for m, c in zip(m_show, qual_colors):
    w = get_warm(f_show, m)
    axL.semilogx(K_NODES, w['pk']/cold['pk'], color=c,
                 label=r'$m=%.0e$ GeV ($\eta=%.1e$)' % (m, eta_of(m)))
    axR.semilogx(cold['ell'], w['tt']/cold['tt'] - 1.0, color=c)
axL.axhline(1, color='k', ls='--', lw=1); axL.set_xlabel(r'$k\ [\mathrm{Mpc}^{-1}]$')
axL.set_ylabel(r'$P_{\rm acc}/P_{\rm cold}$'); axL.legend(fontsize=8); axL.grid(True, which='both', alpha=0.3)
axR.axhline(0, color='k', ls='--', lw=1); axR.set_xlabel(r'$\ell$')
axR.set_ylabel(r'$\Delta C_\ell^{TT}/C_\ell^{TT}$'); axR.grid(True, which='both', alpha=0.3)
fig.suptitle(r'Free-streaming signature shrinks as $m$ grows ($f_{\rm acc}=0.1$, vs cold limit)')
plt.show()
```

- [ ] **Step 4: Verify in the classy env**

Run (user): execute the notebook; confirm the table prints thresholds that increase with
`f_acc` (more accelerated fraction needs a heavier/colder daughter to hide), the 2x2 curves
fall monotonically with mass and cross the dotted cuts, and the diagnostic ratios flatten
toward 1 (P) and 0 (ΔC) at the largest mass.

- [ ] **Step 5: Commit**

```bash
git add notebooks_test/13_test_accDM_CDM_indistinguishability.ipynb
git commit -m "nb13: threshold table + distinguishability and diagnostic plots"
```

---

### Task 7: Regression asserts, q-grid convergence check, caveats

**Files:**
- Modify: `notebooks_test/13_test_accDM_CDM_indistinguishability.ipynb`

**Interfaces:**
- Consumes: `RESULTS`, `get_warm`, `pk_maxdev`, `Q_BINS`, `accdm_params`.
- Produces: hard asserts (nbmake regression) + a numeric-floor estimate `Q_FLOOR`.

- [ ] **Step 1: Add the monotonic-trend + cold-end regression assert cell**

Insert a markdown cell `## Regression asserts (colder -> more CDM-like)` then a code cell:

```python
def _mostly_decreasing(a, tol=0.15):
    """Allow small non-monotonic wiggles from q-grid noise; require net decrease."""
    a = np.asarray(a)
    steps = np.diff(a)
    assert a[0] > a[-1], 'metric must be smaller at the cold (large-mass) end'
    assert np.mean(steps <= tol*abs(a[0])) > 0.7, 'metric not broadly decreasing with mass'

for f_acc in F_SEQ:
    for ref_name in ('cold', 'lcdm'):
        _mostly_decreasing(RESULTS[f_acc][ref_name]['pk_dev'])
        _mostly_decreasing(RESULTS[f_acc][ref_name]['pk_sig'])
    # at the heaviest mass, accDM must be within the fixed P(k) tolerance of the cold limit
    assert RESULTS[f_acc]['cold']['pk_dev'][-1] < CUTS['pk_dev'], \
        'cold-end P(k) still distinguishable at f_acc=%g' % f_acc
print('Regression asserts PASSED (monotone trend + cold-end indistinguishability)')
```

- [ ] **Step 2: Add the q-grid numerical-floor cell**

Insert a markdown cell `## q-grid convergence: is the cold-end floor physical or numerical?`
then a code cell:

```python
# Re-run the coldest mass at a finer daughter grid; the accDM<->accDM difference
# bounds the numeric floor. Thresholds sitting below Q_FLOOR are numerics-limited.
_m = MASS_GRID[-1]
p_coarse = accdm_params(0.1, _m)
p_fine = accdm_params(0.1, _m); p_fine['ncdm_N_momentum_bins'] = '15, {:d}'.format(4*Q_BINS)
coarse = run(p_coarse, ('warm', 0.1, _m))                 # cached from the scan
fine   = run(p_fine,   ('qconv', 0.1, _m, 4*Q_BINS))
Q_FLOOR = pk_maxdev(coarse['pk'], fine['pk'])
print('q-grid numeric floor on max|dP/P|  = {:.2e}  (Q_BINS={} vs {})'.format(
    Q_FLOOR, Q_BINS, 4*Q_BINS))
print('P(k) tolerance cut = {:.0e}. Floor is {} the cut.'.format(
    CUTS['pk_dev'], 'BELOW' if Q_FLOOR < CUTS['pk_dev'] else 'ABOVE'))
assert Q_FLOOR < CUTS['pk_dev'], 'q-grid noise exceeds the P(k) cut; raise Q_BINS'
```

- [ ] **Step 2b: Consider spawning a follow-up if the floor is marginal**

If `Q_FLOOR` is within ~2x of `CUTS['pk_dev']`, note in the final markdown that `Q_BINS`
should be raised for a production threshold; this is expected given memory
`accdm-fluid-f-boundary` (q-grid convergence scales with f).

- [ ] **Step 3: Add the closing caveats markdown cell**

Insert a markdown cell:

```markdown
### Notes and caveats
- **Two "CDM" references.** Solid curves compare to the model's own cold limit (`eta -> 0`),
  isolating free-streaming; dashed curves compare to plain LCDM and also fold in the decay's
  background effect. The cold-limit threshold is the cleaner physical statement.
- **Metrics.** (A) `max|ratio-1|` on the spectra vs a fixed tolerance; (B) cosmic-variance-limited
  `sqrt(chi^2)` (P(k): mode counting in `V_SURVEY`; CMB: Knox over TT+EE+TE, cross-covariance
  neglected as a detectability proxy — see the non-goals in the spec). These are optimistic
  ("if CV-limited can't tell them apart, nothing can").
- **q-grid floor.** The cold-end signal is tiny; the convergence cell bounds the numeric floor.
  Thresholds below `Q_FLOOR` are numerics-limited, not physical — raise `Q_BINS` to push lower.
- **CMB vs P(k).** For cold daughters the CMB effect enters mainly via late growth/lensing, so
  `P(k)` sets the binding (largest-mass) threshold; CMB typically becomes indistinguishable at
  lower mass.
- **Scope.** Fixed decay sector (`kappa_acc`, `a_t_acc`); no large-`kappa_acc` regime (known
  WONTFIX). The chi^2 is a single-parameter detectability proxy, not a Fisher/MCMC forecast.
- Run: `pytest --nbmake notebooks_test/13_test_accDM_CDM_indistinguishability.ipynb`.
```

- [ ] **Step 4: Verify full notebook in the classy env**

Run (user): `pytest --nbmake notebooks_test/13_test_accDM_CDM_indistinguishability.ipynb`.
Expected: all cells execute, all asserts pass end to end.

- [ ] **Step 5: Commit**

```bash
git add notebooks_test/13_test_accDM_CDM_indistinguishability.ipynb
git commit -m "nb13: regression asserts, q-grid convergence floor, caveats"
```

---

## Self-Review

**Spec coverage:**
- Parameter mapping (eta=1e11/m, masses, daughter m_ncdm, exact quadrature, fixed decay) → Task 1. ✓
- Both references (cold-limit per f_acc + plain LCDM) → Tasks 1–2, used in Task 5. ✓
- P(k) metric A + B (tolerance + survey chi^2 with V_SURVEY) → Task 3. ✓
- CMB metric A + B (tolerance + Knox TT+TE+EE, f_sky) → Task 3. ✓
- Scan f_acc {0.01,0.05,0.1} x mass grid, cached → Task 5. ✓
- Threshold extraction / table → Tasks 4 + 6. ✓
- Distinguishability-vs-mass curves + diagnostic panels → Task 6. ✓
- Hard asserts (monotone + cold-end) → Task 7. ✓
- Risks: q-grid floor (Task 7), gauge/sub-horizon (k-floor in Task 1), CMB subtlety (caveats) → covered. ✓

**Placeholder scan:** No TBDs; every code step carries full code. Verification of CLASS-running
cells is explicitly delegated to the user's classy env (a hard environment constraint, not a
placeholder).

**Type consistency:** `run()` returns `{'pk','ell','tt','te','ee'}`; `cmb_significance` consumes
exactly those keys; `RESULTS[f_acc][ref_name][metric]` keys `pk_dev/pk_sig/cmb_dev/cmb_sig` match
`CUTS` and the plot loop; `threshold_mass(masses, values, cut)` signature consistent across Tasks 4/6.

---

## Notes for the implementer

- **You cannot run CLASS while authoring.** Build every cell with `NotebookEdit`. The pure-numpy
  test cells (Tasks 3, 4) you *can* run locally if numpy is importable; the CLASS cells are marked
  "verify in the classy env" and are confirmed by the user.
- Follow notebook 5 (`5_test_Pk_freestreaming.ipynb`) for house style: shared massive-nu sector,
  `_cache` pattern, qualitative/monotonic asserts.
- Keep `Q_BINS`, `V_SURVEY`, `MASS_GRID`, `F_SEQ`, `CUTS` as top-of-notebook knobs so the user can
  trade accuracy vs runtime without hunting through cells.
