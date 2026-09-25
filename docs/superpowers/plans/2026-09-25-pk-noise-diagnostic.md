# P(k) Noise Diagnostic Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A notebook that identifies the source of the P_on/P_off jitter at m_acc = 1e11 GeV.

**Architecture:** One notebook: setup with a `pair(extra)` helper returning P(k) for sink off and on under the same precision settings, a noise metric, a null pair that shifts only the k nodes, a case scan, a table and a residual plot.

**Tech Stack:** classy, numpy, matplotlib.

**Spec:** `docs/superpowers/specs/2026-09-25-pk-noise-diagnostic-design.md`

## Global Constraints

- m_acc = 1e11 GeV, η = 1, f_acc = 0.1, κ = 12.1, a_t = 0.133, `100*theta_s` = 1.041783.
- accDM settings as nb29: `ncdm_quadrature_strategy = '0, 5'`, 51 bins, `evolver = 0`, synchronous, `ncdm_fluid_approximation = 3`.
- Metric on k ∈ [0.02, 2]/Mpc, degree-6 polynomial in ln k; target max abs r < 1e-3.
- Run with Run All in the user's IDE (`accDM` kernel).

---

### Task 1: Notebook `30_pk_noise_diagnostic.ipynb`

**Files:**
- Create: `notebooks_test/30_pk_noise_diagnostic.ipynb` (cells below, in order)

**Interfaces:**
- Produces (inside the notebook): `run(params) -> (pk array on KM, seconds)`, `params(sink, extra, n_q) -> dict`, `noise(p1, p2) -> (r, rms, max)`, `results` list of dicts.

- [ ] **Step 1: Write the cells**

Cell 1:
```markdown
# 30 - Where does the P(k) jitter come from?

Notebook 29 shows ±0.5% jitter in P_on/P_off at m_acc = 10¹¹ GeV for k ≈ 0.05–0.5/Mpc. Leading
hypothesis: the on and off runs sample different k nodes (with C_l output, CLASS sets
k_max_cmb ∝ 1/τ0 and the sink changes τ0), so their spline-interpolation errors do not cancel.

**Null pair:** two sink-off runs that differ only in `k_step_sub` (0.05 vs 0.0505): same physics,
nodes shifted by up to half a step. **Cases:** each is an on/off pair with one precision setting changed.

**Metric:** r(k) = ln(P₁/P₂) minus a smooth degree-6 fit in ln k, on k ∈ [0.02, 2]/Mpc.
Target: max|r| < 10⁻³ (emulator accuracy).
```

Cell 2:
```python
import time
import numpy as np
import matplotlib.pyplot as plt
from classy import Class

BASE = {'omega_b': 0.022383, 'A_s': 2.1005829616811546e-9, 'n_s': 0.96605,
        'tau_reio': 0.0543, 'N_ur': 0.00441, '100*theta_s': 1.041783}
OMEGA_CDM0 = 0.12011
A_REC = 1.0/1091.0
MASS, F_ACC, KAPPA, A_T = 1e11, 0.1, 12.1, 0.133
KM = np.logspace(np.log10(0.02), np.log10(2.0), 400)       # 1/Mpc
# same outputs as nb29: with Cls, k_max_cmb scales as 1/tau0 and shapes the k list
COMMON = {'output': 'tCl,pCl,lCl,mPk', 'lensing': 'yes', 'l_max_scalars': 2500,
          'P_k_max_1/Mpc': 10.0, 'z_max_pk': 0.0,
          'gauge': 'synchronous', 'evolver': 0, 'ncdm_fluid_approximation': 3}


def params(sink, extra=None, n_q=51):
    ocdm = OMEGA_CDM0/(1 + F_ACC*(1 - A_REC**KAPPA)/(1 + (A_REC/A_T)**KAPPA))
    p = {**BASE, **COMMON, 'omega_cdm': ocdm,
         'vary_Gamma_acc': 'yes', 'kappa_acc': KAPPA, 'a_t_acc': A_T,
         'f_acc': F_ACC, 'eta_acc': 1e11/MASS,
         'm_acc_in_GeV': MASS, 'm_cdm_in_GeV': MASS,
         'N_ncdm': 2, 'deg_ncdm': '3, 1', 'm_ncdm': '0.02, {:.6e}'.format(MASS*1e9),
         'T_ncdm': '0.71611, 1', 'ncdm_quadrature_strategy': '0, 5',
         'ncdm_N_momentum_bins': '15, {:d}'.format(n_q),
         'acc_de_sink': 'yes' if sink else 'no'}
    p.update(extra or {})
    return p


def run(p):
    cosmo = Class()
    cosmo.set(p)
    t0 = time.time()
    try:
        cosmo.compute()
        pk = np.array([cosmo.pk(k, 0.0) for k in KM])
    finally:
        cosmo.struct_cleanup()
        cosmo.empty()
    return pk, time.time() - t0


def noise(p1, p2):
    """Residual of ln(p1/p2) about a smooth degree-6 fit in ln k, with its rms and max."""
    x, y = np.log(KM), np.log(p1/p2)
    r = y - np.polyval(np.polyfit(x, y, 6), x)
    return r, float(np.sqrt(np.mean(r**2))), float(np.max(np.abs(r)))


print('ready: m = {:.0e} GeV, {} k points in [{}, {}] /Mpc'.format(MASS, len(KM), KM[0], KM[-1]))
```

Cell 3:
```python
CASES = [('default', {}, 51),
         ('k_step_sub 0.02', {'k_step_sub': 0.02}, 51),
         ('k_step_sub 0.01', {'k_step_sub': 0.01}, 51),
         ('k_per_decade_for_bao 200', {'k_per_decade_for_bao': 200}, 51),
         ('tol 1e-6', {'tol_perturbations_integration': 1e-6}, 51),
         ('tol 1e-7', {'tol_perturbations_integration': 1e-7}, 51),
         ('sampling 0.03', {'perturbations_sampling_stepsize': 0.03}, 51),
         ('101 daughter bins', {}, 101),
         ('smooth births', {'accdm_smooth_births': 1}, 51),
         ('mPk only (no Cls)', {'output': 'mPk', 'lensing': 'no'}, 51)]

results = []

# null pair: same physics, k nodes shifted
p_a, t_a = run(params(False))
p_b, t_b = run(params(False, {'k_step_sub': 0.0505}))
r, rms, mx = noise(p_a, p_b)
results.append(dict(case='NULL (off vs off, shifted k nodes)', r=r, rms=rms, max=mx, sec=t_a + t_b))
print('{:<36s} rms {:.1e}  max {:.1e}  ({:.0f} s)'.format(results[-1]['case'], rms, mx, t_a + t_b))

for name, extra, n_q in CASES:
    try:
        p_off, t_off = run(params(False, extra, n_q))
        p_on, t_on = run(params(True, extra, n_q))
    except Exception as err:
        print('{:<36s} FAILED - {}'.format(name, err))
        continue
    r, rms, mx = noise(p_on, p_off)
    results.append(dict(case=name, r=r, rms=rms, max=mx, sec=t_off + t_on))
    print('{:<36s} rms {:.1e}  max {:.1e}  ({:.0f} s)'.format(name, rms, mx, t_off + t_on))
```

Cell 4:
```python
fig, axes = plt.subplots(len(results), 1, figsize=(8, 1.6*len(results)), sharex=True,
                         sharey=True, constrained_layout=True)
for ax, res in zip(axes, results):
    ax.semilogx(KM, res['r'], lw=0.9)
    ax.axhspan(-1e-3, 1e-3, color='0.88', lw=0)
    ax.axhline(0, color='k', lw=0.5)
    ax.text(0.01, 0.8, res['case'], transform=ax.transAxes, fontsize=9)
    ax.grid(alpha=0.3)
axes[-1].set_xlabel(r'$k\ [\mathrm{Mpc}^{-1}]$')
fig.supylabel(r'residual of $\ln(P_1/P_2)$')
plt.show()
```

Cell 5:
```python
print('{:<36s} {:>9} {:>9} {:>8}'.format('case', 'rms', 'max', 'seconds'))
for res in results:
    print('{:<36s} {:>9.1e} {:>9.1e} {:>8.0f}'.format(res['case'], res['rms'], res['max'], res['sec']))

passing = [res for res in results[1:] if res['max'] < 1e-3]
null = results[0]
print()
print('null pair max = {:.1e}: grid mismatch alone {} the jitter'.format(
    null['max'], 'reproduces' if null['max'] > 0.3*results[1]['max'] else 'does not reproduce'))
if passing:
    best = min(passing, key=lambda res: res['sec'])
    print('cheapest case below 1e-3: {} ({:.0f} s per pair)'.format(best['case'], best['sec']))
else:
    print('no single setting brings max|r| below 1e-3; combine the two best next')
```

- [ ] **Step 2: Assemble** `notebooks_test/30_pk_noise_diagnostic.ipynb` (nbformat 4.4, kernel metadata from nb28) from the five cells.

- [ ] **Step 3: Run** with Run All in the IDE. Expected: the null pair and default rows print; all cases complete; table and verdict lines print.

- [ ] **Step 4: Commit**

```bash
git add notebooks_test/30_pk_noise_diagnostic.ipynb
git commit -m "Add P(k) noise diagnostic notebook for the DE sink comparison"
```
