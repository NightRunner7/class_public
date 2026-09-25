# DE Sink Observables Notebook Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A notebook comparing background densities, lensed CMB spectra and P(k) for accDM with the DE sink off and on, over a mass scan, against ΛCDM.

**Architecture:** One self-contained notebook: a setup cell with parameters and a `run()` helper, one cell that performs all runs into a dict, a sanity-check cell, three figure cells and a summary table. No CLASS changes.

**Tech Stack:** classy, numpy, matplotlib.

**Spec:** `docs/superpowers/specs/2026-09-25-de-sink-observables-notebook-design.md`

## Global Constraints

- `FIX = '100*theta_s'` (1.041783) by default, or `'H0'` (67.32).
- accDM: `ncdm_quadrature_strategy = '0, 5'`, 51 daughter bins, `evolver = 0`, synchronous, `ncdm_fluid_approximation = 3`, ω_cdm rescaled as in nb21/nb28.
- `MASSES = [1e11, 1e12, 1e13, 1e16]` GeV, η = 1e11/m, f_acc = 0.1, κ = 12.1, a_t = 0.133.
- Lensed C_ℓ to ℓ = 2500; P(k, z = 0) on k ∈ [1e-4, 5] /Mpc.
- Sanity bound: on/off < 1e-3 in TT and P(k) at the heaviest mass.
- Build/run in the user's `accDM` env; notebooks are run with Run All in the IDE.

---

### Task 1: Notebook `29_de_sink_observables.ipynb`

**Files:**
- Create: `notebooks_test/29_de_sink_observables.ipynb` (cells below, in order; markdown cells from `markdown` fences, code cells from `python` fences)

**Interfaces:**
- Produces (inside the notebook): `run(params) -> dict(bg, cl, pk, h, '100*theta_s', sigma8)`, `runs` dict keyed by `'lcdm'` and `(mass, sink)`, `ok` list of masses with both runs, `sorted_bg(bg) -> (a, dict)`.

- [ ] **Step 1: Write the cells**

Cell 1:
```markdown
# 29 - DE sink: background, CMB and P(k)

Compares accDM with `acc_de_sink` off and on, and both against ΛCDM, over a mass scan with the
default kick η = 10¹¹ GeV / m_acc. The sink is a w = −1 component that pays the daughters' kick
energy, so it is larger in the past: ρ_de_acc(a) = η f_acc ρ_cdm,0 J(a), with J(0) ≈ 471 for
κ = 12.1, a_t = 0.133.

`FIX` selects what is held fixed between runs: `'100*theta_s'` (H0 is solved by shooting, closest
to what a fit sees) or `'H0'` (the peak shift from the changed sound horizon then dominates ΔC_ℓ).
```

Cell 2:
```python
import numpy as np
import matplotlib.pyplot as plt
from classy import Class

FIX = '100*theta_s'                       # or 'H0'
THETA_S, H0 = 1.041783, 67.32
BASE = {'omega_b': 0.022383, 'A_s': 2.1005829616811546e-9, 'n_s': 0.96605,
        'tau_reio': 0.0543, 'N_ur': 0.00441}
OMEGA_CDM0 = 0.12011
A_REC = 1.0/1091.0

MASSES = [1e11, 1e12, 1e13, 1e16]         # GeV; eta = 1e11/m
F_ACC, KAPPA, A_T = 0.1, 12.1, 0.133
N_Q = 51                                  # qm_acc_birth daughter bins (odd)
L_MAX = 2500
K = np.logspace(-4, np.log10(5.0), 300)   # 1/Mpc

# rk evolver (ndf15 cannot handle the daughter hierarchy); exact ncdm hierarchy
COMMON = {'output': 'tCl,pCl,lCl,mPk', 'lensing': 'yes', 'l_max_scalars': L_MAX,
          'P_k_max_1/Mpc': 10.0, 'z_max_pk': 0.0,
          'gauge': 'synchronous', 'evolver': 0, 'ncdm_fluid_approximation': 3}


def fixed():
    return {FIX: THETA_S if FIX == '100*theta_s' else H0}


def lcdm_params():
    return {**BASE, **COMMON, **fixed(), 'omega_cdm': OMEGA_CDM0,
            'N_ncdm': 1, 'deg_ncdm': '3', 'm_ncdm': '0.02', 'T_ncdm': '0.71611',
            'ncdm_quadrature_strategy': '0', 'ncdm_N_momentum_bins': '15'}


def accdm_params(mass, sink):
    # omega_cdm rescaled so the total matter at recombination matches LCDM
    ocdm = OMEGA_CDM0/(1 + F_ACC*(1 - A_REC**KAPPA)/(1 + (A_REC/A_T)**KAPPA))
    return {**BASE, **COMMON, **fixed(), 'omega_cdm': ocdm,
            'vary_Gamma_acc': 'yes', 'kappa_acc': KAPPA, 'a_t_acc': A_T,
            'f_acc': F_ACC, 'eta_acc': 1e11/mass,
            'm_acc_in_GeV': mass, 'm_cdm_in_GeV': mass,
            'N_ncdm': 2, 'deg_ncdm': '3, 1', 'm_ncdm': '0.02, {:.6e}'.format(mass*1e9),
            'T_ncdm': '0.71611, 1', 'ncdm_quadrature_strategy': '0, 5',
            'ncdm_N_momentum_bins': '15, {:d}'.format(N_Q),
            'acc_de_sink': 'yes' if sink else 'no'}


def run(params):
    cosmo = Class()
    cosmo.set(params)
    try:
        cosmo.compute()
        out = dict(bg=cosmo.get_background(), cl=cosmo.lensed_cl(L_MAX),
                   pk=np.array([cosmo.pk(k, 0.0) for k in K]))
        out.update(cosmo.get_current_derived_parameters(['h', '100*theta_s', 'sigma8']))
    finally:
        cosmo.struct_cleanup()
        cosmo.empty()
    return out


def sorted_bg(bg):
    """Background columns sorted by increasing a."""
    a = 1.0/(1.0 + np.asarray(bg['z']))
    order = np.argsort(a)
    return a[order], {key: np.asarray(val)[order] for key, val in bg.items()}


print('FIX = {}, masses = {}'.format(FIX, MASSES))
```

Cell 3:
```python
runs = {'lcdm': run(lcdm_params())}
for m in MASSES:
    for sink in (False, True):
        try:
            runs[(m, sink)] = run(accdm_params(m, sink))
        except Exception as err:
            print('m = {:.0e} GeV, sink {}: FAILED - {}'.format(m, 'on' if sink else 'off', err))
ok = [m for m in MASSES if (m, False) in runs and (m, True) in runs]
print('completed masses:', ['{:.0e}'.format(m) for m in ok])
```

Cell 4:
```python
# Sanity: the toggle works, and a negligible sink (heaviest mass) changes nothing.
m_ref = max(ok)
off, on = runs[(m_ref, False)], runs[(m_ref, True)]
assert '(.)rho_de_acc' in on['bg'] and '(.)rho_de_acc' not in off['bg']
d_tt = float(np.max(np.abs(on['cl']['tt'][2:]/off['cl']['tt'][2:] - 1)))
d_pk = float(np.max(np.abs(on['pk']/off['pk'] - 1)))
print('m = {:.0e} GeV (eta = {:.0e}): max|dTT/TT| = {:.1e}, max|dP/P| = {:.1e}'.format(
    m_ref, 1e11/m_ref, d_tt, d_pk))
assert max(d_tt, d_pk) < 1e-3, 'negligible sink changed the spectra'
```

Cell 5:
```markdown
## Background

Top: energy densities in units of today's critical density, sink on. Bottom: change in the
expansion rate from switching the sink on.
```

Cell 6:
```python
COMPONENTS = [('(.)rho_cdm', 'CDM'), ('(.)rho_acc_cdm', 'parent'), ('(.)rho_ncdm[1]', 'daughter'),
              ('(.)rho_lambda', r'$\Lambda$'), ('(.)rho_de_acc', 'DE sink')]

fig, axes = plt.subplots(2, len(ok), figsize=(4*len(ok), 6.5), sharex=True, sharey='row',
                         constrained_layout=True, squeeze=False)
for j, m in enumerate(ok):
    a, b = sorted_bg(runs[(m, True)]['bg'])
    a_off, b_off = sorted_bg(runs[(m, False)]['bg'])
    rho_crit0 = b['(.)rho_crit'][-1]
    ax = axes[0, j]
    for key, label in COMPONENTS:
        ax.loglog(a, np.where(b[key] > 0, b[key], np.nan)/rho_crit0, label=label)
    ax.axvline(A_T, color='grey', ls=':')
    ax.set_title(r'$m = 10^{{{:.0f}}}$ GeV, $\eta = {:.0e}$'.format(np.log10(m), 1e11/m))
    ax.set_xlim(1e-4, 1)
    ax.set_ylim(1e-4, 1e13)
    ax.grid(alpha=0.3)
    ax = axes[1, j]
    H_off = np.interp(np.log(a), np.log(a_off), b_off['H [1/Mpc]'])
    ax.semilogx(a, b['H [1/Mpc]']/H_off - 1)
    ax.axvline(A_T, color='grey', ls=':')
    ax.set_xlabel(r'$a$')
    ax.grid(alpha=0.3)
axes[0, 0].set_ylabel(r'$\rho_i/\rho_{\rm crit,0}$')
axes[1, 0].set_ylabel(r'$H_{\rm on}/H_{\rm off} - 1$')
axes[0, 0].legend(fontsize=8)
plt.show()
```

Cell 7:
```markdown
## CMB

Relative differences in the lensed spectra. TE is normalised by √(C_ℓ^TT C_ℓ^EE) because it
crosses zero. Grey: cosmic variance √(2/(2ℓ+1)) per multipole.
```

Cell 8:
```python
ell = runs['lcdm']['cl']['ell'][2:]
cv = np.sqrt(2.0/(2*ell + 1))


def cl(r, key):
    return r['cl'][key][2:]


def delta(x, y, key):
    if key == 'te':
        return (cl(x, 'te') - cl(y, 'te'))/np.sqrt(cl(y, 'tt')*cl(y, 'ee'))
    return cl(x, key)/cl(y, key) - 1


ROWS = [('tt', 'TT'), ('ee', 'EE'), ('te', 'TE'), ('pp', r'$\phi\phi$')]
COLS = [('sink off vs LCDM', lambda m: (runs[(m, False)], runs['lcdm'])),
        ('sink on vs LCDM', lambda m: (runs[(m, True)], runs['lcdm'])),
        ('sink on vs off', lambda m: (runs[(m, True)], runs[(m, False)]))]

fig, axes = plt.subplots(len(ROWS), len(COLS), figsize=(13, 11), sharex=True,
                         constrained_layout=True)
for i, (key, name) in enumerate(ROWS):
    for j, (title, pair) in enumerate(COLS):
        ax = axes[i, j]
        if key in ('tt', 'ee'):
            ax.fill_between(ell, -cv, cv, color='0.88', lw=0)
        for m in ok:
            ax.plot(ell, delta(*pair(m), key), lw=1,
                    label=r'$10^{{{:.0f}}}$ GeV'.format(np.log10(m)))
        ax.axhline(0, color='k', lw=0.6)
        ax.set_xscale('log')
        ax.grid(alpha=0.3)
        if i == 0:
            ax.set_title(title)
        if j == 0:
            ax.set_ylabel(r'$\Delta$' + name)
        if i == len(ROWS) - 1:
            ax.set_xlabel(r'$\ell$')
axes[0, 0].legend(fontsize=8)
plt.show()
```

Cell 9:
```markdown
## Matter power spectrum (z = 0)
```

Cell 10:
```python
fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), sharex=True, constrained_layout=True)
p_ref = runs['lcdm']['pk']
for m in ok:
    label = r'$10^{{{:.0f}}}$ GeV'.format(np.log10(m))
    p_off, p_on = runs[(m, False)]['pk'], runs[(m, True)]['pk']
    axes[0].semilogx(K, p_off/p_ref - 1, label=label)
    axes[1].semilogx(K, p_on/p_ref - 1, label=label)
    axes[2].semilogx(K, p_on/p_off - 1, label=label)
for ax, title in zip(axes, ['sink off vs LCDM', 'sink on vs LCDM', 'sink on vs off']):
    ax.axhline(0, color='k', lw=0.6)
    ax.set_title(title)
    ax.set_xlabel(r'$k\ [\mathrm{Mpc}^{-1}]$')
    ax.grid(alpha=0.3)
axes[0].set_ylabel(r'$\Delta P/P$')
axes[0].legend(fontsize=8)
plt.show()
```

Cell 11:
```python
free = 'H0' if FIX == '100*theta_s' else '100*theta_s'
value = (lambda r: 100*r['h']) if free == 'H0' else (lambda r: r['100*theta_s'])

print('FIX = {}; LCDM: {} = {:.4f}, sigma8 = {:.4f}'.format(
    FIX, free, value(runs['lcdm']), runs['lcdm']['sigma8']))
print('{:>8} {:>8} {:>10} {:>10} {:>10} {:>8} {:>8} {:>10}'.format(
    'm [GeV]', 'eta', 'max Om_de', free + ' off', free + ' on', 's8 off', 's8 on', 'max dTT'))
for m in ok:
    off, on = runs[(m, False)], runs[(m, True)]
    _, b = sorted_bg(on['bg'])
    om_de = float(np.max(b['(.)rho_de_acc']))/b['(.)rho_crit'][-1]
    d_tt = float(np.max(np.abs(cl(on, 'tt')/cl(off, 'tt') - 1)))
    print('{:>8.0e} {:>8.0e} {:>10.2e} {:>10.4f} {:>10.4f} {:>8.4f} {:>8.4f} {:>10.2e}'.format(
        m, 1e11/m, om_de, value(off), value(on), off['sigma8'], on['sigma8'], d_tt))
```

- [ ] **Step 2: Assemble the notebook**

Build `notebooks_test/29_de_sink_observables.ipynb` (nbformat 4.4, kernel metadata copied from `28_test_de_sink_perturbations.ipynb`) from the eleven cells above, in order.

- [ ] **Step 3: Run it**

In the IDE, with the `accDM` kernel: Run All with `FIX = '100*theta_s'`, then again with `FIX = 'H0'`.
Expected: cell 4 passes; three figures render; the summary table lists all four masses (or the
failed ones are reported in cell 3).

- [ ] **Step 4: Commit**

```bash
git add notebooks_test/29_de_sink_observables.ipynb
git commit -m "Add notebook comparing background, CMB and P(k) with the DE sink off and on"
```
