"""accDM daughter births as integration breakpoints: rk must integrate through the birth peak.
Run after building classy:  python -m pytest notebooks_test/test_birth_breakpoints.py -v -s
"""
import time
import numpy as np
from classy import Class

KM = np.logspace(np.log10(0.02), np.log10(2.0), 400)    # 1/Mpc
A_REC = 1.0/1091.0


def params(extra=None, f_acc=0.1, mass=1e11, kappa=12.1, a_t=0.133, n_q=51):
    """accDM run on the qm_acc_birth grid, configured as notebook 30 (lensed C_l, theta_s fixed)."""
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


def test_tight_tolerance_completes():
    """Without breakpoints the rk step underflows at the birth peak (a ~ 0.11) at this tolerance."""
    _, t_tight = pk(params({'tol_perturbations_integration': 1e-7}))
    _, t_default = pk(params())
    print('run time: default tolerance {:.0f} s, tol 1e-7 {:.0f} s'.format(t_default, t_tight))
