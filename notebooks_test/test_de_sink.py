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
