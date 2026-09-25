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
