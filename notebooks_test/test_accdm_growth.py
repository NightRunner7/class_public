"""Scale-independent growth D, f ('gr.fac.'): the accDM parent and the clustering daughter count.
Run after building classy:  python notebooks_test/test_accdm_growth.py
"""
import sys

import numpy as np
from classy import Class
from scipy.interpolate import CubicSpline

from test_accdm_input_checks import accdm_params, expect_error

ZFID = 0.525                     # pyfowl's fiducial redshift for maglim


def chain_params(mass, f_tilde, **overrides):
    """Chain transition (kappa 12.1, a_t 0.133) at f_tilde; daughter at the default l_max_ncdm."""
    return accdm_params(m_acc_in_GeV=mass, m_cdm_in_GeV=mass, m_ncdm="0.02, {:.6e}".format(mass*1e9),
                        eta_acc=1e11/mass, kappa_acc=12.1, a_t_acc=0.133, f_acc=f_tilde/(1 - f_tilde),
                        output="mPk", z_max_pk=1.5, **overrides)


def growth(params):
    """gr.fac. f(ZFID), and f(ZFID) from linear P_cb at k = 0.1 h/Mpc."""
    cosmo = Class()
    cosmo.set(params)
    try:
        cosmo.compute(["fourier"])
        k = 0.1*cosmo.h()
        z = np.linspace(0., 1.5, 61)
        lnp = CubicSpline(z, np.log([cosmo.pk_cb_lin(k, zi) for zi in z]))
        return cosmo.scale_independent_growth_factor_f(ZFID), -0.5*(1 + ZFID)*lnp(ZFID, 1)
    finally:
        cosmo.struct_cleanup()
        cosmo.empty()


def test_negative_pivot_is_rejected():
    expect_error(accdm_params(accdm_growth_k=-0.1), "'accdm_growth_k'")


def test_cold_daughter_matches_cb_growth():
    # before the fix gr.fac. f was ~28% low here (rho_M = b + cdm only)
    f, f_cb = growth(chain_params(10**16.699, 0.5))
    assert abs(f/f_cb - 1) < 1e-2, f/f_cb - 1


def test_hot_daughter_is_left_out():
    # at 1e11 GeV the daughter free-streams on all scales of interest: the default weight
    # drops it; counting it fully (accdm_growth_k = 0) overshoots by ~40%
    f, f_cb = growth(chain_params(1e11, 0.5))
    f_all, _ = growth(chain_params(1e11, 0.5, accdm_growth_k=0.))
    assert abs(f/f_cb - 1) < 0.08, f/f_cb - 1          # parent memory leaves ~+5% (see notes)
    assert f_all/f_cb - 1 > 0.2, f_all/f_cb - 1


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
