"""accdm_q_bins_a_t_slope/_ref: more strategy-5 daughter bins for early transitions (notebook 24).
Run after building classy:  python notebooks_test/test_accdm_q_bins_a_t.py
"""
import sys

import numpy as np
from classy import Class

from test_accdm_input_checks import accdm_params, c_stdout, expect_error

# an early, steep transition: notebook 24 needs ~800 bins at a_t = 1e-3
EARLY = dict(kappa_acc=12.1, a_t_acc=1e-3, f_acc=1.0, m_acc_in_GeV=1e16, m_cdm_in_GeV=1e16,
             m_ncdm="0.02, {:.6e}".format(1e16*1e9), output="mPk")


def observables(params):
    cosmo = Class()
    cosmo.set(params)
    try:
        cosmo.compute(["fourier"])
        bg = cosmo.get_background()
        return {"rho_acc": bg["(.)rho_ncdm[1]"], "pk": np.array([cosmo.pk_cb(k, 0.) for k in (0.01, 0.1, 0.5)]),
                "sigma8_cb": cosmo.sigma8_cb()}
    finally:
        cosmo.struct_cleanup()
        cosmo.empty()


def same(a, b):
    return all(np.array_equal(a[key], b[key]) for key in a)


def q_size(params, tag="early transition"):
    """Daughter bin count CLASS reports at input_verbose 1, from the line containing tag."""
    out = c_stdout(dict(params, input_verbose=1), ["background"])
    line = [l for l in out.splitlines() if tag in l]
    return int(line[-1].split("q_size = ")[1].split()[0]) if line else None


def test_negative_slope_is_rejected():
    expect_error(accdm_params(accdm_q_bins_a_t_slope=-0.1), "'accdm_q_bins_a_t_slope'")


def test_nonpositive_ref_is_rejected():
    expect_error(accdm_params(accdm_q_bins_a_t_ref=0.), "'accdm_q_bins_a_t_ref'")


def test_default_is_off():
    p = accdm_params(**EARLY, ncdm_N_momentum_bins="150, 201")
    assert same(observables(p), observables(dict(p, accdm_q_bins_a_t_slope=0.)))


def test_late_transition_is_unchanged():
    # accdm_params' a_t = 0.13 lies above the reference 1e-2
    p = accdm_params(output="mPk", ncdm_N_momentum_bins="150, 201")
    assert same(observables(p), observables(dict(p, accdm_q_bins_a_t_slope=0.6)))


def test_scaling_of_explicit_bins():
    # 201 (1e-2/1e-3)^0.6 = 800.4 -> 801
    assert q_size(accdm_params(**EARLY, ncdm_N_momentum_bins="150, 201", accdm_q_bins_a_t_slope=0.6)) == 801


def test_scaling_of_per_decade_bins():
    base = dict(accdm_params(**EARLY), accdm_q_bins_a_t_slope=0.6)
    n0 = q_size(base, tag="bins per decade")                  # the per-decade count, before scaling
    n = q_size(base)
    expected = int(np.ceil(n0*10**0.6))
    assert n in (expected, expected + 1), (n0, n)


def test_scaled_equals_explicit():
    scaled = observables(accdm_params(**EARLY, ncdm_N_momentum_bins="150, 201", accdm_q_bins_a_t_slope=0.6))
    explicit = observables(accdm_params(**EARLY, ncdm_N_momentum_bins="150, 801"))
    assert same(scaled, explicit)


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
