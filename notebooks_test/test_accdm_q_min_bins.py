"""accdm_q_min_bins: floor on the strategy-5 daughter bins set by accdm_q_bins_per_decade.
Run after building classy:  python notebooks_test/test_accdm_q_min_bins.py
"""
import sys

import numpy as np
from classy import Class

from test_accdm_input_checks import accdm_params, expect_error

# kappa = 20, a_t = 1: births squeezed into a in [0.48, 1], so 50 bins per decade give 17 bins
NARROW = dict(kappa_acc=20.0, a_t_acc=1.0, f_acc=1.0, output="mPk")


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


def test_negative_is_rejected():
    expect_error(accdm_params(accdm_q_min_bins=-1), "'accdm_q_min_bins'")


def test_default_is_no_floor():
    assert same(observables(accdm_params(**NARROW)), observables(accdm_params(**NARROW, accdm_q_min_bins=0)))


def test_floor_below_the_per_decade_count_changes_nothing():
    # accdm_params' transition (kappa 4, a_t 0.13) gets 121 bins from the per-decade rule
    p = dict(output="mPk")
    assert same(observables(accdm_params(**p)), observables(accdm_params(**p, accdm_q_min_bins=11)))


def test_floor_equals_explicit_bins():
    floor = observables(accdm_params(**NARROW, accdm_q_min_bins=201))
    explicit = observables(accdm_params(**NARROW, ncdm_N_momentum_bins="150, 201"))
    assert same(floor, explicit)


def test_explicit_bins_win():
    explicit = observables(accdm_params(**NARROW, ncdm_N_momentum_bins="150, 51"))
    both = observables(accdm_params(**NARROW, ncdm_N_momentum_bins="150, 51", accdm_q_min_bins=201))
    assert same(explicit, both)


def test_floor_converges_a_narrow_window():
    ref = observables(accdm_params(**NARROW, accdm_q_min_bins=401))["sigma8_cb"]
    default = observables(accdm_params(**NARROW))["sigma8_cb"]
    floor = observables(accdm_params(**NARROW, accdm_q_min_bins=201))["sigma8_cb"]
    assert abs(floor/ref - 1) < abs(default/ref - 1), (default/ref - 1, floor/ref - 1)


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
