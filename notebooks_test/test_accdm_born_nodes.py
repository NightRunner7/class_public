"""accdm_q_log_share: daughter nodes placed partly by born fraction (strategy 5).
Run after building classy:  python notebooks_test/test_accdm_born_nodes.py
"""
import sys

import numpy as np
from classy import Class

from test_accdm_input_checks import accdm_params, expect_error

# chain settings (connect/new): late, fairly sharp transition and a large daughter share
CHAIN = dict(kappa_acc=12.1, a_t_acc=0.133, f_acc=1.0)


def run(params, level, extract):
    cosmo = Class()
    cosmo.set(params)
    try:
        cosmo.compute(level)
        return extract(cosmo)
    finally:
        cosmo.struct_cleanup()
        cosmo.empty()


def test_share_out_of_range_is_rejected():
    for share in (0.0, -0.5, 1.5):
        expect_error(accdm_params(**CHAIN, accdm_q_log_share=share), "must lie in (0,1]")


def test_share_with_smooth_births_is_rejected():
    expect_error(accdm_params(**CHAIN, accdm_q_log_share=0.4, accdm_smooth_births=1),
                 "cannot be combined with 'accdm_smooth_births'")


def test_share_one_is_bit_identical_to_the_default():
    cl = lambda c: c.lensed_cl(1000)
    p = dict(output="tCl,pCl,lCl", lensing="yes", l_max_scalars=1000)
    a = run(accdm_params(**CHAIN, **p), ["lensing"], cl)
    b = run(accdm_params(**CHAIN, **p, accdm_q_log_share=1.0), ["lensing"], cl)
    for s in ("tt", "ee", "te", "pp"):
        assert np.array_equal(a[s], b[s]), s


def daughter_omega(params):
    def extract(c):
        bg = c.get_background()
        return bg["(.)rho_ncdm[1]"][-1]/bg["(.)rho_crit"][-1]
    return run(params, ["background"], extract)


def test_born_nodes_keep_the_daughter_abundance():
    # fewer nodes in the tails: Omega_acc moves by 2.5e-5 at s = 0.4 (1e-6 at 0.6, 6e-5 at 0.25)
    ref = daughter_omega(accdm_params(**CHAIN))
    new = daughter_omega(accdm_params(**CHAIN, accdm_q_log_share=0.4))
    assert abs(new/ref - 1) < 1e-4, new/ref - 1


def test_born_nodes_move_the_perturbations_slightly():
    s8 = lambda c: c.sigma8()
    p = dict(output="mPk", **CHAIN)
    ref = run(accdm_params(**p), ["fourier"], s8)
    new = run(accdm_params(**p, accdm_q_log_share=0.4), ["fourier"], s8)
    assert 1e-7 < abs(new/ref - 1) < 1e-2, new/ref - 1


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
