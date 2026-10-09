"""accdm_l_max: length of the daughter's Boltzmann hierarchy, separate from l_max_ncdm.
Run after building classy:  python notebooks_test/test_accdm_l_max.py
"""
import sys

import numpy as np
from classy import Class

from test_accdm_input_checks import accdm_params, expect_error

WARM = 1e11                      # m_acc in GeV: eta = 1, the warmest daughter (nb19)


def warm_params(**overrides):
    """Chain transition at f_tilde = 0.5, where l_max_ncdm = 17 biases sigma8_cb by ~1% (nb19)."""
    return accdm_params(m_acc_in_GeV=WARM, m_cdm_in_GeV=WARM, m_ncdm="0.02, {:.6e}".format(WARM*1e9),
                        eta_acc=1.0, kappa_acc=12.1, a_t_acc=0.133, f_acc=1.0, output="mPk",
                        **overrides)


def run(params, level, extract):
    cosmo = Class()
    cosmo.set(params)
    try:
        cosmo.compute(level)
        return extract(cosmo)
    finally:
        cosmo.struct_cleanup()
        cosmo.empty()


def test_out_of_range_is_rejected():
    for l_max in (-1, 1, 3):
        expect_error(accdm_params(accdm_l_max=l_max), "'accdm_l_max'")


def test_default_follows_l_max_ncdm():
    cl = lambda c: c.lensed_cl(1000)
    p = dict(output="tCl,pCl,lCl", lensing="yes", l_max_scalars=1000)
    ref = run(accdm_params(**p), ["lensing"], cl)
    for extra in (dict(accdm_l_max=0), dict(accdm_l_max=17)):
        new = run(accdm_params(**p, **extra), ["lensing"], cl)
        for s in ("tt", "ee", "te", "pp"):
            assert np.array_equal(ref[s], new[s]), (extra, s)


def test_daughter_l_max_lengthens_only_the_daughter():
    # 17 vs 50 on the daughter alone moves sigma8_cb by ~1%; the neutrinos' 17 vs 50 by ~1e-4
    s8 = lambda c: c.sigma8_cb()
    short = run(warm_params(), ["fourier"], s8)
    daughter = run(warm_params(accdm_l_max=50), ["fourier"], s8)
    both = run(warm_params(l_max_ncdm=50), ["fourier"], s8)
    assert abs(short/both - 1) > 3e-3, short/both - 1
    assert abs(daughter/both - 1) < 1e-3, daughter/both - 1


def test_daughter_l_max_can_be_shorter():
    s8 = lambda c: c.sigma8_cb()
    ref = run(accdm_params(output="mPk"), ["fourier"], s8)
    new = run(accdm_params(output="mPk", accdm_l_max=6), ["fourier"], s8)
    assert ref != new
    assert abs(new/ref - 1) < 1e-2, new/ref - 1


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
