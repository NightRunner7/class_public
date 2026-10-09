"""Flat models: the C_l multipole sampling must not depend on roundoff in angular_rescaling.
Run after building classy:  python notebooks_test/test_cl_sampling_flat.py

angular_rescaling = ra_rec/(tau0 - tau_rec) is 1 in flat space but came out as 1 - 1e-15, and the
l list truncates expressions in it to integers. Changing H0 by 5e-10 then moved every C_l by up to
2e-3 (EE), a Delta chi2 of 0.041 (the "bin-independent jumps" of notebooks 12, 16-18).
"""
import sys

import numpy as np
from classy import Class

# flat LCDM with the chains' neutrino sector; at these H0 the old code took two different l lists
LCDM = {'omega_b': 0.02252667, 'omega_cdm': 0.1177134, 'ln10^{10}A_s': 3.054243, 'n_s': 0.9699094,
        'tau_reio': 0.06047878, 'k_pivot': 0.05, 'N_ur': 2.0308, 'N_ncdm': 1, 'deg_ncdm': 1, 'm_ncdm': 0.06,
        'T_ncdm': 0.71611, 'reionization_z_start_max': 80, 'output': 'tCl,pCl,lCl', 'lensing': 'yes',
        'l_max_scalars': 2508}
H0 = 68.3547


def lensed_cl(h0):
    cosmo = Class()
    cosmo.set({**LCDM, 'H0': h0})
    try:
        cosmo.compute(['lensing'])
        return cosmo.lensed_cl(2500)
    finally:
        cosmo.struct_cleanup()
        cosmo.empty()


def test_tiny_h0_change_keeps_the_l_sampling():
    # a flip moves TT/EE by 1.4e-3; what remains is rk step noise, 3.5e-5 at the default tolerance
    # (1.4e-6 at 1e-6), so 2e-4 separates the two
    a = lensed_cl(H0)
    for eps in (5e-10, -1.5e-9):
        b = lensed_cl(H0*(1 + eps))
        for s in ('tt', 'ee'):
            dev = np.max(np.abs(b[s][2:]/a[s][2:] - 1))
            assert dev < 2e-4, (eps, s, dev)


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
