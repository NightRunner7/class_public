"""Tests for the 'm_nu' scalar neutrino-mass input (input.c section 5.d).
Run after building classy:  python notebooks_test/test_m_nu_input.py
"""
import sys

import numpy as np
from classy import Class, CosmoSevereError

MASS = 1e16                      # m_acc in GeV
ETA = 0.1
KAPPA = 4.0
A_T = 0.13                       # accDM trigger scale factor (nb10 fiducial)
A_REC = 1.0 / (1.0 + 1090.0)     # recombination scale factor
OMEGA_B = 0.022383
OMEGA_CDM0 = 0.12011
M_NU = 0.02                      # eV, per species


def plain_params(**overrides):
    """Single massive neutrino species, no accDM."""
    params = {
        "omega_b": OMEGA_B, "omega_cdm": OMEGA_CDM0, "H0": 67.32,
        "A_s": 2.1005829616811546e-9, "n_s": 0.96605, "tau_reio": 0.0543,
        "N_ncdm": 1, "deg_ncdm": 3, "T_ncdm": 0.71611,
        "N_ur": 0.00441,
    }
    params.update(overrides)
    return params


def accdm_params(f_acc=0.01, **overrides):
    """Notebook-10 fiducial accDM model with a neutrino in slot 0."""
    omega_cdm = OMEGA_CDM0 * (1 + f_acc * (1 - A_REC**KAPPA) / (1 + (A_REC / A_T)**KAPPA))**(-1)
    params = {
        "omega_b": OMEGA_B, "omega_cdm": omega_cdm, "H0": 67.32,
        "A_s": 2.1005829616811546e-9, "n_s": 0.96605, "tau_reio": 0.0543,
        "evolver": 0, "gauge": "synchronous", "background_Nloga": 5000,
        "vary_Gamma_acc": "yes", "kappa_acc": KAPPA, "a_t_acc": A_T,
        "f_acc": f_acc, "eta_acc": ETA,
        "m_acc_in_GeV": MASS, "m_cdm_in_GeV": MASS,
        "N_ncdm": 2, "deg_ncdm": "3, 1",
        "T_ncdm": "0.71611, 1", "ncdm_quadrature_strategy": "0, 4",
        "N_ur": 0.00441, "ncdm_fluid_approximation": 3,
    }
    params.update(overrides)
    return params


def derived_first_ncdm_mass(params):
    """Return ba.m_ncdm_in_eV[0] after a background-only compute."""
    cosmo = Class()
    cosmo.set(params)
    try:
        cosmo.compute(["background"])
        return cosmo.get_current_derived_parameters(["m_ncdm_in_eV"])["m_ncdm_in_eV"]
    finally:
        cosmo.struct_cleanup()
        cosmo.empty()


def linear_pk(params):
    """Return P(k) on a fixed k grid, in (Mpc/h)^3."""
    cosmo = Class()
    # 'P_k_max_h/Mpc' contains a slash (input.c:5265), so it cannot be a kwarg.
    cosmo.set(dict(params, output="mPk", **{"P_k_max_h/Mpc": 1.0}))
    try:
        cosmo.compute()
        h = cosmo.h()
        k_over_h = np.logspace(-3, 0, 40)
        return np.array([cosmo.pk_lin(k * h, 0.0) * h**3 for k in k_over_h])
    finally:
        cosmo.struct_cleanup()
        cosmo.empty()


def expect_error(params, needle, label):
    """Assert that computing `params` raises CosmoSevereError mentioning `needle`."""
    cosmo = Class()
    cosmo.set(params)
    try:
        cosmo.compute(["background"])
    except CosmoSevereError as error:
        assert needle in str(error), \
            "{}: expected '{}' in error, got:\n{}".format(label, needle, error)
        return
    finally:
        try:
            cosmo.struct_cleanup()
            cosmo.empty()
        except Exception:
            pass
    raise AssertionError("{}: expected CosmoSevereError, none raised".format(label))


def test_m_nu_sets_the_neutrino_mass():
    mass = derived_first_ncdm_mass(plain_params(m_nu=M_NU))
    assert np.isclose(mass, M_NU, rtol=1e-12), \
        "expected m_ncdm_in_eV[0] == {}, got {}".format(M_NU, mass)


def test_m_nu_matches_m_ncdm_in_a_plain_run():
    from_m_nu = linear_pk(plain_params(m_nu=M_NU))
    from_m_ncdm = linear_pk(plain_params(m_ncdm=M_NU))
    assert np.allclose(from_m_nu, from_m_ncdm, rtol=1e-10), \
        "plain run: max rel. dev. {:.3e}".format(np.max(np.abs(from_m_nu / from_m_ncdm - 1)))


def test_m_nu_matches_m_ncdm_in_an_accdm_run():
    from_m_nu = linear_pk(accdm_params(m_nu=M_NU))
    from_m_ncdm = linear_pk(accdm_params(m_ncdm="{}, {:.6e}".format(M_NU, MASS * 1e9)))
    assert np.allclose(from_m_nu, from_m_ncdm, rtol=1e-10), \
        "accDM run: max rel. dev. {:.3e}".format(np.max(np.abs(from_m_nu / from_m_ncdm - 1)))


def test_both_m_nu_and_m_ncdm_is_an_error():
    expect_error(plain_params(m_nu=M_NU, m_ncdm=M_NU),
                 "only enter one of 'm_nu' or 'm_ncdm'",
                 "mutual exclusion")


def test_m_nu_with_accdm_requires_two_ncdm_species():
    expect_error(accdm_params(m_nu=M_NU, N_ncdm=1, deg_ncdm=1,
                              T_ncdm=1, ncdm_quadrature_strategy=4),
                 "would be silently discarded",
                 "accDM N_ncdm guard")


def test_negative_m_nu_is_an_error():
    expect_error(plain_params(m_nu=-0.02),
                 "which makes no sense",
                 "negative mass")


def test_no_mass_input_keeps_the_ultrarelativistic_default():
    mass = derived_first_ncdm_mass(plain_params())
    assert np.isclose(mass, 1.e-5, rtol=1e-12), \
        "expected the 1e-5 eV default, got {}".format(mass)


if __name__ == "__main__":
    # An optional argument filters tests by substring; a leading '-' excludes.
    # The two '_matches_m_ncdm_' tests each run two full computes and dominate
    # the runtime, so '-_matches_' is the fast subset:
    #   python notebooks_test/test_m_nu_input.py -_matches_
    name_filter = sys.argv[1] if len(sys.argv) > 1 else ""
    if name_filter.startswith("-"):
        keep = lambda name: name_filter[1:] not in name
    else:
        keep = lambda name: name_filter in name
    tests = [value for name, value in sorted(globals().items())
             if name.startswith("test_") and keep(name)]
    failed = 0
    for test in tests:
        try:
            test()
            print("PASS", test.__name__)
        except AssertionError as error:
            failed += 1
            print("FAIL", test.__name__, "--", error)
        except Exception as error:
            # An unexpected CLASS error must not abort the remaining tests.
            failed += 1
            print("ERROR", test.__name__, "--", type(error).__name__, error)
    sys.exit(1 if failed else 0)
