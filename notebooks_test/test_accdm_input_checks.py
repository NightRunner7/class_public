"""Input checks that reject accDM settings which would otherwise run silently wrong.
Run after building classy:  python notebooks_test/test_accdm_input_checks.py
"""
import sys

import numpy as np
from classy import Class, CosmoSevereError

MASS = 1e16                      # m_acc in GeV


def accdm_params(**overrides):
    """Valid accDM run with a neutrino in slot 0 and the daughter in slot 1."""
    params = {
        "omega_b": 0.022383, "omega_cdm": 0.12011, "H0": 67.32,
        "evolver": 0, "gauge": "synchronous",
        "vary_Gamma_acc": "yes", "kappa_acc": 4.0, "a_t_acc": 0.13,
        "f_acc": 0.01, "eta_acc": 0.1,
        "m_acc_in_GeV": MASS, "m_cdm_in_GeV": MASS,
        "N_ncdm": 2, "deg_ncdm": "3, 1", "m_ncdm": "0.02, {:.6e}".format(MASS * 1e9),
        "T_ncdm": "0.71611, 1", "ncdm_quadrature_strategy": "0, 5",
        "N_ur": 0.00441, "ncdm_fluid_approximation": 3,
    }
    params.update(overrides)
    return params


def compute(params, level):
    cosmo = Class()
    cosmo.set(params)
    try:
        cosmo.compute(level)
    finally:
        try:
            cosmo.struct_cleanup()
            cosmo.empty()
        except Exception:
            pass


def expect_error(params, needle, level=("background",)):
    """Assert that computing `params` up to `level` raises CosmoSevereError mentioning `needle`."""
    try:
        compute(params, list(level))
    except CosmoSevereError as error:
        assert needle in str(error), "expected '{}' in error, got:\n{}".format(needle, error)
        return
    raise AssertionError("expected CosmoSevereError mentioning '{}', none raised".format(needle))


def test_valid_run_passes_the_checks():
    compute(accdm_params(output="mPk"), ["perturbations"])


def test_fluid_approximation_is_rejected():
    expect_error(accdm_params(output="mPk", ncdm_fluid_approximation=2),
                 "ncdm_fluid_approximation = 3", level=("perturbations",))


def test_default_fluid_approximation_is_rejected():
    params = accdm_params(output="mPk")
    del params["ncdm_fluid_approximation"]
    expect_error(params, "ncdm_fluid_approximation = 3", level=("perturbations",))


def test_omega_ncdm_on_daughter_is_rejected():
    expect_error(accdm_params(Omega_ncdm="0, 0.01"), "abundance is set by 'f_acc'")


def test_omega_ncdm_on_neutrino_only_is_allowed():
    compute(accdm_params(m_ncdm="0, {:.6e}".format(MASS * 1e9), Omega_ncdm="0.001, 0"),
            ["background"])


def test_daughter_degeneracy_is_rejected():
    expect_error(accdm_params(deg_ncdm="3, 2"), "needs 'deg_ncdm' = 1")


def test_daughter_psd_file_is_rejected():
    expect_error(accdm_params(use_ncdm_psd_files="0, 1", ncdm_psd_filenames="psd_FD_single.dat"),
                 "cannot take its p.s.d. from a file")


def test_omega_ini_dcdm_with_accdm_is_rejected():
    params = accdm_params(Omega_ini_dcdm=0.01)
    del params["f_acc"]
    expect_error(params, "do not set the accDM abundance")


def test_negative_f_acc_is_rejected():
    expect_error(accdm_params(f_acc=-0.1), "'f_acc' must be >= 0")


def test_missing_f_acc_is_rejected():
    params = accdm_params()
    del params["f_acc"]
    expect_error(params, "accDM needs 'f_acc'")


def test_nonpositive_kappa_is_rejected():
    expect_error(accdm_params(kappa_acc=0.), "'kappa_acc' must be > 0")


def test_nonpositive_a_t_is_rejected():
    expect_error(accdm_params(a_t_acc=-0.1), "'a_t_acc' must be > 0")


def test_eta_and_e_acc_together_are_rejected():
    expect_error(accdm_params(E_acc_in_GeV=1e11), "only enter one of 'eta_acc' or 'E_acc_in_GeV'")


def background_daughter(params):
    """Background rho of the daughter (last ncdm species)."""
    cosmo = Class()
    cosmo.set(params)
    try:
        cosmo.compute(["background"])
        return cosmo.get_background()["(.)rho_ncdm[1]"].copy()
    finally:
        cosmo.struct_cleanup()
        cosmo.empty()


def test_default_kick_is_1e11_gev():
    params = accdm_params()
    del params["eta_acc"]
    default = background_daughter(params)
    explicit = background_daughter(accdm_params(eta_acc=1e11 / MASS))
    from_energy = background_daughter(dict(params, E_acc_in_GeV=1e11))
    assert np.array_equal(default, explicit), "default kick differs from eta_acc = 1e11/m_acc"
    assert np.array_equal(default, from_energy), "default kick differs from E_acc_in_GeV = 1e11"


def pk_and_sigma8(params):
    cosmo = Class()
    cosmo.set(dict(params, output="mPk", **{"P_k_max_1/Mpc": 1.0}))
    try:
        cosmo.compute()
        k = np.logspace(-3, 0, 40)
        return np.array([cosmo.pk_lin(ki, 0.0) for ki in k]), cosmo.sigma8()
    finally:
        cosmo.struct_cleanup()
        cosmo.empty()


def test_zero_f_acc_leaves_the_daughter_slot_empty():
    bg = background_daughter(accdm_params(f_acc=0.))
    assert np.all(bg == 0.), "f_acc = 0: daughter density is not zero"
    # empty slot reproduces the neutrino-only run up to integration tolerance
    pk_acc, s8_acc = pk_and_sigma8(accdm_params(f_acc=0.))
    plain = {key: value for key, value in accdm_params().items()
             if not key.endswith("_acc") and key not in ("m_acc_in_GeV", "m_cdm_in_GeV", "f_acc")}
    plain.update({"N_ncdm": 1, "deg_ncdm": 3, "m_ncdm": 0.02, "T_ncdm": 0.71611,
                  "ncdm_quadrature_strategy": 0})
    pk_plain, s8_plain = pk_and_sigma8(plain)
    assert np.allclose(pk_acc, pk_plain, rtol=1e-3), \
        "f_acc = 0 vs neutrino-only: max rel. dev. {:.3e}".format(np.max(np.abs(pk_acc / pk_plain - 1)))
    assert abs(s8_acc / s8_plain - 1) < 1e-3, "sigma8 {} vs {}".format(s8_acc, s8_plain)


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
