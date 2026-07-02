"""Smoke tests for the accDM daughter q(f) schedule (input.c).
Run after building classy:  python notebooks_test/test_q_schedule_smoke.py
"""
import os
import sys
import tempfile
from contextlib import contextmanager

import numpy as np
from classy import Class


@contextmanager
def capture_class_stdout():
    """Capture C-level stdout (printf from CLASS) by redirecting fd 1 to a temp file."""
    sys.stdout.flush()
    saved_stdout_fd = os.dup(1)
    with tempfile.TemporaryFile(mode="w+b") as capture_file:
        os.dup2(capture_file.fileno(), 1)
        captured = {}
        try:
            yield captured
        finally:
            sys.stdout.flush()
            os.dup2(saved_stdout_fd, 1)
            os.close(saved_stdout_fd)
            capture_file.seek(0)
            captured["text"] = capture_file.read().decode(errors="replace")


OMEGA_B = 0.022383
OMEGA_CDM0 = 0.12011
MASS = 1e16
ETA = 0.1
KAPPA = 4.0
A_T = 0.13                       # accDM trigger scale factor (nb10 fiducial)
A_REC = 1.0 / (1.0 + 1090.0)     # recombination scale factor


def base_params(f_acc, output=""):
    """Notebook-10 fiducial accDM model; no ncdm_N_momentum_bins (schedule decides)."""
    omega_cdm = OMEGA_CDM0 * (1 + f_acc * (1 - A_REC**KAPPA) / (1 + (A_REC / A_T)**KAPPA))**(-1)
    return {
        "omega_b": OMEGA_B, "omega_cdm": omega_cdm, "H0": 67.32,
        "A_s": 2.1005829616811546e-9, "n_s": 0.96605, "tau_reio": 0.0543,
        "output": output, "evolver": 0, "gauge": "synchronous",
        "background_Nloga": 5000,
        "vary_Gamma_acc": "yes", "kappa_acc": KAPPA, "a_t_acc": A_T,
        "f_acc": f_acc, "eta_acc": ETA,
        "m_acc_in_GeV": MASS, "m_cdm_in_GeV": MASS,
        "N_ncdm": 2, "deg_ncdm": "3, 1",
        "m_ncdm": "0.02, {:.6e}".format(MASS * 1e9),
        "T_ncdm": "0.71611, 1", "ncdm_quadrature_strategy": "0, 4",
        "N_ur": 0.00441, "ncdm_fluid_approximation": 3,
        "input_verbose": 1,
    }


def run_and_capture(params):
    cosmo = Class()
    cosmo.set(params)
    with capture_class_stdout() as captured:
        cosmo.compute()
    cosmo.struct_cleanup()
    cosmo.empty()
    return captured["text"]


def expect_in(text, needle, label):
    assert needle in text, "{}: expected '{}' in CLASS stdout, got:\n{}".format(label, needle, text)


def test_low_f_gets_coarse_grid():
    text = run_and_capture(base_params(f_acc=0.01))
    expect_in(text, "q_size = 250", "low-f schedule")


def test_mid_f_gets_production_grid():
    text = run_and_capture(base_params(f_acc=0.2))
    expect_in(text, "q_size = 1000", "mid-f schedule")


def test_high_f_gets_fine_grid():
    text = run_and_capture(base_params(f_acc=0.5))
    expect_in(text, "q_size = 2000", "high-f schedule")


def test_explicit_bins_bypass_schedule():
    params = base_params(f_acc=0.01)
    params["ncdm_N_momentum_bins"] = "15, 777"
    text = run_and_capture(params)
    expect_in(text, "schedule bypassed", "explicit-bins bypass")


def test_custom_schedule_table():
    params = base_params(f_acc=0.01)
    params["accdm_q_schedule_f_edges"] = "0.05"
    params["accdm_q_schedule_q_sizes"] = "100, 900"
    text = run_and_capture(params)
    expect_in(text, "q_size = 100", "custom table")


def test_schedule_can_be_disabled():
    params = base_params(f_acc=0.01)
    params["accdm_q_schedule"] = "no"
    text = run_and_capture(params)
    assert "accDM q(f) schedule:" not in text, \
        "schedule-off: no schedule line expected, got:\n" + text


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
    sys.exit(1 if failed else 0)
