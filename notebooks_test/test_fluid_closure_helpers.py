import numpy as np
import pytest
from fluid_closure_helpers import (
    ca2_from_kfs, sound_speed_response, shear_response,
    log_upper_envelope, collapse_band,
)

def test_ca2_from_kfs_inverts_convention():
    # k_fs = sqrt(3/2)*aH/sqrt(ca2)  =>  ca2 = (3/2)*(aH/k_fs)^2
    aH, ca2_true = 2.0, 0.05
    k_fs = np.sqrt(1.5) * aH / np.sqrt(ca2_true)
    assert np.isclose(ca2_from_kfs(k_fs, aH), ca2_true)

def test_ca2_from_kfs_array_and_zero_guard():
    out = ca2_from_kfs(np.array([1.0, 0.0]), np.array([1.0, 1.0]))
    assert np.isfinite(out[0]) and out[1] == 0.0   # k_fs=0 -> 0, no divide error

def test_sound_speed_response_is_ratio():
    r = sound_speed_response(np.array([0.1, 0.2]), np.array([0.05, 0.05]))
    assert np.allclose(r, [2.0, 4.0])

def test_shear_response_scales_with_k_and_guards_zero_theta():
    r = shear_response(2.0, np.array([0.3, 0.3]), np.array([0.6, 0.0]))
    assert np.isclose(r[0], 2.0 * 0.3 / 0.6)       # = 1.0
    assert np.isfinite(r[1])                        # theta=0 guarded, no inf

def test_log_upper_envelope_takes_abs_max_per_bin():
    x = np.array([1.0, 1.1, 10.0, 11.0])
    y = np.array([-5.0, 2.0, 1.0, -0.5])            # bin1 max|y|=5, bin2 max|y|=1
    xc, env = log_upper_envelope(x, y, n_bins=2)
    assert env[0] == 5.0 and env[-1] == 1.0
    assert np.all(np.diff(xc) > 0)

def test_collapse_band_zero_for_identical_curves():
    x = np.logspace(0, 2, 20)
    curves = [(x, np.sqrt(x)), (x, np.sqrt(x))]
    band, mx = collapse_band(x, curves)
    assert mx < 1e-9

def test_collapse_band_measures_spread():
    x = np.logspace(0, 2, 20)
    curves = [(x, np.ones_like(x)), (x, 2.0 * np.ones_like(x))]
    band, mx = collapse_band(x, curves)
    # (max-min)/median = (2-1)/1.5 = 0.6667 everywhere
    assert np.isclose(mx, (2.0 - 1.0) / 1.5, atol=1e-6)
