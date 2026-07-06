import numpy as np
import pytest
from fluid_closure_helpers import (
    ca2_from_kfs, sound_speed_response, shear_response,
    log_upper_envelope, collapse_band, mask_small_denom,
    smooth_step, two_regime_ceff2, saturating_cfs,
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
    # theta=0 is guarded with a NaN sentinel (NOT inf): downstream consumers
    # (log_upper_envelope, mask_small_denom, collapse_band) all drop NaN.
    assert np.isnan(r[1]) and not np.isinf(r[1])

def test_log_upper_envelope_takes_abs_max_per_bin():
    x = np.array([1.0, 1.1, 10.0, 11.0])
    y = np.array([-5.0, 2.0, 1.0, -0.5])            # bin1 max|y|=5, bin2 max|y|=1
    xc, env = log_upper_envelope(x, y, n_bins=2)
    assert env[0] == 5.0 and env[-1] == 1.0
    assert np.all(np.diff(xc) > 0)

def test_mask_small_denom_nans_smallest_denominator_samples():
    y = np.array([1., 2., 3., 4., 5.])
    denom = np.array([0.01, 1.0, 2.0, 3.0, 4.0])   # smallest 20% -> index 0
    out = mask_small_denom(y, denom, drop_frac=0.2)
    assert np.isnan(out[0]) and np.allclose(out[1:], y[1:])

def test_mask_small_denom_masks_nonfinite_denominator():
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    denom = np.array([np.nan, 5.0, 6.0, 7.0, 8.0])    # nan + smallest (5.0) drop out
    out = mask_small_denom(y, denom, drop_frac=0.01)
    assert np.isnan(out[0])                            # non-finite denom masked
    assert np.allclose(out[2:], y[2:])                 # large-|denom| samples survive

def test_smooth_step_half_at_transition_and_limits():
    assert np.isclose(smooth_step(5.0, 5.0, 2.0), 0.5)
    assert smooth_step(1e-3, 5.0, 2.0) < 1e-4          # x << x_t -> 0
    assert smooth_step(1e4, 5.0, 2.0) > 0.999          # x >> x_t -> 1

def test_two_regime_ceff2_limits_to_ca2_and_cfs():
    ca2, c_fs, x_t, p = 0.05, 0.4, 5.0, 2.0
    lo = two_regime_ceff2(1e-3, ca2, c_fs, x_t, p)
    hi = two_regime_ceff2(1e4, ca2, c_fs, x_t, p)
    assert np.isclose(lo, ca2, atol=1e-3)              # below x_t -> adiabatic ca2
    assert np.isclose(hi, c_fs, atol=1e-3)             # above x_t -> free-streaming plateau

def test_saturating_cfs_small_argument_is_linear():
    # small A*ca2: c_fs ~ A*ca2 (unsaturated regime)
    assert np.isclose(saturating_cfs(1e-5, A=13.0), 13.0e-5, rtol=1e-3)

def test_saturating_cfs_saturates_at_one_third():
    # large A*ca2: c_fs -> 1/3 (relativistic free-gas ceiling). At A*ca2=13
    # exp(-39) is below float64 eps, so the result equals 1/3 to machine
    # precision - never above it.
    c = saturating_cfs(1.0, A=13.0)
    assert c <= 1./3. + 1e-15 and np.isclose(c, 1./3., atol=1e-6)
    # strictly below the ceiling while still resolvable:
    c_mid = saturating_cfs(0.05, A=13.0)   # exp(-1.95) ~ 0.14
    assert 0.0 < c_mid < 1./3.

def test_saturating_cfs_matches_nb15_eta1_break():
    # nb15: eta=1 measured c_fs=0.2277 at ca2_today=3.053e-2; the map with
    # A~12.5 should land near it (within ~15%)
    assert abs(saturating_cfs(3.053e-2, A=12.5)/0.2277 - 1.0) < 0.15

def test_saturating_cfs_zero_and_negative_ca2_give_zero():
    out = saturating_cfs(np.array([0.0, -1e-3]), A=13.0)
    assert np.allclose(out, 0.0)

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
