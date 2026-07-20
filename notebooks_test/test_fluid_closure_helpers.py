import numpy as np
import pytest
from fluid_closure_helpers import (
    ca2_from_kfs, sound_speed_response, shear_response,
    log_upper_envelope, collapse_band, mask_small_denom,
    smooth_step, two_regime_ceff2, saturating_cfs, eta_plateau_cfs,
    desaturate_cfs, fit_eta_slope, fit_A_of_f, A_eff_of_f, ceff2_f_eta,
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

def test_eta_plateau_cfs_small_eta_is_linear():
    # unsaturated regime: c_eff^2 ~ A*eta with the default A=0.55
    assert np.isclose(eta_plateau_cfs(1e-5), 0.55e-5, rtol=1e-3)

def test_eta_plateau_cfs_matches_exact_plateau_scan():
    # exact-hierarchy ceff2 plateaus at kappa=6, a_t=0.13 (eta-scan, read off
    # the measured plateau levels; A=0.55 fit by eye). 10% tolerance until the
    # scan numbers are re-fit properly.
    measured = {0.01: 0.0055, 0.05: 0.027, 0.1: 0.053, 0.3: 0.13, 0.5: 0.18}
    for eta, plateau in measured.items():
        assert abs(float(eta_plateau_cfs(eta))/plateau - 1.0) < 0.10, \
            'eta={}: model {} vs measured {}'.format(eta, eta_plateau_cfs(eta), plateau)

def test_eta_plateau_cfs_ceiling_and_zero_guard():
    # saturates at the relativistic free-gas ceiling 1/3, never above
    c = eta_plateau_cfs(1e3)
    assert c <= 1./3. + 1e-15 and np.isclose(c, 1./3., atol=1e-6)
    # eta=0 (no kick) and unphysical negative eta both give 0
    out = eta_plateau_cfs(np.array([0.0, -1e-3]))
    assert np.allclose(out, 0.0)

def test_desaturate_cfs_round_trips_saturating_cfs():
    # desaturate is the exact inverse of the saturating map: y = A*x recovered
    x = np.array([1e-4, 1e-2, 0.05, 0.2])
    y = desaturate_cfs(saturating_cfs(x, A=7.3))
    assert np.allclose(y, 7.3 * x, rtol=1e-10)

def test_desaturate_cfs_guards():
    out = desaturate_cfs(np.array([0.0, -1e-3, 1.0/3.0, 0.4]))
    assert out[0] == 0.0 and out[1] == 0.0            # no/unphysical pressure -> 0
    assert np.isinf(out[2]) and np.isinf(out[3])      # at/above the 1/3 ceiling -> inf

def test_fit_eta_slope_recovers_known_A():
    etas = np.array([0.01, 0.05, 0.1, 0.3, 0.5])
    plateaus = saturating_cfs(etas, A=0.62)           # synthetic exact plateaus
    assert np.isclose(fit_eta_slope(etas, plateaus), 0.62, rtol=1e-10)

def test_fit_eta_slope_drops_saturated_points():
    # a plateau pinned at the 1/3 ceiling carries no slope information; it must
    # be dropped (y = inf), not poison the fit
    etas = np.array([0.01, 0.1, 1e3])
    plateaus = np.array([float(saturating_cfs(0.01, A=0.62)),
                         float(saturating_cfs(0.1, A=0.62)), 1.0/3.0])
    assert np.isclose(fit_eta_slope(etas, plateaus), 0.62, rtol=1e-10)

def test_fit_eta_slope_all_bad_gives_nan():
    assert np.isnan(fit_eta_slope([0.1, 0.2], [1.0/3.0, np.nan]))

def test_fit_A_of_f_recovers_A0_and_B():
    f = np.array([0.03, 0.1, 0.2, 0.3])
    A = 0.55 * (1.0 + 0.8 * f)
    A0, B = fit_A_of_f(f, A)
    assert np.isclose(A0, 0.55, rtol=1e-10) and np.isclose(B, 0.8, rtol=1e-10)

def test_fit_A_of_f_flat_gives_B_zero():
    f = np.array([0.03, 0.1, 0.2, 0.3])
    A0, B = fit_A_of_f(f, np.full(f.shape, 0.55))
    assert np.isclose(A0, 0.55, rtol=1e-10) and np.isclose(B, 0.0, atol=1e-10)

def test_A_eff_of_f_is_the_amplitude_law():
    assert np.isclose(A_eff_of_f(0.3, A0=0.5, B=2.0), 0.5 * 1.6)

def test_ceff2_f_eta_reduces_to_eta_plateau_at_B_zero():
    etas = np.array([0.01, 0.1, 0.5])
    assert np.allclose(ceff2_f_eta(0.3, etas, A0=0.55, B=0.0),
                       eta_plateau_cfs(etas, A=0.55))

def test_ceff2_f_eta_monotone_in_f_for_positive_B():
    lo = float(ceff2_f_eta(0.03, 0.1, A0=0.55, B=0.8))
    hi = float(ceff2_f_eta(0.3, 0.1, A0=0.55, B=0.8))
    assert hi > lo
    # and consistent with passing A_eff through the eta-only law (what the C
    # run receives via ncdm_ceff2_eta_A)
    assert np.isclose(hi, float(eta_plateau_cfs(0.1, A=A_eff_of_f(0.3, 0.55, 0.8))))

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
