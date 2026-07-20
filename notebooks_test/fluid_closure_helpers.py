"""Pure-numpy helpers for the fluid-closure feasibility diagnostic (notebook 15).

No CLASS dependency: everything here is unit-tested offline. The notebook feeds
these the daughter perturbation time-series the exact hierarchy already emits
(delta_ncdm, theta_ncdm, shear_ncdm, cs2_ncdm=delta_p/delta_rho, k_fss_acc).
"""
from __future__ import annotations
import numpy as np

_KFS_FACTOR = 1.5  # k_fs = sqrt(3/2)*aH/sqrt(ca2) -> ca2 = (3/2)*(aH/k_fs)^2


def ca2_from_kfs(k_fs, aH):
    """Recover base adiabatic sound speed ca2 from the free-streaming scale.
    k_fs <= 0 (unborn/degenerate daughter) maps to 0."""
    k_fs = np.asarray(k_fs, float)
    aH = np.asarray(aH, float)
    good = k_fs > 0.0
    out = np.zeros(np.broadcast(k_fs, aH).shape, float)
    ratio = np.divide(aH, k_fs, out=np.zeros_like(out), where=good)
    return np.where(good, _KFS_FACTOR * ratio * ratio, 0.0)


def sound_speed_response(delta_p_over_delta_rho, ca2, eps=1e-30):
    """R_c = (delta_p/delta_rho) / ca2 -- the quantity the Eq-38 fit models as
    1 + amp*W*sqrt(k/k_fs)."""
    dpr = np.asarray(delta_p_over_delta_rho, float)
    ca2 = np.asarray(ca2, float)
    return dpr / np.where(np.abs(ca2) < eps, np.nan, ca2)


def shear_response(k, shear, theta, eps=1e-30):
    """R_v = k*sigma/theta -- the anisotropic-stress (cvis2) signature."""
    shear = np.asarray(shear, float)
    theta = np.asarray(theta, float)
    denom = np.where(np.abs(theta) < eps, np.nan, theta)
    return k * shear / denom


def mask_small_denom(y, denom, drop_frac=0.2):
    """Return y with NaN wherever |denom| is in the smallest drop_frac fraction
    of its finite magnitudes -- removes ratio poles at denom zero-crossings
    (delta_p/delta_rho, sigma/delta, k*sigma/theta all blow up when the
    denominator crosses zero in the free-streaming regime)."""
    y = np.asarray(y, float).astype(float, copy=True)
    denom = np.asarray(denom, float)
    a = np.abs(denom)
    finite = np.isfinite(a)
    if finite.sum() == 0:
        return np.full(y.shape, np.nan)
    thr = np.quantile(a[finite], drop_frac)
    y[~(a > thr)] = np.nan          # masks small |denom| and non-finite denom
    return y


def log_upper_envelope(x, y, n_bins=40):
    """Upper envelope of |y| over log-spaced x bins. Returns (x_centers, env)
    for non-empty bins only. x must be positive."""
    x = np.asarray(x, float)
    y = np.abs(np.asarray(y, float))
    m = (x > 0) & np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if x.size == 0:
        return np.array([]), np.array([])
    edges = np.logspace(np.log10(x.min()), np.log10(x.max()), n_bins + 1)
    edges[-1] *= 1.0 + 1e-9  # include the max
    idx = np.digitize(x, edges) - 1
    xc, env = [], []
    for b in range(n_bins):
        sel = idx == b
        if np.any(sel):
            xc.append(np.sqrt(edges[b] * edges[b + 1]))
            env.append(y[sel].max())
    return np.asarray(xc), np.asarray(env)


def smooth_step(x, x_t, p):
    """Smooth 0->1 step S(x) = x^p / (x_t^p + x^p); S(x_t) = 1/2."""
    x = np.asarray(x, float)
    xp = np.power(x, p)
    return xp / (np.power(x_t, p) + xp)


def two_regime_ceff2(x, ca2, c_fs, x_t, p):
    """Two-regime effective sound speed: adiabatic ca2 below the transition x_t,
    free-streaming plateau c_fs above, joined by the smooth step S(x)."""
    S = smooth_step(x, x_t, p)
    return np.asarray(ca2, float) * (1.0 - S) + c_fs * S


def saturating_cfs(ca2_today, A=13.0):
    """Mode-2 plateau: c_fs = (1/3)(1 - exp(-3*A*ca2_today)).

    Saturates at the relativistic free-gas ceiling 1/3 (radiation sound
    speed c/sqrt(3)); linear ~A*ca2 when unsaturated. Mirrors the C
    computation of pba->cfs_acc in background_init - keep in sync."""
    ca2 = np.clip(np.asarray(ca2_today, float), 0.0, None)
    return (1.0/3.0)*(1.0 - np.exp(-3.0*A*ca2))


def eta_plateau_cfs(eta_acc, A=0.55):
    """Mode-3 plateau: c_eff^2 = (1/3)(1 - exp(-3*A*eta_acc)).

    Same saturating form as mode 2 but driven directly by the injection
    kick eta_acc (an input parameter) instead of the background ca2_bg(a=1):
    the exact-hierarchy plateau is linear in eta (~0.55*eta at small eta),
    i.e. pinned to the kick of the freshly injected daughters, not to the
    redshifted bath temperature. Mirrors the C computation of pba->cfs_acc
    for ncdm_ceff2_mode=3 in background_init - keep in sync."""
    return saturating_cfs(eta_acc, A=A)


def desaturate_cfs(cfs):
    """Invert the saturating map: return y such that cfs = (1/3)(1 - exp(-3y)).

    Linearizes the plateau for fitting (y = A*eta when the eta-law holds).
    cfs <= 0 (no pressure) -> 0; cfs at/above the 1/3 ceiling -> inf, so
    saturated points drop out of finite-only fits instead of biasing them."""
    c = np.clip(np.asarray(cfs, float), 0.0, None)
    with np.errstate(divide='ignore', invalid='ignore'):
        y = -np.log(1.0 - 3.0*c)/3.0
    return np.where(c >= 1.0/3.0, np.inf, y)


def fit_eta_slope(etas, plateaus):
    """Least-squares slope A through the origin of desaturated plateau vs eta:
    desaturate_cfs(plateau) = A*eta. Saturated (inf) and non-finite points are
    dropped; returns NaN if nothing usable remains."""
    etas = np.asarray(etas, float)
    y = desaturate_cfs(plateaus)
    good = np.isfinite(y) & np.isfinite(etas) & (etas > 0)
    if not np.any(good):
        return float('nan')
    return float(np.sum(y[good]*etas[good]) / np.sum(etas[good]**2))


def fit_A_of_f(f_values, A_values):
    """Fit the amplitude law A(f) = A0*(1 + B*f) by ordinary least squares on
    A = a + b*f. Returns (A0, B) = (a, b/a)."""
    f = np.asarray(f_values, float)
    A = np.asarray(A_values, float)
    good = np.isfinite(f) & np.isfinite(A)
    slope, intercept = np.polyfit(f[good], A[good], 1)
    return float(intercept), float(slope/intercept)


def A_eff_of_f(f, A0, B):
    """Effective eta-law amplitude A(f) = A0*(1 + B*f) -- the value passed to
    CLASS as ncdm_ceff2_eta_A so mode 3 realizes ceff2_f_eta without C changes."""
    return A0*(1.0 + B*np.asarray(f, float))


def ceff2_f_eta(f, eta, A0=0.55, B=0.0):
    """Deliverable formula: ceff2(f, eta) = (1/3)(1 - exp(-3*A0*(1+B*f)*eta)).

    B = 0 reduces to eta_plateau_cfs (the f-independent eta-law). Calibrated at
    kappa=6, a_t=0.13 only (notebook 18); universality in (kappa, a_t) untested."""
    return saturating_cfs(np.asarray(eta, float), A=A_eff_of_f(f, A0, B))


def collapse_band(x_grid, curves, eps=1e-30):
    """Quantify collapse of several (x_i, y_i) curves onto x_grid (log-interp).
    At each x covered by >=2 curves, band = (max-min)/|median|. Returns
    (band_of_x, max_band). NaN where <2 curves cover x."""
    x_grid = np.asarray(x_grid, float)
    logg = np.log10(x_grid)
    stack = np.full((len(curves), x_grid.size), np.nan)
    for i, (xi, yi) in enumerate(curves):
        xi = np.asarray(xi, float)
        yi = np.asarray(yi, float)
        m = (xi > 0) & np.isfinite(xi) & np.isfinite(yi)
        if m.sum() < 2:
            continue
        order = np.argsort(xi[m])
        xs, ys = np.log10(xi[m][order]), yi[m][order]
        inside = (logg >= xs[0]) & (logg <= xs[-1])
        stack[i, inside] = np.interp(logg[inside], xs, ys)
    band = np.full(x_grid.size, np.nan)
    for j in range(x_grid.size):
        col = stack[:, j]
        col = col[np.isfinite(col)]
        if col.size >= 2:
            med = np.median(col)
            band[j] = (col.max() - col.min()) / (abs(med) + eps)
    finite = band[np.isfinite(band)]
    return band, (float(finite.max()) if finite.size else float("nan"))
