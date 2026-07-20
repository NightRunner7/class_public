"""nb18 (ceff2(f, eta) formula: extraction + P(k) validation) cache generator.

Reproduces the pickles of notebooks_test/18_test_ceff2_f_eta_formula.ipynb
under nb18_cache/:

  Stage A (extraction grid, F_SCAN x ETA_SCAN, q=501):
    {TAG}_f{f}_eta{eta}_kap6_at0.13_q501_nk18.pkl
        payload: {k: dict of daughter tau-series arrays}
  Stage C (P(k) validation at the corners):
    ref_f{f}_eta{eta}_q{q}.pkl                     exact reference (q = 1001/5001 by f)
    fluid3_f{f}_eta{eta}_trig{trig}_A{A:.4f}.pkl   mode-3 fluid with A(f)
        payload: tuple (pk array on K_PK, wall seconds)

The fluid tag embeds A(f) = A0*(1 + B*f) from the notebook's Stage-B fit, so
Stage C needs the Stage-A pickles (this script recomputes the fit exactly,
including the >= 30% adoption rule) - OR pass --a0/--b to pin the constants.

Run order on a cluster: --stage A first (array over 20 jobs), then --stage C
(array over 10 jobs). --index counts within the selected stage.
"""
import os
import pickle
import sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _SCRIPT_DIR)
sys.path.insert(0, os.path.dirname(_SCRIPT_DIR))     # notebooks_test/, for the helpers
import numpy as np

import common
from fluid_closure_helpers import (
    mask_small_denom, fit_eta_slope, fit_A_of_f, A_eff_of_f, ceff2_f_eta)

# --- Stage A: extraction grid (verbatim nb18 setup cell) ---
F_SCAN = [0.03, 0.1, 0.2, 0.3]
ETA_SCAN = [0.01, 0.05, 0.1, 0.3, 0.5]
K_GRID = np.logspace(-2, 0.0, 18)
Q_EXTRACT = 501
DROP_FRAC = 0.2
X_WINDOW = (0.05, 30.0)

# --- Stage C: P(k) validation ---
K_PK = np.logspace(-3, 0.0, 40)
TRIGGER = 10.0
CORNERS = [(0.1, 0.05), (0.1, 0.5), (0.3, 0.05), (0.3, 0.5), (0.2, 0.1)]


def q_exact_of_f(f_acc):
    """Converged exact-reference q-grid (memory: accdm-fluid-f-boundary)."""
    return 1001 if f_acc <= 0.1 else 5001


def stage_a_tag(f_acc, eta, cache_tag):
    return '{}_f{:g}_eta{:g}_kap{:g}_at{:g}_q{}_nk{}'.format(
        cache_tag, f_acc, eta, common.KAPPA, common.A_T, Q_EXTRACT, K_GRID.size)


def stage_a_job(f_acc, eta, cache_tag):
    tag = stage_a_tag(f_acc, eta, cache_tag)

    def thunk():
        return common.extract_daughter_series(
            common.accdm_params(f_acc, eta, Q_EXTRACT), K_GRID)

    return common.Job(tag, tag + '.pkl', thunk)


def plateau_of_series(ser, x_lo=X_WINDOW[0], x_hi=X_WINDOW[1]):
    """Verbatim nb18 cell 4: median pole-masked delta_p/delta_rho on the flat window."""
    vals = []
    for k, s in ser.items():
        g = np.isfinite(s['k_fs']) & (s['k_fs'] > 0) & np.isfinite(s['aH'])
        if not np.any(g):
            continue
        x = k / s['k_fs'][g]
        ce = np.abs(mask_small_denom(s['dpr'][g], s['delta'][g], DROP_FRAC))
        m = np.isfinite(ce) & (ce > 0) & (x >= x_lo) & (x <= x_hi)
        if np.any(m):
            vals.append(ce[m])
    return float(np.median(np.concatenate(vals))) if vals else float('nan')


def stage_b_fit(cache_dir, cache_tag):
    """Recompute the notebook's Stage-B fit (incl. the fixed >= 30% adoption
    rule) from the Stage-A pickles; returns (A0_FINAL, B_FINAL)."""
    plateaus = {}
    for f_acc in F_SCAN:
        for eta in ETA_SCAN:
            path = os.path.join(cache_dir, stage_a_tag(f_acc, eta, cache_tag) + '.pkl')
            if not os.path.exists(path):
                raise SystemExit(
                    'Stage-A pickle missing: {}\nRun --stage A to completion first, '
                    'or pin the constants with --a0/--b.'.format(path))
            with open(path, 'rb') as fh:
                plateaus[(f_acc, eta)] = plateau_of_series(pickle.load(fh))

    per_f_amplitude = {f: fit_eta_slope(ETA_SCAN, [plateaus[(f, e)] for e in ETA_SCAN])
                       for f in F_SCAN}
    eta_pool = np.array([eta for f in F_SCAN for eta in ETA_SCAN])
    c_pool = np.array([plateaus[(f, eta)] for f in F_SCAN for eta in ETA_SCAN])
    a0_null = fit_eta_slope(eta_pool, c_pool)
    a0_fit, b_fit = fit_A_of_f(F_SCAN, [per_f_amplitude[f] for f in F_SCAN])

    def max_grid_residual(a0, b):
        residuals = [float(ceff2_f_eta(f, eta, A0=a0, B=b)) / plateaus[(f, eta)] - 1.0
                     for f in F_SCAN for eta in ETA_SCAN]
        return float(np.max(np.abs(residuals)))

    max_null, max_fcor = max_grid_residual(a0_null, 0.0), max_grid_residual(a0_fit, b_fit)
    if max_fcor < 0.7 * max_null:
        print('[fit] adopted f-corrected model: A0 = {:.4f}, B = {:+.4f} '
              '(max residual {:.1%} -> {:.1%})'.format(a0_fit, b_fit, max_null, max_fcor),
              flush=True)
        return a0_fit, b_fit
    print('[fit] adopted eta-only model: A0 = {:.4f}, B = 0 '
          '(f-correction bought only {:.1%} -> {:.1%})'.format(a0_null, max_null, max_fcor),
          flush=True)
    return a0_null, 0.0


def fluid_params(f_acc, eta, a_eff, trigger):
    """Verbatim nb18 mode-3 fluid builder."""
    p = common.accdm_params(f_acc, eta, q_exact_of_f(f_acc))
    p['ncdm_fluid_approximation'] = 2
    p['ncdm_fluid_trigger_rho_accDM_over_rho_dcdm'] = trigger
    p['switch_off_shear_acc'] = 'no'      # never judge the fluid with frozen shear
    p['ncdm_ceff2_mode'] = 3
    p['ncdm_ceff2_eta_A'] = float(a_eff)
    return p


def pk_job(params, tag):
    return common.Job(tag, tag + '.pkl', lambda: common.compute_pk_elapsed(params, K_PK))


def stage_c_jobs(args, a0_final, b_final):
    jobs = []
    for f_acc, eta in CORNERS:
        q = q_exact_of_f(f_acc)
        jobs.append(pk_job(common.accdm_params(f_acc, eta, q),
                           'ref_f{:g}_eta{:g}_q{}'.format(f_acc, eta, q)))
    for f_acc, eta in CORNERS:
        a_eff = float(A_eff_of_f(f_acc, a0_final, b_final))
        jobs.append(pk_job(fluid_params(f_acc, eta, a_eff, args.trigger),
                           'fluid3_f{:g}_eta{:g}_trig{:g}_A{:.4f}'.format(
                               f_acc, eta, args.trigger, a_eff)))
    return jobs


def main():
    parser = common.make_parser(__doc__)
    parser.add_argument('--cache-tag', default='v1',
                        help="nb18 CACHE_TAG for Stage A; must match the notebook (default v1)")
    parser.add_argument('--stage', choices=['A', 'C', 'all'], default='all',
                        help='A = extraction grid, C = P(k) corners (needs Stage A done '
                             'or --a0/--b); --index requires an explicit A or C')
    parser.add_argument('--trigger', type=float, default=TRIGGER,
                        help='Stage-C fluid trigger (default 10, as in the notebook)')
    parser.add_argument('--a0', type=float, default=None,
                        help='pin A0_FINAL instead of refitting from Stage-A pickles')
    parser.add_argument('--b', type=float, default=None,
                        help='pin B_FINAL instead of refitting (default 0 when --a0 given)')
    args = parser.parse_args()

    if args.index is not None and args.stage == 'all':
        raise SystemExit('--index needs --stage A or --stage C (indices count within a stage)')

    if args.stage in ('A', 'all'):
        common.run_jobs([stage_a_job(f, eta, args.cache_tag)
                         for f in F_SCAN for eta in ETA_SCAN], args, 'nb18_cache')
    if args.stage in ('C', 'all'):
        if args.a0 is not None:
            a0_final, b_final = args.a0, (args.b if args.b is not None else 0.0)
            print('[fit] pinned by CLI: A0 = {:.4f}, B = {:+.4f}'.format(a0_final, b_final))
        else:
            a0_final, b_final = stage_b_fit(
                os.path.join(args.cache_root, 'nb18_cache'), args.cache_tag)
        common.run_jobs(stage_c_jobs(args, a0_final, b_final), args, 'nb18_cache')


if __name__ == '__main__':
    main()
