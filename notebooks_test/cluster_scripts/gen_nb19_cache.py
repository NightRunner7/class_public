"""nb19 (fluid error vs signal on P/P_LCDM) cache generator.

Reproduces the run_pk() pickles of
notebooks_test/19_test_fluid_error_to_signal.ipynb under nb19_cache/:

    {TAG}_lcdm.pkl                                     LCDM reference
    {TAG}_exact_f{f}_eta{eta}.pkl                      exact reference per corner
    {TAG}_fluid_t{trig}_shear{yes|no}_f{f}_eta{eta}.pkl  mode-3 fitted-formula fluid

Payload: dict(pk on K_PK, sigma8, seconds).

The exact references do not depend on --trigger/--shear, so rerunning with a
different trigger or shear setting only adds the 5 cheap fluid files.

Knobs NOT in the tags: --a0-eta / --b-f (the nb18 Stage-B constants fed to the
fluid as ncdm_ceff2_eta_A = A0*(1+B*f)). Keep the notebook's A0_ETA/B_F in
sync with what you pass here.
"""
import os
import sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _SCRIPT_DIR)
sys.path.insert(0, os.path.dirname(_SCRIPT_DIR))     # notebooks_test/, for the helpers
import numpy as np

import common
from fluid_closure_helpers import A_eff_of_f

K_PK = np.logspace(-3, 0.0, 40)
CORNERS = [(0.1, 0.05), (0.1, 0.5), (0.3, 0.05), (0.3, 0.5), (0.2, 0.1)]
TRIGGER = 0.4
SHEAR = 'yes'          # 'yes' = frozen shear (nb19 default plot); 'no' = dynamical
A0_ETA, B_F = 0.5223, 0.0


def q_exact_of_f(f_acc):
    """Converged exact-reference q-grid (memory: accdm-fluid-f-boundary)."""
    return 1001 if f_acc <= 0.1 else 5001


def lcdm_params():
    """Verbatim nb19: identical early-time densities and neutrino sector, no acc sector."""
    p = dict(common.BASE_PARAMS); p.update(common.prec_block(1.0))
    p.update({'N_ncdm': 1, 'deg_ncdm': '3', 'm_ncdm': '0.02', 'T_ncdm': '0.71611',
              'ncdm_quadrature_strategy': '0', 'ncdm_N_momentum_bins': '15',
              'N_ur': 0.00441, 'gauge': 'synchronous'})
    return p


def fluid_params(f_acc, eta, trigger, shear_off, a0_eta, b_f):
    """Verbatim nb19 mode-3 fitted-formula fluid builder."""
    p = common.accdm_params(f_acc, eta, q_exact_of_f(f_acc))
    p['ncdm_fluid_approximation'] = 2
    p['ncdm_fluid_trigger_rho_accDM_over_rho_dcdm'] = trigger
    p['switch_off_shear_acc'] = shear_off
    p['ncdm_ceff2_mode'] = 3
    p['ncdm_ceff2_eta_A'] = float(A_eff_of_f(f_acc, a0_eta, b_f))
    return p


def record_job(params, tag, cache_tag):
    filename = '{}_{}.pkl'.format(cache_tag, tag)
    return common.Job(tag, filename, lambda: common.compute_pk_record(params, K_PK))


def parse_corners(corner_strings):
    corners = []
    for text in corner_strings:
        f_text, eta_text = text.split(',')
        corners.append((float(f_text), float(eta_text)))
    return corners


def build_jobs(args):
    corners = parse_corners(args.corners)
    jobs = [record_job(lcdm_params(), 'lcdm', args.cache_tag)]
    for f_acc, eta in corners:
        corner_tag = 'f{:g}_eta{:g}'.format(f_acc, eta)
        jobs.append(record_job(common.accdm_params(f_acc, eta, q_exact_of_f(f_acc)),
                               'exact_' + corner_tag, args.cache_tag))
    for f_acc, eta in corners:
        corner_tag = 'f{:g}_eta{:g}'.format(f_acc, eta)
        jobs.append(record_job(
            fluid_params(f_acc, eta, args.trigger, args.shear, args.a0_eta, args.b_f),
            'fluid_t{:g}_shear{}_{}'.format(args.trigger, args.shear, corner_tag),
            args.cache_tag))
    return jobs


def main():
    parser = common.make_parser(__doc__)
    parser.add_argument('--cache-tag', default='v1',
                        help="nb19 CACHE_TAG; must match the notebook's setup cell (default v1)")
    parser.add_argument('--trigger', type=float, default=TRIGGER,
                        help='abundance-gate trigger for the fluid runs (default 0.4)')
    parser.add_argument('--shear', choices=['yes', 'no'], default=SHEAR,
                        help="switch_off_shear_acc: 'yes' = frozen shear (default), 'no' = dynamical")
    parser.add_argument('--a0-eta', type=float, default=A0_ETA,
                        help='nb18 Stage-B A0 constant (default 0.5223; not in the tag)')
    parser.add_argument('--b-f', type=float, default=B_F,
                        help='nb18 Stage-B B constant (default 0; not in the tag)')
    parser.add_argument('--corners', nargs='+',
                        default=['{:g},{:g}'.format(f, eta) for f, eta in CORNERS],
                        help='corners as f,eta pairs (default: the nb18 Stage-C corners)')
    args = parser.parse_args()
    common.run_jobs(build_jobs(args), args, 'nb19_cache')


if __name__ == '__main__':
    main()
