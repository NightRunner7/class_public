"""nb17 (eta-plateau mode-3 fluid vs exact, eta scan at f=0.1) cache generator.

Reproduces the run_pk_timed() pickles of
notebooks_test/17_test_eta_plateau_fluid_validation.ipynb under nb17_cache/:

    ref_eta{eta}_f0.1_q1001.pkl        converged exact reference per eta
    fluid3_eta{eta}_trig{trig}.pkl     mode-3 fluid runs

Payload: tuple (pk array on K_PK, wall seconds).

Settings knobs: --a-eta is ncdm_ceff2_eta_A (notebook A_ETA = 0.55). It is NOT
part of the cache tag, so if you generate files with a different value you
must set the same A_ETA in the notebook's setup cell (and clear stale files).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np

import common

K_PK = np.logspace(-3, 0.0, 40)
F_TEST = 0.1
ETA_TEST = [0.01, 0.05, 0.1, 0.3, 0.5]
Q_EXACT = 1001                       # converged at f <= 0.1 (memory: accdm-fluid-f-boundary)
TRIGGERS = [0.4, 0.1]
A_ETA = 0.55


def fluid_params(eta, trigger, a_eta):
    p = common.accdm_params(F_TEST, eta, Q_EXACT)
    p['ncdm_fluid_approximation'] = 2
    p['ncdm_fluid_trigger_rho_accDM_over_rho_dcdm'] = trigger
    p['switch_off_shear_acc'] = 'no'
    p['ncdm_ceff2_mode'] = 3
    p['ncdm_ceff2_eta_A'] = a_eta
    return p


def pk_job(params, tag):
    return common.Job(tag, tag + '.pkl', lambda: common.compute_pk_elapsed(params, K_PK))


def build_jobs(args):
    jobs = []
    for eta in args.etas:
        jobs.append(pk_job(common.accdm_params(F_TEST, eta, Q_EXACT),
                           'ref_eta{:g}_f{:g}_q{}'.format(eta, F_TEST, Q_EXACT)))
    for eta in args.etas:
        for trigger in args.triggers:
            jobs.append(pk_job(fluid_params(eta, trigger, args.a_eta),
                               'fluid3_eta{:g}_trig{:g}'.format(eta, trigger)))
    return jobs


def main():
    parser = common.make_parser(__doc__)
    parser.add_argument('--etas', type=float, nargs='+', default=ETA_TEST)
    parser.add_argument('--triggers', type=float, nargs='+', default=TRIGGERS)
    parser.add_argument('--a-eta', type=float, default=A_ETA,
                        help='ncdm_ceff2_eta_A (default 0.55; not in the tag - keep the '
                             'notebook setup cell in sync if you change it)')
    args = parser.parse_args()
    common.run_jobs(build_jobs(args), args, 'nb17_cache')


if __name__ == '__main__':
    main()
