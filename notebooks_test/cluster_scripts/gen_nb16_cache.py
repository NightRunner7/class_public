"""nb16 (plateau fluid vs exact, f=0.3) cache generator.

Reproduces the run_pk_timed() pickles of
notebooks_test/16_test_plateau_fluid_validation.ipynb under nb16_cache/:

    ref_eta{eta}_f0.3_q5001.pkl          converged exact reference
    fluid_eta{eta}_trig{trig}.pkl        fluid runs, trigger scan
    exact_std_eta{eta}_f0.3_q501.pkl     q-grid attribution runs (cell 10)

Payload: tuple (pk array on K_PK, wall seconds).

NOTE: the notebook's fluid cell sets ncdm_ceff2_mode = 1 (the documented nb16
typo - it never actually tested mode 2). Preserved here verbatim so the files
reproduce what the notebook computes; nb17/nb18/nb19 are the corrected tests.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np

import common

K_PK = np.logspace(-3, 0.0, 40)
F_TEST = 0.3
ETA_TEST = [0.1, 1.0]
Q_REF, Q_STD = 5001, 2501            # converged reference vs production q-grid
Q_ATTRIBUTION = 501                  # cell-10 'exact_std' runs
TRIGGERS = [0.4, 0.2, 0.1, 0.05]


def fluid_params(eta, trigger):
    p = common.accdm_params(F_TEST, eta, Q_STD)
    p['ncdm_fluid_approximation'] = 2
    p['ncdm_fluid_trigger_rho_accDM_over_rho_dcdm'] = trigger
    p['switch_off_shear_acc'] = 'no'
    p['ncdm_ceff2_mode'] = 1         # verbatim nb16 cell 5 (the known typo; see module docstring)
    return p


def pk_job(params, tag):
    return common.Job(tag, tag + '.pkl', lambda: common.compute_pk_elapsed(params, K_PK))


def build_jobs(args):
    jobs = []
    for eta in args.etas:
        jobs.append(pk_job(common.accdm_params(F_TEST, eta, Q_REF),
                           'ref_eta{:g}_f{:g}_q{}'.format(eta, F_TEST, Q_REF)))
    for eta in args.etas:
        for trigger in args.triggers:
            jobs.append(pk_job(fluid_params(eta, trigger),
                               'fluid_eta{:g}_trig{:g}'.format(eta, trigger)))
    for eta in args.etas:
        jobs.append(pk_job(common.accdm_params(F_TEST, eta, Q_ATTRIBUTION),
                           'exact_std_eta{:g}_f{:g}_q{}'.format(eta, F_TEST, Q_ATTRIBUTION)))
    return jobs


def main():
    parser = common.make_parser(__doc__)
    parser.add_argument('--etas', type=float, nargs='+', default=ETA_TEST)
    parser.add_argument('--triggers', type=float, nargs='+', default=TRIGGERS)
    args = parser.parse_args()
    common.run_jobs(build_jobs(args), args, 'nb16_cache')


if __name__ == '__main__':
    main()
