"""nb15 (fluid-closure diagnostic) cache generator.

Reproduces the cached_extract() pickles of
notebooks_test/15_test_fluid_closure_diagnostic.ipynb:

    nb15_cache/{TAG}_eta{eta}_f{f}_kap{kappa}_at{a_t}_nk18.pkl

Main grid: F_LIST x ETA_LIST at kappa=6, a_t=0.13 (30 runs), plus the Task-7
sweep points (kappa {2, 20} at a_t=0.13 and a_t {0.01, 0.5} at kappa=6, both
at eta=0.1, f=0.1). Payload: {k: dict of daughter tau-series arrays}.

Notebook-matching details baked in: CACHE_TAG 'v4', q_size 1001,
P_k_max_1/Mpc = 10 (nb15 only), K_GRID = logspace(-2, 0, 18).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np

import common

K_GRID = np.logspace(-2, 0.0, 18)
F_LIST = [0.01, 0.05, 0.1, 0.2, 0.3]
ETA_LIST = [0.01, 0.05, 0.1, 0.3, 0.5, 1.0]
Q_SIZE = 1001
P_K_MAX = 10.0                       # nb15 PREC (nb16-nb19 use 1.0)
KAPPA_SWEEP = [2.0, 20.0]            # Task 7; kappa=6 is already in the main grid
AT_SWEEP = [0.01, 0.5]               # Task 7; a_t=0.13 is already in the main grid
ETA_SWEEP, F_SWEEP = 0.1, 0.1


def extraction_job(eta, f_acc, kappa, a_t, cache_tag):
    tag = '{}_eta{:g}_f{:g}_kap{:g}_at{:g}_nk{}'.format(
        cache_tag, eta, f_acc, kappa, a_t, K_GRID.size)

    def thunk():
        params = common.accdm_params(f_acc, eta, Q_SIZE, kappa=kappa, a_t=a_t,
                                     p_k_max=P_K_MAX)
        return common.extract_daughter_series(params, K_GRID)

    return common.Job(tag, tag + '.pkl', thunk)


def build_jobs(args):
    jobs = []
    for eta in args.etas:
        for f_acc in args.fs:
            jobs.append(extraction_job(eta, f_acc, common.KAPPA, common.A_T, args.cache_tag))
    if args.sweep:
        for kappa in KAPPA_SWEEP:
            jobs.append(extraction_job(ETA_SWEEP, F_SWEEP, kappa, common.A_T, args.cache_tag))
        for a_t in AT_SWEEP:
            jobs.append(extraction_job(ETA_SWEEP, F_SWEEP, common.KAPPA, a_t, args.cache_tag))
    return jobs


def main():
    parser = common.make_parser(__doc__)
    parser.add_argument('--cache-tag', default='v4',
                        help="nb15 CACHE_TAG; must match the notebook's setup cell (default v4)")
    parser.add_argument('--etas', type=float, nargs='+', default=ETA_LIST)
    parser.add_argument('--fs', type=float, nargs='+', default=F_LIST)
    parser.add_argument('--no-sweep', dest='sweep', action='store_false',
                        help='skip the Task-7 kappa/a_t sweep points')
    args = parser.parse_args()
    common.run_jobs(build_jobs(args), args, 'nb15_cache')


if __name__ == '__main__':
    main()
