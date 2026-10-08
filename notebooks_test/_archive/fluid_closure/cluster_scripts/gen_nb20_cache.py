"""nb20 (q-bin count vs f: accuracy & speed-up scan) cache generator.

Produces the run_pk() pickles of
notebooks_test/20_test_qbins_speed_accuracy.ipynb under nb20_cache/:

    {TAG}_truth_f{f}_eta{eta}_q{Q_TRUTH}.pkl              ground-truth exact run
    {TAG}_exact_f{f}_eta{eta}_q{q}.pkl                    exact run per ladder rung
    {TAG}_fluid_m{M}a{A}_t{trig}_shear{yes|no}_f{f}_eta{eta}_q{q}.pkl
                                                          fluid run per ladder rung

Payload: dict(pk on K_PK, sigma8, seconds) -- common.compute_pk_record.

Run matrix (defaults): 9 f-values x 6 eta-values = 54 corners; per corner one
truth run (exact, q = 10001) plus the 8-rung ladder in BOTH arms (exact and
fluid) -> 54 * (1 + 8 + 8) = 918 jobs. The truth runs dominate the cost;
submit them as their own array with --only truth (and a generous walltime),
then the exact_/fluid_ arrays.

Fluid configuration (encoded in the tag as m{mode}a{amp}): ncdm_ceff2_mode = 0
with ncdm_ceff2_fs_amp = 0, i.e. ceff2 collapses to the pure adiabatic sound
speed -- no free-streaming correction -- and switch_off_shear_acc = 'no'
(dynamical shear, the corrected equation). The abundance-gate trigger goes in
the tag as t{trig}.

Exact/truth files do not depend on the fluid knobs, so rerunning with a
different trigger/mode/amp/shear only adds the cheap fluid files.

Submission (from notebooks_test/, see cluster_scripts/README.md):

    python cluster_scripts/gen_nb20_cache.py --list                  # all 918
    python cluster_scripts/gen_nb20_cache.py --only truth --list     # 54 heavy
    qsub -t 0-53  -v 'GENERATOR=cluster_scripts/gen_nb20_cache.py,GEN_ARGS=--only truth' \
         cluster_scripts/submit_template.pbs
    qsub -t 0-431 -v 'GENERATOR=cluster_scripts/gen_nb20_cache.py,GEN_ARGS=--only exact_' \
         cluster_scripts/submit_template.pbs
    qsub -t 0-431 -v 'GENERATOR=cluster_scripts/gen_nb20_cache.py,GEN_ARGS=--only fluid_' \
         cluster_scripts/submit_template.pbs

(--only is applied before --index, so the array ID enumerates the filtered
list; always --list with the same --only first to confirm the count.)
"""
import os
import sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _SCRIPT_DIR)
import numpy as np

import common

K_PK = np.logspace(-3, 0.0, 40)

# 1e-1 in the requested list collapses onto 0.1; kept deduped here.
F_GRID = [1e-4, 1e-3, 1e-2, 0.03, 0.1, 0.2, 0.3, 0.5, 1.0]
ETA_GRID = [1e-8, 1e-6, 1e-4, 1e-2, 0.1, 1.0]
Q_LADDER = [201, 501, 751, 1001, 2001, 2501, 5001, 7501]
Q_TRUTH = 10001

TRIGGER = 0.4          # abundance-gate rho_accDM/rho_dcdm threshold
SHEAR = 'no'           # switch_off_shear_acc: 'no' = dynamical shear
CEFF2_MODE = 0         # published Eq-38 fit shape ...
CEFF2_FS_AMP = 0.0     # ... with zero amplitude -> pure adiabatic ceff2


def fluid_params(f_acc, eta, q_size, trigger, shear_off, ceff2_mode, fs_amp):
    """Adiabatic-ceff2 fluid: mode 0 with fs_amp = 0 makes ceff2 = ca2."""
    p = common.accdm_params(f_acc, eta, q_size)
    p['ncdm_fluid_approximation'] = 2
    p['ncdm_fluid_trigger_rho_accDM_over_rho_dcdm'] = trigger
    p['switch_off_shear_acc'] = shear_off
    p['ncdm_ceff2_mode'] = ceff2_mode
    p['ncdm_ceff2_fs_amp'] = fs_amp
    return p


def record_job(params, tag, cache_tag):
    filename = '{}_{}.pkl'.format(cache_tag, tag)
    return common.Job(tag, filename, lambda: common.compute_pk_record(params, K_PK))


def corner_tag(f_acc, eta):
    return 'f{:g}_eta{:g}'.format(f_acc, eta)


def build_jobs(args):
    f_values = sorted(set(float(f) for f in args.fs))
    eta_values = sorted(set(float(eta) for eta in args.etas))
    q_ladder = sorted(set(int(q) for q in args.qs))
    corners = [(f, eta) for f in f_values for eta in eta_values]
    fluid_prefix = 'fluid_m{}a{:g}_t{:g}_shear{}'.format(
        args.ceff2_mode, args.fs_amp, args.trigger, args.shear)

    jobs = []
    for f_acc, eta in corners:                       # heavy truth block first
        jobs.append(record_job(
            common.accdm_params(f_acc, eta, args.q_truth),
            'truth_{}_q{}'.format(corner_tag(f_acc, eta), args.q_truth),
            args.cache_tag))
    for f_acc, eta in corners:
        for q_size in q_ladder:
            jobs.append(record_job(
                common.accdm_params(f_acc, eta, q_size),
                'exact_{}_q{}'.format(corner_tag(f_acc, eta), q_size),
                args.cache_tag))
    for f_acc, eta in corners:
        for q_size in q_ladder:
            jobs.append(record_job(
                fluid_params(f_acc, eta, q_size, args.trigger, args.shear,
                             args.ceff2_mode, args.fs_amp),
                '{}_{}_q{}'.format(fluid_prefix, corner_tag(f_acc, eta), q_size),
                args.cache_tag))
    return jobs


def main():
    parser = common.make_parser(__doc__)
    parser.add_argument('--cache-tag', default='v1',
                        help="nb20 CACHE_TAG; must match the notebook's setup cell (default v1)")
    parser.add_argument('--fs', nargs='+', type=float, default=F_GRID,
                        help='f_acc values (deduped/sorted; default: the 9-value ladder)')
    parser.add_argument('--etas', nargs='+', type=float, default=ETA_GRID,
                        help='eta_acc values (deduped/sorted; default: the 6-value ladder)')
    parser.add_argument('--qs', nargs='+', type=int, default=Q_LADDER,
                        help='q-bin ladder tested in both arms (default: 8 rungs 201..7501)')
    parser.add_argument('--q-truth', type=int, default=Q_TRUTH,
                        help='ground-truth exact q-bin count (default 10001)')
    parser.add_argument('--trigger', type=float, default=TRIGGER,
                        help='abundance-gate trigger for the fluid runs (default 0.4; in the tag)')
    parser.add_argument('--shear', choices=['yes', 'no'], default=SHEAR,
                        help="switch_off_shear_acc: 'no' = dynamical shear (default; in the tag)")
    parser.add_argument('--ceff2-mode', type=int, default=CEFF2_MODE,
                        help='ncdm_ceff2_mode for the fluid runs (default 0; in the tag)')
    parser.add_argument('--fs-amp', type=float, default=CEFF2_FS_AMP,
                        help='ncdm_ceff2_fs_amp for the fluid runs (default 0.0; in the tag)')
    args = parser.parse_args()
    common.run_jobs(build_jobs(args), args, 'nb20_cache')


if __name__ == '__main__':
    main()
