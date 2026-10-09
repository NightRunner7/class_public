"""CLASS runs of notebook 21 (daughter hierarchy length accdm_l_max), one per array task.

The grid below is the single definition: notebook 21 imports it. Each job writes
<cache-root>/nb21_cache/<tag>.pkl as {RunCache key: result}, so the notebook reads
them with RunCache(..., also=...). Timing jobs write <cache-root>/nb21_cache/cpu/<tag>.pkl
as {(log10 m, f_tilde, bins, l_max): [CPU seconds, CPU seconds]}.

Run from notebooks_test/:
  python cluster_scripts/gen_nb21_cache.py --list            # jobs, indices, HAVE marks
  python cluster_scripts/gen_nb21_cache.py --index 3         # one job (array mode)
  python cluster_scripts/gen_nb21_cache.py --only cold_      # filter on tags, before --index
  python cluster_scripts/gen_nb21_cache.py --force ...       # recompute existing files
"""
import argparse
import json
import os
import pickle
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))     # notebooks_test/, for accdm_nb

SETTINGS = {'N_ncdm': 2, 'N_ur': 2.0308, 'm_ncdm': '0.06, 0.1', 'deg_ncdm': '1, 1', 'T_ncdm': '0.71611, 1',
            'ncdm_quadrature_strategy': '0, 5', 'ncdm_fluid_approximation': 3, 'accdm_q_log_share': 0.25,
            'kappa_acc': 12.1, 'a_t_acc': 0.133, 'vary_Gamma_acc': 'yes', 'acc_de_sink': 'yes',
            'gauge': 'synchronous', 'reionization_z_start_max': 80, 'evolver': 0, 'k_pivot': 0.05,
            'output': 'mPk, lCl, tCl, pCl', 'P_k_max_1/Mpc': 1.0, 'lensing': 'yes', 'l_max_scalars': 2508,
            'omega_b': 0.02246728, 'omega_dm_tot': 0.1176592, 'H0': 68.34158, 'ln10^{10}A_s': 3.052865,
            'n_s': 0.9694445, 'tau_reio': 0.06127875}
F_NULL = 1e-6
NULL = (14, F_NULL, 17)

# 1. warm convergence at 50 bins/decade, against accdm_l_max 140
WARM, F_WARM, L_WARM, L_REF = [11, 11.5, 12, 12.5], [0.1, 0.3, 0.5, 0.9], [17, 25, 35, 50, 70, 100], 140
# 2. grid x l_max at the hard warm points, against (400, 140)
HARD, BINS, L_GRID = [(11, 0.3), (11, 0.5), (11, 0.9), (11.5, 0.9), (12, 0.9)], [50, 100, 200, 400], [50, 100, 140]
# 3. cold floor, against accdm_l_max 35
COLD, F_COLD, L_COLD, L_COLD_REF = [12.5, 13, 14, 16, 18], [0.1, 0.5, 0.9], [4, 6, 8, 10, 12, 17], 35
# 4. CPU time, (bins, l_max) per point, two runs each
TIME_RUNS = {(11, 0.5): [(50, 17), (50, 50), (50, 100), (100, 50), (100, 100), (200, 100)],
             (14, 0.5): [(50, 17), (50, 4), (50, 6), (50, 8), (50, 10), (50, 12)]}


def params(log10m, f_tilde, l_max, bins=50):
    return {**SETTINGS, 'm_acc_in_GeV': 10.0**log10m, 'f_tilde': f_tilde, 'accdm_l_max': l_max,
            'accdm_q_bins_per_decade': bins}


def key(p):
    """RunCache.key of a full run (level None)."""
    return json.dumps(p, sort_keys=True)


def run_points():
    """(log10 m, f_tilde, l_max, bins) of every CLASS run sections 1-3 read, without duplicates."""
    pts = [NULL + (50,)]
    pts += [(lm, ft, L, 50) for lm in WARM for ft in F_WARM for L in L_WARM + [L_REF]]
    pts += [(lm, ft, L, b) for lm, ft in HARD for b in BINS for L in L_GRID]
    pts += [(lm, ft, L, 50) for lm in COLD for ft in F_COLD for L in L_COLD + [L_COLD_REF]]
    return list(dict.fromkeys(pts))


def jobs():
    out = [('run_m{:g}_f{:g}_L{}_b{}'.format(*pt), 'run', pt) for pt in run_points()]
    out += [('cpu_m{:g}_f{:g}_b{}_L{}'.format(lm, ft, b, L), 'cpu', (lm, ft, b, L))
            for (lm, ft), runs in TIME_RUNS.items() for b, L in runs]
    return out


def path(cache_root, kind, tag):
    d = Path(cache_root)/'nb21_cache'
    return (d/'cpu' if kind == 'cpu' else d)/(tag + '.pkl')


def cpu_seconds(p):
    from classy import Class
    cosmo = Class()
    cosmo.set(p)
    t0 = time.process_time()
    cosmo.compute()
    dt = time.process_time() - t0
    cosmo.struct_cleanup()
    cosmo.empty()
    return dt


def run(kind, spec):
    if kind == 'run':
        from accdm_nb import run_class
        p = params(*spec)
        return {key(p): run_class(p)}
    lm, ft, b, L = spec
    return {spec: [cpu_seconds(params(lm, ft, L, b)) for _ in range(2)]}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--list', action='store_true')
    ap.add_argument('--index', type=int)
    ap.add_argument('--only', default='')
    ap.add_argument('--force', action='store_true')
    ap.add_argument('--cache-root', default='accDM_scans')
    args = ap.parse_args()

    todo = [j for j in jobs() if args.only in j[0]]
    if args.list:
        for i, (tag, kind, _) in enumerate(todo):
            print('{:4d}  {:5s} {}'.format(i, 'HAVE' if path(args.cache_root, kind, tag).exists() else '', tag))
        print('{} jobs'.format(len(todo)))
        return
    if args.index is not None:
        todo = [todo[args.index]]
    for tag, kind, spec in todo:
        out = path(args.cache_root, kind, tag)
        if out.exists() and not args.force:
            print('[have] ' + tag)
            continue
        t0 = time.time()
        result = run(kind, spec)
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_suffix('.tmp{}'.format(os.getpid()))
        tmp.write_bytes(pickle.dumps(result))
        tmp.rename(out)
        status = 'cpu' if kind == 'cpu' else next(iter(result.values())).get('error', 'ok')
        print('[done] {} in {:.0f} s: {}'.format(tag, time.time() - t0, status))


if __name__ == '__main__':
    main()
