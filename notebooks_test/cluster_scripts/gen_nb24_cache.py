"""CLASS runs of notebook 24 (daughter grid at early transitions), one run per array task.

The grid below is the single definition: notebook 24 imports it. Each job writes
<cache-root>/nb24_cache/<tag>.pkl as {RunCache key: result}, so the notebook reads them with
RunCache(..., also=...).

Run from notebooks_test/:
  python cluster_scripts/gen_nb24_cache.py --list            # jobs, indices, HAVE marks
  python cluster_scripts/gen_nb24_cache.py --index 3         # one job (array mode)
  python cluster_scripts/gen_nb24_cache.py --only b1601      # filter on tags, before --index
  python cluster_scripts/gen_nb24_cache.py --force ...       # recompute existing files
"""
import argparse
import json
import os
import pickle
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))     # notebooks_test/, for accdm_nb

from accdm_nb import CHAIN_SETTINGS_OCT07, KAPPA

# the scan's settings; only the daughter bin count changes
SETTINGS = {**CHAIN_SETTINGS_OCT07, 'accdm_l_max': 17}
F_NULL = 1e-6
BINS = [201, 401, 801, 1601]                       # 1601 is the reference
MASSES = [14, 15, 16, 17, 18]
F_TILDES = [0.5, 0.9]
# (kappa, a_t): the a_t-varying chains' kappa down to a_t = 1e-3, and the scan's kappa extremes at 1e-3
TRANSITIONS = [(KAPPA, 1e-3), (KAPPA, 3e-3), (KAPPA, 1e-2), (1, 1e-3), (20, 1e-3)]


def params(log10m, f_tilde, kappa, a_t, bins):
    return {**SETTINGS, 'm_acc_in_GeV': 10.0**log10m, 'f_tilde': f_tilde, 'kappa_acc': kappa, 'a_t_acc': a_t,
            'ncdm_N_momentum_bins': '150, {}'.format(bins)}


NULL = params(14, F_NULL, KAPPA, 0.133, 201)


def key(p):
    """RunCache.key of a full run (level None)."""
    return json.dumps(p, sort_keys=True)


def run_points():
    """(log10 m, f_tilde, kappa, a_t, bins) of every run the notebook reads."""
    return [(lm, ft, k, at, b) for k, at in TRANSITIONS for lm in MASSES for ft in F_TILDES for b in BINS]


def jobs():
    out = [('null', NULL)]
    out += [('m{:g}_f{:g}_k{:g}_at{:g}_b{}'.format(*pt), params(*pt)) for pt in run_points()]
    return out


def path(cache_root, tag):
    return Path(cache_root)/'nb24_cache'/(tag + '.pkl')


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
        for i, (tag, _) in enumerate(todo):
            print('{:4d}  {:5s} {}'.format(i, 'HAVE' if path(args.cache_root, tag).exists() else '', tag))
        print('{} jobs'.format(len(todo)))
        return
    if args.index is not None:
        todo = [todo[args.index]]
    from accdm_nb import run_class
    for tag, p in todo:
        out = path(args.cache_root, tag)
        if out.exists() and not args.force:
            print('[have] ' + tag)
            continue
        t0 = time.time()
        result = {key(p): run_class(p)}
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_suffix('.tmp{}'.format(os.getpid()))
        tmp.write_bytes(pickle.dumps(result))
        tmp.rename(out)
        print('[done] {} in {:.0f} s: {}'.format(tag, time.time() - t0, next(iter(result.values())).get('error', 'ok')))


if __name__ == '__main__':
    main()
