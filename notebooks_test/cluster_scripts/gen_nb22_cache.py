"""CLASS runs of notebook 22 (dependence on κ and a_t), one point per array task.

The grid and the f̃₄ search below are the single definition: notebook 22 imports them and replays
each search from the cached runs. Each job writes <cache-root>/nb22_cache/<tag>.pkl as
{RunCache key: result}, so the notebook reads them with RunCache(..., also=...).

Jobs: the null, the spectra slices at 10^14 GeV, and one search per (log10 m, κ, a_t). A search job
also runs its own null (for Δχ²), and at the corners of the box the grid check at f̃₄. Ladder jobs run
daughter bins × accdm_l_max at fixed f̃ where the 201-bin, l_max 17 grid failed.

Run from notebooks_test/:
  python cluster_scripts/gen_nb22_cache.py --list            # jobs, indices, HAVE marks
  python cluster_scripts/gen_nb22_cache.py --index 3         # one job (array mode)
  python cluster_scripts/gen_nb22_cache.py --only search_    # filter on tags, before --index
  python cluster_scripts/gen_nb22_cache.py --force ...       # recompute existing files
"""
import argparse
import json
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))     # notebooks_test/, for accdm_nb

from accdm_nb import CHAIN_SETTINGS_OCT07, KAPPA, A_T, chi2_parts, ok

# the 4-parameter CLASS scan's daughter grid, on the current chains' settings
SETTINGS = {**CHAIN_SETTINGS_OCT07, 'ncdm_N_momentum_bins': '150, 201', 'accdm_l_max': 17}
HIRES = {'ncdm_N_momentum_bins': '150, 401', 'accdm_l_max': 35}
MASSES = [11, 12, 13, 14, 15, 16]
KAPPAS = [1, 2, 4, 8, 12.1, 20]
A_TS = [0.001, 0.003, 0.01, 0.03, 0.133, 0.3, 1.0]
POINTS = [(lm, k, at) for lm in MASSES for k in KAPPAS for at in A_TS]
CORNERS = [(1, 0.001), (20, 0.001), (1, 1.0), (20, 1.0), (KAPPA, A_T)]
CHI2_LIM, F_MIN, F_MAX = 4.0, 1e-5, 0.999         # F_MAX: the scan's prior bound
F0 = {11: 1e-3, 12: 3e-3, 13: 0.01, 14: 0.03, 15: 0.07, 16: 0.4}     # rough f̃₄ at the chains' choice
F_NULL = 1e-6
LM_SPEC, F_SPEC = 14, 0.05
SLICES = [('a_t', [(KAPPA, at) for at in A_TS]), ('κ', [(k, A_T) for k in KAPPAS])]
SPEC_POINTS = list(dict.fromkeys(kt for _, pts in SLICES for kt in pts))
# daughter bins x accdm_l_max where 201/17 failed (heavy, early, f̃ -> 1), with neighbours as controls
LADDER_POINTS = [(15, 1, 0.001), (15, 20, 0.001), (16, 20, 0.001), (15, 20, 0.01), (16, 20, 0.01), (14, 20, 0.001)]
LADDER_F = [0.5, 0.9]
LADDER_BINS, LADDER_L = [201, 401, 801], [17, 35, 50]


def grid(bins, l_max):
    return {'ncdm_N_momentum_bins': '150, {}'.format(bins), 'accdm_l_max': l_max}


def params(lm, ft, kappa=KAPPA, a_t=A_T, **extra):
    return {**SETTINGS, 'm_acc_in_GeV': 10.0**lm, 'f_tilde': ft, 'kappa_acc': kappa, 'a_t_acc': a_t, **extra}


NULL = params(14, F_NULL)

# ---------------------------------------------------------------------------------------------
# f̃₄: the f̃ at which Δχ² against the null reaches CHI2_LIM
# ---------------------------------------------------------------------------------------------


def sig(f):
    return float('{:.4g}'.format(f))


def bracket(h):
    lo = max((f for f, c in h.items() if c < CHI2_LIM), default=None)
    hi = min((f for f, c in h.items() if c >= CHI2_LIM), default=None)
    return lo, hi


def power_step(h, f1, f2, fallback):
    """f̃ where the power law through (f1, h[f1]) and (f2, h[f2]) reaches CHI2_LIM."""
    p = np.log(h[f2]/h[f1])/np.log(f2/f1) if f1 != f2 else 0
    if not np.isfinite(p) or p <= 0.5:
        p = 2.0
    return fallback(f2*(CHI2_LIM/h[f2])**(1/p))


def next_trial(h, tol=0.05):
    """The next f̃ to run, or [] once a bracketing pair has a run within tol of CHI2_LIM."""
    h = {f: c for f, c in h.items() if np.isfinite(c) and c > 0}
    if not h:
        return []
    lo, hi = bracket(h)
    fs = sorted(h)
    close = min(h, key=lambda f: abs(h[f]/CHI2_LIM - 1))
    if abs(h[close]/CHI2_LIM - 1) < tol:
        if lo is not None and hi is not None:
            return []
        f = close*1.05 if h[close] < CHI2_LIM else close/1.05       # step across
    elif lo is not None and hi is not None:
        f = power_step(h, lo, hi, lambda x: x)
    elif hi is None:                                  # all below: step up, at most 10x
        if fs[-1] >= F_MAX:
            return []
        f = power_step(h, fs[-2] if len(fs) > 1 else fs[-1], fs[-1], lambda x: min(x, 10*fs[-1], F_MAX))
    else:                                             # all above: step down, at most 10x
        if fs[0] <= F_MIN:
            return []
        f = power_step(h, fs[1] if len(fs) > 1 else fs[0], fs[0], lambda x: max(x, fs[0]/10, F_MIN))
    f = sig(min(max(f, F_MIN), F_MAX))
    return [] if f in h else [f]


def search(points, run_many, null, rounds=8):
    """Δχ² histories {point: {f̃: Δχ²}} and the points still unfinished after `rounds`.

    Two starting values per mass, then power-law steps in log Δχ²-log f̃. run_many(list of params)
    returns the CLASS results in order.
    """
    hist = {pt: {} for pt in points}
    trial = {pt: [sig(F0[pt[0]]/2), sig(min(2*F0[pt[0]], F_MAX))] for pt in points}
    for _ in range(rounds):
        todo = [(pt, f) for pt, fs in trial.items() for f in fs]
        if not todo:
            break
        for (pt, f), out in zip(todo, run_many([params(pt[0], f, *pt[1:]) for pt, f in todo])):
            hist[pt][f] = chi2_parts(out, null)['total'] if ok(out) else np.nan
        trial = {pt: nxt for pt in points if (nxt := next_trial(hist[pt]))}
    return hist, trial


def f4(h):
    """Interpolated f̃₄; inf if Δχ² stays below CHI2_LIM up to F_MAX, nan if not bracketed."""
    h = {f: c for f, c in h.items() if np.isfinite(c) and c > 0}
    lo, hi = bracket(h)
    if hi is None:
        return np.inf if h and max(h) >= F_MAX else np.nan
    if lo is None:
        return np.nan
    t = np.log(CHI2_LIM/h[lo])/np.log(h[hi]/h[lo])
    return float(np.exp(np.log(lo) + t*np.log(hi/lo)))


def checks(lm, kappa, a_t, f):
    """Production and high-resolution params of the grid check at f̃₄ (rounded)."""
    p = (lm, sig(f), kappa, a_t)
    return params(*p), params(*p, **HIRES)

# ---------------------------------------------------------------------------------------------
# Jobs
# ---------------------------------------------------------------------------------------------


def key(p):
    """RunCache.key of a full run (level None)."""
    return json.dumps(p, sort_keys=True)


def jobs():
    out = [('null', 'null', None)]
    out += [('spec_k{:g}_at{:g}'.format(*kt), 'spec', kt) for kt in SPEC_POINTS]
    out += [('search_m{:g}_k{:g}_at{:g}'.format(*pt), 'search', pt) for pt in POINTS]
    out += [('ladder_m{:g}_k{:g}_at{:g}_f{:g}_b{}_L{}'.format(*pt, f, b, L), 'ladder', (pt, f, b, L))
            for pt in LADDER_POINTS for f in LADDER_F for b in LADDER_BINS for L in LADDER_L]
    return out


def path(cache_root, tag):
    return Path(cache_root)/'nb22_cache'/(tag + '.pkl')


def run(kind, spec):
    from accdm_nb import run_class
    done = {}

    def run_many(ps):
        for p in ps:
            if key(p) not in done:
                done[key(p)] = run_class(p)
        return [done[key(p)] for p in ps]

    if kind == 'null':
        run_many([NULL])
    elif kind == 'spec':
        run_many([params(LM_SPEC, F_SPEC, *spec)])
    elif kind == 'ladder':
        (lm, k, at), f, b, L = spec
        run_many([params(lm, f, k, at, **grid(b, L))])
    else:
        null = run_many([NULL])[0]
        hist, _ = search([spec], run_many, null)
        f = f4(hist[spec])
        if spec[1:] in CORNERS and np.isfinite(f):
            run_many(list(checks(*spec, f)))
    return done


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
            print('{:4d}  {:5s} {}'.format(i, 'HAVE' if path(args.cache_root, tag).exists() else '', tag))
        print('{} jobs'.format(len(todo)))
        return
    if args.index is not None:
        todo = [todo[args.index]]
    for tag, kind, spec in todo:
        out = path(args.cache_root, tag)
        if out.exists() and not args.force:
            print('[have] ' + tag)
            continue
        t0 = time.time()
        result = run(kind, spec)
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_suffix('.tmp{}'.format(os.getpid()))
        tmp.write_bytes(pickle.dumps(result))
        tmp.rename(out)
        errors = [v['error'] for v in result.values() if 'error' in v]
        print('[done] {} in {:.0f} s, {} runs: {}'.format(tag, time.time() - t0, len(result), errors or 'ok'))


if __name__ == '__main__':
    main()
