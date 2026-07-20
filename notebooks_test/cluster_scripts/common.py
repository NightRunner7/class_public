"""Shared machinery for the nb15-nb19 cache generators.

Each gen_nbNN_cache.py reproduces, run for run, the CLASS calls its notebook
makes and pickles the results under <cache-root>/nbNN_cache/ with the exact
filenames the notebook's cache lookup uses. Copy the produced *.pkl into
notebooks_test/accDM_scans/nbNN_cache/ and the notebook re-run turns into
pure cache hits.

Common CLI (every generator):
  --list             enumerate jobs with their array indices and exit
  --index N          run only job N (SLURM array mode; indices as in --list)
  --only SUBSTR      keep only jobs whose tag contains SUBSTR (applied BEFORE
                     --list/--index, so use the same --only when listing and
                     when submitting the array)
  --force            recompute even if the pickle already exists
  --cache-root DIR   where nbNN_cache/ is created (default ./accDM_scans)
"""
import argparse
import os
import pickle
import time

import numpy as np
from classy import Class

# --- base cosmology, identical across nb15-nb19 ---
OMEGA_B, OMEGA_CDM0 = 0.022383, 0.12011
A_S, N_S, TAU_REIO, H0 = 2.1005829616811546e-9, 0.96605, 0.0543, 67.32
BASE_PARAMS = {'omega_b': OMEGA_B, 'omega_cdm': OMEGA_CDM0, 'H0': H0,
               'A_s': A_S, 'n_s': N_S, 'tau_reio': TAU_REIO}

# --- accDM production knobs shared by all five notebooks ---
KAPPA, A_T, MASS = 6.0, 0.13, 1e16
A_REC = 1.0 / (1.0 + 1090.0)


def prec_block(p_k_max):
    """The notebooks' PREC dict; nb15 uses P_k_max=10, nb16-nb19 use 1.0."""
    return {'output': 'mPk', 'P_k_max_1/Mpc': p_k_max, 'z_max_pk': 0.0,
            'evolver': 0, 'reionization_z_start_max': 80}


def accdm_params(f_acc, eta, q_size, kappa=KAPPA, a_t=A_T, p_k_max=1.0):
    """Exact-hierarchy accDM params (verbatim from the nb15-nb19 builders)."""
    ocdm = OMEGA_CDM0 * (1 + f_acc*(1 - A_REC**kappa)/(1 + (A_REC/a_t)**kappa))**(-1)
    p = dict(BASE_PARAMS); p.update(prec_block(p_k_max))
    p.update({'omega_cdm': ocdm,
              'vary_Gamma_acc': 'yes', 'kappa_acc': kappa, 'a_t_acc': a_t,
              'f_acc': f_acc, 'eta_acc': eta,
              'm_acc_in_GeV': MASS, 'm_cdm_in_GeV': MASS,
              'N_ncdm': 2, 'deg_ncdm': '3, 1',
              'm_ncdm': '0.02, {:.6e}'.format(MASS*1e9),
              'T_ncdm': '0.71611, 1', 'ncdm_quadrature_strategy': '0, 4',
              'ncdm_N_momentum_bins': '15, {}'.format(q_size), 'N_ur': 0.00441,
              'background_Nloga': 5001, 'gauge': 'synchronous',
              'get_perturbations_in_current_gauge': 'yes',
              'ncdm_fluid_trigger_tau_over_tau_k': 25,
              'ncdm_fluid_approximation': 3})           # 3 = none (exact hierarchy)
    return p


def compute_pk_elapsed(params, k_list):
    """nb16/nb17/nb18 cache payload: tuple (pk array on k_list, wall seconds)."""
    M = Class(); M.set(params)
    start = time.perf_counter(); M.compute(); elapsed = time.perf_counter() - start
    pk = np.array([M.pk(float(k), 0.0) for k in k_list])
    M.struct_cleanup(); M.empty()
    return (pk, elapsed)


def compute_pk_record(params, k_list):
    """nb19 cache payload: dict(pk on k_list, sigma8, seconds)."""
    t_start = time.time()
    cosmo = Class(); cosmo.set(params); cosmo.compute()
    pk_values = np.array([cosmo.pk(k_value, 0.0) for k_value in k_list])
    rec = {'pk': pk_values, 'sigma8': cosmo.sigma8(), 'seconds': time.time() - t_start}
    cosmo.struct_cleanup(); cosmo.empty()
    return rec


def _find_key(d, want):
    if want in d:
        return want
    for kk in d:
        if kk.replace(' ', '').startswith(want.replace(' ', '')):
            return kk
    raise KeyError('{!r} not found; available: {}'.format(want, list(d.keys())))


def extract_daughter_series(params, k_list):
    """nb15/nb18 cache payload: {k: dict of daughter tau-series arrays}.

    Verbatim copy of the notebooks' extractor so the pickled structure
    (numpy-float keys, same field names) is identical.
    """
    ks = np.sort(np.asarray(k_list, float))
    p = dict(params); p['k_output_values'] = ', '.join('{:.8e}'.format(k) for k in ks)
    M = Class(); M.set(p); M.compute()
    perts = M.get_perturbations()['scalar']
    bg = M.get_background()
    tau_bg = np.asarray(bg['conf. time [Mpc]'], float)
    a_bg   = 1.0 / (1.0 + np.asarray(bg['z'], float))
    H_bg   = np.asarray(bg['H [1/Mpc]'], float)
    o = np.argsort(tau_bg); tau_bg, a_bg, H_bg = tau_bg[o], a_bg[o], H_bg[o]
    kd  = _find_key(perts[0], 'delta_ncdm[1]'); kt  = _find_key(perts[0], 'theta_ncdm[1]')
    ksh = _find_key(perts[0], 'shear_ncdm[1]'); kc  = _find_key(perts[0], 'cs2_ncdm[1]')
    kf  = _find_key(perts[0], 'k_fss_acc[1]');  ktau = _find_key(perts[0], 'tau')
    out = {}
    for k, d in zip(ks, perts):
        tau = np.asarray(d[ktau], float)
        a   = np.interp(tau, tau_bg, a_bg)
        aH  = a * np.interp(tau, tau_bg, H_bg)
        out[k] = dict(tau=tau, a=a, aH=aH,
                      delta=np.asarray(d[kd], float), theta=np.asarray(d[kt], float),
                      shear=np.asarray(d[ksh], float), dpr=np.asarray(d[kc], float),
                      k_fs=np.asarray(d[kf], float))
    M.struct_cleanup(); M.empty()
    return out


class Job(object):
    """One cache file to produce: tag (notebook cache key), filename, thunk."""

    def __init__(self, tag, filename, thunk):
        self.tag = tag
        self.filename = filename
        self.thunk = thunk


def make_parser(description):
    parser = argparse.ArgumentParser(
        description=description, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--cache-root', default='accDM_scans',
                        help='directory under which nbNN_cache/ is written (default ./accDM_scans)')
    parser.add_argument('--list', action='store_true', help='list jobs (with indices) and exit')
    parser.add_argument('--index', type=int, default=None,
                        help='run only the job with this index (SLURM array mode)')
    parser.add_argument('--only', default=None, help='keep only jobs whose tag contains this substring')
    parser.add_argument('--force', action='store_true', help='recompute even if the pickle exists')
    return parser


def run_jobs(jobs, args, cache_subdir):
    """Filter/list/execute jobs; writes are atomic (tmp file + rename) so
    concurrent array tasks can share one cache directory safely."""
    cache_dir = os.path.join(args.cache_root, cache_subdir)
    if args.only is not None:
        jobs = [job for job in jobs if args.only in job.tag]
    if args.list:
        for i, job in enumerate(jobs):
            have = os.path.exists(os.path.join(cache_dir, job.filename))
            print('{:4d}  {}  {}'.format(i, 'HAVE' if have else '....', job.tag))
        print('# {} jobs -> {}'.format(len(jobs), cache_dir))
        return
    if args.index is not None:
        if not (0 <= args.index < len(jobs)):
            raise SystemExit('--index {} out of range (0..{})'.format(args.index, len(jobs) - 1))
        jobs = [jobs[args.index]]
    os.makedirs(cache_dir, exist_ok=True)
    for job in jobs:
        path = os.path.join(cache_dir, job.filename)
        if os.path.exists(path) and not args.force:
            print('[skip     ] {} (exists)'.format(job.tag), flush=True)
            continue
        print('[computing] {} ...'.format(job.tag), flush=True)
        start = time.perf_counter()
        payload = job.thunk()
        tmp_path = '{}.tmp.{}'.format(path, os.getpid())
        with open(tmp_path, 'wb') as fh:
            pickle.dump(payload, fh)
        os.replace(tmp_path, path)
        print('[done     ] {} in {:.0f} s -> {}'.format(
            job.tag, time.perf_counter() - start, path), flush=True)
