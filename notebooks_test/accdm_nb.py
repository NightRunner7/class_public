"""Shared helpers for the accDM notebooks.

- Model inputs: Planck 2018 + accDM (`lcdm_params`, `accdm_params`) and the settings of the
  fixed-mass chains (`CONNECT_NEW`, `LITE_LIN_11`).
- `RunCache`: CLASS runs in a child process (a crash or hang becomes an error entry), cached on
  disk by their full input dict.
- `observables`: the standard set of outputs (lensed C_l, P_cb, P_m, BAO, sigma8, ...).
- `chi2_parts`: approximate Delta chi2 between two runs (Planck-like CMB, lensing amplitude, BAO).
- The strategy-5 daughter grid in Python: `born_fraction`, `a_min`, `q_size`, `node_lna`.
- Chains: `chains_dir`, `load_chain`, weighted statistics.
"""
import json
import multiprocessing as mp
import pickle
import queue
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from classy import Class

# ---------------------------------------------------------------------------------------------
# Model inputs
# ---------------------------------------------------------------------------------------------

PLANCK18 = {'omega_b': 0.022383, 'omega_cdm': 0.12011, 'H0': 67.32,
            'A_s': 2.1005829616811546e-9, 'n_s': 0.96605, 'tau_reio': 0.0543}
KAPPA, A_T = 12.1, 0.133          # the transition used by the chains
A_REC = 1/1091


def lcdm_params(**extra):
    """Planck 2018 with one massive neutrino species (3 x 0.02 eV), no accDM.

    Same neutrino sector and solver settings as accdm_params, so the two differ only by accDM.
    """
    return {**PLANCK18, 'N_ncdm': 1, 'deg_ncdm': 3, 'm_ncdm': 0.02, 'T_ncdm': 0.71611,
            'N_ur': 0.00441, 'ncdm_fluid_approximation': 3, 'evolver': 0, **extra}


def omega_cdm_rescaled(f_acc, kappa=KAPPA, a_t=A_T):
    """Stable CDM such that stable + parent CDM at recombination equals Planck's omega_cdm."""
    parent = f_acc*(1 - A_REC**kappa)/(1 + (A_REC/a_t)**kappa)
    return PLANCK18['omega_cdm']/(1 + parent)


def accdm_params(f_acc, mass, eta=None, kappa=KAPPA, a_t=A_T, **extra):
    """Planck 2018 + neutrinos + accDM, daughter as the last ncdm species on the exact hierarchy.

    eta defaults to CLASS's 1e11 GeV / mass. The DE sink is on unless extra sets acc_de_sink.
    """
    p = {**PLANCK18, 'omega_cdm': omega_cdm_rescaled(f_acc, kappa, a_t),
         'vary_Gamma_acc': 'yes', 'kappa_acc': kappa, 'a_t_acc': a_t, 'f_acc': f_acc,
         'm_acc_in_GeV': mass, 'm_cdm_in_GeV': mass,
         'N_ncdm': 2, 'deg_ncdm': '3, 1', 'm_ncdm': '0.02, {:.6e}'.format(mass*1e9),
         'T_ncdm': '0.71611, 1', 'N_ur': 0.00441, 'ncdm_quadrature_strategy': '0, 5',
         'ncdm_fluid_approximation': 3, 'evolver': 0}
    if eta is not None:
        p['eta_acc'] = eta
    return {**p, **extra}


# CLASS settings of the fixed-mass CONNECT chains in connect/new (until 2026-10-05). Those chains
# ran before the DE sink and the born-fraction grid became defaults, so both are pinned off.
CONNECT_NEW = {'k_pivot': 0.05, 'lensing': 'yes', 'N_ur': 2.0308, 'N_ncdm': 2, 'deg_ncdm': '1, 1',
               'm_ncdm': '0.06, 0.01', 'T_ncdm': '0.71611, 1.0', 'ncdm_quadrature_strategy': '0, 5',
               'ncdm_fluid_approximation': 3, 'kappa_acc': KAPPA, 'a_t_acc': A_T,
               'vary_Gamma_acc': 'yes', 'gauge': 'synchronous', 'reionization_z_start_max': 80,
               'evolver': 0, 'P_k_max_1/Mpc': 1., 'output': 'tCl,pCl,lCl,mPk', 'l_max_scalars': 2508,
               'acc_de_sink': 'no', 'accdm_q_log_share': 1}

# Best fit of the connect/new 10^14 GeV chain (f_acc ~ 1e-3), as of 2026-10-01 (sampled omega_cdm)
# and 2026-10-05 (sampled omega_dm_tot). Fixed here so the cached runs stay valid when chains change.
LCDM_NEW14_OCT01 = {'omega_b': 0.02249525, 'H0': 68.36639, 'ln10^{10}A_s': 3.051855,
                    'n_s': 0.9703528, 'tau_reio': 0.06007495}
OMEGA_DM_TOT_OCT01 = 0.11764151328772961
LCDM_NEW14_OCT05 = {'omega_b': 0.02252667, 'H0': 68.3547, 'ln10^{10}A_s': 3.054243,
                    'n_s': 0.9699094, 'tau_reio': 0.06047878}
OMEGA_DM_TOT_OCT05 = 0.1177134

# connect/Lite_lin/11 (10^11 GeV, emulator q501): log.param settings and best fit.
LITE_LIN_11 = {'N_ncdm': 2, 'N_ur': 2.0308, 'm_ncdm': '0.06, 0.1', 'deg_ncdm': '1, 1',
               'T_ncdm': '0.71611, 1.0', 'm_acc_in_GeV': 1e11, 'kappa_acc': KAPPA, 'a_t_acc': A_T,
               'vary_Gamma_acc': 'yes', 'ncdm_fluid_approximation': 3, 'gauge': 'synchronous',
               'reionization_z_start_max': 80, 'evolver': 0, 'output': 'mPk, lCl, tCl, pCl',
               'P_k_max_1/Mpc': 1.0, 'lensing': 'yes', 'l_max_scalars': 2508, 'accdm_q_log_share': 1}
LCDM_LITE_LIN_11 = {'omega_b': 0.02256944, 'omega_cdm': 0.1175042, 'H0': 68.47743,
                    'ln10^{10}A_s': 3.056426, 'n_s': 0.972248, 'tau_reio': 0.06151263}

# ---------------------------------------------------------------------------------------------
# Running CLASS
# ---------------------------------------------------------------------------------------------

LMAX = 2500
ELL = np.arange(2, LMAX + 1)
KK = np.logspace(-4, np.log10(0.9), 300)                             # 1/Mpc, inside P_k_max = 1
Z_BAO = np.array([0.295, 0.510, 0.706, 0.934, 1.321, 1.484, 2.330])  # DESI DR2 effective z


def observables(cosmo):
    """Lensed C_l, P_cb and P_m at z = 0, BAO distances, sigma8, theta_s and Omega_acc."""
    cl = cosmo.lensed_cl(LMAX)
    out = {s: np.array(cl[s][2:]) for s in ('tt', 'ee', 'te', 'pp')}
    out['pk_cb'] = np.array([cosmo.pk_cb(k, 0.0) for k in KK])
    out['pk_m'] = np.array([cosmo.pk(k, 0.0) for k in KK])
    rd = cosmo.rs_drag()
    out['DM_rd'] = np.array([cosmo.angular_distance(z)*(1 + z) for z in Z_BAO])/rd
    out['DH_rd'] = np.array([1/cosmo.Hubble(z) for z in Z_BAO])/rd
    out.update(cosmo.get_current_derived_parameters(['100*theta_s', 'Omega_Lambda', 'sigma8']))
    out['sigma8_cb'] = cosmo.sigma8_cb()
    out['T_cmb'] = cosmo.T_cmb()
    bg = cosmo.get_background()
    if '(.)rho_ncdm[1]' in bg:
        out['Omega_acc'] = bg['(.)rho_ncdm[1]'][-1]/bg['(.)rho_crit'][-1]
    return out


def run_class(p, extract=observables, level=None):
    """extract(cosmo) after computing p; {'error': message} if CLASS fails."""
    cosmo = Class()
    cosmo.set(p)
    t0 = time.time()
    try:
        if level:
            cosmo.compute(level)
        else:
            cosmo.compute()
        out = extract(cosmo)
    except Exception as err:
        lines = [s.strip() for s in str(err).splitlines() if s.strip()]
        out = {'error': '{}: {}'.format(type(err).__name__, lines[-1] if lines else err)}
    finally:
        cosmo.struct_cleanup()
        cosmo.empty()
    out['sec'] = time.time() - t0
    if 'error' not in out:
        out['finite'] = all(_finite(v) for v in out.values())
    return out


def _finite(v):
    if isinstance(v, (bool, np.bool_)):
        return True
    if isinstance(v, (int, float, np.number)):
        return bool(np.isfinite(v))
    if isinstance(v, np.ndarray) and v.dtype.kind in 'fc':
        return bool(np.all(np.isfinite(v)))
    return True


def _child(p, extract, level, q):
    q.put(run_class(p, extract, level))


def run_isolated(p, extract=observables, level=None, timeout=1800.0):
    """run_class in a forked child, so a segfault or a hang becomes an error entry."""
    ctx = mp.get_context('fork')
    q = ctx.Queue()
    proc = ctx.Process(target=_child, args=(p, extract, level, q))
    t0 = time.time()
    proc.start()
    out = None
    while out is None:
        try:
            out = q.get(timeout=1.0)
        except queue.Empty:
            if not proc.is_alive():
                try:
                    out = q.get(timeout=2.0)
                except queue.Empty:
                    out = {'error': 'CLASS process died, exit code {}'.format(proc.exitcode)}
            elif time.time() - t0 > timeout:
                proc.kill()
                out = {'error': 'timeout after {:.0f} s'.format(timeout)}
    proc.join()
    out.setdefault('sec', time.time() - t0)
    return out


def ok(out):
    return 'error' not in out and out.get('finite', True)


def status(out):
    return out['error'][:120] if 'error' in out else ('ok' if out.get('finite', True) else 'NON-FINITE')


class RunCache:
    """CLASS results cached in a pickle, keyed by the full input dict.

    cache(p) returns one result; cache.many(ps) runs the missing ones `workers` at a time.
    Files in `also` are read but never written (results of another notebook).
    """

    def __init__(self, path, extract=observables, level=None, workers=1, timeout=1800.0, also=()):
        self.path = Path(path)
        self.extract, self.level, self.workers, self.timeout = extract, level, workers, timeout
        self.data = pickle.loads(self.path.read_bytes()) if self.path.exists() else {}
        self.extra = {}
        for other in also:
            self.extra.update(pickle.loads(Path(other).read_bytes()))

    def key(self, p):
        """The input dict as sorted JSON; runs stopped at a level are keyed apart from full runs."""
        return json.dumps(p if self.level is None else {'params': p, 'level': self.level}, sort_keys=True)

    def _get(self, k):
        return self.data[k] if k in self.data else self.extra.get(k)

    def _run(self, p):
        return run_isolated(p, self.extract, self.level, self.timeout)

    def _save(self):
        """Write, merged with what other caches on the same file have written meanwhile."""
        disk = pickle.loads(self.path.read_bytes()) if self.path.exists() else {}
        disk.update(self.data)
        self.data = disk
        self.path.write_bytes(pickle.dumps(disk))

    def __call__(self, p):
        return self.many([p])[0]

    def many(self, ps, verbose=True):
        todo = list({self.key(p): p for p in ps if self._get(self.key(p)) is None}.values())
        if todo:
            t0 = time.time()
            with ThreadPoolExecutor(self.workers) as pool:
                for i, (p, out) in enumerate(zip(todo, pool.map(self._run, todo))):
                    self.data[self.key(p)] = out
                    if (i + 1) % self.workers == 0 or i + 1 == len(todo):
                        self._save()
            if verbose:
                print('{} new CLASS runs in {:.0f} s'.format(len(todo), time.time() - t0))
        return [self._get(self.key(p)) for p in ps]


# ---------------------------------------------------------------------------------------------
# Approximate Delta chi2 (a sensitivity scale with LCDM fixed, not a likelihood)
# ---------------------------------------------------------------------------------------------

FSKY = 0.6
_ARCMIN = np.pi/(180*60)
BEAM, NOISE_T, NOISE_P = 7.0*_ARCMIN, 33.0*_ARCMIN, 70.0*_ARCMIN    # rad, uK rad
SIGMA_LENS = 0.025      # lensing amplitude error
SIGMA_BAO = 0.01        # per distance, uncorrelated
L_LENS = (ELL >= 8) & (ELL <= 400)


def noise_cl(sigma, t_cmb):
    """Beam-deconvolved white noise in the dimensionless units of lensed_cl."""
    return (sigma/(t_cmb*1e6))**2*np.exp(ELL*(ELL + 1)*BEAM**2/(8*np.log(2)))


def chi2_parts(a, b):
    """Delta chi2 of run a against run b, split into CMB (TT, EE, TE), lensing and BAO.

    CMB: Gaussian covariance with Planck-like noise (7' beam, 33/70 uK'), f_sky = 0.6, l = 2-2500.
    Lensing: shift of the (2L+1)-weighted mean of C_L^phiphi over L = 8-400, 2.5% error.
    BAO: D_M/r_d and D_H/r_d at the 7 DESI DR2 redshifts, 1% each.
    """
    tt = b['tt'] + noise_cl(NOISE_T, b['T_cmb'])
    ee = b['ee'] + noise_cl(NOISE_P, b['T_cmb'])
    te = b['te']
    d = np.array([a['tt'] - b['tt'], a['ee'] - b['ee'], a['te'] - b['te']]).T
    cov = np.array([[2*tt**2, 2*te**2, 2*tt*te],
                    [2*te**2, 2*ee**2, 2*ee*te],
                    [2*tt*te, 2*ee*te, tt*ee + te**2]])/((2*ELL + 1)*FSKY)
    x = np.linalg.solve(np.moveaxis(cov, 2, 0), d[:, :, None])[:, :, 0]
    cmb = float(np.sum(d*x))
    w = (2*ELL + 1)[L_LENS]
    lens = float((np.sum(w*(a['pp']/b['pp'] - 1)[L_LENS])/np.sum(w)/SIGMA_LENS)**2)
    bao = float(np.sum((a['DM_rd']/b['DM_rd'] - 1)**2 + (a['DH_rd']/b['DH_rd'] - 1)**2)/SIGMA_BAO**2)
    return {'total': cmb + lens + bao, 'cmb': cmb, 'lens': lens, 'bao': bao}


def compare(a, b):
    """Delta chi2 parts, max |dP_cb/P_cb|, and relative sigma8_cb and Omega_acc of a against b."""
    out = {**chi2_parts(a, b), 'dpk': float(np.max(np.abs(a['pk_cb']/b['pk_cb'] - 1))),
           'dsigma8': a['sigma8_cb']/b['sigma8_cb'] - 1}
    if 'Omega_acc' in a and 'Omega_acc' in b:
        out['domega'] = a['Omega_acc']/b['Omega_acc'] - 1
    return out


# ---------------------------------------------------------------------------------------------
# The strategy-5 daughter grid, as CLASS builds it
# ---------------------------------------------------------------------------------------------

def born_fraction(a, kappa=KAPPA, a_t=A_T):
    """F(a), the fraction of daughters born by a (background_acc_born_fraction)."""
    with np.errstate(over='ignore'):
        return 1 - (1 - a**kappa)/(1 + (a/a_t)**kappa)


def a_min(kappa=KAPPA, a_t=A_T, eps=1e-6):
    """Earliest birth on the grid: F(a_min) = accdm_q_number_tol (background_acc_a_min)."""
    lo, hi = np.log(1e-14), 0.0
    if born_fraction(np.exp(lo), kappa, a_t) >= eps:
        return np.exp(lo)
    for _ in range(200):
        mid = 0.5*(lo + hi)
        lo, hi = (mid, hi) if born_fraction(np.exp(mid), kappa, a_t) < eps else (lo, mid)
    return np.exp(lo)


def q_size(per_decade, amin):
    """Daughter bins from accdm_q_bins_per_decade (input.c): odd, at least 3."""
    n = max(3, int(np.ceil(per_decade*np.log10(1/amin))) + 1)
    return n + (n % 2 == 0)


def node_lna(n, share=0.25, kappa=KAPPA, a_t=A_T, eps=1e-6):
    """ln a_q of the n grid nodes: a share `share` even in ln a, the rest even in born fraction."""
    amin = a_min(kappa, a_t, eps)
    lna = np.linspace(np.log(amin), 0, 400001)
    F = born_fraction(np.exp(lna), kappa, a_t)
    u = share*(lna - lna[0])/(-lna[0]) + (1 - share)*(F - F[0])/(1 - F[0])
    return np.interp(np.linspace(0, 1, n), u, lna)


# ---------------------------------------------------------------------------------------------
# Chains
# ---------------------------------------------------------------------------------------------

def chains_dir(*parts):
    """chains/fixed_masses/<parts> in the project, found from the working directory."""
    for d in [Path.cwd(), *Path.cwd().parents]:
        if (d/'chains'/'fixed_masses').is_dir():
            return d.joinpath('chains', 'fixed_masses', *parts)
    raise FileNotFoundError('no chains/fixed_masses above {}'.format(Path.cwd()))


def mass_dirs(root):
    """Subfolders of root named by log10 m that hold chain files, sorted by mass."""
    def is_mass(name):
        try:
            float(name)
            return True
        except ValueError:
            return False
    dirs = [d for d in Path(root).iterdir() if d.is_dir() and is_mass(d.name) and any(d.glob('*__*.txt'))]
    return sorted(dirs, key=lambda d: float(d.name))


def load_chain(directory, burn_in=0.3):
    """Post-burn-in samples of the newest MontePython run in directory.

    Adds whichever of omega_cdm, omega_dm_tot, f_acc, f_tilde the chain did not sample, and the
    f_acc prior bound f_max (inf for chains sampling f_tilde).
    """
    directory = Path(directory)
    pn = max(directory.glob('*_.paramnames'))
    names = [line.split()[0] for line in pn.read_text().splitlines() if line.strip()]
    blocks = []
    for path in sorted(directory.glob(pn.name[:-len('.paramnames')] + '_*.txt')):
        x = np.loadtxt(path, ndmin=2)
        if len(x):
            blocks.append(x[int(burn_in*len(x)):])
    x = np.vstack(blocks)
    s = {'weight': x[:, 0], 'mloglike': x[:, 1], **{n: x[:, 2 + i] for i, n in enumerate(names)}}
    for line in open(directory/'log.param'):            # chains store value/scale, e.g. 100 omega_b
        if line.startswith("data.parameters['"):
            name = line.split("'")[1]
            fields = line.split('=', 1)[1].strip(' []\n').split(',')
            if name in s and len(fields) > 4:
                s[name] = s[name]*float(fields[4])
    if 'f_acc' not in s:
        s['f_acc'] = s['f_tilde']/(1 - s['f_tilde'])
    if 'f_tilde' not in s:
        s['f_tilde'] = s['f_acc']/(1 + s['f_acc'])
    if 'omega_dm_tot' not in s:
        s['omega_dm_tot'] = s['omega_cdm']*(1 + s['f_acc'])
    if 'omega_cdm' not in s:
        s['omega_cdm'] = s['omega_dm_tot']/(1 + s['f_acc'])
    s['f_max'] = np.inf
    for line in open(directory/'log.param'):
        if line.startswith("data.parameters['f_acc']"):
            s['f_max'] = float(line.split('=')[1].strip(' []\n').split(',')[2])
    return s


def chain_settings(directory):
    """CLASS input of a chain as MontePython passes it: cosmo_arguments plus fixed per-species parameters
    (name__1, name__2, ... joined into 'v1, v2'). The CONNECT model name is dropped."""
    import ast
    species, settings = {}, {}
    for line in open(Path(directory)/'log.param'):
        if line.startswith("data.parameters['") and '__' in line:
            name, idx = line.split("'")[1].rsplit('__', 1)
            species.setdefault(name, {})[int(idx)] = ast.literal_eval(line.split('=', 1)[1].strip())[0]
        elif line.startswith('data.cosmo_arguments.update('):
            settings = ast.literal_eval(line.strip()[len('data.cosmo_arguments.update('):-1])
    for name, values in species.items():
        settings[name] = ', '.join('{:g}'.format(values[i]) for i in sorted(values))
    settings.pop('connect_model', None)
    return settings


def wmean(x, w):
    return np.sum(w*x)/np.sum(w)


def wstd(x, w):
    return np.sqrt(wmean((x - wmean(x, w))**2, w))


def wquantile(x, w, q):
    order = np.argsort(x)
    cdf = np.cumsum(w[order])
    return np.interp(q*cdf[-1], cdf, x[order])


def ess(w):
    """Effective sample size of weights w."""
    return w.sum()**2/np.sum(w**2)


# ---------------------------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------------------------

def show(df, formats=None, default='{:.3g}'):
    """Display a DataFrame with per-column format strings; missing values as '-'."""
    from IPython.display import display
    out = df.copy().astype(object)
    for col in df.columns:
        f = (formats or {}).get(col, default)
        out[col] = [('-' if v is None or (isinstance(v, float) and np.isnan(v)) else
                     str(v) if isinstance(v, (bool, np.bool_)) else
                     f.format(v) if isinstance(v, (int, float, np.number)) else v) for v in df[col]]
    display(out)


def plot_style():
    """figkit's quick style if installed, else a plain matplotlib setup."""
    try:
        import figkit
        figkit.use('quick', usetex=False)
    except ImportError:
        import matplotlib.pyplot as plt
        plt.rcParams.update({'figure.dpi': 110, 'font.size': 10})
