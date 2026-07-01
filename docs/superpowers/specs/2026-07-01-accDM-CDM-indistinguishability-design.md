# accDM → CDM indistinguishability test — design

**Date:** 2026-07-01
**Deliverable:** `notebooks_test/13_test_accDM_CDM_indistinguishability.ipynb`
**Status:** approved design, ready for implementation plan

## Goal

Determine at what daughter mass the accelerated-DM (accDM) model becomes
observationally indistinguishable from cold dark matter. The daughter is born
with a velocity kick

```
v = sqrt(eta (eta + 2)) / (1 + eta)
```

set by the dimensionless energy boost `eta_acc`. The test adopts the physical
relation

```
eta = 1e11 / m        (m = daughter mass in GeV)
```

so a **heavier daughter has a smaller eta, free-streams less, and is colder**.
Sweeping `m` upward traces the transition from a warm, P(k)-suppressing daughter
to one that is numerically CDM. The notebook locates the mass (equivalently eta)
above which every chosen observable matches "CDM" within the indistinguishability
thresholds.

## Parameter mapping (per accDM run)

| CLASS input        | value                                  |
|--------------------|----------------------------------------|
| `eta_acc`          | `1e11 / m`                             |
| `m_acc_in_GeV`     | `m`                                    |
| `m_cdm_in_GeV`     | `m`                                    |
| daughter `m_ncdm`  | `m * 1e9` (eV)                         |
| `f_acc`            | scan value {0.01, 0.05, 0.1}           |
| `kappa_acc`, `a_t_acc` | fiducial, fixed (as notebook 5): `KAPPA=2.0`, `A_T=0.13` |
| `vary_Gamma_acc`   | `yes`                                  |

Background decay sector (`kappa_acc`, `a_t_acc`) is held fixed across the whole
scan; only `f_acc` (across curves) and `m` (along each curve) vary. `omega_cdm`
is rescaled for the decayed density exactly as in notebook 5
(`ocdm = omega_cdm0 * (1 + f_acc*(1 - A_REC**KAPPA)/(1 + (A_REC/A_T)**KAPPA))**-1`).

**Daughter momentum sector:** exact quadrature (`ncdm_quadrature_strategy = 4`),
`ncdm_fluid_approximation = 0` (the daughter fluid closure is unusable — stiff at
switch-on). `ncdm_N_momentum_bins ≈ 250` as the working value, with a convergence
spot-check against a finer grid (see Risks). `l_max_ncdm` left at the default
(17 is sufficient per prior resolution study).

## References (what "CDM" means)

Two references are computed and each mass point is compared against **both**:

1. **Cold-limit** — the identical accDM setup (same `f_acc`, `kappa_acc`,
   `a_t_acc`, density rescaling) but with `eta_acc = 1e-12`. Isolates the
   free-streaming/warmth effect and removes any background/decay offset. One
   cold-limit run **per `f_acc`**.
2. **Plain ΛCDM** — no acc species, `omega_cdm` set to the total DM density so the
   background matches. One global run. Answers "does it look like vanilla CDM",
   folding in the decay's background effect.

## Observables and metrics

Two observables, two metric families; all four combinations reported.

### Matter power P(k)
- k-grid: `logspace(-3, 0, 60)` /Mpc, `P_k_max_1/Mpc = 1.0`, `z = 0`.
- **Metric A — fixed tolerance:** `dev = max_k |P_acc/P_ref - 1|`.
  Indistinguishability crossings reported at **1%** and **0.1%**.
- **Metric B — survey chi-squared:** cosmic-variance-limited significance from
  mode counting in a fiducial survey volume `V_SURVEY` (configurable, Euclid-like
  ~ (few Gpc)^3). Per k-bin `sigma_P/P = sqrt(2 / N_modes)`,
  `N_modes = k^2 dk V / (2 pi^2)`; `sig = sqrt(sum_k ((P_acc-P_ref)/sigma_P)^2)`.

### CMB (lensed)
- `output` includes `tCl,pCl,lCl`; `lensing = yes`; `l_max_scalars ≈ 2500`.
- **Metric A — fixed tolerance:** `max_l |C_acc/C_ref - 1|` on TT (report EE too).
- **Metric B — Knox chi-squared:** cosmic-variance-limited,
  `sig^2 = sum_l (2l+1)/2 * f_sky * ((C_acc-C_ref)/C_ref)^2` over TT+TE+EE with
  `f_sky ≈ 0.7`.

For Metric B, "indistinguishable" is `sig < 1` (report the 1σ, 2σ, 3σ crossings).

## Scan structure and outputs

- `f_acc ∈ {0.01, 0.05, 0.1}`; mass grid `logspace(11, 19, 12)` GeV
  (`eta` from ~1 down to ~1e-8).
- ~40 CLASS runs (36 accDM + 3 cold refs + 1 ΛCDM), cached in a dict keyed by
  `(f_acc, m, kind)` exactly like notebook 5's `_cache`.

**Deliverables in the notebook:**
1. **Distinguishability-vs-mass curves** — for each metric × reference, one line
   per `f_acc`, plotting the metric against `m` (and a secondary `eta` axis).
2. **Threshold table** — extracted threshold mass / eta where each metric drops
   below its indistinguishability cut, per `f_acc` and per reference (linear
   interpolation on the log-mass axis to find the crossing).
3. **Diagnostic panels** — `P_acc/P_ref` vs k and `ΔC_l/C_l` vs l at a few masses
   spanning the transition (warm → threshold → cold).
4. **Hard asserts (nbmake/pytest):** monotonic trend — every metric decreases as
   `m` increases (colder → more CDM-like) — and at the top of the mass grid all
   metrics sit below their indistinguishability cuts. Qualitative/monotonic in the
   spirit of notebook 5, not fixed magnitudes.

## Risks and caveats (flagged in-notebook)

- **q-grid noise floor at the cold end.** The cold-limit signal is tiny; q-grid
  discretization noise could masquerade as a floor. A convergence spot-check
  (one mass at 250 vs a finer grid) bounds the numerical floor; thresholds below
  it are reported as "numerics-limited, not physical".
- **Super-horizon gauge limitation.** The accDM gauge-invariant construction holds
  only sub-horizon; keep the k-range sub-horizon (the `1e-3` floor is safe at z=0).
- **CMB η-sensitivity is subtle.** For cold daughters the CMB effect enters mainly
  through late-time growth/lensing; expect P(k) to set the binding threshold and
  CMB to cross earlier (become indistinguishable at lower mass).
- **Runtime.** ~40 exact-ncdm CMB runs is the heavy cost; caching + a modest mass
  grid keep it tractable. q_bins and grid size are exposed as top-of-notebook
  constants for the user to trade accuracy vs speed.

## Non-goals

- No MCMC / parameter-inference forecast; the χ² is a single-parameter
  detectability proxy, not a full Fisher/likelihood analysis.
- No variation of the decay sector (`kappa_acc`, `a_t_acc`) — fixed fiducial.
- No claim about the large-`kappa_acc` regime (known WONTFIX density bug).
