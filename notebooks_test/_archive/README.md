# Archived notebooks

Notebooks for problems that are solved or approaches that were dropped. They are kept with their
outputs as the record of what was found; most no longer run against the current build. They keep their
original numbers, which overlap with the renumbered live notebooks: "nb15" in a spec or plan written before
2026-10-09 means an archived notebook or an old live number (table in `../README.md`).
Snapshot before the move: tag `notebooks-pre-cleanup`.

Paths inside these notebooks (`accDM_scans/...`, `golden/`, imports) assume they sit in
`notebooks_test/`; copy one back there to re-run it.

## `fluid_closure/` — daughter fluid approximation (dropped)

Since `7e86acd9`, accDM requires `ncdm_fluid_approximation = 3` (exact hierarchy), so none of these
run. Overall verdict: the fluid switch-on is stiff for small triggers, the closure blows up at high k
and for warm daughters, and the speed-up never exceeded ~1.5-1.8x, against 5-10x from fewer
momentum bins and later strategy 5.

| nb | question | finding |
|---|---|---|
| 6 | fluid vs exact, trigger scan | small triggers crash (`rk` step too small); triggers ≥ 1 run at 1.1-1.4x with ~2-3% P(k) error |
| 7 | Eq-38 `ceff2` fit, 1/3 cap (mode 1), amplitude | amp ~1.0-1.5 halves the P(k) residual (RMS 2.0 → 1.2%); high-k blow-up remains |
| 8 | `c_s^2(tau, k)` time evolution | high-κ blow-up is architectural; nb7's amplitude is valid only at low κ |
| 9 | slaving the daughter to the parent (`acctca`) | daughter does not track the parent after birth; slaving dropped |
| 11 | large triggers (> 1) | stable, 5.5-7x vs exact at q = 1001 (mostly from running at q = 250), but 1.5-7% P(k) error; trigger 1 crashes (two switches at once) |
| 12 | LHC scan, fluid@251 vs exact@1001 | 69/100 points within 1% P(k), worst 39% (low κ, small a_t) |
| 15 | closure diagnostic, plateau `ceff2` | published `W = 1 - 2 eps` has the wrong sign; plateau `c_fs` varies 50% with κ and 170% with a_t, so `c_fs(eta)` alone fails |
| 16 | plateau closure (mode 2), f = 0.3 | invalid: the runs set mode 1 by a typo (see nb17); residuals 50-4000% |
| 17 | eta-plateau closure (mode 3) | ≤ 1.6% for eta ≤ 0.05, catastrophic (10²-10⁴) for eta = 0.1-0.3 |
| 18 | `ceff2(f, eta)` fit + P(k) check | f-dependence absent (B = 0, A = 0.522); P(k) error 2-16%, speed-up 1.3-1.5x |
| 19 | fluid error relative to the accDM signal | warm corner has a secular +5-8% P(k) excess (under-damped shear), not a switch transient |
| 20 | q-bins vs f, exact and fluid arms vs q = 10001 | strategy-4 bins for Δσ8 < 1e-3 grow with f (201-501 up to f = 0.3, 750-7500 at f ≥ 0.5); fluid is ~1.4x faster at equal q but never reaches 1e-3 for f ≥ 0.2 and warm daughters |
| 25 | fluid's extra trace-equation violation | fluid adds ~70% to the drift at k = 1 |

Also here: `fluid_closure_helpers.py` and its tests, and the cluster generators
`cluster_scripts/gen_nb15..20_cache.py` (they import `common` from the live
`notebooks_test/cluster_scripts/`; put that on `PYTHONPATH` to run them). Caches stay in the
git-ignored `notebooks_test/accDM_scans/`.

## `conservation_audit/` — energy non-conservation from the kick (fixed by the DE sink)

The daughter is born with `(1+eta)` times the parent's energy and nothing paid for the extra
`eta`. The DE sink (`acc_de_sink`, spec `2026-09-25-accdm-de-sink-design.md`, on by default for
accDM) pays it from a w = -1 component and restores background conservation; this is now tested in
`test_de_sink.py::test_background_conservation_restored`. The perturbation-level residual left
with the sink on is tracked in the live nb28.

| nb | question | finding |
|---|---|---|
| 21 | does ρ_tot obey the continuity equation? | no: spurious `+eta aQ` injection, structural (Q1 = 1+eta, Q2 = -1, no reservoir) |
| 22 | drift in H from the acceleration equation | Δ(1) up to 5.8%, i.e. Ω_K,eff ~ 0.12 (61σ of Planck): invalidates inference |
| 23 | which combination of eta, κ, f controls the drift | peak rate ∝ eta·κ·Ω_parent; accumulated drift Δ(1) ∝ eta·Ω_parent (2% scatter), κ drops out |
| 24 | perturbation level: Einstein trace equation (MB 21c) | violated at ~8.5e-3; motivates option B |
| 26 | which observable the ambiguity reaches first | θ_* at 12σ for f = 0.1, eta = 0.1; requires eta·f ≤ 8e-4 |

## `superseded/`

| nb | question | superseded by |
|---|---|---|
| 10 | daughter q_size and l_max on strategy 4 | q-schedule (nb14) and strategy 5 (nb27, nb37-39); l_max by nb40 |
| 13 | mass above which accDM looks like CDM | nb35 and the fixed-mass chains (nb13 predates the DE sink and used strategy 4 at κ = 6) |

## `pre_refactor/`

The former `_OLD/`: background and perturbation notebooks from before the June 2026 refactor
(`43a395ae`).
