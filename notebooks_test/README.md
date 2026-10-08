# accDM tests and studies

Build `classy` for this branch first. Fluid-approximation work, the energy-conservation audit and
superseded studies are in [`_archive/`](_archive/README.md), with a one-line finding for each.

## Regression suite

```
python -m pytest notebooks_test
pytest --nbmake notebooks_test/0[1-5]_test_*.ipynb
```

| file | guards |
|---|---|
| `01_test_regression_golden` | background, derived parameters, C_l and P(k) of a fixed model against `golden/` |
| `02_test_limits` | f_acc → 0 and eta → 0 recover ΛCDM |
| `03_test_bg_conservation` | Friedmann closure, Γ_acc consistency, parent analytic form, daughter normalization |
| `04_test_gauge_invariance` | synchronous vs newtonian C_l and P(k), sub-horizon |
| `05_test_pk_freestreaming` | P(k) suppression grows with f_acc and eta |
| `test_accdm_input_checks.py` | input validation (fluid approximation, daughter slot, f_acc) |
| `test_birth_breakpoints.py` | daughter births as integration breakpoints |
| `test_de_sink.py` | DE sink background, including restored energy conservation |
| `test_accdm_born_nodes.py` | born-fraction node placement (`accdm_q_log_share`) |
| `test_q_schedule_smoke.py` | strategy-4 q(f) schedule |
| `test_m_nu_input.py` | `m_nu` input |

## Studies

`NN_cache.pkl` files are local (git-ignored); a notebook recomputes what is missing.

| nb | topic |
|---|---|
| 06 | strategy-4 q(f) schedule calibration |
| 07 | strategy-5 birth-scale-factor q-grid against strategy 4 |
| 08 | DE sink at perturbation level: residual trace-equation violation (open, "option B") |
| 09 | DE sink: background, CMB and P(k) on vs off |
| 10 | P(k) jitter between runs; P_cb is clean |
| 11 | the jitter is a physical oscillation of the daughter density in k |
| 12 | why the 10¹¹ GeV limits differ: quadrature vs DE sink |
| 13 | largest stable κ on the even strategy-5 grid |
| 14 | sampling ω_dm,tot instead of ω_cdm |
| 15 | σ8 and Δχ² along the ω_dm,tot ridge |
| 16 | sampling f̃ = f_acc/(1 + f_acc) |
| 17 | daughter q-grid convergence at large f̃ |
| 18 | born-fraction daughter nodes (`accdm_q_log_share`) |
| 19 | born-fraction nodes across κ |
| 20 | `l_max_ncdm` for warm daughters |

## Old numbers

The notebooks were renumbered on 2026-10-09. Specs, plans, commit messages and the archive use the old
numbers; archived notebooks keep theirs (see `_archive/README.md`).

| old | new | | old | new |
|---|---|---|---|---|
| 1 | 01 | | 31 | 11 |
| 2 | 02 | | 32 | 12 |
| 3 | 03 | | 33 | 13 |
| 4 | 04 | | 34 | 14 |
| 5 | 05 | | 35 | 15 |
| 14 | 06 | | 36 | 16 |
| 27 | 07 | | 37 | 17 |
| 28 | 08 | | 38 | 18 |
| 29 | 09 | | 39 | 19 |
| 30 | 10 | | 40 | 20 |
