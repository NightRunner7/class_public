# accDM tests and studies

Build `classy` for this branch first. Fluid-approximation work, the energy-conservation audit and
superseded studies are in [`_archive/`](_archive/README.md), with a one-line finding for each.

## Regression suite

```
python -m pytest notebooks_test
pytest --nbmake notebooks_test/[1-5]_test_*.ipynb
```

| file | guards |
|---|---|
| `1_test_regression_golden` | background, derived parameters, C_l and P(k) of a fixed model against `golden/` |
| `2_test_limits` | f_acc → 0 and eta → 0 recover ΛCDM |
| `3_test_bg_conservation` | Friedmann closure, Γ_acc consistency, parent analytic form, daughter normalization |
| `4_test_gauge_invariance` | synchronous vs newtonian C_l and P(k), sub-horizon |
| `5_test_Pk_freestreaming` | P(k) suppression grows with f_acc and eta |
| `test_accdm_input_checks.py` | input validation (fluid approximation, daughter slot, f_acc) |
| `test_birth_breakpoints.py` | daughter births as integration breakpoints |
| `test_de_sink.py` | DE sink background, including restored energy conservation |
| `test_accdm_born_nodes.py` | born-fraction node placement (`accdm_q_log_share`) |
| `test_q_schedule_smoke.py` | strategy-4 q(f) schedule |
| `test_m_nu_input.py` | `m_nu` input |

## Studies

Numbered notebooks keep their numbers; gaps are archived ones. `NN_cache.pkl` files are local
(git-ignored); a notebook recomputes what is missing.

| nb | topic |
|---|---|
| 14 | strategy-4 q(f) schedule calibration |
| 27 | strategy-5 birth-scale-factor q-grid against strategy 4 |
| 28 | DE sink at perturbation level: residual trace-equation violation (open, "option B") |
| 29 | DE sink: background, CMB and P(k) on vs off |
| 30 | P(k) jitter between runs; P_cb is clean |
| 31 | the jitter is a physical oscillation of the daughter density in k |
| 32 | why the 10¹¹ GeV limits differ: quadrature vs DE sink |
| 33 | largest stable κ on the even strategy-5 grid |
| 34 | sampling ω_dm,tot instead of ω_cdm |
| 35 | σ8 and Δχ² along the ω_dm,tot ridge |
| 36 | sampling f̃ = f_acc/(1 + f_acc) |
| 37 | daughter q-grid convergence at large f̃ |
| 38 | born-fraction daughter nodes (`accdm_q_log_share`) |
| 39 | born-fraction nodes across κ |
| 40 | `l_max_ncdm` for warm daughters |
