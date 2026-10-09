# accDM tests and studies

Build `classy` for this branch first. Fluid-approximation work, the energy-conservation audit and
superseded studies are in [`_archive/`](_archive/README.md), with a one-line finding for each.

Each notebook starts with its question and its result. Shared code lives in `accdm_nb.py`: model
inputs, CLASS runs with a disk cache (`NN_cache.pkl`, git-ignored; a notebook recomputes what is
missing), the approximate Δχ², the strategy-5 grid in Python, and chain reading. The studies pin the
settings they were run with (e.g. `acc_de_sink = no`, `accdm_q_log_share = 1` before those became
defaults), so a rerun reproduces their recorded results.

## Regression suite

```
python -m pytest notebooks_test
pytest --nbmake notebooks_test/0[1-4]_test_*.ipynb
```

| file | guards |
|---|---|
| `01_test_regression_golden` | background, derived parameters, C_l and P(k) of a fixed model against `golden/` |
| `02_test_limits` | f_acc → 0 and η → 0 recover ΛCDM |
| `03_test_bg_conservation` | Friedmann closure, parent analytic form, Γ_acc consistency, daughter normalization |
| `04_test_pk_freestreaming` | P(k) suppression grows with f_acc and η, cutoff moves to larger scales with η |
| `test_accdm_input_checks.py` | input validation (fluid approximation, daughter slot, f_acc) |
| `test_birth_breakpoints.py` | daughter births as integration breakpoints |
| `test_de_sink.py` | DE sink background, including restored energy conservation |
| `test_accdm_born_nodes.py` | born-fraction node placement (`accdm_q_log_share`) |
| `test_accdm_l_max.py` | daughter hierarchy length `accdm_l_max`, separate from `l_max_ncdm` |
| `test_q_schedule_smoke.py` | strategy-4 q(f) schedule |
| `test_m_nu_input.py` | `m_nu` input |

## Studies

| nb | question → answer |
|---|---|
| 05 | strategy-4 q(f) schedule against 3001 bins → no regime meets 10⁻³ (250 bins: 5·10⁻³); prefer strategy 5 |
| 06 | strategy-5 grid against strategy 4 → 3× more accurate than 501-bin strategy 4 and 10× faster |
| 07 | DE sink, perturbations → a ~6·10⁻³ violation remains with the sink on (open, "option B") |
| 08 | DE sink, observables → matters below ~10¹³ GeV (TT 1% at 10¹², 8% at 10¹¹ GeV for f_acc = 0.2) |
| 09 | P(k) jitter → in the daughter only, not removable by precision; use P_cb |
| 10 | why → a physical oscillation of the daughter density in k, period 1.3·10⁻³/Mpc |
| 11 | 10¹¹ GeV limits, quadrature vs sink → the sink (−10%); the emulator disagreed with both |
| 12 | largest stable κ, even grid → 30 at 51 bins; CLASS crashes from κ ≈ 294 (overflow) |
| 13 | ω_dm,tot instead of ω_cdm → same model, the coordinate the data measure |
| 14 | σ8 and Δχ² along the ω_dm,tot ridge → flat only above ~10^16.5 GeV |
| 15 | f̃ instead of f_acc → same model, closer to linear response, different prior |
| 16 | daughter grid convergence at large f̃ → second order, too few nodes in the birth window |
| 17 | born-fraction nodes → 50–2500× smaller error at equal cost; s = 0.25 is the default |
| 18 | born-fraction nodes across κ → stable to κ = 290 at 51 bins |
| 19 | `l_max_ncdm` for warm daughters → biases large f̃ below 10¹²·⁵ GeV, not the chains |
| 20 | integration tolerance → keep 1e-5: Δχ² ≤ 10⁻³; 1e-4 saves ≤ 11% CPU for up to 0.016 at 10¹¹ GeV |

## Old numbers

The notebooks were renumbered on 2026-10-09. Specs, plans, commit messages and the archive use the old
numbers; archived notebooks keep theirs (see `_archive/README.md`). Commit `1ec30348` briefly used
01–20 with 04 the gauge test; old nb4 is now archived.

| old | new | | old | new |
|---|---|---|---|---|
| 1 | 01 | | 31 | 10 |
| 2 | 02 | | 32 | 11 |
| 3 | 03 | | 33 | 12 |
| 4 | archived | | 34 | 13 |
| 5 | 04 | | 35 | 14 |
| 14 | 05 | | 36 | 15 |
| 27 | 06 | | 37 | 16 |
| 28 | 07 | | 38 | 17 |
| 29 | 08 | | 39 | 18 |
| 30 | 09 | | 40 | 19 |
