# Self-tuning daughter momentum-grid schedule for MCMC — Design Spec

**Status:** approved design (2026-07-02). Successor to the momentum-resolution study
(`2026-06-24-daughter-momentum-resolution-study.md`), which established the measurements
this schedule codifies.

## Context and goal

Production use is MCMC scans (CMB + LSS combined likelihoods) with PyBird/PyFowl one-loop
corrections applied downstream of CLASS. That fixes the requirements:

- The daughter resolution must be chosen **automatically at input time** from the model
  parameters — no hand-tuned config per chain point.
- CLASS must deliver linear P(k) out to the pipeline's k_max (a few h/Mpc), with the
  tightest accuracy demands in the EFT window k ≈ 0.1–0.3 h/Mpc, **and** accurate lensed
  C_l. Lowering k_max is not an available lever.
- Current testing indicates the posterior concentrates at daughter fractions f < 0.1, but
  chains formally scan the full range f ∈ [0, 1]. The schedule must therefore be fast in
  the low-f bulk and **degrade gracefully** (slower, never wrong) at high-f excursions —
  a fast-but-wrong point distorts the likelihood surface itself.

The exact Boltzmann hierarchy stays the production solver. No physics changes, no new
approximation schemes. Only the daughter's momentum-bin count becomes adaptive.

**This is an accuracy fix as much as a speedup.** Measured: exact@1001 vs exact@5001 bins
differs by >1% in P(k) for f ≳ 0.1 (`memory: accdm-fluid-f-boundary`). The current fixed
1001-bin production config is *under*-resolved exactly where chains occasionally wander,
and ~8× *over*-resolved where the posterior lives (`memory: daughter-qsize-overkill`).

## Root cause (why q_size is the lever)

The daughter costs `q_size × (l_max_ncdm+1)` ODE variables per k. Fine q-grids are needed
not to resolve the boosted decay PSD (smooth) but because after production each q-shell
free-streams, so Ψ_l(q) develops oscillations in q with phase ∝ k·q·(τ−τ_birth)/ε —
resolution demand grows with f (how accurately the daughter must be integrated) and with
k_max. The alternatives are exhausted: the 3-moment fluid is architecturally unstable for
the warm daughter at high k (~1.5× at best, late/cold only), and slaving to the parent is
invalid because the daughter free-streams away from it.

## Component 1 — q(f) schedule in `input.c`

At input time, after background parameters are resolved, compute the daughter fraction f
(the same quantity scanned in notebook 12) and set the **daughter's** momentum-bin count
(2nd entry of the manual `ncdm_N_momentum_bins` grid) from a piecewise rule.

**PROVISIONAL — breakpoints, bin counts, and the number of ranges below are placeholders
pending the user's in-progress convergence tests. The validation scan (Component 2) sets
the final table; more/different ranges are expected.**

| regime (provisional) | q_size (provisional) | note |
|---|---|---|
| f < 0.1 | ~250 | coarse regime validated in nb10/nb12 |
| 0.1 ≤ f < 0.3 | ~1000 | current production resolution |
| f ≥ 0.3 | ~2000+ | fixes current under-resolution; requires `evolver = rkck` (ndf15 dense Jacobian OOMs at large q_size, `memory: ndf15-oom-high-q`) |

Behavior and constraints:

- Calibrated **at the pipeline's k_max** (a few h/Mpc for PyBird): phase-mixing grows with
  k·τ, so the calibration k_max is part of the schedule's validity statement. If the scan
  shows strong k_max sensitivity, the rule becomes q(f, k_max).
- An explicit user-supplied `ncdm_N_momentum_bins` **always overrides** the schedule, with
  a printed notice. Existing configs change nothing silently.
- Applies only to the last ncdm species and is `has_acc`-gated
  (`memory: last-species-must-be-has-acc-gated`).
- The manual quadrature strategy ties background and perturbation grids together
  (`background.c:1594`), so the validation scan re-checks background w_ncdm(a) at each
  scheduled q_size (already clean in nb10 down to ~120 bins).
- At high-f regimes the schedule may force/require `evolver = rkck`; error out (or warn
  loudly) if the user pins ndf15 with a large scheduled grid.

## Component 2 — validation & calibration notebook

A calibration-and-regression notebook: for a grid of (m, f) points spanning all schedule
regimes, compare scheduled runs against a fine-grid reference (5001 bins) on:

- max|ΔP/P| over the full PyBird k-range, tightest tolerance in k = 0.1–0.3 h/Mpc;
- max|ΔC_l/C_l| for lensed TT/EE/φφ (CMB is in the likelihood);
- background w_ncdm(a) of the daughter (guards the bg/pert grid coupling).

Acceptance: ≤ 0.1% on P(k) and C_l at every grid point; wall-times recorded per regime.
The final schedule table is set from the measured convergence boundaries with a safety
margin, and the notebook's asserts become the regression gate. Plot style: STIX serif +
ColorBrewer (`memory: notebook-plot-style`).

## Deliberately deferred / out of scope

- **Per-species `l_max_ncdm_acc` (DEFERRED, user decision 2026-07-02).** Cutting the
  daughter's multipole cutoff 17 → ~8 once cold is an additional ~2× and, being
  per-species, would leave the neutrino hierarchy untouched — but it needs additional
  testing on the user's side first. Revisit after the q-schedule lands.
- **Fluid path**: stays as-is, off by default in production. Its ~1.5× applies only in
  the late/cold trigger regime and adds a stability caveat the MCMC doesn't need.
- **q-tail truncation and unborn-bin skipping**: second-order savings, invasive in the
  state-vector layout.
- **Integral-equation solver along characteristics (Ali-Haïmoud & Bird 2012 style) —
  UNVERIFIED claim, needs the user's own check before being relied on.** Candidate
  approach that would remove the phase-mixing cost at its root (analytic free-streaming
  kernels; ~20–40 q-nodes, no l-hierarchy; potential 20–100× on the daughter, valid warm
  or cold). The daughter *appears* to be the textbook case — each q-shell born at a known
  aq(q) with a parent-slaved IC, collisionless afterwards — but neither the method's
  applicability to the decay-sourced daughter nor the quoted gains have been verified
  here. Research-grade rewrite (per-k metric history, global iteration loop) — gets its
  own spec, starting with that verification, only if this schedule proves insufficient
  for MCMC-scale cost.
- **MontePython gotcha, documented not coded**: per-species list parameters truncate
  silently in MP wrappers (`memory: ncdm-fluid-approx-is-scalar-not-per-species`). The
  schedule lives inside `input.c`, so it works identically from Python/MCMC with no
  per-species list inputs required.

## Global constraints

- Edit `class_accDM`, never the pristine `axion_project/class_public` reference.
- No build available in the agent shell; the user builds and runs the C side.
- Only `input.c` (+ any precision-parameter plumbing) changes; perturbation physics code
  paths are untouched.

## Success criteria

1. Chain points with f < 0.1 run at the coarse-grid speed (~5–10× vs the fixed 1001-bin
   config) with ≤ 0.1% deviation in P(k) (PyBird range) and lensed C_l vs the fine-grid
   reference.
2. High-f points meet the same tolerance (resolving the current >1% under-resolution),
   accepting the slower fine grid there.
3. Explicit `ncdm_N_momentum_bins` input reproduces today's behavior bit-for-bit.
4. The validation notebook passes its asserts at every calibration grid point and pins
   the final schedule table.
