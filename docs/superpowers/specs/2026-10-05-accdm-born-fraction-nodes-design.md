# accDM daughter nodes placed by born fraction — Design Spec

**Status:** approved design (2026-10-05). New precision parameter, off by default.

## Problem

Strategy 5 (`qm_acc_birth`) spreads the daughter's nodes evenly in ln a_q over [a_min, 1]. With the chain
settings (κ = 12.1, a_t = 0.133) the default 71 nodes put only 17 inside the central 98% of births
(a = 0.091–0.194) and 37 after the births are over (nb37, section 0). Notebook 37 found the resulting error
converges at second order but is large at big f̃: Δχ² up to 0.25 at 10¹⁴–10¹⁸ GeV and f̃ = 0.9, more for warm
daughters, and larger than the accDM signal itself at 10¹⁸ GeV.

The quadrature of smooth functions of birth time over the births is already accurate to ~1e-7 on this grid,
and the daughter abundance today is right to 2e-7 (nb37). The error comes from the birth window itself: each
node is born at once (born weight 0 → 1 at its birth time), so during the window the daughter appears as a
staircase of ~17 steps of 6–8% of the births each while the parent decays smoothly. More nodes inside the
window give finer steps.

A pure born-fraction grid (nodes evenly spaced in F) starves the tails and breaks the post-birth integrals
(errors up to 180% for ∫ a³ dF). A mixed node density keeps both: with node density ∝ α + dF/dln a, the
71 nodes put 37 (α = 0.5) or 49 (α = 0.2) inside the window while post-birth integrals stay accurate to
6e-6 and 3e-5.

## Design

1. **Parameter.** New precision parameter `accdm_q_log_share` = s, 0 < s ≤ 1, default 1: the share of the
   daughter's nodes spread evenly in ln a_q; the remaining 1 − s follow the born fraction. It applies to
   strategy 5 only. s = 1 keeps the existing code path, so default runs are bit-identical.
2. **Map.** With L = ln(1/a_min) and F_min = F(a_min),

   u(ln a) = s·(ln a − ln a_min)/L + (1 − s)·(F(a) − F_min)/(1 − F_min),

   monotone from u(a_min) = 0 to u(1) = 1, with F from `background_acc_born_fraction`. The node density in
   ln a is du/dln a = s/L + (1 − s)·R(a)/(1 − F_min), R = dF/dln a from `background_acc_birth_rate`.
   s ≈ αL/(αL + 1): α = 0.5 ↔ s ≈ 0.6, α = 0.2 ↔ s ≈ 0.4 at a_min = 0.0425.
3. **Nodes.** u_i = i/(N − 1); ln a_i solves u(ln a_i) = u_i by bisection on [ln a_min, 0] to 1e-14 in ln a.
   The first node is a_min and the last is exactly a = 1. q_i = a_i·P_acc/T_acc as now.
4. **Weights.** Simpson in u with h_u = 1/(N − 1):
   w_i = f0(q_i)·q_i·(dln a/du)_i·h_u·S_i, S = (1, 4, 2, …, 4, 1)/3, f0 from `background_ncdm_distribution`.
   At s = 1 this equals the existing log10-Simpson weights.
5. **Where.** A new function in `background.c`, `background_acc_q_nodes`, builds q and w for the daughter
   and is called in place of `get_qsampling_manual` when strategy 5 has s < 1. The generic quadrature
   routine does not know the birth law. Birth times, perturbation breakpoints and born weights already read
   each node's position (`aq_ncdm_acc`, `lna_birth_lo/hi_acc`), so they need no change.
6. **Node count.** Unchanged: `accdm_q_bins_per_decade` × log10(1/a_min), or an explicit
   `ncdm_N_momentum_bins`. s moves the nodes; it does not change the cost.
7. **Input checks** (`input.c`): 0 < s ≤ 1, and s < 1 together with `accdm_smooth_births` is rejected (smooth
   births tile cells evenly in ln a, and converged at first order in nb37).

## Validation

- `notebooks_test/test_accdm_born_nodes.py` (pytest, m = 10¹⁶ GeV, f̃ = 0.5, background level unless stated):
  - s = 1 given explicitly is bit-identical to the default (lensed C_ℓ);
  - s ≤ 0, s > 1, and s < 1 with smooth births are rejected at input;
  - s = 0.4 runs, and Ω_acc today matches the default grid to 1e-4. (Measured: the tails get fewer nodes, so
    Ω_acc moves by 1e-6, 2.5e-5 and 6e-5 at s = 0.6, 0.4, 0.25 against 2e-7 for the default grid, all relative
    to the daughter density; 7e-9 at s = 0.4 with 100/decade.)
- Notebook 38: at nb37's (mass, f̃) points, s ∈ {0.6, 0.4, 0.25} at 50/decade (71 nodes) and the 100/decade
  node count, Δχ² against nb37's 400/decade log-grid references (read from `37_cache.pkl`; both grids converge
  to the same continuum limit). Success: at equal node count, Δχ² drops at least 10× wherever nb37's 50/decade
  value was above 0.1, no point gets worse, and runtime stays within 10%. The notebook recommends a default s;
  changing the default is a separate decision.
- `test_de_sink.py`, `test_birth_breakpoints.py` and `test_accdm_input_checks.py` still pass.

## Out of scope

- The `l_max_ncdm = 17` truncation error of warm daughters at large f̃ (nb37, 2b).
- The bin-independent Δχ² ≈ 0.04 jumps (nb33, nb37).
- The (a/a_t)^κ overflow above κ ≈ 295 (nb33); the new map uses the same rate and born-fraction functions.
