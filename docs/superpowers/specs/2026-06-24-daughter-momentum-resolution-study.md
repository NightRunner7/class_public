# Daughter momentum/multipole resolution study (the safe speedup) — Design Spec

**Status:** draft (2026-06-24). Supersedes the abandoned Phase-2 `acctca` slaving direction (see `memory: daughter-free-streams-not-slaved`) and the marginal smooth-attractor-switch-on idea (does not move the architectural trigger-0.4 wall; see `memory: fluid-approx-unusable`).

**Goal:** find the smallest daughter momentum grid (`q_size`) and Boltzmann multipole cutoff (`l_max_ncdm`) that keep P(k) and C_l within tolerance of the current high-resolution exact run — buying wall-time directly, with the **exact hierarchy and all physics untouched** (no stability risk, unlike the fluid).

## Why this is the lever

The exact hierarchy is the production path (~245 s, [`6_test_fluid_vs_exact.ipynb`](../../../notebooks_test/6_test_fluid_vs_exact.ipynb)). Its cost is dominated by the daughter, whose perturbation vector is `(l_max_ncdm+1) × q_size` per k. In the production config that is `18 × 1001 ≈ 18 000` ODE variables for the daughter alone — vs ~2 for the dcdm parent. This is also why `ndf15` OOMs (dense `O(neq²)` Jacobian; `memory: ndf15-oom-high-q`) and why `rk` is slow.

The two alternative speedups are exhausted: the 3-moment **fluid** is architecturally unstable for the warm daughter at high k (cannot be patched; usable only at late trigger ≥0.4, ~1.8×), and **slaving to the parent** is invalid because the daughter free-streams away from it (`δ_acc` suppressed to ~0–7% of `δ_dcdm`, k-dependent; `memory: daughter-free-streams-not-slaved`).

**The untested question:** is `q_size = 1001` actually necessary? It is a *manual* grid resolving the boosted decay PSD (`ncdm_quadrature_strategy = 4`, `background.c:1594`). If the PSD and the daughter transfer functions converge at far fewer bins, cost falls ~linearly in `q_size` — a 5–10× safe speedup is plausible. Cost in `l_max` is also roughly linear in the hierarchy, but `l_max_ncdm` is global so it is the riskier knob.

## Cost / knob model

- **`q_size` (daughter only, SAFE primary lever).** Set by the 2nd entry of `ncdm_N_momentum_bins` (e.g. `'15, 1001'`). Manual strategy ties background and perturbation grids together (`q_size_ncdm_bg = q_size_ncdm = ncdm_input_q_size`, `background.c:1594-1595`), so reducing it coarsens the background daughter PSD too — must verify background `ρ_ncdm(a)`, `w(a)` stay within tolerance.
- **`l_max_ncdm` (GLOBAL, secondary/coupled lever).** One int for all ncdm (`precisions.h:312`; no per-species override, `perturbations.c:4128`). Reducing it also lowers the **neutrino** hierarchy accuracy, so the sweep must watch a neutrino-sensitive observable (high-ℓ C_l lensing/damping tail), not just the daughter.

## Method (new notebook `10_test_daughter_resolution.ipynb`, pure config, no rebuild)

1. **Reference.** The current production config (`ncdm_N_momentum_bins '15, 1001'`, `l_max_ncdm 17`), exact (`ncdm_fluid_approximation = none`, `evolver = 0`), κ = 4.0 — same model as notebook 6. Record P(k) on a fixed k-grid, C_l^TT (and lensed C_l for the ℓ tail), background `w_ncdm[1](a)`, and wall-time.
2. **`q_size` sweep (primary).** Hold `l_max_ncdm = 17`; vary the daughter bins `∈ {1001, 500, 250, 120, 60, 30}`. For each: deviation of P(k), C_l, and background `w_ncdm[1](a)` vs reference, plus wall-time. Find the smallest grid with `max|ΔP/P| < TOL_PK` and `max|ΔC_l/C_l| < TOL_CL` **and** background `w` deviation `< TOL_BG`.
3. **`l_max` sweep (secondary, coupled).** At the chosen `q_size`, vary `l_max_ncdm ∈ {17, 12, 8, 6}`. Watch the **neutrino-sensitive** high-ℓ C_l tail explicitly, since this knob is global. Find the smallest `l_max` within tolerance for *both* daughter and neutrino observables.
4. **Report.** Recommended `(q_size, l_max_ncdm)`, the measured speedup vs reference, and the residual P(k)/C_l plots (STIX serif + ColorBrewer, `memory: notebook-plot-style`).

## Global constraints

- Edit `class_accDM`, never the pristine reference (`memory: apply-fixes-to-working-branch`). This study needs **no C edit and no rebuild** — it is config-only against the built `classy`.
- Daughter = last ncdm species, `has_acc`-gated (`memory: last-species-must-be-has-acc-gated`).
- The exact hierarchy at full resolution is the convergence reference; tolerances are measured against it, not against observations.
- Keep κ, η_acc, masses, and the PSD definition fixed; vary only resolution.

## Success criteria

- A recommended `(q_size, l_max_ncdm)` with P(k) and C_l within tolerance (suggest `TOL_PK = TOL_CL = 1e-3`, i.e. well below the ~3% the fluid costs) of the full-resolution reference, and a demonstrated wall-time reduction.
- Background `w_ncdm[1](a)` unchanged within `TOL_BG` (suggest 1e-3) at the reduced `q_size` (guards the manual-grid background coupling).
- If P(k)/C_l do **not** converge until near 1001 / 17, the daughter is irreducibly expensive at the exact level — record that as the verdict and revisit the (research-y) reduced-moment closure.

## Risks / open points

- **Boosted PSD structure.** The decay PSD may be peaked/extended enough to genuinely need many bins; the sweep is exactly the test. If `q_size` cannot drop much, that is a real, informative negative result.
- **Global `l_max` coupling.** Reducing `l_max_ncdm` to speed up the daughter degrades the neutrino. If the neutrino sets the floor, the high-value follow-up is a small C change: a per-species `l_max_ncdm_acc` so the daughter's multipole cutoff can be lowered independently (the daughter cools and needs fewer multipoles than a relativistic relic). Spec that separately only if the global sweep shows the neutrino is the binding constraint.
- **Convergence ≠ accuracy floor.** Converging to the 1001-bin answer means matching *that* run, which is itself assumed converged. If in doubt, add one point *above* 1001 to confirm the reference is itself converged.

## Decision

If the sweep finds `q_size ≪ 1001` (and/or a lower `l_max`) within tolerance → adopt it as the default production config: a real speedup with the exact hierarchy and physics fully intact, no stability caveats. This is the cleanest win available and should be tried before any further fluid/closure work.
