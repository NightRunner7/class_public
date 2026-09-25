# accDM dark-energy sink — Design Spec

**Status:** approved design (2026-09-25). Background physics change behind an opt-in flag.

## Context and goal

Each accDM conversion removes one parent of rest mass M and creates one daughter of energy
M(1+η) (`P_acc = M sqrt(η(η+2))`, `input.c:2664`; default η = 1e11/m_acc, `input.c:2644`).
The parent loses aΓρ_p per conformal time while the daughter gains (1+η)aΓρ_p, so total
stress-energy is not conserved: the background picks up +η·aQ₀ and the perturbations
Σ δQ = η·aΓρ_p δ_p. Notebooks 21, 22 and 24 in `notebooks_test/` measure both.

**Goal.** Draw the extra ηM per conversion from a w = −1 component, restoring energy
conservation in the background.

## Scope

| option | scope | verdict |
|---|---|---|
| background sink + homogeneous DE (A) | this spec | **chosen** |
| interacting vacuum: δρ_DE, momentum transfer to parent, birth dipole (B) | perturbations | deferred; decided by notebook 24 |

Out of scope: option B, chain reruns and emulator regeneration, golden-file changes, and
whether the physical energy source is DE or magnetic fields.

## Physics

Conformal time, parent and daughter unchanged:

- ρ_de_acc′ = −η aΓ_acc ρ_p, with p_de_acc = −ρ_de_acc.
- ρ_de_acc(a) = η f_acc ρ_cdm,0 J(a), with J(a) = ∫ₐ¹ F′(a′) a′⁻³ d ln a′ and
  F′ = `background_acc_birth_rate` (= Γ n_p/H as a fraction of the initial parent number).
- J(1) = 0, so today's budget, the Ω_Λ closure (`input.c:3658`, `input.c:6341`), shooting and the
  derived `Omega_Lambda` are unchanged.
- Closed form (tests only; CLASS has no GSL): with y = (a/a_t)^κ and t = y/(1+y),
  J(a) = (1+a_t^κ) a_t⁻³ [B(t₁; 1−3/κ, 1+3/κ) − B(t(a); 1−3/κ, 1+3/κ)], t₁ = t(a=1),
  B the unregularised incomplete beta function. For κ ≤ 3 the integrand grows as a^(κ−3)
  towards a → 0; J stays finite at every a > 0 but has no a → 0 limit, and the closed form
  does not apply (the beta parameters are invalid). Tests use direct quadrature there.
- The sink uses the analytic birth law, not the daughter's discrete q-grid. The residual
  non-conservation is the daughter's quadrature error and shrinks with q-bins.
- ρ_de_acc decreases in time, so the effective DE equation of state is above −1 (never phantom).

**Size.** J(0) ≈ 471 for κ = 12.1, a_t = 0.133. At f_acc = 0.3 the early-time excess is
Ω_de_acc ≈ 37 (in units of today's ρ_crit) at m_acc = 1e11 GeV, 3.7 at 1e12, 0.37 at 1e13,
0.04 at 1e14. Constraints below ~1e13 GeV will tighten sharply.

## Code changes

1. **Input** (`source/input.c`, next to `eta_acc`): `acc_de_sink = yes/no`, default `no`,
   stored as `pba->has_acc_de_sink`. `class_test` rejects `yes` without accDM.
2. **Table** (`source/background.c`): built in `background_init` right after
   `background_indices` (before anything calls `background_functions`), only when the flag is
   on. J on a uniform ln a grid from ln a = −69 (a ≈ 1e-30, below any background start, since
   `background_initial_conditions` may lower a_ini for ncdm) to 0, by Simpson per cell
   integrating backward from a = 1; lookup by cubic Hermite interpolation in ln a using the
   analytic derivative dJ/d ln a = −F′a⁻³. Depends only on κ and a_t. Freed in
   `background_free_noinput`. Arrays live in `struct background` (`include/background.h`).
3. **Background** (`background_functions`): new index `index_bg_rho_de_acc` (defined only
   with the flag). `rho_tot += ρ_de_acc`, `p_tot -= ρ_de_acc`,
   `dp_dloga += η f_acc ρ_cdm,0 F′(a) a⁻³`. Not added to `rho_m`; `rho_plus_p_tot` untouched.
4. **Output**: column `(.)rho_de_acc`; one verbose line with Ω_de_acc at a_ini.
5. **Perturbations**: no change. δρ_de_acc = 0 in synchronous gauge; the first-order leak
   η aΓρ_p δ_p remains.

## Validation

New `notebooks_test/test_de_sink.py` (pytest, background only):

1. Flag off: background table and P(k) identical to the golden regression.
2. `(.)rho_de_acc` matches η f ρ_cdm,0 J(a) from `scipy.special.betainc` to < 1e-6 relative
   for κ ∈ {5, 12.1} and a_t ∈ {0.05, 0.133}; κ = 2 against direct `scipy.integrate.quad`.
3. ρ_de_acc(1) = 0; `Omega_Lambda` and H(z=0) equal flag-off values to 1e-10 (the age and
   distances change, since the expansion history does). Also at m_acc = 1e11 GeV, f_acc = 0.3.
4. Background conservation (integral residual Ω_K_eff of notebook 21), at `background_Nloga`
   = 40001 and 51 / 101 `qm_acc_birth` bins: flag on cuts the flag-off residual by > 20×, and
   |on(η=0.1) − off(η=1e-6)| < 1e-3 |off(η=0.1)|. The remaining residual is not monotonic in
   n_q: it is the estimator's trapezoid across the daughter's birth staircase plus the ΛCDM
   floor (1.1e-4 at Nloga = 10001, 1.7e-6 at 40001), present with η ≈ 0 and no sink.
5. `p_tot_prime` matches a finite difference of `p_tot` from the table.
6. `acc_de_sink = yes` without accDM fails with a CLASS error.

Rerun notebooks 21 and 22 with the flag on (expect "conserved") and notebook 24 (expect the
first-order leak, which sizes the option-A residual and decides whether B is needed).

Builds and tests run where `classy` is built (cluster or WSL); this Windows host has no toolchain.

## Risks

- **Spline accuracy near a_t** at large κ (sharply peaked F′): covered by test 2.
- **Large early DE** at m_acc ~ 1e11 GeV changes H strongly: one run at that corner in test 3.
- **Sign error in `dp_dloga`** would silently corrupt `p_tot_prime` users: covered by test 5.

## Results (2026-09-25 implementation)

Audit notebooks rerun with every accDM run at `acc_de_sink = yes`; flag-off values are their saved outputs.

| check | flag off | flag on | floor / reference |
|---|---|---|---|
| nb21 Ω_K_eff(1), fiducial (f=0.1, η=0.1) | 1.85e-2 | 1.31e-4 | ΛCDM 1.07e-4 |
| nb21 Ω_K_eff(1), worst point (f=1, η=0.5) | 5.0e-1 | 3.4e-4 | Planck 1.9e-3 (1σ) |
| nb21 max abs R_windowed, fiducial | 1.59e-2 | 4.80e-3 | 4.39e-3 at η=1e-5 (birth staircase) |
| nb22 largest abs Δ(1) over η ≤ 1 | 5.79e-2 | 1.45e-5 | ΛCDM 1.4e-12 |
| nb28 Δ_h(a=1), k = 0.1 | 8.67e-3 | 6.63e-3 | pre-injection floor 1.1e-4 |
| nb28 meas/pred (ΣδQ = η aΓρ_p δ_p), k = 0.01 / 0.1 / 1 | 1.38 / 1.37 / 1.37 | 1.03 / 1.05 / 1.04 | 1 |

- Background: the violation is gone to the estimator floor at every scanned point. The remaining
  max abs R is the windowed estimator aliasing on the daughter's birth staircase; it is present at η ≈ 0
  without the sink and scales with f_acc, so the "above 1e-2" verdict line in nb21 is not an
  energy leak. The remaining Δ(1) in nb22 is the daughter's quadrature (1+η_eff)/(1+η) = 1.0015.
- Perturbations: the flag-off drift was the ΣδQ leak plus a k-independent ~2.3e-3 imprint of the
  background violation; the sink removes the latter, and the flag-on drift matches the ΣδQ
  prediction to 2.5-5%.

Option B needed: yes, for full conservation. With the sink on the only leak left is first order,
ΣδQ = η aΓρ_p δ_p, at 6.6e-3 of max abs h' (60× the floor) for f_acc = 0.1, η = 0.1. Whether it matters
observationally is a C_l / P(k) comparison, not decided here.
