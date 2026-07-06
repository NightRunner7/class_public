# Plateau ceff2 closure (`ncdm_ceff2_mode = 2`) + k<=1 P(k) validation — Design Spec

**Status:** draft (2026-07-06). Implements the closure measured by [`15_test_fluid_closure_diagnostic.ipynb`](../../../notebooks_test/15_test_fluid_closure_diagnostic.ipynb) (Tasks 6-9) for the fluid approximation of the accDM daughter. Scope gated by the nb15 verdict: **valid only at fixed `(kappa, a_t)` with production completed early** (a_t <~ 0.2, kappa >~ 4-6); the production-history maturity effect makes the closure non-universal across those knobs (`memory: ceff2-fit-structure`). The user's production scans fix `(kappa, a_t)` and vary `(eta, f, mass)` — the covered regime.

**Goal:** replace the published Eq-38 `sqrt(k/k_fs)` fit (wrong sign in `W(eta)`, unbounded, blows up) with the measured plateau closure

```
ceff2(a) = min( max( ca2(a), c_fs ), 1/3 )
c_fs     = (1/3) * (1 - exp(-3 * A * ca2_bg(a=1)))
```

where `A` (`ncdm_ceff2_fs_A`, default 13.0) is the single family constant measured by nb15, and `ca2_bg(a=1)` is the daughter's background adiabatic sound speed today — so the `eta` (and `f`, mass) dependence enters automatically within a fixed-`(kappa, a_t)` family. Then validate on **P(k) over k<=1 only** at `f = 0.3`.

## Why this form (measured, not assumed)

- Median exact `ceff2 = delta_p/delta_rho` is a flat, k-independent plateau over the free-streaming bulk; `-> ca2` in the clustering regime; `-> ~1` at relativistic birth (earliest times) where the `1/3` cap is the correct causal ceiling and harmless for k<=1 (negligible daughter density then).
- Plateau is f-independent to 0.5% up to f=0.3; increases with eta (`c_fs = {0.0296, 0.0949, 0.2277}` at eta `{0.1, 0.3, 1.0}`, kappa=6, a_t=0.13).
- `c_fs = A*ca2_bg(a=1)` with `A ~ 12.4` holds within the family; the eta=1 deviation is causal saturation (unsaturated prediction 0.38 > 1/3), captured by the exponential map. De-saturated spread within the family: ~15-20%.
- The plateau is sub-1/3 everywhere in the covered regime -> the closure is causally safe by construction; no runaway like the old fit.

## C implementation (Phase 2)

1. **Input:** new precision/input parameter `ncdm_ceff2_fs_A` (double, default 13.0); `ncdm_ceff2_mode = 2` selects the plateau closure. Existing modes 0 (Eq-38 fit) and 1 (fit capped at 1/3) unchanged.
2. **Background hook:** after background integration, compute the daughter's `ca2_bg` at the last background table entry (a=1) using the same expression the code already uses for `k_fss_acc` (the source-free fallback is fine at a=1 for completed production), then store `pba->cfs_acc = (1/3)*(1 - exp(-3 * A * ca2_bg_today))`. One double on `struct background`.
3. **Closure:** `perturbations_ceff2_ncdm(...)` gains the mode-2 branch: `return min( max(cs2_base, pba->cfs_acc), 1./3. )`. Both existing call sites route through this helper — the source accumulation (`cg2` base, `perturbations.c:7369/7387`) and the fluid derivatives (`ca2` base, `perturbations.c:10121`) — so `delta_p` bookkeeping and the equations of motion stay consistent automatically.
4. **Gating:** daughter-only as now (`n_ncdm == N_ncdm-1 && has_acc`; `memory: last-species-must-be-has-acc-gated`). Keep the Python mirror in nb15 (`model_plateau`) in sync.

## Validation (Phase 3, decisive)

New notebook cells (nb15 extension or nb16), after the user rebuilds:

1. **Reference:** converged exact at `f = 0.3` — per `memory: accdm-fluid-f-boundary` this needs the fine daughter q-grid (q_size ~5001; use `rkck`, not `ndf15` — `memory: ndf15-oom-high-q`) or the nb14 q(f) schedule. One-off cost.
2. **Test matrix:** fluid mode-2 vs exact, `eta in {0.1, 1.0}`, `f = 0.3`, baseline `(kappa=6, a_t=0.13)`; P(k) residual restricted to **k <= 1 Mpc^-1**; record wall-time for both.
3. **Trigger scan:** the plateau closure is stable by construction, so try earlier fluid switch-on (`ncdm_fluid_trigger_rho_accDM_over_rho_dcdm` below the old 0.4 wall) — this is where a speedup beyond the old ~1.5x (`memory: fluid-approx-marginal-speedup`) would come from.
4. **Success criteria:** `max|P_fluid/P_exact - 1| <~ 1%` over k <= 1 at f = 0.3, AND a wall-time win that beats or complements the safe q-size reduction (`memory: daughter-qsize-overkill`). Report both; adoption is a cost-benefit call, not automatic.
5. **Known risk:** only `ceff2` is being fixed; `cvis2` stays at the default `3*w*ca2`. nb15's `sigma/delta` panel shows structure at high x, so if P(k) misses the 1% bar with correct `ceff2`, the shear closure is the next suspect — record that explicitly rather than tuning blindly.

## Out of scope

- Scans varying `(kappa, a_t)`: not covered by this closure (production-history maturity is not a background quantity); those use exact + q(f) (notebook 14).
- The `a_t = 0.01 -> A ~ 21` outlier: possibly a measurement systematic at tiny `c_fs`; flagged, not chased here.
- Recalibration workflow: picking a new fixed `(kappa, a_t)` family means re-running nb15's cached pipeline once to re-measure `A`; document in the notebook, no tooling needed.

## Global constraints

- Edit `class_accDM`, never the pristine reference (`memory: apply-fixes-to-working-branch`). C build happens on the user's side (`memory: build-environment`); pytest and notebooks run on the user's side (`memory: agent-shell-no-python`).
- Keep modes 0/1 bit-identical; mode 2 is additive.
