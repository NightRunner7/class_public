# `m_nu` input parameter — Design Spec

**Status:** approved design (2026-07-27). Input-layer convenience only; no physics change.

## Context and goal

Running accDM alongside massive neutrinos requires `N_ncdm = 2`, with the accDM daughter
occupying the **last** ncdm slot by convention. Every per-species ncdm input is therefore a
length-`N_ncdm` list, and `class_read_list_of_doubles_or_default` (`include/input.h:139`)
hard-errors when a supplied list's length does not match `N_ncdm`. Users must consequently
spell out an `m_ncdm` entry for the daughter — which `input.c:2783` then immediately
discards, overwriting it with `m_acc_in_GeV*1e9`.

The result is a placeholder repeated across the whole repo:

- `test_background.ini:44` — `m_ncdm = 0.02, 1e16  # the 2nd argument is a placeholder`
- `test_perturb.ini:43` — `m_ncdm = 0.02, 1e25  # the 2nd argument is a placeholder`
- ~9 notebooks under `notebooks_test/` — `'m_ncdm': '0.02, {:.6e}'.format(MASS*1e9)`
- `notebooks_test/golden/regression_accDM.json:24` — `"m_ncdm": "0.02, 1e+16*1e9"`

In each case the user hand-computes `m_acc_in_GeV*1e9`, a value already available to CLASS
from `m_acc_in_GeV` and already ignored on read. In MontePython this is worse still: the
wrapper must assemble the list as a formatted string at every chain point.

**Goal.** Let the user pass a scalar `m_nu` (in eV) instead of the `m_ncdm` list. Together
with the existing `m_acc_in_GeV`, that fully determines the ncdm mass vector, and no
placeholder need ever be typed or computed.

## Scope

This spec covers `m_ncdm` **only**. It is option (a) of three considered:

| option | scope | verdict |
|---|---|---|
| (a) | `m_nu` + `m_acc_in_GeV` fill `m_ncdm` | **chosen** |
| (b) | (a) + accDM slot self-supplies `deg_ncdm`, `T_ncdm`, `ncdm_N_momentum_bins` defaults | deferred |
| (c) | (b) + scalar broadcast for any per-species list | rejected (YAGNI at `N_ncdm` ≤ 2) |

Explicitly **out of scope**: `deg_ncdm`, `T_ncdm`, and `ncdm_N_momentum_bins` remain
length-`N_ncdm` lists, each with a trailing accDM entry the user must still write. Option
(a) removes the placeholder the user must *compute*, not the ones they must *type*. It is
worth stating plainly that this leaves three quarters of the ergonomic problem in place;
(b) remains available as a follow-up if the residual friction proves annoying in practice.

## Semantics

- **`m_nu` is the mass of one neutrino species, in eV.** All massive neutrino species are
  assumed to share it.
- **It is per-species, not the sum.** With the repo's usual `deg_ncdm = '3, 1'`,
  `m_nu = 0.02` means Σm_ν = 0.06 eV. This is the opposite of the MontePython convention,
  where priors are normally placed on Σm_ν; the MP wrapper must divide by the degeneracy.
- **It broadcasts to every ncdm slot**, then the accDM slot (if `has_acc`) is overwritten
  from `m_acc_in_GeV` by the pre-existing logic. It is therefore equally usable in plain
  non-accDM runs, where it is simply a scalar shorthand for a uniform `m_ncdm` list.
- **`m_nu` and `m_ncdm` are mutually exclusive.** Supplying both is a hard error.
- **`m_nu` is not stored on `pba`.** `pba->m_ncdm_in_eV` remains the single canonical store,
  so nothing downstream of the input module changes.

### Rationale for the hard error

The original proposal was for `m_nu` to silently overwrite `m_ncdm`. A hard error is used
instead, for two reasons:

1. It matches every other mutually-exclusive input pair in `input_read_parameters`:
   `Omega_ini_dcdm`/`omega_ini_dcdm`, `Gamma_acc`/`tau_acc`,
   `kappa_acc`/`log10kappa_acc`, `Omega_ncdm`/`omega_ncdm` all `class_test` rather than
   pick a winner.
2. Silent precedence fails worst in the target use case. A stale `m_ncdm` left in a
   MontePython param file would quietly override the varied `m_nu`, and the run would look
   healthy while scanning the wrong model — the same class of silent-mismatch bug already
   encountered with `ncdm_fluid_approximation` list truncation.

## Implementation

One block in `source/input.c`, section 5.d (currently lines 2774–2789).

### Change 1 — explicit read replacing the macro

`class_read_list_of_doubles_or_default` cannot distinguish "user supplied `m_ncdm`" from
"user supplied nothing", which the mutual-exclusion test requires. Replace it with a direct
`parser_read_list_of_doubles` call, reproducing the macro's length test and default fill:

```c
    /** 5.d) Mass and/or Omega of each ncdm species */
    /* Read. 'm_nu' is a convenience alternative to 'm_ncdm': it gives the mass
       in eV of a single massive neutrino species, taken to be common to all of
       them, and spares the user from spelling out a length-N_ncdm list whose
       accDM entry is a discarded placeholder anyway (see just below). */
    int flag_m_nu, flag_m_ncdm;
    double m_nu_in_eV;

    class_call(parser_read_double(pfc,"m_nu",&m_nu_in_eV,&flag_m_nu,errmsg),
               errmsg,
               errmsg);
    class_call(parser_read_list_of_doubles(pfc,"m_ncdm",&entries_read,&(pba->m_ncdm_in_eV),&flag_m_ncdm,errmsg),
               errmsg,
               errmsg);

    /* Test */
    class_test((flag_m_nu == _TRUE_) && (flag_m_ncdm == _TRUE_),
               errmsg,
               "You can only enter one of 'm_nu' or 'm_ncdm'.");

    /* Complete set of parameters */
    if (flag_m_ncdm == _TRUE_){
      class_test(entries_read != N_ncdm,
                 errmsg,
                 "Number of entries of 'm_ncdm' (%d) does not match expected number (%d).",
                 entries_read,N_ncdm);
    }
    else {
      /* Neither given: fall back to 0.0, which the loop further down turns
         into the ultra-relativistic default of 1e-5 eV. */
      if (flag_m_nu == _FALSE_) m_nu_in_eV = 0.0;
      class_alloc(pba->m_ncdm_in_eV,N_ncdm*sizeof(double),errmsg);
      for (n=0; n<N_ncdm; n++){ pba->m_ncdm_in_eV[n] = m_nu_in_eV; }
    }
```

Dedicated locals `flag_m_nu` / `flag_m_ncdm` are used rather than the shared `flag1`/`flag2`
scratch variables, because `flag_m_nu` must survive into Change 2 below and the surrounding
code reuses `flag1`/`flag2` freely.

A negative `m_nu` needs no dedicated test: it lands in `m_ncdm_in_eV[n]` and is caught by the
existing negativity loop at `input.c:2785`. The resulting message names `m_ncdm[0]` rather
than `m_nu`, which is mildly confusing but not worth a second code path — the value and the
diagnosis are both correct.

### Change 2 — guard the accDM overwrite

The existing overwrite is kept, with an added test and an expanded comment:

```c
    /* The last ncdm slot is reserved for the accDM daughter, whose mass always
       comes from 'm_acc_in_GeV'; whatever sits in m_ncdm[N_ncdm-1] (a
       placeholder, or the m_nu broadcast above) is discarded here.
       Guard on has_acc: without it, a plain (non-accDM) run would have its last
       ncdm species' mass overwritten with m_acc_in_GeV*1e9 = 0, silently
       turning e.g. the neutrino massless. */
    if (pba->N_ncdm > 0 && pba->has_acc == _TRUE_) {
      class_test((flag_m_nu == _TRUE_) && (N_ncdm < 2),
                 errmsg,
                 "You set 'm_nu' together with accDM, but 'N_ncdm = %d': the only ncdm species is the accDM daughter, which takes its mass from 'm_acc_in_GeV', so 'm_nu' would be silently discarded. Set 'N_ncdm = 2' for both massive neutrinos and accDM.",
                 N_ncdm);
      pba->m_ncdm_in_eV[pba->N_ncdm-1] = pba->m_acc_in_GeV*1e9;
    }
```

Without this guard, `N_ncdm = 1` plus accDM would accept `m_nu` and then discard it
entirely — the precise silent failure the mutual-exclusion error exists to prevent.

### Ordering constraint

`pba->has_acc` and `pba->m_acc_in_GeV` are set in the accDM block at `input.c:2613-2730`,
before `N_ncdm` is read at `input.c:2733`. Both are therefore available in section 5.d. No
reordering is needed.

### Change 3 — documentation

- `explanatory.ini`: document `m_nu` in the ncdm section (5.d), stating the per-species
  semantics, the mutual exclusion with `m_ncdm`, and the accDM last-slot convention.
- `default.ini`: add a commented `#m_nu = 0.06` line beside the existing `#m_ncdm = 0.06`.

## Behaviour when `N_ncdm = 0`

Section 5.d sits inside `if (N_ncdm > 0)`, so a `m_nu` passed with `N_ncdm = 0` is never
read. The two front ends then diverge, and both are left as they are:

- The **C binary** reports it under the opt-in `write_warnings` path (`input.c:5946`) as an
  unused input line, and otherwise ignores it.
- **classy** raises `CosmoSevereError: Class did not read input parameter(s): m_nu`, because
  the wrapper treats any unread parameter as fatal (`python/classy.pyx:357-364`).

This is how CLASS already treats every other inapplicable parameter, so no special handling
is added. Note the practical consequence for the Python front end: an `m_nu` that CLASS
cannot apply is an error rather than a silent no-op, which is the desirable direction.

## Backwards compatibility

Complete. Every existing `.ini`, notebook, and `notebooks_test/golden/regression_accDM.json`
supplies `m_ncdm` and takes the `flag_m_ncdm == _TRUE_` branch, which reproduces the macro's
behaviour exactly. The golden regression continues to pass untouched, and is the primary
guard that Change 1 is behaviour-preserving.

## Testing

1. **Golden regression unchanged** — `notebooks_test/1_test_regression_golden.ipynb` passes
   with no edits. Confirms the `m_ncdm` path is byte-for-byte equivalent after Change 1.
2. **Equivalence** — an accDM run with `m_nu = 0.02` produces P(k) and C_l identical to the
   same run with `m_ncdm = '0.02, <m_acc*1e9>'`. This is the core assertion.
3. **Plain-run equivalence** — a non-accDM run with `N_ncdm = 1, deg_ncdm = 3, m_nu = 0.02`
   matches `m_ncdm = 0.02`.
4. **Error paths** — each raises `CosmoSevereError`: both `m_nu` and `m_ncdm` given, and
   `m_nu` with accDM at `N_ncdm = 1`, both matching on their dedicated messages; negative
   `m_nu`, matching only on the generic `m_ncdm` negativity message noted above.
5. **Default preserved** — neither given, `N_ncdm = 1`: mass falls through to the 1e-5 eV
   ultra-relativistic default at `input.c:2812`.

Tests 2–5 go in a new `notebooks_test/21_test_m_nu_input.py` (plain script, not a notebook —
these are fast assertions with no plots). The user runs it; the agent environment has no
Python or compiler.

## Follow-up, not included here

- Migrating the ~9 notebooks and the `.ini` files to `m_nu`. Deliberately deferred: the
  point of full backwards compatibility is that migration can happen lazily, and touching
  the golden regression's input in the same change would undermine its value as the control.
- The MontePython wrapper change (Σm_ν → `m_nu` division by degeneracy).
- Option (b), if the residual `deg_ncdm` / `T_ncdm` / `ncdm_N_momentum_bins` placeholders
  prove annoying in practice.
