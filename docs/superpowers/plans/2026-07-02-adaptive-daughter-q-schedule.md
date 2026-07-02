# Adaptive Daughter q(f) Schedule Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Choose the accDM daughter's momentum-bin count automatically at input time from the daughter fraction f, so MCMC chains run coarse (fast) grids at low f and fine (accurate) grids at high f, with explicit user input always winning.

**Architecture:** A schedule block in `input.c` (species section) that overwrites the daughter's entry of `pba->ncdm_input_q_size` from a piecewise-in-f table, configurable via two new input lists; a Python smoke-test script and a calibration/regression notebook that validate against the built `classy`. (An ndf15 guard was DEFERRED per user decision 2026-07-02 — rkck is the working production evolver; ndf15 is not part of the supported configuration.)

**Tech Stack:** C (CLASS fork `class_accDM`), Python/`classy` + numpy for tests, Jupyter notebook for calibration.

**Spec:** `docs/superpowers/specs/2026-07-02-adaptive-daughter-q-schedule-design.md`

## Global Constraints

- Edit `class_accDM` only — never the pristine `axion_project/class_public` reference.
- **No compiler in the agent shell.** All C code is verified by the USER building (`make classy` or equivalent on their side) and then running the Python tests. Tasks below mark these steps `USER-BUILD`.
- The daughter is always the **last** ncdm species; every accDM special case must be `has_acc`-gated (a plain-ncdm run must be bit-for-bit unaffected).
- Schedule table values (breakpoints 0.1/0.3, bin counts 250/1000/2000) are **PROVISIONAL** — the user's convergence tests and the Task 4 calibration notebook set the final values. Keep them in exactly one place in the C code and mark them `PROVISIONAL` in comments.
- Explicit `ncdm_N_momentum_bins` (or deprecated `Number of momentum bins`) input must reproduce today's behavior exactly.
- Descriptive variable names (spell things out; no `hi`/`lo`-style abbreviations).
- Commit messages end with `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`.

---

### Task 1: Smoke-test script (written first; runnable only after the Task 3 build)

**Files:**
- Create: `notebooks_test/test_q_schedule_smoke.py`

**Interfaces:**
- Produces: `capture_class_stdout()` context manager and `base_params(f_acc)` dict builder, reused verbatim by the Task 4 notebook.
- Consumes (from Task 2, once built): the stdout lines `accDM q(f) schedule: ... q_size = N ...` and `accDM q(f) schedule bypassed ...` printed at `input_verbose >= 1`, and the input parameters `accdm_q_schedule` (flag), `accdm_q_schedule_f_edges` (double list), `accdm_q_schedule_q_sizes` (int list).

- [ ] **Step 1: Write the script**

The base model is the notebook-10 fiducial (m = 1e16 GeV, eta = 0.1, kappa = 4.0). The script talks to `classy` and captures the C-level stdout via fd redirection (works on Windows; `wurlitzer` is POSIX-only).

```python
"""Smoke tests for the accDM daughter q(f) schedule (input.c).
Run after building classy:  python notebooks_test/test_q_schedule_smoke.py
"""
import os
import sys
import tempfile
from contextlib import contextmanager

import numpy as np
from classy import Class


@contextmanager
def capture_class_stdout():
    """Capture C-level stdout (printf from CLASS) by redirecting fd 1 to a temp file."""
    sys.stdout.flush()
    saved_stdout_fd = os.dup(1)
    with tempfile.TemporaryFile(mode="w+b") as capture_file:
        os.dup2(capture_file.fileno(), 1)
        captured = {}
        try:
            yield captured
        finally:
            sys.stdout.flush()
            os.dup2(saved_stdout_fd, 1)
            os.close(saved_stdout_fd)
            capture_file.seek(0)
            captured["text"] = capture_file.read().decode(errors="replace")


OMEGA_B = 0.022383
OMEGA_CDM0 = 0.12011
MASS = 1e16
ETA = 0.1
KAPPA = 4.0
A_T = 0.13                       # accDM trigger scale factor (nb10 fiducial)
A_REC = 1.0 / (1.0 + 1090.0)     # recombination scale factor


def base_params(f_acc, output=""):
    """Notebook-10 fiducial accDM model; no ncdm_N_momentum_bins (schedule decides)."""
    omega_cdm = OMEGA_CDM0 * (1 + f_acc * (1 - A_REC**KAPPA) / (1 + (A_REC / A_T)**KAPPA))**(-1)
    return {
        "omega_b": OMEGA_B, "omega_cdm": omega_cdm, "H0": 67.32,
        "A_s": 2.1005829616811546e-9, "n_s": 0.96605, "tau_reio": 0.0543,
        "output": output, "evolver": 0, "gauge": "synchronous",
        "background_Nloga": 5000,
        "vary_Gamma_acc": "yes", "kappa_acc": KAPPA, "a_t_acc": A_T,
        "f_acc": f_acc, "eta_acc": ETA,
        "m_acc_in_GeV": MASS, "m_cdm_in_GeV": MASS,
        "N_ncdm": 2, "deg_ncdm": "3, 1",
        "m_ncdm": "0.02, {:.6e}".format(MASS * 1e9),
        "T_ncdm": "0.71611, 1", "ncdm_quadrature_strategy": "0, 4",
        "N_ur": 0.00441, "ncdm_fluid_approximation": 3,
        "input_verbose": 1,
    }


def run_and_capture(params):
    cosmo = Class()
    cosmo.set(params)
    with capture_class_stdout() as captured:
        cosmo.compute()
    cosmo.struct_cleanup()
    cosmo.empty()
    return captured["text"]


def expect_in(text, needle, label):
    assert needle in text, "{}: expected '{}' in CLASS stdout, got:\n{}".format(label, needle, text)


def test_low_f_gets_coarse_grid():
    text = run_and_capture(base_params(f_acc=0.01))
    expect_in(text, "q_size = 250", "low-f schedule")


def test_mid_f_gets_production_grid():
    text = run_and_capture(base_params(f_acc=0.2))
    expect_in(text, "q_size = 1000", "mid-f schedule")


def test_high_f_gets_fine_grid():
    text = run_and_capture(base_params(f_acc=0.5))
    expect_in(text, "q_size = 2000", "high-f schedule")


def test_explicit_bins_bypass_schedule():
    params = base_params(f_acc=0.01)
    params["ncdm_N_momentum_bins"] = "15, 777"
    text = run_and_capture(params)
    expect_in(text, "schedule bypassed", "explicit-bins bypass")


def test_custom_schedule_table():
    params = base_params(f_acc=0.01)
    params["accdm_q_schedule_f_edges"] = "0.05"
    params["accdm_q_schedule_q_sizes"] = "100, 900"
    text = run_and_capture(params)
    expect_in(text, "q_size = 100", "custom table")


def test_schedule_can_be_disabled():
    params = base_params(f_acc=0.01)
    params["accdm_q_schedule"] = "no"
    text = run_and_capture(params)
    assert "accDM q(f) schedule:" not in text, \
        "schedule-off: no schedule line expected, got:\n" + text


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    failed = 0
    for test in tests:
        try:
            test()
            print("PASS", test.__name__)
        except AssertionError as error:
            failed += 1
            print("FAIL", test.__name__, "--", error)
    sys.exit(1 if failed else 0)
```

- [ ] **Step 2: Sanity-run against the CURRENT build to confirm the tests fail for the right reason**

Run: `python notebooks_test/test_q_schedule_smoke.py`
Expected: `FAIL test_low_f_gets_coarse_grid` (and the other schedule tests) with "expected 'q_size = 250' in CLASS stdout" — the schedule does not exist yet. If `classy` itself fails to import or the base model errors, fix the base params first (they must match notebook 10).

- [ ] **Step 3: Commit**

```powershell
git add notebooks_test/test_q_schedule_smoke.py
git commit -m @'
test: smoke tests for accDM daughter q(f) schedule (red)

Written ahead of the input.c implementation; asserts schedule stdout
lines, explicit-bins bypass, custom table, off switch, and ndf15 guard.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>
'@
```

---

### Task 2: q(f) schedule in `input.c`

**Files:**
- Modify: `source/input.c:2866-2874` (momentum-bins read — capture the "explicitly provided" flag) and insert the schedule block immediately after it, before the `background_ncdm_init` call at `source/input.c:2877`.

**Interfaces:**
- Consumes: `pba->f_acc`, `pba->Omega_ini_dcdm`, `pba->Omega0_cdm`, `pba->Omega0_acc_cdm`, `pba->has_acc` (all already set earlier in this function, `source/input.c:2576-2729`), `pba->ncdm_input_q_size` (allocated by the reads below), local `N_ncdm`, `input_verbose`, `pfc`, `errmsg`.
- Produces: new input parameters `accdm_q_schedule` (yes/no, default yes), `accdm_q_schedule_f_edges` (double list), `accdm_q_schedule_q_sizes` (int list); stdout lines `accDM q(f) schedule: daughter fraction f = <f> -> q_size = <N> ...` and `accDM q(f) schedule bypassed: momentum bins supplied explicitly ...` at `input_verbose >= 1`. Task 1's tests and Task 4's notebook match on these exact substrings.

- [ ] **Step 1: Replace the momentum-bins read so an explicit user list is detectable**

The current code (`source/input.c:2866-2874`):

```c
    /** 5.h.2) Number of momentum bins */
    class_call(parser_read_list_of_integers(pfc, "Number of momentum bins", &entries_read, &(pba->ncdm_input_q_size), &flag1, errmsg),
               errmsg, errmsg); //Deprecated parameter, still read to keep compatibility
    if (flag1 == _TRUE_) {
      class_test(entries_read != N_ncdm, errmsg, "Number of entries in Number of momentum bins, %d, is different from the number of N_cdm species, %d", entries_read, N_ncdm);
    }
    else {
      class_read_list_of_integers_or_default("ncdm_N_momentum_bins", pba->ncdm_input_q_size, 150, N_ncdm);
    }
```

becomes (the `class_read_list_of_integers_or_default` macro hides its found-flag, so inline its logic):

```c
    /** 5.h.2) Number of momentum bins */
    int momentum_bins_provided_explicitly = _FALSE_;
    class_call(parser_read_list_of_integers(pfc, "Number of momentum bins", &entries_read, &(pba->ncdm_input_q_size), &flag1, errmsg),
               errmsg, errmsg); //Deprecated parameter, still read to keep compatibility
    if (flag1 == _TRUE_) {
      class_test(entries_read != N_ncdm, errmsg, "Number of entries in Number of momentum bins, %d, is different from the number of N_cdm species, %d", entries_read, N_ncdm);
      momentum_bins_provided_explicitly = _TRUE_;
    }
    else {
      int flag_momentum_bins, entries_momentum_bins;
      class_call(parser_read_list_of_integers(pfc, "ncdm_N_momentum_bins", &entries_momentum_bins, &(pba->ncdm_input_q_size), &flag_momentum_bins, errmsg),
                 errmsg, errmsg);
      if (flag_momentum_bins == _TRUE_) {
        class_test(entries_momentum_bins != N_ncdm, errmsg,
                   "Number of entries of 'ncdm_N_momentum_bins' (%d) does not match expected number (%d).",
                   entries_momentum_bins, N_ncdm);
        momentum_bins_provided_explicitly = _TRUE_;
      }
      else {
        class_alloc(pba->ncdm_input_q_size, N_ncdm*sizeof(int), errmsg);
        for (n=0; n < N_ncdm; n++) pba->ncdm_input_q_size[n] = 150;
      }
    }
```

(`n`, `flag1`, `entries_read` are existing locals in this function.)

- [ ] **Step 2: Insert the schedule block between the code above and the `background_ncdm_init` call (`source/input.c:2877`)**

```c
    /* accDM daughter q(f) schedule: pick the daughter's momentum-bin count from
       the daughter fraction f, so MCMC chains run coarse grids where the
       posterior lives (f < ~0.1) and fine grids at rare high-f excursions.
       Explicit user-provided momentum bins always win. Table values are
       PROVISIONAL pending calibration
       (docs/superpowers/specs/2026-07-02-adaptive-daughter-q-schedule-design.md). */
    if (pba->has_acc == _TRUE_) {

      int accdm_q_schedule = _TRUE_;
      class_read_flag("accdm_q_schedule", accdm_q_schedule);

      if (momentum_bins_provided_explicitly == _TRUE_) {
        if ((accdm_q_schedule == _TRUE_) && (input_verbose > 0))
          printf("accDM q(f) schedule bypassed: momentum bins supplied explicitly (daughter q_size = %d).\n",
                 pba->ncdm_input_q_size[N_ncdm-1]);
      }
      else if (accdm_q_schedule == _TRUE_) {

        /* PROVISIONAL defaults: f < 0.1 -> 250 bins (coarse regime validated in
           notebooks 10/12), 0.1 <= f < 0.3 -> 1000 (current production),
           f >= 0.3 -> 2000 (fixes measured >1% under-resolution at 1001 bins).
           Override without rebuilding via the two input lists below. */
        double schedule_f_edges_default[2] = {0.1, 0.3};
        int schedule_q_sizes_default[3] = {250, 1000, 2000};

        double * schedule_f_edges = NULL;
        int * schedule_q_sizes = NULL;
        int number_of_edges = 2, number_of_sizes = 3;
        int flag_edges, flag_sizes, index_edge;

        class_call(parser_read_list_of_doubles(pfc, "accdm_q_schedule_f_edges",
                                               &number_of_edges, &schedule_f_edges, &flag_edges, errmsg),
                   errmsg, errmsg);
        class_call(parser_read_list_of_integers(pfc, "accdm_q_schedule_q_sizes",
                                                &number_of_sizes, &schedule_q_sizes, &flag_sizes, errmsg),
                   errmsg, errmsg);
        class_test(flag_edges != flag_sizes, errmsg,
                   "Provide both 'accdm_q_schedule_f_edges' and 'accdm_q_schedule_q_sizes', or neither.");
        if (flag_edges == _FALSE_) {
          number_of_edges = 2;
          number_of_sizes = 3;
          class_alloc(schedule_f_edges, number_of_edges*sizeof(double), errmsg);
          class_alloc(schedule_q_sizes, number_of_sizes*sizeof(int), errmsg);
          for (index_edge=0; index_edge < number_of_edges; index_edge++)
            schedule_f_edges[index_edge] = schedule_f_edges_default[index_edge];
          for (index_edge=0; index_edge < number_of_sizes; index_edge++)
            schedule_q_sizes[index_edge] = schedule_q_sizes_default[index_edge];
        }
        class_test(number_of_sizes != number_of_edges+1, errmsg,
                   "'accdm_q_schedule_q_sizes' must have exactly one more entry (has %d) than 'accdm_q_schedule_f_edges' (has %d).",
                   number_of_sizes, number_of_edges);
        for (index_edge=0; index_edge < number_of_edges; index_edge++) {
          class_test((schedule_f_edges[index_edge] <= 0.) || (schedule_f_edges[index_edge] >= 1.),
                     errmsg, "'accdm_q_schedule_f_edges' entries must lie strictly inside (0,1).");
          if (index_edge > 0)
            class_test(schedule_f_edges[index_edge] <= schedule_f_edges[index_edge-1],
                       errmsg, "'accdm_q_schedule_f_edges' must be strictly increasing.");
        }
        for (index_edge=0; index_edge < number_of_sizes; index_edge++)
          class_test(schedule_q_sizes[index_edge] < 2, errmsg,
                     "'accdm_q_schedule_q_sizes' entries must be >= 2.");

        /* daughter fraction driving the schedule; -1 means undeterminable */
        double daughter_fraction_for_schedule = -1.;
        if (pba->f_acc > 0.)
          daughter_fraction_for_schedule = pba->f_acc;
        else if ((pba->Omega_ini_dcdm > 0.) && (pba->Omega0_cdm > 0.))
          daughter_fraction_for_schedule = pba->Omega_ini_dcdm / pba->Omega0_cdm;
        else if (pba->Omega0_acc_cdm > 0.)
          daughter_fraction_for_schedule = pba->Omega0_acc_cdm / (pba->Omega0_cdm + pba->Omega0_acc_cdm);

        /* undeterminable fraction -> most conservative (finest) grid, never fast-wrong */
        int daughter_q_size_scheduled = schedule_q_sizes[number_of_sizes-1];
        if (daughter_fraction_for_schedule >= 0.) {
          for (index_edge=0; index_edge < number_of_edges; index_edge++) {
            if (daughter_fraction_for_schedule < schedule_f_edges[index_edge]) {
              daughter_q_size_scheduled = schedule_q_sizes[index_edge];
              break;
            }
          }
        }
        pba->ncdm_input_q_size[N_ncdm-1] = daughter_q_size_scheduled;

        if (input_verbose > 0)
          printf("accDM q(f) schedule: daughter fraction f = %g -> q_size = %d for ncdm species %d.\n",
                 daughter_fraction_for_schedule, daughter_q_size_scheduled, N_ncdm-1);

        free(schedule_f_edges);
        free(schedule_q_sizes);
      }
    }
```

Notes for the implementer:
- This lives inside the `if (N_ncdm > 0)` species block, so `N_ncdm >= 1` is guaranteed; `has_acc` gating keeps plain runs untouched.
- Boundary semantics are strict: f exactly equal to an edge falls in the *higher* regime (conservative).
- During parameter shooting (`Omega_acc_cdm` given), this code re-runs each iteration with trial values; harmless because each iteration is self-consistent, and the MCMC path (`vary_Gamma_acc` + `f_acc`) does not shoot.

- [ ] **Step 3: Visually re-read the diff (`git diff source/input.c`) — no compiler here, so check: braces balance, every `class_call`/`class_test` has `errmsg`, `free` on both lists in the schedule path only (the defaults are `class_alloc`ed so they are freeable), no change outside the two blocks**

- [ ] **Step 4: Commit**

```powershell
git add source/input.c
git commit -m @'
feat: accDM daughter q(f) schedule in input.c

Daughter momentum-bin count picked at input time from the daughter
fraction (f_acc, else Omega_ini_dcdm/Omega0_cdm, else acc/(cdm+acc));
PROVISIONAL table 250/1000/2000 at edges 0.1/0.3, overridable via
accdm_q_schedule_f_edges / accdm_q_schedule_q_sizes; explicit
ncdm_N_momentum_bins always bypasses; accdm_q_schedule=no disables.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>
'@
```

---

### Task 3: USER-BUILD + green smoke tests

**Files:**
- Modify (only if fixes needed): `source/input.c`, `notebooks_test/test_q_schedule_smoke.py`

**Interfaces:**
- Consumes: Tasks 1–2 outputs.
- Produces: a built `classy` with the schedule active and all 6 smoke tests passing — the gate for Task 4.

- [ ] **Step 1 (USER-BUILD): user rebuilds CLASS/classy from the branch** (no compiler in the agent shell; report any compile errors back verbatim)

- [ ] **Step 2: Run the smoke tests**

Run: `python notebooks_test/test_q_schedule_smoke.py`
Expected: `PASS` for all 6 tests, exit code 0.

- [ ] **Step 3: If any test fails, fix the C code (or a wrong test expectation), re-run Steps 1-2 until green**

- [ ] **Step 4: Commit any fixes**

```powershell
git add source/input.c notebooks_test/test_q_schedule_smoke.py
git commit -m @'
fix: green q(f) schedule smoke tests after first build

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>
'@
```

---

### Task 4: Calibration & regression notebook (`14_test_q_schedule_calibration.ipynb`)

**Files:**
- Create: `notebooks_test/14_test_q_schedule_calibration.ipynb`

**Interfaces:**
- Consumes: `capture_class_stdout` and `base_params` from `notebooks_test/test_q_schedule_smoke.py` (import them — do not copy), the built `classy`.
- Produces: the measured convergence table that finalizes the PROVISIONAL schedule defaults in `source/input.c` and the spec; regression asserts (`max|ΔP/P| <= 1e-3`, `max|ΔC_l/C_l| <= 1e-3`) that become the permanent gate.

- [ ] **Step 1: Create the notebook with the following cells**

**Cell 1 (markdown):**

```markdown
# 14 — accDM daughter q(f) schedule: calibration & regression

Validates the input-time q(f) schedule (spec: docs/superpowers/specs/2026-07-02-adaptive-daughter-q-schedule-design.md).
For each (m, eta, f) grid point: run the **scheduled** configuration (no explicit bins;
input.c picks q_size) against a fine-grid **reference** (5001 bins) and require
- max|ΔP/P| ≤ 1e-3 over the full PyBird k-range, reported separately in the EFT window k = 0.1–0.3 h/Mpc,
- max|ΔC_l/C_l| ≤ 1e-3 for **lensed** TT/EE and φφ (CMB is in the likelihood),
- wall-times per regime.

The schedule table (edges 0.1/0.3 → sizes 250/1000/2000) is PROVISIONAL; the final cell
prints the measured verdict per point — update `source/input.c` defaults and the spec from it.
**Reference runs at 5001 bins are slow (tens of minutes each): set `FAST = True` to subsample the grid.**
```

**Cell 2 (code) — imports and plot style:**

```python
import sys, time
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from classy import Class

sys.path.insert(0, ".")
from test_q_schedule_smoke import capture_class_stdout, base_params

mpl.rcParams.update({
    "font.family": "serif", "mathtext.fontset": "stix", "font.serif": ["STIXGeneral"],
    "axes.prop_cycle": mpl.cycler(color=[
        "#1b9e77", "#d95f02", "#7570b3", "#e7298a", "#66a61e", "#e6ab02"]),
})
```

**Cell 3 (code) — grid and run helpers:**

```python
FAST = False
TOL_PK = 1e-3
TOL_CL = 1e-3
LMAX = 2500
K_NODES = np.logspace(-3, 1, 60)          # 1/Mpc, full PyBird input range
H_LITTLE = 0.6732
EFT_WINDOW = (0.1 * H_LITTLE, 0.3 * H_LITTLE)  # 1/Mpc

# (m [GeV], eta, f) — spans all three PROVISIONAL schedule regimes; edit freely.
GRID = [
    (1e16, 0.1, 0.01),
    (1e16, 0.1, 0.05),
    (1e16, 0.1, 0.15),
    (1e16, 0.1, 0.30),
    (1e16, 0.1, 0.50),
]
if FAST:
    GRID = [GRID[0], GRID[2], GRID[4]]

REFERENCE_BINS = 5001


def observables_params(f_acc, mass, eta):
    p = base_params(f_acc, output="tCl,pCl,lCl,mPk")
    p.update({"lensing": "yes", "l_max_scalars": LMAX, "P_k_max_1/Mpc": 10.0,
              "z_max_pk": 0.0, "reionization_z_start_max": 80,
              "m_acc_in_GeV": mass, "m_cdm_in_GeV": mass,
              "m_ncdm": "0.02, {:.6e}".format(mass * 1e9),
              "eta_acc": eta})
    return p


def run_point(params):
    cosmo = Class()
    cosmo.set(params)
    start = time.perf_counter()
    with capture_class_stdout() as captured:
        cosmo.compute()
    wall = time.perf_counter() - start
    pk = np.array([cosmo.pk(float(k), 0.0) for k in K_NODES])
    lensed = cosmo.lensed_cl(LMAX)
    result = dict(wall=wall, pk=pk, stdout=captured["text"],
                  ell=np.asarray(lensed["ell"]),
                  tt=np.asarray(lensed["tt"]), ee=np.asarray(lensed["ee"]),
                  pp=np.asarray(lensed["pp"]))
    cosmo.struct_cleanup(); cosmo.empty()
    return result


def max_rel_dev(candidate, reference, mask=None):
    c, r = np.asarray(candidate, float), np.asarray(reference, float)
    if mask is not None:
        c, r = c[mask], r[mask]
    scale = np.where(np.abs(r) > 0, np.abs(r), 1.0)
    return float(np.max(np.abs(c - r) / scale))
```

**Cell 4 (code) — the scan:**

```python
RESULTS = []
for mass, eta, f_acc in GRID:
    scheduled_params = observables_params(f_acc, mass, eta)          # no bins key -> schedule
    reference_params = dict(scheduled_params,
                            ncdm_N_momentum_bins="15, {:d}".format(REFERENCE_BINS))
    scheduled = run_point(scheduled_params)
    reference = run_point(reference_params)

    scheduled_q_size = None
    for line in scheduled["stdout"].splitlines():
        if "accDM q(f) schedule:" in line:
            scheduled_q_size = int(line.split("q_size = ")[1].split()[0].rstrip("."))
    ell_mask = scheduled["ell"] >= 2
    eft_mask = (K_NODES >= EFT_WINDOW[0]) & (K_NODES <= EFT_WINDOW[1])
    RESULTS.append(dict(
        mass=mass, eta=eta, f=f_acc, q_size=scheduled_q_size,
        wall_scheduled=scheduled["wall"], wall_reference=reference["wall"],
        dpk_full=max_rel_dev(scheduled["pk"], reference["pk"]),
        dpk_eft=max_rel_dev(scheduled["pk"], reference["pk"], eft_mask),
        dcl_tt=max_rel_dev(scheduled["tt"], reference["tt"], ell_mask),
        dcl_ee=max_rel_dev(scheduled["ee"], reference["ee"], ell_mask),
        dcl_pp=max_rel_dev(scheduled["pp"], reference["pp"], ell_mask),
    ))
    r = RESULTS[-1]
    print("f={f:.2f} q={q_size} | dPk full={dpk_full:.2e} eft={dpk_eft:.2e} | "
          "dCl TT={dcl_tt:.2e} EE={dcl_ee:.2e} PP={dcl_pp:.2e} | "
          "t={wall_scheduled:.0f}s vs ref {wall_reference:.0f}s".format(**r))
```

**Cell 5 (code) — results table, speedup, and regression asserts:**

```python
print("{:>6} {:>7} {:>10} {:>10} {:>10} {:>9}".format(
    "f", "q_size", "dPk_eft", "dCl_TT", "dCl_PP", "speedup"))
for r in RESULTS:
    print("{f:6.2f} {q_size:7d} {dpk_eft:10.2e} {dcl_tt:10.2e} {dcl_pp:10.2e} "
          "{speedup:9.1f}x".format(speedup=r["wall_reference"]/r["wall_scheduled"], **r))

for r in RESULTS:
    label = "(m={mass:.0e}, eta={eta}, f={f})".format(**r)
    assert r["q_size"] is not None, "schedule line not found in stdout " + label
    assert r["dpk_full"] <= TOL_PK, "P(k) tolerance exceeded {}: {:.2e}".format(label, r["dpk_full"])
    for key in ("dcl_tt", "dcl_ee", "dcl_pp"):
        assert r[key] <= TOL_CL, "C_l ({}) tolerance exceeded {}: {:.2e}".format(key, label, r[key])
print("ALL REGRESSION GATES PASSED")
```

**Cell 6 (code) — residual plots:**

```python
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for r in RESULTS:
    axes[0].axhline(TOL_PK, color="k", lw=0.5, ls="--")
    axes[0].scatter([r["f"]], [r["dpk_eft"]], label="f={f}".format(**r))
    axes[1].scatter([r["f"]], [r["dcl_tt"]])
axes[0].set(xlabel="$f$", ylabel=r"max$|\Delta P/P|$ (EFT window)", yscale="log")
axes[1].axhline(TOL_CL, color="k", lw=0.5, ls="--")
axes[1].set(xlabel="$f$", ylabel=r"max$|\Delta C_\ell^{TT}/C_\ell^{TT}|$ (lensed)", yscale="log")
fig.tight_layout()
```

**Cell 7 (markdown):**

```markdown
## Finalizing the table

If any point fails its gate: raise that regime's `q_size` (or add an edge) via
`accdm_q_schedule_f_edges` / `accdm_q_schedule_q_sizes`, re-run this notebook, and once
green copy the working table into the `schedule_f_edges_default` / `schedule_q_sizes_default`
arrays in `source/input.c` and update the spec's provisional table. If a coarse regime passes
with large margin, try lowering it — the margin is wall-time on every MCMC point.
```

- [ ] **Step 2 (USER-BUILD/RUN): user runs the notebook against the built classy** (hours at full grid; `FAST = True` for a first pass)

- [ ] **Step 3: From the measured results, update `schedule_f_edges_default` / `schedule_q_sizes_default` in `source/input.c` and the spec's provisional table (remove the PROVISIONAL marker only when the user declares calibration done)**

- [ ] **Step 4: Commit**

```powershell
git add notebooks_test/14_test_q_schedule_calibration.ipynb source/input.c docs/superpowers/specs/2026-07-02-adaptive-daughter-q-schedule-design.md
git commit -m @'
nb14: q(f) schedule calibration + regression gate; pin measured table

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>
'@
```
