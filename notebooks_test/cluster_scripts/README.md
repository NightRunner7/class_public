# Cluster cache generators

`gen_nb21_cache.py` (notebook 21, daughter `accdm_l_max`) and `gen_nb22_cache.py` (notebook 22, κ and
a_t) are the live generators. The nb15-nb20 generators of the fluid era moved to
`notebooks_test/_archive/fluid_closure/`; `common.py` stays here for them and for
`emulator_chi2_validation.py`. The setup and submission notes below apply to any generator run through
`submit_template.pbs`.

## nb21: daughter hierarchy length

`gen_nb21_cache.py` defines the grid, and notebook 21 imports it, so the two cannot drift. Each array
task runs one CLASS job (257 runs, then 12 CPU-timing jobs) and writes
`accDM_scans/nb21_cache/<tag>.pkl` as `{RunCache key: result}` (timings in `nb21_cache/cpu/`). The
notebook reads them with `RunCache(..., also=...)` and stops if any run is missing.

```
cd $HOME/Workplace/class_accDM_refactor/notebooks_test
python cluster_scripts/gen_nb21_cache.py --list | tail -1        # -> 269 jobs
qsub -t 0-268 -v 'GENERATOR=cluster_scripts/gen_nb21_cache.py' cluster_scripts/submit_template.pbs
# after it finishes, on the laptop:
rsync -av kruk:Workplace/class_accDM_refactor/notebooks_test/accDM_scans/nb21_cache/ notebooks_test/accDM_scans/nb21_cache/
```

The env's `classy` must be built from `cbeb3e86` or later (`accdm_l_max`, flat-space ℓ sampling).
Most tasks take seconds to minutes; the 400 bins/decade × `accdm_l_max` 140 references at 10¹¹ GeV
take about 40 CPU-minutes, so up to an hour on the 4 threads of one task. Resubmitting the same
array after a walltime kill only redoes the missing files. The timing jobs measure CPU time on the
node they land on, two runs each.

## nb22: dependence on κ and a_t

`gen_nb22_cache.py` defines the grid (6 masses × 6 κ × 7 a_t) and the f̃₄ search, and notebook 22
imports both. Each search task runs one (m, κ, a_t) point's whole search serially (its own null, then
3–6 runs until Δχ² = 4 is bracketed); at the corners of the box it adds the 401-bin grid check at f̃₄.
The notebook replays every search from `accDM_scans/nb22_cache/` and stops if a run is missing.

```
cd $HOME/Workplace/class_accDM_refactor/notebooks_test
python cluster_scripts/gen_nb22_cache.py --list | tail -1        # -> 373 jobs, the first 265 without ladder
qsub -t 0-264 -l mem=2gb,walltime=06:00:00 -N nb22-cache -v 'GENERATOR=cluster_scripts/gen_nb22_cache.py' \
     cluster_scripts/submit_template.pbs
# section 5, bins x accdm_l_max up to (801, 50):
qsub -t 0-107 -l mem=4gb,walltime=12:00:00 -N nb22-ladder \
     -v 'GENERATOR=cluster_scripts/gen_nb22_cache.py,GEN_ARGS=--only ladder_' cluster_scripts/submit_template.pbs
rsync -av kruk:Workplace/class_accDM_refactor/notebooks_test/accDM_scans/nb22_cache/ notebooks_test/accDM_scans/nb22_cache/
```

Needs `CHAIN_SETTINGS_OCT07` in `accdm_nb.py` and `classy` from `209be92a` or later. Tasks take a few
minutes; the earliest transitions (a_t = 10⁻³, κ = 1) and the 401-bin checks are the slow ones.

## Old generators (nb15-nb20, archived)

Each `gen_nbNN_cache.py` reproduces, run for run, the CLASS calls its notebook
makes and writes pickles with the **exact filenames** the notebook's cache
lookup uses. Generate on the cluster, copy the `*.pkl` into
`notebooks_test/accDM_scans/nbNN_cache/`, and the notebook re-run becomes pure
cache hits (nb13/nb14 only cache in memory, so they have no generator).

| script | notebook | cache dir | payload | jobs (defaults) |
|---|---|---|---|---|
| `gen_nb15_cache.py` | 15 fluid-closure diagnostic | `nb15_cache/` | `{k: tau-series dict}` | 30 grid + 4 sweep |
| `gen_nb16_cache.py` | 16 plateau fluid, f=0.3 | `nb16_cache/` | `(pk, seconds)` | 2 ref + 8 fluid + 2 attribution |
| `gen_nb17_cache.py` | 17 mode-3 eta scan, f=0.1 | `nb17_cache/` | `(pk, seconds)` | 5 ref + 10 fluid |
| `gen_nb18_cache.py` | 18 ceff2(f, eta) formula | `nb18_cache/` | series / `(pk, seconds)` | Stage A: 20, Stage C: 10 |
| `gen_nb19_cache.py` | 19 fluid error vs signal | `nb19_cache/` | `{'pk','sigma8','seconds'}` | 1 LCDM + 5 exact + 5 fluid |
| `gen_nb20_cache.py` | 20 q-bins vs f accuracy/speed | `nb20_cache/` | `{'pk','sigma8','seconds'}` | 54 truth + 432 exact + 432 fluid |

## Cluster setup

Copy the whole `notebooks_test/` folder (or at least `cluster_scripts/` +
`fluid_closure_helpers.py`, keeping that relative layout — nb18/nb19 import
the helpers from the parent directory) into `$HOME/Workplace/class_accDM_refactor/` on
the cluster. The `accDM` conda env must contain the **rebuilt** `classy` from
this branch (modes 2/3 wired in — the env used for `param_scan.sh` needs a
`classy` rebuild if it predates the mode-2/3 work).

Always submit from `notebooks_test/` and let `--cache-root` default to
`accDM_scans`: the pickles then land in `notebooks_test/accDM_scans/nbNN_cache/`,
the identical layout to the laptop, so copying back is a single rsync of that
tree.

## Running

Common CLI on every generator:

```
python cluster_scripts/gen_nb19_cache.py --list        # enumerate jobs + indices, HAVE marks existing files
python cluster_scripts/gen_nb19_cache.py               # run everything sequentially (skips existing files)
python cluster_scripts/gen_nb19_cache.py --index 3     # run one job (array mode)
python cluster_scripts/gen_nb19_cache.py --only fluid  # substring filter on tags (applied before --index)
python cluster_scripts/gen_nb19_cache.py --force ...   # recompute despite existing files
```

Writes are atomic (tmp file + rename), so array tasks can share one cache dir
and a resubmit after walltime kills only redoes missing files.
`submit_template.pbs` follows the `param_scan.sh` conventions (Torque/Maui,
`accDM` conda env, thread pinning); the generator and its settings go in via
`qsub -v`, and the **0-based** array ID maps to `--index` (unlike
`param_scan.sh`, where the ID is a seed and starts at 1):

```
cd $HOME/Workplace/class_accDM_refactor/notebooks_test
python cluster_scripts/gen_nb18_cache.py --stage A --list     # -> 20 jobs
qsub -t 0-19 -v 'GENERATOR=cluster_scripts/gen_nb18_cache.py,GEN_ARGS=--stage A' \
     cluster_scripts/submit_template.pbs
# after Stage A completes (it feeds the Stage-B fit):
qsub -t 0-9 -v 'GENERATOR=cluster_scripts/gen_nb18_cache.py,GEN_ARGS=--stage C' \
     cluster_scripts/submit_template.pbs
```

`qsub -v` separates variables with commas, so `GEN_ARGS` can hold spaces but
not commas — for nb19's `--corners f,eta` pairs, edit a copy of the template
instead of passing them through `-v`.

Heaviness guide: the q=5001 exact references (f=0.3 corners in nb16/nb18/nb19)
dominate; everything else is minutes. `evolver: 0` (rkck) is baked in — do not
switch to ndf15 at large q (memory: ndf15-oom-high-q).

nb20 is the big one (918 jobs by default): 54 exact q=10001 ground-truth runs
plus the 8-rung q-ladder in both arms per (f, eta) corner. Submit it as three
arrays via `--only` (the filter is applied before `--index`, so always `--list`
with the same `--only` first to confirm the count):

```
python cluster_scripts/gen_nb20_cache.py --only truth --list          # -> 54 jobs
qsub -t 0-53  -v 'GENERATOR=cluster_scripts/gen_nb20_cache.py,GEN_ARGS=--only truth' \
     cluster_scripts/submit_template.pbs
qsub -t 0-431 -v 'GENERATOR=cluster_scripts/gen_nb20_cache.py,GEN_ARGS=--only exact_' \
     cluster_scripts/submit_template.pbs
qsub -t 0-431 -v 'GENERATOR=cluster_scripts/gen_nb20_cache.py,GEN_ARGS=--only fluid_' \
     cluster_scripts/submit_template.pbs
```

The truth array is the expensive one — q=10001 is ~2x a q=5001 run, so bump
the copy's walltime if the 5001 references were already close to the limit.
Exact/truth files do not depend on the fluid knobs (`--trigger`, `--shear`,
`--ceff2-mode`, `--fs-amp`, all encoded in the fluid tag), so re-scanning a
different fluid configuration only adds the cheap `fluid_` array.

## Changing settings

Knobs that are **encoded in the filename** (safe — the notebook picks up the
matching file automatically once its setup cell uses the same value):

- nb16/nb17: `--triggers`, `--etas`
- nb18: `--trigger`, `--cache-tag`
- nb19: `--trigger`, `--shear`, `--cache-tag`
- nb15: `--etas`, `--fs`, `--cache-tag`

Knobs that are **not in the filename** — if you change them you must set the
same value in the notebook's setup cell *and* regenerate/clear stale files
(same name, different contents is a silent poison):

- nb17 `--a-eta` (ncdm_ceff2_eta_A)
- nb19 `--a0-eta` / `--b-f` (must match the notebook's `A0_ETA, B_F`)
- nb18 `--a0`/`--b` if pinned by hand (normally derived from Stage A exactly
  like the notebook's Stage-B fit, adoption rule included, so the fluid tag's
  `_A{...}` matches what the notebook computes)

## Copying back

```
rsync -av kruk:Workplace/class_accDM_refactor/notebooks_test/accDM_scans/ notebooks_test/accDM_scans/
```

Then run the notebook and check every run line prints `[cache hit]` /
`[cache]` — any `[computing]`/`[run]` line means a tag mismatch (or a knob
out of sync with the setup cell).

## Caveats

- **numpy pickle compatibility**: pickles written under numpy >= 2 do not load
  under numpy 1.x. Match major versions between cluster and laptop (or
  downgrade the cluster env to the laptop's numpy major).
- `gen_nb16_cache.py` preserves the notebook's known `ncdm_ceff2_mode = 1`
  typo in the fluid runs (documented in the nb17 header) so the files match
  what nb16 actually computes. nb17/nb18/nb19 are the corrected tests.
- The elapsed-time fields (`seconds`) will reflect cluster wall times; the
  notebooks' speedup tables will report cluster timings, which is usually
  what you want anyway.

## Monitoring kruk (chains, CONNECT)

`kruk_chains.py` and `kruk_connect.py` moved to `project_monopoles/scripts/`
(see the README there); run them as `kruk-chains` / `kruk-connect`.
