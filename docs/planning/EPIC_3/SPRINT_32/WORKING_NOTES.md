# Sprint 32 Working Notes

## Day 1

**Objective:** Convert the Sprint 31 handoff into a precise Sprint 32 inventory by rerunning the authoritative Apple Clang CMake baseline for the current branch, inspecting the named Sprint 32 test files, and recording the current `tests/test_reorder_nd.c` truthfulness state before any structural edits begin.

### Commands Run

1. Read Sprint 32 scope and Sprint 31 handoff inputs:
   - `sed -n '79,111p' docs/planning/EPIC_3/PROJECT_PLAN.md`
   - `sed -n '1,220p' docs/planning/EPIC_3/SPRINT_31/HANDOFF.md`
   - `sed -n '1,220p' docs/planning/EPIC_3/SPRINT_32/PLAN.md`
2. Inspect the leading Sprint 32 truthfulness target and the named warning files:
   - `sed -n '1,320p' tests/test_reorder_nd.c`
   - `rg -n "RUN_TEST|^static void test_|SPARSE_.*opts_t .*\\{" tests/test_reorder_nd.c tests/test_ldlt.c tests/test_colamd.c tests/test_chol_csc.c tests/test_cholesky.c tests/test_sprint12_integration.c tests/test_sprint18_integration.c tests/test_sprint19_integration.c tests/test_sprint20_integration.c tests/test_reorder.c tests/test_etree.c tests/test_svd.c tests/test_sprint6_integration.c tests/test_bidiag.c tests/test_qr.c tests/test_sprint10_integration.c tests/test_block_solvers.c tests/test_ilu.c tests/test_lu_csr.c tests/test_sprint5_integration.c`
3. Run a clean serialized Apple Clang CMake baseline capture for the current branch:
   - `rm -rf build/sprint32-day1-cmake`
   - `cmake -S . -B build/sprint32-day1-cmake`
   - `cmake --build build/sprint32-day1-cmake --parallel 1 --clean-first`
4. Derive Sprint 32 test-warning counts from the new CMake stderr artifact:
   - counts by area
   - counts by warning class
   - counts by file
   - counts by file and warning class
5. Derive the current `tests/test_reorder_nd.c` structure counts:
   - static test functions
   - active `RUN_TEST(...)` sites
   - commented-out `RUN_TEST(...)` sites

All command stdout/stderr and derived counts were written under `docs/planning/EPIC_3/SPRINT_32/artifacts/`.

### Raw Counts

- CMake configure warnings: `0`
- CMake full-build warnings: `98`
- `src` warnings: `0`
- `tests` warnings: `98`
- `benchmarks` warnings: `0`
- `examples` warnings: `0`
- warning-bearing test files: `20`

### Sprint 32 Warning Inventory

- `tests/test_ldlt.c`
  - `18` warnings
  - all `-Wmissing-field-initializers`
  - highest-volume mechanical designated-initializer cleanup target
- `tests/test_sprint20_integration.c`
  - `9` warnings
  - split between `-Wmissing-field-initializers` (`5`) and `-Wdouble-promotion` (`4`)
  - main mixed-class integration cleanup target
- `tests/test_colamd.c`
  - `8` warnings
  - mostly designated-initializer drift (`7`) plus one `-Wdouble-promotion`
- `tests/test_chol_csc.c`
  - `8` warnings
  - all `-Wmissing-field-initializers`
- `tests/test_reorder_nd.c`
  - `7` warnings
  - `4` `-Wmissing-field-initializers`
  - `3` `-Wunused-function`
  - leading truthfulness / dormant-scaffold target for the sprint
- `tests/test_svd.c`
  - `6` warnings
  - all `-Wdouble-promotion`
- `tests/test_sprint6_integration.c`
  - `6` warnings
  - all `-Wdouble-promotion`
- `tests/test_sprint18_integration.c`
  - `6` warnings
  - mixed initializer and double-promotion cleanup
- `tests/test_sprint12_integration.c`
  - `5` warnings
  - all `-Wmissing-field-initializers`
- `tests/test_cholesky.c`
  - `5` warnings
  - all `-Wmissing-field-initializers`
- `tests/test_reorder.c`
  - `4` warnings
  - all `-Wmissing-field-initializers`
- remaining named deferred queue
  - `tests/test_sprint19_integration.c`: `3`
  - `tests/test_bidiag.c`: `3`
  - `tests/test_sprint10_integration.c`: `2`
  - `tests/test_qr.c`: `2`
  - `tests/test_sprint5_integration.c`: `1`
  - `tests/test_lu_csr.c`: `1`
  - `tests/test_ilu.c`: `1`
  - `tests/test_etree.c`: `1`
  - `tests/test_block_solvers.c`: `1`

### Current `test_reorder_nd.c` Truthfulness State

- static test functions compiled in file: `26`
- active `RUN_TEST(...)` sites: `23`
- commented-out `RUN_TEST(...)` sites: `3`
- commented-out test bodies still exist as compiled static functions:
  - `test_finest_fm_annealing_pres_poisson_close_to_target`
  - `test_nd_root_spectral_pres_poisson_close_to_target`
  - `test_non_pipeline_pres_poisson_close_to_target`
- the three `-Wunused-function` warnings line up with those dormant helpers rather than with active suite coverage
- the file also still carries `4` positional options-struct initialization warnings, so its truthfulness cleanup and mechanical cleanup are coupled

### Warning Taxonomy

- `-Wmissing-field-initializers`: `62`
- `-Wdouble-promotion`: `33`
- `-Wunused-function`: `3`

Interpretation:

- designated-initializer drift is still the dominant raw warning class and is spread across the named files already called out by Sprint 30 and Sprint 31
- double-promotion cleanup is real but more concentrated in numeric and integration tests
- `-Wunused-function` is not a broad tree problem; it is localized to the dormant `test_reorder_nd.c` scaffold

### Day 1 Interpretation

- Sprint 32 starts exactly at the validated Sprint 31 end state: the entire remaining warning queue is in `tests/`, with `src`, `benchmarks`, and `examples` already at zero.
- `tests/test_reorder_nd.c` is the highest-signal structural target because it combines honesty drift in the executed protection surface with the only residual `-Wunused-function` warnings in the tree.
- The remaining warning debt is now sharply partitioned:
  - truthfulness / dormant scaffold: `tests/test_reorder_nd.c`
  - designated initializers: the majority of the named deferred files
  - mechanical double-promotion cleanup: a smaller numeric/integration subset
- No new non-test warning sites appeared in the Day 1 baseline rerun.

### Day 1 Outputs

- `artifacts/day1-test-suite-baseline.md`
- `artifacts/day1-warning-counts-by-area.txt`
- `artifacts/day1-warning-counts-by-class.txt`
- `artifacts/day1-warning-counts-by-file.txt`
- `artifacts/day1-warning-counts-by-file-and-class.txt`
- `artifacts/day1-test-reorder-nd-structure.txt`
- raw configure/build stdout/stderr logs for the clean Day 1 CMake rerun

## Day 2

**Objective:** Audit `tests/test_reorder_nd.c` precisely enough to separate active coverage from dormant scaffold, classify each commented-out test as delete vs formalize vs keep-active material, and write the structural end-state note before code edits begin.

### Commands Run

1. Re-read the Sprint 32 baseline and the current `test_reorder_nd.c` source:
   - `sed -n '1,220p' docs/planning/EPIC_3/SPRINT_32/WORKING_NOTES.md`
   - `sed -n '1,320p' tests/test_reorder_nd.c`
   - `sed -n '320,760p' tests/test_reorder_nd.c`
   - `sed -n '1460,1755p' tests/test_reorder_nd.c`
2. Enumerate active and dormant test entry points:
   - `rg -n "test_finest_fm_annealing_pres_poisson_close_to_target|test_nd_root_spectral_pres_poisson_close_to_target|test_non_pipeline_pres_poisson_close_to_target|RUN_TEST\\(" tests/test_reorder_nd.c`
   - Python one-off to list:
     - all `static void test_*` functions
     - active `RUN_TEST(...)` calls
     - commented-out `RUN_TEST(...)` calls
     - any unreferenced `static void test_*` functions
3. Inspect supporting framework conventions and nearby policy signals:
   - `sed -n '1,260p' tests/test_framework.h`
   - `rg -n "RUN_TEST\\(|skipped \\(|TF_FAIL_|return;|tf_setenv|EXPERIMENT|opt-in|slow" tests`
4. Re-read the sprint decision documents that the dormant ND stubs cite:
   - `sed -n '1,240p' docs/planning/EPIC_2/SPRINT_27/headline_summary.md`
   - `sed -n '1,120p' docs/planning/EPIC_2/SPRINT_27/thick_restart_decision.md`
   - `sed -n '1,220p' docs/planning/EPIC_2/SPRINT_28/non_pipeline_decision.md`
5. Reconfirm the local branch/commit state before writing the Day 2 note:
   - `git show --stat --oneline --no-patch HEAD`

### Audit Findings

- `tests/test_reorder_nd.c` contains `26` `static void test_*` functions.
- `23` of those functions are active through `RUN_TEST(...)`.
- `3` are intentionally dormant through commented-out `RUN_TEST(...)` lines:
  - `test_finest_fm_annealing_pres_poisson_close_to_target`
  - `test_nd_root_spectral_pres_poisson_close_to_target`
  - `test_non_pipeline_pres_poisson_close_to_target`
- There are `0` hidden or unreferenced `static void test_*` functions beyond that active-vs-commented split.
- The file’s `3` `-Wunused-function` warnings map exactly to the dormant trio.
- The file’s `4` `-Wmissing-field-initializers` warnings are separate mechanical debt and are not the reason the dormant trio exists.

### Classification of the Dormant Trio

- `test_finest_fm_annealing_pres_poisson_close_to_target`
  - status: historical failing-as-expected scaffold
  - evidence: its own comment says the test was kept commented out because Sprint 27 Day 12 measured `0.943x` vs the `0.87x` tolerance gate; Sprint 27 Day 13 then concluded the default path was already the Pres_Poisson best and annealing remained advisory-only
  - Day 2 classification: delete from active suite code or move its evidence into docs; do not formalize as opt-in coverage
- `test_nd_root_spectral_pres_poisson_close_to_target`
  - status: historical failing-as-expected scaffold
  - evidence: its own comment and Sprint 27 notes say root-spectral landed `0.944x`, still far from the `0.87x` tolerance gate, and shipped advisory-only
  - Day 2 classification: same as annealing; historical evidence, not a truthful opt-in test candidate
- `test_non_pipeline_pres_poisson_close_to_target`
  - status: retired-target scaffold
  - evidence: Sprint 28 `non_pipeline_decision.md` formally retired the literal `0.85x` Pres_Poisson target after six consecutive sprint misses and explains why the supernodal-etree post-pass cannot change fill by construction
  - Day 2 classification: strongest delete/docs-only candidate of the three because its target is explicitly retired, not merely currently unmet

### Active Coverage That Should Stay Active

- The advisory-axis smoke tests already present the truthful active suite surface:
  - `test_finest_fm_annealing_differs_from_baseline`
  - `test_nd_root_spectral_pres_poisson_smoke`
  - `test_finest_fm_thick_restart_returns_to_anchor`
  - `test_hcc_kuu_safe_corpus_parity`
  - `test_per_vertex_fixed_k_three_schemes_differentiate`
- These tests assert real current contracts:
  - dispatch fires
  - output changes or remains within an explicit parity budget
  - the library’s shipped advisory paths behave as implemented today
- They do not pretend that the retired or unmet `0.85x` Pres_Poisson goal is currently protected by CI.

### Day 2 Interpretation

- The dormant-scaffold problem in `tests/test_reorder_nd.c` is narrower than it first looked: it is not a generic “slow test” issue and it is not scattered through the file.
- All three dormant helpers are descendants of a single historical pattern:
  - write a failing-as-expected close-to-target stub
  - leave it compiled
  - comment out the `RUN_TEST(...)`
- That pattern is no longer acceptable for a truthful suite because it leaves compile-visible code that implies active protection where none exists.
- The current test framework has no first-class experimental or skipped-test category; today it only supports active `RUN_TEST(...)` execution or ordinary early-return skip behavior inside a test body.
- Day 3 should therefore decide between:
  - deleting these three historical stubs from the suite file and preserving their evidence only in sprint docs, or
  - introducing a minimal explicit non-default category before any such tests remain in-tree
- Based on the Sprint 27 and Sprint 28 decisions, the leading recommendation is deletion/docs-only for this specific trio rather than preserving them as opt-in tests.

### Day 2 Outputs

- `artifacts/day2-test-reorder-nd-audit.md`
