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

## Day 3

**Objective:** Choose the project-level rule for active, slow, experimental, and historical tests so Sprint 32 can remove dormant scaffold without inventing a test model that the current Makefile/CMake harness cannot support.

### Commands Run

1. Re-read the Sprint 32 plan, project-plan scope, and Day 2 audit:
   - `sed -n '1,260p' docs/planning/EPIC_3/SPRINT_32/PLAN.md`
   - `sed -n '1,220p' docs/planning/EPIC_3/PROJECT_PLAN.md`
   - `sed -n '1,220p' docs/planning/EPIC_3/SPRINT_32/artifacts/day2-test-reorder-nd-audit.md`
2. Inspect the actual test execution model:
   - `rg -n "add_test\\(|ctest|make test|TEST_SUITE_BEGIN|RUN_TEST\\(" CMakeLists.txt tests Makefile`
   - `sed -n '1,260p' Makefile`
   - `sed -n '120,150p' CMakeLists.txt`
   - `sed -n '1,260p' tests/test_framework.h`
3. Search for repo-level policy signals and dormant-test guidance:
   - `rg -n "slow test|experimental test|historical|opt-in|advisory|failing-as-expected|commented out|RUN_TEST commented out|active suite|coverage-honesty|skip behavior" docs tests README.md`
   - `sed -n '30,60p' docs/planning/EPIC_3/reviews/todo-codex-2026-05-15.md`
4. Reconfirm whether the commented-out `RUN_TEST(...)` anti-pattern is localized or widespread:
   - `rg -n '^\\s*/\\*\\s*RUN_TEST\\(' tests`
5. Sample adjacent historical stub patterns for policy calibration:
   - `sed -n '2985,3035p' tests/test_svd.c`
   - `sed -n '940,990p' tests/test_eigs.c`
   - `sed -n '2210,2255p' tests/test_chol_csc.c`

### Harness Constraints

- `make test` builds every `tests/test_*.c` binary and runs each one directly.
- CMake/CTest mirrors the same shape: one executable per `tests/test_*.c`, each registered with `add_test(...)`.
- Inside each binary, the current first-class notion is simply `RUN_TEST(fn)`.
- The framework already supports early-return skip behavior inside a test body, but that skip is reported as a pass because `RUN_TEST` only distinguishes "failed" from "did not fail".

Interpretation:

- Sprint 32 should not invent a new runner architecture.
- Any new policy must fit one-binary-per-file execution and the existing macro-driven framework.
- If we want truthful opt-in coverage, the minimal missing primitive is explicit skip accounting and explicit opt-in wrappers, not a broader harness rewrite.

### Chosen Policy

- **Active tests**
  - run under default `make test` and `ctest`
  - must assert a current supported behavior or measured bound
  - must pass on the normal green path
- **Slow opt-in tests**
  - still assert a current supported behavior or measured bound
  - are excluded from the default path only because of runtime or fixture cost
  - must pass when explicitly enabled
- **Experimental opt-in tests**
  - assert a current non-default path or under-evaluation behavior whose contract is still live
  - are excluded from the default path because they are non-default and not yet worth always running
  - must still pass when explicitly enabled
- **Historical evidence**
  - documents missed targets, retired goals, or dead investigative branches
  - does not compile into the normal test binary
  - belongs in sprint notes / artifacts / decision docs, not behind commented-out `RUN_TEST(...)`

### Anti-Pattern Rule

- Do not merge commented-out `RUN_TEST(...)` lines as long-lived scaffolding.
- Do not keep compiled failing-as-expected stubs in a normal suite file once the sprint that introduced them is over.
- Do not use "experimental" as a parking lot for stale or already-retired target assertions.

### Decision Framework

- Keep a test **active** when it is cheap enough and expresses a live current contract.
- Move a test to **slow opt-in** when the contract is live but the runtime cost is too high for default CI/local `make test`.
- Move a test to **experimental opt-in** only when:
  - the asserted behavior is still current
  - the path is intentionally non-default
  - there is value in rerunning it as code evolves
- Move code to **docs-only historical evidence** when:
  - the asserted target is explicitly missed or retired
  - the sprint decision docs already carry the relevant measurements
  - keeping the code would imply coverage the suite does not actually provide

### Chosen Representation For Day 4

- Add explicit skip accounting to `tests/test_framework.h`.
- Add a true skip macro for use inside test bodies.
- Add minimal opt-in wrappers for top-level test registration rather than inventing a separate runner:
  - one wrapper for slow tests
  - one wrapper for experimental tests
- Gate them with simple environment variables so `make test` / `ctest` behavior stays unchanged by default.

Recommended names:

- `SPARSE_TEST_SLOW=1`
- `SPARSE_TEST_EXPERIMENTAL=1`

Recommended behavior when disabled:

- print `[SKIP] <test-name> (set SPARSE_TEST_SLOW=1)` or `[SKIP] <test-name> (set SPARSE_TEST_EXPERIMENTAL=1)`
- increment a skipped counter
- do not report the disabled test as pass

Why this is the least invasive fit:

- it works inside the existing per-binary `main()` pattern
- it preserves default `make test` and `ctest`
- it makes opt-in behavior auditable in source instead of hidden in comments

### Policy Implications For `test_reorder_nd.c`

- The three dormant Pres_Poisson close-to-target stubs do **not** qualify for the new slow/experimental categories.
- Their contracts are stale:
  - two encode known-missed Sprint 27 target claims
  - one encodes a Sprint 28 target that was formally retired
- Day 5 should therefore delete those stubs from suite code and keep the supporting evidence in the existing Sprint 27 / Sprint 28 docs.
- The currently active advisory smoke/parity tests should remain active because they express live behavior today.

### Repo-Wide Scope Implication

- The current commented-out `RUN_TEST(...)` anti-pattern is localized to `tests/test_reorder_nd.c`.
- That makes Sprint 32's first structural cleanup well-bounded.
- Day 3 still establishes a repo-wide rule so future sprint-day stub work does not reintroduce the same pattern elsewhere.

### Day 3 Interpretation

- The right split is not "everything non-default becomes experimental."
- Cheap, stable advisory-path checks can stay active if they assert truthful current behavior.
- The new opt-in category should exist for live checks with real rerun value, not as a storage layer for old failed targets.
- Historical evidence is already well-served by this repo's sprint artifact discipline; moving dead scaffolding out of suite code is aligned with existing practice, not a new burden.

### Day 3 Outputs

- `artifacts/day3-test-truthfulness-policy.md`

## Day 4

**Objective:** Add the smallest real framework support needed for truthful slow/experimental test categories, prove it works under both Makefile and CMake flows, and keep default active-suite behavior unchanged.

### Commands Run

1. Re-read the Day 3 policy and the current framework:
   - `sed -n '1,260p' docs/planning/EPIC_3/SPRINT_32/artifacts/day3-test-truthfulness-policy.md`
   - `sed -n '1,360p' tests/test_framework.h`
2. Implement the framework support and self-check wiring:
   - edit `tests/test_framework.h`
   - add `tests/test_framework_optin.c`
   - edit `Makefile`
   - edit `CMakeLists.txt`
3. Format the touched code:
   - `make format`
4. Validate the Makefile path:
   - `make build/test_framework_optin build/test_reorder_nd`
   - `./build/test_framework_optin`
   - `./build/test_reorder_nd`
5. Validate the CMake path:
   - `cmake -S . -B build/sprint32-day1-cmake`
   - `cmake --build build/sprint32-day1-cmake --parallel 1 --target test_framework_optin test_reorder_nd`
   - `ctest --test-dir build/sprint32-day1-cmake --output-on-failure -R '^test_framework_optin$'`
   - `ctest --test-dir build/sprint32-day1-cmake --output-on-failure -R 'test_framework_optin|test_reorder_nd'`
     - first combined rerun raced the fresh `test_framework_optin` link and marked it `Not Run`
     - the rerun still proved the rebuilt CMake-side `test_reorder_nd` target passed
     - `test_framework_optin` was then rerun cleanly once the binary existed

### Implementation

- `tests/test_framework.h`
  - added `tf_tests_skipped`
  - added `tf_current_skipped`
  - summary output now reports skipped-count explicitly
  - added `SKIP_TEST(reason)` for test-body-level truthful skip reporting
  - added `RUN_TEST_SLOW(fn)` gated by `SPARSE_TEST_SLOW`
  - added `RUN_TEST_EXPERIMENTAL(fn)` gated by `SPARSE_TEST_EXPERIMENTAL`
  - added `tf_env_enabled(...)` so the env gate accepts ordinary truthy values and rejects `0` / `false` / `off` / `no`
- `tests/test_framework_optin.c`
  - new dedicated self-check binary for the Day 4 support
  - verifies:
    - normal `RUN_TEST(...)` behavior still works
    - `SKIP_TEST(...)` records a skip instead of a pass
    - disabled slow/experimental wrappers emit `[SKIP]` and do not execute the body
    - enabled slow/experimental wrappers execute normally
- `Makefile` and `CMakeLists.txt`
  - both now register `test_framework_optin` so the support path is available in both local test entry points

### Validation Results

- `./build/test_framework_optin`
  - passed
  - summary: `Tests run: 8`, `Tests failed: 0`, `Tests skipped: 3`
  - confirmed:
    - one body-level `SKIP_TEST(...)`
    - one disabled slow wrapper
    - one disabled experimental wrapper
    - both wrappers execute once enabled
- `./build/test_reorder_nd`
  - passed
  - `23` active tests still pass under the modified framework
- CMake build:
  - `test_framework_optin` built and linked cleanly
  - `test_reorder_nd` rebuilt cleanly
- CTest:
  - `test_framework_optin` passed once rerun against the completed build tree
  - `test_reorder_nd` passed in the rebuilt CMake tree (`106.45 s`)

### Day 4 Interpretation

- The framework extension stayed appropriately small: no new runner, no Makefile/CTest architecture change, and no xfail layer.
- The new support is source-auditable:
  - active tests are still obvious `RUN_TEST(...)`
  - opt-in tests will be explicit `RUN_TEST_SLOW(...)` or `RUN_TEST_EXPERIMENTAL(...)`
  - historical evidence no longer needs to hide behind commented-out `RUN_TEST(...)`
- The existing active ND suite still passes without any source changes in `tests/test_reorder_nd.c`, so Day 5 can now focus on structural cleanup rather than infrastructure risk.

### Day 4 Outputs

- `artifacts/day4-test-framework-support.md`

## Day 5

**Objective:** Apply the Sprint 32 truthfulness model to `tests/test_reorder_nd.c` by removing the dormant historical scaffold, clearing the file’s `-Wunused-function` debt, and closing the file’s remaining designated-initializer warnings in the same pass.

### Commands Run

1. Re-read the Day 4 support note and inspect the Day 5 target regions:
   - `sed -n '1,240p' docs/planning/EPIC_3/SPRINT_32/artifacts/day4-test-framework-support.md`
   - `sed -n '780,1035p' tests/test_reorder_nd.c`
   - `sed -n '1080,1760p' tests/test_reorder_nd.c`
2. Inspect the public option-struct layouts before converting the active initializers:
   - `sed -n '1,180p' include/sparse_analysis.h`
   - `sed -n '1,180p' include/sparse_cholesky.h`
   - `sed -n '1,220p' include/sparse_lu.h`
   - `sed -n '1,200p' include/sparse_ldlt.h`
3. Edit `tests/test_reorder_nd.c`:
   - remove the three dormant historical Pres_Poisson close-to-target stubs
   - remove their commented-out `RUN_TEST(...)` lines and stale scaffolding comments
   - convert the active Cholesky/LU/LDLT option structs to designated initializers
   - drop the now-unused `<errno.h>` include
4. Validate the Makefile path:
   - `make format`
   - `make build/test_framework_optin build/test_reorder_nd`
   - `./build/test_framework_optin`
   - `./build/test_reorder_nd`
5. Re-run a clean serialized Apple Clang CMake build to measure the warning delta:
   - `cmake --build build/sprint32-day1-cmake --parallel 1 --clean-first`
   - summarized from `/tmp/sprint32_day5_build.stderr`

### Structural Cleanup Landed

- Removed:
  - `test_finest_fm_annealing_pres_poisson_close_to_target`
  - `test_nd_root_spectral_pres_poisson_close_to_target`
  - `test_non_pipeline_pres_poisson_close_to_target`
- Removed the corresponding commented-out `RUN_TEST(...)` lines from `main()`.
- Rewrote the surrounding narrative comments so the file now describes the preserved active checks instead of shipping dormant compiled target stubs.
- Converted the remaining active warning sites to designated initialization:
  - `sparse_cholesky_opts_t opts_amd`
  - `sparse_cholesky_opts_t opts_nd`
  - `sparse_lu_opts_t opts`
  - `sparse_ldlt_opts_t opts`

### Validation Results

- `./build/test_framework_optin`
  - passed
  - confirms the Day 4 support path still behaves correctly after the Day 5 cleanup
- `./build/test_reorder_nd`
  - passed
  - all `23` active tests still pass
- clean serialized Apple Clang CMake rebuild
  - passed
  - full-tree warnings: `98 -> 91`
  - warning-class delta:
    - `-Wmissing-field-initializers`: `62 -> 58`
    - `-Wdouble-promotion`: `33 -> 33`
    - `-Wunused-function`: `3 -> 0`
  - `tests/test_reorder_nd.c` no longer appears in the clean-build warning output

### Day 5 Interpretation

- The Day 2 / Day 3 recommendation held up under implementation: the dormant trio was historical evidence, not live opt-in coverage.
- Deleting the dormant stubs and keeping the active advisory smoke/parity tests gives `tests/test_reorder_nd.c` an honest executed protection surface without reducing real coverage.
- The file’s warning queue is now fully closed:
  - no dormant compiled scaffold
  - no commented-out `RUN_TEST(...)`
  - no residual initializer warnings
  - no residual unused-function warnings
- Sprint 32 can now move on to the remaining test-warning queue with the highest-signal truthfulness target already resolved.

### Day 5 Outputs

- `artifacts/day5-test-reorder-nd-cleanup.md`

## Day 6

**Objective:** Document the now-landed active-vs-opt-in test split, cross-check that the maintainer-facing docs match the Day 5 `tests/test_reorder_nd.c` end state, and record the small residual truthfulness queue that remains out of scope for Sprint 32's ND cleanup.

### Commands Run

1. Re-read the Day 6 scope in the Sprint 32 plan and the Day 2 / Day 3 policy artifacts:
   - `sed -n '1,220p' docs/planning/EPIC_3/SPRINT_32/PLAN.md`
   - `sed -n '1,220p' docs/planning/EPIC_3/SPRINT_32/artifacts/day2-test-reorder-nd-audit.md`
   - `sed -n '1,220p' docs/planning/EPIC_3/SPRINT_32/artifacts/day3-test-truthfulness-policy.md`
2. Inspect the current public testing docs and the post-Day-5 ND suite state:
   - `sed -n '538,620p' README.md`
   - `sed -n '1,220p' tests/test_reorder_nd.c`
   - `sed -n '1,220p' tests/test_framework.h`
3. Search the tree for remaining truthfulness markers and sample the surviving hits:
   - `rg -n "commented-out RUN_TEST|RUN_TEST_SLOW|RUN_TEST_EXPERIMENTAL|SPARSE_TEST_SLOW|SPARSE_TEST_EXPERIMENTAL|failing-as-expected|expected fail" README.md docs tests`
   - `sed -n '930,990p' tests/test_eigs.c`
   - `sed -n '2990,3045p' tests/test_svd.c`
   - `sed -n '2218,2250p' tests/test_chol_csc.c`
4. Edit the maintainer-facing docs and Sprint 32 notes:
   - `README.md`
   - `docs/planning/EPIC_3/SPRINT_32/WORKING_NOTES.md`
   - `docs/planning/EPIC_3/SPRINT_32/artifacts/day6-coverage-honesty-docs.md`
5. Run targeted doc sanity checks:
   - `git diff -- README.md docs/planning/EPIC_3/SPRINT_32/WORKING_NOTES.md docs/planning/EPIC_3/SPRINT_32/artifacts/day6-coverage-honesty-docs.md`
   - `rg -n "RUN_TEST_SLOW|RUN_TEST_EXPERIMENTAL|SPARSE_TEST_SLOW|SPARSE_TEST_EXPERIMENTAL|commented-out RUN_TEST" README.md docs/planning/EPIC_3/SPRINT_32`

### Documentation Landed

- `README.md`
  - added a `Test Category Policy` subsection under `## Testing`
  - documented the default active surface (`RUN_TEST(...)`)
  - documented the explicit opt-in commands:
    - `SPARSE_TEST_SLOW=1 make test`
    - `SPARSE_TEST_EXPERIMENTAL=1 make test`
  - documented the rule that historical or retired targets belong in `docs/planning/` artifacts, not as commented-out `RUN_TEST(...)` lines in normal suite files
- `artifacts/day6-coverage-honesty-docs.md`
  - records the Day 5 before/after state for `tests/test_reorder_nd.c`
  - captures the active / slow / experimental / historical category model in maintainer-facing form
  - records the small residual out-of-scope queue

### Cross-Check Results

- The Day 5 `tests/test_reorder_nd.c` state matches the Day 6 docs:
  - `23` active tests remain registered
  - `0` commented-out `RUN_TEST(...)` lines remain
  - the dormant historical Pres_Poisson close-to-target trio is gone from suite code
- The new README policy matches the Day 4 framework support:
  - `RUN_TEST_SLOW(...)` / `SPARSE_TEST_SLOW=1`
  - `RUN_TEST_EXPERIMENTAL(...)` / `SPARSE_TEST_EXPERIMENTAL=1`
- The repo-wide residual truthfulness queue is narrower than the original Day 1 risk implied:
  - sampled files such as `tests/test_eigs.c`, `tests/test_svd.c`, and `tests/test_chol_csc.c` still carry older "failing-as-expected" narrative comments
  - those sampled hits are comment-history debt, not live commented-out registrations or compiled dormant helpers
  - Sprint 32 should treat them as future comment-cleanup candidates, not as evidence that the Day 5 ND structural problem still exists elsewhere

### Day 6 Interpretation

- Sprint 32 now has both halves of the truthfulness fix:
  - code-level representation support from Day 4
  - maintainer-facing documentation from Day 6
- The project rule is now explicit:
  - active current contracts run by default
  - slow and experimental current contracts are opt-in but still green when enabled
  - historical evidence stays in docs
- The remaining out-of-scope debt is mostly wording cleanup in older test comments, not hidden execution drift in the default test surface.

### Day 6 Outputs

- `artifacts/day6-coverage-honesty-docs.md`

## Day 7

**Objective:** Turn the remaining post-Day-5 test-side designated-initializer warning queue into a concrete implementation plan by rerunning the authoritative clean build, auditing the warning-bearing files by API family, and choosing validation-friendly edit batches for Days 8 and 9.

### Commands Run

1. Re-read the Day 7 scope and the Sprint 32 starting inventory:
   - `sed -n '145,245p' docs/planning/EPIC_3/SPRINT_32/PLAN.md`
   - `sed -n '1,140p' docs/planning/EPIC_3/SPRINT_32/WORKING_NOTES.md`
   - `sed -n '1,220p' docs/planning/EPIC_3/SPRINT_31/artifacts/day7-designated-initializers-batch1.md`
   - `sed -n '1,220p' docs/planning/EPIC_3/SPRINT_31/artifacts/day8-designated-initializers-batch2.md`
2. Rerun the authoritative clean serialized Apple Clang CMake build for the current branch:
   - `cmake --build build/sprint32-day1-cmake --parallel 1 --clean-first > /tmp/sprint32_day7b_build.stdout 2> /tmp/sprint32_day7b_build.stderr`
3. Derive the current post-Day-5 warning counts from the fresh stderr capture:
   - counts by area
   - counts by warning class
   - counts by file
   - counts by file and warning class
4. Inspect the remaining initializer-bearing test files and public option layouts:
   - `rg -n "sparse_[a-z_]+_opts_t [A-Za-z0-9_]+ = \\{" tests/test_ldlt.c tests/test_colamd.c tests/test_chol_csc.c tests/test_cholesky.c tests/test_sprint12_integration.c tests/test_reorder.c tests/test_sprint18_integration.c tests/test_sprint19_integration.c tests/test_sprint20_integration.c tests/test_etree.c`
   - `wc -l tests/test_ldlt.c tests/test_colamd.c tests/test_chol_csc.c tests/test_cholesky.c tests/test_sprint12_integration.c tests/test_reorder.c tests/test_sprint18_integration.c tests/test_sprint19_integration.c tests/test_sprint20_integration.c tests/test_etree.c`
   - `sed -n '1,220p' include/sparse_cholesky.h`
   - `sed -n '1,240p' include/sparse_ldlt.h`
   - `sed -n '1,260p' include/sparse_analysis.h`
   - `sed -n '1,220p' include/sparse_qr.h`
   - `sed -n '1,220p' include/sparse_lu.h`
5. Edit the Sprint 32 notes and write the Day 7 batch-design artifact:
   - `docs/planning/EPIC_3/SPRINT_32/WORKING_NOTES.md`
   - `docs/planning/EPIC_3/SPRINT_32/artifacts/day7-initializer-batch-design.md`

### Current Post-Day-5 Warning State

Fresh Day 7 clean-build counts:

- full-tree warnings: `91`
- `src`: `0`
- `tests`: `91`
- `benchmarks`: `0`
- `examples`: `0`
- `-Wmissing-field-initializers`: `58`
- `-Wdouble-promotion`: `33`

Initializer-bearing files now total `10`:

- `tests/test_ldlt.c`: `18`
- `tests/test_sprint20_integration.c`: `5`
- `tests/test_chol_csc.c`: `8`
- `tests/test_colamd.c`: `7`
- `tests/test_cholesky.c`: `5`
- `tests/test_sprint12_integration.c`: `5`
- `tests/test_reorder.c`: `4`
- `tests/test_sprint18_integration.c`: `3`
- `tests/test_sprint19_integration.c`: `2`
- `tests/test_etree.c`: `1`

Interpretation:

- Day 5 closed `tests/test_reorder_nd.c`, so the designated-initializer queue is now `62 -> 58`.
- No non-test warnings reappeared.
- The remaining initializer work is cleaner to batch by API family than by raw descending file size.

### Pattern Audit

The remaining `-Wmissing-field-initializers` sites are still the same Sprint 31 mechanical pattern: public option structs gained trailing backend, telemetry, or callback/context fields, while tests kept positional initializers that stop early.

Main sub-patterns:

- pre-backend Cholesky / LDLT forms
  - examples: `sparse_cholesky_opts_t opts = {SPARSE_REORDER_AMD};`
  - examples: `sparse_ldlt_opts_t opts = {SPARSE_REORDER_AMD, 0.0};`
  - current first warning field: `backend`
- pre-callback backend/telemetry forms
  - examples: `{SPARSE_REORDER_NONE, 0.0, SPARSE_LDLT_BACKEND_AUTO, &used_csc}`
  - examples: `{SPARSE_REORDER_AMD, SPARSE_CHOL_BACKEND_AUTO, &used}`
  - current first warning field: `progress_cb`
- pre-callback LU / QR forms
  - LU examples: `{SPARSE_PIVOT_PARTIAL, SPARSE_REORDER_AMD, 1e-12}`
  - QR examples: `{SPARSE_REORDER_COLAMD, 0, 0}` or `{SPARSE_REORDER_COLAMD, 0, 1}`
  - current first warning field: `progress_cb`

One Day 7 high-signal special case:

- `tests/test_chol_csc.c:4045` currently uses `sparse_cholesky_opts_t opts = {use_amd ? SPARSE_REORDER_AMD : SPARSE_REORDER_NONE, 0.0};`
- that positional `0.0` is landing in the enum-valued `backend` slot only because `0` currently maps to `SPARSE_CHOL_BACKEND_AUTO`
- this is exactly the kind of brittle positional coupling Day 8 / Day 9 should remove

### Chosen Batch Order

#### Day 8 Batch I: LDLT family

Files:

- `tests/test_ldlt.c`
- `tests/test_sprint12_integration.c`
- `tests/test_sprint20_integration.c`

Why this batch first:

- same `sparse_ldlt_opts_t` family across all three files
- same two sub-patterns:
  - old two-field `{reorder, tol}` forms that now need designated `.reorder` / `.tol`
  - older four-field backend/telemetry forms that now need designated `.backend` / `.used_csc_path`
- highest raw initializer payoff: `28` of the remaining `58`
- `tests/test_sprint20_integration.c` is mixed-class, but its five initializer warnings are still mechanically LDLT-specific and belong with the LDLT family rather than the later double-promotion batch

Planned Day 8 validation targets:

- `make format`
- `make build/test_ldlt build/test_sprint12_integration build/test_sprint20_integration`
- run the three touched binaries
- clean serialized CMake rebuild for warning delta

#### Day 9 Batch II: Cholesky + QR + LU companion family

Files:

- `tests/test_chol_csc.c`
- `tests/test_cholesky.c`
- `tests/test_sprint18_integration.c`
- `tests/test_sprint19_integration.c`
- `tests/test_colamd.c`
- `tests/test_reorder.c`
- `tests/test_etree.c`

Why this is the right second batch:

- it closes the remaining `30` initializer warnings
- the files fall into small, internally coherent subgroups:
  - Cholesky backend/callback family:
    - `test_chol_csc.c`
    - `test_cholesky.c`
    - `test_sprint18_integration.c`
    - `test_sprint19_integration.c`
  - QR callback family:
    - `test_colamd.c`
  - LU callback family:
    - `test_reorder.c`
    - `test_etree.c`
- `tests/test_etree.c` is only `1` warning, so splitting it into a later sprint would just strand a same-pattern LU cleanup behind `tests/test_reorder.c`

Planned Day 9 validation targets:

- `make format`
- targeted rebuild of the touched binaries
- run the touched binaries with the highest-signal coverage:
  - `test_chol_csc`
  - `test_cholesky`
  - `test_colamd`
  - `test_reorder`
  - `test_etree`
  - plus the touched sprint integrations
- clean serialized CMake rebuild for final initializer delta

### Day 7 Interpretation

- The Day 7 inventory confirms that the designated-initializer queue is now sharply separable from the double-promotion queue:
  - initializer cleanup: `58` warnings in `10` files
  - double-promotion cleanup: `33` warnings in `11` files
- The Sprint 31 public-facing rule reuses cleanly for Sprint 32 tests:
  - name only the fields the test intentionally overrides
  - let default-valued trailing backend/telemetry/callback fields zero-initialize
  - do not mirror evolving public struct layouts positionally
- The chosen Day 8 / Day 9 batch order keeps the edit surfaces coherent and gives each implementation day a validation set that matches the edited API family instead of forcing a scattered whole-tree pass after every small change.

### Day 7 Outputs

- `artifacts/day7-initializer-batch-design.md`

## Day 8

**Objective:** Land the first designated-initializer cleanup batch from the Day 7 plan by converting the remaining LDLT-family positional option-struct forms in `tests/test_ldlt.c`, `tests/test_sprint12_integration.c`, and `tests/test_sprint20_integration.c`, then validate that the touched binaries still pass and the clean-build warning queue drops by the expected amount.

### Commands Run

1. Re-read the Day 8 target batch and inspect the exact LDLT warning sites:
   - `sed -n '1,220p' docs/planning/EPIC_3/SPRINT_32/artifacts/day7-initializer-batch-design.md`
   - `rg -n "sparse_ldlt_opts_t [A-Za-z0-9_]+ = \\{" tests/test_ldlt.c tests/test_sprint12_integration.c tests/test_sprint20_integration.c`
   - `sed -n '800,840p' tests/test_ldlt.c`
   - `sed -n '1070,1125p' tests/test_ldlt.c`
   - `sed -n '1148,1310p' tests/test_ldlt.c`
   - `sed -n '1422,1450p' tests/test_ldlt.c`
   - `sed -n '2098,2165p' tests/test_ldlt.c`
   - `sed -n '2288,2565p' tests/test_ldlt.c`
   - `sed -n '90,390p' tests/test_sprint12_integration.c`
   - `sed -n '140,270p' tests/test_sprint20_integration.c`
2. Edit the three Day 8 batch files:
   - `tests/test_ldlt.c`
   - `tests/test_sprint12_integration.c`
   - `tests/test_sprint20_integration.c`
3. Format the touched sources:
   - `make format`
4. Validate the targeted Makefile path:
   - `make build/test_ldlt build/test_sprint12_integration build/test_sprint20_integration`
   - `./build/test_ldlt`
   - `./build/test_sprint12_integration`
   - `./build/test_sprint20_integration`
5. Re-run a clean serialized Apple Clang CMake build to measure the warning delta:
   - `cmake --build build/sprint32-day1-cmake --parallel 1 --clean-first > /tmp/sprint32_day8_build.stdout 2> /tmp/sprint32_day8_build.stderr`
   - derive fresh counts by area, class, file, and file/class from `/tmp/sprint32_day8_build.stderr`

### Changes Landed

- `tests/test_ldlt.c`
  - converted all remaining `sparse_ldlt_opts_t` positional initializers to designated form
  - covered both main sub-patterns:
    - reorder/tolerance-only forms
    - backend/telemetry forms used by the Sprint 20 backend-routing tests
- `tests/test_sprint12_integration.c`
  - converted the AMD and RCM LDLT option initializers to designated form
- `tests/test_sprint20_integration.c`
  - converted the AUTO / LINKED_LIST / CSC backend-routing option initializers to designated form

Chosen Day 8 rule:

- name the specific LDLT fields the test intends to control:
  - `.reorder`
  - `.tol`
  - `.backend`
  - `.used_csc_path` where telemetry is part of the test contract
- leave trailing callback/context fields at their documented default `NULL` values rather than mirroring the evolving public struct layout positionally

### Validation Results

- `make build/test_ldlt build/test_sprint12_integration build/test_sprint20_integration`
  - passed
- `./build/test_ldlt`
  - passed
  - summary: `83` tests run, `0` failed, `0` skipped
- `./build/test_sprint12_integration`
  - passed
  - summary: `8` tests run, `0` failed, `0` skipped
- `./build/test_sprint20_integration`
  - passed
  - summary: `20` tests run, `0` failed, `0` skipped
- clean serialized Apple Clang CMake rebuild
  - passed

### Warning Delta

Relative to the Day 7 baseline:

- full-tree warnings: `91 -> 63`
- `tests` warnings: `91 -> 63`
- `-Wmissing-field-initializers`: `58 -> 30`
- `-Wdouble-promotion`: unchanged at `33`

Per-file Day 8 reduction:

- `tests/test_ldlt.c`: `18 -> 0`
- `tests/test_sprint12_integration.c`: `5 -> 0`
- `tests/test_sprint20_integration.c`: `9 -> 4`
  - the `5` initializer warnings are gone
  - the remaining `4` warnings are its already-planned `-Wdouble-promotion` sites

### Day 8 Interpretation

- The Day 7 batch design was accurate: the LDLT family was a coherent mechanical cleanup unit and closed cleanly in one pass.
- No behavior regressions appeared in the touched high-signal binaries.
- The remaining initializer queue is now exactly the planned Day 9 surface:
  - `tests/test_chol_csc.c`: `8`
  - `tests/test_colamd.c`: `7`
  - `tests/test_cholesky.c`: `5`
  - `tests/test_reorder.c`: `4`
  - `tests/test_sprint18_integration.c`: `3`
  - `tests/test_sprint19_integration.c`: `2`
  - `tests/test_etree.c`: `1`

### Day 8 Outputs

- `artifacts/day8-designated-initializers-batch1.md`

## Day 9

**Objective:** Close the remaining designated-initializer queue by converting the companion Cholesky, QR, and LU positional option-struct forms in the Day 7 Batch II files, then validate that the touched binaries still pass and the clean-build warning queue becomes pure double-promotion debt.

### Commands Run

1. Re-read the Day 9 target batch and inspect the exact remaining warning sites:
   - `sed -n '1,240p' docs/planning/EPIC_3/SPRINT_32/artifacts/day7-initializer-batch-design.md`
   - `rg -n "warning: missing field .*\\[-Wmissing-field-initializers\\]" /tmp/sprint32_day8_build.stderr`
   - `rg -n "sparse_(cholesky|analysis|qr|lu)_opts_t [A-Za-z0-9_]+ = \\{" tests/test_chol_csc.c tests/test_cholesky.c tests/test_sprint18_integration.c tests/test_sprint19_integration.c tests/test_colamd.c tests/test_reorder.c tests/test_etree.c`
   - `sed -n '260,460p' tests/test_cholesky.c`
   - `sed -n '760,910p' tests/test_colamd.c`
   - `sed -n '420,520p' tests/test_sprint19_integration.c`
   - `sed -n '120,280p' tests/test_sprint18_integration.c`
   - `sed -n '760,920p' tests/test_reorder.c`
   - `sed -n '1228,1260p' tests/test_etree.c`
   - `sed -n '760,810p' tests/test_chol_csc.c`
   - `sed -n '4036,4395p' tests/test_chol_csc.c`
2. Edit the Day 9 batch files:
   - `tests/test_chol_csc.c`
   - `tests/test_cholesky.c`
   - `tests/test_sprint18_integration.c`
   - `tests/test_sprint19_integration.c`
   - `tests/test_colamd.c`
   - `tests/test_reorder.c`
   - `tests/test_etree.c`
3. Format the touched sources:
   - `make format`
4. Validate the targeted Makefile path:
   - `make build/test_chol_csc build/test_cholesky build/test_sprint18_integration build/test_sprint19_integration build/test_colamd build/test_reorder build/test_etree`
   - `./build/test_chol_csc`
   - `./build/test_cholesky`
   - `./build/test_sprint18_integration`
   - `./build/test_sprint19_integration`
   - `./build/test_colamd`
   - `./build/test_reorder`
   - `./build/test_etree`
5. Re-run a clean serialized Apple Clang CMake build to measure the final initializer delta:
   - `cmake --build build/sprint32-day1-cmake --parallel 1 --clean-first > /tmp/sprint32_day9_build.stdout 2> /tmp/sprint32_day9_build.stderr`
   - derive fresh counts by area, class, file, and file/class from `/tmp/sprint32_day9_build.stderr`

### Changes Landed

- `tests/test_cholesky.c`
  - converted all remaining reorder-only `sparse_cholesky_opts_t` forms to designated initialization
- `tests/test_sprint18_integration.c`
  - converted the AUTO / LINKED_LIST / CSC Cholesky dispatch-routing options to designated form
- `tests/test_sprint19_integration.c`
  - converted the forced linked-list and forced CSC Cholesky comparison options to designated form
- `tests/test_colamd.c`
  - converted the remaining QR options to designated form
  - preserved explicit `.sparse_mode = 1` where sparse-mode behavior is part of the test contract
- `tests/test_reorder.c`
  - converted all remaining LU option initializers to designated form
- `tests/test_etree.c`
  - converted the LU-with-AMD containment check options to designated form
- `tests/test_chol_csc.c`
  - converted the remaining Cholesky option sites to designated form
  - replaced the brittle two-field `{reorder, 0.0}` form with a reorder-only designated initializer, making the default AUTO backend explicit by omission rather than by enum-zero coercion

Chosen Day 9 rule:

- name only the fields each test intentionally overrides:
  - Cholesky:
    - `.reorder`
    - `.backend`
    - `.used_csc_path`
  - QR:
    - `.reorder`
    - `.sparse_mode` where needed
  - LU:
    - `.pivot`
    - `.reorder`
    - `.tol`
- leave callback/context fields at default `NULL`
- avoid positional mirroring of the evolving public option-struct layouts

### Validation Results

- `make build/test_chol_csc build/test_cholesky build/test_sprint18_integration build/test_sprint19_integration build/test_colamd build/test_reorder build/test_etree`
  - passed
- direct binary runs:
  - `./build/test_chol_csc`
    - passed
    - summary: `137` tests run, `0` failed, `0` skipped
  - `./build/test_cholesky`
    - passed
    - summary: `21` tests run, `0` failed, `0` skipped
  - `./build/test_sprint18_integration`
    - passed
    - summary: `10` tests run, `0` failed, `0` skipped
  - `./build/test_sprint19_integration`
    - passed
    - summary: `8` tests run, `0` failed, `0` skipped
  - `./build/test_colamd`
    - passed
    - summary: `70` tests run, `0` failed, `0` skipped
  - `./build/test_reorder`
    - passed
    - summary: `38` tests run, `0` failed, `0` skipped
  - `./build/test_etree`
    - passed
    - summary: `94` tests run, `0` failed, `0` skipped
- clean serialized Apple Clang CMake rebuild
  - passed

### Warning Delta

Relative to the Day 8 baseline:

- full-tree warnings: `63 -> 33`
- `tests` warnings: `63 -> 33`
- `-Wmissing-field-initializers`: `30 -> 0`
- `-Wdouble-promotion`: unchanged at `33`

Per-file Day 9 reduction:

- `tests/test_chol_csc.c`: `8 -> 0`
- `tests/test_colamd.c`: `8 -> 1`
  - the `7` initializer warnings are gone
  - the remaining `1` warning is the already-planned `-Wdouble-promotion` site
- `tests/test_cholesky.c`: `5 -> 0`
- `tests/test_reorder.c`: `4 -> 0`
- `tests/test_sprint18_integration.c`: `6 -> 3`
  - the `3` initializer warnings are gone
  - the remaining `3` warnings are its already-planned `-Wdouble-promotion` sites
- `tests/test_sprint19_integration.c`: `3 -> 1`
  - the `2` initializer warnings are gone
  - the remaining `1` warning is its already-planned `-Wdouble-promotion` site
- `tests/test_etree.c`: `1 -> 0`

### Day 9 Interpretation

- The remaining Batch II queue matched the Day 7 plan exactly and closed cleanly in one pass.
- Sprint 32’s designated-initializer work is now complete:
  - `-Wmissing-field-initializers`: `62 -> 0` versus the Day 1 baseline
- The entire residual warning queue is now purely numeric/mechanical promotion cleanup:
  - full-tree warnings: `33`
  - all `33` are `-Wdouble-promotion`
- No touched-binary regressions appeared in the Cholesky, QR, LU, or companion integration coverage.

### Day 9 Outputs

- `artifacts/day9-designated-initializers-batch2.md`
