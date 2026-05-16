# Sprint 30 Working Notes

## Day 1

**Objective:** Capture a clean bounded warning baseline from the CMake and Makefile build paths and initialize Sprint 30 working artifacts.

### Commands Run

1. Clean Day 1 build directories:
   - `rm -rf build/sprint30-day1-cmake build/sprint30-day1-make`
2. CMake configure:
   - `cmake -S . -B build/sprint30-day1-cmake`
3. CMake full build:
   - `cmake --build build/sprint30-day1-cmake -j4`
4. Makefile default build path:
   - `make BUILDDIR=build/sprint30-day1-make all`

All command stdout/stderr were redirected into `docs/planning/EPIC_3/SPRINT_30/artifacts/`.

### Observations

- The rerun completed normally in bounded polling loops.
- The earlier apparent 13-hour runtime was not representative of actual Day 1 work.
- CMake configure completed cleanly with no warnings.
- The CMake full build is the meaningful full-tree warning baseline for Sprint 30.
- The Makefile default `all` target produced no warnings, but its scope is library-only.

### Raw Counts

- CMake configure warnings: `0`
- CMake full-build warnings: `123`
- Makefile default-build warnings: `0`

### CMake Full-Build Breakdown

By area:

- `tests`: `98`
- `benchmarks`: `13`
- `src`: `11`
- `examples`: `1`

By warning class:

- `-Wmissing-field-initializers`: `72`
- `-Wdouble-promotion`: `45`
- `-Wunused-function`: `3`
- `-Wimplicit-function-declaration`: `2`
- `-Wswitch`: `1`

Top warning-bearing files:

- `tests/test_ldlt.c`: `18`
- `tests/test_sprint20_integration.c`: `9`
- `tests/test_colamd.c`: `8`
- `tests/test_chol_csc.c`: `8`
- `tests/test_reorder_nd.c`: `7`
- `tests/test_svd.c`: `6`
- `tests/test_sprint6_integration.c`: `6`
- `tests/test_sprint18_integration.c`: `6`
- `src/sparse_svd.c`: `5`
- `benchmarks/bench_main.c`: `5`
- `src/sparse_qr.c`: `4`

### Day 1 Interpretation

- The highest-signal Sprint 30 core-library work remains the `-Wdouble-promotion` cleanup in the four source files identified in the Epic 3 review.
- The largest raw warning volume is outside the core library, especially in tests, which matches the Epic 3 review’s claim that the suite is carrying warning debt and dormant scaffolding.
- The Makefile path needs a more comparable non-default compile strategy later if it is going to be used as a repository-wide warning baseline rather than a library-only baseline.

### Day 1 Outputs

- `artifacts/day1-warning-baseline.md`
- `artifacts/day1-cmake-warning-counts-by-area.txt`
- `artifacts/day1-cmake-warning-counts-by-class.txt`
- `artifacts/day1-cmake-warning-counts-by-file.txt`
- raw stdout/stderr logs for bounded reruns

## Day 2

**Objective:** Classify the Day 1 baseline into a usable warning taxonomy, separate library-proper warnings from auxiliary-code warnings, and rank warning classes by Sprint 30 urgency versus later-sprint urgency.

### Commands Run

Day 2 reused the Day 1 CMake full-build stderr artifact and derived:

1. counts by area and warning class
2. counts by library versus auxiliary ownership
3. ranked top warning-bearing files for handoff planning

All derived outputs were written under `docs/planning/EPIC_3/SPRINT_30/artifacts/`.

### Day 2 Classification

By area and warning class:

- `src`
  - `-Wdouble-promotion`: `11`
- `tests`
  - `-Wdouble-promotion`: `33`
  - `-Wmissing-field-initializers`: `62`
  - `-Wunused-function`: `3`
- `benchmarks`
  - `-Wdouble-promotion`: `1`
  - `-Wimplicit-function-declaration`: `2`
  - `-Wmissing-field-initializers`: `9`
  - `-Wswitch`: `1`
- `examples`
  - `-Wmissing-field-initializers`: `1`

Ownership split:

- library-proper warnings (`src/`): `11`
- auxiliary warnings (`tests/`, `benchmarks/`, `examples/`): `112`

### Day 2 Severity Ordering

Ranked cleanup queue:

1. `src` `-Wdouble-promotion`
   - Sprint 30 must-fix
   - only warning class currently present in the library proper
2. auxiliary `-Wmissing-field-initializers`
   - high-volume later-sprint cleanup
   - tracks evolving options-struct maintenance drift
3. auxiliary `-Wunused-function`
   - low-volume but structurally important
   - strong signal for dormant test scaffolding
4. benchmark `-Wimplicit-function-declaration`
   - portability/tooling issue
   - natural Sprint 31 target
5. benchmark `-Wswitch`
   - stale enum/tooling drift
   - natural Sprint 31 target
6. auxiliary `-Wdouble-promotion`
   - real cleanup work, but lower priority than the library-proper cluster

### Day 2 Interpretation

- Sprint 30’s first code-edit target is now confirmed quantitatively, not just qualitatively: the library-proper warning debt is exactly the `-Wdouble-promotion` cluster in four `src/` files.
- The bulk of the raw warning volume is auxiliary-code debt, especially tests with evolving options structs and dormant scaffolding.
- The taxonomy supports the Epic 3 sequencing:
  - Sprint 30: core-library hygiene
  - Sprint 31: benchmark/example tooling drift
  - Sprint 32: dormant test scaffolding and test-honesty cleanup

### Day 2 Outputs

- `artifacts/day2-warning-taxonomy.md`
- `artifacts/day2-warning-counts-by-area-and-class.txt`
- `artifacts/day2-warning-counts-library-vs-auxiliary.txt`

## Day 3

**Objective:** Audit the four Sprint 30 core warning files before editing them, determine whether the warning sites are compile-hygiene-only or signs of deeper issues, and choose the implementation idiom for Day 4-5.

### Files Audited

- `src/sparse_lu.c`
- `src/sparse_ldlt.c`
- `src/sparse_qr.c`
- `src/sparse_svd.c`

### Findings

- The library-proper warning cluster remains tightly scoped:
  - `src/sparse_lu.c`: `1`
  - `src/sparse_ldlt.c`: `1`
  - `src/sparse_qr.c`: `4`
  - `src/sparse_svd.c`: `5`
- Every current library-proper warning is a `-Wdouble-promotion` warning caused by using the float-typed `INFINITY` macro in `double` contexts on the current Apple Clang toolchain.

Per-file audit verdict:

- `src/sparse_lu.c`
  - compile-hygiene only
  - current `condest` behavior is semantically correct
- `src/sparse_ldlt.c`
  - compile-hygiene only
  - mirrors the LU condition-estimation path correctly
- `src/sparse_qr.c`
  - compile-hygiene only
  - sentinel and `condest` uses are consistent with the public header contract
- `src/sparse_svd.c`
  - compile-hygiene only
  - documented API contract already says the function returns infinity for singular matrices and on failure

### Chosen Day 4-5 Cleanup Idiom

Use `HUGE_VAL` for implementation-side double infinity values in the four audited `src/` files.

Reasoning:

- preserves the current runtime semantics
- avoids the current `INFINITY` -> `HUGE_VALF` float-promotion warning path
- minimal-risk source edit
- no need for a new internal abstraction at Sprint 30 scope

Rejected alternatives:

- `(double)INFINITY`
- new internal infinity macro/helper
- changing public docs from “infinity” semantics to implementation-specific wording

### Day 3 Interpretation

- Sprint 30 Day 4-5 can proceed as a narrow source cleanup, not a semantic redesign.
- No deeper algorithmic inconsistency was found at the audited warning sites.
- The cleanup should preserve all current public behavior while removing the warning class from the core library.

### Day 3 Outputs

- `artifacts/day3-core-warning-audit.md`

## Day 4

**Objective:** Apply the first narrow code-edit batch from the Day 3 audit, limited to `src/sparse_lu.c` and `src/sparse_ldlt.c`, then measure the warning reduction with a fresh bounded build.

### Files Edited

- `src/sparse_lu.c`
- `src/sparse_ldlt.c`

### Change Applied

Per the Day 3 decision, replaced implementation-side `INFINITY` uses with `HUGE_VAL` in the two Day 4 target files:

- `src/sparse_lu.c`
  - `*condest = INFINITY;` -> `*condest = HUGE_VAL;`
- `src/sparse_ldlt.c`
  - `*condest = INFINITY;` -> `*condest = HUGE_VAL;`

No public API contract changed.

### Validation Performed

Fresh bounded build:

- configured and built `build/sprint30-day4-cmake`
- captured logs in `artifacts/day4-cmake-*`

Warning deltas relative to Day 1:

- total warnings: `123` -> `121`
- `src` warnings: `11` -> `9`
- `src/sparse_lu.c`: `1` -> `0`
- `src/sparse_ldlt.c`: `1` -> `0`

Targeted regression slice:

- ran `ctest --test-dir build/sprint30-day4-cmake --output-on-failure -R 'test_sparse_lu|test_ldlt|test_reorder|test_edge_cases'`
- executed `7` tests because the regex also matched `test_ldlt_csc`, `test_reorder_nd`, and `test_reorder_amd_qg`
- result: `7/7` passed in `88.14 sec`

### Day 4 Interpretation

- The Day 3 audit decision was correct: the targeted edits removed the warnings without disturbing behavior.
- Day 4 removed the full warning contribution of the two edited files.
- The remaining Sprint 30 library-proper warning work is now exactly:
  - `src/sparse_qr.c`: `4`
  - `src/sparse_svd.c`: `5`

### Day 4 Outputs

- `artifacts/day4-core-fixes-batch1.md`
- `artifacts/day4-cmake-configure.stdout.txt`
- `artifacts/day4-cmake-configure.stderr.txt`
- `artifacts/day4-cmake-build.stdout.txt`
- `artifacts/day4-cmake-build.stderr.txt`
- `artifacts/day4-targeted-ctest.stdout.txt`
- `artifacts/day4-targeted-ctest.stderr.txt`

## Day 5

**Objective:** Apply the remaining Day 3 core-library cleanup in `src/sparse_qr.c` and `src/sparse_svd.c`, confirm the targeted `src/` warning class is gone, and re-verify both primary local build paths.

### Files Edited

- `src/sparse_qr.c`
- `src/sparse_svd.c`

### Change Applied

Per the Day 3 decision, replaced the remaining implementation-side `INFINITY` uses with `HUGE_VAL` in the two Day 5 target files:

- `src/sparse_qr.c`
  - `double prev_rnorm = INFINITY;` -> `double prev_rnorm = HUGE_VAL;` at two refinement sites
  - `info->condest = INFINITY;` -> `info->condest = HUGE_VAL;`
  - `return INFINITY;` -> `return HUGE_VAL;`
- `src/sparse_svd.c`
  - five `return INFINITY;` sites -> `return HUGE_VAL;`

This completed the local consistency sweep for the audited infinity-related warning pattern inside the Sprint 30 core files.

### Validation Performed

Fresh bounded CMake build:

- configured and built `build/sprint30-day5-cmake`
- captured logs in `artifacts/day5-cmake-*`

Makefile verification build:

- built `build/sprint30-day5-make`
- captured logs in `artifacts/day5-make-*`

Warning deltas relative to Day 4:

- total warnings: `121` -> `112`
- `src` warnings: `9` -> `0`
- `src/sparse_qr.c`: `4` -> `0`
- `src/sparse_svd.c`: `5` -> `0`

Warning deltas relative to Day 1:

- total warnings: `123` -> `112`
- library-proper warnings: `11` -> `0`

Remaining Day 5 CMake full-build warnings are all auxiliary:

- `tests`: `98`
- `benchmarks`: `13`
- `examples`: `1`

Makefile warning count:

- `0`

Targeted regression slice:

- ran `ctest --test-dir build/sprint30-day5-cmake --output-on-failure -R 'test_qr|test_svd|test_bidiag|test_dense'`
- executed `4` tests
- result: `4/4` passed in `5.32 sec`

### Day 5 Interpretation

- Sprint 30’s targeted core-library warning sites are now fully resolved.
- The warning reduction matched the exact remaining `src/` warning count, which confirms the Day 3 hotspot audit was complete and accurate.
- The remaining warning debt is exclusively in auxiliary code and stays aligned with the Day 2 priority queue for later sprint work.

### Day 5 Outputs

- `artifacts/day5-core-fixes-batch2.md`
- `artifacts/day5-cmake-configure.stdout.txt`
- `artifacts/day5-cmake-configure.stderr.txt`
- `artifacts/day5-cmake-build.stdout.txt`
- `artifacts/day5-cmake-build.stderr.txt`
- `artifacts/day5-make-build.stdout.txt`
- `artifacts/day5-make-build.stderr.txt`
- `artifacts/day5-targeted-ctest.stdout.txt`
- `artifacts/day5-targeted-ctest.stderr.txt`

## Day 6

**Objective:** Write the Sprint 30 compile-hygiene playbook so later Epic 3 warning cleanup can use explicit rules about scope, urgency, build-path authority, and closure evidence.

### Documents Added

- `COMPILE_HYGIENE_PLAYBOOK.md`
- `artifacts/day6-compile-hygiene-playbook.md`

### Day 6 Policy Decisions

The playbook records these main rules:

- the clean Apple Clang CMake build remains the authoritative full-tree warning inventory
- the Makefile `all` path remains a library-only cross-check, not a repository-wide warning baseline
- `src/` warnings are not acceptable on supported build surfaces
- pre-existing auxiliary warnings may remain temporarily only if they are measured and explicitly queued
- new warnings in any area are not acceptable
- warning closure claims require before/after counts, captured logs, and proportional regression evidence
- global `-Werror` remains deferred until the baseline debt is substantially reduced

### Day 6 Scope Notes

- Day 6 is a documentation-and-policy checkpoint, not a source-behavior change.
- The playbook intentionally reflects Sprint 30’s measured state after Day 5:
  - `src`: `0`
  - `tests`: `98`
  - `benchmarks`: `13`
  - `examples`: `1`
- The policy therefore separates “must stay clean now” from “documented debt queued for later cleanup.”

### Validation Performed

- ran `make format`
- ran `make lint`
- ran `make test`
- result: all three completed successfully

### Day 6 Interpretation

- Sprint 30 now has an explicit compile-quality decision framework rather than relying on informal judgments.
- Later warning-cleanup and gating work can reference a concrete rule set tied to the measured baseline.
- The playbook stays within Sprint 30 scope by focusing on existing warning debt and validation standards, not feature expansion.

### Day 6 Outputs

- `COMPILE_HYGIENE_PLAYBOOK.md`
- `artifacts/day6-compile-hygiene-playbook.md`

## Day 7

**Objective:** Automate the Sprint 30 rebuild and warning-capture workflow so clean baseline reproduction and post-edit validation no longer require manual command reconstruction.

### Files Added Or Updated

- `scripts/epic3_warning_workflow.sh`
- `Makefile`
- `REBUILD_WORKFLOW.md`
- `artifacts/day7-rebuild-workflow.md`

### Day 7 Workflow Decisions

- the workflow uses label-based build directories:
  - `build/<label>-cmake`
  - `build/<label>-make`
- the workflow writes all logs and derived summaries into `docs/planning/EPIC_3/SPRINT_30/artifacts`
- the authoritative full-tree warning inventory remains the clean CMake build
- the Makefile path remains a library-only cross-check
- the workflow includes `ctest` so it validates the CMake path rather than only compiling it

### Validation Performed

Workflow validation run:

- ran `make warning-workflow WARNING_WORKFLOW_LABEL=day7-workflow`
- ran `make format`
- ran `make lint`
- ran `make test`

Observed results:

- CMake configure warnings: `0`
- CMake build warnings: `112`
- Makefile build warnings: `0`
- `ctest`: `52/52` passed in `205.75 sec`
- end-of-day validation sequence: passed

Derived warning profile from the validated workflow run:

- by area
  - `tests`: `98`
  - `benchmarks`: `13`
  - `examples`: `1`
- by warning class
  - `-Wmissing-field-initializers`: `72`
  - `-Wdouble-promotion`: `34`
  - `-Wunused-function`: `3`
  - `-Wimplicit-function-declaration`: `2`
  - `-Wswitch`: `1`

### Day 7 Interpretation

- Sprint 30 now has a single documented command entry point for warning-baseline reproduction.
- The new workflow is suitable for both pre-change and post-change capture because labels prevent artifact overwrites.
- The validated workflow reproduced the current post-Day-5 warning state exactly, which confirms that it is measuring the intended build surfaces.

### Day 7 Outputs

- `REBUILD_WORKFLOW.md`
- `artifacts/day7-rebuild-workflow.md`
- `artifacts/day7-workflow-workflow-summary.md`
- `artifacts/day7-workflow-cmake-warning-counts-by-area.txt`
- `artifacts/day7-workflow-cmake-warning-counts-by-class.txt`
- `artifacts/day7-workflow-cmake-warning-counts-by-file.txt`

## Day 8

**Objective:** Re-run the warning baseline on the Apple Clang CMake path and the Makefile path after the core cleanup batch, then separate shared warning debt from path-scope-only findings.

### Files Added Or Updated

- `artifacts/day8-cross-path-analysis.md`
- `scripts/epic3_warning_workflow.sh`
- `Makefile`
- `REBUILD_WORKFLOW.md`

### Day 8 Workflow Adjustment

The first Day 8 rerun exposed a warning-capture stability issue:

- parallel CMake builds could interleave stderr warning lines
- total warning counts stayed correct
- area/file attribution could drift because split warning lines stopped matching the parser assumptions

Day 8 fixed this by changing the workflow default to:

- `WARNING_WORKFLOW_JOBS=1`

This keeps the warning-capture artifacts parse-stable for baseline comparison work.

### Validation Performed

Cross-path workflow run:

- ran `make warning-workflow WARNING_WORKFLOW_LABEL=day8-crosspath`
- ran `make format`
- ran `make lint`
- ran `make test`

Observed results:

- CMake configure warnings: `0`
- CMake build warnings: `112`
- Makefile build warnings: `0`
- `ctest`: `52/52` passed in `159.95 sec`
- end-of-day validation sequence: passed

Remaining CMake warning profile:

- by area
  - `tests`: `98`
  - `benchmarks`: `13`
  - `examples`: `1`
  - `src`: `0`
- by class
  - `-Wmissing-field-initializers`: `72`
  - `-Wdouble-promotion`: `34`
  - `-Wunused-function`: `3`
  - `-Wimplicit-function-declaration`: `2`
  - `-Wswitch`: `1`

Cross-path comparison result:

- shared warnings across both measured paths: none
- CMake-only remaining warnings: `112`
- Makefile-only warnings: `0`

Interpretation:

- the remaining warnings are not library-proper warnings
- they are not false positives created by the CMake path
- they remain because the default Makefile `all` path does not compile the same auxiliary-code scope as the CMake full-tree build

### Day 8 Interpretation

- Sprint 30 now has a clean cross-path statement: the core library is warning-clean on both measured paths.
- The remaining warning debt is entirely auxiliary and CMake-visible because of build-scope differences, not because of new Day 8 regressions.
- The warning workflow itself is now better suited for later Epic 3 comparisons because its default capture mode is stable rather than interleaved.

### Day 8 Outputs

- `artifacts/day8-cross-path-analysis.md`
- `artifacts/day8-crosspath-workflow-summary.md`
- `artifacts/day8-crosspath-cmake-warning-counts-by-area.txt`
- `artifacts/day8-crosspath-cmake-warning-counts-by-class.txt`
- `artifacts/day8-crosspath-cmake-warning-counts-by-file.txt`

## Day 9

**Objective:** Run a stricter compile-only or warning-focused pass beyond the default baseline, identify any new findings, decide what is Sprint 30-actionable, and confirm the core cleanup still holds under the stricter settings.

### Files Added Or Updated

- `src/sparse_types.c`
- `artifacts/day9-stricter-compile-sweep.md`
- `artifacts/day9-strict-final-cmake-warning-counts-by-area.txt`
- `artifacts/day9-strict-final-cmake-warning-counts-by-class.txt`

### Stricter Passes Run

Compiler-stricter CMake pass:

- configured `build/day9-strict-final-cmake` with `-DCMAKE_C_FLAGS='-Wstrict-prototypes -Wmissing-prototypes'`
- built with `cmake --build build/day9-strict-final-cmake --parallel 1`

Warning-focused quality pass:

- ran `make lint`
- result: passed

End-of-day validation:

- ran `make format`
- ran `make lint`
- ran `make test`
- result: passed

### New Finding From The Strict Sweep

The initial strict CMake pass surfaced one new `src/` warning beyond the Day 8 baseline:

- `src/sparse_types.c`
  - `-Wmissing-prototypes`
  - function: `sparse_set_errno_`

This was Sprint 30-actionable because it was a core-library warning under stricter supported warning settings.

### Fix Applied

Added a prior declaration in `src/sparse_types.c` before the `sparse_set_errno_` definition:

- [src/sparse_types.c](/Users/jeff/experiments/linalg_sparse_orthogonal/src/sparse_types.c:5)

This kept the helper internal while satisfying the stricter prototype warning.

### Post-Fix Strict Results

Strict CMake build after the fix:

- warnings: `112`
- warning delta versus Day 8 baseline: `0`

By area:

- `tests`: `98`
- `benchmarks`: `13`
- `examples`: `1`
- `src`: `0`

By class:

- `-Wmissing-field-initializers`: `72`
- `-Wdouble-promotion`: `34`
- `-Wunused-function`: `3`
- `-Wimplicit-function-declaration`: `2`
- `-Wswitch`: `1`

Targeted strict-tree regression slice:

- ran `ctest --test-dir build/day9-strict-final-cmake --output-on-failure -R 'test_sparse_matrix|test_sparse_io|test_sparse_lu|test_ldlt|test_qr|test_svd'`
- observed `7/7` passed in `6.91 sec`

### Day 9 Interpretation

- The stricter pass found one real new `src/` issue rather than a large hidden second warning backlog.
- That issue was narrow and easy to fix, which supports the Sprint 30 strategy of using stricter passes as controlled discovery rather than as uncontrolled baseline replacement.
- After the fix, the stricter-pass warning profile collapsed back to the same auxiliary-code classes already present in the Day 8 baseline.

### Day 9 Outputs

- `artifacts/day9-stricter-compile-sweep.md`
- `artifacts/day9-strict-final-cmake-warning-counts-by-area.txt`
- `artifacts/day9-strict-final-cmake-warning-counts-by-class.txt`
- `artifacts/day9-strict-final-cmake-build.stderr.txt`
- `artifacts/day9-strict-final-cmake-build.stdout.txt`
- `artifacts/day9-strict-final-ctest.stdout.txt`
- `artifacts/day9-strict-final-ctest.stderr.txt`
- `artifacts/day9-make-lint.stdout.txt`
- `artifacts/day9-make-lint.stderr.txt`

## Day 10

**Objective:** Turn the test-warning backlog into an actionable queue by separating dormant scaffolding, options-struct initializer drift, and lower-priority incidental numeric-literal warnings.

### Files Added

- `artifacts/day10-test-warning-triage.md`
- `artifacts/day10-test-warning-counts-by-file.txt`
- `artifacts/day10-test-warning-counts-by-file-and-class.txt`

### Test Warning Inventory

Using the Day 8 serialized CMake baseline:

- total test warnings: `98`
- warning-bearing test files: `21`

By warning class:

- `-Wmissing-field-initializers`: `62`
- `-Wdouble-promotion`: `33`
- `-Wunused-function`: `3`

Top warning-bearing files:

- `tests/test_ldlt.c`: `18`
- `tests/test_sprint20_integration.c`: `9`
- `tests/test_chol_csc.c`: `8`
- `tests/test_colamd.c`: `8`
- `tests/test_reorder_nd.c`: `7`

### Day 10 Triage Result

Category 1: positional option-struct drift

- dominant warning class: `-Wmissing-field-initializers`
- main files:
  - `test_ldlt.c`
  - `test_chol_csc.c`
  - `test_colamd.c`
  - `test_cholesky.c`
  - `test_reorder.c`
  - `test_reorder_nd.c`
  - `test_sprint12_integration.c`
  - `test_sprint18_integration.c`
  - `test_sprint19_integration.c`
  - `test_sprint20_integration.c`
- interpretation:
  - maintainability drift around evolving option structs
  - strong candidate for designated-initializer cleanup in a later sprint

Category 2: dormant or non-executed scaffolding

- only file: `test_reorder_nd.c`
- only warning class: `-Wunused-function`
- interpretation:
  - strongest test-honesty issue in the current warning set
  - inactive future-check bodies should not remain as ordinary dead functions in the normal suite

Category 3: incidental numeric-literal precision warnings

- warning class: `-Wdouble-promotion`
- broad but lower-priority spread across:
  - `test_sprint6_integration.c`
  - `test_svd.c`
  - `test_bidiag.c`
  - `test_sprint20_integration.c`
  - smaller one- and two-site files
- interpretation:
  - mostly mechanical fixture/assertion literal cleanup rather than hidden algorithmic defects

### First-Fix Queue For Later Cleanup

Priority A:

- `tests/test_reorder_nd.c`

Priority B:

- `tests/test_ldlt.c`
- `tests/test_chol_csc.c`
- `tests/test_colamd.c`
- `tests/test_cholesky.c`
- `tests/test_sprint12_integration.c`
- `tests/test_sprint18_integration.c`
- `tests/test_sprint19_integration.c`
- `tests/test_sprint20_integration.c`
- `tests/test_reorder.c`

Priority C:

- remaining `-Wdouble-promotion` test files after the higher-signal cleanup groups

### Day 10 Interpretation

- Test warnings are now organized by structural meaning instead of raw count alone.
- The biggest maintainability issue is positional initializer drift.
- The biggest test-honesty issue is dormant ND scaffolding.
- The double-promotion cluster is real cleanup work, but it is lower priority than the first two categories.

### Day 10 Outputs

- `artifacts/day10-test-warning-triage.md`
- `artifacts/day10-test-warning-counts-by-file.txt`
- `artifacts/day10-test-warning-counts-by-file-and-class.txt`

### Day 10 Validation

End-of-day verification passed:

- `make format`
- `make lint`
- `make test`
