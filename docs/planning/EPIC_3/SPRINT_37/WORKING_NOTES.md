# Sprint 37 Working Notes

## Day 1

**Objective:** Turn the Sprint 36 closeout and the Sprint 37 project-plan scope
into a concrete maintainability baseline by confirming the inherited validated
quality contract, inventorying the auxiliary-code surfaces most likely to drive
cleanup cost, and naming the first audit targets for helper consolidation,
quality-target normalization, and large-file cleanup.

### Commands Run

1. Confirm branch and starting state:
   - `git status --short --branch`
   - `git rev-parse --short HEAD`
   - `git branch --show-current`
2. Re-read the Sprint 37 scope and inherited Sprint 36 constraints:
   - `sed -n '262,292p' docs/planning/EPIC_3/PROJECT_PLAN.md`
   - `cat docs/planning/EPIC_3/SPRINT_36/HANDOFF.md`
   - `cat docs/planning/EPIC_3/SPRINT_36/RETROSPECTIVE.md`
   - `sed -n '1,220p' docs/planning/EPIC_3/SPRINT_37/PLAN.md`
3. Inventory current large-file and workflow/helper surfaces:
   - `python3` line-count sweep over `tests/`, `benchmarks/`, `scripts/`,
     `docs/planning/EPIC_3/`, and `.github/workflows/`
   - `wc -l Makefile scripts/*.sh scripts/*.py .github/workflows/*.yml`
   - targeted line-count checks for:
     - `tests/test_chol_csc.c`
     - `tests/test_svd.c`
     - `tests/test_ldlt_csc.c`
     - `tests/test_qr.c`
     - `tests/test_etree.c`
     - `tests/test_iterative.c`
     - `benchmarks/bench_eigs.c`
     - `benchmarks/bench_main.c`
     - `scripts/deadcode_report.py`
     - `scripts/deadcode_workflow.sh`
     - `Makefile`
     - `.github/workflows/ci.yml`
4. Inventory current maintained target surface:
   - `rg -n "^([A-Za-z0-9_./-]+):" Makefile`
5. Sample duplicated helper-pattern hotspots in tests and benchmarks:
   - `rg -n "static .*helper|static .*create_|static .*make_|static .*build_|static .*assert_|static .*load_|static .*setup_" tests benchmarks --glob '*.[ch]'`
   - `python3` helper-name frequency scan over `tests/*.c` and
     `benchmarks/*.c`
6. Reconfirm the inherited maintained suite baseline:
   - `ctest -N --test-dir build/quality-review-cmake`

### Day 1 Findings

#### 1. Sprint 37 starts from a validated auxiliary baseline, not a regression queue

Sprint 37 inherits the Sprint 36 close state exactly as intended:

- direct maintained gates were green at handoff:
  - `make format`
  - `make lint`
  - `make test`
- reviewed wrapper paths were green at handoff:
  - `make quality-review-compile`
  - `make quality-review`
  - `make quality-review-cmake-compile`
  - `make quality-review-cmake`
- supporting maintained paths were also green at handoff:
  - `make wall-check`
  - `make deadcode-report`
  - `make deadcode-check`
  - `make sanitize`
- the active CMake suite baseline remains:
  - `ctest -N --test-dir build/quality-review-cmake` = `53`

Interpretation:

- Sprint 37 is not a warning-debt sprint
- Sprint 37 is not a parity-definition sprint
- Sprint 37 is a maintainability and ownership sprint layered on top of a
  validated quality contract

#### 2. The auxiliary maintainability load is concentrated in tests, Makefile workflow plumbing, and dead-code tooling

The largest non-core surfaces at Day 1 are heavily skewed toward auxiliary
maintenance code:

- test tree:
  - `54` `.c` files
  - `62,005` total lines
- benchmark tree:
  - `14` `.c` files
  - `5,170` total lines
- largest test files:
  - `tests/test_chol_csc.c` = `4,643`
  - `tests/test_svd.c` = `3,712`
  - `tests/test_ldlt_csc.c` = `3,637`
  - `tests/test_qr.c` = `3,259`
  - `tests/test_etree.c` = `2,890`
  - `tests/test_iterative.c` = `2,819`
- largest benchmark/helper surfaces:
  - `benchmarks/bench_eigs.c` = `958`
  - `benchmarks/bench_main.c` = `774`
- largest workflow/helper files:
  - `Makefile` = `812`
  - `scripts/deadcode_report.py` = `472`
  - `scripts/deadcode_workflow.sh` = `189`
  - `.github/workflows/ci.yml` = `231`

Interpretation:

- the Sprint 37 large-file pass should focus on auxiliary maintenance hotspots,
  not core numerical kernels
- the highest-value one-or-two-file targets are likely to come from the test
  layer plus `Makefile` / dead-code workflow plumbing

#### 3. The current quality-target surface is rich but structurally busy

The current maintained target layout already exposes many named entry points:

- direct/build surfaces:
  - `all`
  - `examples-build`
  - `examples`
  - `smoke`
  - `test`
  - `bench`
  - `bench-build`
  - `tooling-build`
  - `bench-fast`
- sanitizer/openmp/platform-support surfaces:
  - `sanitize`
  - `asan`
  - `sanitize-all`
  - `omp`
  - `tsan`
  - `sanitize-thread`
- formatting/lint/reviewed quality surfaces:
  - `format`
  - `format-check`
  - `lint`
  - `check`
  - `quality-review-compile`
  - `quality-review`
  - `quality-review-cmake-compile`
  - `quality-review-cmake`
- reporting/dead-code/coverage surfaces:
  - `warning-workflow`
  - `deadcode-compile-db`
  - `deadcode`
  - `deadcode-report`
  - `deadcode-check`
  - `wall-check`
  - `coverage`
  - `coverage-lcov`
  - `coverage-gcovr`

Interpretation:

- Sprint 37 does not need new quality targets first
- it needs clearer ownership and normalization of an already large target graph
- Sprint 36's sanitizer caveat belongs directly in this normalization work

#### 4. The sanitizer/build-tree caveat is a real maintainability concern, not just a Day 13 footnote

Sprint 36 handed off one operational constraint that belongs squarely inside
Sprint 37:

- a prior `make sanitize` run can leave an instrumented `build/` tree behind
- a later direct or reviewed validation sweep may then fail unless the tree is
  cleaned first

Interpretation:

- target normalization should treat this as a first-class workflow problem
- the fix surface is probably a mix of:
  - Makefile target organization
  - explicit clean-build expectations in maintainer docs
  - tighter naming/ownership of direct vs reviewed paths

#### 5. Helper duplication is visible already in both tests and benchmarks

The Day 1 helper sweep was intentionally shallow, but it already exposed likely
Sprint 37 consolidation candidates:

- repeated matrix-construction helpers across tests:
  - `build_spd_tridiag`
  - `make_tridiag`
  - `build_kkt`
  - `make_kkt`
  - `build_identity`
  - `make_identity`
- repeated benchmark-side synthetic matrix builders:
  - `make_tridiag`
  - `make_dense_*`
  - `build_*_spd`
  - `load_or_null`
- repeated small utility/helper names across tests and benchmarks:
  - `wall_time`
  - `relative_residual`
  - `compute_rel_residual`
  - `run_one`
  - `make_jacobi`
  - `is_valid_perm` / `is_valid_permutation`

Interpretation:

- Sprint 37 should not assume these all belong in a single shared helper layer
- but Day 2 and Day 3 have a strong real audit surface instead of a vague
  “look for duplication” task

#### 6. The first implementation surfaces are already clear

Highest-value Sprint 37 files at Day 1:

- test/helper-heavy auxiliary files:
  - `tests/test_chol_csc.c`
  - `tests/test_svd.c`
  - `tests/test_ldlt_csc.c`
  - `tests/test_qr.c`
  - `tests/test_etree.c`
  - `tests/test_iterative.c`
- benchmark/helper-heavy files:
  - `benchmarks/bench_eigs.c`
  - `benchmarks/bench_main.c`
  - `benchmarks/bench_chol_csc.c`
  - `benchmarks/bench_ldlt_csc.c`
- quality/workflow plumbing:
  - `Makefile`
  - `scripts/deadcode_report.py`
  - `scripts/deadcode_workflow.sh`
  - `.github/workflows/ci.yml`

Likely first audit split:

- Day 2:
  - test-helper consolidation
- Day 3:
  - benchmark-helper consolidation
- Day 4:
  - quality-target normalization design

### Day 1 Interpretation

- Sprint 37 starts from a validated quality and parity baseline, so the real
  job is to reduce future cleanup cost in the non-core layers.
- The biggest maintainability surfaces are already obvious: very large test
  files, dense Makefile workflow plumbing, and dead-code scripts that now carry
  a large share of repo-quality behavior.
- The Day 1 inventory supports a bounded maintainability sprint:
  - helper consolidation where it improves ownership
  - target normalization where it reduces operator ambiguity
  - one or two large-file refactors where they materially improve locality

### Day 1 Outputs

- `artifacts/day1-auxiliary-maintainability-baseline.md`
- `artifacts/day1-auxiliary-surface-inventory.txt`

## Day 2

**Objective:** Audit the maintained test tree for real helper duplication,
separate true consolidation candidates from intentionally local helpers, and
define a bounded first cleanup queue that preserves the Sprint 32
truthfulness/opt-in test contract.

### Commands Run

1. Re-read the Day 2 plan and the Day 1 baseline:
   - `git status --short --branch`
   - `git rev-parse --short HEAD`
   - `sed -n '39,68p' docs/planning/EPIC_3/SPRINT_37/PLAN.md`
   - `cat docs/planning/EPIC_3/SPRINT_37/artifacts/day1-auxiliary-maintainability-baseline.md`
2. Measure test-side helper density:
   - `python3` static-function count sweep across `tests/*.c`
3. Identify repeated helper names and their file spread:
   - `python3` helper-name frequency scan across `tests/*.c`
   - `rg -n "static .*build_spd_tridiag|static .*make_spd_tridiag|static .*make_tridiag|static .*build_kkt|static .*make_kkt|static .*build_identity|static .*make_identity|static .*make_jacobi|static .*compute_relative_residual|static .*relative_residual|static .*compute_rel_residual" tests --glob '*.c'`
   - `python3` cluster summary for:
     - SPD/tridiagonal builders
     - KKT builders
     - identity builders
     - Jacobi/precondition helpers
     - residual helpers
4. Re-check the current shared test-support layer:
   - `ls tests`
   - `sed -n '1,260p' tests/test_framework.h`
   - `rg -n "#include \\\"test_.*\\.h\\\"|#include \\\".*_test.*\\.h\\\"|#include \\\"test_framework.h\\\"" tests --glob '*.[ch]'`
5. Reconfirm the Sprint 32 truthfulness floor:
   - `rg -n "^[[:space:]]*//.*RUN_TEST|^[[:space:]]*/\\*.*RUN_TEST" tests --glob '*.c'`
6. Re-check the build model that constrains shared-helper landing options:
   - `sed -n '1,180p' Makefile`
   - `sed -n '180,260p' CMakeLists.txt`
   - `rg -n "TEST_SRCS|TEST_BINS|smoke_test|test_framework_optin" Makefile CMakeLists.txt`

### Day 2 Findings

#### 1. The test tree has a real helper-duplication queue, but it is concentrated rather than uniform

The largest helper-heavy tests are clearly carrying disproportionate local
support logic:

- `tests/test_chol_csc.c`
  - `154` static functions
  - `4,643` lines
- `tests/test_ldlt_csc.c`
  - `115` static functions
  - `3,637` lines
- `tests/test_etree.c`
  - `101` static functions
  - `2,890` lines
- `tests/test_svd.c`
  - `99` static functions
  - `3,712` lines
- `tests/test_ldlt.c`
  - `85` static functions
  - `2,774` lines
- `tests/test_iterative.c`
  - `82` static functions
  - `2,819` lines

Interpretation:

- Sprint 37 should not treat the whole test tree as equally ripe for helper
  extraction
- the best consolidation yield is in large files and integration clusters where
  repeated support logic is already obvious

#### 2. The strongest repeated helper families are matrix builders and residual calculators

The clearest repeated clusters from the Day 2 scan:

- SPD/tridiagonal builders: `14` hits
  - spread across solver, preconditioner, and integration tests
- KKT builders: `7` hits
  - spread across LDLT, MINRES, eigs, and integration tests
- identity builders: `3` hits
  - smaller cluster, lower urgency
- Jacobi/precondition helper family: `8` hits
  - limited to a few iterative/integration surfaces
- residual helper family: `26` hits
  - strongest semantic duplication signal in the audit

Representative repeated surfaces:

- `build_spd_tridiag` / `make_spd_tridiag`
  - `test_iterative.c`
  - `test_bicgstab.c`
  - `test_ilu.c`
  - `test_ic.c`
  - `test_minres.c`
  - several `test_sprint*_integration.c` files
- `build_kkt` / `make_kkt`
  - `test_minres.c`
  - `test_ldlt.c`
  - `test_eigs.c`
  - `test_sprint12_integration.c`
  - `test_sprint13_integration.c`
  - `test_sprint18_integration.c`
  - `test_sprint20_integration.c`
- residual helpers
  - `compute_relative_residual`
  - `relative_residual`
  - `compute_rel_residual`
  - `rel_residual`

Interpretation:

- Sprint 37 has real consolidation targets instead of just naming-style drift
- matrix-builder and residual helpers are the first serious batch candidates

#### 3. The current shared test-support layer is intentionally minimal

At Day 2, the only pervasive shared test-support file is still:

- `tests/test_framework.h`

There is not an existing general-purpose `test_helpers.*`,
`test_matrix_builders.*`, or assertion-support library already in use across
the test tree.

Interpretation:

- the test tree has intentionally avoided a large shared helper substrate so far
- Sprint 37 should therefore add shared test helpers only where the ownership
  win is clear
- broad “frameworkization” would likely hurt locality more than it helps

#### 4. The test build model sharply constrains where shared helpers can land

Current build model:

- Makefile:
  - each test binary is built from its own single `tests/test_*.c` file plus
    the library
- CMake:
  - each test is added explicitly via `add_sparse_test(...)`

Interpretation:

- the safest consolidation landing options are:
  - small shared headers with `static inline` or macro helpers
  - tightly-scoped helper headers for related test clusters
  - local helper renaming/sectioning inside very large files
- the riskiest first move would be a broad new multi-file test-support `.c`
  layer that requires touching every build surface immediately

#### 5. Sprint 32 truthfulness constraints remain intact and must stay non-negotiable

Day 2 re-checks confirmed:

- no commented-out `RUN_TEST(...)` registrations remain in `tests/*.c`
- `test_framework.h` remains the live shared policy layer for:
  - `RUN_TEST`
  - `RUN_TEST_SLOW`
  - `RUN_TEST_EXPERIMENTAL`
  - `SKIP_TEST`

Interpretation:

- helper consolidation must not blur the line between active tests, opt-in
  tests, and historical evidence
- Sprint 37 should not centralize test registration behavior beyond the
  existing `test_framework.h` contract

#### 6. The first consolidation batch should stay narrow and cluster-based

The highest-value Day 2 cleanup order is now clear:

Priority A:

- residual helper consolidation across iterative/integration clusters
- likely files:
  - `tests/test_iterative.c`
  - `tests/test_bicgstab.c`
  - `tests/test_ilu.c`
  - `tests/test_minres.c`
  - selected `test_sprint*_integration.c` files

Priority B:

- SPD/tridiagonal builder consolidation across iterative/preconditioner tests
- likely files:
  - `tests/test_iterative.c`
  - `tests/test_bicgstab.c`
  - `tests/test_ilu.c`
  - `tests/test_ic.c`
  - `tests/test_minres.c`

Priority C:

- KKT builder consolidation inside the LDLT/MINRES/integration family

Keep local for now:

- heavy file-specific assertion helpers in:
  - `tests/test_chol_csc.c`
  - `tests/test_ldlt_csc.c`
  - `tests/test_svd.c`
  - `tests/test_qr.c`
- these are large, but most of their support logic is tightly coupled to local
  factorization structures and invariants

### Day 2 Interpretation

- Sprint 37 does have a meaningful test-helper consolidation queue, but the
  right shape is small cluster-based extraction, not a generic global helper
  layer.
- The first real payoff is in repeated matrix builders and residual helpers
  across iterative and integration tests.
- The largest test files matter for the sprint, but not because all of their
  helpers should become shared. Many of their assertions are still local by
  design and should stay that way.

### Day 2 Outputs

- `artifacts/day2-test-helper-consolidation-audit.md`
