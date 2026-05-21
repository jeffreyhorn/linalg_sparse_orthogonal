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

## Day 3

**Objective:** Audit the maintained benchmark tree for duplicated setup,
timing, reporting, and synthetic-fixture logic, separate true shared-helper
opportunities from benchmark-local behavior, and define a bounded first cleanup
queue that preserves the Sprint 31 benchmark contract.

### Commands Run

1. Re-read the Day 3 plan and the Day 1 baseline:
   - `git status --short --branch`
   - `git rev-parse --short HEAD`
   - `sed -n '69,98p' docs/planning/EPIC_3/SPRINT_37/PLAN.md`
   - `cat docs/planning/EPIC_3/SPRINT_37/artifacts/day1-auxiliary-maintainability-baseline.md`
2. Measure benchmark-side helper density:
   - `python3` static-function count sweep across `benchmarks/*.c`
3. Identify repeated helper names and their file spread:
   - `python3` helper-name frequency scan across `benchmarks/*.c`
4. Inspect the strongest candidate benchmark files directly:
   - `sed -n '1,220p' benchmarks/bench_chol_csc.c`
   - `sed -n '1,220p' benchmarks/bench_ldlt_csc.c`
   - `sed -n '1,220p' benchmarks/bench_main.c`
   - `sed -n '1,220p' benchmarks/bench_reorder.c`
5. Re-check the current benchmark build/support layer:
   - `ls benchmarks`
   - `sed -n '1,220p' benchmarks/README.md`
   - `rg -n "BENCH_SRCS|BENCH_BINS|bench-fast|bench-build|add_executable\\(bench_|NOT WIN32" Makefile CMakeLists.txt`
   - `rg -n "#include \\\".*bench.*\\.h\\\"|typedef struct \\{|static const .*kFixtures|static const .*kReorderings|Usage:" benchmarks --glob '*.[ch]'`

### Day 3 Findings

#### 1. The benchmark-helper queue is smaller than the test queue, but more structurally repetitive

The benchmark tree is much smaller than the test tree:

- `14` `.c` files
- `5,170` total lines

But the helper density is still meaningful, especially in a few maintained
drivers:

- `benchmarks/bench_eigs.c`
  - `19` static functions
  - `958` lines
- `benchmarks/bench_chol_csc.c`
  - `10` static functions
  - `446` lines
- `benchmarks/bench_main.c`
  - `9` static functions
  - `774` lines
- `benchmarks/bench_ldlt_csc.c`
  - `9` static functions
  - `579` lines
- `benchmarks/bench_amd_qg.c`
  - `9` static functions
  - `332` lines
- `benchmarks/bench_reorder.c`
  - `8` static functions
  - `321` lines

Interpretation:

- Sprint 37's benchmark work is not about size pressure alone
- it is about a few structurally similar harnesses that are now expensive to
  keep behaviorally aligned

#### 2. The strongest repeated helper families are timer, residual, and backend-comparison harness helpers

The clearest repeated helper clusters from the Day 3 scan:

- wall-clock timers: `11` hits
  - almost every maintained benchmark defines its own `wall_time()`
- residual helpers:
  - `rel_residual`: `3` hits
  - `compute_rel_residual`: `2` hits
- backend comparison helpers:
  - `bench_matrix`: `4` hits
  - `bench_csc_path`: `2` hits
  - `bench_linked_list`: `2` hits
- smaller repeated utility helpers:
  - `run_one`: `3` hits
  - `ends_with`: `2` hits
  - `symbolic_nnz_L`: `2` hits
  - `now_ms`: `2` hits

Interpretation:

- the queue is not generic “all benchmark helpers”
- it is a few repeated harness shapes plus a ubiquitous timer helper

#### 3. `bench_chol_csc.c` and `bench_ldlt_csc.c` are the strongest first consolidation pair

These two files are nearly mirror-image backend-comparison harnesses:

- same `_POSIX_C_SOURCE 200809L` portability baseline
- same local `wall_time()` helper shape
- same relative-residual helper contract
- same `bench_result_t` pattern
- same linked-list vs CSC backend-comparison structure
- same `bench_matrix` / dispatch-style harness naming

Interpretation:

- this pair is the cleanest first extraction target in the whole benchmark
  tree
- helper landing options can stay narrow and benchmark-local in spirit while
  still reducing real duplication

#### 4. `bench_main.c` and `bench_reorder.c` repeat smaller utility patterns, but much of their behavior should stay local

Shared-looking pieces:

- timer helpers (`wall_time` / `now_ms`)
- path/filter helpers like `ends_with`
- per-matrix runner structure (`benchmark*` / `run_one*`)
- reorder/reporting utilities

Reasons not to over-centralize them immediately:

- `bench_main.c` is the broad solver harness with several modes:
  - LU / `--cholesky`
  - SpMV
  - iterative
  - directory sweeps
- `bench_reorder.c` is a specialized reorder/factor comparison tool with:
  - fixed reorder table
  - fixed fixture table
  - analyze-vs-preapply path split

Interpretation:

- there is some real small-helper duplication
- but the high-risk part of these files is behavior breadth, not raw helper
  count
- Day 3 does not justify a first batch that merges their CLI/reporting logic

#### 5. The current benchmark support layer is intentionally minimal

At Day 3, there is still no shared benchmark support header or source layer:

- no `bench_helpers.h`
- no `bench_common.c`
- no shared fixture/timing/reporting helper library

Current structure is:

- standalone benchmark executables
- README-level contract documentation in `benchmarks/README.md`
- explicit Makefile and CMake executable lists

Interpretation:

- the benchmark tree resembles the test tree in one important way:
  shared support should be added only where the ownership win is obvious
- the audit does not justify broad benchmark framework extraction as a first
  move

#### 6. The benchmark build model constrains the safest landing shapes

Current build model:

- Makefile:
  - explicit `BENCH_SRCS`
  - one binary per benchmark source
- CMake:
  - individual `add_executable(bench_...)` entries
  - several benchmark binaries gated out on Windows via `NOT WIN32`

Interpretation:

- safest landing options are:
  - small shared benchmark headers
  - tightly scoped helper extraction for benchmark clusters
  - local helper cleanup and sectioning inside the largest harnesses
- riskiest first move is a broad new benchmark helper `.c` library that
  requires touching every bench build path immediately

#### 7. Sprint 31 benchmark-contract constraints remain load-bearing

The current README and file shapes still encode the Sprint 31 benchmark split:

- `bench_main --reorder none|rcm|amd|nd`
- `bench_reorder` owns the broader `none|rcm|amd|colamd|nd` comparison surface
- `bench_colamd` and `example_colamd` remain the QR/COLAMD-focused comparison
  tools
- `bench_chol_csc` and `bench_ldlt_csc` are backend-comparison tools, not
  general reorder sweep tools

Interpretation:

- consolidation must not blur reorder-surface ownership
- shared helpers are fine for timer/result plumbing
- shared CLI or reorder-surface logic needs much stronger justification

#### 8. The first consolidation batch should stay narrow and cluster-based

Highest-value Day 3 cleanup order:

Priority A:

- `bench_chol_csc.c` + `bench_ldlt_csc.c`
  - timer helper
  - residual helper
  - `bench_result_t`-style comparison harness helpers
  - benchmark-matrix dispatch plumbing

Priority B:

- light utility cleanup around:
  - `wall_time` / `now_ms`
  - simple path/filter helpers like `ends_with`
  - only if this can be done without obscuring the mode-specific harnesses

Keep local for now:

- `bench_main.c` CLI breadth and mode dispatch
- `bench_reorder.c` fixture/reorder table ownership
- `bench_eigs.c` backend-specific benchmark logic
- `bench_amd_qg.c` bitset-specific local helpers

### Day 3 Interpretation

- Sprint 37 does have a meaningful benchmark-helper consolidation queue, but it
  is smaller and more structurally obvious than the test queue.
- The best first win is the nearly parallel `bench_chol_csc` /
  `bench_ldlt_csc` harness pair.
- `bench_main.c` and `bench_reorder.c` should be treated as behavior-rich
  owners of their own CLI/reporting contracts, with only small utility cleanup
  considered early.

### Day 3 Outputs

- `artifacts/day3-benchmark-helper-consolidation-audit.md`

## Day 4

**Objective:** Define a clearer quality-target ownership model before changing
the Makefile layout, with explicit separation between maintained operator entry
points, helper plumbing, and tree-mutating instrumentation flows, while folding
in the Sprint 36 sanitizer/build-tree caveat as a first-class design
constraint.

### Commands Run

1. Re-read the Day 4 plan and inherited Sprint 36 constraint:
   - `git status --short --branch`
   - `git rev-parse --short HEAD`
   - `sed -n '99,128p' docs/planning/EPIC_3/SPRINT_37/PLAN.md`
   - `cat docs/planning/EPIC_3/SPRINT_36/HANDOFF.md`
2. Re-read the current quality-target surface in the Makefile:
   - `sed -n '380,760p' Makefile`
3. Re-read the current operator-facing quality command map:
   - `sed -n '680,760p' README.md`
4. Inventory tree-mutating target behavior and `clean` usage:
   - `rg -n "^(sanitize|asan|sanitize-all|omp|tsan|coverage-lcov|coverage-gcovr):|clean test|clean \\$\\(TEST_BINS\\)" Makefile`
5. Inventory the current target entry points by name:
   - `python3` target-location summary for:
     - direct gates
     - reviewed wrappers
     - dead-code targets
     - sanitizer/coverage targets
     - build helpers
     - `clean`

### Day 4 Findings

#### 1. The repo already has the needed quality targets; the problem is ownership clarity

Current maintained quality/validation surfaces already cover the needed
behaviors:

- direct atomic gates:
  - `format`
  - `format-check`
  - `lint`
  - `test`
  - `check`
- reviewed wrappers:
  - `quality-review-compile`
  - `quality-review`
  - `quality-review-cmake-compile`
  - `quality-review-cmake`
- dead-code surfaces:
  - `deadcode-compile-db`
  - `deadcode`
  - `deadcode-report`
  - `deadcode-check`
- supporting validation helpers:
  - `wall-check`
  - `warning-workflow`
- instrumentation / alternate-build flows:
  - `sanitize`
  - `asan`
  - `sanitize-all`
  - `tsan`
  - `omp`
  - `coverage`
  - `coverage-lcov`
  - `coverage-gcovr`
- build plumbing:
  - `bench-build`
  - `examples-build`
  - `tooling-build`
  - `clean`

Interpretation:

- Sprint 37 does not need more entry points first
- it needs a clearer target topology and clearer operator ownership rules

#### 2. The correct normalization split is operator entry points vs helper plumbing vs tree-mutating modes

The Day 4 design split is:

Category A — maintained operator entry points

- atomic direct gates:
  - `format`
  - `format-check`
  - `lint`
  - `test`
  - `check`
- reviewed local/CMake wrappers:
  - `quality-review-compile`
  - `quality-review`
  - `quality-review-cmake-compile`
  - `quality-review-cmake`
- maintained specialized validation/report surfaces:
  - `deadcode-report`
  - `deadcode-check`
  - `wall-check`

Category B — helper / prerequisite plumbing

- `bench-build`
- `examples-build`
- `tooling-build`
- `deadcode-compile-db`
- `deadcode`
- `warning-workflow`

Category C — tree-mutating instrumentation or alternate-build modes

- `sanitize`
- `asan`
- `sanitize-all`
- `tsan`
- `omp`
- `coverage`
- `coverage-lcov`
- `coverage-gcovr`

Interpretation:

- target normalization should make these categories more obvious in comments,
  layout, and docs
- helper targets should not read like peer operator entry points
- tree-mutating modes should be called out as special modes, not ordinary
  neighbors of the stable reviewed/direct gates

#### 3. The Sprint 36 sanitizer/build-tree caveat generalizes to a broader “tree-mutating mode” rule

Current Makefile behavior already shows the pattern:

- `sanitize`, `asan`, `sanitize-all`, `omp`, and `tsan` all run through
  `clean test`
- `coverage-lcov` and `coverage-gcovr` both run through `clean $(TEST_BINS)`

Interpretation:

- the repo already treats these as alternate build modes that cannot safely
  reuse the normal object tree without reset
- Sprint 36's UBSan-aftereffects issue is therefore not isolated to `sanitize`
  alone
- the correct normalization rule is:
  - tree-mutating instrumentation/build-mode targets own a clean-tree reset on
    entry
  - maintainers should treat `clean` as the canonical reset when returning from
    those modes to the normal direct/reviewed path

#### 4. Reviewed wrappers and direct gates should stay semantically stable

The Day 4 design keeps these meanings unchanged:

- `lint` remains the direct compile/static-analysis gate
- `test` remains the direct runtime gate
- `check` remains the simple legacy direct aggregate
- `quality-review-compile` remains the reviewed local compile-quality wrapper
- `quality-review` remains the reviewed local end-to-end wrapper
- `quality-review-cmake-compile` remains the reviewed CMake parity wrapper
- `quality-review-cmake` remains the full reviewed CMake execution wrapper
- `deadcode-check` remains a sibling quality category, not part of the
  warning-clean definition itself

Interpretation:

- normalization should improve readability and safety
- it should not silently repurpose Sprint 34's reviewed contract

#### 5. The simplest safe sanitizer-caveat fix is to normalize around the existing `clean` target, not invent a parallel reset name

Day 4 decision:

- do **not** introduce a second alias like `quality-clean-build` or
  `reset-build-tree` in the first normalization pass
- keep `clean` as the single canonical tree reset
- instead:
  - make tree-mutating target ownership explicit
  - make reviewed/direct operator docs point back to `make clean` when
    returning from instrumentation or alternate-build modes
  - tighten comments and banners so this is hard to miss

Why:

- a new alias would increase target count without adding new behavior
- the maintainability problem is ambiguity, not lack of a reset command
- the existing `clean` target is already understood and widely referenced

#### 6. The next implementation batch should be narrow and comment/layout focused first

Highest-value Day 7 implementation order:

Priority A:

- normalize Makefile comments and section layout around the three target
  classes:
  - operator entry points
  - helper plumbing
  - tree-mutating modes

Priority B:

- make the tree-mutating-mode rule explicit in:
  - target comments
  - reviewed wrapper/operator docs
  - maintainer workflow docs

Priority C:

- reduce low-value naming or placement ambiguity where it helps:
  - keep helper targets physically near the operator targets they support
  - keep dead-code helper/report/check layering easy to see

Avoid in the first batch:

- renaming established Sprint 34 or Sprint 36 targets
- merging dead-code into reviewed warning-clean semantics
- claiming fake Windows Makefile parity or fake dead-code universality

### Day 4 Interpretation

- the repo's problem is not missing targets, but a crowded flat quality surface
  with too little ownership signaling
- the right normalization contract is a three-way split:
  - stable operator entry points
  - helper plumbing
  - tree-mutating alternate-build modes
- the Sprint 36 sanitizer caveat should be treated as one instance of the
  broader tree-mutating-mode rule, with `clean` remaining the canonical reset

### Day 4 Outputs

- `artifacts/day4-quality-target-normalization-design.md`

## Day 5

**Objective:** Convert the Day 2 audit into the safest first test-helper
consolidation slice by extracting the repeated L2 residual helpers used across
the iterative/preconditioner/integration cluster, while keeping the landing
shape narrow, header-only, and easy to review.

### Commands Run

1. Re-read the Day 5 plan and the Day 2 helper audit:
   - `git status --short --branch`
   - `sed -n '114,145p' docs/planning/EPIC_3/SPRINT_37/PLAN.md`
   - `cat docs/planning/EPIC_3/SPRINT_37/artifacts/day2-test-helper-consolidation-audit.md`
2. Re-read the candidate test files from the residual-helper cluster:
   - `rg -n "compute_relative_residual|relative_residual|block_relative_residual|local_norm2|compute_rel_residual|rel_residual" tests`
   - `sed -n` spot reads in:
     - `tests/test_iterative.c`
     - `tests/test_bicgstab.c`
     - `tests/test_ilu.c`
     - `tests/test_minres.c`
     - `tests/test_sprint5_integration.c`
     - `tests/test_sprint10_integration.c`
     - `tests/test_sprint12_integration.c`
     - `tests/test_sprint13_integration.c`
3. Implement the narrow shared-helper batch:
   - add `tests/test_solver_helpers.h`
   - replace duplicated local residual helpers in the selected cluster files
4. Validate the touched batch:
   - `make format`
   - `make lint`
   - `make test`
5. Fix the one fallout from the first `make test` run:
   - `tests/test_sprint10_integration.c` still referenced removed
     `local_norm2(...)`
   - replace it with `tf_vec_norm2(...)`
6. Re-run the required full gate:
   - `make format`
   - `make lint`
   - `make test`

### Day 5 Findings

#### 1. The safest first shared landing shape was a small header-only solver helper layer

Day 2 identified several real duplication families, but the test build model
still makes a broad shared `.c` support layer a poor first move.

The batch therefore landed as:

- new header:
  - `tests/test_solver_helpers.h`
- intentionally narrow helper surface:
  - `tf_vec_norm2(...)`
  - `tf_relative_residual_l2(...)`
  - `tf_block_relative_residual_l2(...)`

Why this shape was the right first batch:

- each touched test binary still compiles from its own test source
- the shared logic is pure helper code with no separate link-time ownership
- behavior stays explicit at the call site
- the new layer is small enough that it does not read like a new generic test
  framework

#### 2. The clearest residual-helper duplication is now closed for the first cluster

The Day 5 batch removed duplicated local L2 residual helpers from:

- `tests/test_iterative.c`
- `tests/test_bicgstab.c`
- `tests/test_ilu.c`
- `tests/test_minres.c`
- `tests/test_sprint5_integration.c`
- `tests/test_sprint10_integration.c`
- `tests/test_sprint12_integration.c`
- `tests/test_sprint13_integration.c`

That included:

- single-RHS residual helpers
- one block-RHS residual helper
- one local vector-norm helper that only existed to support the residual math

Important preservation detail:

- the new helper takes an explicit allocation-failure sentinel so each file can
  preserve its previous failure semantics
- `tests/test_iterative.c` kept its existing `-1.0` sentinel behavior
- the other touched files kept `HUGE_VAL` behavior where that was already the
  local convention

#### 3. The batch stayed narrow and did not overreach into other audit families

Day 5 deliberately did **not** try to absorb the full Day 2 queue.

Still deferred after this batch:

- SPD / tridiagonal matrix builders
- KKT builders
- file-specific residual helpers in other clusters such as:
  - `tests/test_qr.c`
  - `tests/test_ldlt_csc.c`
  - `tests/test_sprint19_integration.c`

Reason:

- those surfaces either carry more file-local semantics or belong to different
  solver families
- forcing them into the first batch would have widened the review surface
  beyond the Day 5 “narrow batch” contract

#### 4. One real fallout appeared immediately and was fixed before the authoritative rerun

The first `make test` run found one legitimate cleanup fallout:

- `tests/test_sprint10_integration.c` still used removed helper
  `local_norm2(...)` in its CSR-vs-linked-list comparison

Fix:

- replace that remaining use with `tf_vec_norm2(...)`

Interpretation:

- this validates the Day 5 approach of doing the narrow extraction with a full
  rerun immediately
- the fallout was mechanical and local, not a design failure in the shared
  helper contract

#### 5. The required code-quality gate passed after the local cleanup

Because this batch changed test `*.c` and `*.h` files, the required gate was:

- `make format`
- `make lint`
- `make test`

Authoritative result after the one local cleanup:

- `make format`: passed
- `make lint`: passed
- `make test`: passed

This keeps Sprint 37 on the validated Sprint 36 / Sprint 35 baseline rather
than opening a new regression queue.

### Day 5 Interpretation

- Day 5 converted the Day 2 audit into a real first consolidation batch
  without creating a broad new shared test framework
- the right early consolidation surface was the repeated residual math cluster,
  not builders or broader support scaffolding
- the remaining helper queue is now narrower and better partitioned:
  - residual helpers outside this cluster
  - SPD/tridiagonal builders
  - KKT builders

### Day 5 Outputs

- `tests/test_solver_helpers.h`
- `artifacts/day5-test-helper-consolidation-batch1.md`

## Day 6

**Objective:** Convert the Day 3 benchmark-helper audit into the safest first
benchmark consolidation slice by extracting the repeated backend-comparison
support logic shared by `bench_chol_csc.c` and `bench_ldlt_csc.c`, while
keeping the landing shape header-only and preserving the Sprint 31 benchmark
behavior contract.

### Commands Run

1. Re-read the Day 6 plan and the Day 3 benchmark-helper audit:
   - `git status --short --branch`
   - `sed -n '146,176p' docs/planning/EPIC_3/SPRINT_37/PLAN.md`
   - `cat docs/planning/EPIC_3/SPRINT_37/artifacts/day3-benchmark-helper-consolidation-audit.md`
2. Re-read the backend-comparison benchmark pair:
   - `sed -n '1,260p' benchmarks/bench_chol_csc.c`
   - `sed -n '1,360p' benchmarks/bench_ldlt_csc.c`
   - targeted `rg -n` scans for:
     - `wall_time`
     - `rel_residual`
     - `bench_result_t`
     - `bench_matrix`
     - dispatch helpers
3. Implement the narrow shared-helper batch:
   - add `benchmarks/bench_backend_compare_helpers.h`
   - replace repeated timer / residual / matrix-load / unit-RHS setup logic in:
     - `benchmarks/bench_chol_csc.c`
     - `benchmarks/bench_ldlt_csc.c`
4. Validate touched benchmark paths directly:
   - `make format`
   - `make build/bench_chol_csc build/bench_ldlt_csc`
   - `./build/bench_chol_csc --small-corpus --repeat 1`
   - `./build/bench_ldlt_csc tests/data/suitesparse/nos4.mtx --repeat 1`
   - `./build/bench_ldlt_csc --dispatch --repeat 1`
5. Run the required full gate because benchmark `*.c` and helper `*.h` changed:
   - `make format`
   - `make lint`
   - `make test`

### Day 6 Findings

#### 1. The safest first benchmark consolidation surface really was the backend-comparison pair

The Day 3 audit held up under implementation review:

- `benchmarks/bench_chol_csc.c`
- `benchmarks/bench_ldlt_csc.c`

shared the cleanest low-risk support layer:

- wall-clock timer helper
- max-relative-residual checker
- file-backed matrix load + display-label extraction
- repeated `b = A * 1` / workspace allocation setup
- identical factor/solve/residual status result struct shape

This was a better first batch than:

- timer-only repo-wide extraction
- CLI/report formatting unification
- broader helper moves touching `bench_main.c`, `bench_reorder.c`, or
  `bench_eigs.c`

#### 2. The right landing shape was a pair-scoped header-only helper, not a broad benchmark framework

Day 6 landed as:

- new header:
  - `benchmarks/bench_backend_compare_helpers.h`

with intentionally narrow support:

- `bench_backend_result_t`
- `bench_backend_wall_time(...)`
- `bench_backend_rel_residual_max(...)`
- `bench_backend_load_matrix(...)`
- `bench_backend_make_unit_rhs(...)`

Why this shape was the right first batch:

- benchmarks still build one binary per source in both Makefile and CMake
- the extracted logic is pure support code with no new link-time ownership
- the helper layer is scoped to the backend-comparison pair instead of reading
  like a new generic benchmark subsystem
- the larger LDLT dispatch and Cholesky small-corpus logic stay local where
  their behavior contracts still differ

#### 3. The batch removed real duplication without changing benchmark semantics or CLI surfaces

The implementation reduced duplicated support logic in:

- `benchmarks/bench_chol_csc.c`
- `benchmarks/bench_ldlt_csc.c`

Shared now:

- timer helper
- residual helper
- path loading + CSV label extraction
- `b = A * [1, ..., 1]` workspace setup
- common factor/solve result struct

Still local by design:

- Cholesky scalar vs supernodal elimination path definitions
- LDLT native vs wrapper vs supernodal kernel timing logic
- LDLT `--dispatch` benchmark mode
- Cholesky synthetic-corpus builders
- each benchmark's CSV schema and CLI contract

Interpretation:

- the batch reduced maintenance burden where the pair truly matched
- it did not overreach into behavior-owner code paths

#### 4. Direct benchmark validation stayed aligned with the existing Sprint 31 benchmark contract

Touched benchmark validation passed directly:

- `./build/bench_chol_csc --small-corpus --repeat 1`
- `./build/bench_ldlt_csc tests/data/suitesparse/nos4.mtx --repeat 1`
- `./build/bench_ldlt_csc --dispatch --repeat 1`

That confirmed the Day 6 batch did not disturb:

- the fixed backend-comparison role of `bench_chol_csc`
- the fixed backend-comparison + dispatch role of `bench_ldlt_csc`
- the existing CSV/output surfaces those tools expose

The Sprint 31 benchmark behavior split also remained intact because this batch
did not touch:

- `bench_main`
- `bench_reorder`
- `bench_colamd`

#### 5. The required full code-quality gate passed

Because this batch changed benchmark `*.c` and helper `*.h` files, the
required gate was:

- `make format`
- `make lint`
- `make test`

Authoritative result:

- `make format`: passed
- `make lint`: passed
- `make test`: passed

This keeps Sprint 37 on the validated baseline rather than opening a new
benchmark regression queue.

#### 6. The residual benchmark-helper queue is now narrower and better partitioned

Still deferred after Day 6:

- timer/residual drift outside this pair, especially:
  - `bench_refactor_csc.c`
  - `bench_bicgstab.c`
  - `bench_convergence.c`
- larger behavior-owner benchmark surfaces:
  - `bench_main.c`
  - `bench_reorder.c`
  - `bench_eigs.c`
- benchmark-specific reporting / CLI helpers

Reason:

- those surfaces either carry broader CLI/report ownership or diverge more
  structurally than the backend-comparison pair
- forcing them into Day 6 would have widened the batch beyond the intended
  low-risk scope

### Day 6 Interpretation

- Day 6 converted the Day 3 audit into a real first benchmark-helper batch
  without inventing a broad benchmark framework
- the backend-comparison pair was the right first target because its helper
  duplication was both real and behaviorally stable
- the remaining queue is now narrower and more clearly partitioned:
  - pair-external timer/residual helpers
  - larger behavior-owner benchmark files
  - benchmark-specific reporting / CLI helpers

### Day 6 Outputs

- `benchmarks/bench_backend_compare_helpers.h`
- `artifacts/day6-benchmark-helper-consolidation-batch1.md`

## Day 7

**Objective:** Implement the first Makefile normalization batch from the Day 4
design by making the maintained quality-target ownership model more explicit in
comments, section layout, and operator guidance, while reducing the visibility
gap around the Sprint 36 sanitizer/build-tree caveat without changing target
semantics.

### Commands Run

1. Re-read the Day 7 plan and the Day 4 target-normalization design:
   - `git status --short --branch`
   - `sed -n '177,208p' docs/planning/EPIC_3/SPRINT_37/PLAN.md`
   - `cat docs/planning/EPIC_3/SPRINT_37/artifacts/day4-quality-target-normalization-design.md`
2. Re-read the current maintained quality surface:
   - `sed -n '300,780p' Makefile`
   - `sed -n '680,780p' README.md`
3. Recheck the tree-mutating target family and reset behavior:
   - `rg -n "^(sanitize|asan|sanitize-all|omp|tsan|coverage-lcov|coverage-gcovr):|^\\.PHONY: sanitize|^\\.PHONY: asan|^\\.PHONY: sanitize-all|^\\.PHONY: omp|^\\.PHONY: tsan|^\\.PHONY: coverage" Makefile`
4. Implement the normalization batch:
   - clarify category boundaries in `Makefile`
   - add explicit `make clean` return guidance to reviewed wrappers
   - mirror the same operator rule in `README.md`
5. Validate the touched operator paths directly:
   - `make -n quality-review-compile`
   - `make -n quality-review`
   - `make -n quality-review-cmake-compile`
   - `make -n sanitize`
   - `make -n coverage-gcovr`

### Day 7 Findings

#### 1. The highest-value first batch was comment/layout normalization plus explicit reset guidance, not target renaming

The Day 4 design still held:

- the repo already had the needed direct, reviewed, helper, and instrumentation
  targets
- the main maintainability problem was that they still read too much like one
  flat surface

So Day 7 deliberately did **not**:

- rename established targets
- add a parallel reset alias
- change the `lint` / `test` / `check` / reviewed-wrapper semantics

It instead normalized:

- category boundaries in the Makefile
- operator-facing wrapper guidance
- README operator documentation

#### 2. The Makefile now reflects the Day 4 three-way ownership model more directly

The Day 7 batch reorganized the quality section signaling around the Day 4
classes:

Category A — maintained operator entry points:

- direct gates
- reviewed local wrappers
- reviewed CMake wrappers
- maintained specialized top-level surfaces such as `deadcode-check` and
  `wall-check`

Category B — helper / prerequisite plumbing:

- warning-workflow
- dead-code compile/report layering
- build-plumbing support for the maintained operator flows

Category C — tree-mutating instrumentation or alternate-build modes:

- sanitize / asan / sanitize-all / tsan / omp
- coverage / coverage-lcov / coverage-gcovr

Interpretation:

- the target list is still the same
- the ownership model is no longer implicit
- helper targets read less like peer operator entry points
- instrumentation/coverage modes read less like ordinary neighboring commands

#### 3. The Sprint 36 sanitizer caveat is now surfaced where operators actually see it

Day 4 said the right first fix was visibility and ownership, not a new reset
command.

Day 7 implemented that directly:

- reviewed wrapper banners now say:
  - if returning from `sanitize` / `asan` / `sanitize-all` / `tsan` / `omp` /
    `coverage*`, reset first with `make clean`
- the Makefile now labels sanitizer/coverage targets as tree-mutating
  alternate-build modes
- the README operator command map now mirrors the same rule instead of leaving
  it as retrospective tribal knowledge

This materially reduces the risk that maintainers jump straight from an
instrumented build mode back into the normal direct/reviewed path without a
reset.

#### 4. The batch preserved the established reviewed-quality contract

The Day 7 normalization kept these meanings unchanged:

- `format`, `format-check`, `lint`, `test`, `check`
- `quality-review-compile`
- `quality-review`
- `quality-review-cmake-compile`
- `quality-review-cmake`
- `deadcode-report`
- `deadcode-check`

Validation evidence from the dry-run path checks:

- wrapper order is unchanged
- lower-level invoked commands are unchanged
- tree-mutating targets still own `clean` on entry
- the new guidance is additive operator messaging rather than behavioral drift

#### 5. The residual target-layout queue is now smaller and more bounded

Still deferred after Day 7:

- any deeper physical reordering of target blocks beyond the first category
  signaling pass
- any later cleanup in the coverage / instrumentation area beyond ownership
  comments
- any future build-tree isolation work for dead-code or alternate modes

Reason:

- Day 7 was the right scope for improving clarity without reopening target
  semantics or inventing new command names

### Day 7 Interpretation

- Day 7 converted the Day 4 design into a real first normalization batch
  without changing the maintained quality contract
- the most valuable first move was making ownership and reset expectations
  explicit where operators actually encounter them
- the Sprint 36 sanitizer/build-tree caveat is now materially less implicit,
  while `make clean` remains the single canonical reset

### Day 7 Outputs

- `artifacts/day7-quality-target-normalization-batch1.md`

## Day 8

**Objective:** Audit the largest auxiliary files across tests, benchmarks,
scripts, docs, and Makefile-adjacent surfaces to separate files that are merely
large from files whose size reflects real maintainability problems such as
mixed concerns, helper drift, or review-hostile structure, then define the
highest-value one-or-two-file cleanup batch for Day 9.

### Commands Run

1. Re-read the Day 8 plan and the Day 1 baseline:
   - `git status --short --branch`
   - `sed -n '209,240p' docs/planning/EPIC_3/SPRINT_37/PLAN.md`
   - `cat docs/planning/EPIC_3/SPRINT_37/artifacts/day1-auxiliary-maintainability-baseline.md`
2. Inventory the largest auxiliary files across the repo:
   - `python3` line-count inventory for:
     - `tests/**/*.c`
     - `benchmarks/**/*.c`
     - `scripts/**/*`
     - `docs/**/*.md`
     - `Makefile`
     - `README.md`
3. Compare candidate structure rather than line count alone:
   - `python3` marker/function summary for:
     - `tests/test_chol_csc.c`
     - `tests/test_svd.c`
     - `tests/test_ldlt_csc.c`
     - `tests/test_qr.c`
     - `tests/test_etree.c`
     - `tests/test_iterative.c`
     - `benchmarks/bench_eigs.c`
     - `benchmarks/bench_main.c`
     - `Makefile`
     - `scripts/deadcode_report.py`
     - `scripts/deadcode_workflow.sh`
4. Re-read representative candidate files:
   - `sed -n '1,220p' tests/test_chol_csc.c`
   - `sed -n '1,220p' benchmarks/bench_eigs.c`
   - `sed -n '1,220p' scripts/deadcode_report.py`
   - `sed -n '220,520p' scripts/deadcode_report.py`
   - `sed -n '1,220p' scripts/deadcode_workflow.sh`
   - `sed -n '380,845p' Makefile`
   - `sed -n '680,780p' README.md`

### Day 8 Findings

#### 1. The largest files are not automatically the best refactor targets

The raw line-count leaders are dominated by tests:

- `tests/test_chol_csc.c` = `4,643`
- `tests/test_svd.c` = `3,712`
- `tests/test_ldlt_csc.c` = `3,637`
- `tests/test_qr.c` = `3,259`
- `tests/test_etree.c` = `2,890`
- `tests/test_iterative.c` = `2,796`

But those files are not just large; they are also strongly sectioned behavior
owners:

- `tests/test_chol_csc.c`
  - `155` static functions
  - `138` section/marker lines
- `tests/test_svd.c`
  - `99` static functions
  - `115` section/marker lines
- `tests/test_ldlt_csc.c`
  - `123` static functions
  - `142` section/marker lines

Interpretation:

- these files are expensive to navigate, but they are also heavily structured
  and map closely to broad feature surfaces
- under the repo’s one-binary-per-test-source build model, mechanical splitting
  would introduce more ownership and build-model complexity than Day 8 justifies
- they are large, but not the clearest first maintainability win

#### 2. The strongest Day 9 targets are mixed-concern auxiliary files, not the longest tests

Two files stood out as large **and** structurally mixed:

##### `Makefile`

- `845` lines
- `39` `.PHONY` declarations
- about `101` target-like rule lines

Why it is a stronger maintainability target than its raw size alone suggests:

- it mixes:
  - build products
  - direct quality gates
  - reviewed wrappers
  - helper plumbing
  - dead-code workflow layering
  - instrumentation / sanitizer / coverage modes
  - install / pkg-config logic
- even after Day 7’s ownership pass, the file still concentrates too many
  operational surfaces in one place
- review locality is weak: small target changes still require large-context
  rereads because the file is a broad command map as well as the implementation

##### `scripts/deadcode_report.py`

- `472` lines
- `15` top-level functions

Why it is a stronger maintainability target than its raw size alone suggests:

- it combines several responsibilities in one file:
  - parsing coverage notes
  - parsing `xunused`
  - parsing `cppcheck`
  - classification policy
  - TSV generation
  - markdown rendering
  - validator logic
- the file has very low visual section signaling compared with the larger tests
- Sprint 33 / 34 / 36 added capability incrementally, so the script now reads
  as a layered workflow core rather than one clear single-purpose utility

Interpretation:

- these two files are not the largest in the repo
- they are the clearest mixed-concern maintainability surfaces

#### 3. `scripts/deadcode_workflow.sh` is not a first-pass large-file refactor target

`scripts/deadcode_workflow.sh` is only `189` lines and still reads as a fairly
cohesive workflow runner:

- prerequisite checks
- compile-db validation
- coverage-note write
- `cppcheck` run
- `xunused` run

It has complexity, but not the same mixed-concern sprawl as:

- `Makefile`
- `scripts/deadcode_report.py`

Interpretation:

- it can stay in the residual queue
- it is not the highest-value Day 9 target

#### 4. The large benchmark owners are real, but they are not the best structural cleanup target yet

Largest benchmark files:

- `benchmarks/bench_eigs.c` = `958`
- `benchmarks/bench_main.c` = `774`

Those files are genuinely large, but they still read as behavior-owner CLIs:

- `bench_eigs.c`
  - owns multi-mode eigensolver CLI/report behavior
  - size comes from real sweep / compare / single-matrix behavior breadth
- `bench_main.c`
  - owns the narrower solver-harness benchmark contract from Sprint 31

Interpretation:

- they may deserve later cleanup
- but they are not as strong a first large-file maintainability target as the
  mixed-concern `Makefile` + dead-code reporting surfaces

#### 5. Very large sprint working notes and planning docs are intentionally long and should not drive the refactor batch

The repo contains many very large planning docs and sprint working notes, for
example:

- `docs/planning/EPIC_3/SPRINT_34/WORKING_NOTES.md` = `2,076`
- `docs/algorithm.md` = `1,508`
- `docs/planning/EPIC_3/SPRINT_37/WORKING_NOTES.md` = `1,439`

Interpretation:

- these are intentionally archival / narrative surfaces
- they are not the right Day 9 maintainability target
- refactoring them for line count alone would not materially improve the active
  code-maintenance workflow

#### 6. The highest-value Day 9 batch is now well-bounded

Recommended Day 9 target set:

Priority A:

- `Makefile`

Priority B:

- `scripts/deadcode_report.py`

Intended shape of the Day 9 cleanup:

- structural simplification
- clearer sectioning / helper boundaries
- local helper extraction or layout cleanup where it reduces reread cost
- no behavior change

Avoid for Day 9:

- splitting giant feature-owner tests mechanically
- broad benchmark CLI refactors
- narrative/planning doc churn

### Day 8 Interpretation

- line count alone would have pointed at giant tests, but the better
  maintainability targets are the mixed-concern auxiliary surfaces
- the strongest Day 9 cleanup batch is now explicit and bounded:
  - `Makefile`
  - `scripts/deadcode_report.py`
- the large tests and large benchmark owners remain in the residual queue, but
  they are not the best first structural cleanup win

### Day 8 Outputs

- `artifacts/day8-large-file-maintainability-audit.md`

## Day 9

**Objective:** Implement the first large-file maintainability refactor batch on
the two mixed-concern auxiliary surfaces chosen on Day 8, improve structure
without changing behavior, and leave the residual queue narrower and explicit.

### Commands Run

1. Re-read the Day 9 plan and Day 8 target selection:
   - `sed -n '214,250p' docs/planning/EPIC_3/SPRINT_37/PLAN.md`
   - `cat docs/planning/EPIC_3/SPRINT_37/artifacts/day8-large-file-maintainability-audit.md`
2. Inspect the current target shapes before editing:
   - `sed -n '1,240p' scripts/deadcode_report.py`
   - `sed -n '240,520p' scripts/deadcode_report.py`
   - `sed -n '140,360p' Makefile`
   - `sed -n '360,780p' Makefile`
   - `rg -n "for t in \\$\\(TEST_BINS\\)|Some tests failed|All tests passed" Makefile`
3. Apply the Day 9 structural cleanup:
   - shared `Makefile` test-loop helpers for `test`, `coverage-lcov`, and
     `coverage-gcovr`
   - `scripts/deadcode_report.py` markdown-render helper extraction
4. Revalidate the touched support paths directly:
   - `python3 -m py_compile scripts/deadcode_report.py`
   - `make test`
   - `make deadcode-report`
   - `make deadcode-check`
5. Capture post-change shape details for the notes:
   - `wc -l Makefile scripts/deadcode_report.py`
   - `python3` function-count scan over `scripts/deadcode_report.py`
   - `python3` line-location scan for the new Makefile test-loop helpers

### Day 9 Findings

#### 1. The right Day 9 maintainability win was helper extraction, not file splitting

Day 8's target choice held up under implementation:

- `Makefile` still benefits more from internal structure cleanup than from
  artificial splitting
- `scripts/deadcode_report.py` still benefits more from internal phase
  boundaries than from behavior or CLI changes

The shipped batch therefore stayed deliberately narrow:

- one shared test-loop seam in `Makefile`
- one shared markdown-section seam in `scripts/deadcode_report.py`

Interpretation:

- this is a maintainability batch, not a shrink-the-file-count batch
- line count alone did not improve, but reread locality did

#### 2. `Makefile` now has one authoritative maintained test-run loop instead of three near-copies

The direct `test` target and both coverage targets previously each carried
their own all-test execution loop.

Day 9 replaced that drift-prone repetition with two named helpers:

- `RUN_TEST_BINS_WITH_BANNERS`
- `RUN_TEST_BINS_QUIET`

Current landing points:

- helper definitions at lines `213` and `223`
- `test` target uses the bannered helper
- `coverage-lcov` and `coverage-gcovr` use the quiet helper

Behavior stayed the same:

- `test` still prints per-binary banners and ends with `All tests passed.`
- coverage paths still fail the same way on any failing test binary
- no target names or operator entry points changed

Interpretation:

- the maintained test-bin execution contract now has one clear ownership point
- future edits to failure handling or shared loop behavior no longer require
  three synchronized changes

#### 3. `scripts/deadcode_report.py` now reads as sectioned report assembly instead of one long markdown block

The script remains a single utility, but its markdown-render phase is no longer
one large mixed block.

Day 9 extracted the major report sections into dedicated helpers:

- `append_run_metadata`
- `append_coverage_gaps`
- `append_internal_candidates`
- `append_public_surface_items`
- `append_secondary_signals`
- `append_noise_summary`
- `append_next_action_queue`

Supporting helpers were also added for repeated report structure:

- `append_section`
- `append_symbol_rows`
- `public_bucket_reviewed_keeps`

Post-change shape:

- `scripts/deadcode_report.py` is now `504` lines
- it now has `25` top-level functions
- that is larger in raw count than Day 8's `472` / `15`, but the added lines
  are almost entirely internal structure and section boundaries rather than new
  workflow semantics

Interpretation:

- the file is easier to navigate by report phase
- Sprint 33 / 34 / 36 logic is now easier to audit without reading the full
  renderer top-to-bottom
- this is the right kind of growth for a maintainability pass

#### 4. The touched support paths stayed behavior-stable

Direct Day 9 validation passed:

- `python3 -m py_compile scripts/deadcode_report.py`
- `make test`
- `make deadcode-report`
- `make deadcode-check`

Meaning:

- the shared Makefile loop still drives the full maintained test suite cleanly
- the dead-code report/check flow still produces the same artifacts and
  validation contract after the renderer refactor

#### 5. The residual large-file queue is now narrower and more explicit

Closed or materially reduced by Day 9:

- `Makefile` test-loop duplication across direct and coverage paths
- `scripts/deadcode_report.py` markdown-render monolith

Residual queue after Day 9:

- `scripts/deadcode_workflow.sh`
- `benchmarks/bench_eigs.c`
- `benchmarks/bench_main.c`
- large feature-owner tests such as:
  - `tests/test_chol_csc.c`
  - `tests/test_svd.c`
  - `tests/test_ldlt_csc.c`
  - `tests/test_qr.c`
  - `tests/test_etree.c`
  - `tests/test_iterative.c`
- wider `Makefile` target-graph density outside the now-shared test loop
- broader `scripts/deadcode_report.py` parse/classify/validate layering, which
  remains intentionally in one file for now

Interpretation:

- Day 9 improved the two chosen auxiliary surfaces without pretending to solve
  every large-file concern at once
- later Sprint 37 work can now focus on comment debt and final cleanup from a
  smaller structural queue

### Day 9 Interpretation

- the best first large-file maintainability batch was still `Makefile` plus
  `scripts/deadcode_report.py`
- the right move was behavior-preserving helper extraction, not aggressive
  splitting
- the residual queue is now smaller and more concrete

### Day 9 Outputs

- `artifacts/day9-large-file-maintainability-batch1.md`

## Day 10

**Objective:** Audit auxiliary comments and maintainer-facing support docs for
stale sprint-history framing, duplicated workflow explanation, and wording that
now obscures the actual Sprint 34-Sprint 36 quality contract.

### Commands Run

1. Re-read the Day 10 plan and the Day 9 residual queue:
   - `sed -n '241,320p' docs/planning/EPIC_3/SPRINT_37/PLAN.md`
   - `tail -n 220 docs/planning/EPIC_3/SPRINT_37/WORKING_NOTES.md`
2. Sweep the main maintainer-facing workflow surfaces for reviewed/dead-code /
   staged/excluded / sanitizer wording:
   - `rg -n "Sprint 33|Sprint 34|Sprint 35|Sprint 36|Sprint 37|dead-code|staged|excluded|sanitize|make clean|quality-review|quality-review-cmake|reviewed" README.md Makefile scripts/deadcode_report.py scripts/deadcode_workflow.sh .github/workflows/*.yml`
3. Inspect the most likely high-duplication surfaces directly:
   - `sed -n '1,120p' .github/workflows/ci.yml`
   - `sed -n '1,140p' .github/workflows/macos-ci.yml`
   - `sed -n '1,120p' .github/workflows/windows-ci.yml`
   - `sed -n '632,790p' README.md`
   - `sed -n '1,220p' scripts/deadcode_workflow.sh`
   - `sed -n '1,220p' scripts/deadcode_report.py`

### Day 10 Findings

#### 1. The main Day 10 debt is duplication and sprint-history overframing, not contract breakage

The current maintainer-facing workflow language is mostly accurate. The problem
is that the same accurate contract is restated too many times, often with more
historical sprint framing than operators need in the maintained surface.

Most visible pattern:

- workflow YAML headers explain the platform contract in detail
- `README.md` explains the same platform contract again in more detail
- individual script headers still frame themselves as the result of a specific
  sprint day rather than as current maintained utilities
- `Makefile` comments are already closer to the right shape, but still include
  some very dense historical framing near the reviewed-wrapper / dead-code /
  alternate-build sections

Interpretation:

- this is not a truthfulness failure
- it is a clarity and reread-cost problem

#### 2. Workflow file headers are the highest-value “simplify, don’t expand” target

Top-of-file workflow comments in:

- `.github/workflows/ci.yml`
- `.github/workflows/macos-ci.yml`
- `.github/workflows/windows-ci.yml`

are all accurate, but they currently do too much at once:

- define the enforced/staged/supplemental contract
- explain why a platform differs
- preserve sprint-history breadcrumbs
- preserve detailed portability caveats

Why this is the best Day 11 cleanup target:

- operators hit these files directly when CI behavior changes
- long header blocks slow down simple step-name or job-scope edits
- the README already owns the fuller cross-platform contract summary

Best cleanup shape:

- keep a short pointer-style top comment in each workflow
- state only the current platform role and major exclusions
- point maintainers back to the authoritative README section for the fuller
  contract

#### 3. The dead-code script headers contain the clearest stale wording

Two files still present themselves mainly as Sprint 33 artifacts:

- `scripts/deadcode_workflow.sh`
  - `Sprint 33 Day 5 raw dead-code workflow`
  - `Sprint 33 Day 5 dead-code coverage notes`
- `scripts/deadcode_report.py`
  - `Generate and validate Sprint 33 dead-code reports.`
  - `Generate or validate dead-code reports from raw Sprint 33 artifacts.`
  - report title `# Sprint 33 Dead-Code Report`

This wording is now stale for maintainers:

- the workflow is still current and maintained
- the report/check path is now part of the Sprint 34-Sprint 36 reviewed and CI
  contract
- “Sprint 33” is useful provenance, but not the right primary label anymore

Interpretation:

- these files should move from sprint-origin wording to current maintained
  utility wording
- provenance can stay in a shorter note, but should not dominate the header or
  rendered report title

#### 4. `README.md` is authoritative, but its maintainer-workflow section is more repetitive than it needs to be

The key maintained README workflow section currently includes:

- dead-code workflow overview
- reviewed local quality path
- cross-platform CI contract
- operator command map
- rerun guidance
- tree-mutating mode reset guidance

All of those topics belong there, but the section currently repeats some of the
same command names and contract language in multiple sub-blocks.

Main duplication pattern:

- command names appear in the reviewed-wrapper intro, the cross-platform table,
  the operator command map, and the rerun guidance
- the dead-code section and the reviewed-wrapper section both restate
  conservative interpretation caveats that could be slightly tighter
- the reset guidance after sanitizer/coverage modes is correct, but could be
  made more pointer-style now that the Makefile wrappers themselves also print
  the same guidance

Interpretation:

- Day 11 should simplify and compress this section, not relocate ownership away
  from README
- README remains the authoritative maintainer workflow surface

#### 5. Not every historical comment is debt

Some comments should remain as-is, because they encode real platform or tooling
constraints rather than historical clutter:

- TSan / libomp caveats in `.github/workflows/ci.yml`
- Apple Clang / libomp / GCC OpenMP details in `Makefile`
- coverage backend split comments in `Makefile`
- partial-success handling comments in `scripts/deadcode_workflow.sh`

Interpretation:

- Day 11 should not become a broad anti-comment pass
- the target is stale or duplicative maintainer guidance, not dense but still
  useful implementation rationale

#### 6. The Day 11 cleanup batch is now concrete and bounded

Ranked wording cleanup queue:

Priority A:

- workflow top-of-file headers in:
  - `.github/workflows/ci.yml`
  - `.github/workflows/macos-ci.yml`
  - `.github/workflows/windows-ci.yml`
- stale Sprint 33-first wording in:
  - `scripts/deadcode_workflow.sh`
  - `scripts/deadcode_report.py`

Priority B:

- maintainer-workflow compression in `README.md`

Priority C:

- any small matching `Makefile` comment simplifications discovered while doing
  the above, but only if they reduce duplication without weakening the target
  ownership model

Explicit non-targets for Day 11:

- numerical algorithm docs
- public API tutorials
- benchmark behavior-owner commentary
- broad deletion of still-useful platform/tooling rationale

### Day 10 Interpretation

- the current workflow contract is already accurate
- the debt is mainly too much repeated explanation and too much sprint-history
  framing in maintained operator-facing files
- the highest-value Day 11 batch is to simplify workflow/script headers and
  compress the README maintainer workflow section while preserving the
  authoritative contract

### Day 10 Outputs

- `artifacts/day10-comment-and-maintainer-doc-audit.md`

## Day 11

**Objective:** Implement the bounded maintainer-workflow wording cleanup from
Day 10, reduce duplicated or sprint-history-first guidance in active operator
surfaces, and keep the current Sprint 34-Sprint 36 contract unchanged.

### Commands Run

1. Re-read the Day 11 scope and the Day 10 audit:
   - `sed -n '241,320p' docs/planning/EPIC_3/SPRINT_37/PLAN.md`
   - `sed -n '1,220p' docs/planning/EPIC_3/SPRINT_37/artifacts/day10-comment-and-maintainer-doc-audit.md`
2. Re-open the targeted maintainer-facing files:
   - `sed -n '1,80p' .github/workflows/ci.yml`
   - `sed -n '1,90p' .github/workflows/macos-ci.yml`
   - `sed -n '1,80p' .github/workflows/windows-ci.yml`
   - `sed -n '1,80p' scripts/deadcode_workflow.sh`
   - `sed -n '1,80p' scripts/deadcode_report.py`
   - `sed -n '632,790p' README.md`
3. Validate the touched support surfaces directly:
   - `ruby -e 'require "yaml"; ...'` for `.github/workflows/{ci,macos-ci,windows-ci}.yml`
   - `bash -n scripts/deadcode_workflow.sh`
   - `python3 -m py_compile scripts/deadcode_report.py`
   - `make deadcode-report`
   - `make deadcode-check`
4. Re-run the dead-code path serially after one attempted concurrent rerun hit
   the known shared-path race:
   - `make deadcode-report && make deadcode-check`

### Day 11 Findings

#### 1. The highest-value cleanup was compression, not relocation

The Day 10 audit was right about shape:

- the workflow contract already lived in the correct files
- the problem was repeated explanation and sprint-history-first framing
- the best fix was to compress and relabel the active operator surfaces rather
  than move ownership elsewhere

The Day 11 batch therefore stayed narrow:

- shorter workflow file headers
- current-purpose dead-code utility wording
- tighter README maintainer-workflow prose

#### 2. Workflow YAML headers now point to the contract instead of retelling it

Updated files:

- `.github/workflows/ci.yml`
- `.github/workflows/macos-ci.yml`
- `.github/workflows/windows-ci.yml`

What changed:

- long Sprint 36 history-heavy top comments were replaced by short
  pointer-style summaries
- each workflow now states only:
  - its current platform role
  - the major enforced or staged boundary
  - the README section that owns the fuller contract

Why this is better:

- maintainers can reach the jobs faster
- the README remains the human-readable contract owner
- the workflows still expose their role clearly without carrying the full
  platform narrative at the top of every file

#### 3. The dead-code utilities now describe current purpose instead of Sprint 33 origin first

Updated files:

- `scripts/deadcode_workflow.sh`
- `scripts/deadcode_report.py`

What changed:

- the workflow script header now describes itself as a raw dead-code evidence
  refresh utility
- the generated coverage-notes banner now says `Dead-code coverage notes`
- the report script docstring and CLI description now refer to raw workflow
  artifacts instead of Sprint 33 artifacts
- the rendered report title is now `# Dead-Code Report`

Why this is better:

- these tools now read like maintained utilities
- Sprint 33 provenance is no longer the dominant operator-facing label
- the reviewed/CI contract that grew around them in later sprints is not hidden
  behind stale origin wording

#### 4. `README.md` now says the same workflow contract with less duplication

Updated maintainer-workflow areas:

- dead-code workflow intro
- reviewed local quality path
- cross-platform CI contract interpretation
- tree-mutating mode reset wording

What changed:

- removed unnecessary Sprint-introduction framing where the commands
  themselves already communicate the current contract
- compressed repeated interpretation bullets
- kept the command map and rerun guidance, but reduced repeated restatement of
  the same wrapper relationships

What did **not** change:

- README still owns the authoritative maintainer workflow summary
- the enforced/staged/supplemental platform contract is unchanged
- the rerun guidance and `make clean` reset guidance are still explicit

#### 5. The one operational limitation remains the known dead-code shared-path race

During Day 11 validation, one attempted concurrent dead-code rerun reproduced
the known shared-path limitation:

- `make deadcode-check` passed
- a simultaneous `make deadcode-report` run tripped the expected
  `report.md missing coverage-gap section` race

The authoritative validation was the required serialized rerun:

- `make deadcode-report && make deadcode-check`

Interpretation:

- the wording cleanup did not introduce a new regression
- the Sprint 33 / Sprint 34 shared-path limitation remains real and still must
  be respected in maintainer workflows

#### 6. Residual maintainer-doc debt is now smaller and narrow

Closed or materially reduced:

- workflow top-of-file over-explanation
- Sprint 33-first dead-code utility wording
- repeated README maintainer-workflow framing

Residual queue after Day 11:

- dense but still-useful platform/tooling rationale in `Makefile` and Linux
  TSan workflow comments
- the known dead-code serial-only limitation, which remains a workflow
  constraint until the shared-path work lands in later sprints
- any further README compression beyond this point would risk losing operator
  clarity rather than improving it

### Day 11 Interpretation

- Day 10’s chosen batch was the right one
- the best improvement was to make active operator files point to the contract
  instead of repeatedly narrating its sprint history
- the current workflow model is now easier to scan without weakening the actual
  maintained expectations

### Day 11 Outputs

- `artifacts/day11-maintainer-workflow-docs-batch.md`
