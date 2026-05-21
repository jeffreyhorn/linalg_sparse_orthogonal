# Sprint 38 Working Notes

## Day 1

**Objective:** Turn the Sprint 34, Sprint 36, and Sprint 37 handoff state plus
the Sprint 38 project-plan scope into a concrete regression-proofing baseline by
confirming the inherited validated quality contract, inventorying the current
coverage/gate/reporting surfaces, and naming the first audit targets before any
implementation begins.

### Commands Run

1. Confirm branch and starting state:
   - `git status --short --branch`
   - `git rev-parse --short HEAD`
   - `git branch --show-current`
2. Re-read the Sprint 38 scope and inherited constraints:
   - `sed -n '296,332p' docs/planning/EPIC_3/PROJECT_PLAN.md`
   - `cat docs/planning/EPIC_3/SPRINT_34/HANDOFF.md`
   - `cat docs/planning/EPIC_3/SPRINT_36/HANDOFF.md`
   - `cat docs/planning/EPIC_3/SPRINT_37/HANDOFF.md`
   - `sed -n '1,220p' docs/planning/EPIC_3/SPRINT_38/PLAN.md`
3. Reconfirm the inherited reviewed CMake suite baseline:
   - `ctest -N --test-dir build/quality-review-cmake`
4. Recheck local prerequisite tool availability:
   - `command -v cppcheck`
   - `command -v clang-tidy`
   - `command -v xunused`
   - `command -v gcovr`
   - `command -v ctest`
5. Inventory current maintained target and reporting surfaces:
   - `rg -n '^\\.PHONY: (quality-review|quality-review-cmake|deadcode|coverage|format|lint|test|wall-check)|^(quality-review|quality-review-cmake|deadcode|coverage|format|lint|test|wall-check)[^[:alnum:]_-]' Makefile`
   - `make -n quality-review quality-review-cmake deadcode-report deadcode-check coverage-lcov coverage-gcovr`
6. Reconfirm current dead-code artifact/report surfaces:
   - `ls build/deadcode/`
   - targeted checks for:
     - `build/deadcode/report.md`
     - `build/deadcode/report.tsv`
     - `build/deadcode/cppcheck.txt`
     - `build/deadcode/xunused.txt`
     - `build/deadcode/coverage-notes.txt`
     - `build/deadcode/.workflow.stamp`
     - `build/deadcode/.report.stamp`
7. Cross-check inherited open limitations in current docs:
   - `rg -n 'bench_svd|example_basic_solve|example_condition|example_iterative|example_least_squares|example_matrix_free|example_svd_lowrank|build/deadcode-cmake|build/deadcode/' README.md docs/planning/EPIC_3/SPRINT_{34,36,37}/HANDOFF.md`

### Day 1 Findings

#### 1. Sprint 38 starts from a validated regression baseline, not unresolved warning or parity debt

Sprint 38 inherits the Sprint 37 close state exactly as intended:

- direct maintained gates were green at handoff:
  - `make format`
  - `make lint`
  - `make test`
- reviewed wrapper paths were green at handoff:
  - `make quality-review-compile`
  - `make quality-review`
  - `make quality-review-cmake-compile`
  - `make quality-review-cmake`
- maintained support/reporting paths were green at handoff:
  - `make deadcode-report`
  - `make deadcode-check`
  - `make wall-check`
- the active reviewed CMake suite baseline remains:
  - `ctest -N --test-dir build/quality-review-cmake` = `53`

Interpretation:

- Sprint 38 is not a warning-cleanup sprint
- Sprint 38 is not a dead-code-baseline sprint
- Sprint 38 is a regression-proofing and signaling sprint layered on top of a
  validated quality contract

#### 2. The current quality surface is broad enough; the main Sprint 38 risk is signaling drift, not missing target names

The maintained target graph already exposes the surfaces Sprint 38 is supposed
to harden:

- direct/reviewed quality paths:
  - `format`
  - `format-check`
  - `lint`
  - `test`
  - `check`
  - `quality-review-compile`
  - `quality-review`
  - `quality-review-cmake-compile`
  - `quality-review-cmake`
- dead-code/reporting paths:
  - `deadcode-compile-db`
  - `deadcode`
  - `deadcode-report`
  - `deadcode-check`
- compile-only / auxiliary validation paths:
  - `tooling-build`
  - `wall-check`
- coverage/reporting paths:
  - `coverage`
  - `coverage-lcov`
  - `coverage-gcovr`

Interpretation:

- Sprint 38 does not start by needing a whole new target family
- it starts by needing more truthful, better-scoped regression signaling on
  top of existing targets and reports

#### 3. The dead-code workflow is operational, but its inherited limitations remain real and must stay explicit

Current local dead-code/report artifacts are present and named consistently:

- `build/deadcode/report.md`
- `build/deadcode/report.tsv`
- `build/deadcode/cppcheck.txt`
- `build/deadcode/xunused.txt`
- `build/deadcode/coverage-notes.txt`
- `build/deadcode/.workflow.stamp`
- `build/deadcode/.report.stamp`

Still open from Sprint 34 through Sprint 37:

- compile-db exclusion list:
  - `bench_svd`
  - `example_basic_solve`
  - `example_condition`
  - `example_iterative`
  - `example_least_squares`
  - `example_matrix_free`
  - `example_svd_lowrank`
- shared execution paths:
  - `build/deadcode-cmake`
  - `build/deadcode/`

Interpretation:

- Sprint 38 can mature dead-code reporting and gate behavior
- but it must not blur the difference between actionable improvement and fully
  solved concurrency/coverage readiness

#### 4. Coverage/readiness truthfulness is now the highest-signal wording risk

The cleaned-up test contract from Sprint 32 is still in force:

- active tests are what default `make test` / `ctest` runs
- slow and experimental tests are opt-in surfaces
- historical evidence is supposed to live in docs/artifacts, not dormant test
  scaffold

At the same time, the repo now has multiple quality/reporting layers:

- direct test execution
- reviewed Makefile wrappers
- reviewed CMake parity wrappers
- dead-code reports/checks
- coverage targets and reports
- CI workflow summaries/artifacts

Interpretation:

- the leading Sprint 38 risk is no longer stale helper code or warning debt
- it is overstated or ambiguous wording about what is actually covered,
  enforced, staged, or merely reported

#### 5. Compile-only protection is still the most concrete unresolved regression surface

Sprint 34 and later handoffs preserved a named unresolved compile-only queue
rather than claiming it was already closed:

- `bench_svd`
- `example_basic_solve`
- `example_condition`
- `example_iterative`
- `example_least_squares`
- `example_matrix_free`
- `example_svd_lowrank`

The current local reviewed/dead-code/reporting paths make that queue visible,
but do not by themselves prove it has been closed as a routine regression gate.

Interpretation:

- Sprint 38 Day 3 and Day 6 are real implementation work, not paperwork
- this is the most concrete named gap list entering the sprint

#### 6. The first audit/implementation batches are already clear

Highest-value Sprint 38 surfaces at Day 1:

- coverage truthfulness:
  - `README.md`
  - coverage target/report wording
  - test-category/report wording tied to the Sprint 32 opt-in contract
- compile-only protection:
  - named exclusion-list binaries from Sprint 34 handoff
- dead-code maturation:
  - `scripts/deadcode_workflow.sh`
  - `scripts/deadcode_report.py`
  - `build/deadcode/report.md` / `report.tsv` contract
- readiness/reporting polish:
  - quality-wrapper output
  - CI artifact/report wording
  - concise release/readiness checklist surface

Interpretation:

- Sprint 38 already has a bounded implementation queue
- the initial days should stay audit-first so later gate changes remain
  truthful rather than aspirational

## Day 2

**Objective:** Audit the repo's coverage language, target names, summaries, and
artifact wording against the actual active/opt-in test contract so later Sprint
38 coverage cleanup can stay narrow and truthful instead of drifting into fake
"more coverage" claims.

### Commands Run

1. Re-read the Sprint 38 Day 2 plan section:
   - `sed -n '1,220p' docs/planning/EPIC_3/SPRINT_38/PLAN.md`
2. Re-read the actual test-framework opt-in contract:
   - `sed -n '1,260p' tests/test_framework.h`
   - `sed -n '1,220p' tests/test_framework_optin.c`
3. Sweep coverage/category wording across docs, targets, and workflows:
   - `rg -n "coverage|gcovr|lcov|RUN_TEST_SLOW|RUN_TEST_EXPERIMENTAL|SPARSE_TEST_SLOW|SPARSE_TEST_EXPERIMENTAL|skipped|slow|experimental" README.md INSTALL.md docs include tests Makefile .github/workflows -g '!docs/planning/**'`
4. Re-read the main user-facing coverage/test sections:
   - `sed -n '540,780p' README.md`
   - `sed -n '110,190p' INSTALL.md`
   - `sed -n '195,240p' .github/workflows/ci.yml`
5. Re-read the actual coverage-target implementation and threshold:
   - `sed -n '680,780p' Makefile`
6. Confirm current suite-count and opt-in-category facts:
   - `python3` count of `tests/test_*.c`
   - `python3` count of `add_sparse_test(...)` in `CMakeLists.txt`
   - `sed -n '240,300p' tests/test_suitesparse.c`
7. Re-read the short README command map where coverage is introduced:
   - `sed -n '110,140p' README.md`

### Day 2 Findings

#### 1. The largest coverage-honesty problem is stale top-level README language, not missing gate machinery

The strongest current mismatch is concentrated in one high-visibility README
sentence:

- `README.md` still says the test suite has:
  - `1453 unit tests`
  - `42 test suites`
  - `>=95% line coverage (CI-enforced)`

Current repo truth:

- `ctest -N --test-dir build/quality-review-cmake` = `53`
- `tests/test_*.c` count = `53`
- `add_sparse_test(...)` count in `CMakeLists.txt` = `53`
- `Makefile` sets:
  - `COV_THRESHOLD = 80`

Interpretation:

- this is a wording/drift problem, not an execution-path problem
- the most important Day 5 cleanup should start in `README.md`, because that is
  currently the highest-signal place where Sprint 29's coverage calibration and
  Sprint 32's test-truthfulness work are still being overstated

#### 2. The actual opt-in test contract is broader than the README currently teaches

The framework-level contract is truthful for two explicit opt-in wrappers:

- `RUN_TEST_SLOW(...)` gated by `SPARSE_TEST_SLOW=1`
- `RUN_TEST_EXPERIMENTAL(...)` gated by `SPARSE_TEST_EXPERIMENTAL=1`

But the live suite also still contains at least one separate documented opt-in
surface outside those wrappers:

- `tests/test_suitesparse.c` gates large-matrix cases behind
  `SPARSE_TEST_LARGE=1`

Interpretation:

- the current README test-category section is incomplete, not fully wrong
- it teaches the wrapper-based opt-in categories correctly, but it still reads
  too much like those are the only non-default live categories
- this is again primarily a wording/truthfulness issue, not a demand for new
  gating infrastructure

#### 3. Coverage today measures the default active suite, not the full universe of opt-in or intentionally-skipped paths

The coverage targets currently do the following:

- rebuild tests with coverage instrumentation
- run the default test-binary set
- aggregate failures across the full run
- check line coverage against `COV_THRESHOLD = 80`

They do not, by default, enable:

- `SPARSE_TEST_SLOW=1`
- `SPARSE_TEST_EXPERIMENTAL=1`
- `SPARSE_TEST_LARGE=1`

Interpretation:

- this is not a bug by itself
- it means Sprint 38 coverage wording must say "default active regression
  surface + current intentional skips/opt-ins as configured" rather than
  implying that coverage is measuring every live optional path automatically

#### 4. The current coverage wording mixes three different ideas that should be separated

Today the repo uses "coverage" to refer to three different surfaces:

- line-coverage instrumentation reports:
  - `make coverage`
  - `make coverage-lcov`
  - `make coverage-gcovr`
- default regression execution:
  - `make test`
  - `ctest`
- broad feature presence / category truthfulness:
  - live slow/experimental/large test paths
  - intentionally skipped fixture/environment-sensitive branches

Interpretation:

- the implementation is mostly fine
- the language is what needs cleanup
- later Sprint 38 docs should separate:
  - default executed regression surface
  - opt-in executed regression surface
  - instrumented source line coverage surface

#### 5. The CI coverage story is real, but it is supplemental rather than part of the reviewed baseline

Current repo truth:

- Linux CI has a dedicated supplemental `coverage` job
- `README.md` cross-platform contract already classifies coverage as
  supplemental on Linux
- coverage is not part of:
  - `make quality-review-compile`
  - `make quality-review`
  - `make quality-review-cmake-compile`
  - `make quality-review-cmake`

Interpretation:

- the underlying contract is already sound
- the main remaining risk is that older README wording ("CI-enforced" and the
  broader testing intro) can still be read as if coverage were part of the same
  reviewed baseline as format/lint/test/dead-code

#### 6. The keep/fix/defer split is already clear

Keep as-is:

- `tests/test_framework.h` opt-in wrapper mechanics
- `tests/test_framework_optin.c` self-check surface
- `Makefile` `COV_THRESHOLD = 80`
- Linux supplemental coverage CI job
- `INSTALL.md` explanation of backend split:
  - GCC + lcov path
  - Apple Clang + gcovr path

Fix in the first Sprint 38 coverage batch:

- stale README threshold/suite-count sentence
- README wording that implies `make test` or `make coverage` means "all live
  tests" without qualification
- README test-category wording to mention the separate `SPARSE_TEST_LARGE`
  opt-in surface
- coverage wording that should distinguish supplemental CI coverage from the
  reviewed baseline

Defer for later Sprint 38 or beyond:

- any attempt to make coverage include all opt-in categories by default
- any new cross-platform coverage parity expansion
- any threshold recalibration work; the current Sprint 29 `80` decision remains
  the operative truth source unless new measured evidence justifies reopening it
