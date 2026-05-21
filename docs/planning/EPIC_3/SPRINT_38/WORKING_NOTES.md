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

## Day 3

**Objective:** Audit the named Sprint 34 exclusion-list binaries directly so
Sprint 38 can distinguish true compile-only protection gaps from dead-code
compile-db/reporting gaps that are merely being described too broadly.

### Commands Run

1. Re-read the Sprint 38 Day 3 plan section:
   - `sed -n '1,220p' docs/planning/EPIC_3/SPRINT_38/PLAN.md`
2. Re-scan current docs and prior sprint notes for the named exclusion list:
   - `rg -n "bench_svd|example_basic_solve|example_condition|example_iterative|example_least_squares|example_matrix_free|example_svd_lowrank|tooling-build|examples-build|bench-build|BENCH_BINS|EX_BINS" Makefile CMakeLists.txt README.md INSTALL.md docs/planning/EPIC_3/SPRINT_{34,36,37,38}/* docs/planning/EPIC_3/SPRINT_{34,36,37,38}/artifacts/*`
3. Re-read the Makefile benchmark/example build surface:
   - `sed -n '120,270p' Makefile`
   - `make -n tooling-build`
4. Re-read the CMake benchmark/example registration surface:
   - `sed -n '220,320p' CMakeLists.txt`
   - `rg -n "add_executable\\((bench_svd|example_basic_solve|example_condition|example_iterative|example_least_squares|example_matrix_free|example_svd_lowrank)|NOT WIN32|add_sparse_test\\(" CMakeLists.txt`
5. Confirm the current file inventory for benchmarks/examples:
   - `ls examples/*.c benchmarks/*.c`
   - `python3` count/list of `examples/*.c` and `benchmarks/*.c`
6. Recheck actual dead-code compile-db and coverage-notes status:
   - `python3` membership check of the seven named source files in `build/deadcode-cmake/compile_commands.json`
   - `sed -n '1,220p' build/deadcode/coverage-notes.txt`
7. Re-read the current README dead-code limitation wording:
   - `sed -n '640,680p' README.md`

### Day 3 Findings

#### 1. The named exclusion list is no longer a Makefile compile-only protection gap

Current Makefile truth:

- `BENCH_SRCS` explicitly includes:
  - `bench_svd.c`
- `EX_SRCS = $(wildcard examples/*.c)` includes:
  - `example_basic_solve.c`
  - `example_condition.c`
  - `example_iterative.c`
  - `example_least_squares.c`
  - `example_matrix_free.c`
  - `example_svd_lowrank.c`
- `tooling-build` depends on:
  - `bench-build`
  - `examples-build`
- `lint` depends on:
  - `tooling-build`

Interpretation:

- all seven named binaries already compile under the maintained Makefile
  compile-only path
- the Sprint 34 wording is now too broad if it still implies these are missing
  from compile-only protection in general

#### 2. The seven named binaries are still absent from the dead-code CMake compile database

Current `build/deadcode-cmake/compile_commands.json` truth:

- `bench_svd.c` = missing
- `example_basic_solve.c` = missing
- `example_condition.c` = missing
- `example_iterative.c` = missing
- `example_least_squares.c` = missing
- `example_matrix_free.c` = missing
- `example_svd_lowrank.c` = missing

Current `build/deadcode/coverage-notes.txt` truth:

- benchmarks = `13`
- examples = `6`
- `missing_benchmarks`:
  - `bench_svd`
- `missing_examples`:
  - `example_basic_solve`
  - `example_condition`
  - `example_iterative`
  - `example_least_squares`
  - `example_matrix_free`
  - `example_svd_lowrank`

Interpretation:

- the exclusion list is still real
- but it is specifically a dead-code compile-db/reporting gap, not a Makefile
  compile-only protection gap

#### 3. The root cause is narrower than “CMake parity misses random tooling surface”

Current CMake truth:

- examples registered in `CMakeLists.txt`:
  - `example_ldlt`
  - `example_ic_minres`
  - `example_analysis`
  - `example_minnorm`
  - `example_colamd`
  - `example_eigs`
- the six named missing examples are not registered
- benchmark registration includes many bench binaries, but still omits the named
  `bench_svd` target from the dead-code compile-db tree

Interpretation:

- the exclusion list is caused by a bounded CMake registration gap
- this is smaller and more actionable than a generic "compile_commands is
  unreliable" story

#### 4. The repo currently mixes two different compile-protection stories that should be separated

Current repo behavior:

- Makefile reviewed/compile-only path:
  - already compiles all `14` benchmarks
  - already compiles all `12` examples
- dead-code compile-db/reporting path:
  - still covers only `13` benchmarks
  - still covers only `6` examples

Interpretation:

- Day 3's main audit result is a vocabulary split:
  - compile-only regression protection through `tooling-build` is largely closed
  - dead-code compile-db/reporting coverage for those seven files remains open
- later Sprint 38 work should stop using the phrase "compile-only gap" for the
  whole list unless it is explicitly talking about the dead-code CMake path

#### 5. The keep/fix/defer split is already clear

Keep as-is:

- `tooling-build` as the Makefile compile-only regression surface
- `lint -> tooling-build` as the maintained local ingress for benchmark/example
  compile protection
- current README statement that the dead-code compilation database under-covers
  part of the Makefile tooling surface

Fix in Sprint 38:

- docs/notes that still describe the seven files as generic compile-only drift
  instead of dead-code compile-db/reporting drift
- whichever strongest safe follow-through Day 6 chooses:
  - broaden the dead-code CMake compile-db to include some or all of the seven
    files, or
  - re-document the exclusion list more precisely as a dead-code/reporting limit

Defer unless later evidence changes:

- any attempt to move routine runtime execution of these binaries into the
  reviewed baseline
- any claim that reviewed CMake parity should own the full benchmark/example
  compile-only surface by default

## Day 4

**Objective:** Reassess the current dead-code workflow artifacts, buckets, and
execution model so Sprint 38 can distinguish worthwhile signal-quality
improvements from changes that would overclaim concurrency safety or cleanup
readiness.

### Commands Run

1. Re-read the Sprint 38 Day 4 plan section:
   - `sed -n '1,260p' docs/planning/EPIC_3/SPRINT_38/PLAN.md`
2. Re-read the current generated dead-code report artifacts:
   - `sed -n '1,260p' build/deadcode/report.md`
   - `sed -n '1,220p' build/deadcode/report.tsv`
   - `sed -n '1,260p' build/deadcode/cppcheck.txt`
3. Re-read the current dead-code workflow implementation:
   - `sed -n '1,260p' scripts/deadcode_workflow.sh`
   - `sed -n '1,260p' scripts/deadcode_report.py`
4. Summarize current report buckets/dispositions from `report.tsv`:
   - `python3` bucket/disposition counter over `build/deadcode/report.tsv`
5. Cross-check the inherited Sprint 33 dead-code contract and deferred queue:
   - `sed -n '1,180p' docs/planning/EPIC_3/SPRINT_33/HANDOFF.md`
   - `sed -n '1,140p' docs/planning/EPIC_3/SPRINT_33/RETROSPECTIVE.md`
6. Re-scan current maintainer docs for dead-code limitation wording:
   - `rg -n "deadcode\\*|serially|shared .*build/deadcode-cmake|coverage-gap|secondary signals|supporting evidence|noise" README.md docs/planning/EPIC_3/SPRINT_{33,34,36,37}/HANDOFF.md docs/planning/EPIC_3/SPRINT_{33,34,36,37}/RETROSPECTIVE.md`

### Day 4 Findings

#### 1. The dead-code workflow is stable as an advisory/report-completeness gate, not yet as a stronger cleanup gate

Current live report buckets are:

- `coverage-gap` = `7`
- `public-surface-review` = `4`
- `secondary-candidate-signal` = `35`
- `non-deadcode-static-analysis-noise` = `6`
- `definitely-unused-internal-candidate` = `0`

Current dispositions are also stable:

- `defer-until-compile-db-expanded` = `7`
- `keep-public-api-day8-audited` = `4`
- `summarize-only-supporting-evidence` = `35`
- `appendix-only-not-cleanup-candidate` = `6`

Interpretation:

- the current workflow already succeeds at one important job:
  - it turns raw scanner output into explicit, reviewable classes
- but it does **not** currently surface a new cleanup-ready queue beyond the
  already-audited public keeps and deferred evidence buckets

#### 2. Since Sprint 33, the workflow has been revalidated more than fundamentally matured

Compared to the Sprint 33 handoff/retrospective:

- the bucket counts are unchanged except for the already-closed internal
  candidate queue staying at `0`
- the same three deferred areas remain:
  - compile-db coverage gap
  - shared-path serialization
  - residual `cppcheck` supporting-signal / noise review

Interpretation:

- Sprint 38 Day 4 does not uncover a hidden new dead-code debt queue
- it confirms that the next maturity step should focus on report signal and
  scope clarity, not on pretending the workflow is ready for a more aggressive
  cleanup or concurrency contract

#### 3. The strongest currently actionable dead-code signal is still report completeness, not content-based failure

Current `deadcode-check` truth model remains:

- generated report exists
- every `xunused` finding is categorized
- coverage-gap section is present

Current report content shows why that remains the right enforced boundary:

- `coverage-gap` rows are still expected and truthful
- public-surface rows are already explicitly audited keeps
- `cppcheck` secondary signals are still summarized as supporting evidence only
- `cppcheck` noise rows are still appendix-only and not cleanup candidates

Interpretation:

- a stronger content-based failure rule would still be premature
- the more promising Sprint 38 refinement space is:
  - clearer report structure
  - tighter explanation of what each bucket means
  - better alignment between the report and the named deferred queue

#### 4. The shared-path execution model remains the main blocker for stronger local/CI enforcement assumptions

Current workflow implementation still shares:

- `build/deadcode-cmake`
- `build/deadcode/`

Current maintainer docs still correctly say:

- run `deadcode*` targets serially
- concurrent invocation can race on the shared CMake build tree

Interpretation:

- Sprint 38 should not treat dead-code as concurrency-safe yet
- any future stronger local/CI enforcement claims still depend on either:
  - path isolation, or
  - preserving explicit serialized execution

#### 5. Day 3 narrowed the compile-gap story, which changes the best dead-code framing

Day 3 established that the seven named files are:

- already compile-protected by Makefile `tooling-build`
- still omitted from dead-code compile-db/reporting coverage

Interpretation:

- the dead-code workflow's leading unresolved gap is now more precisely named:
  - partial compile-db/report coverage
  - not generic benchmark/example compile drift
- that makes the dead-code report easier to improve, because the coverage-gap
  section can be framed as a bounded tooling-scope limit rather than a broad
  compile-health warning

#### 6. The actionable/staged/defer split is already clear

Actionable enough for Sprint 38 refinement:

- report wording/structure for the existing buckets
- alignment between `report.md`, `report.tsv`, `coverage-notes.txt`, and README
- clearer next-action guidance that distinguishes:
  - no current cleanup-ready internal queue
  - audited public keeps
  - supporting-only `cppcheck` evidence

Still staged / not ready for stronger enforcement:

- treating `secondary-candidate-signal` as failure-worthy content
- treating `non-deadcode-static-analysis-noise` as cleanup instructions
- assuming concurrent-safe dead-code invocation
- treating compile-db silence on the seven excluded files as meaningful

Best Day 7 design target:

- improve signal quality and operator clarity inside the current staged model
- do not turn the advisory report into a stronger correctness gate until the
  compile-db coverage and execution-model limits are addressed

## Day 6

**Objective:** Close the strongest remaining compile-only/dead-code coverage
gap from the Day 3 audit by bringing the seven named exclusion-list binaries
under the dead-code CMake compile-db/reporting surface, while keeping the batch
structural and low-risk.

### Commands Run

1. Re-read the Sprint 38 Day 6 plan section:
   - `sed -n '1,220p' docs/planning/EPIC_3/SPRINT_38/PLAN.md`
2. Re-read the current CMake benchmark/example registration block:
   - `sed -n '220,340p' CMakeLists.txt`
3. Re-read the omitted source files to confirm their CMake target shape:
   - `sed -n '1,220p' benchmarks/bench_svd.c`
   - `sed -n '1,200p' examples/example_basic_solve.c`
   - `sed -n '1,200p' examples/example_condition.c`
   - `sed -n '1,200p' examples/example_iterative.c`
   - `sed -n '1,200p' examples/example_least_squares.c`
   - `sed -n '1,200p' examples/example_matrix_free.c`
   - `sed -n '1,200p' examples/example_svd_lowrank.c`
4. Reconfirm the dead-code/report entry points before editing:
   - `make -n deadcode-report`
   - `make -n deadcode-check`
5. Validate the batch after editing:
   - `make deadcode-report && make deadcode-check`
   - `python3 -m py_compile scripts/deadcode_report.py`
   - direct checks of:
     - `build/deadcode-cmake/compile_commands.json`
     - `build/deadcode/coverage-notes.txt`
     - `build/deadcode/report.md`
     - `build/deadcode/report.tsv`

### Day 6 Findings

#### 1. The strongest Day 3 follow-through was the implementation path, not another documentation-only pass

Day 3 established that the seven named files were already under the maintained
Makefile compile-only path, but still absent from the dead-code compile
database. Day 6 chose the stronger follow-through:

- add `bench_svd` to the dead-code CMake benchmark registration surface
- add the six omitted examples to the dead-code CMake example registration
  surface

Interpretation:

- Sprint 38 now closes the concrete dead-code/reporting exclusion list instead
  of just re-describing it more precisely

#### 2. One real target-specific detail surfaced during implementation

The initial `bench_svd` CMake target registration was not sufficient by itself:

- `bench_svd.c` includes the private header:
  - `sparse_svd_internal.h`
- so the new target also needed:
  - private `${CMAKE_CURRENT_SOURCE_DIR}/src` include dirs

Interpretation:

- this was a genuine implementation detail, not a conceptual blocker
- the final Day 6 batch remains structural and low-risk because the fix stayed
  localized to target registration

#### 3. The dead-code compile-db coverage gap is now closed for the named Sprint 34 exclusion list

Post-batch artifact truth:

- `build/deadcode/coverage-notes.txt` now reports:
  - `benchmarks 14`
  - `examples 12`
- `missing_benchmarks` is empty
- `missing_examples` is empty

`build/deadcode/report.md` now reports:

- compile-db translation-unit counts:
  - `src=25`
  - `tests=53`
  - `benchmarks=14`
  - `examples=12`
- `## Coverage Gaps`
  - `No current benchmark/example compile-db coverage gaps are recorded in this run.`

`build/deadcode/report.tsv` bucket counts are now:

- `public-surface-review` = `4`
- `secondary-candidate-signal` = `35`
- `non-deadcode-static-analysis-noise` = `6`
- `coverage-gap` = `0`

Interpretation:

- the Day 3 named exclusion list is now closed in the dead-code/reporting path
- the remaining dead-code maturity queue is now more cleanly about:
  - report/check signaling
  - shared-path serialization
  - residual `cppcheck` supporting evidence

#### 4. The report wording also needed a small truthfulness adjustment once the gap closed

After the compile-db expansion, the old report intro for `## Coverage Gaps`
became misleading when the bucket was empty.

Day 6 therefore also updated `scripts/deadcode_report.py` so the markdown now
switches cleanly between:

- gap-present wording when rows exist
- `No current benchmark/example compile-db coverage gaps are recorded in this run.`
  when the bucket is empty

Interpretation:

- this is still part of the compile-only batch because otherwise the new
  zero-gap state would have been described with stale language

#### 5. The residual compile-only queue is now smaller and better scoped

Closed in Day 6:

- `bench_svd`
- `example_basic_solve`
- `example_condition`
- `example_iterative`
- `example_least_squares`
- `example_matrix_free`
- `example_svd_lowrank`

Still open after Day 6:

- none from the named Sprint 34 compile-db exclusion list

Residual adjacent work belongs to later Sprint 38 days:

- dead-code report/check maturation
- shared-path serialization / execution-model limits
- readiness/reporting polish

## Day 5

**Objective:** Apply the safest first coverage-honesty cleanup slice from the
Day 2 audit by fixing the highest-signal README wording drift without changing
test execution behavior, coverage threshold policy, or reviewed gate semantics.

### Commands Run

1. Re-read the Sprint 38 Day 5 plan section:
   - `sed -n '1,220p' docs/planning/EPIC_3/SPRINT_38/PLAN.md`
2. Re-read the current README testing/coverage sections before editing:
   - `sed -n '548,700p' README.md`
   - `sed -n '110,140p' README.md`
3. Reconfirm the current maintained CTest count:
   - `ctest -N --test-dir build/quality-review-cmake`
4. Reconfirm the exact stale README matches before editing:
   - `rg -n "1453 unit tests|42 test suites|95% line coverage|CI-enforced|SPARSE_TEST_LARGE|RUN_TEST_SLOW|RUN_TEST_EXPERIMENTAL|make coverage" README.md`
5. After editing, run a doc sanity sweep against the relevant truth sources:
   - `rg -n "53|80%|SPARSE_TEST_LARGE|make coverage" README.md`
   - targeted cross-check against:
     - `Makefile`
     - `tests/test_framework.h`
     - `tests/test_suitesparse.c`

### Day 5 Findings

#### 1. The highest-value coverage-honesty batch really was concentrated in the README

The Day 2 audit was accurate: the strongest current truthfulness drift was not
distributed across many surfaces. It was concentrated in the highest-visibility
README language:

- stale top-level suite-count claim
- stale `95%` threshold claim
- stale "CI-enforced" wording that could be read too broadly
- missing mention of the large-matrix live opt-in test surface

Interpretation:

- Day 5 stayed intentionally narrow for the right reason
- the batch improves the main user/operator truth source without reopening the
  underlying gate machinery

#### 2. The README now separates default regression execution from supplemental coverage more honestly

The updated README now says directly:

- the maintained default regression surface is `53` registered CTest test
  binaries
- coverage is a supplemental signal
- Linux coverage enforcement is an `80%` line-coverage threshold on `src/` for
  the default instrumented run path

Interpretation:

- this is a meaningful truthfulness improvement because it removes the old
  implied equivalence between:
  - default regression execution
  - full live test surface
  - coverage instrumentation

#### 3. The opt-in test contract is now more complete in the main user-facing doc

The README test-category policy already described:

- `RUN_TEST_SLOW(...)`
- `RUN_TEST_EXPERIMENTAL(...)`

Day 5 added the missing suite-local live opt-in surface:

- `SPARSE_TEST_LARGE=1 make test`

Interpretation:

- the public contract is now closer to the actual repo shape
- the wrapper-based categories remain the main model, but the README no longer
  implies they are the only non-default live test surface

#### 4. This batch intentionally did not change behavior, thresholds, or reviewed-gate ownership

Day 5 did **not** change:

- `COV_THRESHOLD = 80`
- coverage backend selection
- reviewed local wrapper behavior
- default `make test` behavior
- automatic enabling of slow / experimental / large opt-in paths inside
  `make coverage`

Interpretation:

- this was the right first batch because it improves truthfulness with minimal
  semantic risk
- any later stronger changes still need a separate audited justification

#### 5. The residual coverage-honesty queue is now narrower

After Day 5, the remaining Sprint 38 coverage-honesty work is smaller:

- check for nearby docs that still flatten:
  - default regression execution
  - opt-in execution
  - line-coverage instrumentation
- reinforce in later readiness/report wording that coverage is supplemental,
  not part of the reviewed baseline

Interpretation:

- the highest-signal README drift is now addressed
- the remaining queue is mostly secondary consistency work, not another large
  wording correction batch

## Day 7

**Objective:** Turn the Day 4 audit plus the Day 6 compile-db closure into a
concrete dead-code maturation design that improves routine signal quality
without overstating concurrency safety or content-based readiness.

### Commands Run

1. Re-read the Sprint 38 Day 7 plan section:
   - `sed -n '1,260p' docs/planning/EPIC_3/SPRINT_38/PLAN.md`
2. Re-read the Day 4 dead-code audit:
   - `sed -n '1,260p' docs/planning/EPIC_3/SPRINT_38/artifacts/day4-deadcode-workflow-maturation-audit.md`
3. Re-read the current dead-code report after Day 6:
   - `sed -n '1,220p' build/deadcode/report.md`
4. Re-check the current dead-code report implementation surface:
   - `sed -n '1,260p' scripts/deadcode_report.py`

### Day 7 Findings

#### 1. The compile-db closure changes the dead-code design target materially

After Day 6, the dead-code report no longer has the old named benchmark/example
coverage gap:

- `coverage-gap = 0`
- `definitely-unused-internal-candidate = 0`
- `public-surface-review = 4`
- `secondary-candidate-signal = 35`
- `non-deadcode-static-analysis-noise = 6`

Interpretation:

- the workflow is now narrower and more stable
- the next maturity step should improve operator signal inside this cleaner
  state rather than reopening compile-surface work

#### 2. The strongest remaining dead-code limitation is still execution-model truthfulness

The report is cleaner, but the workflow still depends on shared serialized
paths:

- `build/deadcode-cmake`
- `build/deadcode/`

Interpretation:

- Sprint 38 still should not imply concurrent-safe `deadcode*` execution
- shared-path isolation remains later work, not part of the next report batch

#### 3. The safest Day 8 batch is report/check refinement, not stronger failure logic

The current check boundary still proves:

- report generation succeeded
- every `xunused` finding was categorized
- report completeness invariants held

It still does **not** prove:

- zero findings
- zero `cppcheck` noise
- cleanup-ready internal removals exist
- fully parallel-safe execution

Interpretation:

- the next useful step is to make the report/check output say this more
  directly
- it is still too early to add content-based pass/fail rules over the
  `cppcheck` buckets

#### 4. The current next-action wording can now become more direct

With the compile-db gap closed and the internal cleanup queue empty, the report
can now speak more plainly:

- no current compile-db coverage gaps
- no current definitely-unused internal cleanup queue
- public-surface rows are audited keeps
- `cppcheck` rows remain supporting evidence only

Interpretation:

- this is the highest-value Day 8 improvement because it reduces operator
  ambiguity without changing the staged contract

#### 5. The Day 8 batch is now tightly bounded

Chosen next batch:

- refine `build/deadcode/report.md` wording/section shape for the zero-gap state
- refine `deadcode-check` / report messaging only if needed to make the
  completeness-vs-cleanup boundary more explicit
- preserve serial authoritative validation:
  - `make deadcode-report`
  - `make deadcode-check`

Not in the next batch:

- stronger content-based failure rules
- shared-path isolation
- multi-platform dead-code expansion
- reclassification of public audited keeps into cleanup work

Interpretation:

- the dead-code maturity batch is now small, deliberate, and low-risk

## Day 8

**Objective:** Implement the first dead-code maturation batch by refining the
report/check signal for the current zero-gap state without changing bucket
semantics, adding content-based failure logic, or implying concurrent-safe
execution.

### Commands Run

1. Re-read the Sprint 38 Day 8 design note:
   - `sed -n '1,260p' docs/planning/EPIC_3/SPRINT_38/artifacts/day7-deadcode-workflow-maturation-design.md`
2. Re-read the current report generator and operator-facing messages:
   - `sed -n '1,260p' scripts/deadcode_report.py`
   - `sed -n '600,635p' Makefile`
3. Re-read the current rendered report before editing:
   - `sed -n '1,220p' build/deadcode/report.md`
4. Authoritative serial validation after editing:
   - `python3 -m py_compile scripts/deadcode_report.py`
   - `make deadcode-report`
   - `make deadcode-check`

### Day 8 Findings

#### 1. The Day 8 batch improved signal quality without changing any bucket semantics

The batch changed wording and next-action structure only:

- no bucket names changed
- no classifier rules changed
- no `deadcode-check` content-based failure logic changed

Interpretation:

- the dead-code workflow remains the same staged completeness gate
- the batch reduced operator ambiguity without creating new cleanup claims

#### 2. The zero-gap state is now described directly as closure, not lingering debt

The rendered `## Coverage Gaps` section now says:

- no current benchmark/example compile-db gaps are recorded
- the dead-code compile database currently covers the maintained
  benchmark/example tooling surface

Interpretation:

- this closes the old Sprint 34 exclusion-list story explicitly in the report
- the workflow no longer reads like it is still waiting on that compile-db gap

#### 3. The empty internal queue is now described as an empty cleanup batch

The rendered `## Definitely-Unused Internal Candidates` section now says:

- no current definitely-unused internal cleanup batch is classified in this run

Interpretation:

- this is more useful than the older generic "none in this bucket" wording
- it makes the current actionable state clearer for maintainers skimming the
  report

#### 4. Supporting evidence and audited keeps are now framed more explicitly as non-actionable

The report now says more directly:

- public rows are audited keeps, not cleanup
- `cppcheck` secondary signals are not direct removal instructions or pass/fail
  criteria
- the next-action queue keeps those rows visible for context rather than
  implying a pending removal batch

Interpretation:

- this was the highest-value signal refinement left after the Day 6 compile-db
  closure
- the report now better matches the real staged boundary

#### 5. The operator-facing `deadcode-check` message now matches the staged contract more closely

`make deadcode-check` now ends with:

- report completeness checks passed
- not a zero-findings gate
- authoritative execution remains serialized

Interpretation:

- this aligns the target output with the actual workflow contract
- it reduces the risk of reading a successful check as "no dead-code evidence
  remains"

#### 6. The residual dead-code queue is now narrower and cleaner

Still remaining after Day 8:

- shared-path serialization / execution-model limits
- future review of `cppcheck` supporting evidence and summarized noise
- later readiness/reporting integration work

Closed by Day 8:

- zero-gap report wording drift
- generic empty-queue wording drift
- ambiguous `deadcode-check` success wording

## Day 9

**Objective:** Re-audit the current reviewed gate boundaries after the Day 5
through Day 8 truthfulness/compile-db/dead-code work and choose the smallest
meaningful next-tier gate expansion that strengthens regression protection
without collapsing the Sprint 36 enforced/staged/excluded platform contract.

### Commands Run

1. Re-read the Sprint 38 Day 9 plan section:
   - `sed -n '180,320p' docs/planning/EPIC_3/SPRINT_38/PLAN.md`
2. Re-read the Sprint 38 project-plan item wording:
   - `sed -n '296,332p' docs/planning/EPIC_3/PROJECT_PLAN.md`
3. Re-read the current quality command map and cross-platform CI contract:
   - `sed -n '1,260p' README.md`
4. Re-read the current reviewed/dead-code Makefile surface:
   - `sed -n '470,640p' Makefile`
5. Re-read the current CI workflow contract surfaces:
   - `sed -n '1,260p' .github/workflows/ci.yml`
   - `sed -n '1,260p' .github/workflows/macos-ci.yml`
   - `sed -n '1,260p' .github/workflows/windows-ci.yml`

### Day 9 Findings

#### 1. The current enforced baseline is already broad; the next useful expansion is aggregation, not a new primitive gate

The repo already has stable maintained reviewed primitives for:

- reviewed Makefile compile-quality:
  - `make quality-review-compile`
- reviewed local Makefile quality:
  - `make quality-review`
- reviewed CMake parity:
  - `make quality-review-cmake-compile`
  - `make quality-review-cmake`
- dead-code completeness/reporting:
  - `make deadcode-report`
  - `make deadcode-check`

Interpretation:

- the next useful hardening step is not another low-level target
- it is a clearer next-tier aggregate over the maintained reviewed paths that
  already exist

#### 2. Coverage still should not be folded into the reviewed baseline

After Day 5, coverage wording is now explicit:

- coverage is supplemental
- the Linux coverage job enforces the current `80%` threshold
- it is not part of the cross-platform reviewed baseline

Interpretation:

- Sprint 38 still should not expand the reviewed gate by folding `make coverage`
  into a top-level reviewed wrapper
- that would blur the truthfulness work from Day 2 and Day 5

#### 3. macOS dead-code and Windows local Makefile parity are still staged for good reasons

The Sprint 36 contract remains accurate even after Day 6 and Day 8:

- macOS dead-code is still staged
- Windows local Makefile reviewed wrappers are still staged
- Windows dead-code is still excluded

Why this remains correct:

- dead-code still depends on serialized shared paths
- `xunused` setup remains more toolchain-sensitive than the reviewed CMake path
- Windows still has an honest enforced subset centered on direct CMake

Interpretation:

- the next gate-expansion batch should not try to "promote" these staged paths
  by documentation fiat
- cross-platform honesty remains more important than symmetry

#### 4. The highest-value narrow expansion is a single local aggregate reviewed baseline target

Current local operator reality:

- maintainers can run the full practical reviewed baseline today
- but it is split across two top-level commands:
  - `make quality-review`
  - `make quality-review-cmake`

That means there is still no single named top-level local command for:

- reviewed Makefile path
- reviewed CMake parity path

Interpretation:

- adding one explicit aggregate wrapper is the smallest meaningful expansion
- it strengthens routine local regression-proofing without changing any staged
  platform boundary

#### 5. The Day 10 batch is now tightly bounded

Chosen Day 10 batch:

- add one serial top-level reviewed aggregate wrapper over:
  - `make quality-review`
  - `make quality-review-cmake`
- document it as the strongest local reviewed baseline command
- keep `coverage`, `wall-check`, `sanitize`, and cross-platform staged/excluded
  paths outside that aggregate

Not chosen for Day 10:

- enforcing macOS dead-code
- enforcing Windows local Makefile reviewed wrappers
- folding `make coverage` into the reviewed baseline
- changing the current Linux/macOS/Windows CI contract

Interpretation:

- the next expansion is real but low-risk
- it improves local regression-proofing and future readiness-checklist clarity
  without destabilizing current CI or platform-truthfulness boundaries

## Day 10

**Objective:** Implement the smallest meaningful next-tier gate expansion from
the Day 9 design by adding one explicit local aggregate wrapper over the
already-stable reviewed Makefile and reviewed CMake paths.

### Commands Run

1. Re-read the Sprint 38 Day 10 design note:
   - `sed -n '1,260p' docs/planning/EPIC_3/SPRINT_38/artifacts/day9-quality-gate-expansion-design.md`
2. Re-read the current reviewed wrapper surfaces before editing:
   - `sed -n '470,560p' Makefile`
   - `sed -n '100,140p' README.md`
   - `sed -n '688,722p' README.md`
3. Validate the new aggregate wrapper wiring:
   - `make -n quality-review-full`
4. Validate the new aggregate wrapper live:
   - `make quality-review-full`

### Day 10 Findings

#### 1. The new gate expansion is real, but it stays entirely within the existing reviewed contract

The new top-level wrapper is:

- `make quality-review-full`

It runs:

- `make quality-review`
- `make quality-review-cmake`

Interpretation:

- Sprint 38 expanded regression protection in one concrete area
- it did so without inventing any new lower-level gate semantics

#### 2. The wrapper makes the strongest local reviewed baseline explicit

Before Day 10, the full local reviewed baseline existed but required maintainers
to remember two separate top-level commands.

After Day 10:

- one command now names the strongest local reviewed baseline directly
- the README command map now reflects that explicitly

Interpretation:

- this is a small but meaningful reduction in routine regression-proofing drift
- it also sets up the later readiness checklist more cleanly

#### 3. Failure attribution stayed clear because the wrapper remained serial and bannered

The new wrapper preserves the same Day 9 design principles:

- serial execution
- explicit banners
- direct rerun guidance:
  - `make quality-review`
  - `make quality-review-cmake`

Interpretation:

- the expansion remains attributable
- the wrapper does not hide which half of the local reviewed baseline failed

#### 4. The batch preserved the Sprint 36 enforced/staged/excluded contract

Day 10 intentionally did **not** change:

- Linux CI job ownership
- macOS dead-code staging
- Windows local Makefile reviewed-wrapper staging
- Windows dead-code exclusion
- coverage being supplemental rather than part of the reviewed baseline

Interpretation:

- the expansion strengthened local routine use without collapsing the
  cross-platform honesty model

#### 5. The live reviewed baseline passed under the new wrapper

`make quality-review-full` passed end to end:

- reviewed Makefile path passed
- reviewed CMake parity path passed
- `ctest -N` remained `53`
- full reviewed CMake `ctest` passed `53 / 53`

Interpretation:

- the new wrapper is not just a naming change
- it is a validated aggregate over the maintained local reviewed baseline

## Day 11

**Objective:** Define a concise quality-readiness checklist that matches the
current cleaned-up repo reality, with one authoritative landing surface and
clear links outward to supporting detail rather than another broad duplicated
documentation layer.

### Commands Run

1. Re-read the Sprint 38 Day 11 plan section:
   - `sed -n '220,360p' docs/planning/EPIC_3/SPRINT_38/PLAN.md`
2. Re-read the Sprint 38 project-plan item wording:
   - `sed -n '296,332p' docs/planning/EPIC_3/PROJECT_PLAN.md`
3. Re-read the current README/INSTALL quality command and contract sections:
   - `rg -n "quality-review-full|quality-review|deadcode-check|coverage|Cross-Platform CI Contract|readiness|checklist|ctest -N|53 registered" README.md INSTALL.md docs -g '!docs/planning/**'`
4. Re-read the current Sprint 38 working-state tail:
   - `tail -n 180 docs/planning/EPIC_3/SPRINT_38/WORKING_NOTES.md`

### Day 11 Findings

#### 1. The checklist should live in README, not in a separate planning-only artifact

Current reality:

- `README.md` already owns the operator command map
- `README.md` already owns the cross-platform CI contract
- `INSTALL.md` already defers to the README for the canonical command map

Interpretation:

- the authoritative readiness checklist should land in `README.md`
- it should sit near the maintained quality command/contract material, not in a
  sprint-local or planning-only file

#### 2. The checklist should be compact and criterion-based, not a duplicate procedure manual

The repo already has detailed supporting material for:

- command entry points
- dead-code semantics
- coverage semantics
- cross-platform CI staging/exclusion boundaries

Interpretation:

- the checklist should answer "what must be true?" rather than "how does every
  target work?"
- supporting detail should be linked or referenced, not re-explained inline

#### 3. The canonical checklist scope is now narrow and stable

Chosen readiness criteria:

- local reviewed baseline:
  - `make quality-review-full`
- dead-code completeness/report truthfulness:
  - `make deadcode-report`
  - `make deadcode-check`
- active test surface / CMake parity truthfulness:
  - `53` registered CTest tests
  - `ctest -N` / reviewed CMake parity contract
- coverage truthfulness:
  - coverage remains supplemental
  - current enforced threshold is `80%` on `src/` in the Linux coverage path
- docs/examples truthfulness:
  - no stale contract drift between README, tutorial, headers, and maintained
    examples
- cross-platform parity truthfulness:
  - Linux reviewed baseline enforced
  - macOS/Windows staged or excluded surfaces remain named honestly

Interpretation:

- this is enough to be a real readiness checklist
- it avoids reopening broad design discussion about warning debt or dormant test
  scaffolding that earlier sprints already closed

#### 4. The checklist should explicitly separate enforced baseline from staged/supplemental signals

The most important truthfulness line to preserve is:

- reviewed local baseline
- reviewed cross-platform enforced baseline
- supplemental signals
- staged/excluded signals

Interpretation:

- the checklist should not imply that coverage, wall-check, sanitize, or staged
  dead-code/platform paths are required readiness gates in the same sense as
  the reviewed local baseline
- it should mention them only where they matter to truthful release/readiness
  interpretation

#### 5. The Day 12 batch is now tightly bounded

Chosen Day 12 shape:

- add one compact README readiness-checklist section
- place it near the maintained quality command/contract material
- apply only small surrounding wording polish if needed so the checklist reads
  cleanly

Keep:

- README as canonical checklist landing surface
- INSTALL as supporting setup doc
- sprint artifacts as audit history only

Link/reference:

- reviewed command map
- dead-code report/check explanation
- cross-platform CI contract

Defer:

- any large restructure of README sections
- any new CI job or target semantics
- any attempt to make staged/excluded surfaces look enforced

## Day 12

**Objective:** Ship the concise README quality-readiness checklist from the Day
11 design and keep any surrounding polish limited to the minimum needed for the
checklist to read cleanly in the maintained quality-contract section.

### Commands Run

1. Re-read the Day 11 readiness-checklist design note:
   - `sed -n '1,260p' docs/planning/EPIC_3/SPRINT_38/artifacts/day11-readiness-checklist-design.md`
2. Re-read the current README quality-contract area before editing:
   - `sed -n '680,790p' README.md`
   - `sed -n '100,140p' README.md`
3. Validate the touched docs surface after editing:
   - `rg -n "Quality Readiness Checklist|quality-review-full|deadcode-check|53|80%|Cross-Platform CI Contract|supplemental" README.md`
   - `sed -n '718,790p' README.md`

### Day 12 Findings

#### 1. The checklist lands cleanly in the README quality-contract section

The new checklist now lives in `README.md` directly under the existing quality
command/CI-contract material.

Interpretation:

- this matches the Day 11 landing-surface decision
- the checklist is now where maintainers already look for the maintained
  quality contract

#### 2. The checklist stayed criterion-based instead of becoming another command manual

The new section focuses on what must remain true:

- strongest local reviewed baseline
- dead-code report/check truthfulness
- reviewed CMake parity / current test-surface truthfulness
- supplemental coverage truthfulness
- docs/examples/header consistency
- honest cross-platform enforced/staged/excluded boundaries

Interpretation:

- the batch preserved the Day 11 scope discipline
- it reuses the existing command map, dead-code explanation, and CI contract
  instead of duplicating them

#### 3. The batch adds readability without changing any gate semantics

Day 12 did **not** change:

- any Makefile target behavior
- any CI workflow behavior
- any dead-code classifier/check semantics
- any coverage threshold or classification

Interpretation:

- this is exactly the right shape for a readiness/reporting batch
- the checklist improves operator clarity without inventing new enforcement

#### 4. The README now has one concise release/readiness pass that matches the current repo reality

The checklist now points to the real maintained truths:

- `make quality-review-full` as the strongest local reviewed baseline
- `make deadcode-report` / `make deadcode-check` as the dead-code truthfulness
  pair
- `53` current reviewed CTest registrations
- `80%` Linux supplemental coverage threshold on `src/`
- supporting links back to the existing dead-code and cross-platform contract
  sections

Interpretation:

- Sprint 38 now has the concise readiness surface it planned for
- later validation and closeout can reference this checklist directly

#### 5. The residual Day 12 queue is narrow

Still remaining after Day 12:

- any final small CI/report-output polish identified on Day 13 validation
- sprint closeout routing and final baseline recording

Closed by Day 12:

- lack of a concise canonical readiness checklist
- need to infer readiness criteria from multiple README sections by hand
