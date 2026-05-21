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
