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
