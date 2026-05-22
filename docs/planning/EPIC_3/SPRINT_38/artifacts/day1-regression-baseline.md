# Sprint 38 Day 1 Regression Baseline

**Date:** 2026-05-21  
**Branch:** `sprint-38`

## Objective

Turn the Sprint 34, Sprint 36, and Sprint 37 handoff state into a concrete
Sprint 38 starting inventory by confirming the inherited validated quality
contract, auditing the current coverage/gate/reporting surfaces, and naming the
first regression-proofing targets before implementation begins.

## Baseline Summary

Sprint 38 starts from the Sprint 37 close exactly as intended:

- no inherited warning-cleanup queue
- no inherited dead-code-baseline cleanup queue
- no inherited cross-platform contract ambiguity
- maintained direct gates already validated at close:
  - `make format`
  - `make lint`
  - `make test`
- maintained reviewed wrapper paths already validated at close:
  - `make quality-review-compile`
  - `make quality-review`
  - `make quality-review-cmake-compile`
  - `make quality-review-cmake`
- maintained support/reporting paths already validated at close:
  - `make deadcode-report`
  - `make deadcode-check`
  - `make wall-check`
- active reviewed CMake suite baseline remains:
  - `ctest -N --test-dir build/quality-review-cmake` = `53`

Current branch head during the Day 1 baseline capture:

- `6e5b9a3`

This means Sprint 38 is not a debt-burn-down sprint. It is a
regression-proofing sprint focused on truthfulness, compile-only protection,
dead-code maturity, and clearer readiness signaling.

## Current Regression-Proofing Surface

### Maintained direct and reviewed quality paths

- direct paths:
  - `format`
  - `format-check`
  - `lint`
  - `test`
  - `check`
- reviewed wrapper paths:
  - `quality-review-compile`
  - `quality-review`
  - `quality-review-cmake-compile`
  - `quality-review-cmake`

Interpretation:

- the repo already has named routine quality entry points
- Sprint 38 should improve what those paths prove and how their scope is
  reported, not invent a parallel quality-command layer

### Dead-code and reporting paths

- `deadcode-compile-db`
- `deadcode`
- `deadcode-report`
- `deadcode-check`

Current local artifacts present:

- `build/deadcode/report.md`
- `build/deadcode/report.tsv`
- `build/deadcode/cppcheck.txt`
- `build/deadcode/xunused.txt`
- `build/deadcode/coverage-notes.txt`
- `build/deadcode/.workflow.stamp`
- `build/deadcode/.report.stamp`

Interpretation:

- the dead-code workflow is operational enough to mature further
- Sprint 38 should treat it as a report/check system with known staged limits,
  not as a freshly invented workflow

### Compile-only and coverage/reporting paths

- compile-only or compile-adjacent protection:
  - `tooling-build`
  - `wall-check`
- coverage/reporting:
  - `coverage`
  - `coverage-lcov`
  - `coverage-gcovr`

Interpretation:

- Sprint 38 has a real surface for compile-only and coverage-honesty work
- the leading task is to reconcile actual protection/reporting behavior with
  what the repo claims those paths mean

### Local prerequisite tools currently available

- `cppcheck`
- `clang-tidy`
- `xunused`
- `gcovr`
- `ctest`

Interpretation:

- the local environment is ready for Sprint 38 audit and implementation work
- Day 1 is not blocked by missing tooling

## Inherited Constraints That Remain Open

### 1. Dead-code compile-db exclusion list is still open work

Still explicitly preserved from earlier handoffs:

- `bench_svd`
- `example_basic_solve`
- `example_condition`
- `example_iterative`
- `example_least_squares`
- `example_matrix_free`
- `example_svd_lowrank`

Implication:

- Sprint 38 compile-only regression work must either close or honestly
  re-document these surfaces
- it must not assume this queue is already solved

### 2. Dead-code shared-path serialization is still a real limitation

Still explicitly preserved from earlier handoffs:

- `build/deadcode-cmake`
- `build/deadcode/`

Implication:

- Sprint 38 can improve signal quality, report classification, or workflow
  topology
- it must not silently assume concurrent-safe dead-code enforcement before that
  limitation is addressed

### 3. Cross-platform reviewed/dead-code status remains intentionally asymmetric

Inherited platform contract still in force:

- Linux:
  - enforced reviewed Makefile path
  - enforced reviewed CMake path
  - enforced dead-code path
- macOS:
  - enforced Apple Clang reviewed path
  - staged dead-code path
  - supplemental GCC leg
- Windows:
  - enforced reviewed CMake subset
  - staged local Makefile reviewed-wrapper parity
  - excluded dead-code path

Implication:

- Sprint 38 quality-gate expansion must preserve staged vs enforced wording
- it should not collapse the platform contract into fake symmetry

### 4. The instrumentation/build-tree reset rule remains part of the maintainer contract

Still inherited from Sprint 36 and preserved through Sprint 37:

- tree-mutating or instrumented modes can pollute the normal `build/` tree
- `make clean` remains the canonical reset before returning to direct/reviewed
  baseline validation

Implication:

- Sprint 38 readiness and gate wording should keep this explicit where it
  affects operator expectations

## First Audit Targets

### Coverage-honesty audit

Most likely Day 2 concentration surface:

- coverage wording in maintainer/public docs
- interaction between default active tests and opt-in slow/experimental tests
- wording in report/check surfaces that could overstate what is truly covered

### Compile-only regression audit

Most likely Day 3 concentration surface:

- `bench_svd`
- `example_basic_solve`
- `example_condition`
- `example_iterative`
- `example_least_squares`
- `example_matrix_free`
- `example_svd_lowrank`

### Dead-code workflow maturation audit

Most likely Day 4 concentration surface:

- `scripts/deadcode_workflow.sh`
- `scripts/deadcode_report.py`
- current `report.md` / `report.tsv` bucket contract
- residual `cppcheck` evidence and noise buckets preserved from Sprint 33

### Readiness/reporting polish

Most likely later concentration surface:

- reviewed wrapper output
- CI artifact/report output
- concise release/readiness checklist covering:
  - warnings
  - dead code
  - test truthfulness
  - docs/examples consistency
  - cross-platform parity

## Day 1 Conclusion

Sprint 38 starts from a strong validated quality baseline and a clear bounded
problem statement:

- the repo already has maintained direct, reviewed, dead-code, coverage, and
  CI/reporting surfaces
- the main remaining work is not “add more targets”
- it is “make the existing targets, reports, exclusions, and readiness signals
  more truthful, more explicit, and harder to regress”

That is the right starting point for the rest of Sprint 38.
