# Sprint 50 Day 2 Artifact: Validation Baseline and Truthfulness Recheck

## Purpose

Freeze the maintained validation and truthfulness baseline Sprint 50 must
preserve before the sprint goes deeper into direct-solver public-surface
inventory, lifecycle gap analysis, and bounded API design.

## Starting Truth

Sprint 50 inherits the same reviewed validation contract preserved at Epic 4
closeout:

- strongest local reviewed baseline:
  - `make quality-review-full`
- reviewed CMake parity anchor:
  - `ctest -N --test-dir build/quality-review-cmake`
- current reviewed CMake test count:
  - `53`

This means Sprint 50 does not need to define a new quality contract. It needs
to carry the maintained one forward while keeping its direct-solver design work
honest about what has and has not been rerun.

## Highest-Value Day 2 Rechecks

### 1. Maintained reviewed baseline wording

The live wrapper surface still says exactly:

- `quality-review-full: strongest local reviewed baseline`

That wording is reinforced in:

- `README.md`
- `docs/maintainer_guide.md`
- `Makefile`

This remains the authoritative local close-state phrase Sprint 50 should use.

### 2. Reviewed CMake parity truthfulness anchor

The current reviewed CMake parity surface still resolves to:

- `ctest -N --test-dir build/quality-review-cmake` = `53`

The suite still includes the direct-solver and structural tests most relevant
to the later Sprint 50 implementation surface:

- `test_cholesky`
- `test_ldlt`
- `test_etree`
- `test_chol_csc`
- `test_ldlt_csc`

That makes the reviewed CMake path the clearest count-based truthfulness anchor
for later public direct-lifecycle changes.

### 3. Authority split the sprint should preserve

The maintained quality contract is intentionally layered:

- `make quality-review-full`
  - strongest local reviewed baseline
- `make quality-review`
  - reviewed Makefile path
  - `format-check + lint + test + deadcode-check`
- `make quality-review-cmake`
  - reviewed CMake configure / rebuild / `ctest -N` / `ctest`
- `make deadcode-check`
  - report-completeness gate, not a zero-findings or removal-ready gate

Sprint 50 should keep using that split rather than rephrasing the contract in a
new sprint-local form.

## Sprint 50 Validation Boundary

### Design-sprint boundary

For docs-only and design-only days, Sprint 50 only needs to:

- preserve the maintained wording and meaning of:
  - `make quality-review-full`
  - reviewed CMake parity
  - `53` tests
- run targeted sanity checks appropriate to the touched docs and planning
  artifacts

### Later implementation boundary

When Sprint 50 or later direct-solver lifecycle work changes `*.c` or `*.h`,
the minimum required gate remains:

- `make format`
- `make lint`
- `make test`

For substantial public direct-lifecycle API batches, the stronger reviewed
baseline should also be rerun:

- `make quality-review-full`

## Candidate Touched-Surface Follow-On Set

Before implementation begins, the likely highest-signal direct-solver follow-on
surface is already explicit:

- examples:
  - `./build/example_analysis`
- benchmarks:
  - `./build/bench_refactor`
  - `./build/bench_refactor_csc`
- regression tests:
  - `./build/test_cholesky`
  - `./build/test_ldlt`
  - `./build/test_etree`
  - `./build/test_chol_csc`
  - `./build/test_ldlt_csc`

These do not replace the authoritative full gates. They are the most likely
high-signal follow-ons for later direct-solver lifecycle implementation days.

## Highest-Value Day 2 Conclusions

### 1. Sprint 50 should keep using the exact maintained “strongest local reviewed baseline” wording

The phrase is still live and authoritative in the wrapper surface, README, and
maintainer guide. Rewording it locally would add ambiguity instead of clarity.

### 2. The exact `53`-test reviewed CMake count remains the key truthfulness anchor

Later implementation work must preserve both the count and the Makefile/CMake
parity contract rather than treating the reviewed CMake path as a looser smoke
check.

### 3. The design sprint has a narrow validation boundary; later implementation sprints do not

Day 2 makes the split explicit:

- design-only work:
  - targeted sanity checks
  - preserved truthfulness language
- code-touch implementation work:
  - `make format`
  - `make lint`
  - `make test`
  - often `make quality-review-full`

### 4. The highest-value later direct-solver follow-ons are already explicit

The main implementation-era follow-on list is already stable enough to name:

- `example_analysis`
- `bench_refactor`
- `bench_refactor_csc`
- `test_cholesky`
- `test_ldlt`
- `test_etree`
- `test_chol_csc`
- `test_ldlt_csc`

That lets Day 3 and later design days stay focused on lifecycle and API shape
instead of drifting into validation-policy uncertainty.
