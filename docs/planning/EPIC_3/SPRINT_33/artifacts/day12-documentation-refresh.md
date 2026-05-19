# Sprint 33 Day 12: Documentation Refresh

## Goal

Update maintainer-facing documentation so the Sprint 33 dead-code workflow is
usable without reading the sprint notes or inferring policy from the scripts.

## Files Updated

- `README.md`
- `docs/planning/EPIC_3/SPRINT_33/WORKING_NOTES.md`

## README Changes

### 1. Build command list now includes the Sprint 33 targets

The Makefile quick-reference block now lists:

- `make deadcode`
- `make deadcode-report`
- `make deadcode-check`

This keeps the dead-code workflow visible alongside `make lint`, `make test`,
and the Sprint 31 `make tooling-build` compile-only gate.

### 2. Added a dedicated maintainer-facing "Dead-Code Workflow" section

The new section documents:

- what each target does
- the raw and classified artifact paths under `build/deadcode/`
- the local prerequisites:
  - `cppcheck`
  - `xunused`

### 3. Documented the interpretation contract explicitly

The README now states that the workflow is conservative evidence rather than
perfect reachability proof, and it separates:

- active code
- opt-in test code
- historical sprint evidence
- exported/public-surface findings
- secondary static-analysis signals

That keeps the Sprint 32 truthfulness policy and the Sprint 33 dead-code policy
visible in the normal maintainer docs instead of only in planning artifacts.

### 4. Documented the known workflow limits

The README now records two important Day 10 / Day 11 limits:

1. `xunused` currently relies on a compilation database that still misses:
   - `bench_svd`
   - `example_basic_solve`
   - `example_condition`
   - `example_iterative`
   - `example_least_squares`
   - `example_matrix_free`
   - `example_svd_lowrank`
2. `make deadcode*` targets should be run serially because they share
   `build/deadcode-cmake` and `build/deadcode/`; concurrent invocation can race
   on the shared CMake build tree.

## Documentation Outcome

After Day 12, a maintainer can answer the practical Sprint 33 questions from the
README alone:

- which commands exist
- which artifacts they produce
- what the report does and does not prove
- which classes of findings are deletion candidates versus review/defer buckets
- which local workflow limitations are known and intentional

## Validation

Day 12 was docs-only. Validation was a documentation sanity sweep:

1. re-read the edited README sections in place
2. checked the command names and artifact paths against the current Makefile and
   scripts
3. confirmed the listed coverage-gap examples match the current
   `build/deadcode/coverage-notes.txt`
