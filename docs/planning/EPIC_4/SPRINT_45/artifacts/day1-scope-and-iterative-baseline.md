# Sprint 45 Day 1 Artifact: Scope and Iterative Baseline

## Purpose

Capture the Sprint 45 starting baseline before iterative workspace reuse,
compatibility-wrapper normalization, and repeated-solve benchmark work begin.

## Starting Truth

Sprint 45 starts from a stable preserved Sprint 40/41/42/44 baseline:

- strongest local reviewed baseline already exists:
  - `make quality-review-full`
- reviewed CMake parity remains explicit and measurable:
  - `ctest -N --test-dir build/quality-review-cmake` = `53`
- maintained dead-code surfaces already exist:
  - `make deadcode-report`
  - `make deadcode-check`
- dead-code execution remains serialized because `deadcode*` still shares:
  - `build/deadcode-cmake`
  - `build/deadcode/`
- Sprint 41 already left a reusable internal safety/helper layer:
  - `src/sparse_alloc_internal.h`
  - `src/sparse_alloc_internal.c`
- Sprint 42 already left a compatibility-preserving internal-first refactor
  model:
  - internal lifecycle scaffolding
  - shared state-guard helpers
  - wrapper-preserving migration rules
- Sprint 44 already closed from a validated Epic 4 baseline, so Sprint 45 does
  not need to spend time re-establishing graph/lifecycle/quality contracts

This means Sprint 45 is not opening with baseline repair or public API churn.
It is opening with bounded iterative workspace reuse and repeated-solve
efficiency work on top of a preserved reviewed baseline and an already-written
Epic 4 execution contract.

## Day 1 Workstreams

Sprint 45 Day 1 confirms the sprint's eight bounded workstreams:

1. iterative workspace seam inventory
2. reusable workspace API design
3. shared workspace-backed internal helper layer
4. CG / GMRES migration
5. block iterative migration
6. compatibility wrapper preservation
7. repeated-solve benchmark batch
8. validation closeout

These come directly from the Sprint 45 section of
`docs/planning/EPIC_4/PROJECT_PLAN.md` and stay consistent with the earlier
Epic 4 rule that structural performance/maintainability changes should land
internally first before any broader public-surface redesign.

## Highest-Value Authoritative Inputs

### Epic 4 planning and architecture inputs

- `docs/planning/EPIC_4/PROJECT_PLAN.md`
- `docs/planning/EPIC_4/SPRINT_45/PLAN.md`
- `docs/planning/EPIC_4/SPRINT_44/artifacts/day14-closeout-and-handoff.md`

### Inherited execution-rule inputs

- `docs/planning/EPIC_4/SPRINT_41/artifacts/day12-safety-style-and-prep-rules.md`
- `docs/planning/EPIC_4/SPRINT_42/artifacts/day14-closeout-and-handoff.md`
- `docs/planning/EPIC_4/SPRINT_40/artifacts/day13-validation-anchor-and-command-matrix.md`
- `src/sparse_alloc_internal.h`
- `src/sparse_alloc_internal.c`
- `src/sparse_bicgstab_internal.h`

### Inherited reviewed-quality / policy inputs

- `README.md`
- `Makefile`
- `CMakeLists.txt`
- `.github/workflows/ci.yml`
- `.github/workflows/macos-ci.yml`
- `.github/workflows/windows-ci.yml`

### Highest-risk Day 1 iterative implementation and regression inputs

- `src/sparse_iterative.c`
- `tests/test_iterative.c`
- `tests/test_block_solvers.c`
- `tests/test_bicgstab.c`
- `tests/test_minres.c`
- `tests/test_stagnation.c`

### Highest-risk Day 1 repeated-solve support inputs

- `examples/example_iterative.c`
- `examples/example_matrix_free.c`
- `benchmarks/bench_convergence.c`
- `benchmarks/bench_refactor.c`

## Highest-Value Day 1 Conclusions

### 1. Sprint 45 is an internal iterative-workspace sprint, not a public API sprint

The preserve-not-reopen boundary is explicit:

- keep work internal-first
- preserve current one-shot public iterative entry points
- preserve Sprint 40 validation-anchor truth
- preserve Sprint 41 shared-helper rules where generic count/allocation work is
  needed
- preserve Sprint 42 compatibility-preserving refactor style
- avoid bundling eigensolver workspace work that belongs to Sprint 46

### 2. The main iterative hotspot and support surfaces are already explicit

The live repo now shows:

- `src/sparse_iterative.c` = `2357` lines
- iterative regression concentration:
  - `tests/test_iterative.c` = `2795`
  - `tests/test_block_solvers.c` = `507`
  - `tests/test_bicgstab.c` = `1586`
  - `tests/test_minres.c` = `1588`
  - `tests/test_stagnation.c` = `1361`
- repeated-solve teaching / benchmark support:
  - `examples/example_iterative.c` = `144`
  - `examples/example_matrix_free.c` = `122`
  - `benchmarks/bench_convergence.c` = `421`
  - `benchmarks/bench_refactor.c` = `159`

That means Sprint 45 does not need another exploratory sprint before it begins
workspace work or repeated-solve measurement.

### 3. The iterative subsystem already contains one useful workspace precedent

The live code shows:

- CG, matrix-free CG, GMRES, block CG, and MINRES still rely on one-shot
  packed buffer allocation inside `src/sparse_iterative.c`
- BiCGSTAB already uses:
  - `bicgstab_workspace_t`
  - `bicgstab_workspace_alloc(...)`
  - `bicgstab_workspace_free(...)`

That means Sprint 45 is not inventing the first iterative internal workspace
concept. It is generalizing the reusable-workspace direction to the larger
repeated-allocation seams.

### 4. The front-half order of the sprint is fixed

The correct early sprint order is:

1. baseline and seam inventory
2. workspace API design
3. shared buffer-layer design
4. shared buffer landing
5. CG / GMRES migration
6. block-path migration

That ordering preserves Sprint 40's core rule: structural refactors should be
guided by measured seams and explicit ownership boundaries before code movement
lands.
