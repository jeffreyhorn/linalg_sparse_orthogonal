# Sprint 46 Day 1 Artifact: Scope and Eigensolver Baseline

## Purpose

Capture the Sprint 46 starting baseline before eigensolver workspace reuse,
compatibility-wrapper preservation, repeated-run benchmark work, and
maintainer-facing memory-behavior documentation begin.

## Starting Truth

Sprint 46 starts from a stable preserved Sprint 40/41/42/45 baseline:

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
- Sprint 45 already left a reusable internal workspace precedent:
  - `src/sparse_iterative_workspace_internal.h`
  - `src/sparse_iterative_workspace_internal.c`
  - compatibility-preserving one-shot wrapper routing

This means Sprint 46 is not opening with baseline repair or public API churn.
It is opening with bounded eigensolver workspace/state reuse and repeated-run
efficiency work on top of a preserved reviewed baseline and an already-proven
internal reusable-workspace pattern.

## Day 1 Workstreams

Sprint 46 Day 1 confirms the sprint's eight bounded workstreams:

1. eigensolver seam inventory
2. reusable workspace/state design
3. shared buffer layer
4. grow-m / thick-restart migration
5. LOBPCG migration
6. wrapper preservation
7. repeated-run benchmark batch
8. memory-behavior documentation and validation closeout

These come directly from the Sprint 46 section of
`docs/planning/EPIC_4/PROJECT_PLAN.md` and stay consistent with the earlier
Epic 4 rule that structural performance/maintainability changes should land
internally first before any broader public-surface redesign.

## Highest-Value Authoritative Inputs

### Epic 4 planning and architecture inputs

- `docs/planning/EPIC_4/PROJECT_PLAN.md`
- `docs/planning/EPIC_4/SPRINT_46/PLAN.md`
- `docs/planning/EPIC_4/SPRINT_45/artifacts/day14-closeout-and-handoff.md`

### Inherited execution-rule inputs

- `docs/planning/EPIC_4/SPRINT_41/artifacts/day12-safety-style-and-prep-rules.md`
- `docs/planning/EPIC_4/SPRINT_42/artifacts/day14-closeout-and-handoff.md`
- `docs/planning/EPIC_4/SPRINT_40/artifacts/day13-validation-anchor-and-command-matrix.md`
- `src/sparse_alloc_internal.h`
- `src/sparse_alloc_internal.c`
- `src/sparse_iterative_workspace_internal.h`
- `src/sparse_iterative_workspace_internal.c`

### Inherited reviewed-quality / policy inputs

- `README.md`
- `Makefile`
- `CMakeLists.txt`
- `.github/workflows/ci.yml`
- `.github/workflows/macos-ci.yml`
- `.github/workflows/windows-ci.yml`

### Highest-risk Day 1 eigensolver implementation and regression inputs

- `src/sparse_eigs.c`
- `tests/test_eigs.c`
- `tests/test_eigs_thick_restart.c`
- `tests/test_eigs_lobpcg.c`

### Highest-risk Day 1 repeated-run support inputs

- `examples/example_eigs.c`
- `benchmarks/bench_eigs.c`
- `benchmarks/bench_iterative_reuse.c`

## Highest-Value Day 1 Conclusions

### 1. Sprint 46 is an internal eigensolver-workspace sprint, not a public API sprint

The preserve-not-reopen boundary is explicit:

- keep work internal-first
- preserve current one-shot public eigensolver entry points
- preserve Sprint 40 validation-anchor truth
- reuse Sprint 41 helper patterns and Sprint 45 workspace patterns instead of
  creating a second competing internal model
- keep the work bounded to repeated-run eigensolver efficiency rather than
  broad benchmark or documentation churn

### 2. The main eigensolver hotspot and support surfaces are already explicit

The live repo now shows:

- `src/sparse_eigs.c` = `3151` lines
- eigensolver regression concentration:
  - `tests/test_eigs.c` = `1269`
  - `tests/test_eigs_thick_restart.c` = `1161`
  - `tests/test_eigs_lobpcg.c` = `1196`
- repeated-run teaching / benchmark support:
  - `examples/example_eigs.c` = `284`
  - `benchmarks/bench_eigs.c` = `958`
- direct reusable-workspace precedent:
  - `src/sparse_iterative_workspace_internal.h` = `76`
  - `src/sparse_iterative_workspace_internal.c` = `215`

That means Sprint 46 does not need another exploratory sprint before it begins
workspace/state work or repeated-run measurement.

### 3. The real repeated-run eigensolver targets are now fixed up front

The live code shows the primary repeated-allocation / repeated-run families are:

- grow-m Lanczos
- thick-restart Lanczos
- LOBPCG

The strongest common allocation shapes are:

- basis / vector bundles
- tridiagonal / Ritz / restart scratch
- block `(n * k)` bundles
- dense projected-subproblem intermediates

That means Sprint 46 should be driven by real buffer/state ownership shapes
rather than by a generic "add context objects everywhere" approach.

### 4. Sprint 46 already has one partial eigensolver-state precedent and one stronger neighboring workspace precedent

The live code already shows:

- thick-restart-specific restart-state ownership via:
  - `lanczos_restart_state_t`
  - `lanczos_restart_state_free(...)`
- a broader proven reusable-workspace pattern next door from Sprint 45

That means Sprint 46 is not inventing all state/lifetime rules from scratch.
It is expanding them into a coherent shared reusable eigensolver workspace/state
model for the main repeated-run paths.

### 5. The front-half order of the sprint is fixed

The correct early sprint order is:

1. baseline and seam inventory
2. reusable workspace/state design
3. shared buffer-layer design
4. shared buffer landing
5. grow-m / thick-restart migration
6. LOBPCG migration

That ordering preserves Sprint 40's core rule: structural refactors should be
guided by measured seams and explicit ownership boundaries before code movement
lands.
