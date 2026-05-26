# Sprint 42 Day 1 Artifact: Scope and Lifecycle Baseline

## Purpose

Capture the Sprint 42 starting baseline before lifecycle-seam inventory,
internal handle scaffolding, and shared matrix-state guard work begins.

## Starting Truth

Sprint 42 starts from a stable preserved Sprint 40/Sprint 41 baseline:

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
- Sprint 40 already left a concrete lifecycle and migration contract:
  - lifecycle/state taxonomy
  - lifecycle contract map
  - handle-model migration strategy
  - quality-truth ownership map
  - public migration-risk audit
  - validation anchor
- Sprint 41 already left a concrete internal safety/helper package:
  - `src/sparse_alloc_internal.h`
  - `src/sparse_alloc_internal.c`
  - broader `src/` migration pattern
  - bounded example-side helper alignment

This means Sprint 42 is not opening with reviewed-quality repair work or
generic helper cleanup. It is opening with lifecycle-boundary implementation on
top of a stable reviewed baseline, explicit architecture contract, and a proven
shared internal utility seam.

## Day 1 Workstreams

Sprint 42 Day 1 confirms the sprint's eight bounded workstreams:

1. lifecycle seam inventory refresh
2. internal handle scaffolding
3. matrix-state guard helpers
4. factor-path normalization
5. cancellation-contract normalization
6. compatibility bridge planning
7. focused lifecycle tests
8. validation closeout

These come directly from the Sprint 42 section of
`docs/planning/EPIC_4/PROJECT_PLAN.md` and from Sprint 40's Day 14 handoff,
which identified LU/Cholesky internal payload separation and
`sparse_factors_t` bridge normalization as the first major lifecycle-handle
landing seams.

## Highest-Value Authoritative Inputs

### Epic 4 planning and architecture inputs

- `docs/planning/EPIC_4/PROJECT_PLAN.md`
- `docs/planning/EPIC_4/SPRINT_42/PLAN.md`
- `docs/planning/EPIC_4/SPRINT_40/artifacts/day14-architecture-contract-synthesis.md`
- `docs/planning/EPIC_4/SPRINT_40/artifacts/day10-handle-model-design-2-and-migration-strategy.md`
- `docs/planning/EPIC_4/SPRINT_40/artifacts/day12-public-migration-risk-audit.md`

### Inherited implementation and handoff inputs

- `docs/planning/EPIC_4/SPRINT_41/artifacts/day14-closeout-and-handoff.md`
- `docs/planning/EPIC_4/SPRINT_41/artifacts/day12-safety-style-and-prep-rules.md`
- `src/sparse_alloc_internal.h`
- `src/sparse_alloc_internal.c`

### Inherited reviewed-quality / policy inputs

- `README.md`
- `Makefile`
- `CMakeLists.txt`
- `.github/workflows/ci.yml`
- `.github/workflows/macos-ci.yml`
- `.github/workflows/windows-ci.yml`

### Highest-risk Day 1 lifecycle seam inputs

- LU factorization internals
- Cholesky factorization internals
- `sparse_factors_t`
- lifecycle-sensitive precondition checks across:
  - QR
  - SVD
  - analysis
  - direct factorization paths

## Highest-Value Day 1 Conclusions

### 1. Sprint 42 is an internal lifecycle-groundwork sprint, not a public-handle rollout sprint

The preserve-not-reopen boundary is now explicit:

- keep work internal-first
- preserve current public API compatibility
- preserve Sprint 40 validation and migration-risk contracts
- preserve Sprint 41's shared-helper seam as the default internal safety layer
- avoid opportunistic README/tutorial/header churn before new lifecycle seams
  are real

### 2. The first landing seams are explicit before code changes begin

The first Sprint 42 lifecycle landing cluster is:

- LU factorization internals
- Cholesky factorization internals
- `sparse_factors_t`
- shared original-state / identity-permutation / factored-state guard paths

This gives the sprint a bounded first-wave target set before broader later Epic
4 lifecycle or public-handle work starts.

### 3. The front-half order of the sprint is fixed

The correct early sprint order is:

1. baseline and scope confirmation
2. lifecycle seam inventory refresh
3. internal handle scaffolding design
4. matrix-state guard design
5. first bounded implementation batch

That ordering preserves Sprint 40's core rule: architecture and compatibility
truth should guide lifecycle implementation rather than being reverse-engineered
after the code changes land.
