# Sprint 49 Day 1 Artifact: Scope and Lifecycle API Baseline

## Purpose

Capture the Sprint 49 starting baseline before public lifecycle API landing,
migration-path documentation, cross-surface compatibility reconciliation,
final residual review, full validation, and Epic 4 closeout begin.

## Starting Truth

Sprint 49 starts from a stable preserved Sprint 40/42/45/46/48 baseline:

- strongest local reviewed baseline already exists:
  - `make quality-review-full`
- reviewed CMake parity remains explicit and measurable:
  - `ctest -N --test-dir build/quality-review-cmake` = `53`
- Sprint 42 already left the compatibility-preserving lifecycle groundwork:
  - factor-state scaffolding
  - shared state-guard helpers
  - internal-first wrapper-preserving migration rules
- Sprint 45 already left a reusable internal iterative workspace precedent:
  - `src/sparse_iterative_workspace_internal.h`
  - `src/sparse_iterative_workspace_internal.c`
- Sprint 46 already left a reusable internal eigensolver workspace/state
  precedent:
  - `src/sparse_eigs_workspace_internal.h`
  - `src/sparse_eigs_workspace_internal.c`
- Sprint 48 already left the final maintainer-facing policy / migration-doc
  home structure:
  - `docs/maintainer_guide.md`

This means Sprint 49 is not opening with baseline recovery, decomposition
repair, or generic docs redistribution. It is opening with the final bounded
public-lifecycle exposure and Epic 4 integration/closeout work on top of an
already-validated structural baseline.

## Day 1 Workstreams

Sprint 49 Day 1 confirms the sprint's seven bounded workstreams:

1. public lifecycle API landing
2. migration-path documentation
3. cross-surface compatibility sweep
4. final residual review
5. full integration validation
6. Epic 4 summary artifacts
7. closeout and handoff

These come directly from the Sprint 49 section of
`docs/planning/EPIC_4/PROJECT_PLAN.md` and stay consistent with the earlier
Epic 4 rule that public-facing cleanup should only happen after the internal
ownership groundwork and validation anchors are already stable.

## Highest-Value Authoritative Inputs

### Epic 4 planning and architecture inputs

- `docs/planning/EPIC_4/PROJECT_PLAN.md`
- `docs/planning/EPIC_4/SPRINT_49/PLAN.md`
- `docs/planning/EPIC_4/SPRINT_48/artifacts/day14-closeout-and-handoff.md`

### Inherited execution-rule and lifecycle-groundwork inputs

- `docs/planning/EPIC_4/SPRINT_40/artifacts/day13-validation-anchor-and-command-matrix.md`
- `docs/planning/EPIC_4/SPRINT_42/artifacts/day14-closeout-and-handoff.md`
- `docs/planning/EPIC_4/SPRINT_45/artifacts/day14-closeout-and-handoff.md`
- `docs/planning/EPIC_4/SPRINT_46/artifacts/day14-closeout-and-handoff.md`
- `src/sparse_matrix_internal.h`
- `src/sparse_iterative_internal.h`
- `src/sparse_iterative_workspace_internal.h`
- `src/sparse_iterative_workspace_internal.c`
- `src/sparse_eigs_internal.h`
- `src/sparse_eigs_workspace_internal.h`
- `src/sparse_eigs_workspace_internal.c`

### Inherited reviewed-quality / policy inputs

- `README.md`
- `docs/maintainer_guide.md`
- `Makefile`
- `CMakeLists.txt`
- `.github/workflows/ci.yml`
- `.github/workflows/macos-ci.yml`
- `.github/workflows/windows-ci.yml`

### Highest-risk Day 1 public lifecycle/workspace inputs

- `include/sparse_analysis.h`
- `include/sparse_iterative.h`
- `include/sparse_eigs.h`
- `src/sparse_iterative.c`
- `src/sparse_eigs.c`

### Highest-risk Day 1 migration / compatibility support inputs

- `examples/example_iterative.c`
- `examples/example_matrix_free.c`
- `examples/example_eigs.c`
- `benchmarks/bench_iterative_reuse.c`
- `benchmarks/bench_eigs_reuse.c`
- `tests/test_iterative.c`
- `tests/test_block_solvers.c`
- `tests/test_minres.c`
- `tests/test_bicgstab.c`
- `tests/test_stagnation.c`
- `tests/test_eigs.c`
- `tests/test_eigs_thick_restart.c`
- `tests/test_eigs_lobpcg.c`

## Highest-Value Day 1 Conclusions

### 1. Sprint 49 is a bounded public-lifecycle exposure sprint, not an internal-workspace invention sprint

The preserve-not-reopen boundary is explicit:

- preserve Sprint 40 validation-anchor truth
- preserve the existing compatibility-oriented one-shot public APIs where
  required
- reuse the Sprint 42 lifecycle scaffolding rather than creating a second
  competing lifecycle model
- reuse the Sprint 45/46 internal workspace precedents rather than inventing
  a new repeated-run ownership model
- avoid reopening the large subsystem decompositions already closed in Sprints
  43-48

### 2. The repo already has one public reusable-lifecycle precedent

The main public precedent already exists in `include/sparse_analysis.h`:

- `sparse_analysis_t`
- `sparse_factors_t`
- analyze / numeric factor / refactor / free lifecycle

That means Sprint 49 is not inventing explicit lifecycle concepts from
nothing. It is aligning the final public-facing iterative/eigensolver story
with a reusable-handle pattern the library already teaches in one subsystem.

### 3. The newer repeated-run gains are still internal-only today

The main newer repeated-run improvements remain behind private seams:

- iterative reusable workspace:
  - `src/sparse_iterative_workspace_internal.h`
  - `src/sparse_iterative_workspace_internal.c`
- eigensolver reusable workspace/state:
  - `src/sparse_eigs_workspace_internal.h`
  - `src/sparse_eigs_workspace_internal.c`

The public iterative and eigensolver headers remain primarily one-shot entry
surfaces. That is the exact final Epic 4 gap Sprint 49 is meant to close.

### 4. The direct Sprint 49 hotspots are already explicit

The live Day 1 sizes make the primary landing surface obvious:

- `include/sparse_iterative.h` = `585`
- `include/sparse_eigs.h` = `592`
- `include/sparse_analysis.h` = `334`
- `src/sparse_iterative.c` = `2276`
- `src/sparse_eigs.c` = `3060`

The support and regression surfaces are already concentrated:

- examples:
  - `example_iterative.c` = `144`
  - `example_matrix_free.c` = `122`
  - `example_eigs.c` = `285`
- repeated-run benchmarks:
  - `bench_iterative_reuse.c` = `251`
  - `bench_eigs_reuse.c` = `201`
- core regression concentration:
  - `tests/test_iterative.c` = `2795`
  - `tests/test_eigs.c` = `1269`
  - `tests/test_eigs_thick_restart.c` = `1161`
  - `tests/test_eigs_lobpcg.c` = `1196`

### 5. Migration-path documentation is a real Sprint 49 deliverable, not a postscript

The Day 1 public/internal split is now explicit:

- existing one-shot public callers still have a supported path
- internal reusable-workspace-backed paths already prove the repeated-run value
- the repo now has both:
  - a reusable public lifecycle precedent (`sparse_analysis.h`)
  - internal repeated-run precedents (iterative and eigensolver)

That means Sprint 49 can and should document a concrete migration path instead
of a generic “handles are better” story.

### 6. The sprint must reserve real space for final residual review and epic closeout

Sprint 49 is not only an API landing sprint. It also owns:

- cross-surface compatibility agreement
- final classification of `review-codex-2026-05-21.md`
- full integrated validation
- Epic 4 final summary artifacts
- final residual-risk routing

The correct Sprint 49 shape is therefore:

1. baseline and seam inventory
2. bounded public lifecycle design
3. API landing
4. compatibility and migration proof
5. residual review
6. final validation
7. Epic 4 closeout
