# Sprint 49 Day 2 Artifact: Public Lifecycle/API Surface Inventory

## Purpose

Refresh the live public lifecycle/workspace seam map so Sprint 49's API
landing, migration-path documentation, compatibility sweep, residual review,
and final validation are sequenced from the actual repo state rather than only
from the Sprint 49 project-plan headings.

## Day 2 Starting Point

Day 1 already fixed the macro baseline:

- Sprint 49 starts from a preserved Sprint 40/42/45/46/48 validation and
  lifecycle/documentation baseline
- one reusable public lifecycle precedent already exists in
  `include/sparse_analysis.h`
- iterative and eigensolver repeated-run gains are still internal-only

Day 2 therefore focuses on reducing the remaining work to explicit public and
compatibility ownership seams.

## Refreshed Surface Map

### 1. Reusable public lifecycle precedent already exists

The current public reusable-handle model lives in `include/sparse_analysis.h`:

- `sparse_analysis_t`
- `sparse_factors_t`
- `sparse_analyze(...)`
- `sparse_factor_numeric(...)`
- `sparse_refactor_numeric(...)`
- `sparse_factor_free(...)`

This is the existing public precedent for:

- analyze once
- reuse across repeated numeric work
- free explicit owned state

Sprint 49 should treat this as the public terminology and lifecycle anchor, not
as the main missing implementation seam.

### 2. Iterative and eigensolver public usage is still primarily one-shot

The current public headers still primarily teach one-shot usage:

- `include/sparse_iterative.h`
  - `sparse_solve_cg(...)`
  - `sparse_solve_gmres(...)`
  - matrix-free and block convenience variants
- `include/sparse_eigs.h`
  - `sparse_eigs_sym(...)`

The top-level README and the main examples reinforce that one-shot caller
shape:

- `README.md`
- `examples/example_iterative.c`
- `examples/example_eigs.c`

This is the main Sprint 49 public-surface gap.

### 3. The repeated-run implementation gains already exist behind private seams

The repeated-run benchmark and internal-entry surfaces show that the real
reusable-workspace/state paths are already landed:

#### Iterative internal reuse

- `src/sparse_iterative_workspace_internal.h`
- `src/sparse_iterative_workspace_internal.c`
- `src/sparse_iterative_internal.h`
- `sparse_solve_cg_with_workspace_internal(...)`
- `sparse_solve_gmres_with_workspace_internal(...)`
- `benchmarks/bench_iterative_reuse.c`

#### Eigensolver internal reuse

- `src/sparse_eigs_workspace_internal.h`
- `src/sparse_eigs_workspace_internal.c`
- `src/sparse_eigs_internal.h`
- `sparse_eigs_sym_with_workspace_internal(...)`
- `benchmarks/bench_eigs_reuse.c`

This means Sprint 49 is not blocked on new internal groundwork. The main
remaining task is bounded public exposure and compatibility integration.

## Day 2 Seam Buckets

### Bucket 1: Explicit lifecycle/workspace public exposure

This is the main first-landing seam:

- choose the bounded public handle/type/function surface
- expose only what is now safe after the groundwork sprints
- align its terminology with the existing analysis/factor lifecycle precedent

Primary target surfaces:

- `include/sparse_iterative.h`
- `include/sparse_eigs.h`
- corresponding implementation routing in:
  - `src/sparse_iterative.c`
  - `src/sparse_eigs.c`

### Bucket 2: Compatibility-preserving one-shot wrapper routing

This is the second main seam:

- old one-shot public callers must keep working
- wrapper semantics should become more explicit and easier to explain
- internal reusable-workspace/state paths should remain the backing seam

Primary target surfaces:

- `src/sparse_iterative.c`
- `src/sparse_eigs.c`

### Bucket 3: Caller guidance and migration rules

This seam is now clearly separate from API design itself:

- when old one-shot style remains appropriate
- when explicit lifecycle/workspace style is preferable
- how the existing public analysis/factor lifecycle relates to the new surface

Primary target surfaces:

- top-level docs
- touched examples
- any migration-focused Sprint 49 artifact/docs text

### Bucket 4: Cross-surface compatibility alignment

This is the later verification seam:

- README
- examples
- repeated-run benchmarks
- tests/comments reflecting the final contract

These are important, but they should follow the bounded public API landing
instead of driving it.

### Bucket 5: Final residual-review and Epic 4 bookkeeping

This seam is not code-first:

- revisit `review-codex-2026-05-21.md`
- classify remaining findings
- map fixed vs deferred vs accepted tradeoff outcomes
- fold the final public lifecycle/workspace contract into Epic 4 closeout

## First-Landing Targets vs Later Verification Surfaces

### Real first landing targets

- `include/sparse_iterative.h`
- `include/sparse_eigs.h`
- `src/sparse_iterative.c`
- `src/sparse_eigs.c`

These are the direct Sprint 49 API exposure and compatibility-routing surfaces.

### Public precedent / terminology anchor

- `include/sparse_analysis.h`

This should guide the final public lifecycle wording, but it is not the main
missing landing target.

### Later verification and migration-proof surfaces

- `README.md`
- `examples/example_iterative.c`
- `examples/example_matrix_free.c`
- `examples/example_eigs.c`
- `benchmarks/bench_iterative_reuse.c`
- `benchmarks/bench_eigs_reuse.c`
- iterative/eigensolver regression tests

These should be reconciled after the bounded header/source shape is stable.

## Highest-Value Day 2 Conclusions

### 1. Sprint 49 is reconciling two already-existing public styles, not inventing a new API category

The repo already has:

- explicit reusable public lifecycle in `sparse_analysis.h`
- compatibility-oriented one-shot solver/eigensolver APIs

The real Sprint 49 job is to make these coexist coherently at the public edge.

### 2. The iterative and eigensolver headers are the true first API landing zone

The remaining missing public work is concentrated in:

- `include/sparse_iterative.h`
- `include/sparse_eigs.h`

Everything else is either precedent, implementation backing seam, or later
verification surface.

### 3. Migration and compatibility work should follow, not lead, the public landing

Examples, README, and repeated-run benchmarks already prove caller expectations
and internal reuse value, but they should not define the public API first.

The correct order is:

1. API design
2. bounded public header / source landing
3. migration docs
4. compatibility sweep
5. residual review
6. validation / closeout
