# Sprint 55 Day 9 - iterative decomposition batch 1 design

Date: 2026-06-04
Branch: `sprint-55`

## Scope

Freeze the first `src/sparse_iterative.c` extraction boundary before any
iterative code movement begins by defining:

- the exact Day 10 file split
- the private-header strategy
- the behavior invariants the batch must preserve
- the minimal comment-cleanup policy for the moved `MINRES` ownership band

## First-batch target

The first iterative extraction target remains:

- `MINRES`

But the Day 10 landing should be narrower than the original Day 8 sketch.

## Exact file-boundary split

Recommended new file:

- `src/sparse_iterative_minres.c`

Move into the new file:

- `sparse_solve_minres_with_workspace_internal(...)`
- `sparse_solve_minres(...)`
- `sparse_solve_minres_with_handle(...)`

Retain in `src/sparse_iterative.c`:

- public handle init/free and growth helpers
- shared staging / residual-history / reporting helpers
- `CG`
- `GMRES`
- shared block-column orchestration:
  - `iter_block_column_solver_fn`
  - `solve_block_independent_columns(...)`
- block wrapper entry points:
  - `solve_block_minres_column(...)`
  - `sparse_minres_solve_block(...)`
- `BiCGSTAB`

## Why the block MINRES wrapper stays in the main file for Batch 1

The block MINRES entrypoint is not numerically complex on its own; its main
coupling is to the generic shared block-column helper:

- `solve_block_independent_columns(...)`

Moving the block MINRES wrapper in the first batch would widen the extraction
surface unnecessarily by forcing that shared wrapper seam into cross-file
visibility too early.

Phase-1 maintainability conclusion:

- move the coherent scalar/handle `MINRES` family first
- keep shared block-wrapper scaffolding in the retained main iterative file

## Private declaration strategy

Use the existing iterative internal headers:

- `src/sparse_iterative_internal.h`
- `src/sparse_iterative_workspace_internal.h`

Expected Day 10 widening:

- add `sparse_solve_minres_with_workspace_internal(...)` to
  `src/sparse_iterative_internal.h`

Do not add:

- `src/sparse_iterative_minres_internal.h`

Why:

- the current internal header already owns the workspace-backed internal solver
  entry surface for `CG` and `GMRES`
- adding `MINRES` there matches the established pattern
- the first batch stays ownership-focused instead of taxonomy-focused

## Invariants the extraction must preserve

Day 10 must preserve:

- public repeated-run handle semantics for:
  - `CG`
  - `GMRES`
  - `MINRES`
- one-shot/default caller semantics for `sparse_solve_minres(...)`
- handle growth/reuse behavior for `sparse_solve_minres_with_handle(...)`
- workspace capacity and typing through:
  - `sparse_iter_workspace_prepare_minres(...)`
- result/reporting fields:
  - `iterations`
  - `residual_norm`
  - `converged`
  - `stagnated`
  - `breakdown`
- parity across the main proof/adoption surfaces:
  - `tests/test_minres.c`
  - `tests/test_iterative.c`
  - `benchmarks/bench_iterative_reuse.c`
  - `build/example_ic_minres`

This is an ownership change, not a behavior change.

## Comment policy for the first iterative batch

The moved `MINRES` block still contains stale sprint-history narrative such as:

- `Sprint 29 Day 7`

Day 10 should:

- keep durable algorithm commentary
- keep comments that explain numerical checks, recurrence state, or
  convergence-verification logic
- remove or rewrite stale sprint-history narration only inside the moved
  `MINRES` ownership band

Day 10 should not:

- normalize the entire remaining `src/sparse_iterative.c` comment body
- turn the extraction into a whole-file comment rewrite

## Expected Day 10 touched files

- `src/sparse_iterative.c`
- `src/sparse_iterative_minres.c` (new)
- `src/sparse_iterative_internal.h`
- `Makefile`
- `CMakeLists.txt`

## Validation checklist

Required code-day validation:

- `make format`
- `make lint`
- `make test`
- `make quality-review-full`

Primary proof surfaces:

- `tests/test_minres.c`
- `tests/test_iterative.c`

Secondary parity surfaces:

- `benchmarks/bench_iterative_reuse.c`
- `examples/example_iterative.c`
- `build/example_ic_minres`

## Conclusion

Sprint 55 Day 9 turns the Day 8 `MINRES` recommendation into a concrete Batch 1
design:

- new owned file:
  - `src/sparse_iterative_minres.c`
- move only the scalar/handle `MINRES` family
- keep block-wrapper scaffolding in the retained main file
- widen the existing internal header instead of adding a new private one
- keep the first comment cleanup tightly bounded to the moved `MINRES` block

That gives Day 10 a precise landing checklist and avoids refining the
extraction boundary mid-implementation.
