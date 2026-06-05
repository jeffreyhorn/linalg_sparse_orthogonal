# Sprint 55 Day 8 - `sparse_iterative.c` seam audit

Date: 2026-06-04
Branch: `sprint-55`

## Scope

Reduce the post-Day-7 `src/sparse_iterative.c` ownership problem to explicit
extraction seams, rank the viable iterative split targets by maintainability
value and behavioral risk, and define the first bounded iterative extraction
boundary without reopening the public repeated-run solver contract.

## Live ownership map

The current iterative implementation now reads as six real ownership bands
rather than one undifferentiated large-file problem:

1. public handle orchestration and growth helpers
2. shared staging / residual-history / reporting utilities
3. `CG` execution path
4. `GMRES` execution path
5. `MINRES` execution path
6. block-wrapper plus `BiCGSTAB` compatibility surfaces

The strongest retained large-file hotspot is still:

- `src/sparse_iterative.c` = `2377`

But the decomposition question is now shaped more by ownership coherence than
by raw line count alone.

## Ranked extraction targets

### 1. First target: `MINRES`

Why it ranks first:

- the internal reusable-workspace seam already exists:
  - `sparse_iter_workspace_prepare_minres(...)`
- the public repeated-run handle path is already real:
  - `sparse_iter_handle_prepare_minres(...)`
  - `sparse_solve_minres_with_handle(...)`
- direct public repeated-run proof already exists:
  - `tests/test_iterative.c`
- family-local numerical proof already exists:
  - `tests/test_minres.c`

Maintainability argument:

- the `MINRES` cluster is already family-coherent
- extracting it would reduce the main file without changing the public support
  boundary or the one-shot/default caller story
- this is the lowest-risk first split that still buys real ownership clarity

### 2. Later target: `GMRES`

Why it ranks second:

- it is a meaningful ownership band
- it would eventually remove a substantial amount of code from the main file

Why it is not first:

- it is more entangled with matrix-free adapters, restart-state flow, public
  reuse orchestration, and block-column wrappers
- a first split here would carry more orchestration risk than `MINRES`

### 3. Later target: block-wrapper scaffolding

Why it matters:

- the block surfaces are real code and do contribute to file size

Why it is not first:

- the dominant block shape is wrapper-oriented rather than a numerically
  cohesive backend family
- moving it first would do less to reduce the main reasoning burden

### 4. Defer target: `BiCGSTAB`

Why it is not a Phase 1 extraction target:

- it still uses its own family-local workspace model:
  - `sparse_bicgstab_internal.h`
  - `bicgstab_workspace_t`
- Sprint 54 explicitly left `BiCGSTAB` outside the public repeated-run handle
  support set

Maintainability conclusion:

- `BiCGSTAB` is a real seam, but not the right first seam
- extracting it early would mix decomposition work with a consciously excluded
  public-handle family

## Proposed first extraction boundary

Recommended new file:

- `src/sparse_iterative_minres.c`

Move target set:

- `sparse_solve_minres_with_workspace_internal(...)`
- `sparse_solve_minres(...)`
- `sparse_solve_minres_with_handle(...)`
- `solve_block_minres_column(...)`
- `sparse_minres_solve_block(...)`

Retain in `src/sparse_iterative.c`:

- public handle init/free and growth helpers
- shared staging/residual/reporting helpers
- `CG`
- `GMRES`
- block-shared wrapper scaffolding
- `BiCGSTAB`

Why this boundary is the right first move:

- it extracts one coherent repeated-run solver family
- it preserves the Sprint 54 steady-state support boundary
- it keeps the shared public/orchestration layer intact
- it does not require public API changes or a private-header taxonomy rewrite

## Rejected split strategies

The audit explicitly rejects:

- splitting by arbitrary line ranges
- starting with tiny utility-only moves that leave the main reasoning load in
  place
- reopening the Sprint 54 repeated-run support boundary
- treating `BiCGSTAB` as equivalent to the supported handle-backed families
- combining the first iterative extraction with a broad comment-taxonomy or
  architecture rewrite

## Proof/adoption guidance for the first iterative batch

Primary proof surfaces for a `MINRES` extraction:

- `tests/test_minres.c`
- `tests/test_iterative.c`

Secondary parity surfaces:

- `benchmarks/bench_iterative_reuse.c`
- `examples/example_iterative.c`
- `build/example_ic_minres`

The benchmark/example surfaces should stay parity checks, not redesign targets.

## Conclusion

Sprint 55 Day 8 reduces the iterative large-source problem to an explicit
landing order:

1. `MINRES` extraction first
2. `GMRES` later if the remaining orchestration layer still needs a second
   family split
3. block-wrapper cleanup only after the main family ownership improves
4. `BiCGSTAB` deferred out of the first extraction fence

That gives Day 9 a maintainability-shaped implementation target rather than a
generic file-size reduction objective.
