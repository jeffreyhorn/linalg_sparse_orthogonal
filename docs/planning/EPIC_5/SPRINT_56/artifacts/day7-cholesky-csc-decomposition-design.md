# Sprint 56 Day 7 - Cholesky CSC decomposition design

Date: 2026-06-05
Branch: `sprint-56`

## Scope

Freeze the first bounded Cholesky CSC extraction boundary before editing
permanent implementation files, using the Day 6 supernodal-first ranking as
the design anchor.

## Selected first extraction seam

Sprint 56 Batch 2 should extract the full Cholesky-owned supernodal backend
into its own source file:

- new file:
  - `src/sparse_chol_csc_supernodal.c`

The first moved function set should be:

- `columns_in_same_supernode(...)`
- `chol_csc_detect_supernodes(...)`
- `chol_dense_factor(...)`
- `chol_dense_solve_lower(...)`
- `chol_csc_eliminate_supernodal(...)`
- `chol_csc_bsearch_row_map(...)`
- `chol_csc_supernode_extract(...)`
- `chol_csc_supernode_eliminate_diag(...)`
- `chol_csc_supernode_eliminate_panel(...)`
- `chol_csc_supernode_writeback(...)`

These functions already form a contiguous backend-owned block inside
`src/sparse_chol_csc.c`, share one SPD-only vocabulary, and have the strongest
dedicated proof surface in `tests/test_chol_csc.c`, with repeated-run CSC
behavioral confirmation in `benchmarks/bench_refactor_csc.c`.

## File-boundary ownership map

### Keep in `src/sparse_chol_csc.c`

- lifecycle / storage / structural conversion
- sparse-to-CSC and analysis-aware conversion entry points
- validation
- scalar workspace and native elimination/solve core
- wrapper / dispatch-specific glue
- `chol_csc_writeback_to_sparse(...)`
- shared dense indefinite primitive helpers:
  - `ldlt_dense_sym_swap(...)`
  - `ldlt_dense_factor(...)`

### Move to `src/sparse_chol_csc_supernodal.c`

- supernode detection
- dense Cholesky factor/solve primitives
- supernodal row-map lookup helper
- supernode extract logic
- supernode diagonal-block eliminate helper
- supernode panel eliminate helper
- supernode CSC writeback helper
- top-level supernodal elimination driver

## Internal declaration strategy

Sprint 56 Phase 2 should keep the moved declarations in the existing:

- `src/sparse_chol_csc_internal.h`

Reason:

- Batch 2 already changes one major ownership axis:
  - source-file extraction
- opening a new private-header taxonomy in the same batch would mix:
  - source extraction
  - private-header redesign
- the current internal header already owns the private contract for:
  - `CholCsc`
  - `CholCscWorkspace`
  - scalar/native helpers
  - Cholesky supernodal helper declarations
  - shared dense helper declarations

Deferred by design:

- creation of `src/sparse_chol_csc_supernodal_internal.h`
- broader repartitioning of `src/sparse_chol_csc_internal.h`

## Invariants the first batch must preserve

### Public contract invariants

- no public header/API changes
- no user-visible direct-solver lifecycle behavior change
- no change to the public analysis/factors path that ultimately reaches the
  Cholesky CSC completion seam

### Scalar/supernodal parity invariants

- scalar versus supernodal result parity unchanged
- supernode-detection semantics unchanged
- `min_size` threshold behavior unchanged
- dense Cholesky diagonal/panel behavior unchanged

### Dispatch/writeback invariants

- one-shot and shared analysis-aware CSC routing unchanged
- `chol_csc_factor(...)` and `chol_csc_factor_solve(...)` behavior unchanged
- `chol_csc_writeback_to_sparse(...)` semantics unchanged
- drop-threshold and diagonal-preservation behavior unchanged

### Proof-surface invariants

- `tests/test_chol_csc.c` remains the primary direct proof surface
- `tests/test_cholesky.c` remains unchanged in meaning
- `tests/test_integration.c` remains unchanged in meaning
- `benchmarks/bench_refactor_csc.c` keeps its current repeated-run CSC proof
- `examples/example_analysis.c` keeps its current caller-facing repeated direct
  workflow proof

## Bounded non-goal fence

This first Cholesky batch should not:

- redesign CSC dispatch
- change public APIs or header shape
- create a new private-header taxonomy
- move the shared dense LDLT primitive into the Cholesky-owned file
- widen into broader Cholesky/LDLT CSC code reconciliation
- reopen benchmark or example design beyond parity checks

## Minimal comment policy for the first batch

Preserve:

- durable algorithm meaning
- backend ownership boundaries
- threshold/drop/writeback invariants
- supernode detection and panel semantics

Reduce where touched:

- stale sprint chronology
- implementation-history narrative
- comments that explain landing order instead of present code truth

Do not try in Batch 2:

- repo-wide Cholesky CSC comment normalization
- CSC private-header taxonomy cleanup
- broader direct-solver doc rewriting

## Expected Day 8 touched files

Primary expected touched set:

- `src/sparse_chol_csc.c`
- `src/sparse_chol_csc_supernodal.c` (new)
- `src/sparse_chol_csc_internal.h`
- `Makefile`
- `CMakeLists.txt`

Secondary touch only if truly needed:

- `tests/test_chol_csc.c`

Avoid by default:

- `include/sparse_cholesky.h`
- `tests/test_cholesky.c`
- `tests/test_integration.c`
- `benchmarks/bench_refactor_csc.c`
- `src/sparse_ldlt_csc.c`

## Landing checklist

Before calling Batch 2 complete:

1. The Cholesky-owned supernodal backend lives in
   `src/sparse_chol_csc_supernodal.c`.
2. `src/sparse_chol_csc.c` still owns lifecycle/conversion, scalar/native
   elimination/solve, wrapper glue, CSC writeback-to-sparse, and shared dense
   LDLT helpers.
3. No public header/API changes are introduced.
4. Touched comments reflect durable code truth rather than sprint history.
5. Validation passes:
   - `make format`
   - `make lint`
   - `make test`
   - `make quality-review-full`
6. High-signal follow-ons remain green:
   - `./build/test_chol_csc`
   - `./build/test_cholesky`
   - `./build/test_integration`
   - `./build/bench_refactor_csc`
   - `./build/example_analysis`

## Conclusion

Day 7 fixes the first Cholesky CSC extraction boundary explicitly:

- move the full Cholesky-owned supernodal backend into
  `src/sparse_chol_csc_supernodal.c`
- keep lifecycle/conversion, scalar/native core, wrapper glue, CSC writeback,
  and shared dense LDLT helpers in `src/sparse_chol_csc.c`
- reuse the existing internal header for Phase 2
- preserve the full scalar/supernodal, dispatch, threshold, and proof contract
  exactly

That gives Sprint 56 a concrete, bounded, maintainability-first Day 8 landing
plan.
