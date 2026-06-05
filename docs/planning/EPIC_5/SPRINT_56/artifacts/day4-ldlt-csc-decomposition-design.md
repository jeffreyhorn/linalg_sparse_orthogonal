# Sprint 56 Day 4 - LDLT CSC decomposition design

Date: 2026-06-05
Branch: `sprint-56`

## Scope

Freeze the first LDLT CSC extraction boundary before editing permanent
implementation files, using the Day 3 supernodal-first ranking as the design
anchor.

## Selected first extraction seam

Sprint 56 Batch 1 should extract the supernodal LDLT CSC helper cluster into
its own source file:

- new file:
  - `src/sparse_ldlt_csc_supernodal.c`

The first moved function set should be:

- `ldlt_csc_bsearch_row_map(...)`
- `ldlt_csc_supernode_extract(...)`
- `ldlt_csc_supernode_writeback(...)`
- `ldlt_csc_supernode_eliminate_diag(...)`
- `ldlt_csc_supernode_eliminate_panel(...)`
- `ldlt_csc_eliminate_supernodal(...)`

These functions already form a contiguous backend-owned block inside
`src/sparse_ldlt_csc.c` and have the strongest dedicated proof surface in
`tests/test_ldlt_csc.c`, with repeated-run behavioral confirmation in
`benchmarks/bench_refactor_csc.c`.

## File-boundary ownership map

### Keep in `src/sparse_ldlt_csc.c`

- lifecycle / storage / structural conversion
- row-adjacency management
- supernode detection
- wrapper compatibility path
- scalar/native Bunch-Kaufman kernel core
- native elimination driver
- solve path
- validation and CSC-to-public writeback

### Move to `src/sparse_ldlt_csc_supernodal.c`

- supernodal row-map lookup helper
- supernode extract logic
- supernode CSC writeback logic
- supernodal diagonal-block eliminate helper
- supernodal panel eliminate helper
- supernodal elimination driver

## Internal declaration strategy

Sprint 56 Phase 2 should keep the moved declarations in the existing:

- `src/sparse_ldlt_csc_internal.h`

Reason:

- the first batch already changes one major ownership axis:
  - source-file extraction
- adding a dedicated new private-header taxonomy in the same batch would mix:
  - source extraction
  - private-header redesign
- the existing internal header already contains the right private contract for
  both the scalar/native and supernodal CSC paths

Deferred by design:

- creation of `src/sparse_ldlt_csc_supernodal_internal.h`
- broader repartitioning of `src/sparse_ldlt_csc_internal.h`

## Invariants the first batch must preserve

### Public contract invariants

- no public header/API changes
- no user-visible direct-solver lifecycle behavior change
- no change to the public analysis/factors path that ultimately reaches the CSC
  completion seam

### Native/wrapper invariants

- `ldlt_csc_eliminate(...)` runtime override behavior unchanged
- native versus wrapper routing unchanged
- linked-list comparison path retained for regression and benchmark use

### Numerical/storage invariants

- permutation semantics unchanged
- pivot-size semantics unchanged
- `D` / `D_offdiag` handoff unchanged
- residual and inertia parity unchanged
- row-adjacency assumptions unchanged

### Proof-surface invariants

- `tests/test_ldlt_csc.c` remains the primary direct proof surface
- `tests/test_integration.c` remains unchanged in meaning
- `benchmarks/bench_refactor_csc.c` keeps both:
  - SPD repeated-run Cholesky CSC evidence
  - indefinite repeated-run LDLT CSC evidence

## Minimal comment policy for the first batch

Preserve:

- durable algorithm meaning
- ownership boundaries
- pivot/permutation invariants
- writeback/drop-threshold semantics

Reduce where touched:

- sprint chronology
- implementation-history narrative
- comments that explain landing order instead of current code truth

Do not try in Batch 1:

- repo-wide LDLT CSC comment normalization
- private-header taxonomy cleanup
- broad CSC doc rewriting

## Expected Day 5 touched files

Primary expected touched set:

- `src/sparse_ldlt_csc.c`
- `src/sparse_ldlt_csc_supernodal.c` (new)
- `src/sparse_ldlt_csc_internal.h`
- `Makefile`
- `CMakeLists.txt`

Secondary touch only if truly needed:

- `tests/test_ldlt_csc.c`

Avoid by default:

- `include/sparse_ldlt.h`
- `tests/test_integration.c`
- `benchmarks/bench_refactor_csc.c`
- `src/sparse_chol_csc.c`

## Landing checklist

Before calling Batch 1 complete:

1. The supernodal LDLT CSC helper/driver cluster lives in
   `src/sparse_ldlt_csc_supernodal.c`.
2. `src/sparse_ldlt_csc.c` still owns lifecycle/conversion, wrapper
   compatibility, and the scalar/native kernel.
3. No public header/API changes are introduced.
4. Touched comments reflect durable code truth rather than sprint history.
5. Validation passes:
   - `make format`
   - `make lint`
   - `make test`
   - `make quality-review-full`
6. High-signal follow-ons remain green:
   - `./build/test_ldlt_csc`
   - `./build/test_ldlt`
   - `./build/test_integration`
   - `./build/bench_refactor_csc`
   - `./build/example_analysis`

## Conclusion

Day 4 fixes the first LDLT CSC extraction boundary explicitly:

- move the supernodal helper cluster into `src/sparse_ldlt_csc_supernodal.c`
- keep lifecycle/conversion, wrapper compatibility, and scalar/native kernel
  ownership in `src/sparse_ldlt_csc.c`
- reuse the existing internal header for Phase 2
- preserve the full native/wrapper/permutation/proof contract exactly

That gives Sprint 56 a concrete, bounded, maintainability-first Day 5 landing
plan.
