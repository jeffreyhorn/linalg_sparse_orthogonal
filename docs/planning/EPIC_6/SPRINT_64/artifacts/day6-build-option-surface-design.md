# Sprint 64 Day 6: Build/Option Surface Design

Date: 2026-06-11
Branch: `sprint-64`

## Purpose

Convert the Day 5 backend abstraction contract into an exact build/options
wiring plan for the first Sprint 64 landing without widening the public surface
or weakening the self-contained default build.

## Main Decision

### 1. The first Sprint 64 landing should not add a new public runtime/backend option

The repository already exposes public backend selectors where the product
contract truly needs them:

- `sparse_cholesky_opts_t::backend`
- `sparse_ldlt_opts_t::backend`
- `sparse_eigs_opts_t::backend`

The first Sprint 64 Cholesky CSC supernodal landing does not justify that same
surface expansion.

Therefore:

- no new public header field should be added
- no new runtime/backend forcing knob should be documented
- no env-var control should be introduced

## Build/Option Contract

### 2. If a toggle is needed, it should be build-time, target-private, and default-safe

The current build surfaces already express optional implementation features via
compile-time switches:

- `SPARSE_OPENMP`
- `SPARSE_MUTEX`

Sprint 64 should follow the same broad pattern only if the implementation
truly needs a selectable branch, and then keep it narrower:

- the toggle remains internal-first
- the compile definition remains `PRIVATE` to `sparse_lu_ortho`
- the default preserves today’s authoritative self-contained path
- the first landing does not turn the new path into a broad public product
  promise

## Preferred Wiring Order

### 3. The minimum viable plan is now explicit

Preferred order:

1. no new build toggle if the selected kernel modernization can land directly
   on the existing authoritative path
2. if a toggle is actually needed:
   - `CMakeLists.txt` gains one bounded option for the selected Cholesky CSC
     supernodal lane
   - `Makefile` mirrors that option with the same semantics
   - both surfaces emit only a target-private compile definition
3. no public runtime forcing in Sprint 64 Phase 1

This keeps the default path truthful while still permitting a bounded internal
selection seam if the code landing needs it.

## Internal Ownership

### 4. The internal policy home is the Cholesky CSC seam

The natural policy/config home for the first landing is:

- `src/sparse_chol_csc_internal.h`

That means:

- local policy enums, macros, or helper declarations belong there
- Sprint 64 should not add a new repository-wide backend-config header
- Sprint 64 should not widen `include/` to expose implementation-local policy

### 5. `src/sparse_dense.c` remains support-only

If the first landing needs shared dense-helper support, it can touch:

- `src/sparse_dense.c`

But that file remains:

- a bounded helper seam
- not a new global backend center
- not a justification for QR/SVD-wide dense unification

## Day 7-10 Implementation Fence

### 6. Exact touched-surface map

Required first-landing implementation surface:

- `src/sparse_chol_csc_supernodal.c`

Likely internal support surface:

- `src/sparse_chol_csc_internal.h`
- `src/sparse_dense.c`

Likely proof surfaces:

- `tests/test_chol_csc.c`
- `tests/test_integration.c`
- `benchmarks/bench_chol_csc.c`

Conditional build/config surfaces only if the implementation proves they are
needed:

- `CMakeLists.txt`
- `Makefile`

Later truth surfaces only after implementation shape is real:

- `benchmarks/README.md`
- `docs/maintainer_guide.md`

## Explicit Non-Goals

The first Sprint 64 landing should not widen into:

- `include/sparse_cholesky.h` public option growth
- a new repository-wide backend-config layer
- `src/sparse_ldlt_csc_supernodal.c`
- `src/sparse_qr.c`
- `src/sparse_svd.c`
- benchmark-governance redesign
- packaging/platform work

## Exit State

Sprint 64 now has an explicit build/options plan before implementation:

- no new public runtime/backend option is justified
- any needed toggle should be build-time, target-private, and default-safe
- `src/sparse_chol_csc_internal.h` is the natural internal policy home
- `CMakeLists.txt` and `Makefile` are conditional support surfaces rather than
  mandatory first-batch edits
- the Day 7-10 implementation fence is explicit before code moves
