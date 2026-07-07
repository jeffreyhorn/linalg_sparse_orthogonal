# Day 3 Matrix Builder Ownership Decision

## Purpose

Day 3 turns the Matrix builder audit into a concrete ownership decision and
publishes the prerequisite contract for Matrix Market source movement. This is
the gate that prevents `sparse_load_mm` and `sparse_save_mm` from moving before
their shared bulk-entry construction behavior has a stable private owner.

## Source Inputs

- `docs/planning/EPIC_10/SPRINT_110/PLAN.md`, Day 3.
- `docs/planning/EPIC_10/SPRINT_110/artifacts/day2-matrix-builder-ownership-audit.md`.
- `src/sparse_matrix.c`.
- `src/sparse_matrix_internal.h`.
- `src/sparse_alloc_internal.h`.
- `Makefile`.
- `CMakeLists.txt`.
- `build-metadata/library_sources.txt`.

## Decision

`SparseBuildEntry`, `sparse_build_entry_cmp`, and
`sparse_matrix_build_from_entries` should move from file-static ownership in
`src/sparse_matrix.c` into a private builder implementation source before
Matrix Market load/save moves to a Matrix I/O source.

Provisional owner:

- `src/sparse_matrix_build_internal.c`

Provisional private declaration location:

- a narrowly scoped internal declaration in `src/sparse_matrix_internal.h`, or a
  new private builder header if the implementation needs a smaller include
  boundary.

This is an internal implementation decision only. It must not add public API,
installed headers, helper targets, or reviewed CTest registrations.

## Go Rationale

The private builder source path is acceptable because:

- `src/sparse_matrix_internal.h` already exposes the internal `SparseMatrix`
  and `Node` structures to multiple implementation owners;
- pool allocation operations are already available through private declarations;
- the builder can preserve the current direct row/column link construction
  without using public insertion APIs;
- copy, transpose, and Matrix Market load should continue sharing one
  duplicate/zero/order policy instead of growing separate implementations;
- Matrix Market source movement becomes cleaner after builder ownership is no
  longer tied to the central matrix shell file;
- no public or installed header surface is needed.

## Rejected Options

### Keep Builder Central

Rejected for Sprint 110 because it would leave Matrix Market load/save coupled
to `src/sparse_matrix.c` and would force Day 4 to publish a no-split deferral.
This option remains a fallback if implementation later proves the private
builder source unsafe.

### Move Matrix Market Without Moving Builder

Rejected because `sparse_load_mm` would need either duplicated bulk-entry
construction behavior or an unsafe private dependency on file-static helpers in
`src/sparse_matrix.c`.

### Expose Builder Publicly

Rejected outright. The builder is a private storage-construction detail and
must not become part of `include/` or install/export surfaces.

## Owner Contract

The private builder owner must preserve these behaviors exactly:

| Contract Area | Required Behavior |
|---|---|
| Entry record | Preserve row, column, scalar value, and original-order fields. |
| Sorting | Sort by row, column, then original order when entries are not already sorted. |
| Duplicate handling | Collapse duplicate `(row, col)` entries so the last ordered value wins. |
| Zero handling | Drop entries whose final collapsed value is `0.0`. |
| Bounds handling | Return the existing error semantics without leaving a partial matrix behind. |
| Allocation failure | Free row tails, column tails, and partial matrix state before returning allocation failure. |
| Empty input | Return a valid empty matrix created with the requested shape. |
| Row/column links | Build both row and column linked lists consistently and update `nnz` only for retained nonzeros. |
| Copy separation | Do not absorb `sparse_copy` permutation, cached norm, factor-state, or reorder-permutation cloning. |
| Public surface | Do not add public headers, install headers, public functions, helper targets, or CTest registrations. |

## Source-List Contract

If the private builder source is implemented, update all library source lists in
one change:

- `Makefile` `LIB_SRCS`;
- `CMakeLists.txt` library source list;
- `build-metadata/library_sources.txt`.

Required order:

1. `src/sparse_matrix.c`
2. future `src/sparse_matrix_build_internal.c`
3. future `src/sparse_matrix_io.c`, if Matrix Market load/save moves later

Run `make source-list-check` after any source-list edit.

## Matrix Market Prerequisite Checklist

Matrix Market load/save may move toward `src/sparse_matrix_io.c` only after:

- the private builder source has a concrete implementation plan;
- copy, transpose, and Matrix Market load still use the same builder contract;
- Matrix Market load/save has a planned private include path that does not
  expose builder helpers publicly;
- checked stream-write and errno behavior ownership is documented;
- focused matrix tests and solver-smoke lanes are selected;
- Makefile, CMake, and `build-metadata/library_sources.txt` ordering is known;
- no public API, install-header, helper-target, or CTest drift is expected.

## Focused Validation Checklist

Any builder movement must validate at least:

- `make source-list-check`;
- `make test TEST=test_sparse_matrix`, if supported locally, or the equivalent
  focused `test_sparse_matrix` binary;
- `make test TEST=test_sparse_io`, if supported locally, or the equivalent
  focused `test_sparse_io` binary;
- `test_csr` or equivalent CSR/CSC constructor coverage to prove adjacent
  compressed front-door behavior did not drift;
- one solver-smoke lane that loads Matrix Market data before solving, such as
  `test_qr`, `test_suitesparse`, or `test_integration`;
- `make format && make lint && make test` before committing any C or build-file
  implementation.

If the local Makefile does not support per-test selection, Day 4/5 should
record the direct binary command or use full `make test`.

## Day 4/5 Implementation Boundaries

Allowed:

- move builder record/comparator/build helper into a private source;
- add private declarations required by `src/sparse_matrix.c` and future
  Matrix I/O source;
- update Makefile, CMake, and source manifest in reviewed order;
- plan Matrix Market movement after builder source ownership is explicit.

Not allowed:

- public API changes;
- install-header changes;
- new compiled test helper targets;
- reviewed CTest registration changes;
- broad matrix-shell extraction;
- CSR/CSC constructor rewiring;
- changes to duplicate-last-write-wins or final-zero-drop behavior.

## Completion Criteria Status

- The Matrix builder owner decision is explicit: private builder source.
- Matrix Market movement is unblocked only after the private builder source
  plan and validation gates are satisfied.
- No public API or install-header change is implied.
- Focused validation gates are defined.
- Downstream days can proceed without revisiting builder ownership unless
  implementation evidence invalidates the private source path.
