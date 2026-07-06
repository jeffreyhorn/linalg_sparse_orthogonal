# Day 13 Central Matrix Shell Deferral Contract

## Purpose

Day 13 documents why `src/sparse_matrix.c` remains central public API and
compatibility territory in Sprint 107. The goal is to prevent opportunistic
source extraction from bypassing public behavior review, install-header
review, solver-entry compatibility, or the Sprint 101 compressed-first product
model.

## Current Owner Snapshot

- `src/sparse_matrix.c`: 1,359 lines.
- Public header: `include/sparse_matrix.h`.
- Private storage/state headers:
  - `src/sparse_matrix_internal.h`
  - `src/sparse_matrix_state_internal.h`
- Central responsibilities:
  - node pool allocation and reuse;
  - bulk shell construction from normalized entries;
  - public lifecycle: `sparse_create`, `sparse_free`, `sparse_copy`,
    `sparse_transpose`;
  - mutation: `sparse_insert`, `sparse_remove`, `sparse_set`, `sparse_scale`,
    `sparse_add_inplace`;
  - physical and logical access through row/column permutations;
  - matrix information, norm caching, symmetry checks, memory accounting;
  - arithmetic: add, in-place add, matrix multiply, matvec, block matvec;
  - Matrix Market load/save;
  - debug/printing helpers;
  - permutation and one-shot factor compatibility reset behavior.

## Sprint 101 Contract To Preserve

Sprint 101 established a bounded compressed-first model:

- callers with CSR/CSC arrays can use public compressed constructors as the
  clearer front door;
- those constructors validate and copy caller-owned data into an independent
  public `SparseMatrix` shell;
- normal solver families still accept `SparseMatrix` as the public
  coefficient object;
- mutable matrix-shell construction remains supported compatibility;
- the project does not claim broad direct CSR/CSC solver parity or replacement
  of the public matrix shell.

Any `src/sparse_matrix.c` split must preserve that product model. A source
split must not imply:

- deprecation of `sparse_create` / `sparse_insert` mutable construction;
- no-copy or adopt ownership for CSR/CSC data;
- direct compressed solver entry across solver families;
- a new public storage ABI;
- install-header expansion.

## Why Sprint 107 Does Not Extract `src/sparse_matrix.c`

`src/sparse_matrix.c` is smaller than several already-addressed proof owners
and has unusually high public behavior density. Its boundaries are not just
implementation convenience; they encode compatibility:

- `sparse_create`, shell buffer allocation, and pool initialization define the
  public object lifecycle.
- `sparse_insert`, `sparse_remove`, `sparse_set`, `sparse_scale`, and
  `sparse_add_inplace` clear or preserve factor compatibility state in ways
  solver callers observe.
- `sparse_get` and `sparse_matvec` depend on logical-to-physical permutation
  semantics after factorization or reordering.
- `sparse_copy` preserves permutation and factor compatibility state by
  design, while `sparse_reset_perms` deliberately drops compatibility when
  appropriate.
- Matrix Market load/save and print paths expose logical ordering and parsing
  contracts to users and tests.
- Internal headers expose `SparseMatrix`, `Node`, `NodePool`, and factor-state
  hooks to other implementation files, so splitting without a header-boundary
  review can accidentally widen private coupling.

For Sprint 107, a source extraction would create more review and validation
surface than it removes.

## Future Split Preconditions

Future matrix-shell extraction must begin with a fresh public-behavior review
and should satisfy all applicable prerequisites below before code moves.

| Candidate Area | Preconditions | Required Evidence |
|---|---|---|
| Node pool owner | Prove pool helpers remain private and no solver/source file starts depending on pool internals directly. | `test_sparse_matrix`, allocator/failure-path coverage, full quality gate. |
| Bulk entry builder | Define duplicate-entry, zero-drop, sort-order, bounds, Matrix Market, CSR/CSC import, copy, and transpose semantics before extraction. | `test_sparse_matrix`, `test_sparse_io`, `test_csr`, Matrix Market tests, compressed constructor smoke tests. |
| Matrix Market I/O owner | Preserve errno behavior, parse errors, symmetric/pattern handling, logical ordering, and 1-based file coordinate conversion. | `test_sparse_io`, known-matrix fixtures, docs/example smoke if changed. |
| Arithmetic/matvec owner | Preserve logical permutation semantics, cache invalidation, factor-state clearing, OpenMP behavior, and physical-index caveats. | `test_sparse_arith`, `test_matmul`, `test_sparse_matrix`, iterative/eigensolver smoke if matvec behavior changes. |
| Factor compatibility owner | Preserve `sparse_copy`, `sparse_mark_factored`, `sparse_reset_perms`, row/col permutation accessors, and solver compatibility hooks. | direct solver tests, analysis/refactor tests, Cholesky/LU compatibility tests. |
| Public header cleanup | Separate documentation-only wording from ABI or install-header changes. | install/export checks, downstream consumer checks, public docs review. |

## Required Build And Validation For Future Extraction

Any future source split from `src/sparse_matrix.c` must update and validate:

- Makefile object/source membership.
- CMake target source membership.
- source-list parity expectations.
- public install-header set, with an explicit no-change statement unless a
  public API review approves a new header.
- focused matrix-shell tests for the moved behavior.
- solver-entry smoke tests if factor compatibility, permutation semantics, or
  matvec behavior changes.
- full quality gate:

```sh
make format && make lint && make test
```

## Maintainer Guidance

Do not extract from `src/sparse_matrix.c` as a line-count cleanup. Extract only
when the candidate has:

1. a named public-behavior boundary;
2. a private-header dependency plan;
3. a Make/CMake/source-list parity plan;
4. focused tests that observe the exact behavior being moved;
5. an explicit statement that Sprint 101's compressed-first story still
   routes through the public matrix shell unless a separate API project says
   otherwise.

## Sprint 107 Outcome

Sprint 107 records `src/sparse_matrix.c` as deferred central API debt, not a
failed cleanup item. The correct handoff is a future boundary-first project
around one candidate area, with public behavior and build-system parity
defined before any source movement.
