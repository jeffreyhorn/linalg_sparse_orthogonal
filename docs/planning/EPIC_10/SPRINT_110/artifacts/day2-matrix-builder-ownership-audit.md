# Day 2 Matrix Builder Ownership Audit

## Purpose

Day 2 audits the private Matrix builder seam before any Matrix Market source
movement. Sprint 110 cannot safely move `sparse_load_mm` toward a Matrix I/O
owner until the shared bulk-entry builder contract is understood, because the
same helper is used by copy, transpose, and Matrix Market load behavior.

## Source Inputs

- `docs/planning/EPIC_10/PROJECT_PLAN.md`, Sprint 110 Item 2.
- `docs/planning/EPIC_10/SPRINT_110/PLAN.md`, Day 2.
- `docs/planning/EPIC_10/SPRINT_109/artifacts/day8-matrix-shell-candidate-boundary-contract.md`.
- `docs/planning/EPIC_10/SPRINT_109/artifacts/day14-sprint-closeout-residual-queue.md`.
- `src/sparse_matrix.c`.
- `src/sparse_csr.c`.
- `tests/test_sparse_matrix.c`.
- `tests/test_sparse_io.c`.
- `tests/test_csr.c`.

## Live Builder Objects

| Object | Current Owner | Visibility | Role |
|---|---|---|---|
| `SparseBuildEntry` | `src/sparse_matrix.c` | file-static typedef | Row, column, scalar value, and insertion order record for bulk matrix construction. |
| `sparse_build_entry_cmp` | `src/sparse_matrix.c` | file-static helper | Sorts entries by row, column, then order. |
| `sparse_matrix_build_from_entries` | `src/sparse_matrix.c` | file-static helper | Builds a linked-list `SparseMatrix` from bulk entries with optional sorting, duplicate collapse, zero dropping, and row/column link construction. |

## Direct Builder Callers

| Caller | Input Stream | `entries_sorted` | Public Behavior Coupled To Builder |
|---|---|---:|---|
| `sparse_copy` | existing physical row-order traversal | `1` | Copies physical nonzeros into a fresh shell, then separately clones permutations, cached norm, factor state, and reorder permutation. |
| `sparse_transpose` | physical nonzeros with row/column swapped | `0` | Produces a matrix with shape `(cols, rows)`, sorted transposed physical storage, and preserved nonzero values. |
| `sparse_load_mm` | parsed Matrix Market coordinates after one-based to zero-based conversion | `0` | Builds loaded matrices after duplicate, symmetric, pattern, coordinate, allocation, and parse handling. |

No other source directly calls `sparse_matrix_build_from_entries` today.

## CSR/CSC Constructor Relationship

CSR/CSC constructors are adjacent Matrix front-door behavior, but they do not
use the builder seam today:

- `sparse_from_csr` validates compressed rows, sorted columns, duplicates, and
  bounds in `src/sparse_csr.c`, then populates through `sparse_insert`.
- `sparse_from_csc` validates compressed columns, sorted rows, duplicates, and
  bounds in `src/sparse_csr.c`, then populates through `sparse_insert`.
- `sparse_create_from_csr` and `sparse_create_from_csc` are convenience wrappers
  over the explicit conversion APIs.

Implication: moving the builder does not automatically touch CSR/CSC
constructors, but future builder ownership should not be described as the
compressed-constructor owner unless those constructors are deliberately
rewired and revalidated.

## Builder-Coupled Public Behavior

| Behavior | Current Implementation Detail | Public Risk If Moved Incorrectly |
|---|---|---|
| Entry ordering | Optional `qsort` by row, column, and original order. | Transpose and Matrix Market load could change physical traversal, duplicate resolution, or deterministic roundtrips. |
| Duplicate handling | Consecutive equal `(row, col)` entries collapse to the final value after sorting. | Matrix Market duplicate-last-write-wins behavior can regress. |
| Zero handling | The final collapsed value is skipped if it is `0.0`. | Explicit zero semantics could diverge from the existing sparse-get zero-as-absent contract. |
| Bounds handling | Builder returns `SPARSE_ERR_BOUNDS` if an entry falls outside requested dimensions. | Matrix Market parse errors must not become unexpected bounds errors for coordinate validation paths already checked before build. |
| Allocation handling | Matrix shell, row-tail, col-tail, and node allocation failures return allocation errors and free partial state. | Copy, transpose, and load could leak partial matrices or return half-built objects. |
| Empty matrices | `nentries == 0` returns a valid empty matrix after `sparse_create`. | Empty Matrix Market roundtrip and transpose behavior could regress. |
| Row/column links | Builder constructs both row and column linked lists directly. | Matvec, get/set, transpose, factorization, and solver consumers can observe corrupt storage if one side is incomplete. |
| Factor/permutation state | Builder only creates a fresh shell; `sparse_copy` separately clones state afterward. | Moving builder must not accidentally take ownership of factor-state clone behavior. |

## Tests Currently Covering The Coupled Behavior

| Behavior Surface | Current Tests |
|---|---|
| Copy shape, values, independence, and null handling | `tests/test_sparse_matrix.c` copy tests. |
| Transpose identity, double transpose, rectangular, symmetric, row/column vector, nnz, SuiteSparse symmetric/unsymmetric cases | `tests/test_sparse_matrix.c` transpose tests. |
| Matrix Market roundtrip, rectangular, precision, nnz, 1x1, empty, negative values, permutation roundtrip | `tests/test_sparse_io.c`. |
| Matrix Market symmetric and pattern loading | `tests/test_sparse_io.c`. |
| Matrix Market malformed headers, negative dimensions, rectangular symmetric headers, out-of-range and zero coordinates | `tests/test_sparse_io.c`. |
| Matrix Market errno capture and clear-on-success behavior | `tests/test_sparse_io.c`. |
| Duplicate-last-write-wins | `tests/test_sparse_matrix.c::test_load_mm_duplicate_last_write_wins`. |
| CSR/CSC constructor behavior outside builder path | `tests/test_csr.c`. |

## Ownership Options For Day 3

### Option A: Keep Builder Central

Keep `SparseBuildEntry`, `sparse_build_entry_cmp`, and
`sparse_matrix_build_from_entries` file-static in `src/sparse_matrix.c`.

Pros:
- lowest public behavior risk;
- no new private header or source-list membership;
- copy/transpose/load remain on the same private storage owner.

Cons:
- Matrix Market load cannot move cleanly into `src/sparse_matrix_io.c`;
- `src/sparse_matrix.c` remains the owner for both shell behavior and file I/O.

Best fit if Day 3 decides the builder is still too coupled to central shell
storage.

### Option B: Split Builder Into A Private Source Owner

Move builder record and helper behavior into a private source such as
`src/sparse_matrix_build_internal.c`, with private declarations in an internal
header consumed by `src/sparse_matrix.c` and a future `src/sparse_matrix_io.c`.

Pros:
- unblocks a later Matrix Market source split without exposing builder helpers
  publicly;
- keeps copy, transpose, and load on one shared builder implementation;
- makes the builder contract explicit.

Cons:
- requires careful private access to `SparseMatrix`, `Node`, pool allocation,
  and row/column header internals;
- requires Makefile, CMake, and `build-metadata/library_sources.txt` parity;
- may need a private header that exposes low-level matrix internals more widely.

Best fit only if private header boundaries can stay internal and focused tests
can prove copy, transpose, and Matrix Market behavior.

### Option C: Defer Builder Split And Matrix Market Movement

Publish a no-split deferral if builder extraction would require unsafe internal
exposure or too much matrix-shell movement.

Pros:
- preserves all public behavior;
- avoids turning a source-boundary sprint into a broad matrix-shell rewrite.

Cons:
- leaves Matrix Market load/save in `src/sparse_matrix.c`;
- pushes file-I/O source ownership into a downstream sprint.

Best fit if Day 3 cannot prove Option B is low risk.

## Extraction Risk Checklist

Before choosing Option B, Day 3 must prove:

- no public header or install-header change is needed;
- private declarations do not expose builder helpers to installed surfaces;
- `SparseMatrix`, `Node`, pool allocation, row headers, column headers, and
  `nnz` mutations remain internally consistent;
- copy, transpose, and Matrix Market load keep the same error behavior;
- duplicate-last-write-wins and final-zero-drop behavior remain unchanged;
- empty Matrix Market and empty transpose/copy behavior remains unchanged;
- `sparse_copy` continues to clone permutation arrays, cached norms, factor
  state, and reorder permutation outside the builder helper;
- source membership changes are applied consistently to Makefile, CMake, and
  `build-metadata/library_sources.txt`;
- focused validation covers `test_sparse_matrix`, `test_sparse_io`, `test_csr`,
  and at least one solver-smoke lane that loads Matrix Market data.

## Day 3 Decision Criteria

Day 3 should choose:

- **private builder source** only if the private boundary can be implemented
  without public headers, install headers, helper targets, or broad shell
  behavior movement;
- **central builder ownership** if builder internals remain too tightly coupled
  to `SparseMatrix` storage and factor/permutation compatibility;
- **no-split deferral** if neither option can be validated within the Sprint
  110 risk budget.

## Completion Criteria Status

- All direct builder callers are known: `sparse_copy`, `sparse_transpose`, and
  `sparse_load_mm`.
- CSR/CSC constructors are confirmed adjacent but not direct builder users.
- Matrix Market movement prerequisites are explicit.
- Public behavior risks are documented before a Day 3 decision.
- No Matrix Market source split has begun before the builder boundary decision.
