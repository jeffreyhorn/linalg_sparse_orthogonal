# Sprint 101 Day 4 Compressed-First API Design

## Purpose

Day 4 turns the Day 2 storage audit and Day 3 solver audit into a bounded
CSR/CSC-first API design. The design intentionally avoids a broad solver API
rewrite. Sprint 101 should make the existing compressed constructors the clear
front door, preserve the mutable matrix shell as the compatibility object, and
prove the selected behavior with focused tests and documentation.

## Reconciled Audit Inputs

| input | strongest signal | design response |
|---|---|---|
| Day 2 storage audit | `sparse_create_from_csr` and `sparse_create_from_csc` already are real public compressed-first constructors | promote existing constructors rather than inventing a new parallel constructor family |
| Day 2 storage audit | `sparse_from_csr` and `sparse_from_csc` keep explicit `sparse_err_t` diagnostics but read like older conversion wrappers | clarify them as diagnostic compressed constructors in header/docs/tests |
| Day 2 storage audit | tutorial and examples still begin from incremental matrix-shell insertion | add a compressed-input learning path after design and implementation scope freeze |
| Day 2 storage audit | CSR/CSC import still builds the public shell through insertion | treat build-path optimization as optional and bounded, not required for the product-model claim |
| Day 3 solver audit | solver APIs mostly accept `SparseMatrix` even when internals dispatch to compressed kernels | define the front door as compressed data to public matrix shell to solver entry |
| Day 3 solver audit | repeated direct lifecycle is a strong explicit owner model | include repeated direct lifecycle in the compressed-input workflow narrative |
| Day 3 solver audit | matrix-free iterative paths can wrap caller-native compressed storage | document as an expert option, not the default CSR/CSC front door |
| Day 3 solver audit | QR, SVD, eigensolver, and preconditioners remain `SparseMatrix`-first | record broad direct CSR/CSC solver parity as a non-goal |

## Selected API Refinements

| selection | type | files likely affected | rationale |
|---|---|---|---|
| clarify `sparse_create_from_csr/csc` as the simple compressed-first front door | header/docs refinement | `include/sparse_csr.h`, `README.md`, `docs/tutorial.md` | these functions already provide the intended product path for callers with CSR/CSC arrays |
| clarify `sparse_from_csr/csc` as diagnostic front-door constructors | header/docs/test refinement | `include/sparse_csr.h`, `tests/test_csr.c`, `README.md` | retains existing ABI while making explicit-status construction discoverable |
| strengthen bad-input and ownership tests for CSR/CSC constructors | test refinement | `tests/test_csr.c` or a focused helper/test addition | validates null, shape, monotonicity, index range, duplicate/order, output-null, and caller-owned input semantics |
| add a compressed-input-to-solver adoption example or tutorial subsection | docs/example refinement | `docs/tutorial.md`, `examples/README.md`, possibly one focused `examples/example_*` file | closes the learning-path gap without changing solver APIs |
| prove CSR/CSC-built matrices work with representative solver entry paths | test refinement | bounded tests in existing suites | supports Sprint 102 oracle seeds without claiming full solver parity |

## Deferred or Rejected API Ideas

| idea | decision | reason |
|---|---|---|
| add new direct CSR/CSC solver entry points for LU, Cholesky, LDLT, QR, SVD, eigensolvers, and preconditioners | defer | too broad for Sprint 101 and would imply solver-family parity that the audits did not prove |
| replace `SparseMatrix` as the public solver coefficient object | reject | breaks compatibility and exceeds the sprint goal |
| rename `sparse_from_csr/csc` | reject | unnecessary ABI churn; documentation can clarify diagnostic use |
| add adopt/no-copy CSR/CSC constructors | defer | useful but requires lifetime, aliasing, mutation, and failure cleanup contracts that need separate design |
| change Matrix Market import to publish compressed objects | defer | lower value than constructor/docs/tests and not needed for the Sprint 101 front door |
| optimize the internal CSR/CSC build path unconditionally | optional/defer | valuable only if Day 5 can bound code risk and validation; front-door clarity is higher value |
| promote `lu_csr_factor_solve` as the default solver path | reject | it is an expert working-format API, while normal callers should use the public matrix shell and solver family APIs |

## Ownership and Lifetime Contract

| object | owner | lifetime rule | mutation rule |
|---|---|---|---|
| caller-provided `SparseCsr` / `SparseCsc` | caller | must remain valid for the duration of constructor call only | constructor must not mutate arrays |
| returned `SparseMatrix *` from `sparse_create_from_csr/csc` | caller | caller frees with `sparse_free` | behaves like any other public matrix shell after construction |
| returned `SparseMatrix **` from `sparse_from_csr/csc` | caller on success | set to `NULL` on failure | same matrix-shell behavior on success |
| exported `SparseCsr *` / `SparseCsc *` from `sparse_to_csr/csc` | caller | caller frees with `sparse_csr_free` or `sparse_csc_free` | exported arrays are independent of source matrix |
| solver factors and handles | caller after initialization/factorization | free with family-specific free functions | factor/handle reuse rules remain family-specific |

No selected Day 4 refinement changes ownership from copy/build semantics to
adopted external storage. A successful compressed constructor owns a new
`SparseMatrix`; the caller still owns and may free or reuse the CSR/CSC arrays
after construction.

## Error Semantics

| API | current behavior to preserve | Day 4 design requirement |
|---|---|---|
| `sparse_create_from_csr` | returns `SparseMatrix *` or `NULL` | keep simple constructor semantics; document that `NULL` covers invalid input and allocation failure |
| `sparse_create_from_csc` | returns `SparseMatrix *` or `NULL` | same as CSR |
| `sparse_from_csr` | returns `sparse_err_t`; sets output to `NULL` before validation/build | clarify as the diagnostic constructor for callers that need status |
| `sparse_from_csc` | returns `sparse_err_t`; sets output to `NULL` before validation/build | same as CSR |
| `sparse_to_csr/csc` | returns `sparse_err_t`; output set to `NULL` on error | preserve existing export semantics |

Validation expectations for diagnostic constructors:

- null input returns `SPARSE_ERR_NULL`;
- null output pointer returns `SPARSE_ERR_NULL`;
- negative dimensions or `nnz` return `SPARSE_ERR_BADARG`;
- missing pointer arrays with nonzero `nnz` return `SPARSE_ERR_BADARG`;
- pointer arrays must start at zero, be monotonic, stay within `nnz`, and end
  at `nnz`;
- row/column indices must be in range;
- per-row CSR columns and per-column CSC rows must be strictly increasing;
- duplicate structural entries are rejected by the strict-order rule;
- allocation failures remain `SPARSE_ERR_ALLOC`;
- output matrix remains `NULL` on failure.

## Compatibility Behavior Table

| caller pattern | compatibility status | Day 4 behavior |
|---|---|---|
| small/ad hoc matrix creation with `sparse_create` and `sparse_insert` | fully supported | keep as mutable-shell compatibility path |
| callers already holding CSR/CSC arrays | promoted path | use `sparse_create_from_csr/csc` for simple construction or `sparse_from_csr/csc` for diagnostics |
| one-shot direct callers | fully supported | construct shell from CSR/CSC if needed, copy before in-place LU/Cholesky as usual |
| repeated direct callers | fully supported | construct shell from CSR/CSC if needed, then use `sparse_analyze` / `sparse_factor_numeric` / `sparse_refactor_numeric` |
| iterative callers with native compressed storage | supported expert option | use matrix-free callbacks when avoiding shell construction is more important than the standard matrix API |
| QR/SVD/eigensolver/preconditioner callers | `SparseMatrix`-first | construct or load a public matrix shell, then use existing family APIs |
| lower-level LU CSR users | expert API | keep `sparse_lu_csr.h` documented as a working-format path, not the general front door |

## Validation Requirements

| refinement | required validation |
|---|---|
| header wording for constructor/front-door semantics | `make format && make lint && make test` if `.h` changes land |
| diagnostic constructor tests | focused `test_csr` run if available, then full required quality chain for code/header/test changes |
| compressed-input solver smoke tests | focused suite for touched tests plus full required quality chain |
| tutorial/examples wording | `git diff --check` and trailing-whitespace scan; full quality chain only if `.c` or `.h` examples/source change |
| optional internal build-path improvement | focused CSR tests and solver smoke tests, then `make format && make lint && make test` |

## Day 5 Boundary Recommendation

Day 5 should freeze a small implementation batch:

1. Update `include/sparse_csr.h` wording only if needed to make simple versus
   diagnostic constructor roles unmistakable.
2. Add or strengthen `tests/test_csr.c` cases for explicit status, failure
   output nulling, strict ordering, duplicates, and caller-owned input arrays.
3. Add a compressed-input solver smoke proof using existing constructors and
   one or two representative solver families.
4. Schedule documentation/example updates after the implementation proof, not
   before.
5. Defer adopt/no-copy constructors, broad direct CSR/CSC solver APIs, and
   internal build-path optimization unless Day 5 can show a very small, easily
   validated patch.

## Non-Claims

This design does not claim:

- direct CSR/CSC solver parity across all solver families;
- replacement of `SparseMatrix` as the public coefficient object;
- elimination of the mutable matrix shell;
- no-copy CSR/CSC ownership transfer;
- Matrix Market compressed-object publication;
- state-of-the-art sparse solver performance from constructor clarity alone.

## Day 4 Conclusion

Sprint 101 should make the existing CSR/CSC constructors the clear product
front door and prove that path through tests and documentation. The selected
scope preserves ABI compatibility, avoids broad solver API churn, and gives
Day 5 a concrete implementation boundary: clarify the existing API contract,
prove diagnostics and ownership, and connect compressed input to existing
solver workflows without overstating solver-family parity.
