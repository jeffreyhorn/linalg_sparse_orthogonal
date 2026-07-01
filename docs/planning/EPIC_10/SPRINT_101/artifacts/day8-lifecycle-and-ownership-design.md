# Sprint 101 Day 8 Lifecycle and Ownership Design

## Purpose

Day 8 defines the ownership, lifetime, mutation, and repeated-run rules that
compressed-first callers need after the Day 6 constructor/import batch. This
is a design artifact. It does not change APIs or code.

## Current Ownership Surface Audit

| surface | owner | lifetime rule | current clarity |
|---|---|---|---|
| caller-provided `SparseCsr` / `SparseCsc` | caller | must be valid only during constructor call | clear in `include/sparse_csr.h` after Day 6 |
| `sparse_create_from_csr/csc` result | caller | free returned matrix with `sparse_free` | clear in header after Day 6 |
| `sparse_from_csr/csc` result | caller on success | free returned matrix with `sparse_free`; output is set to `NULL` on error | mostly clear; README still calls them compatibility wrappers |
| `sparse_to_csr/csc` result | caller | free with `sparse_csr_free` / `sparse_csc_free` | clear |
| mutable `SparseMatrix` shell | caller | free with `sparse_free`; mutation APIs may change factor state | clear but spread across docs |
| one-shot LU/Cholesky factor shell | caller | factorization mutates a working matrix shell; use a copy when original is needed | clear in README/tutorial |
| LDLT / QR / SVD / ILU / IC result objects | caller | free with family-specific free functions | clear per family, but not integrated into compressed-input workflow |
| direct repeated-run analysis | caller | `sparse_analysis_t` owns symbolic/permutation setup; free with `sparse_analysis_free` | clear in `include/sparse_analysis.h` |
| direct repeated-run factors | caller | `sparse_factors_t` owns numeric factor state; free with `sparse_factor_free` | clear in `include/sparse_analysis.h` |
| iterative handle | caller | zero/init, optionally prepare, solve repeatedly, free with `sparse_iter_handle_free` | clear in `include/sparse_iterative.h` and README |
| eigensolver handle | caller | zero/init, optionally prepare, solve repeatedly, free with `sparse_eigs_handle_free` | clear in `include/sparse_eigs.h` and README |
| eigensolver result buffers | caller | caller allocates eigenvalue/eigenvector buffers and frees them | clear in `include/sparse_eigs.h` |

## Compressed-First Rule Set

| rule | contract |
|---|---|
| construction is copy/build, not adopt | CSR/CSC arrays are read and copied into a new `SparseMatrix`; caller keeps ownership of the arrays |
| compressed input enters solvers through the public matrix shell | normal solver APIs still take `SparseMatrix`; constructor success does not imply direct CSR/CSC solver APIs |
| `SparseMatrix` remains mutable compatibility storage | callers may still use insertion/removal/set APIs; Sprint 101 does not deprecate them |
| one-shot mutating direct solvers need working copies | LU and Cholesky mutate their matrix shell; keep the original matrix or CSR/CSC arrays separately if needed |
| repeated direct reuse is analysis/factor ownership, not matrix ownership | `sparse_analysis_t` and `sparse_factors_t` own reusable state; they do not own the source matrix |
| iterative/eigensolver handles preserve capacity only | handles do not preserve Krylov/Ritz/search state as a numerical feature |
| matrix-free is an expert compressed-native escape hatch | callers can wrap native compressed storage through callbacks, but that is not the default CSR/CSC front door |

## Repeated-Run Lifecycle Implications

| workflow | compressed-first path | ownership implication |
|---|---|---|
| one-shot direct solve | CSR/CSC arrays -> `sparse_create_from_csr/csc` -> optional `sparse_copy` -> one-shot factor/solve | caller owns arrays, matrix shell, and any working factor copy |
| repeated direct solve | CSR/CSC arrays -> `SparseMatrix` shell -> `sparse_analyze` -> `sparse_factor_numeric` / `sparse_refactor_numeric` | analysis owns symbolic/permutation state; factors own numeric factor state; source matrix is not retained |
| iterative one-shot | CSR/CSC arrays -> `SparseMatrix` shell -> `sparse_solve_*` | solver reads matrix; no solver-owned matrix reference survives |
| iterative repeated-run | CSR/CSC arrays -> `SparseMatrix` shell -> iterative handle prepare/solve/free | handle owns workspace capacity only; matrix remains caller-owned |
| eigensolver one-shot | CSR/CSC arrays -> `SparseMatrix` shell -> caller-owned result buffers -> `sparse_eigs_sym` | result buffers are caller-owned; matrix is read-only |
| eigensolver repeated-run | CSR/CSC arrays -> `SparseMatrix` shell -> eigensolver handle prepare/solve/free | handle owns workspace capacity only; result buffers remain caller-owned |
| preconditioned iterative solve | CSR/CSC arrays -> `SparseMatrix` shell/copy -> preconditioner factor -> iterative solve | preconditioner object owns factor state; build from fresh/original shell when factor state may be present |

## Mutation and Factored-State Rule Map

| state | allowed action | caution |
|---|---|---|
| freshly constructed CSR/CSC matrix shell | any normal matrix operation | it is a normal `SparseMatrix`, not a compressed object |
| matrix needed later in original coefficient form | use `sparse_copy` before mutating direct factorization | LU/Cholesky factorization overwrites the working shell |
| matrix already factored or reordered | avoid CSR/CSC export and preconditioner/fresh-analysis use unless reset/copied as appropriate | conversion operates in physical index space and headers warn against non-identity permutations |
| repeated direct analysis object exists | matrix may be freed or replaced if later factor/refactor inputs preserve required dimensions/pattern | analysis does not retain source matrix |
| repeated direct factors exist | `sparse_refactor_numeric` may replace numeric state on success | failed refactor preserves old usable factors |
| iterative/eigensolver handle prepared | can reuse for same/smaller compatible work | prepare/solve reuse is allocation reuse only, not numerical-state reuse |
| caller mutates original CSR/CSC arrays after construction | returned matrix remains unchanged | Day 6 tests prove copy ownership |

## Day 9 Follow-Through Queue

| priority | item | likely file | validation |
|---:|---|---|---|
| 1 | update README API wording that still calls `sparse_from_csr/csc` retained compatibility wrappers | `README.md` | docs hygiene only |
| 2 | add compact compressed-input lifecycle wording to tutorial or README workflow section | `docs/tutorial.md` or `README.md` | docs hygiene only |
| 3 | add focused no-op/free lifecycle tests only if a clear gap appears | existing test suites | full C quality gate if code changes |
| 4 | add additional solver smoke beyond LU | defer to Day 12 unless Day 9 uncovers a narrow ownership gap | full C quality gate if code changes |

## Non-Goals

Day 8 does not select:

- direct CSR/CSC solver entry APIs;
- adopt/no-copy constructors;
- replacement of `SparseMatrix` as the public coefficient object;
- broad compressed solver parity tests;
- Matrix Market compressed-object publication;
- internal build-path optimization.

## Day 8 Conclusion

The lifecycle model is coherent: compressed input is copied into a normal
caller-owned `SparseMatrix`, then existing solver/factor/handle ownership
rules apply. The biggest remaining gap is narrative coherence, especially the
README wording around diagnostic constructors and a compact user-facing
compressed-input lifecycle path. Day 9 should prefer a narrow docs/header
wording batch unless it finds a concrete test-backed lifecycle gap.
