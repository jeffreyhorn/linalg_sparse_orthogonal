# Sprint 101 Day 3 Solver Entry Path Audit

## Purpose

Day 3 audits solver-facing entry paths for compressed-first adoption costs.
It separates solver-path risks from the storage-constructor issues recorded
on Day 2. This is an audit-only artifact: it does not change APIs, source,
tests, examples, or public documentation.

## Audited Surfaces

| surface | role |
|---|---|
| `include/sparse_lu.h` | public one-shot LU factor, solve, refinement, block solve, condition estimate |
| `include/sparse_lu_csr.h` | public CSR LU working-format conversion, elimination, solve, and one-shot factor-solve |
| `include/sparse_cholesky.h` | public one-shot Cholesky factor and solve |
| `include/sparse_ldlt.h` | public LDL^T factor object, solve, inertia, refinement, condition estimate |
| `include/sparse_analysis.h` | public analyze-once / factor-many direct lifecycle |
| `include/sparse_qr.h` | public QR factor, solve, rank, nullspace, minimum-norm solve |
| `include/sparse_svd.h` | public full and partial SVD surfaces |
| `include/sparse_iterative.h` | public CG, GMRES, MINRES, BiCGSTAB, matrix-free, block, and handle paths |
| `include/sparse_eigs.h` | public symmetric eigensolver and explicit eigensolver handle |
| `include/sparse_ilu.h` | ILU/ILUT factor objects and iterative preconditioner callback |
| `include/sparse_ic.h` | IC(0) factor object and iterative preconditioner callback |
| `README.md` | public workflow chooser and API summary |
| `docs/tutorial.md` | longer user learning path |
| `examples/README.md` | shipped example adoption map |

## Solver Entry Path Map

| family | primary public entry | accepted matrix object | compressed path today | mutation behavior | current classification |
|---|---|---|---|---|---|
| LU one-shot | `sparse_lu_factor`, `sparse_lu_solve`, `sparse_lu_refine` | `SparseMatrix *` | CSR working format is available separately through `sparse_lu_csr.h` | factorization mutates a caller-supplied matrix shell | compatibility-shell with compressed backend support |
| LU CSR working format | `lu_csr_from_sparse`, `lu_csr_eliminate`, `lu_csr_solve`, `lu_csr_factor_solve` | `SparseMatrix *` converted to `LuCsr *` | CSR kernel is explicit, but starts from a matrix shell | working object is caller-owned `LuCsr`; shell input is read-only for factor-solve | compressed working-format specialist surface |
| Cholesky one-shot | `sparse_cholesky_factor`, `sparse_cholesky_factor_opts`, `sparse_cholesky_solve` | `SparseMatrix *` | implementation routes large/eligible cases through CSC internals | factorization mutates the matrix shell into solve-ready factor state | compatibility-shell public API with CSC backend |
| LDL^T | `sparse_ldlt_factor`, `sparse_ldlt_factor_opts`, `sparse_ldlt_solve` | `const SparseMatrix *` into `sparse_ldlt_t` | implementation has CSC/native paths behind public object | factor object owns `L`, `D`, pivots, permutations; input is not the factor owner | compressed backend behind factor object |
| repeated direct | `sparse_analyze`, `sparse_factor_numeric`, `sparse_refactor_numeric`, `sparse_factor_solve` | `const SparseMatrix *`, `sparse_analysis_t`, `sparse_factors_t` | symbolic structure is compressed-column, and numeric paths may dispatch to CSR/CSC | preserves symbolic/permutation setup; refactor refreshes numeric factor contents | strongest public repeated-run direct lifecycle |
| QR | `sparse_qr_factor`, `sparse_qr_factor_opts`, `sparse_qr_solve` | `const SparseMatrix *` into `sparse_qr_t` | no public CSR/CSC QR entry; COLAMD ordering is available | factor object owns QR state; input expected as original matrix view | compatibility-shell solver entry |
| SVD | `sparse_svd_compute`, `sparse_svd_partial`, `sparse_cond`, low-rank helpers | `const SparseMatrix *` | partial SVD is iterative/matvec oriented; no CSR/CSC public constructor entry beyond shell build | result object owns dense outputs; low-rank sparse helper returns `SparseMatrix` | compatibility-shell solver entry |
| CG / GMRES / MINRES | `sparse_solve_*`, `sparse_solve_*_with_handle` | `const SparseMatrix *` or matrix-free callbacks | matrix-free variants bypass `SparseMatrix`; explicit CSR/CSC callbacks are caller-provided if desired | one-shot entries allocate local workspace; handles preserve capacity only | partly compressed-friendly through matrix-free path |
| BiCGSTAB and block iterative | `sparse_solve_bicgstab`, block solve helpers | `const SparseMatrix *` or BiCGSTAB matrix-free callback | BiCGSTAB has matrix-free path; block helpers are shell-based | one-shot compatibility surfaces | bounded compatibility surface |
| symmetric eigensolver | `sparse_eigs_sym`, `sparse_eigs_sym_with_handle` | `const SparseMatrix *`, optional handle | shift-invert composes with LDL^T dispatch; no direct CSR/CSC public entry | result buffers are caller-owned; handle preserves workspace capacity | shell entry with compressed-backed inner paths |
| ILU / ILUT / IC preconditioners | `sparse_ilu_factor`, `sparse_ilut_factor`, `sparse_ic_factor` | `const SparseMatrix *` into factor object | no public CSR/CSC preconditioner constructor | factor object owns preconditioner state; caller passes callback to iterative solvers | compatibility-shell preconditioner entry |

## Compressed-First Readiness by Family

| family | readiness | evidence | remaining adoption cost |
|---|---|---|---|
| repeated direct lifecycle | high | `sparse_analysis_t` stores symbolic compressed-column structure and README teaches analyze/factor/refactor/solve | callers still enter through `SparseMatrix`, so compressed input must first build the public shell |
| Cholesky | high internally | public docs note CSC-backed large-`n` Cholesky and old-factor preservation on refactor | public API is still in-place shell factorization for one-shot use |
| LDL^T | high internally | public eigensolver shift-invert composes with LDL^T and reports CSC backend use | factor object hides backend, but compressed-input callers still start by constructing `SparseMatrix` |
| LU | medium-high | `sparse_lu_csr.h` exposes CSR working format and `lu_csr_factor_solve` | the specialist CSR API is not the main one-shot front door and still converts from `SparseMatrix` |
| iterative solvers | medium | matrix-free CG, GMRES, and BiCGSTAB can wrap caller-native compressed storage without shell construction | no built-in CSR/CSC matvec adapter is promoted as the default compressed-input solve story |
| eigensolver | medium | explicit handle exists, and shift-invert uses LDL^T backend dispatch | all public eigensolver entries take `SparseMatrix`; no compressed public entry or standard CSR/CSC adapter |
| QR | low-medium | public QR factor object is clear and supports COLAMD | no direct compressed input path; tutorial tells users to pass original shell view |
| SVD | low-medium | partial SVD can benefit from iterative/matvec implementation style | public API remains `SparseMatrix`-first, with no compressed-public or matrix-free SVD front door |
| ILU / IC preconditioners | low-medium | factor objects fit iterative callback model | preconditioners require `SparseMatrix` shell construction and copy discipline |

## Ownership and Mutation Ambiguities

| ambiguity | affected surface | risk | likely owner |
|---|---|---|---|
| compressed-input direct solve still has a two-step story | CSR/CSC constructors plus LU/Cholesky/LDL^T | users may assume there is a direct `csr -> solve` front door because backend kernels exist | Day 4 API design and Day 10 docs/examples |
| in-place one-shot direct factorization competes with compressed-first narrative | LU and Cholesky | public API remains shell mutation even when backend work is compressed | Day 4 compatibility behavior table |
| specialist LU CSR APIs are public but lower-level | `sparse_lu_csr.h` | callers may not know whether to use `lu_csr_factor_solve` or normal LU after `sparse_create_from_csr` | Day 4 non-goal and docs decision |
| matrix-free iterative path can be compressed-native but is not packaged as CSR/CSC adapter | CG/GMRES/BiCGSTAB matrix-free | advanced users can avoid shell construction, but the product story does not make that obvious | Day 4 design candidate or defer |
| preconditioner factor objects require shell discipline | ILU/ILUT/IC | iterative examples must still teach copying/original-view preservation before preconditioning | Day 10-11 documentation |
| repeated direct lifecycle has strong reuse rules but no compressed-input example | `example_analysis`, README, tutorial | strongest direct lifecycle does not demonstrate starting from CSR/CSC data | Day 10-12 docs/example candidate |
| SVD and QR retain original-shell requirements | QR/SVD tutorial and headers | compressed-first claim must not imply every solver family accepts compressed public input directly | Day 4 non-goal and Day 13 claim discipline |

## Solver-Path Risks Separated from Storage Risks

| risk type | Day 2 storage owner | Day 3 solver owner |
|---|---|---|
| compressed constructor diagnostics | `sparse_create_from_csr/csc`, `sparse_from_csr/csc` | solvers depend on the constructed `SparseMatrix`; they do not add diagnostics for bad compressed arrays |
| internal shell-insertion build cost | `src/sparse_csr.c` | solver backends start after shell construction, except lower-level working-format conversions |
| public compressed-input learning path | tutorial/examples storage construction | solver examples need to show what happens after compressed construction |
| solve/factor ownership | matrix-shell lifecycle | direct factor objects, iterative/eigs handles, SVD/QR results, preconditioner objects |
| backend capability claims | compressed import/export headers | per-family API and documentation must avoid implying direct CSR/CSC solver entry where none exists |

## Implementation Candidate Ranking

| rank | candidate | user value | compatibility risk | Day 3 recommendation |
|---:|---|---|---|---|
| 1 | document a compressed-input-to-solver workflow using existing constructors plus one-shot and repeated direct paths | high | low | strong Day 10-11 docs/example candidate |
| 2 | clarify that `sparse_create_from_csr/csc` are the public compressed front door before solver entry | high | low | Day 4 design should select wording/API-local refinements |
| 3 | add focused tests proving CSR/CSC-constructed matrices work across representative solver families | high | low-medium | Day 12 regression candidate; choose small LU, Cholesky or LDLT, iterative, QR/SVD/eigs smoke only if bounded |
| 4 | provide or document a CSR/CSC matrix-free adapter pattern for iterative solvers | medium-high | medium | candidate if Day 4 keeps scope docs/example-only; new API may be too broad for Sprint 101 |
| 5 | promote `lu_csr_factor_solve` as an expert working-format path without making it the general product front door | medium | low | docs-only candidate, not a constructor rewrite |
| 6 | add direct CSR/CSC solver entry points for QR/SVD/eigs/preconditioners | medium | high | defer; this would exceed Sprint 101's bounded front-door scope |
| 7 | replace public `SparseMatrix` solver entry model | high theoretical | very high | explicit non-goal for Sprint 101 |

## Sprint 102 Direct-Solver Oracle Dependencies

Sprint 102 can use Sprint 101 outputs most directly if Sprint 101 leaves:

- one accepted compressed-input construction workflow for solver tests;
- representative CSR/CSC-built matrices that exercise direct solver families;
- a clear distinction between public one-shot direct APIs and the
  analyze-once / factor-many lifecycle;
- an explicit non-claim that compressed constructors do not prove broad
  solver parity by themselves;
- any selected Day 12 regression tests named as direct-solver oracle seeds.

## Day 3 Conclusion

The solver surface is more compressed-ready internally than it is
compressed-first publicly. Cholesky, LDL^T, LU CSR, shift-invert eigensolver,
and repeated direct analysis all have meaningful compressed or compressed-like
backend structure, but almost every public solver family still starts from
`SparseMatrix`. Sprint 101 should therefore focus Day 4 on a bounded front
door: make CSR/CSC construction and solver entry feel like one coherent
workflow, preserve the mutable shell as the compatibility object, and avoid
claiming direct CSR/CSC solver parity across families that still have
`SparseMatrix`-first APIs.
