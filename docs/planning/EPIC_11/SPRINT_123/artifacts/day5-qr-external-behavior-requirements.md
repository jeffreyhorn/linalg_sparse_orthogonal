# Sprint 123 Day 5 QR External Behavior Requirements

## Purpose

Day 5 defines the QR external behavior evidence requirements before Sprint 123
decides whether to implement or explicitly defer compatible, rank-deficient,
underdetermined/minimum-norm, and Q/economy QR evidence. The goal is to keep
each candidate behavior-specific and avoid turning one small Python
standard-library reference lane into broad QR, LAPACK, NumPy, SciPy, or
least-squares parity.

This is a requirements artifact only. No C source, header, Python helper,
build, CMake, CTest, workflow, public API, or public wording changes are made
by Day 5.

## Inputs Reviewed

| Input | Relevant Content |
| --- | --- |
| Sprint 123 Plan Day 5 | Requires QR external and deterministic evidence inventory, behavior separation, fixture requirements, tolerance/skip/failure rules, minimum-norm ownership boundaries, and a Day 6 checklist. |
| Sprint 121 Day 3 QR audit | Defines current QR factorization, solve, rank, nullspace, minimum-norm, refinement, economy, sparse-mode, and reordering proof owners. |
| Sprint 121 Day 8 rank-deficient expansion | Defines dependent-row and diagonal-threshold QR fixtures and duplicate fences for deterministic rank evidence. |
| Sprint 121 Day 9 least-squares expansion | Defines compatible tall, incompatible tall, underdetermined minimum-norm, and SVD pseudoinverse deterministic evidence. |
| Sprint 122 Day 5 QR lane requirements | Defines the original QR external-lane selection criteria and external parity non-claims. |
| Sprint 122 Day 6 QR lane design | Records completed `qr_overdetermined_incompatible_4x2` external least-squares evidence. |
| `tests/qr_external_dense_reference.py` | Current QR external-reference helper with one bounded incompatible overdetermined least-squares fixture. |
| `tests/test_qr_solve.c` | Current external QR test owner plus compatible, incompatible, rank-deficient, minimum-norm, SuiteSparse, QR-vs-LU, and synthetic solve owners. |
| `tests/test_qr.c` | Current QR factorization, rank, nullspace, Q application, economy, sparse-mode, and reordering owner. |

## Current QR External Fixture Inventory

| Fixture Key | Owner | Behavior Class | Compared Quantity | Current Trust Boundary | Duplicate Fence |
| --- | --- | --- | --- | --- | --- |
| `qr_overdetermined_incompatible_4x2` | `tests/qr_external_dense_reference.py`, `tests/test_qr_solve.c` | Small dense tall full-column-rank incompatible least-squares | Solution vector and residual norm | Python standard-library normal-equation reference for one tiny fixed fixture | Do not add another incompatible 4x2 external least-squares fixture unless it adds a new behavior class. |

## Deterministic QR Coverage Classes

| Coverage Class | Current Evidence Owner | External Evidence Implication |
| --- | --- | --- |
| Basic factorization and reconstruction | `tests/test_qr.c` | External lanes should not duplicate Householder, permutation, or reconstruction smoke checks. |
| Q application and orthogonality | `tests/test_qr.c` | Basis comparisons require sign, shape, and orientation policy before externalization. |
| Compatible tall least-squares | `tests/test_qr_solve.c` | Candidate external lane if a reference fixture adds independent solution/residual evidence beyond generated-RHS exactness. |
| Incompatible tall least-squares | `tests/test_qr_solve.c` plus `qr_overdetermined_incompatible_4x2` | Already has one bounded external lane; new work should not repeat this shape. |
| Rank-deficient QR and nullspace | `tests/test_qr.c`, `tests/test_qr_solve.c` | Candidate only if rank threshold, residual, and minimum-norm behavior are separated. |
| Underdetermined minimum-norm | `tests/test_qr_solve.c`, historical `tests/test_colamd.c`, SVD pseudoinverse cross-checks | Candidate only if ownership remains behavior-specific and does not hide COLAMD/SVD-pinv/refinement semantics. |
| Economy mode | `tests/test_qr.c` | External evidence must define thin-Q shape and avoid comparing non-unique basis orientation. |
| Sparse mode | `tests/test_qr.c` | External dense references should not claim dense-vs-sparse backend parity or performance. |
| Refinement | `tests/test_qr.c`, `tests/test_qr_solve.c` | Keep separate from external QR behavior unless residual semantics and owner are explicit. |
| Reordering and fill | `tests/test_qr.c`, `tests/test_colamd.c` | Not a Day 5 external behavior candidate; keep as QR-adjacent evidence. |

## QR External Candidate Table

| Candidate Class | Example Fixture Key | Adds New Evidence? | Risk | Day 5 Disposition |
| --- | --- | --- | --- | --- |
| Compatible tall least-squares | `qr_overdetermined_compatible_5x3` | Moderate. Adds external reference for a generated-RHS exact tall system and expected near-zero residual. | Low to moderate; duplicate risk is high unless fixture shape and residual semantics differ from current deterministic tests. | Candidate for Day 6 only if paired with an explicit duplicate fence and output protocol. |
| Rank-deficient least-squares | `qr_rankdef_duplicate_5x4_ls` | High. Adds external behavior for duplicate-column least-squares residual and rank-deficient solve handling. | High; overlaps rank thresholds, nullspace, and minimum-norm ownership. | Candidate for Day 6 only if Day 6 keeps rank and minimum-norm claims out of scope. |
| Underdetermined minimum-norm | `qr_underdetermined_minnorm_3x5` | High. Adds independent norm/residual evidence for `sparse_qr_solve_minnorm`. | High; overlaps QR, COLAMD, SVD pseudoinverse, refinement, fallback, and SuiteSparse ownership. | Defer to Day 7 minimum-norm decision unless ownership can be named precisely. |
| Q/economy shape evidence | `qr_economy_q_shape_5x3` | Moderate. Could externally validate thin-Q shape or projection behavior. | High; Q bases are sign/orientation-dependent, and economy/full shape semantics differ. | Defer to Day 7 Q/economy decision. |
| Square QR solve | `qr_square_3x3_external_solve` | Low. Duplicates exact square solve and QR-vs-LU checks. | Moderate claim risk around direct-solver parity. | Reject for Sprint 123. |
| SuiteSparse external QR | `qr_suitesparse_external_ls` | Broad but tempting. | Very high; optional corpus, runtime, platform, and broad-corpus interpretation risk. | Reject for Sprint 123 Day 6-8. |
| Sparse-mode external parity | `qr_sparse_mode_external_dense_compare` | Low for external oracle value; deterministic dense-vs-sparse checks already exist. | High backend/performance claim risk. | Reject for Sprint 123 external behavior evidence. |

## Behavior-Specific Fixture Requirements

| Behavior | Required Fixture Fields | Required Metrics | Explicit Non-Metrics |
| --- | --- | --- | --- |
| Compatible tall least-squares | Matrix, RHS, known solution, full-column-rank statement, residual expectation, output count | solution max difference, reported residual absolute difference, optional true residual confirmation | no rank-threshold, Q-basis, minimum-norm, or direct-solver parity claim |
| Rank-deficient least-squares | Matrix, RHS, structural rank model, expected rank if asserted, residual metric, explicit minnorm exclusion or inclusion | solve status, residual norm, optional rank value if threshold is pinned | no global rank policy, nullspace basis parity, or minnorm claim unless separately designed |
| Underdetermined minimum-norm | Matrix, RHS, expected minimum-norm solution, norm comparator, residual metric, owner of fallback behavior | solution max difference, `||A*x-b||`, solution norm, optional comparison against a named alternate solution | no broad global optimality, COLAMD parity, or pseudoinverse parity unless separately named |
| Q/economy evidence | Matrix, mode, expected Q shape, sign/orientation policy, projection or orthogonality metric | shape, projection residual, orthogonality bound | no column-by-column basis equality unless sign and subspace policy is explicit |

## Basis and Ownership Rules

| Topic | Rule |
| --- | --- |
| Q basis orientation | Do not compare raw Q columns externally unless sign and repeated/degenerate basis rules are written first. |
| Economy shape | Any economy evidence must state whether Q is full `m x m`, thin `m x n`, or otherwise mode-specific. |
| Rank-deficient solves | Rank, residual, and minimum-norm behavior must be asserted separately; a residual-only fixture must not imply rank-policy proof. |
| Minimum-norm ownership | Keep QR, COLAMD, SVD pseudoinverse, refinement, fallback, and SuiteSparse minimum-norm owners visible at the test or artifact boundary. |
| Helper migration | Do not move minimum-norm helpers into generic QR helpers before Day 11 decides helper ownership. |
| Reordering | QR external behavior evidence must not hide AMD/COLAMD/fill ownership. |

## Tolerance, Skip, and Failure Rules

| Rule | Requirement |
| --- | --- |
| Reference implementation | Python standard library only; no NumPy, SciPy, LAPACK, BLAS, SuiteSparse, or external package dependency. |
| Fixture size | Prefer tiny fixed fixtures with dimensions no larger than roughly 6 in either direction unless a later day justifies otherwise. |
| Solution tolerance | Default absolute max difference target is `1e-8` for accepted external solution-vector comparisons. |
| Residual tolerance | Compare residual norms separately from solution values; default absolute difference target is `1e-8`. |
| Rank tolerance | If rank is asserted, state the threshold and whether it is structural or numerical. |
| Missing Python | Preserve the existing helper skip behavior. |
| Windows behavior | Preserve explicit Windows skip unless a future platform-support owner promotes it. |
| Helper `ERROR` output | Treat as test failure, not skip. |
| Failure diagnostics | Identify fixture key, reference status, product QR status, solution max difference, residual difference, and whether the mismatch is solve, residual, rank, basis, or protocol-related. |

## Day 6 Decision Checklist

Day 6 should decide compatible and rank-deficient QR evidence using this order:

1. Treat `qr_overdetermined_incompatible_4x2` as completed and fenced.
2. Decide whether a compatible tall external fixture adds enough evidence beyond
   deterministic generated-RHS tests.
3. Decide whether a rank-deficient residual-only fixture can be accepted without
   claiming minimum-norm or broad rank-threshold behavior.
4. For any accepted fixture, define key, matrix, RHS, expected outputs,
   tolerances, skip behavior, affected test owners, and failure diagnostics.
5. For any deferred fixture, name the future owner and promotion gate.

## Non-Claim Register

Day 5 does not claim:

- LAPACK, NumPy, SciPy, BLAS, SuiteSparse, PETSc, Trilinos, Eigen, or broad
  external dense-library parity;
- broad QR factorization or least-squares parity;
- direct-solver parity;
- rank-deficient QR external parity;
- underdetermined or minimum-norm global optimality;
- Q-basis, Q-sign, Q-orientation, economy-mode, sparse-mode, reorder, or
  backend parity;
- package, ABI, platform, public API, CMake, Makefile, CI, or CTest expansion;
- performance, scalability, memory behavior, or state-of-the-art behavior.

## Validation Notes

Day 5 changed documentation only. Required validation is:

1. `git diff --check`
2. Focused trailing-whitespace scan over `docs/planning/EPIC_11/SPRINT_123`

The branch already contains Day 4 `.c` and Python helper changes; Day 4 ran
the full `make format && make lint && make test` gate after those changes.

## Completion Criteria Status

| Criterion | Status | Evidence |
| --- | --- | --- |
| QR candidates are behavior-specific. | Complete | See candidate table and behavior-specific fixture requirements. |
| Basis-dependent evidence is not conflated with solve residual evidence. | Complete | Q/economy evidence is deferred to Day 7 with basis and shape rules. |
| QR external parity remains a non-claim unless separately earned. | Complete | See non-claim register and trust-boundary rules. |
