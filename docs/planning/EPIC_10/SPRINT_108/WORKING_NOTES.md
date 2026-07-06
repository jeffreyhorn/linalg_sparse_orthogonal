# Sprint 108 Working Notes

## Sprint Goal

Sprint 108 converts Sprint 107's residual deferred debt into bounded
proof-owner cleanup and source-boundary planning. The sprint must not duplicate
completed Sprint 107 helper extractions, and it must not turn deferred source
risks into opportunistic public API, install-header, helper-target, source-list,
or reviewed test-count changes.

## Starting Constraints

- Keep public API and install-header surfaces unchanged unless a later explicit
  review approves otherwise.
- Do not add a compiled test helper target.
- Do not change reviewed test counts unintentionally.
- Do not extract implementation source without a source-list, Make/CMake, and
  focused validation plan.
- Keep proof assertions visible at test call sites when extracting setup
  helpers.
- Treat `src/sparse_matrix.c` as central public behavior territory until a
  public-behavior review and private-header dependency plan exists.

## Sprint 107 Completed Work Excluded From Sprint 108

The following Sprint 107 work is not unresolved Sprint 108 debt:

- Sprint 106 deferred debt intake and residual owner re-rank.
- LDLT CSC row-adjacency helper extraction.
- QR small 4x3 fixture-builder cleanup.
- Iterative matrix-free fixture cleanup.
- SVD diagonal and rank-1 fixture cleanup.
- Eigensolver source boundary and explicit no-split deferral.
- Matrix shell deferral contract.
- Sprint 107 final validation and drift checks.

## Sprint 108 Owner Inventory

| Owner | Carry-Forward Scope | Sprint 108 Guardrail |
|---|---|---|
| `tests/test_ldlt_csc.c` | Remaining broad direct-solver proof and oracle logic. | Extract at most one additional named proof helper after a fresh boundary. |
| `tests/test_qr.c` | Generated fixtures, tall/economy builders, diagonal/singleton setup, and SuiteSparse exact-RHS setup. | Preserve visible rank, solve, residual, refinement, and reconstruction assertions. |
| `tests/test_iterative.c` | Convergence-sensitive setup and repeated solver fixture material. | Do not hide solver options, restart values, preconditioners, convergence results, or direct comparisons. |
| `tests/test_svd.c` | Rank, oracle, reconstruction, pseudoinverse, low-rank, partial-SVD, and condition-number proof logic. | Create a dedicated validation lane before moving any remaining proof helper family. |
| `src/sparse_eigs.c` | Source-owner risk around dense Jacobi feasibility and grow-m refinement boundaries. | Prepare build-system/source-list and cross-backend spectral validation before any future split. |
| `src/sparse_matrix.c` | Central public behavior, private-header dependency, allocation/state ownership, and compatibility territory. | Perform public-behavior review and private dependency mapping before any future shell extraction. |

## Day-Level Ownership

| Day | Planned Focus | Project Plan Item |
|---:|---|---|
| 1 | Carry-forward intake and validation expectations. | Item 1, Item 8 |
| 2 | Residual proof-owner boundary refresh. | Item 1 |
| 3 | LDLT CSC oracle boundary. | Item 2 |
| 4 | LDLT CSC helper follow-through. | Item 2 |
| 5 | QR residual fixture boundary. | Item 3 |
| 6 | QR fixture follow-through. | Item 3 |
| 7 | Iterative convergence boundary. | Item 4 |
| 8 | Iterative convergence cleanup. | Item 4 |
| 9 | SVD validation lane boundary. | Item 5 |
| 10 | SVD oracle/reconstruction cleanup. | Item 5 |
| 11 | Eigensolver source feasibility boundary. | Item 6 |
| 12 | Eigensolver feasibility closeout. | Item 6 |
| 13 | Matrix shell public-behavior review. | Item 7 |
| 14 | Validation, metrics, residual queue, and closeout. | Item 8 |

## Validation Expectations

| Touched Surface | Required Checks |
|---|---|
| Documentation only | `git diff --check`; trailing-whitespace scan over touched docs. |
| Test `.c` files | Focused touched test binary or suite; `make format && make lint && make test`; `git diff --check`. |
| Test headers | Focused impacted tests; `make format && make lint && make test`; `git diff --check`. |
| Implementation `.c` or `.h` files | Focused family tests; source-list/build-system checks if membership changes; `make format && make lint && make test`; `git diff --check`. |
| Makefile or CMake membership | Source-list parity checks, Make and CMake build/test surfaces, and full quality gate. |
| Public headers or install metadata | Public API/install/export review, downstream consumer or package checks, and full quality gate. |
| Mixed docs and code | Apply the strongest requirement for any touched file type. |

## Day 1 Notes

- Created Sprint 108 working notes and artifact directory.
- Converted Sprint 107 residual deferred debt into an explicit Sprint 108 owner
  inventory.
- Recorded completed Sprint 107 extractions as exclusions to prevent duplicate
  work.
- Established validation expectations before boundary or cleanup work begins.

## Day 2 Notes

- Captured live residual owner metrics after the Sprint 107 merge.
- Ranked remaining proof-owner cleanup by failure-localization value,
  reviewability, validation cost, and dependency order.
- Confirmed Sprint 108 should continue boundary-first:
  1. LDLT CSC oracle helper follow-through.
  2. QR fixture follow-through.
  3. Iterative convergence-sensitive cleanup.
  4. SVD validation-lane cleanup.
  5. Eigensolver source feasibility planning.
  6. Matrix shell public-behavior review.
- Reconfirmed completed Sprint 107 work remains excluded from Sprint 108:
  row-adjacency helper, QR 4x3 builders, iterative matrix-free fixture reuse,
  SVD diagonal/rank-1 builders, eigensolver no-split record, and matrix shell
  deferral contract.

### Day 2 Live Metrics

| Owner | Lines | Static Functions | Tests | Assertions | Sparse Creates | Sparse Inserts | Day 2 Disposition |
|---|---:|---:|---:|---:|---:|---:|---|
| `tests/test_ldlt_csc.c` | 3,887 | 132 | 100 | 489 | 56 | 170 | Highest-value narrow oracle helper candidate. |
| `tests/test_qr.c` | 3,213 | 86 | 73 | 337 | 56 | 224 | Fixture follow-through after LDLT boundary. |
| `tests/test_iterative.c` | 2,828 | 89 | 77 | 316 | 26 | 103 | Cleanup only after convergence guardrails are explicit. |
| `tests/test_svd.c` | 2,897 | 81 | 75 | 363 | 55 | 160 | Requires dedicated validation lane before helper movement. |
| `src/sparse_eigs.c` | 1,538 | n/a | n/a | n/a | n/a | n/a | Feasibility plan only before any future source split. |
| `src/sparse_matrix.c` | 1,359 | n/a | n/a | n/a | n/a | n/a | Public-behavior review before any future shell split. |

## Day 3 Notes

- Inspected remaining LDLT CSC factor-state, residual, external dense-reference,
  and row-adjacency proof patterns.
- Rejected broad movement of `ldlt_csc_factor_state_matches` and
  `assert_ldlt_external_dense_reference` because they already encode large
  oracle semantics and allocation/skip behavior.
- Rejected additional row-adjacency movement because Sprint 107 already
  completed that extraction path.
- Selected one narrow Day 4 candidate:
  `assert_s20_solve_residual_below`, a local assertion helper around existing
  `s20_solve_residual`.
- The selected candidate keeps KKT/direct CSC proof intent visible at call
  sites while localizing the repeated residual-threshold assertion and failure
  message.

## Day 4 Notes

- Implemented the Day 3 approved local helper
  `assert_s20_solve_residual_below` in `tests/test_ldlt_csc.c`.
- Updated only the four approved Sprint 20 supernodal with-analysis residual
  call sites:
  - `test_s20_supernodal_with_analysis_kkt_5x5`
  - `test_s20_supernodal_with_analysis_kkt_10x10`
  - `test_s20_supernodal_with_analysis_random_indefinite_30x30`
  - `test_s20_supernodal_heuristic_vs_with_analysis_residuals`
- Preserved visible fixture construction, factorization path,
  `ldlt_csc_validate`, and explicit `1e-10` tolerance at call sites.
- Left external dense-reference, factor-state equality, row-adjacency, and
  unrelated LDLT solve/dispatch tests unchanged.
- Remaining LDLT CSC residual debt is deferred to a future dedicated oracle
  review rather than expanded inside Sprint 108 Day 4.

### Day 4 LDLT CSC Metrics

| Metric | Before Day 4 | After Day 4 |
|---|---:|---:|
| `tests/test_ldlt_csc.c` lines | 3,887 | 3,896 |
| Local `s20_solve_residual` assertion helper | no | yes |
| Approved residual call sites using labeled helper | 0 | 4 |
| New compiled helper target | 0 | 0 |
| Public headers touched | 0 | 0 |

## Day 5 Notes

- Inventoried remaining QR generated fixtures, tall/economy builders,
  diagonal/singleton setup, SuiteSparse exact-RHS setup, reconstruction checks,
  residual assertions, sparse-mode parity checks, and refinement setup.
- Confirmed Sprint 107's completed QR 4x3 builders remain excluded:
  `make_qr_small_banded_4x3`, `make_qr_duplicate_column_4x3`, and
  `make_qr_near_duplicate_4x3`.
- Rejected broad movement of SuiteSparse exact-RHS setup because allocation,
  Matrix Market loading, factorization, solve, residual, and comparison
  behavior are intentionally adjacent in those tests.
- Rejected moving rank, reconstruction, residual, refinement, and dense/sparse
  parity assertions behind a new helper.
- Selected one bounded Day 6 candidate: a local tall diagonal-dominant fixture
  builder for the repeated setup in:
  - `test_economy_solve_tall`
  - `test_sparse_mode_tall`
  - `test_qr_refine_overdetermined`
- The Day 6 helper must build only the matrix. RHS vectors, QR options,
  factorization calls, solve/refinement calls, rank checks, residual checks,
  reconstruction checks, and dense/sparse parity assertions must remain visible
  at call sites.

### Day 5 QR Metrics

| Metric | Current Value |
|---|---:|
| `tests/test_qr.c` lines | 3,213 |
| Existing QR fixture builders | 3 |
| Existing QR residual/reconstruction helpers | 4 |
| Approved Day 6 tall diagonal-dominant call sites | 3 |
| New helper target approved | 0 |

## Day 6 Notes

- Implemented the Day 5 approved local fixture builder
  `make_qr_tall_diagonal_dominant` in `tests/test_qr.c`.
- Updated only the three approved QR call sites:
  - `test_economy_solve_tall`
  - `test_sparse_mode_tall`
  - `test_qr_refine_overdetermined`
- Preserved visible dimensions, RHS setup, QR option values, factorization
  calls, solve/refinement calls, economy checks, residual comparisons, and
  dense/sparse parity assertions at call sites.
- Left SuiteSparse exact-RHS setup, residual helpers, reconstruction helpers,
  diagonal/singleton setup, public headers, implementation sources, build
  membership, and CTest registration unchanged.
- Remaining QR cleanup debt is deferred to future boundary work rather than
  expanded inside Day 6.

### Day 6 QR Metrics

| Metric | Before Day 6 | After Day 6 |
|---|---:|---:|
| `tests/test_qr.c` lines | 3,213 | 3,210 |
| Local tall diagonal-dominant fixture builder | no | yes |
| Approved call sites using the new builder | 0 | 3 |
| New compiled helper target | 0 | 0 |
| Public headers touched | 0 | 0 |

## Day 7 Notes

- Inventoried `tests/test_iterative.c` repeated matrix builders, RHS setup,
  solver options, restart values, preconditioner setup, result structures,
  convergence assertions, residual checks, and direct comparisons.
- Confirmed Sprint 107's matrix-free tridiagonal and sequential-RHS helper work
  remains excluded from Sprint 108 follow-through.
- Rejected broad movement of solver options, restarts, convergence flags,
  residual assertions, direct CG/GMRES/LU/Cholesky comparisons, SuiteSparse
  corpus setup, and matrix-free comparison assertions.
- Selected one bounded Day 8 candidate: a local helper that builds the
  repeated poorly scaled unsymmetric tridiagonal matrix and matching
  diagonal-inverse vector for the diagonal-preconditioned GMRES tests.
- Approved Day 8 call sites:
  - `test_gmres_right_precond_diag`
  - `test_gmres_diagonal_preconditioner`
- The Day 8 helper must not hide `sparse_gmres_opts_t` values,
  `precond_side`, restart values, convergence assertions, reported/true
  residual comparisons, or left-vs-right preconditioner semantics.

### Day 7 Iterative Metrics

| Metric | Current Value |
|---|---:|
| `tests/test_iterative.c` lines | 2,828 |
| Existing matrix builders | 4 |
| Existing matrix-free helper callbacks/builders | 4 |
| Approved Day 8 diagonal-preconditioner fixture call sites | 2 |
| New helper target approved | 0 |

## Day 8 Notes

- Implemented the Day 7 approved local fixture builder
  `build_scaled_unsym_tridiag_with_diag_inv` in `tests/test_iterative.c`.
- Updated only the two approved GMRES diagonal-preconditioner call sites:
  - `test_gmres_right_precond_diag`
  - `test_gmres_diagonal_preconditioner`
- Preserved visible `idx_t n`, RHS setup, `diag_precond_t` setup,
  `sparse_gmres_opts_t` literals, restart/tolerance values, `precond_side`,
  solve calls, convergence assertions, iteration reporting, and reported-vs-true
  residual checks at call sites.
- Added fixture allocation/insert failure cleanup inside the new helper without
  changing public headers, implementation sources, build membership, helper
  targets, or CTest registration.
- Left broader CG convergence lanes, GMRES restart lanes, SuiteSparse corpus
  lanes, matrix-free comparison lanes, and direct solver comparison assertions
  unchanged.

### Day 8 Iterative Metrics

| Metric | Before Day 8 | After Day 8 |
|---|---:|---:|
| `tests/test_iterative.c` lines | 2,828 | 2,849 |
| Local scaled unsymmetric tridiagonal + diagonal-inverse builder | no | yes |
| Approved diagonal-preconditioner call sites using the new builder | 0 | 2 |
| New compiled helper target | 0 | 0 |
| Public headers touched | 0 | 0 |

## Day 9 Notes

- Inventoried remaining SVD rank, oracle, reconstruction, pseudoinverse,
  low-rank, partial-SVD, and condition-number proof logic in
  `tests/test_svd.c`.
- Confirmed Sprint 107's completed diagonal and rank-1 fixture builders remain
  excluded from Sprint 108 follow-through:
  - `make_svd_diag_matrix`
  - `make_svd_rank1_row_progression`
- Rejected broad movement of singular-value assertions, rank-threshold checks,
  reconstruction loops, orthogonality checks, Moore-Penrose identities,
  low-rank Frobenius comparisons, partial-SVD vector checks, and condition
  estimates because those assertions are the proof.
- Selected one bounded Day 10 candidate: a local deterministic dense `16x8`
  full-SVD fixture builder for the repeated setup in:
  - `test_svd_full_u_v_orthonormality`
  - `test_svd_full_u_v_economy_mode_unchanged`
  - `test_svd_full_u_v_reconstruction`
- The Day 10 helper may build only the repeated matrix fixture. Full/economy
  options, SVD calls, dimension assertions, singular-triplet parity,
  orthogonality checks, reconstruction loops, residual thresholds, and logging
  must remain visible at call sites.

### Day 9 SVD Metrics

| Metric | Current Value |
|---|---:|
| `tests/test_svd.c` lines | 2,897 |
| Static functions | 81 |
| Registered tests | 98 |
| Assertion/proof macro references | 461 |
| Sparse creates | 55 |
| Sparse inserts | 160 |
| Approved Day 10 full-SVD fixture call sites | 3 |
| New helper target approved | 0 |

## Day 10 Notes

- Implemented the Day 9 approved local fixture builder
  `make_svd_full_uv_fixture_16x8` in `tests/test_svd.c`.
- Updated only the three approved full-SVD fixture call sites:
  - `test_svd_full_u_v_orthonormality`
  - `test_svd_full_u_v_economy_mode_unchanged`
  - `test_svd_full_u_v_reconstruction`
- Preserved visible dimensions, economy/full option literals, SVD calls,
  dimension assertions, U/Vt assertions, singular-triplet parity loops,
  U/Vt orthogonality loops, reconstruction loops, residual thresholds, and
  diagnostic logging at call sites.
- Added fixture insert failure handling through the existing
  `svd_insert_or_free` helper without changing public headers,
  implementation sources, build membership, helper targets, or CTest
  registration.
- Left rank-threshold checks, pseudoinverse proofs, low-rank dense/sparse
  comparisons, partial-SVD vector/corpus checks, and condition-number behavior
  unchanged.

### Day 10 SVD Metrics

| Metric | Before Day 10 | After Day 10 |
|---|---:|---:|
| `tests/test_svd.c` lines | 2,897 | 2,896 |
| Local deterministic full-SVD `16x8` fixture builder | no | yes |
| Approved full-SVD call sites using the new builder | 0 | 3 |
| New compiled helper target | 0 | 0 |
| Public headers touched | 0 | 0 |

## Day 11 Notes

- Inspected the current eigensolver source ownership surface:
  - `src/sparse_eigs.c`
  - `src/sparse_eigs_internal.h`
  - `src/sparse_eigs_workspace_internal.c`
  - `src/sparse_eigs_workspace_internal.h`
  - `src/sparse_eigs_thick_restart.c`
  - `src/sparse_eigs_lobpcg.c`
  - `include/sparse_eigs.h`
- Reviewed existing build membership and source-list parity owners:
  - `Makefile`
  - `CMakeLists.txt`
  - `build-metadata/library_sources.txt`
  - `scripts/check_library_sources.py`
- Re-read Sprint 107's eigensolver source-boundary and no-split deferral
  artifacts, plus Sprint 103 spectral comparison artifacts for LOBPCG,
  thick-restart, grow-m, shift-invert, SVD cross-check, and Sprint 29
  integration evidence.
- Confirmed existing split owners already cover the largest backend bodies:
  workspace storage, thick-restart Lanczos, and LOBPCG.
- Identified `s21_dense_sym_jacobi` as the least risky future source seam, but
  still requiring Make/CMake/manifest parity and focused thick-restart plus
  LOBPCG validation before movement.
- Identified grow-m shift-invert/refinement as a higher-risk future seam
  because it crosses public behavior, LDLT setup, residual reporting,
  `NEAREST_SIGMA`, and Sprint 29 refinement evidence.
- Performed no source, header, Makefile, CMake, manifest, public API,
  install-header, helper-target, or CTest registration change for Day 11.

### Day 11 Eigensolver Metrics

| Owner | Lines | Day 11 Disposition |
|---|---:|---|
| `src/sparse_eigs.c` | 1,538 | Keep intact; future seam planning only. |
| `src/sparse_eigs_internal.h` | 631 | Keep declarations unchanged. |
| `src/sparse_eigs_workspace_internal.c` | 267 | Existing split owner; no change. |
| `src/sparse_eigs_workspace_internal.h` | 82 | Existing private workspace contract; no change. |
| `src/sparse_eigs_thick_restart.c` | 915 | Existing backend split owner; validation target for future dense helper movement. |
| `src/sparse_eigs_lobpcg.c` | 401 | Existing backend split owner; validation target for future dense helper movement. |
| `include/sparse_eigs.h` | 651 | Public API/install surface; no movement approved. |

## Day 12 Notes

- Closed the eigensolver feasibility workstream with an explicit
  documentation-only handoff instead of landing a source split.
- Confirmed Sprint 108's only Day 12 preparatory change is the closeout
  artifact:
  `docs/planning/EPIC_10/SPRINT_108/artifacts/day12-eigensolver-feasibility-closeout.md`.
- Reaffirmed the future first candidate as a private dense spectral helper
  source for `s21_dense_sym_jacobi`, with no public header movement and no
  new CTest registration.
- Recorded the required source membership updates for any future extraction:
  `Makefile`, `CMakeLists.txt`, and `build-metadata/library_sources.txt`,
  followed by `make source-list-check`.
- Captured focused future validation requirements for eigensolver movement:
  `test_eigs`, `test_eigs_thick_restart`, `test_eigs_lobpcg`, and
  `test_sprint29_integration`.
- Preserved explicit non-goals:
  - no `src/sparse_eigs.c` split in Sprint 108;
  - no movement of grow-m Lanczos, shift-invert/refinement, dispatch,
    handle/workspace glue, or shared Lanczos kernels;
  - no public API, install-header, Makefile, CMake, manifest, helper-target,
    or CTest registration change.

### Day 12 Eigensolver Closeout Queue

| Queue Item | Future Gate | Sprint 108 Disposition |
|---|---|---|
| Dense Jacobi helper owner | Move only `s21_dense_sym_jacobi`; update Make/CMake/manifest; run source-list and cross-backend spectral validation. | Future candidate; no Day 12 source change. |
| Grow-m refinement audit | Strengthen residual/refinement evidence before moving shift-invert or Rayleigh-quotient iteration helpers. | Deferred as behavior-sensitive. |
| Dispatch and defaults boundary | Preserve backend selection, options normalization, and public result semantics until reviewed behavior tests justify movement. | Deferred as public-behavior owner. |
| Handle/workspace glue | Keep storage split separate from public handle preparation and workspace-backed solve behavior. | Deferred; workspace storage already split. |
| Shared Lanczos kernels | Validate thick-restart, LOBPCG, grow-m, and focused internal tests together before any shared-kernel owner appears. | Deferred as cross-backend numerical surface. |

## Day 13 Notes

- Reviewed `src/sparse_matrix.c`, `include/sparse_matrix.h`,
  `src/sparse_matrix_internal.h`, and
  `src/sparse_matrix_state_internal.h` as the central matrix-shell ownership
  surface.
- Confirmed `src/sparse_matrix.c` still owns public lifecycle, mutation,
  logical/physical access, permutation compatibility, factor-state reset,
  arithmetic, matvec/block matvec, Matrix Market I/O, memory/norm reporting,
  and print/debug helpers.
- Mapped private matrix-header dependencies across direct solvers,
  iterative/spectral solvers, graph/reorder code, compressed constructors,
  dense/SVD/bidiag paths, and focused tests.
- Confirmed any future matrix-shell source split must update Makefile,
  CMake, and `build-metadata/library_sources.txt`, then run
  `make source-list-check`.
- Identified public behavior guardrails for future movement:
  `test_sparse_matrix`, `test_sparse_io`, `test_sparse_arith`,
  `test_matmul`, `test_csr`, `test_reorder`, `test_sparse_lu`,
  `test_cholesky`, and solver smoke tests when matvec, permutation, or
  factor compatibility changes.
- Performed no source, header, public API, install-header, Makefile, CMake,
  manifest, helper-target, or CTest registration change for Day 13.

### Day 13 Matrix Shell Metrics

| Owner | Lines | Day 13 Disposition |
|---|---:|---|
| `src/sparse_matrix.c` | 1,359 | Keep intact; public-behavior review only. |
| `include/sparse_matrix.h` | 614 | Public API/install-header contract; no movement. |
| `src/sparse_matrix_internal.h` | 251 | Broad private struct/state dependency surface; no split approved. |
| `src/sparse_matrix_state_internal.h` | 58 | Factor/permutation compatibility helper surface; no split approved. |
| `tests/test_sparse_matrix.c` | 1,296 | Primary behavior guardrail for lifecycle, mutation, access, copy, matvec, perms, memory, and get semantics. |
| `tests/test_sparse_io.c` | 511 | Matrix Market I/O and errno guardrail. |
| `tests/test_csr.c` | 704 | Compressed constructor and public matrix-shell entry guardrail. |

## Day 14 Notes

- Closed Sprint 108 by validating the actual touched-file set:
  - code/test files changed:
    - `tests/test_ldlt_csc.c`
    - `tests/test_qr.c`
    - `tests/test_iterative.c`
    - `tests/test_svd.c`
  - planning files added under `docs/planning/EPIC_10/SPRINT_108/`
- Confirmed no Sprint 108 change touched:
  - public headers;
  - installed-header declarations;
  - implementation sources under `src/`;
  - `Makefile`;
  - `CMakeLists.txt`;
  - `build-metadata/library_sources.txt`;
  - helper targets;
  - CTest registration.
- Captured final proof-owner metrics and residual queue in
  `docs/planning/EPIC_10/SPRINT_108/artifacts/day14-validation-metrics-closeout.md`.
- Required quality gate for the branch is:
  - `make format`
  - `make lint`
  - `make test`
  - `git diff --check`
  - trailing whitespace scan for Sprint 108 planning artifacts
- Day 14 validation result:
  - `make format && make lint && make test` passed;
  - `git diff --check` passed;
  - `rg -n "[ \t]+$" docs/planning/EPIC_10/SPRINT_108` reported no trailing
    whitespace;
  - post-format status still shows only the expected tracked test-file
    changes plus untracked Sprint 108 planning artifacts.

### Day 14 Final Metrics

| Owner | HEAD Lines | Final Lines | Net Line Delta | Sprint 108 Disposition |
|---|---:|---:|---:|---|
| `tests/test_ldlt_csc.c` | 3,887 | 3,896 | +9 | One bounded oracle/helper follow-through; direct proof intent retained. |
| `tests/test_qr.c` | 3,213 | 3,210 | -3 | Small fixture cleanup; solve/rank/reconstruction/refinement assertions retained. |
| `tests/test_iterative.c` | 2,828 | 2,849 | +21 | Convergence helper cleanup; solver options and comparison visibility retained. |
| `tests/test_svd.c` | 2,897 | 2,896 | -1 | Full-SVD fixture helper applied to approved call sites only. |
| `src/sparse_eigs.c` | 1,538 | 1,538 | 0 | Future extraction plan only; no source split. |
| `src/sparse_matrix.c` | 1,359 | 1,359 | 0 | Public-behavior review only; no source split. |

### Sprint 108 Residual Queue

| Priority | Residual | Next Gate |
|---:|---|---|
| 1 | Eigensolver dense Jacobi helper source candidate. | Move only `s21_dense_sym_jacobi` after Make/CMake/manifest parity and focused eigensolver validation are explicit. |
| 2 | Eigensolver grow-m refinement, dispatch, handle glue, and shared kernel boundaries. | Add behavior evidence before any source movement. |
| 3 | Matrix-shell source ownership candidates. | Start with one named public-behavior boundary and matching focused tests. |
| 4 | Additional giant-test proof-owner cleanup. | Keep future extractions bounded to one helper family at a time with visible call-site proof logic. |
