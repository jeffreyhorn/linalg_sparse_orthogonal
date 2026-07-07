# Sprint 110 Working Notes

## Sprint Goal

Sprint 110 converts Sprint 109's residual deferred debt into a bounded
Matrix I/O, eigensolver behavior-owner, and proof-owner follow-through pass.
The sprint must not duplicate completed Sprint 109 dense Jacobi extraction,
Matrix Market owner selection, QR exact-RHS cleanup, or validation/drift
closeout work. It must also avoid treating residual maintainability work as
permission to broaden public API, install headers, helper targets, source-list
membership, or reviewed CTest surfaces without explicit evidence and
validation.

## Starting Constraints

- Keep public API and install-header surfaces unchanged unless a later explicit
  review proves the change is required.
- Do not add a compiled test helper target.
- Do not change reviewed test counts unintentionally.
- Do not move Matrix Market code until Matrix builder ownership is resolved.
- Do not move behavior-sensitive eigensolver code unless one owner is selected,
  directly validated, and fenced from public-header drift.
- Keep QR, LDLT CSC, iterative, and SVD proof values visible at call sites when
  extracting setup helpers.
- Treat `src/sparse_matrix.c` as central public matrix-shell territory until
  builder ownership, Matrix Market load/save coupling, focused matrix tests,
  and solver-smoke fixtures are documented.

## Sprint 109 Completed Work Excluded From Sprint 110

The following Sprint 109 work is not unresolved Sprint 110 debt:

- Sprint 109 residual intake and dependency ordering.
- Dense Jacobi boundary review.
- Dense Jacobi extraction into `src/sparse_eigs_dense_internal.c`.
- Dense Jacobi Makefile, CMake, manifest, and source-list registration.
- Focused dense Jacobi Make and CMake validation.
- Eigensolver grow-m/refinement/shared-kernel no-move audit.
- Eigensolver dispatch/defaults/handle/workspace/shift-invert no-move audit.
- Matrix Market future-owner selection.
- QR exact-RHS helper cleanup in `tests/test_qr.c`.
- Sprint 109 focused integration, full validation, metrics, and drift checks.

## Sprint 110 Owner Inventory

| Owner | Carry-Forward Scope | Sprint 110 Guardrail |
|---|---|---|
| `src/sparse_matrix.c` | Central matrix shell, Matrix Market load/save, builder helpers, copy/transpose, allocation/state ownership, and factor-compatibility behavior. | Resolve builder ownership before any Matrix Market split; do not move broad matrix-shell behavior opportunistically. |
| `SparseBuildEntry` and `sparse_matrix_build_from_entries` | Internal build path currently shared by copy, transpose, and Matrix Market load. | Decide private builder source vs central shell ownership before touching Matrix Market source ownership. |
| Matrix Market load/save logic | File I/O, duplicate handling, symmetric expansion, pattern handling, errno behavior, and loaded-matrix solver use. | Move toward `src/sparse_matrix_io.c` only after builder ownership and validation gates are documented. |
| eigensolver behavior owners beyond dense Jacobi | Defaults, dispatch, workspace/growth, refinement, shift-invert, and shared Lanczos behavior. | Select at most one behavior owner; otherwise publish a no-move contract with direct validation requirements. |
| `tests/test_qr.c` | Remaining sequential RHS and proof setup after Sprint 109 exact-RHS helper cleanup. | Do not duplicate `make_qr_exact_rhs`; preserve least-squares and refinement proof values at call sites. |
| `tests/test_ldlt_csc.c` | External dense-reference oracle and LDLT CSC proof logic. | Treat oracle cleanup as a dedicated lane because it couples Python references, Windows skips, factorization, and dense solve comparison. |
| `tests/test_iterative.c` | Remaining exact-RHS setup for CG, GMRES, BiCGSTAB, and MINRES families. | Split cleanup by solver family; do not introduce one broad cross-solver helper that hides options or convergence evidence. |
| `tests/test_svd.c` | Storage-layout, stride, rank, orthogonality, reconstruction, pseudoinverse, low-rank, partial-SVD, and condition-number proof loops. | Extract at most one safe setup helper family and keep proof values visible. |

## Live Owner Snapshot

| Owner | Current Lines | Sprint 110 Disposition |
|---|---:|---|
| `src/sparse_matrix.c` | 1,359 | Builder ownership and Matrix Market source-boundary candidate; no broad shell movement without evidence. |
| `src/sparse_eigs.c` | 1,412 | Behavior-sensitive owner validation only; dense Jacobi work is already complete and excluded. |
| `src/sparse_eigs_dense_internal.c` | 129 | Completed dense Jacobi owner; no duplicate Sprint 110 work planned. |
| `tests/test_qr.c` | 3,234 | Eligible only for non-duplicate sequential RHS/proof setup cleanup if proof values stay visible. |
| `tests/test_ldlt_csc.c` | 3,896 | Eligible for one oracle-lane cleanup only if dense-reference proof remains visible. |
| `tests/test_iterative.c` | 2,849 | Eligible for one per-solver-family exact-RHS cleanup if convergence evidence remains visible. |
| `tests/test_svd.c` | 2,890 | Eligible for one setup helper family after proof-loop boundary review. |

## Day-Level Ownership

| Day | Planned Focus | Project Plan Item |
|---:|---|---|
| 1 | Residual intake, duplicate-work exclusions, dependency ordering, and validation expectations. | Item 1 |
| 2 | Matrix builder dependency audit. | Item 2 |
| 3 | Matrix builder ownership decision and prerequisite contract. | Item 2 |
| 4 | Matrix Market source-boundary plan or no-split deferral setup. | Item 3 |
| 5 | Matrix Market source split follow-through or no-split closure. | Item 3 |
| 6 | Matrix Market focused matrix and solver-smoke validation. | Item 3 |
| 7 | Eigensolver behavior-owner selection. | Item 4 |
| 8 | Eigensolver behavior-owner validation or no-move contract. | Item 4 |
| 9 | QR, LDLT CSC, and iterative proof-owner boundary selection. | Item 5 |
| 10 | One bounded QR, LDLT CSC, or iterative proof-owner cleanup. | Item 5 |
| 11 | SVD proof-loop boundary review. | Item 6 |
| 12 | One SVD proof-loop cleanup or explicit deferral. | Item 6 |
| 13 | Integrated validation, drift checks, and maintainability metrics. | Item 7 |
| 14 | Sprint closeout and downstream residual handoff. | Item 7 |

## Dependency Order

1. Residual intake and duplicate-work exclusions must precede all source or
   proof-owner movement.
2. Matrix builder ownership must precede any Matrix Market source split.
3. Matrix Market source-boundary planning must precede Matrix Market movement
   or no-split closure.
4. Matrix Market focused validation must close before downstream docs or
   package work can describe stable file-I/O ownership.
5. Eigensolver behavior-owner selection must precede any behavior-sensitive
   movement.
6. Direct/iterative proof-owner boundary selection must precede the bounded
   proof-owner cleanup batch.
7. SVD proof-loop boundary review must precede any SVD setup helper extraction.
8. Focused validation must precede full quality gate and residual closeout.

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

- Created Sprint 110 working notes and artifact directory.
- Converted Sprint 109 residual deferred debt into an explicit Sprint 110
  owner inventory.
- Recorded completed Sprint 109 work as exclusions to prevent duplicate
  cleanup.
- Ordered Matrix builder, Matrix Market, eigensolver behavior, proof-owner, SVD,
  validation, and closeout work so no day depends on a later item.
- Established validation expectations before source-boundary or proof-owner
  cleanup work begins.

## Day 2 Notes

- Audited the live Matrix builder seam in `src/sparse_matrix.c`.
- Confirmed the private builder objects are:
  - `SparseBuildEntry`;
  - `sparse_build_entry_cmp`;
  - `sparse_matrix_build_from_entries`.
- Confirmed direct builder callers:
  - `sparse_copy`;
  - `sparse_transpose`;
  - `sparse_load_mm`.
- Confirmed CSR/CSC constructors do not use the builder today; they validate
  compressed arrays and populate the matrix through `sparse_insert`.
- Captured builder-coupled public behavior:
  - row/column/order sorting for unsorted entry streams;
  - last duplicate entry wins;
  - final zero-valued entries are dropped;
  - copy preserves physical entries before separately cloning permutations,
    cached norms, factor state, and reorder permutation;
  - transpose changes shape and sorts transposed physical entries;
  - Matrix Market load expands symmetric entries, supports pattern matrices,
    validates one-based coordinates, and translates parser/build failures into
    existing public error codes.
- Recorded Day 3 decision options:
  - keep builder central and defer Matrix Market source movement;
  - split builder into a private source used by matrix shell and Matrix I/O;
  - publish a no-split deferral if private sharing would require unsafe header
    exposure or public behavior risk.

## Day 3 Notes

- Chose the private builder source option as the Sprint 110 Matrix builder
  ownership decision.
- Decision: `SparseBuildEntry`, `sparse_build_entry_cmp`, and
  `sparse_matrix_build_from_entries` should move into a private builder owner,
  provisionally `src/sparse_matrix_build_internal.c`, before Matrix Market
  load/save moves to any Matrix I/O source.
- Rationale:
  - the existing private `src/sparse_matrix_internal.h` already exposes
    `SparseMatrix`, `Node`, row/column headers, pool operations, and `nnz` to
    multiple internal implementation owners;
  - the builder can remain internal without public or install-header changes;
  - copy, transpose, and Matrix Market load should continue sharing one builder
    implementation instead of duplicating bulk-entry construction semantics.
- Required source-list order for implementation:
  - `src/sparse_matrix.c`;
  - future `src/sparse_matrix_build_internal.c`;
  - future `src/sparse_matrix_io.c`, if Day 4/5 proceeds with Matrix Market
    movement.
- Day 4 may plan Matrix Market movement only after the private builder source
  plan includes Makefile, CMake, `build-metadata/library_sources.txt`, focused
  copy/transpose/load tests, and at least one loaded-matrix solver-smoke lane.

## Day 4 Notes

- Planned Matrix Market movement as a two-step implementation:
  1. move the shared builder seam into `src/sparse_matrix_build_internal.c`;
  2. only after that, move Matrix Market load/save into
     `src/sparse_matrix_io.c`.
- Confirmed Matrix Market public declarations stay in
  `include/sparse_matrix.h`; no public API or install-header change is needed.
- Identified Matrix Market helper dependencies:
  - `sparse_errno_internal.h` for errno capture/reset;
  - `sparse_alloc_internal.h` for checked allocation and size conversions;
  - `sparse_matrix_internal.h` for private matrix traversal and logical
    row/column save behavior;
  - `sparse_stream_printf_checked`/`sparse_stream_vprintf_checked`, which must
    either remain central, move with Matrix I/O, or become private stream
    helpers without public exposure.
- Confirmed source-list insertion order for Day 5 implementation:
  - `src/sparse_matrix.c`;
  - `src/sparse_matrix_build_internal.c`;
  - `src/sparse_matrix_io.c`;
  - `src/sparse_factor_state_internal.c`.
- Recorded focused validation lanes:
  - `build/test_sparse_matrix`;
  - `build/test_sparse_io`;
  - `build/test_csr`;
  - `build/test_integration`;
  - `build/test_suitesparse` or `build/test_qr` as loaded-matrix solver smoke.
- No Matrix Market code moved on Day 4; Day 5 is the first implementation day.

## Day 5 Notes

- Implemented the planned Matrix Market source-boundary split.
- Moved the shared bulk-entry builder into
  `src/sparse_matrix_build_internal.c` with an internal
  `SparseBuildEntry` contract declared in `src/sparse_matrix_internal.h`.
- Moved `sparse_save_mm` and `sparse_load_mm` into
  `src/sparse_matrix_io.c`.
- Kept the public Matrix Market declarations in `include/sparse_matrix.h` and
  made no install-header or public API changes.
- Kept checked stream printing helpers in the central matrix owner because
  dense/entry/info debug printing still uses them outside Matrix Market save.
- Registered the new implementation files in:
  - `Makefile`;
  - `CMakeLists.txt`;
  - `build-metadata/library_sources.txt`.
- Preserved Matrix Market semantics:
  - one-based coordinate parsing;
  - symmetric expansion;
  - pattern matrices;
  - duplicate-entry last-write behavior through the shared builder;
  - final zero-entry elision;
  - errno capture/reset behavior.
- Required validation for Day 5 is implementation-level validation:
  focused matrix/I/O checks, source-list parity, `make format`, `make lint`,
  `make test`, and `git diff --check`.
- Day 5 validation completed:
  - `make source-list-check`;
  - focused build and execution for `test_sparse_matrix`, `test_sparse_io`,
    `test_csr`, `test_integration`, `test_suitesparse`, and `test_qr`;
  - `make format && make lint && make test`;
  - `git diff --check`;
  - trailing-whitespace scan over Sprint 110 docs and the new Matrix Market
    source files.

## Day 6 Notes

- Closed the Matrix Market focused validation lane for the Day 5 split.
- Re-ran source-list parity after the new Matrix Market implementation files
  were registered:
  - `make source-list-check` passed with 48 library sources.
- Ran the reviewed CMake compile/parity path:
  - `make quality-review-cmake-compile` passed;
  - CMake clean rebuild included `src/sparse_matrix_build_internal.c` and
    `src/sparse_matrix_io.c`;
  - `ctest -N --test-dir build/quality-review-cmake` reported 54 tests;
  - Makefile/CMake test-count parity reported 54 Makefile tests and 54 CMake
    tests.
- Ran selected Matrix Market and loaded-matrix CTest lanes:
  - `test_sparse_matrix`;
  - `test_sparse_io`;
  - `test_integration`;
  - `test_suitesparse`;
  - `test_csr`;
  - `test_qr`.
- Confirmed no public API, install-header, install/export, or CTest
  registration drift is part of the Matrix Market split.
- Matrix I/O work is closed for Sprint 110; Day 7 can begin eigensolver
  behavior-owner selection without depending on unresolved Matrix Market
  validation.

## Day 7 Notes

- Re-read the Sprint 109 eigensolver no-go artifacts:
  - `day6-growm-refinement-shared-kernel-audit.md`;
  - `day7-dispatch-handle-shift-invert-audit.md`.
- Confirmed dense Jacobi extraction is complete Sprint 109 work and is excluded
  from Sprint 110 duplication.
- Inventoried the remaining eigensolver behavior-owner candidates:
  - defaults and option validation;
  - backend dispatch;
  - public handle and workspace preparation;
  - grow-m sizing and retry behavior;
  - refinement defaults and budgets;
  - shift-invert setup and LDLT backend reporting;
  - shared Lanczos kernels.
- Selected exactly one Day 8 validation target:
  the public handle/workspace bridge around
  `sparse_eigs_handle_prepare`, `sparse_eigs_sym_with_handle`,
  `s49_eigs_handle_prepare_backend`, and the grow-m, thick-restart, and
  LOBPCG workspace prepare calls.
- Selection rationale:
  - it is narrower than dispatch/defaults, shift-invert, refinement, grow-m,
    or shared Lanczos movement;
  - it already has direct public-handle tests across all three backend
    workspace views;
  - validation can prove reuse/growth behavior without moving code or changing
    public headers.
- Day 8 should validate the selected owner and publish a no-move contract
  unless a strictly smaller internal cleanup proves unnecessary public-surface
  risk.
- Explicit Day 7 deferrals:
  - no dispatch/default source movement;
  - no shift-invert source movement;
  - no refinement source movement;
  - no grow-m backend movement;
  - no shared Lanczos-kernel movement;
  - no public header or install-header change.

## Day 8 Notes

- Validated the Day 7 selected eigensolver owner without moving code.
- Published the public handle/workspace bridge no-move contract for Sprint 110.
- Confirmed the selected owner remains covered by direct public-handle tests:
  - grow-m prepare/reuse/growth;
  - generic prepare/reuse;
  - validation and on-demand allocation;
  - thick-restart prepare/reuse/growth;
  - LOBPCG prepare/reuse/growth.
- Ran focused Make eigensolver validation:
  - `make build/test_eigs build/test_eigs_thick_restart
    build/test_eigs_lobpcg build/test_sprint29_integration`;
  - `build/test_eigs`;
  - `build/test_eigs_thick_restart`;
  - `build/test_eigs_lobpcg`;
  - `build/test_sprint29_integration`.
- Ran CTest no-drift and focused CMake validation:
  - `ctest -N --test-dir build/quality-review-cmake` reported 54 tests;
  - `ctest --test-dir build/quality-review-cmake --output-on-failure -R
    '^(test_eigs|test_eigs_thick_restart|test_eigs_lobpcg|test_sprint29_integration)$'`
    passed 4 of 4 tests.
- Re-ran `make source-list-check`; it passed with 48 library sources.
- Confirmed Day 8 introduced no public header, install-header, source-list,
  helper-target, CTest registration, or eigensolver source drift.
- Unsafe behavior-sensitive eigensolver movement remains explicitly deferred:
  dispatch/defaults, shift-invert, refinement, grow-m, and shared
  Lanczos-kernel movement all require future direct owner-specific proof.

## Day 9 Notes

- Reviewed the remaining direct and iterative proof-owner candidates:
  - QR sequential RHS setup after the completed Sprint 109 exact-RHS helper;
  - LDLT CSC external dense-reference oracle cleanup;
  - iterative exact-RHS setup split by solver family.
- Explicitly excluded the completed Sprint 109 QR exact-RHS cleanup:
  `make_qr_exact_rhs` and its seven call-site replacements are not Sprint 110
  work.
- Confirmed LDLT CSC external dense-reference cleanup remains too coupled for
  the Day 10 batch because it spans the Python oracle, Windows skip behavior,
  LDLT CSC factorization, permutation handling, dense-reference comparison,
  and residual proof.
- Confirmed QR sequential RHS cleanup is safe only as a later setup-only pass
  because the remaining QR sites are mostly least-squares, null-residual, and
  refinement smoke paths where literal RHS values explain the proof.
- Selected one Day 10 cleanup family:
  `tests/test_iterative.c` CG exact-RHS allocation/setup.
- Day 10 scope is limited to a local static setup helper in
  `tests/test_iterative.c` for dynamically allocated CG exact-solution RHS
  cases. It may allocate/fill `x_exact`, allocate `b`, and compute
  `b = A*x_exact`.
- Day 10 must keep all solver calls, options, convergence assertions,
  iteration comparisons, residual thresholds, residual recomputation, prints,
  preconditioner setup, and cleanup visible at call sites.
- Focused validation for a Day 10 code change:
  `make build/test_iterative && build/test_iterative`, followed by
  `make format && make lint && make test` because `tests/test_iterative.c`
  is a `.c` file.

## Day 10 Notes

- Implemented the selected iterative CG exact-RHS cleanup in
  `tests/test_iterative.c`.
- Added a local callback-based helper that allocates/fills `x_exact`, allocates
  `b`, and computes `b = A*x_exact` through the existing `compute_rhs` helper.
- Added a fatal wrapper for allocation/setup failure so null proof buffers are
  not dereferenced after a non-fatal test assertion.
- Replaced repeated dynamic exact-RHS setup in:
  - `test_cg_laplacian_2d`;
  - `test_cg_initial_guess`;
  - `test_cg_large_tridiag`;
  - `test_cg_max_iter_exceeded`;
  - `test_cg_nos4`;
  - `test_cg_bcsstk04`;
  - `test_cg_suitesparse_initial_guess`;
  - `test_cg_tight_tolerance`;
  - `test_cg_loose_tolerance`;
  - `test_cg_residual_accuracy`.
- Kept solver calls, options, convergence assertions, residual thresholds,
  independent residual recomputation, iteration comparisons, printed labels,
  exact initial-guess setup, and cleanup visible at call sites.
- Did not move preconditioner-specific CG setup, stack/literal CG proof
  vectors, GMRES, BiCGSTAB, MINRES, or handle-helper setup.
- Day 10 validation completed:
  - `make build/test_iterative` passed;
  - `build/test_iterative` passed with 80 tests, 0 failures, 0 skipped;
  - first `make format && make lint && make test` run stopped at
    `make lint` because `cppcheck` correctly required the local callback
    context to be `const void *`;
  - patched the helper callback type and wrapper context to `const void *`;
  - reran `make build/test_iterative && build/test_iterative`, which passed
    with 80 tests, 0 failures, 0 skipped;
  - reran `make format && make lint && make test`, which passed.

## Day 11 Notes

- Reviewed `tests/test_svd.c` proof-loop boundaries for the Sprint 110 SVD
  cleanup item.
- Explicitly excluded completed SVD cleanup work from prior sprints:
  - `make_svd_diag_matrix`;
  - `make_svd_rank1_row_progression`;
  - `make_svd_full_uv_fixture_16x8`;
  - Sprint 103's diagonal rank/full-UV claim fixture;
  - existing reconstruction, orthogonality, and partial-SVD helpers.
- Mapped the remaining SVD proof surfaces:
  - Golub-Kahan extraction;
  - full SVD singular values;
  - full/economy UV behavior;
  - rank comparison;
  - pseudoinverse;
  - dense low-rank;
  - sparse low-rank;
  - partial SVD;
  - condition number.
- Rejected moving reconstruction loops, U/Vt orthogonality loops,
  Moore-Penrose products, low-rank Frobenius residuals, partial-SVD vector
  residuals, and condition-number assertions because those loops encode the
  proof intent directly.
- Selected exactly one Day 12 setup-helper family:
  a local `make_svd_rank_deficient_colpair_5x4` fixture builder for the
  repeated 5x4 rank-deficient column-pair setup used by
  `test_svd_rank_vs_qr` and `test_svd_rank_deficient`.
- Day 12 must keep SVD rank counting, `tol_svd`, QR comparison, expected rank
  values, printed labels, and cleanup visible at call sites.
- Day 11 made documentation-only changes; no SVD code, headers, helper
  targets, source lists, public API, install headers, or CTest registrations
  changed.

## Day 12 Notes

- Implemented the Day 11-selected SVD setup-helper cleanup in
  `tests/test_svd.c`.
- Added one local static helper:
  `make_svd_rank_deficient_colpair_5x4`.
- Kept the helper limited to repeated 5x4 rank-deficient matrix construction
  where column 1 duplicates column 0 and column 3 duplicates column 2.
- Replaced duplicated setup in:
  - `test_svd_rank_vs_qr`;
  - `test_svd_rank_deficient`.
- Kept SVD rank counting, `tol_svd`, QR comparison, expected rank value `2`,
  printed rank labels, and cleanup visible at call sites.
- Did not move reconstruction loops, U/Vt orthogonality loops,
  Moore-Penrose products, low-rank Frobenius residuals, partial-SVD vector
  residuals, or condition-number assertions.
- Day 12 validation completed:
  - `make build/test_svd`;
  - `build/test_svd` with 98 tests, 0 failures, 0 skipped, and 1,562
    assertions;
  - `make format && make lint && make test`.
- Day 12 hygiene checks completed:
  - `git diff --check`;
  - trailing-whitespace scan over touched Sprint 110 docs and
    `tests/test_svd.c`.

## Day 13 Notes

- Completed the integrated validation and metrics pass for Sprint 110 code,
  test, build-system, and documentation changes.
- Re-ran source-list parity:
  - `make source-list-check` passed with 48 library sources.
- Verified reviewed CTest registration before and during the reviewed CMake
  compile/parity path:
  - `ctest -N --test-dir build/quality-review-cmake` reported 54 tests;
  - `make quality-review-cmake-compile` passed;
  - Makefile/CMake test-count parity reported 54 Makefile tests and 54 CMake
    tests.
- Re-ran the required full quality gate because Sprint 110 changed
  implementation `.c`, private `.h`, test `.c`, and build-system files:
  - `make format && make lint && make test` passed.
- Confirmed public API and install-header no-drift:
  - `git diff --name-only -- include` produced no output.
- Captured maintainability metrics:
  - `src/sparse_matrix.c`: 1,359 lines on `master`, 1,053 lines after the
    Matrix builder and Matrix I/O split;
  - `src/sparse_matrix_build_internal.c`: 111 lines;
  - `src/sparse_matrix_io.c`: 198 lines;
  - `src/sparse_matrix_internal.h`: 251 lines on `master`, 267 lines after
    adding the private builder contract;
  - `tests/test_iterative.c`: 2,849 lines on `master`, 2,908 lines after the
    CG exact-RHS setup helper;
  - `tests/test_svd.c`: 2,890 lines on `master`, 2,893 lines after the
    rank-deficient SVD setup helper.
- Wrote the Day 13 integrated validation artifact:
  `artifacts/day13-integrated-validation-and-metrics.md`.

## Day 14 Notes

- Reconciled every Sprint 110 project-plan item against the final artifact set.
- Confirmed all seven Sprint 110 project-plan items have a final disposition:
  residual intake, Matrix builder ownership, Matrix Market source split,
  eigensolver behavior-owner validation, direct/iterative proof-owner cleanup,
  SVD proof-loop cleanup, and validation/residual handoff.
- Confirmed the duplicate-work fence held:
  - no duplicate dense Jacobi extraction;
  - no duplicate Sprint 109 Matrix Market owner-selection-only work;
  - no duplicate Sprint 109 QR exact-RHS cleanup;
  - no unproven eigensolver source movement;
  - no broad proof-helper target;
  - no public API, install-header, helper-target, or reviewed CTest drift.
- Published the dependency-ordered residual queue:
  - Sprint 111 should use the Matrix I/O split evidence for user-facing docs
    without claiming a public Matrix I/O module or public builder API;
  - Sprint 112 should use Sprint 110's no-public-header-drift result as package
    and platform baseline evidence, without inferring shared-library/ABI or
    broader Windows support;
  - Sprint 113 or later maintainability work should handle eigensolver
    behavior-owner movement, direct/iterative proof-owner cleanup, and SVD
    proof-owner cleanup only after fresh boundary artifacts.
- Wrote the Day 14 closeout artifact:
  `artifacts/day14-closeout-and-residual-handoff.md`.
- Sprint 110 is ready for retrospective creation.
