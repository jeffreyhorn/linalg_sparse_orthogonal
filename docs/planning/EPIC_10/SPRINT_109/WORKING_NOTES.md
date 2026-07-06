# Sprint 109 Working Notes

## Sprint Goal

Sprint 109 converts Sprint 108's residual deferred debt into a bounded
implementation/source-boundary pass. The sprint must not duplicate Sprint 108
helper work, and it must not treat residual source-boundary planning as
permission to broaden public API, install headers, helper targets, source-list
membership, or reviewed CTest surfaces without explicit evidence and validation.

## Starting Constraints

- Keep public API and install-header surfaces unchanged unless a later explicit
  review proves the change is required.
- Do not add a compiled test helper target.
- Do not change reviewed test counts unintentionally.
- Do not move implementation source without Makefile, CMake, manifest, and
  source-list parity checks.
- Keep proof assertions visible at giant-test call sites when extracting setup
  helpers.
- Treat `src/sparse_eigs.c` behavior-sensitive paths as no-move candidates
  unless the sprint records stronger shift-invert, backend-selection,
  handle/workspace, and cross-backend numerical evidence.
- Treat `src/sparse_matrix.c` as central public matrix-shell territory until
  one future public-behavior owner, private-header dependencies, source-list
  requirements, focused behavior tests, and solver-smoke gates are documented.

## Sprint 108 Completed Work Excluded From Sprint 109

The following Sprint 108 work is not unresolved Sprint 109 debt:

- Sprint 108 residual intake and proof-owner boundary refresh.
- LDLT CSC residual assertion helper follow-through:
  `assert_s20_solve_residual_below`.
- QR tall diagonal-dominant fixture helper:
  `make_qr_tall_diagonal_dominant`.
- Iterative diagonal-preconditioner fixture helper:
  `make_iterative_diagonal_precond_matrix`.
- SVD full-UV fixture helper:
  `make_svd_full_uv_fixture_16x8`.
- Sprint 108 eigensolver feasibility boundary and closeout handoff.
- Sprint 108 matrix-shell public-behavior review.
- Sprint 108 final validation, metrics, and drift checks.

## Sprint 109 Owner Inventory

| Owner | Carry-Forward Scope | Sprint 109 Guardrail |
|---|---|---|
| `src/sparse_eigs.c` | Dense Jacobi helper candidate plus behavior-sensitive grow-m, refinement, dispatch/defaults, handle/workspace, shift-invert, and shared Lanczos territory. | Move only `s21_dense_sym_jacobi` if Days 2-3 prove the boundary is low risk and source-list parity is complete. |
| eigensolver private/internal headers and source lists | Internal declaration placement, helper source ownership, Make/CMake/manifest membership, and focused eigensolver validation. | No public header or install-header change; no source-list drift between build systems. |
| `src/sparse_matrix.c` | Central public matrix shell, allocation/state ownership, mutation, copy, transpose, norms, Matrix Market, and factor compatibility behavior. | Choose one future public-behavior owner contract; do not move shell code without independent low-risk evidence. |
| `tests/test_ldlt_csc.c` | Remaining direct-solver proof owner and oracle logic after Sprint 108 helper cleanup. | Do not repeat Day 4 residual helper work; select only one future helper family if proof visibility remains strong. |
| `tests/test_qr.c` | Remaining generated fixture, exact-RHS, solve/rank/reconstruction/refinement proof setup after Sprint 108 fixture cleanup. | Do not repeat tall diagonal-dominant helper work; keep proof assertions visible. |
| `tests/test_iterative.c` | Remaining convergence-sensitive setup after Sprint 108 diagonal-preconditioner fixture cleanup. | Do not hide solver options, restarts, preconditioners, convergence results, or comparisons. |
| `tests/test_svd.c` | Remaining rank, oracle, reconstruction, pseudoinverse, low-rank, partial-SVD, and condition-number proof logic after Sprint 108 full-UV fixture cleanup. | Do not repeat full-UV fixture work; validate any new cleanup family before movement. |

## Day-Level Ownership

| Day | Planned Focus | Project Plan Item |
|---:|---|---|
| 1 | Residual intake, exclusions, dependency ordering, and validation expectations. | Item 1 |
| 2 | Dense Jacobi source-boundary revalidation. | Item 2 |
| 3 | Source-list parity and focused validation harness prep. | Item 2 |
| 4 | Dense Jacobi extraction or no-split deferral. | Item 3 |
| 5 | Dense Jacobi focused cross-lane validation. | Item 3 |
| 6 | Grow-m, refinement, and shared-kernel behavior audit. | Item 4 |
| 7 | Dispatch/defaults, handle/workspace, and shift-invert audit. | Item 4 |
| 8 | Matrix-shell candidate public-behavior boundary contract. | Item 5 |
| 9 | Matrix-shell validation and move/no-move decision. | Item 5 |
| 10 | Giant-test cleanup candidate selection. | Item 6 |
| 11 | Giant-test cleanup follow-through. | Item 6 |
| 12 | Focused integration and drift check. | Item 7 |
| 13 | Full quality gate and maintainability metrics. | Item 7 |
| 14 | Sprint 109 residual closeout and downstream handoff. | Item 7 |

## Dependency Order

1. Residual intake and duplicate-work exclusions must precede all source or
   proof-owner movement.
2. Dense Jacobi boundary and source-list planning must precede any eigensolver
   extraction.
3. Dense Jacobi validation must close before broader eigensolver behavior
   audits are used for future planning.
4. Eigensolver behavior audits must precede downstream claims about additional
   source movement.
5. Matrix-shell contract work must precede any future matrix-shell extraction.
6. Giant-test cleanup candidate selection must precede the cleanup pass.
7. Focused validation must precede full quality gate and residual closeout.

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

- Created Sprint 109 working notes and artifact directory.
- Converted Sprint 108 residual deferred debt into an explicit Sprint 109 owner
  inventory.
- Recorded completed Sprint 108 helper work as exclusions to prevent duplicate
  cleanup.
- Ordered eigensolver, matrix-shell, giant-test, validation, and closeout work
  so no day depends on a later item.
- Established validation expectations before source-boundary or proof-owner
  cleanup work begins.

## Day 2 Notes

- Revalidated `s21_dense_sym_jacobi` as the only Sprint 109 eigensolver source
  movement candidate.
- Confirmed live ownership:
  - implementation: `src/sparse_eigs.c`;
  - private declaration: `src/sparse_eigs_internal.h`;
  - direct runtime callers: `src/sparse_eigs_thick_restart.c` and
    `src/sparse_eigs_lobpcg.c`.
- Confirmed the helper depends only on dense buffers, `idx_t`, `sparse_err_t`,
  `SPARSE_*` return codes, `sqrt`, and `fabs`; it does not own public
  eigensolver options, results, handles, workspaces, shift-invert setup, or
  dispatch/default behavior.
- Proposed `src/sparse_eigs_dense_internal.c` as the private source owner while
  keeping the existing declaration in `src/sparse_eigs_internal.h`.
- Recorded source-list/build touch points for any future extraction:
  `Makefile`, `CMakeLists.txt`, and
  `build-metadata/library_sources.txt`.
- Recorded focused validation expectations:
  `make source-list-check`, focused `test_eigs`,
  `test_eigs_thick_restart`, `test_eigs_lobpcg`,
  `test_sprint29_integration`, and the broad quality gate if code moves.
- Day 2 moved no code; it approves only planning for a Day 4 extraction attempt
  after Day 3 source-list and validation prep.

## Day 3 Notes

- Inspected live source-membership owners:
  - `Makefile` `LIB_SRCS`;
  - `CMakeLists.txt` static library source list;
  - `build-metadata/library_sources.txt`.
- Confirmed all three currently order eigensolver library sources as:
  `sparse_eigs_workspace_internal.c`, `sparse_eigs_lobpcg.c`,
  `sparse_eigs_thick_restart.c`, `sparse_eigs.c`.
- Proposed adding `src/sparse_eigs_dense_internal.c` after
  `src/sparse_eigs_workspace_internal.c` and before backend owners in all
  three source-list surfaces if Day 4 extracts the helper.
- Confirmed the parity gate is `make source-list-check`, backed by
  `scripts/check_library_sources.py`.
- Confirmed focused eigensolver tests are registered in both Makefile and
  CMake surfaces:
  `test_eigs`, `test_eigs_thick_restart`, `test_eigs_lobpcg`, and
  `test_sprint29_integration`.
- Recorded Day 4 no-drift expectations:
  no public header, install-header, pkg-config/install/export, helper-target,
  test-source, or CTest registration changes.
- Day 3 moved no code and changed no build/source-list files.

## Day 4 Notes

- Extracted only `s21_dense_sym_jacobi` from `src/sparse_eigs.c` into the new
  private source owner `src/sparse_eigs_dense_internal.c`.
- Left the private declaration in `src/sparse_eigs_internal.h` unchanged.
- Preserved both direct callers unchanged:
  - `src/sparse_eigs_thick_restart.c`;
  - `src/sparse_eigs_lobpcg.c`.
- Added the new private source to all library source-list owners in matching
  order:
  - `Makefile`;
  - `CMakeLists.txt`;
  - `build-metadata/library_sources.txt`.
- Confirmed source-list parity with `make source-list-check`.
- Confirmed focused eigensolver behavior with:
  `test_eigs`, `test_eigs_thick_restart`, `test_eigs_lobpcg`, and
  `test_sprint29_integration`.
- Ran the required code-change quality gate:
  `make format && make lint && make test`.
- Confirmed diff hygiene with `git diff --check` and no trailing whitespace in
  Sprint 109 docs.
- Recorded source-size movement:
  - `src/sparse_eigs.c`: 1538 lines before, 1412 lines after;
  - `src/sparse_eigs_dense_internal.c`: 129 lines after extraction.
- Confirmed no public header, install-header, helper-target, test-source, or
  CTest registration change was needed for the extraction.

## Day 5 Notes

- Re-ran the focused Make eigensolver validation lanes after the Day 4
  extraction:
  - `test_eigs`: passed, 31 tests;
  - `test_eigs_thick_restart`: passed, 21 tests;
  - `test_eigs_lobpcg`: passed, 27 tests;
  - `test_sprint29_integration`: passed, 3 tests.
- Reconfirmed source-list parity with `make source-list-check`, which passed
  with 46 library sources.
- Verified matching source ordering across `Makefile`, `CMakeLists.txt`, and
  `build-metadata/library_sources.txt`:
  `sparse_eigs_workspace_internal.c`,
  `sparse_eigs_dense_internal.c`,
  `sparse_eigs_lobpcg.c`,
  `sparse_eigs_thick_restart.c`, and `sparse_eigs.c`.
- Configured `build/day5-cmake-ctest` for CMake registration and generated
  compile-command inspection.
- Confirmed generated CMake compile commands include
  `src/sparse_eigs_dense_internal.c` between the workspace owner and backend
  owners.
- Built and ran the four focused CMake/CTest eigensolver targets:
  `test_sprint29_integration`, `test_eigs`, `test_eigs_thick_restart`, and
  `test_eigs_lobpcg`; all 4 passed.
- Confirmed local CMake CTest registration reports 54 tests, with the focused
  eigensolver lanes registered as tests 46-49.
- Recorded the only residual risk as broader eigensolver behavior outside the
  dense Jacobi helper: grow-m/refinement/shared-kernel boundaries remain Day 6
  audit scope, and dispatch/default/handle/shift-invert boundaries remain
  Day 7 audit scope.
- Day 5 made no additional code or build-system changes.

## Day 6 Notes

- Audited grow-m, refinement, and shared-kernel eigensolver boundaries without
  moving additional code.
- Classified `s46_run_growm_backend` as a no-move Sprint 109 candidate because
  it owns m-cap sizing, retry growth, progress callback timing, cumulative
  iteration accounting, partial-result emission, `peak_basis_size`,
  `residual_norm`, and workspace view assumptions.
- Classified refinement helpers `s29_refine_pair`,
  `s29_refine_eigenpairs`, and `s29_maybe_refine` as no-move Sprint 109
  candidates because they mutate public result eigenpairs in place, allocate
  direct-solver scratch, build shifted matrix copies, perturb singular shifts,
  and preserve backend return-code semantics.
- Classified shared kernels as needs-more-proof before movement:
  `s21_mgs_reorth`, `s20_lanczos_starting_vector`,
  `s20_spectrum_scale`, `s20_select_indices`, and
  `s20_lift_ritz_vectors`.
- Recorded that these shared kernels are not behavior-free utilities; they
  encode reorthogonalization stability, deterministic starting-vector
  behavior, residual scale anchoring, selection ordering, shift-invert
  ordering, and vector-lift conventions used by multiple backends and tests.
- Identified future split prerequisites:
  dedicated private owner naming, source-list parity, focused direct tests for
  each moved helper, cross-backend eigensolver validation, refinement
  validation, and no public header/install-header drift.
- Left dispatch/defaults, public handle/workspace glue, and shift-invert
  factoring to Day 7.
- Day 6 made no code or build-system changes.

## Day 7 Notes

- Audited dispatch/default behavior, public handle/workspace ownership, and
  shift-invert setup without moving additional code.
- Classified `s46_default_public_opts`, `s46_validate_public_entry`,
  `s49_eigs_effective_max_iters`, `s46_select_backend`,
  `s46_run_backend`, and `s46_sparse_eigs_sym_impl` as no-move Sprint 109
  candidates because they jointly implement public defaults, option
  validation, backend routing, result initialization, cleanup, and refinement
  post-pass semantics.
- Classified public handle helpers `s49_eigs_handle_ensure`,
  `s49_eigs_handle_prepare_backend`, `sparse_eigs_handle_prepare`,
  `sparse_eigs_sym_with_handle`, and `sparse_eigs_handle_free` as no-move
  Sprint 109 candidates because they own reusable workspace allocation,
  capacity matching, on-demand growth, and public lifetime behavior.
- Classified shift-invert setup in `s46_sparse_eigs_sym_impl` as no-move for
  Sprint 109 because it copies and shifts `A`, factors `(A - sigma I)` through
  LDLT AUTO dispatch, records `used_csc_path_ldlt`, swaps the Lanczos operator
  callback, post-processes inverse Ritz values, and cleans up factorization
  state before optional refinement.
- Confirmed current tests already cover the relevant contracts:
  handle reuse/growth in `tests/test_eigs.c`, AUTO routing in
  `tests/test_eigs_thick_restart.c` and `tests/test_eigs_lobpcg.c`,
  shift-invert and LDLT path reporting in `tests/test_eigs.c`, and
  refinement/progress/cancellation interaction in
  `tests/test_sprint29_integration.c`.
- Recorded future split prerequisites for any dispatch, handle/workspace, or
  shift-invert movement: public-contract owner naming, direct validation for
  defaults and option rejection, CMake/Make/source-list parity, focused
  backend tests, consumer-facing docs review, and full quality gate if code
  moves.
- Closed the Sprint 109 eigensolver audit with no additional movement beyond
  the Day 4 dense Jacobi extraction.
- Day 7 made no code or build-system changes.

## Day 8 Notes

- Re-read Sprint 108's matrix-shell public-behavior review and current
  `src/sparse_matrix.c`, `include/sparse_matrix.h`, and focused matrix I/O
  tests.
- Reconfirmed `src/sparse_matrix.c` owns lifecycle, mutation, access, copy,
  transpose, matrix properties, factor/permutation compatibility, arithmetic,
  matvec/block-matvec, Matrix Market I/O, and print/debug helpers.
- Selected one future public-behavior owner candidate:
  `src/sparse_matrix_io.c` for Matrix Market load/save only.
- Explicitly excluded lifecycle, mutation, arithmetic/matvec,
  factor/permutation compatibility, copy/transpose, and print/debug helpers
  from the selected Day 8 candidate.
- Recorded private dependency constraints for the future I/O split:
  `sparse_save_mm` needs logical row/column permutation access and checked
  stream writes; `sparse_load_mm` needs Matrix Market parsing, errno handling,
  symmetric/pattern expansion, and a private bulk-entry builder.
- Flagged `SparseBuildEntry` and `sparse_matrix_build_from_entries` as the
  main prerequisite decision because the builder is currently static in
  `src/sparse_matrix.c` and also supports copy/transpose behavior.
- Defined focused tests for a future I/O split:
  `test_sparse_io`, `test_sparse_matrix` duplicate-load/permuted-save cases,
  `test_known_matrices`, `test_integration`, and `test_csr` load-backed smoke.
- Defined solver smoke guardrails for Matrix Market movement because many
  solver suites consume `sparse_load_mm` fixtures.
- Day 8 made no code, header, or build-system changes.

## Day 9 Notes

- Validated the Day 8 Matrix Market future-owner contract against focused
  matrix public-behavior tests:
  - `test_sparse_io`: passed, 26 tests;
  - `test_sparse_matrix`: passed, 63 tests;
  - `test_known_matrices`: passed, 15 tests;
  - `test_integration`: passed, 58 tests;
  - `test_csr`: passed, 19 tests.
- Ran representative solver-smoke lanes that consume Matrix Market fixtures:
  - `test_sparse_lu`: passed, 40 tests;
  - `test_cholesky`: passed, 21 tests;
  - `test_ldlt`: passed, 89 tests;
  - `test_iterative`: passed, 80 tests;
  - `test_eigs`: passed, 31 tests;
  - `test_svd`: passed, 98 tests.
- Confirmed `sparse_load_mm` fixture use is broad across direct, iterative,
  eigensolver, SVD, graph, reorder, and integration tests, so a future I/O
  move needs solver-smoke validation in addition to `test_sparse_io`.
- Confirmed no matrix source, matrix private header, public header, or install
  header changed on this branch for Day 9.
- Reconfirmed the selected future owner remains `src/sparse_matrix_io.c`, but
  movement is blocked in Sprint 109 by the static bulk-entry builder
  dependency shared with `sparse_copy` and `sparse_transpose`.
- Published the Sprint 109 matrix-shell no-move decision: public behavior
  remains the proof owner, and a future split must first resolve private
  builder ownership plus source-list parity and focused solver-smoke gates.
- Day 9 made no code, header, or build-system changes.

## Day 10 Notes

- Re-inventoried residual giant-test cleanup candidates in
  `tests/test_ldlt_csc.c`, `tests/test_qr.c`, `tests/test_iterative.c`, and
  `tests/test_svd.c`.
- Recorded current large-test sizes:
  - `tests/test_ldlt_csc.c`: 3896 lines;
  - `tests/test_qr.c`: 3210 lines;
  - `tests/test_iterative.c`: 2849 lines;
  - `tests/test_svd.c`: 2890 lines.
- Excluded duplicate work from Sprint 108 and earlier Sprint 109 days:
  LDLT CSC adjacency/duplicate-entry helpers, `assert_s20_solve_residual_below`,
  QR 4x3 fixture builders, `make_qr_tall_diagonal_dominant`, iterative solver
  helper headers and diagonal-preconditioner follow-through, SVD diagonal,
  rank-1, and full-U/V fixtures, and the Day 4 dense Jacobi extraction.
- Ranked the remaining candidates by proof clarity, review size, validation
  cost, and failure localization.
- Selected one bounded future cleanup batch:
  `tests/test_qr.c` exact-solution RHS setup helper.
- Defined proof-visibility rules for that future QR cleanup:
  helper code may hide only allocation, sequential exact-solution fill, and
  `sparse_matvec(A, x_exact, b)`, while solver calls, expected ranks, residual
  labels, tolerances, reconstruction checks, and comparison assertions remain
  visible at call sites.
- Defined focused validation for the future code-change day:
  `make build/test_qr && ./build/test_qr`, followed by
  `make format && make lint && make test` because that implementation will
  modify a `.c` file.
- Day 10 made no code, header, or build-system changes.

## Day 11 Notes

- Implemented the Day 10 selected cleanup batch in `tests/test_qr.c`.
- Added one local static setup helper:
  `make_qr_exact_rhs(const SparseMatrix *A, idx_t x_len, idx_t b_len,
  double **x_exact_out, double **b_out)`.
- Kept the helper limited to repeated setup:
  allocation, sequential exact-solution fill, and `b = A*x_exact`.
- Replaced seven repeated exact-RHS setup blocks in:
  - `test_qr_solve_nos4`;
  - `test_qr_bcsstk04`;
  - `test_qr_west0067`;
  - `test_qr_vs_lu`;
  - `test_qr_tall_synthetic`;
  - `test_qr_reorder_nos4_fillin`;
  - `test_qr_refine_nos4`.
- Preserved solver calls, expected rank checks, residual labels, residual
  tolerances, reconstruction assertions, refinement assertions, and QR-vs-LU
  comparison loops at their call sites.
- Left tiny literal RHS cases, overdetermined least-squares RHS values,
  rank-deficient RHS values, dense-vs-sparse QR comparisons, and Sprint 108 QR
  fixture builders unchanged.
- Captured before/after maintainability metrics:
  - `tests/test_qr.c`: 3210 lines in the Day 10 inventory, 3194 lines after
    Day 11 formatting;
  - exact-RHS helper calls: 0 before, 7 after;
  - selected repeated exact-RHS fill/matvec blocks: 7 before, 0 after;
  - new helper targets: 0;
  - CTest registration changes: 0.
- Focused QR validation passed:
  `make build/test_qr && ./build/test_qr` completed with 73 tests,
  647 assertions, 0 failures, and 0 skips.
- Required C quality gate passed:
  `make format && make lint && make test`.
- Remaining residuals are deferred explicitly:
  QR sequential RHS fill helper, LDLT CSC external dense-reference oracle
  cleanup, per-solver iterative exact-RHS helper families, and SVD
  storage-layout proof-loop cleanup.

## Day 12 Notes

- Ran focused integration validation for every touched Sprint 109
  implementation and build-system surface:
  `src/sparse_eigs.c`, `src/sparse_eigs_dense_internal.c`, `Makefile`,
  `CMakeLists.txt`, `build-metadata/library_sources.txt`, and
  `tests/test_qr.c`.
- Make focused validation passed:
  - `make source-list-check`: passed with 46 library sources;
  - `test_eigs`: passed, 31 tests;
  - `test_eigs_thick_restart`: passed, 21 tests;
  - `test_eigs_lobpcg`: passed, 27 tests;
  - `test_sprint29_integration`: passed, 3 tests;
  - `test_qr`: passed, 73 tests.
- Configured `build/day12-cmake-ctest` with exported compile commands and
  built the focused CMake targets:
  `test_eigs`, `test_eigs_thick_restart`, `test_eigs_lobpcg`,
  `test_sprint29_integration`, and `test_qr`.
- Focused CTest validation passed:
  5 tests, 0 failures.
- Built the full CMake default target in `build/day12-cmake-ctest` and
  re-ran `ctest -N`; registration remains 54 tests.
- Confirmed focused CTest registration remains:
  - `test_qr`: test #20;
  - `test_sprint29_integration`: test #46;
  - `test_eigs`: test #47;
  - `test_eigs_thick_restart`: test #48;
  - `test_eigs_lobpcg`: test #49.
- Verified public API and install-header no-drift:
  `git diff --name-only -- include src/*.h tests/*.h` produced no output.
- Verified matrix-shell no-drift:
  no diff in `src/sparse_matrix.c`, `include/sparse_matrix.h`,
  `tests/test_sparse_io.c`, or `tests/test_sparse_matrix.c`.
- Verified the only build-system drift is the private
  `src/sparse_eigs_dense_internal.c` source added consistently to
  `Makefile`, `CMakeLists.txt`, and
  `build-metadata/library_sources.txt`.
- Verified CMake `compile_commands.json` includes
  `src/sparse_eigs_dense_internal.c` as a library compile unit.
- Captured Day 12 metrics:
  - `src/sparse_eigs.c`: 1412 lines;
  - `src/sparse_eigs_dense_internal.c`: 129 lines;
  - `tests/test_qr.c`: 3194 lines;
  - `make_qr_exact_rhs` call sites: 7;
  - public/header diffs: 0;
  - helper-target changes: 0;
  - CTest registrations: 54.

## Day 13 Notes

- Ran the required full quality gate for Sprint 109 code, test, and
  build-system changes:
  `make format && make lint && make test`.
- Quality-gate result: passed.
- `make format` completed across source, header, test, benchmark, example, and
  public-header surfaces.
- `make lint` passed:
  - tooling build completed for bench and example binaries without execution;
  - strict `cc -fsyntax-only` warning gate completed for all 46 library
    sources;
  - `clang-tidy` completed for all 46 library sources;
  - `cppcheck` completed for 101 source/test files with
    `--error-exitcode=1`.
- `make test` passed; final output reported `All tests passed.`
- Captured changed-owner metrics:
  - `src/sparse_eigs.c`: 1412 lines;
  - `src/sparse_eigs_dense_internal.c`: 129 lines;
  - `tests/test_qr.c`: 3194 lines;
  - `CMakeLists.txt`: 435 lines;
  - `Makefile`: 972 lines;
  - `build-metadata/library_sources.txt`: 50 lines.
- Captured source/helper metrics:
  - library sources: 46;
  - `s21_dense_sym_jacobi` implementation owners: 1;
  - `s21_dense_sym_jacobi` public/header declarations: 0;
  - `make_qr_exact_rhs` definitions: 1;
  - `make_qr_exact_rhs` call sites: 7;
  - helper-target changes: 0;
  - Day 12 CTest registrations: 54;
  - public/header diffs: 0.
- Confirmed header drift check remains empty:
  `git diff --name-only -- include src/*.h tests/*.h`.
- Recorded no Day 13 validation gaps.
- Deferred work remains residual planning, not failed validation:
  matrix-shell Matrix Market source split, behavior-sensitive eigensolver
  movements, QR sequential RHS helper follow-through, LDLT CSC external
  dense-reference oracle cleanup, per-solver iterative exact-RHS cleanup, and
  SVD storage-layout proof-loop cleanup.

## Day 14 Notes

- Reconciled Sprint 109 outcomes against all seven project-plan items.
- Marked completed:
  - residual debt intake and dependency ordering;
  - dense Jacobi source-boundary review;
  - dense Jacobi extraction into `src/sparse_eigs_dense_internal.c`;
  - behavior-sensitive eigensolver no-move audit;
  - Matrix Market future-owner contract and no-move validation;
  - QR exact-RHS giant-test cleanup;
  - focused integration, full validation, metrics, and residual closeout.
- Published the no-duplicate completed-work list for downstream sprints:
  dense Jacobi extraction and source-list registration, behavior-sensitive
  eigensolver no-move audits, Matrix Market future-owner selection, QR
  exact-RHS cleanup, Day 12 focused integration, and Day 13 full quality gate.
- Confirmed dense Jacobi final status:
  completed, private `.c` implementation owner, no public/header/install
  surface drift.
- Confirmed eigensolver behavior-sensitive final status:
  deferred for future dedicated owner contracts; grow-m, refinement, shared
  kernels, dispatch/defaults, handle/workspace, and shift-invert remain
  no-move in Sprint 109.
- Confirmed matrix-shell final status:
  `src/sparse_matrix_io.c` is the future Matrix Market owner candidate, but
  code movement is deferred behind private builder ownership for
  `SparseBuildEntry` and `sparse_matrix_build_from_entries`.
- Confirmed giant-test final status:
  one bounded QR cleanup completed, with QR sequential RHS, LDLT CSC external
  oracle, iterative exact-RHS, and SVD proof-loop cleanup deferred.
- Published the dependency-ordered residual queue:
  matrix builder ownership, Matrix Market source split, behavior-specific
  eigensolver owner validation, QR sequential RHS cleanup, LDLT CSC oracle
  cleanup, iterative per-solver exact-RHS cleanup, and SVD proof-loop cleanup.
- Confirmed Sprint 109 validation closeout:
  Day 5 focused eigensolver validation passed, Day 9 matrix no-move validation
  passed, Day 11 QR validation passed, Day 12 focused Make/CMake integration
  passed, Day 12 CTest registration remained 54, and Day 13
  `make format && make lint && make test` passed.
- Captured retrospective-ready takeaway:
  Sprint 109 delivered one low-risk private source extraction and one bounded
  proof-owner test cleanup while deferring broader behavior owners with
  explicit no-go conditions.
- Day 14 made no code, header, or build-system changes.
