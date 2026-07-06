# Sprint 109 Retrospective

**Sprint:** 109 - Residual Source Boundary & Proof-Owner Closeout
**Duration:** 14 days (Days 1-14 landed on branch `sprint-109`)
**Status:** Complete

## Definition of Done Checklist

- [x] Sprint 109 started from Sprint 108 residual deferred debt and explicitly
      excluded completed Sprint 108 helper work from duplicate cleanup.
- [x] eigensolver, matrix-shell, and giant-test residuals were ordered before
      implementation work started.
- [x] `s21_dense_sym_jacobi` received a private source owner in
      `src/sparse_eigs_dense_internal.c`.
- [x] `Makefile`, `CMakeLists.txt`, and
      `build-metadata/library_sources.txt` were updated in matching order for
      the new private source.
- [x] broader eigensolver grow-m, refinement, shared-kernel, dispatch,
      defaults, handle/workspace, and shift-invert movement was audited and
      deferred with explicit no-go conditions.
- [x] `src/sparse_matrix.c` received a Matrix Market future-owner contract
      without moving matrix-shell code.
- [x] `tests/test_qr.c` received one bounded exact-RHS setup helper while
      keeping solve, refine, rank, residual, reconstruction, and QR-vs-LU proof
      logic visible at call sites.
- [x] no public API, install-header, private header, shared test-helper header,
      helper target, or reviewed CTest registration drift was introduced.
- [x] source-list parity and focused Make/CMake validation passed.
- [x] final validation passed:
  - `make format && make lint && make test`
  - `git diff --check`
  - trailing-whitespace scan over `docs/planning/EPIC_10/SPRINT_109`
- [x] downstream residuals are dependency-ordered and do not duplicate
      completed Sprint 109 work.

## What Went Well

1. **The sprint converted one source-boundary candidate into code.**
   Sprint 108 identified dense Jacobi as the safest eigensolver split
   candidate. Sprint 109 completed the boundary review, added build-system
   parity, moved only `s21_dense_sym_jacobi`, and validated the direct
   thick-restart, LOBPCG, public eigensolver, and cross-feature lanes.

2. **Build-system parity stayed explicit.**
   The new private source was added consistently to `Makefile`,
   `CMakeLists.txt`, and `build-metadata/library_sources.txt`. The branch
   repeatedly validated that parity with `make source-list-check`, focused
   Make builds, focused CMake builds, generated compile-command inspection,
   and CTest registration checks.

3. **Behavior-sensitive eigensolver movement was not overclaimed.**
   Grow-m execution, refinement, shared kernels, dispatch/defaults,
   handle/workspace behavior, and shift-invert setup were documented as
   behavior owners rather than line-count cleanup. This kept the sprint from
   turning one safe extraction into a risky source split.

4. **The matrix-shell work produced a useful future contract.**
   Sprint 109 selected `src/sparse_matrix_io.c` as the future Matrix Market
   owner, but correctly deferred code movement until private ownership for
   `SparseBuildEntry` and `sparse_matrix_build_from_entries` is resolved.

5. **The giant-test cleanup remained proof-visible.**
   The QR exact-RHS helper removed repeated setup from seven call sites while
   preserving rank checks, solve/refine calls, residual labels, tolerances,
   reconstruction checks, and QR-vs-LU comparison logic inline.

6. **Validation matched the changed surface.**
   Because implementation `.c`, test `.c`, and source-list files changed, the
   sprint ran focused tests, CMake/CTest checks, source-list parity, and the
   full `make format && make lint && make test` gate.

## What Didn't Go Well

1. **Only one implementation source split was low-risk enough to land.**
   The dense Jacobi split succeeded, but the remaining eigensolver candidates
   still couple public defaults, backend selection, shift-invert factoring,
   result mutation, workspace lifecycle, or cross-backend numerical semantics.

2. **The matrix-shell split remains blocked by private builder ownership.**
   Matrix Market load/save is now a clear future owner, but movement would
   either expose a currently static builder or force unrelated copy/transpose
   movement. That prerequisite remains unresolved.

3. **The proof-owner tests remain large.**
   `tests/test_qr.c` got smaller, but QR, LDLT CSC, iterative, and SVD test
   owners still need future one-family cleanup passes. Broad helper extraction
   would still risk hiding proof intent.

4. **Validation remained expensive.**
   The branch touched implementation, tests, and source lists, so full format,
   lint, and test validation was necessary. This is the right cost for the
   change, but it reinforces that future source splits need tight scope.

## Final Metrics

### Validation

| Metric | Sprint 109 close state |
|---|---:|
| source-list parity | `make source-list-check` passed with 46 library sources |
| focused eigensolver Make tests | `test_eigs`, `test_eigs_thick_restart`, `test_eigs_lobpcg`, `test_sprint29_integration` passed |
| focused QR Make test | `test_qr` passed, 73 tests |
| focused CMake/CTest validation | 5 focused tests passed |
| CTest registration | 54 tests |
| full branch-level gate | `make format && make lint && make test` passed |
| diff hygiene | `git diff --check` passed |
| trailing-whitespace scan | passed on `docs/planning/EPIC_10/SPRINT_109` |
| public/install header drift | 0 files |
| helper-target drift | 0 targets |

### Source and Test Owner Movement

| owner | Sprint 109 baseline | final | delta |
|---|---:|---:|---:|
| `src/sparse_eigs.c` | 1538 | 1412 | -126 |
| `src/sparse_eigs_dense_internal.c` | 0 | 129 | +129 |
| `tests/test_qr.c` | 3210 | 3194 | -16 |

### Build and Review Surfaces

| surface | Sprint 109 close state |
|---|---:|
| library sources | 46 |
| public/install headers changed | 0 |
| private headers changed | 0 |
| shared test helper headers changed | 0 |
| Make/CMake/source-list membership changes | 1 private source added consistently |
| reviewed test registration changes | 0 |
| new compiled helper targets | 0 |
| `s21_dense_sym_jacobi` implementation owners | 1 |
| `make_qr_exact_rhs` call sites | 7 |

### Sprint 109 Artifact Package

| Metric | Sprint 109 close state |
|---|---:|
| artifact files under `SPRINT_109/artifacts/` | 14 |
| planning and working-note files | 2 |
| retrospective files | 1 |

Notes:

- scope, boundary, and dependency artifacts:
  - `day1-residual-debt-intake.md`
  - `day2-dense-jacobi-source-boundary.md`
  - `day3-source-list-parity-validation-prep.md`
  - `day6-growm-refinement-shared-kernel-audit.md`
  - `day7-dispatch-handle-shift-invert-audit.md`
  - `day8-matrix-shell-candidate-boundary-contract.md`
  - `day10-giant-test-cleanup-boundary.md`
- implementation, validation, and closeout artifacts:
  - `day4-dense-jacobi-extraction.md`
  - `day5-dense-jacobi-cross-lane-validation.md`
  - `day9-matrix-shell-validation-no-move.md`
  - `day11-giant-test-cleanup-follow-through.md`
  - `day12-focused-integration-drift-check.md`
  - `day13-full-quality-gate-metrics.md`
  - `day14-sprint-closeout-residual-queue.md`

## Residual Deferred Debt

Most important carry-forward work:

- Matrix builder ownership must be resolved before any Matrix Market source
  split. The future decision is whether `SparseBuildEntry` and
  `sparse_matrix_build_from_entries` become a private builder source or remain
  in the central matrix shell.
- Matrix Market load/save can move to `src/sparse_matrix_io.c` only after the
  builder prerequisite is resolved and focused matrix tests plus solver-smoke
  fixture lanes are planned.
- Eigensolver behavior-sensitive movement remains deferred. Any future split
  must select one owner at a time and validate defaults, dispatch, workspace,
  refinement, or shift-invert semantics directly.
- QR sequential RHS cleanup remains possible, but only for setup that does not
  hide least-squares or refinement proof values.
- LDLT CSC external dense-reference oracle cleanup remains a dedicated
  oracle-lane review because it couples external Python references, Windows
  skips, LDLT factorization, and dense solve comparison.
- Iterative exact-RHS cleanup should be split by solver family rather than
  introduced as one broad CG/GMRES/BiCGSTAB/MINRES helper.
- SVD proof-loop cleanup should preserve storage-layout, stride, rank,
  orthogonality, and reconstruction proof at call sites.

Still consciously constrained rather than silently solved:

- no public API or install-header change;
- no private header or shared test-helper header change;
- no new compiled helper target;
- no reviewed test-count change;
- no matrix-shell source movement;
- no behavior-heavy eigensolver movement beyond dense Jacobi;
- no broad giant-test proof abstraction.

Not carried forward as unresolved Sprint 109 debt:

- Sprint 108 duplicate-work exclusion list;
- dense Jacobi boundary review;
- dense Jacobi extraction and source-list registration;
- focused dense Jacobi Make and CMake validation;
- eigensolver grow-m/refinement/shared-kernel no-move audit;
- eigensolver dispatch/defaults/handle/workspace/shift-invert no-move audit;
- Matrix Market future-owner selection;
- QR exact-RHS helper cleanup;
- focused integration, full validation, metrics, and drift checks.

## Key Deliverables

- [PLAN.md](./PLAN.md)
- [WORKING_NOTES.md](./WORKING_NOTES.md)
- [day1-residual-debt-intake.md](./artifacts/day1-residual-debt-intake.md)
- [day2-dense-jacobi-source-boundary.md](./artifacts/day2-dense-jacobi-source-boundary.md)
- [day3-source-list-parity-validation-prep.md](./artifacts/day3-source-list-parity-validation-prep.md)
- [day4-dense-jacobi-extraction.md](./artifacts/day4-dense-jacobi-extraction.md)
- [day5-dense-jacobi-cross-lane-validation.md](./artifacts/day5-dense-jacobi-cross-lane-validation.md)
- [day6-growm-refinement-shared-kernel-audit.md](./artifacts/day6-growm-refinement-shared-kernel-audit.md)
- [day7-dispatch-handle-shift-invert-audit.md](./artifacts/day7-dispatch-handle-shift-invert-audit.md)
- [day8-matrix-shell-candidate-boundary-contract.md](./artifacts/day8-matrix-shell-candidate-boundary-contract.md)
- [day9-matrix-shell-validation-no-move.md](./artifacts/day9-matrix-shell-validation-no-move.md)
- [day10-giant-test-cleanup-boundary.md](./artifacts/day10-giant-test-cleanup-boundary.md)
- [day11-giant-test-cleanup-follow-through.md](./artifacts/day11-giant-test-cleanup-follow-through.md)
- [day12-focused-integration-drift-check.md](./artifacts/day12-focused-integration-drift-check.md)
- [day13-full-quality-gate-metrics.md](./artifacts/day13-full-quality-gate-metrics.md)
- [day14-sprint-closeout-residual-queue.md](./artifacts/day14-sprint-closeout-residual-queue.md)
