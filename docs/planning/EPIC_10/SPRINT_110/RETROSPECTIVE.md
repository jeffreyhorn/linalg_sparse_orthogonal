# Sprint 110 Retrospective

**Sprint:** 110 - Residual Matrix I/O, Behavior Owners & Proof-Owner Follow-Through
**Duration:** 14 days (Days 1-14 landed on branch `sprint-110`)
**Status:** Complete

## Definition of Done Checklist

- [x] Sprint 110 started from Sprint 109 residual deferred debt and explicitly
      excluded completed Sprint 109 dense Jacobi, Matrix Market owner
      selection, QR exact-RHS cleanup, and validation closeout work.
- [x] Matrix builder ownership was decided before Matrix Market load/save
      movement began.
- [x] `SparseBuildEntry` and `sparse_matrix_build_from_entries` received a
      private implementation owner in `src/sparse_matrix_build_internal.c`.
- [x] Matrix Market load/save moved into `src/sparse_matrix_io.c` while public
      declarations remained in `include/sparse_matrix.h`.
- [x] `Makefile`, `CMakeLists.txt`, and
      `build-metadata/library_sources.txt` were updated consistently for the
      new private source owners.
- [x] focused Matrix Market and loaded-matrix solver-smoke validation passed.
- [x] one behavior-sensitive eigensolver owner was selected and validated as a
      no-move contract without moving public handle/workspace code.
- [x] one bounded iterative proof-owner cleanup landed in
      `tests/test_iterative.c` for CG exact-RHS setup while preserving solver
      options, residuals, convergence checks, and printed evidence at call
      sites.
- [x] one bounded SVD setup cleanup landed in `tests/test_svd.c` for the
      repeated rank-deficient 5x4 matrix while preserving rank and QR proof
      values at call sites.
- [x] no public API, install-header, helper-target, or reviewed CTest drift was
      introduced.
- [x] final validation passed:
  - `make source-list-check`
  - `make format && make lint && make test`
  - `make quality-review-cmake-compile`
  - `git diff --check`
  - trailing-whitespace scans over Sprint 110 docs and touched C/H files
- [x] downstream residuals are dependency-ordered for Sprints 111, 112, 113,
      or future maintainability work.

## What Went Well

1. **The Matrix builder prerequisite was resolved before file-I/O movement.**
   Sprint 110 did not move Matrix Market code while the shared builder still
   lived as a static helper inside the central matrix shell. The sprint first
   audited builder behavior, selected a private builder owner, and then moved
   Matrix Market load/save behind a private Matrix I/O source.

2. **The Matrix Market split reduced the central matrix shell without changing
   public API.**
   `src/sparse_matrix.c` dropped from 1,359 lines on `master` to 1,053 lines
   after moving builder and Matrix Market responsibilities into
   `src/sparse_matrix_build_internal.c` and `src/sparse_matrix_io.c`. Public
   Matrix Market declarations stayed in `include/sparse_matrix.h`.

3. **Build-system parity was kept explicit.**
   The new private source files were added consistently to the Makefile,
   CMake, and source-list metadata. The branch validated this with
   `make source-list-check`, focused Make tests, reviewed CMake compile
   parity, and CTest registration checks.

4. **Behavior-sensitive eigensolver work stayed honest.**
   The public handle/workspace bridge was selected for validation, but no code
   moved. The sprint documented why defaults, dispatch, grow-m, refinement,
   shift-invert, shared Lanczos kernels, and public handle/workspace source
   movement still require owner-specific proof before any future split.

5. **Proof-owner cleanup stayed local and readable.**
   The iterative cleanup used a CG-specific setup helper and left solver calls,
   options, residual thresholds, independent residual recomputation,
   convergence checks, and prints visible. The SVD cleanup extracted only the
   repeated rank-deficient matrix setup and left SVD rank counting, QR rank
   comparison, expected rank, and cleanup visible.

6. **Validation matched the changed surface.**
   Because the sprint touched implementation `.c`, private `.h`, test `.c`,
   build-system, source-list, and documentation files, it ran the full quality
   gate plus CMake parity and source-list checks before closeout.

## What Didn't Go Well

1. **The proof-owner test files remain large.**
   The CG and SVD helpers improved repeated setup, but `tests/test_iterative.c`
   and `tests/test_svd.c` remain large proof owners. The sprint correctly
   avoided broad helpers, but that means cleanup is incremental by design.

2. **Most eigensolver source movement remains deferred.**
   The handle/workspace bridge was validated, but the surrounding eigensolver
   behavior is still too coupled to defaults, backend dispatch, refinement,
   shift-invert, and workspace lifecycle to split safely without more direct
   proof.

3. **Documentation needs a careful follow-up pass.**
   The Matrix I/O split is private implementation work. Sprint 111 still needs
   to update user-facing documentation in terms of public behavior rather than
   internal source ownership, avoiding claims about a public Matrix I/O module
   or public builder API.

4. **Validation remained expensive.**
   The branch changed implementation, tests, build metadata, and source lists,
   so full format/lint/test plus CMake compile parity was the right bar. That
   cost is expected, but it reinforces that future source splits should stay
   narrow and evidence-driven.

## Final Metrics

### Validation

| Metric | Sprint 110 close state |
|---|---:|
| source-list parity | `make source-list-check` passed with 48 library sources |
| focused Matrix Market Make tests | `test_sparse_matrix`, `test_sparse_io`, `test_csr`, `test_integration`, `test_suitesparse`, and `test_qr` passed |
| focused eigensolver validation | `test_eigs`, `test_eigs_thick_restart`, `test_eigs_lobpcg`, and `test_sprint29_integration` passed |
| focused iterative validation | `test_iterative` passed with 80 tests |
| focused SVD validation | `test_svd` passed with 98 tests |
| CTest registration | 54 tests |
| Makefile/CMake test-count parity | 54 Makefile tests and 54 CMake tests |
| reviewed CMake compile path | `make quality-review-cmake-compile` passed |
| full branch-level gate | `make format && make lint && make test` passed |
| diff hygiene | `git diff --check` passed |
| trailing-whitespace scan | passed on Sprint 110 docs and touched C/H files |
| public/install header drift | 0 files |
| helper-target drift | 0 targets |

### Source and Test Owner Movement

| owner | Sprint 110 baseline | final | delta |
|---|---:|---:|---:|
| `src/sparse_matrix.c` | 1,359 | 1,053 | -306 |
| `src/sparse_matrix_build_internal.c` | 0 | 111 | +111 |
| `src/sparse_matrix_io.c` | 0 | 198 | +198 |
| `src/sparse_matrix_internal.h` | 251 | 267 | +16 |
| `tests/test_iterative.c` | 2,849 | 2,908 | +59 |
| `tests/test_svd.c` | 2,890 | 2,893 | +3 |

### Build and Review Surfaces

| surface | Sprint 110 close state |
|---|---:|
| library sources | 48 |
| public/install headers changed | 0 |
| private headers changed | 1 |
| implementation source files added | 2 |
| implementation source files reduced | 1 central matrix shell |
| Make/CMake/source-list membership changes | 2 private sources added consistently |
| reviewed test registration changes | 0 |
| new compiled helper targets | 0 |

### Sprint 110 Artifact Package

| Metric | Sprint 110 close state |
|---|---:|
| artifact files under `SPRINT_110/artifacts/` | 14 |
| planning and working-note files | 2 |
| retrospective files | 1 |

Notes:

- scope, decision, and source-boundary artifacts:
  - `day1-residual-debt-intake.md`
  - `day2-matrix-builder-ownership-audit.md`
  - `day3-matrix-builder-ownership-decision.md`
  - `day4-matrix-market-boundary-plan.md`
- implementation and validation artifacts:
  - `day5-matrix-market-source-split-follow-through.md`
  - `day6-matrix-market-focused-validation.md`
  - `day13-integrated-validation-and-metrics.md`
- behavior-owner and proof-owner artifacts:
  - `day7-eigensolver-behavior-owner-selection.md`
  - `day8-eigensolver-handle-workspace-validation.md`
  - `day9-proof-owner-boundary-selection.md`
  - `day10-iterative-cg-proof-owner-cleanup.md`
  - `day11-svd-proof-loop-boundary.md`
  - `day12-svd-proof-loop-cleanup.md`
- closeout artifact:
  - `day14-closeout-and-residual-handoff.md`

## Residual Deferred Debt

Most important carry-forward work:

- Sprint 111 should update user-facing documentation to describe Matrix Market
  load/save behavior as public API while avoiding claims about a public Matrix
  I/O module or public builder API.
- Sprint 112 can use Sprint 110's no-public-header-drift result as package and
  platform baseline evidence, but should not infer shared-library/ABI support
  or expanded Windows coverage from this sprint alone.
- Eigensolver behavior-sensitive movement remains deferred. Defaults and
  option validation, backend dispatch, grow-m sizing and retry behavior,
  refinement defaults and budgets, shift-invert setup, shared Lanczos kernels,
  and public handle/workspace source movement all need direct owner-specific
  tests before any future split.
- Direct and iterative proof-owner cleanup remains open for QR sequential RHS
  setup, LDLT CSC external dense-reference oracle cleanup, CG
  preconditioner-specific exact-RHS setup, GMRES exact-RHS setup, BiCGSTAB
  exact-RHS setup, and MINRES exact-RHS setup.
- SVD proof-owner cleanup remains open for reconstruction helper movement,
  U/Vt orthogonality helper movement, Moore-Penrose helper extraction, dense
  and sparse low-rank proof-loop cleanup, partial-SVD vector/residual cleanup,
  and condition-number proof cleanup.

Still consciously constrained rather than silently solved:

- no public API or install-header change;
- no shared-library/ABI or platform-support expansion claim;
- no public Matrix I/O or public builder API claim;
- no behavior-heavy eigensolver source split;
- no broad cross-solver proof helper;
- no broad SVD proof abstraction;
- no helper-target or reviewed test-count change.

Not carried forward as unresolved Sprint 110 debt:

- Sprint 109 residual intake and duplicate-work exclusion;
- Matrix builder ownership decision;
- Matrix builder private source implementation;
- Matrix Market private source split;
- Matrix Market focused validation and solver-smoke checks;
- eigensolver handle/workspace validation no-move contract;
- iterative CG exact-RHS setup cleanup;
- SVD rank-deficient setup cleanup;
- integrated validation, CMake parity, metrics, and closeout handoff.

## Key Deliverables

- [PLAN.md](./PLAN.md)
- [WORKING_NOTES.md](./WORKING_NOTES.md)
- [day1-residual-debt-intake.md](./artifacts/day1-residual-debt-intake.md)
- [day2-matrix-builder-ownership-audit.md](./artifacts/day2-matrix-builder-ownership-audit.md)
- [day3-matrix-builder-ownership-decision.md](./artifacts/day3-matrix-builder-ownership-decision.md)
- [day4-matrix-market-boundary-plan.md](./artifacts/day4-matrix-market-boundary-plan.md)
- [day5-matrix-market-source-split-follow-through.md](./artifacts/day5-matrix-market-source-split-follow-through.md)
- [day6-matrix-market-focused-validation.md](./artifacts/day6-matrix-market-focused-validation.md)
- [day7-eigensolver-behavior-owner-selection.md](./artifacts/day7-eigensolver-behavior-owner-selection.md)
- [day8-eigensolver-handle-workspace-validation.md](./artifacts/day8-eigensolver-handle-workspace-validation.md)
- [day9-proof-owner-boundary-selection.md](./artifacts/day9-proof-owner-boundary-selection.md)
- [day10-iterative-cg-proof-owner-cleanup.md](./artifacts/day10-iterative-cg-proof-owner-cleanup.md)
- [day11-svd-proof-loop-boundary.md](./artifacts/day11-svd-proof-loop-boundary.md)
- [day12-svd-proof-loop-cleanup.md](./artifacts/day12-svd-proof-loop-cleanup.md)
- [day13-integrated-validation-and-metrics.md](./artifacts/day13-integrated-validation-and-metrics.md)
- [day14-closeout-and-residual-handoff.md](./artifacts/day14-closeout-and-residual-handoff.md)
