# Day 14 Validation, Metrics & Sprint 108 Closeout

## Purpose

Day 14 closes Sprint 108 by validating the touched surfaces, recording final
maintainability metrics, confirming no unsupported public/support-surface drift,
and publishing the residual queue for future sprints.

## Touched-File Set

Sprint 108 changed four test sources:

- `tests/test_ldlt_csc.c`
- `tests/test_qr.c`
- `tests/test_iterative.c`
- `tests/test_svd.c`

Sprint 108 also added planning artifacts under:

- `docs/planning/EPIC_10/SPRINT_108/`

No Sprint 108 change touched:

- public headers under `include/`;
- implementation sources under `src/`;
- `Makefile`;
- `CMakeLists.txt`;
- `build-metadata/library_sources.txt`;
- helper targets;
- CTest registration;
- reviewed Windows, Linux, or macOS test-count expectations.

## Final Metrics

| Owner | HEAD Lines | Final Lines | Net Line Delta | Net Diff | Sprint 108 Outcome |
|---|---:|---:|---:|---:|---|
| `tests/test_ldlt_csc.c` | 3,887 | 3,896 | +9 | +14 / -5 | Added one bounded oracle/helper follow-through while preserving direct LDLT CSC proof intent. |
| `tests/test_qr.c` | 3,213 | 3,210 | -3 | +23 / -26 | Consolidated a small QR fixture path without hiding solve, rank, reconstruction, or refinement proof logic. |
| `tests/test_iterative.c` | 2,828 | 2,849 | +21 | +45 / -24 | Added bounded iterative convergence cleanup while preserving options, restart values, preconditioner setup, and comparisons. |
| `tests/test_svd.c` | 2,897 | 2,896 | -1 | +26 / -27 | Added a dedicated full-SVD fixture helper for approved call sites only. |
| `src/sparse_eigs.c` | 1,538 | 1,538 | 0 | 0 | No source split; future extraction plan recorded. |
| `src/sparse_matrix.c` | 1,359 | 1,359 | 0 | 0 | No source split; public-behavior review recorded. |

## Artifact Inventory

Sprint 108 produced these closeout-relevant artifacts:

- `day1-carry-forward-intake.md`
- `day2-residual-proof-owner-boundary-refresh.md`
- `day3-ldlt-csc-oracle-boundary.md`
- `day4-ldlt-csc-helper-follow-through.md`
- `day5-qr-residual-fixture-boundary.md`
- `day6-qr-fixture-follow-through.md`
- `day7-iterative-convergence-boundary.md`
- `day8-iterative-convergence-cleanup.md`
- `day9-svd-validation-lane-boundary.md`
- `day10-svd-oracle-reconstruction-cleanup.md`
- `day11-eigensolver-source-feasibility-boundary.md`
- `day12-eigensolver-feasibility-closeout.md`
- `day13-matrix-shell-public-behavior-review.md`
- `day14-validation-metrics-closeout.md`

## Reviewed Drift Check

Sprint 108 introduced no public or reviewed support-surface drift:

- no public API change;
- no installed-header change;
- no source-list membership change;
- no Makefile target change;
- no CMake target/source change;
- no helper-target change;
- no CTest registration change;
- no reviewed test-count change.

Because build membership did not change, `make source-list-check` and reviewed
CMake registration/count checks are not required for Sprint 108 closeout.

## Required Validation

Because Sprint 108 modified `.c` test files, the required branch gate is:

```sh
make format
make lint
make test
git diff --check
rg -n "[ \t]+$" docs/planning/EPIC_10/SPRINT_108
```

## Validation Results

Day 14 validation passed:

- `make format && make lint && make test`
- `git diff --check`
- `rg -n "[ \t]+$" docs/planning/EPIC_10/SPRINT_108`

Post-format file status remained bounded to the expected tracked test files
and Sprint 108 planning artifacts. Formatting did not introduce public-header,
implementation-source, build-system, source-list, helper-target, or CTest
registration drift.

## Residual Queue

Future work should remain ordered and non-duplicative:

1. Eigensolver dense Jacobi private source candidate
   - Candidate: move only `s21_dense_sym_jacobi` into a private dense spectral
     helper source.
   - Gate: Make/CMake/manifest parity, `make source-list-check`, focused
     `test_eigs`, `test_eigs_thick_restart`, `test_eigs_lobpcg`, and
     `test_sprint29_integration`.
2. Eigensolver grow-m/refinement/dispatch/handle/shared-kernel audit
   - Candidate: behavior review before any movement.
   - Gate: residual, shift-invert, backend-selection, handle-workspace, and
     cross-backend numerical evidence.
3. Matrix-shell source boundary project
   - Candidate: one public-behavior owner at a time, such as Matrix Market,
     bulk entry construction, arithmetic/matvec, or factor compatibility.
   - Gate: private-header dependency plan, Make/CMake/manifest parity, focused
     public behavior tests, and solver smoke tests for touched semantics.
4. Additional giant-test proof-owner cleanup
   - Candidate: future bounded helper families in remaining large tests.
   - Gate: one helper family per change, visible proof logic retained at
     call sites, and full quality validation for code changes.

## Retrospective Input

Sprint 108 completed the intended residual follow-through without broadening
public support surfaces. The sprint made bounded proof-owner cleanup in LDLT
CSC, QR, iterative, and SVD tests, then converted eigensolver and matrix-shell
source work into explicit future guardrail contracts.

The primary residual is source-boundary work that should not be treated as
line-count cleanup. Eigensolver and matrix-shell movement both require
behavior-first validation and source-list parity before code moves.

## Completion Criteria Status

- Required validation commands are identified for the actual touched-file set.
- Final proof-owner metrics are recorded.
- Public API, install-header, helper-target, source-list, and CTest drift are
  explicitly ruled out.
- Residual work is ordered, non-duplicative, and gated by concrete evidence.
