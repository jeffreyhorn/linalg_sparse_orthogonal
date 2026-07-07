# Sprint 113 Working Notes

## Sprint Goal

Sprint 113 resolves Sprint 110's remaining behavior-sensitive eigensolver and
proof-owner residuals in bounded, evidence-first batches before final Epic 10
integration and closeout. The sprint must not widen public API, install-header,
package/platform, or reviewed CTest claims unless direct evidence justifies the
change.

## Starting Constraints

- Do not repeat Sprint 110 Matrix builder or Matrix Market private source
  movement.
- Do not repeat Sprint 110 iterative CG exact-RHS setup cleanup.
- Do not repeat Sprint 110 SVD rank-deficient 5x4 setup cleanup.
- Do not repeat Sprint 110 public eigensolver handle/workspace validation
  no-move contract.
- Do not infer shared-library/ABI, platform-support, install-validation, or
  package-manager support from Sprint 113 behavior-owner work.
- Keep proof values visible at test call sites unless a boundary artifact
  proves a helper can hide setup without hiding behavior.
- If `.c`, `.h`, build-system, source-list, or test registration changes,
  run the strongest checks required for the touched surface.

## Completed Work Excluded From Sprint 113 Scope

| Completed work | Source evidence | Sprint 113 handling |
|---|---|---|
| Matrix builder ownership decision | Sprint 110 Day 3 and retrospective | Excluded. |
| Matrix builder private source implementation | Sprint 110 Day 3 and retrospective | Excluded. |
| Matrix Market private source split | Sprint 110 Days 4-6 and retrospective | Excluded. |
| Matrix Market focused validation and solver-smoke checks | Sprint 110 Day 6 and retrospective | Excluded. |
| Eigensolver public handle/workspace validation | Sprint 110 Days 7-8 and retrospective | Excluded as completed no-move contract. |
| Iterative CG exact-RHS allocation/setup cleanup | Sprint 110 Days 9-10 and retrospective | Excluded. |
| SVD rank-deficient 5x4 setup cleanup | Sprint 110 Days 11-12 and retrospective | Excluded. |
| Sprint 112 package/platform support truth | Sprint 112 retrospective | Use as boundary; do not widen claims. |

## Remaining Eigensolver Behavior-Owner Candidates

| Candidate | Primary source / tests | Dependency notes | Initial disposition |
|---|---|---|---|
| Defaults and option validation | `src/sparse_eigs.c`, `include/sparse_eigs.h`, `tests/test_eigs.c` | Public behavior around `opts == NULL`, invalid options, backend enums, and error codes. | Candidate for Day 2 comparison. |
| Backend dispatch | `src/sparse_eigs.c`, `tests/test_eigs.c`, `tests/test_eigs_thick_restart.c`, `tests/test_eigs_lobpcg.c` | Requires proof of AUTO priority, `backend_used`, result propagation, and error propagation. | Candidate for Day 2 comparison. |
| Grow-m sizing and retry behavior | `src/sparse_eigs.c`, `src/sparse_eigs_workspace_internal.c`, `tests/test_eigs.c` | Coupled to capacity growth, retry behavior, progress callbacks, partial results, peak basis, and residuals. | Candidate for Day 2 comparison. |
| Refinement defaults and budgets | `src/sparse_eigs.c`, `tests/test_eigs.c`, `tests/test_sprint29_integration.c` | Mutates returned eigenpairs and interacts with cancellation and backend return codes. | Candidate for Day 2 comparison. |
| Shift-invert setup | `src/sparse_eigs.c`, direct solver paths, eigensolver tests | Coupled to `NEAREST_SIGMA`, singular shifts, LDLT path reporting, and inverse Ritz conversion. | Candidate for Day 2 comparison. |
| Shared Lanczos kernels | `src/sparse_eigs.c`, `tests/test_eigs.c`, `tests/test_eigs_thick_restart.c` | Cross-backend behavior for ordering, residual scale, reorthogonalization, and vector lifting. | Candidate for Day 2 comparison. |
| Public handle/workspace source movement | `include/sparse_eigs.h`, `src/sparse_eigs.c`, `src/sparse_eigs_workspace_internal.c` | Validation no-move contract is already complete; movement still needs broader proof. | Defer unless Day 2 finds a narrower non-duplicate owner. |

## Remaining Direct/Iterative Proof-Owner Candidates

| Candidate | Primary file | Proof values at risk | Initial disposition |
|---|---|---|---|
| QR sequential RHS setup | `tests/test_qr.c` | least-squares residuals, refinement before/after residuals, literal RHS values. | Candidate for Day 7, but avoid repeating Sprint 109 exact-RHS helper work. |
| LDLT CSC external dense-reference oracle cleanup | `tests/test_ldlt_csc.c` | dense oracle comparison, Windows skip behavior, permutation handling, solve residuals. | Candidate for Day 7; requires dedicated oracle-lane review. |
| CG preconditioner-specific exact-RHS setup | `tests/test_iterative.c` | preconditioner construction, residual norms, iteration comparisons. | Candidate for Day 7; do not repeat Sprint 110 generic CG setup helper. |
| GMRES exact-RHS setup | `tests/test_iterative.c` | restart settings, convergence status, residual norms, lucky-breakdown behavior. | Candidate for Day 7. |
| BiCGSTAB exact-RHS setup | `tests/test_iterative.c` | breakdown behavior, residual norms, convergence status. | Candidate for Day 7. |
| MINRES exact-RHS setup | `tests/test_iterative.c` | symmetry assumptions, preconditioner behavior, residual norms, convergence status. | Candidate for Day 7. |

## Remaining SVD Proof-Owner Candidates

| Candidate | Primary file | Proof values at risk | Initial disposition |
|---|---|---|---|
| Reconstruction helper movement | `tests/test_svd.c` | reconstruction residuals and matrix dimension/layout checks. | Candidate for Day 9 only after proof-boundary refresh. |
| U/Vt orthogonality helper movement | `tests/test_svd.c` | dot-product orthogonality thresholds and layout loops. | Candidate for Day 9 only after proof-boundary refresh. |
| Moore-Penrose helper extraction | `tests/test_svd.c` | expected inverse entries and Moore-Penrose products. | Candidate for Day 9 only after proof-boundary refresh. |
| Dense low-rank proof-loop cleanup | `tests/test_svd.c` | retained singular-value error bounds and Frobenius residuals. | Candidate for Day 9 only after proof-boundary refresh. |
| Sparse low-rank proof-loop cleanup | `tests/test_svd.c` | dense-vs-sparse residuals, drop tolerance, corpus fixture names. | Candidate for Day 9 only after proof-boundary refresh. |
| Partial-SVD vector/residual cleanup | `tests/test_svd.c`, `tests/test_svd_partial_helpers.h` | vector orthogonality and `A*v ~= sigma*u` residuals. | Candidate for Day 9 only after proof-boundary refresh. |
| Condition-number proof cleanup | `tests/test_svd.c` | expected finite/infinite condition values and rectangular interpretation. | Candidate for Day 9 only after proof-boundary refresh. |

## Day-Level Ownership

| Day | Planned Focus | Project Plan Item |
|---:|---|---|
| 1 | Residual intake, duplicate-work exclusion, candidate inventories. | Item 1 |
| 2 | Eigensolver behavior-owner selection. | Item 2 |
| 3 | Eigensolver behavior proof design. | Item 2 |
| 4 | Eigensolver behavior proof implementation. | Item 2 |
| 5 | Eigensolver movement/no-move decision. | Item 3 |
| 6 | Eigensolver movement or no-move contract execution. | Item 3 |
| 7 | Direct/iterative proof-owner boundary selection. | Item 4 |
| 8 | Direct/iterative proof-owner cleanup. | Item 4 |
| 9 | SVD proof-boundary refresh. | Item 5 |
| 10 | SVD proof-owner cleanup. | Item 5 |
| 11 | Proof-owner metrics and non-claims. | Item 6 |
| 12 | Integrated validation planning. | Item 7 |
| 13 | Integrated validation execution. | Item 7 |
| 14 | Sprint closeout and handoff. | Item 7 |

## Validation Expectations

| Touched Surface | Required Checks |
|---|---|
| Documentation only | `git diff --check`; trailing-whitespace scan over touched docs; local relative Markdown link check when links change. |
| Test `.c` files | Focused test binary build/run; `make format && make lint && make test`; `git diff --check`. |
| Source `.c` or private `.h` files | Focused affected tests; source-list/build checks as applicable; `make format && make lint && make test`. |
| Public headers or install headers | Public/install-header drift review; `make format && make lint && make test`; install/package checks if package semantics change. |
| Make/CMake/source-list metadata | `make source-list-check`; focused Make/CMake build or CTest checks; full quality gate if code changed. |
| CTest registration | `ctest -N` through the relevant CMake build path; document expected count changes explicitly. |

## Day 1 Notes

- Created Sprint 113 working notes and artifact directory.
- Re-read the Sprint 113 section of `docs/planning/EPIC_10/PROJECT_PLAN.md`.
- Re-read Sprint 110 residual deferred debt and closeout notes.
- Reviewed Sprint 110 behavior-owner and proof-owner artifacts:
  - `day7-eigensolver-behavior-owner-selection.md`;
  - `day8-eigensolver-handle-workspace-validation.md`;
  - `day9-proof-owner-boundary-selection.md`;
  - `day11-svd-proof-loop-boundary.md`;
  - Sprint 110 retrospective residual queue.
- Explicitly excluded completed work:
  - Matrix builder ownership and private source implementation;
  - Matrix Market private source split and focused validation;
  - eigensolver public handle/workspace validation no-move contract;
  - iterative CG exact-RHS allocation/setup cleanup;
  - SVD rank-deficient 5x4 setup cleanup.
- Inventoried remaining eigensolver behavior-owner candidates:
  - defaults and option validation;
  - backend dispatch;
  - grow-m sizing and retry behavior;
  - refinement defaults and budgets;
  - shift-invert setup;
  - shared Lanczos kernels;
  - public handle/workspace source movement only as a deferred movement topic.
- Inventoried remaining direct/iterative proof-owner candidates:
  - QR sequential RHS setup;
  - LDLT CSC external dense-reference oracle cleanup;
  - CG preconditioner-specific exact-RHS setup;
  - GMRES exact-RHS setup;
  - BiCGSTAB exact-RHS setup;
  - MINRES exact-RHS setup.
- Inventoried remaining SVD proof-owner candidates:
  - reconstruction helper movement;
  - U/Vt orthogonality helper movement;
  - Moore-Penrose helper extraction;
  - dense and sparse low-rank proof-loop cleanup;
  - partial-SVD vector/residual cleanup;
  - condition-number proof cleanup.
- Added Day 1 artifact:
  - `artifacts/day1-residual-intake-and-boundary.md`.

## Day 2 Notes

- Reviewed Day 1 eigensolver behavior-owner candidates against current source
  and tests:
  - `src/sparse_eigs.c`;
  - `src/sparse_eigs_workspace_internal.c`;
  - `src/sparse_eigs_internal.h`;
  - `include/sparse_eigs.h`;
  - `tests/test_eigs.c`;
  - `tests/test_eigs_thick_restart.c`;
  - `tests/test_eigs_lobpcg.c`;
  - `tests/test_sprint29_integration.c`.
- Confirmed existing coverage already strongly pins several candidates:
  - backend dispatch has AUTO/explicit coverage across LOBPCG and
    thick-restart tests;
  - shift-invert has diagonal, indefinite, singular, eigenvector,
    wide-spectrum, CSC/linked-list threshold, thick-restart, and LOBPCG
    nearest-sigma coverage;
  - refinement defaults and budgets have focused tests and Sprint 29
    integration coverage;
  - public handle/workspace validation was completed in Sprint 110 and remains
    excluded from duplicate Sprint 113 work.
- Selected **grow-m sizing and retry behavior** as the Sprint 113 eigensolver
  behavior owner for direct proof.
- Selected owner implementation surfaces:
  - `s49_eigs_effective_max_iters`;
  - `s49_eigs_growm_capacity`;
  - `s46_run_growm_backend`;
  - grow-m workspace preparation;
  - progress callback emission at grow-m retry boundaries;
  - partial-result publication when `m_cap` is exhausted.
- Required invariants for Day 3 proof design:
  - default max-iteration budget remains bounded;
  - too-small explicit max-iteration budgets reject with `SPARSE_ERR_BADARG`;
  - grow-m capacity is clamped by `n`, `max_iterations`, and minimum basis
    requirements;
  - `peak_basis_size` reports grow-m upper-bound allocation;
  - retry growth accumulates `iterations`;
  - progress callbacks and cancellation remain clean;
  - `SPARSE_ERR_NOT_CONVERGED` partial-result behavior stays visible.
- Source movement remains blocked until Day 4 proof lands and Day 5 makes an
  evidence-backed movement/no-move decision.
- Added Day 2 artifact:
  - `artifacts/day2-eigensolver-behavior-owner-selection.md`.

## Day 3 Notes

- Located the exact grow-m behavior-owner surfaces:
  - `s49_eigs_effective_max_iters`;
  - `s49_eigs_growm_capacity`;
  - `s46_run_growm_backend`;
  - `sparse_eigs_workspace_prepare_growm`;
  - public `sparse_eigs_sym` and `sparse_eigs_sym_with_handle` observations.
- Confirmed existing tests cover adjacent behavior but do not yet provide a
  focused grow-m sizing/retry proof:
  - public handle grow-m reuse/growth exists;
  - progress plus refinement exists;
  - first-callback cancellation exists;
  - broad SuiteSparse and stability tests exercise grow-m incidentally.
- Designed five focused Day 4 tests:
  - default grow-m capacity pins `peak_basis_size`;
  - explicit `max_iterations` pins `peak_basis_size`;
  - too-small explicit `max_iterations` rejects with `SPARSE_ERR_BADARG`;
  - retry-boundary progress steps are monotonic and consistent with
    accumulated `iterations`;
  - cancellation after a nonzero retry-boundary step exits with
    `SPARSE_ERR_CANCELLED`.
- Selected deterministic fixtures:
  - diagonal matrices for capacity and validation-path tests;
  - SPD tridiagonal matrix for retry/progress tests.
- Kept proof values visible:
  - `n`;
  - `k`;
  - explicit `max_iterations`;
  - expected `peak_basis_size`;
  - expected diagonal eigenvalues;
  - progress callback step history.
- Defined Day 4 focused validation:
  - `make build/test_eigs`;
  - `build/test_eigs`;
  - plus `make build/test_sprint29_integration` and
    `build/test_sprint29_integration` only if cancellation coverage lands
    there instead of `tests/test_eigs.c`.
- Reconfirmed that Day 4 code changes will require
  `make format && make lint && make test` before sprint closeout.
- Added Day 3 artifact:
  - `artifacts/day3-eigensolver-behavior-proof-design.md`.

## Day 4 Notes

- Implemented the focused grow-m behavior proof in `tests/test_eigs.c`.
- Added local test support:
  - `build_shifted_tridiag`;
  - `growm_progress_record_t`;
  - `growm_progress_record_cb`.
- Added five grow-m owner tests:
  - `test_growm_default_capacity_pins_peak_basis_size`;
  - `test_growm_explicit_capacity_pins_peak_basis_size`;
  - `test_growm_too_small_explicit_iteration_budget_rejected`;
  - `test_growm_retry_progress_steps_accumulate_iterations`;
  - `test_growm_retry_boundary_cancellation_exits_cleanly`.
- Adjusted the explicit-capacity test from the Day 3 design:
  - the initial `n = 64`, `k = 2`, repeated-tail diagonal fixture could return
    a duplicate top Ritz value for the second pair;
  - the final test uses `k = 1`, still proving `max_iterations = 24` maps to
    `peak_basis_size = 24` without making a clustered-spectrum ordering claim.
- Focused validation passed:
  - `make build/test_eigs && build/test_eigs`;
  - `test_eigs`: `36` tests, `0` failed, `345` assertions.
- Required full quality chain passed for the `.c` test change:
  - `make format && make lint && make test`.
- No public API, install-header, helper-target, Make/CMake, source-list, or
  reviewed CTest registration drift was introduced.
- Source movement remains blocked until the Day 5 movement/no-move decision.
- Added Day 4 artifact:
  - `artifacts/day4-eigensolver-behavior-proof.md`.

## Day 5 Notes

- Reviewed Day 4 grow-m proof and the selected owner coupling.
- Decided **not** to move eigensolver source in Day 6.
- Decision rationale:
  - Day 4 proves grow-m behavior through public results and callbacks;
  - the actual owner, `s46_run_growm_backend`, remains tightly coupled to
    shared static Lanczos helpers inside `src/sparse_eigs.c`;
  - moving only `s49_eigs_effective_max_iters` and
    `s49_eigs_growm_capacity` would be a cosmetic split;
  - moving `s46_run_growm_backend` now would require exposing or relocating a
    broader helper cluster that Day 4 did not prove as a separate owner;
  - grow-m workspace preparation is already isolated in
    `src/sparse_eigs_workspace_internal.c`.
- Rejected movement options:
  - sizing-helper-only split;
  - immediate grow-m backend source split;
  - public handle grow-m preparation movement;
  - extra workspace-preparation movement.
- Defined Day 6 no-move execution:
  - publish no-move contract;
  - capture owner metrics and focused guard tests;
  - verify no source-list, Make/CMake, or CTest drift;
  - run focused `test_eigs` validation unless Day 6 changes code, in which
    case the full quality chain must run again.
- Added Day 5 artifact:
  - `artifacts/day5-eigensolver-movement-decision.md`.

## Day 6 Notes

- Executed the Day 5 no-move decision.
- Published the grow-m eigensolver no-move contract.
- Captured current owner metrics:
  - `src/sparse_eigs.c`: 1412 lines;
  - `src/sparse_eigs_workspace_internal.c`: 267 lines;
  - `src/sparse_eigs_workspace_internal.h`: 82 lines;
  - `tests/test_eigs.c`: 1758 lines.
- Recorded current grow-m owner locations:
  - `s49_eigs_effective_max_iters` at `src/sparse_eigs.c:769`;
  - `s49_eigs_growm_capacity` at `src/sparse_eigs.c:793`;
  - grow-m handle preparation branch at `src/sparse_eigs.c:838`;
  - grow-m executor at `src/sparse_eigs.c:965`;
  - executor workspace preparation call at `src/sparse_eigs.c:1029`;
  - backend dispatch into grow-m at `src/sparse_eigs.c:1223`;
  - workspace implementation at `src/sparse_eigs_workspace_internal.c:84`;
  - workspace declaration at `src/sparse_eigs_workspace_internal.h:72`.
- Preserved the no-move boundary:
  - no public API changes;
  - no install-header changes;
  - no helper-target changes;
  - no Make/CMake/source-list changes;
  - no reviewed CTest registration changes;
  - no eigensolver source movement.
- Deferred broader movement until future proof covers:
  - `lanczos_iterate_op`;
  - Ritz selection on repeated/clustered spectra;
  - Ritz vector lifting;
  - partial-result publication after `m_cap` exhaustion;
  - shift-invert grow-m conversion;
  - shared helper visibility rules.
- Added Day 6 artifact:
  - `artifacts/day6-eigensolver-no-move-contract.md`.

## Day 7 Notes

- Reviewed remaining direct/iterative proof-owner cleanup candidates:
  - QR sequential RHS setup;
  - LDLT CSC external dense-reference oracle cleanup;
  - CG preconditioner-specific exact-RHS setup;
  - GMRES exact-RHS setup;
  - BiCGSTAB exact-RHS setup;
  - MINRES exact-RHS setup.
- Re-read Sprint 110 Day 9 direct/iterative proof-owner boundary to avoid
  duplicating completed work.
- Explicitly excluded completed cleanup:
  - Sprint 109 QR exact-RHS helper work;
  - Sprint 110 generic CG exact-RHS setup helper.
- Selected **LDLT CSC external dense-reference oracle cleanup** for Day 8.
- Selection rationale:
  - one local direct-solver oracle lane;
  - repeated allocation and cleanup noise is concentrated in one helper;
  - three existing call sites can remain intact;
  - solver proof values can remain visible at the oracle boundary.
- Day 8 must keep visible:
  - fixture key;
  - fixture builder;
  - tolerance;
  - exact-RHS construction;
  - permutation and unpermutation flow;
  - `ldlt_csc_solve`;
  - dense-reference read status;
  - max-difference and residual assertions.
- Day 8 validation if `tests/test_ldlt_csc.c` changes:
  - `make build/test_ldlt_csc`;
  - `build/test_ldlt_csc`;
  - `make format && make lint && make test`;
  - `git diff --check`.
- Added Day 7 artifact:
  - `artifacts/day7-direct-iterative-proof-owner-boundary.md`.

## Day 8 Notes

- Implemented the selected LDLT CSC external dense-reference oracle cleanup in
  `tests/test_ldlt_csc.c`.
- Added a local oracle state owner and cleanup helpers:
  - `ldlt_external_dense_reference_state_t`;
  - `ldlt_external_dense_reference_state_alloc`;
  - `ldlt_external_dense_reference_state_free`.
- Preserved proof visibility for:
  - exact solution setup;
  - exact RHS construction;
  - two-pass indefinite factorization;
  - RHS permutation;
  - `ldlt_csc_solve`;
  - solution unpermutation;
  - dense-reference read status;
  - max-difference checks;
  - relative residual checks.
- Focused validation passed:
  - `make build/test_ldlt_csc && build/test_ldlt_csc`;
  - `test_ldlt_csc`: 100 tests, 0 failed, 0 skipped, 3556 assertions.
- Full required C quality chain passed:
  - `make format && make lint && make test`.
- Confirmed no public API, install-header, helper-target, Make/CMake
  source-list, or reviewed CTest registration drift.
- Added Day 8 artifact:
  - `artifacts/day8-direct-iterative-proof-owner-cleanup.md`.

## Day 9 Notes

- Reviewed remaining SVD proof-owner cleanup candidates:
  - reconstruction helper movement;
  - U/Vt orthogonality helper movement;
  - Moore-Penrose product helper extraction;
  - dense low-rank proof-loop cleanup;
  - sparse low-rank proof-loop cleanup;
  - partial-SVD vector/residual cleanup;
  - condition-number proof cleanup.
- Re-read current SVD proof locations in:
  - `tests/test_svd.c`;
  - `tests/test_svd_partial_helpers.h`.
- Captured current SVD test-surface metrics:
  - `tests/test_svd.c`: 2893 lines;
  - `tests/test_svd_partial_helpers.h`: 915 lines.
- Ran focused SVD baseline validation:
  - `make build/test_svd && build/test_svd`;
  - `test_svd`: 98 tests, 0 failed, 0 skipped, 1562 assertions.
- Selected **partial-SVD vector/residual cleanup** for Day 10.
- Day 10 selected owner:
  - duplicated `A*v ~= sigma*u` residual loops in
    `tests/test_svd_partial_helpers.h`;
  - primary target tests: `test_partial_svd_vectors_Av` and
    `test_partial_svd_vectors_wide`.
- Day 10 must keep visible:
  - fixture shape and inserted values;
  - selected partial rank `k`;
  - SVD options;
  - expected singular-value tolerances;
  - residual threshold `1e-6`;
  - diagnostic labels;
  - `sparse_svd_partial` call and ownership checks.
- Day 10 may centralize only the mechanical residual loop:
  - temporary `Av` and `v` allocation;
  - `Vt` row extraction;
  - `sparse_matvec`;
  - per-vector residual computation;
  - maximum residual tracking.
- Explicitly deferred broad SVD helper abstractions:
  - reconstruction;
  - U/Vt orthogonality;
  - Moore-Penrose products;
  - dense and sparse low-rank proof loops;
  - condition-number proof cleanup.
- Added Day 9 artifact:
  - `artifacts/day9-svd-proof-boundary-refresh.md`.

## Day 10 Notes

- Implemented the Day 9-selected partial-SVD vector/residual cleanup in
  `tests/test_svd_partial_helpers.h`.
- Added local helper `partial_svd_max_av_residual` for the mechanical
  `A*v ~= sigma*u` residual loop.
- Updated the two selected owner tests:
  - `test_partial_svd_vectors_Av`;
  - `test_partial_svd_vectors_wide`.
- Preserved proof visibility for:
  - fixture shapes and inserted values;
  - selected partial ranks;
  - SVD options;
  - expected singular-value tolerances;
  - residual diagnostic labels;
  - residual threshold `1e-6`;
  - `sparse_svd_partial` calls and cleanup ownership.
- Captured before/after metrics:
  - `tests/test_svd_partial_helpers.h`: 915 lines before, 907 lines after;
  - `tests/test_svd.c`: 2893 lines unchanged.
- Focused validation passed:
  - `make build/test_svd && build/test_svd`;
  - `test_svd`: 98 tests, 0 failed, 0 skipped, 1562 assertions.
- Full required C/header quality chain passed:
  - `make format && make lint && make test`.
- Confirmed no public API, install-header, helper-target, Make/CMake
  source-list, or reviewed CTest registration drift.
- Added Day 10 artifact:
  - `artifacts/day10-svd-proof-owner-cleanup.md`.

## Day 11 Notes

- Captured proof-owner metrics through Day 10.
- Code/test file metrics:
  - `tests/test_eigs.c`: 1560 baseline lines, 1758 current lines,
    +198 / -0;
  - `tests/test_ldlt_csc.c`: 3896 baseline lines, 3915 current lines,
    +79 / -60;
  - `tests/test_svd_partial_helpers.h`: 915 baseline lines, 907 current
    lines, +42 / -50.
- Confirmed no current diffs under:
  - `Makefile`;
  - `CMakeLists.txt`;
  - `cmake/`;
  - `include/`;
  - `src/`.
- Confirmed changed build/test membership is limited to existing test files:
  - `tests/test_eigs.c`;
  - `tests/test_ldlt_csc.c`;
  - `tests/test_svd_partial_helpers.h`.
- Documented membership drift status:
  - no helper-target drift;
  - no Make/CMake source-list drift;
  - no public API or install-header drift;
  - no reviewed CTest registration drift.
- Documented remaining proof-owner residual queues for:
  - eigensolver grow-m adjacent internals;
  - direct/iterative exact-RHS and oracle lanes;
  - SVD reconstruction, orthogonality, Moore-Penrose, low-rank, and
    condition-number owners.
- Reaffirmed broad-abstraction non-claims:
  - no broad cross-solver proof abstraction is proven safe;
  - no broad SVD proof abstraction is proven safe.
- Added Day 11 artifact:
  - `artifacts/day11-proof-owner-metrics-and-non-claims.md`.

## Day 12 Notes

- Built the integrated validation matrix for Sprint 113 touched surfaces.
- Validation owners assigned:
  - `tests/test_eigs.c` -> focused `test_eigs` plus full quality gate;
  - `tests/test_ldlt_csc.c` -> focused `test_ldlt_csc` plus full quality gate;
  - `tests/test_svd_partial_helpers.h` -> focused `test_svd` plus full
    quality gate;
  - Sprint 113 docs and artifacts -> doc hygiene and local Markdown link
    checks.
- Defined Day 13 focused validation commands:
  - `make build/test_eigs && build/test_eigs`;
  - `make build/test_ldlt_csc && build/test_ldlt_csc`;
  - `make build/test_svd && build/test_svd`.
- Confirmed Day 13 must run the full quality gate because `.c` and `.h` files
  are changed:
  - `make format && make lint && make test`.
- Defined build/source/API drift checks:
  - `git diff --name-only -- Makefile CMakeLists.txt cmake include src tests |
    sort`;
  - `git diff --name-only -- Makefile CMakeLists.txt cmake include src |
    sort`.
- Defined documentation hygiene checks:
  - `git diff --check`;
  - trailing-whitespace scan;
  - local Markdown link check over Sprint 113 plan, working notes, and
    artifacts.
- Added Day 12 artifact:
  - `artifacts/day12-integrated-validation-plan.md`.

## Day 13 Notes

- Executed the Day 12 integrated validation matrix.
- Focused eigensolver validation passed:
  - `make build/test_eigs && build/test_eigs`;
  - 36 tests, 0 failed, 0 skipped, 345 assertions.
- Focused LDLT CSC validation passed:
  - `make build/test_ldlt_csc && build/test_ldlt_csc`;
  - 100 tests, 0 failed, 0 skipped, 3556 assertions.
- Focused SVD validation passed:
  - `make build/test_svd && build/test_svd`;
  - 98 tests, 0 failed, 0 skipped, 1562 assertions.
- Build and membership drift checks passed:
  - only `tests/test_eigs.c`, `tests/test_ldlt_csc.c`, and
    `tests/test_svd_partial_helpers.h` are changed under build/test scope;
  - no `Makefile`, `CMakeLists.txt`, `cmake/`, `include/`, or `src/` drift.
- Full required quality gate passed:
  - `make format && make lint && make test`.
- Documentation hygiene checks passed:
  - `git diff --check`;
  - trailing-whitespace scan;
  - local Markdown link check.
- No blocking validation failures were found.
- Added Day 13 artifact:
  - `artifacts/day13-integrated-validation-execution.md`.

## Day 14 Notes

- Reviewed all Sprint 113 artifacts, working notes, code/test changes, metrics,
  and validation output.
- Confirmed all seven Sprint 113 project-plan items are complete or explicitly
  deferred:
  - residual intake and boundary refresh complete;
  - eigensolver grow-m behavior proof complete;
  - eigensolver source movement closed as no-move with proof requirements;
  - LDLT CSC external dense-reference oracle cleanup complete;
  - partial-SVD vector/residual cleanup complete;
  - proof-owner metrics and non-claims complete;
  - validation and closeout complete.
- Summarized eigensolver outcome:
  - grow-m behavior proof added in `tests/test_eigs.c`;
  - broader source movement deferred behind explicit proof requirements.
- Summarized direct/iterative outcome:
  - LDLT CSC external dense-reference oracle cleanup completed in
    `tests/test_ldlt_csc.c`.
- Summarized SVD outcome:
  - partial-SVD `A*v ~= sigma*u` residual helper cleanup completed in
    `tests/test_svd_partial_helpers.h`.
- Recorded dependency-ordered residual deferred debt for:
  - eigensolver source-movement proof;
  - direct/iterative exact-RHS and oracle owners;
  - SVD proof owners.
- Reaffirmed final no-drift status:
  - no public API drift;
  - no install-header drift;
  - no helper-target drift;
  - no Make/CMake source-list drift;
  - no reviewed CTest membership drift.
- Added Day 14 artifact:
  - `artifacts/day14-closeout-and-handoff.md`.
