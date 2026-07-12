# Sprint 120 Day 5 Split Ranking and Proof Plan

## Purpose

Day 5 ranks direct and iterative proof-owner split candidates from the Day 2
and Day 3 audits using the Day 4 shared fixture architecture. It selects one
direct split batch and one iterative split batch for later Sprint 120 design
and implementation, and records validation, source-list, CMake/CTest,
rollback, and deferral criteria before any C source or header edit begins.

This artifact is a planning and proof-lane artifact only. No test owner is
moved by Day 5.

## Selection Rules Applied

| Rule | Day 5 application |
|---|---|
| Prefer focused scenario splits over broad helper extraction. | Selected batches isolate named proof blocks first; shared helpers are introduced only if Day 6 or Day 9 design proves they are needed. |
| Keep solver-specific meaning local. | Tolerances, convergence status, residual interpretation, lifecycle, callback behavior, and failure modes remain in named tests. |
| Avoid high-lifecycle splits before state ownership is designed. | LDLT CSC two-pass/external-reference work and public handle movement are deferred. |
| Treat new test executables as CTest-count changes. | Any split that adds a test executable must update Makefile/CMake membership and record `ctest -N` impact. |
| Preserve non-claims. | No selected split broadens direct/iterative parity, external-oracle completeness, platform/package support, public API surface, or performance claims. |

## Ranked Direct Split Candidates

| Rank | Candidate | Proof value | Risk | Rollback cost | Decision |
|---:|---|---|---|---|---|
| 1 | QR solve scenario split from `tests/test_qr.c` | High: isolates square, overdetermined, analytical, rank-deficient, SuiteSparse, null-residual, and QR-vs-LU solve proofs from a 3,234-line hotspot while preserving generated-RHS/residual owner visibility. | Medium: QR reported residual and fixture-specific tolerances must stay local. | Medium: restore moved solve tests and remove new source-list/CMake entries. | Select as direct batch. |
| 2 | LDLT Matrix Market and KKT fixture helper extraction from `tests/test_ldlt.c` | Medium-high: repeated `A * ones`, residual, and KKT fixture setup could support future direct oracle cleanup. | Medium-high: inertia expectations are LDLT-specific and easy to hide accidentally. | Medium: helper extraction can be reversed, but multiple LDLT tests may depend on it. | Defer until after QR solve split proves fixture placement. |
| 3 | LDLT cross-backend scenario split from `tests/test_ldlt.c` | Medium-high: valuable bounded direct oracle with backend agreement semantics. | High: backend telemetry, linked-list/CSC routing, and environment behavior are more complex than QR solve. | Medium-high: rollback touches backend scenario ownership and CMake/Makefile membership. | Defer to residual queue after direct QR batch. |
| 4 | LDLT CSC solve scenario split from `tests/test_ldlt_csc.c` | High: reduces largest direct hotspot and isolates many solve cases. | High: relative infinity residuals, AMD, inertia, in-place solve, linked-list agreement, and singular behavior are tightly coupled. | High: large ownership move with CTest and source-list impact. | Defer until LDLT CSC state owner design exists. |
| 5 | LDLT CSC external dense-reference split | Medium-high: clear oracle role and repeated KKT dense-reference behavior. | Very high: external process, platform skip policy, permutation/unpermutation, analysis state, and cleanup paths are coupled. | High: rollback must restore state helpers and platform behavior. | Defer explicitly. |
| 6 | QR exact-RHS/residual helper extraction only | Medium: reduces repeated fixture setup. | Medium: may obscure QR-specific `A * [1,2,...]` and reported-residual semantics if done before scenario split. | Low-medium: helper can be inlined back. | Fold into selected QR solve design only if needed. |
| 7 | QR reconstruction or sparse-mode split | Medium: reduces QR hotspot. | Medium-high for Sprint 120 scope: less aligned with direct/iterative generated-RHS oracle architecture. | Medium. | Defer outside selected Sprint 120 batch. |

## Ranked Iterative Split Candidates

| Rank | Candidate | Proof value | Risk | Rollback cost | Decision |
|---:|---|---|---|---|---|
| 1 | Block BiCGSTAB scenario split from `tests/test_bicgstab.c` | High: isolates multi-RHS, mixed convergence, preconditioned block, single-RHS equivalence, aggregation, and error-propagation behavior from a 1,826-line hotspot. | Medium: block status aggregation and per-column semantics must stay local. | Medium: restore block tests and remove new source-list/CMake entries. | Select as iterative batch. |
| 2 | Block MINRES scenario split from `tests/test_minres.c` | High: isolates multi-RHS SPD/indefinite, zero-column, many-RHS, preconditioned, and sequential-equivalence behavior. | Medium-high: symmetric-indefinite/KKT interpretation and preconditioner semantics are more nuanced. | Medium: restore block tests and CMake/Makefile entries. | Defer behind BiCGSTAB block batch. |
| 3 | GMRES SuiteSparse/restart/right-preconditioner split from `tests/test_iterative.c` | High: reduces a 2,924-line combined CG/GMRES hotspot and isolates high-value GMRES proof blocks. | High: restart outcomes, relaxed SuiteSparse behavior, reported-vs-true residual, and right-preconditioner lifecycle are claim-sensitive. | Medium-high: source movement touches central iterative file and shared helpers. | Defer until after one simpler iterative split. |
| 4 | Matrix-free BiCGSTAB split from `tests/test_bicgstab.c` | Medium-high: isolates callback equivalence and failure propagation. | Medium-high: callback semantics overlap with broader iterative callback architecture. | Medium. | Defer until callback helper boundary is needed. |
| 5 | MINRES preconditioner/direct-comparison split from `tests/test_minres.c` | Medium-high: valuable direct/iterative comparison and preconditioner proof owner. | High: LDLT/GMRES comparison boundaries and preconditioner semantics are sensitive. | Medium-high. | Defer. |
| 6 | Public iterative handle helper movement | Medium: removes dependency on `test_iterative.c` static builders. | High: handle validation/reuse/growth and static builder dependencies need exact placement. | Medium. | Defer until helper placement rules are proven by prior splits. |
| 7 | CG/GMRES broad shared fixture extraction | Medium: reduces repeated builders. | High: broad extraction can hide solver-specific convergence/failure contracts. | Medium-high. | Defer explicitly. |

## Selected Direct Split Batch

| Field | Decision |
|---|---|
| Batch name | QR solve scenario owner split |
| Current owner | `tests/test_qr.c` |
| Target owner | To be designed on Day 6; preferred target is a focused QR solve test owner such as `tests/test_qr_solve.c` if source-list/CTest impact is accepted. |
| Candidate tests | QR solve cases, SuiteSparse solve cases, null-residual solve case, and QR-vs-LU solve comparison. Exact list is Day 6 work. |
| Helper policy | Keep `make_qr_exact_rhs`, QR true-residual measurement, and QR-specific tolerances scenario-local unless Day 6 proves a narrow helper is safer. |
| Behavior that stays local | Reported residual versus true residual, square/overdetermined/rank-deficient tolerance budgets, SuiteSparse fixture-specific tolerance, QR factor/solve cleanup, and QR-vs-LU non-parity wording. |
| Expected Makefile impact | If a new test executable is selected, add `$(TESTDIR)/test_qr_solve.c` to `TEST_SRCS`; helper-only changes still require full C quality. |
| Expected CMake/CTest impact | If a new executable is selected, add `add_sparse_test(test_qr_solve)` and expect CTest count to increase by one. If only helper movement occurs, CTest count should not change. |
| Full quality requirement | Any `.c` or `.h` implementation requires `make format && make lint && make test` before proceeding. |

## Selected Iterative Split Batch

| Field | Decision |
|---|---|
| Batch name | Block BiCGSTAB scenario owner split |
| Current owner | `tests/test_bicgstab.c` |
| Target owner | To be designed on Day 9; preferred target is a focused block BiCGSTAB test owner such as `tests/test_bicgstab_block.c` if source-list/CTest impact is accepted. |
| Candidate tests | Block BiCGSTAB argument validation, 2-RHS and 4-RHS solves, single-RHS equivalence, mixed convergence, preconditioned block behavior, result aggregation, error propagation, and `nrhs` validation. Exact list is Day 9 work. |
| Helper policy | Keep per-column expected status, aggregate result semantics, preconditioner failure behavior, and block cleanup scenario-local. A block RHS helper may be introduced only if Day 9 proves it preserves column semantics. |
| Behavior that stays local | Per-column convergence, zero/mixed RHS handling, single-RHS equivalence, block result aggregation, preconditioner error propagation, callback/result cleanup, and expected failure modes. |
| Expected Makefile impact | If a new test executable is selected, add `$(TESTDIR)/test_bicgstab_block.c` to `TEST_SRCS`; helper-only changes still require full C quality. |
| Expected CMake/CTest impact | If a new executable is selected, add `add_sparse_test(test_bicgstab_block)` and expect CTest count to increase by one. If only helper movement occurs, CTest count should not change. |
| Full quality requirement | Any `.c` or `.h` implementation requires `make format && make lint && make test` before proceeding. |

## Focused Validation Command List

These commands are planned validation lanes for implementation days. Day 5
does not run them because it makes documentation-only changes.

| Change | Focused proof before full quality | Count/source proof | Full required proof |
|---|---|---|---|
| QR solve moved into a new executable | `make build/test_qr build/test_qr_solve && ./build/test_qr && ./build/test_qr_solve` | `make source-list-check`; CMake configure/build plus `ctest -N --test-dir <build-dir>` if CMake membership changes | `make format && make lint && make test` |
| QR solve helper-only refactor | `make build/test_qr && ./build/test_qr` | `make source-list-check` if source-list touched; no CTest count change expected | `make format && make lint && make test` |
| Block BiCGSTAB moved into a new executable | `make build/test_bicgstab build/test_bicgstab_block && ./build/test_bicgstab && ./build/test_bicgstab_block` | `make source-list-check`; CMake configure/build plus `ctest -N --test-dir <build-dir>` if CMake membership changes | `make format && make lint && make test` |
| Block BiCGSTAB helper-only refactor | `make build/test_bicgstab && ./build/test_bicgstab` | `make source-list-check` if source-list touched; no CTest count change expected | `make format && make lint && make test` |
| Documentation-only design updates | Not applicable | Not applicable | `git diff --check` and focused trailing-whitespace scan over `docs/planning/EPIC_11/SPRINT_120` |

## Rollback Criteria

Rollback the selected split if any of the following occur:

- Focused QR or BiCGSTAB tests fail after movement.
- `make source-list-check` fails after source-list edits.
- CMake configure/build or `ctest -N` fails after CMake membership edits.
- CTest count changes unexpectedly or platform-gated test membership changes
  without an explicit reviewed reason.
- The split requires moving solver-specific tolerance, convergence, residual,
  callback, lifecycle, or failure interpretation into a generic helper.
- The split widens public claims or implies broad direct/iterative parity.
- The selected batch grows beyond a bounded scenario owner and starts pulling
  unrelated QR, BiCGSTAB, GMRES, MINRES, LDLT, package, benchmark, or public
  API surfaces.

Rollback path:

1. Restore moved tests to the original owner file.
2. Remove the new test executable/helper from `TEST_SRCS` and CMake if added.
3. Restore expected CTest membership.
4. Re-run the focused original test (`./build/test_qr` or
   `./build/test_bicgstab`) and then the required full quality lane if C/header
   files were modified.
5. Record the residual owner and deferral reason in the sprint artifact.

## Deferred Candidate Queue

| Candidate | Residual owner | Defer reason |
|---|---|---|
| LDLT Matrix Market/KKT helper extraction | Future direct oracle cleanup | Wait for QR solve split to prove fixture placement and direct helper naming. |
| LDLT cross-backend split | Future direct oracle cleanup | Backend telemetry and routing semantics are more complex than the first selected batch. |
| LDLT CSC solve split | Future LDLT CSC owner cleanup | Requires explicit CSC state/lifecycle owner before movement. |
| LDLT CSC external dense-reference split | Future LDLT CSC/external-reference cleanup | Platform skip, external process, permutation state, and cleanup are too coupled for the first batch. |
| Block MINRES split | Future iterative owner cleanup | Good candidate after block BiCGSTAB proves block split validation pattern. |
| GMRES SuiteSparse/restart/right-preconditioner split | Future iterative owner cleanup | High-value but claim-sensitive; defer until after simpler iterative split. |
| Matrix-free BiCGSTAB split | Future callback owner cleanup | Depends on callback fixture boundary being needed and proven. |
| Public iterative handle movement | Future handle/helper cleanup | Depends on placement rules being validated by earlier split work. |

## Non-Claims

The selected batches are maintainability and proof-owner cleanup work only.
They do not claim new solver support, broader direct/iterative parity,
external-oracle completeness, package or platform support, public API
expansion, performance improvement, or state-of-the-art validation.

## Completion Criteria Check

| Criterion | Status |
|---|---|
| Items 3 and 4 have evidence-ranked implementation candidates | Complete: QR solve scenario and block BiCGSTAB scenario are selected from ranked direct and iterative tables. |
| Every selected candidate has focused tests and rollback instructions | Complete: validation commands, CTest/source-list impact, full quality requirements, rollback criteria, and rollback path are recorded. |
| No implementation begins without an agreed validation lane | Complete: Day 5 is documentation-only, and implementation is deferred to Day 6+ and Day 9+ design artifacts. |
