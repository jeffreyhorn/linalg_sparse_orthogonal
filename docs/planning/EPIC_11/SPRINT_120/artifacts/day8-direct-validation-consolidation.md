# Sprint 120 Day 8 Direct Validation and Consolidation

## Purpose

Day 8 revalidates the Day 7 QR solve split, consolidates the direct oracle
residual queue, and records the readiness criteria for Day 9 iterative split
design. The direct split remains focused on proof-owner maintainability: it
does not widen QR behavior, direct solver parity, public API, package,
platform, performance, or external-oracle claims.

## Revalidation Evidence

| Command | Result |
|---|---|
| `make build/test_qr build/test_qr_solve && ./build/test_qr && ./build/test_qr_solve` | Passed. `test_qr`: 63 tests, 0 failed. `test_qr_solve`: 10 tests, 0 failed. |
| `make source-list-check` | Passed: 49 library sources. |
| `ctest -N --test-dir build/quality-review-cmake` | Passed membership inspection. Total tests: 55; `test_qr_solve` is registered as test #21. |

Day 7 already completed the required full C quality lane for the C/build
changes:

- `make format`;
- `make lint`;
- `make test`.

Day 8 made documentation-only consolidation changes after revalidation.

## Direct Diff Review

| Review area | Result |
|---|---|
| QR solve tolerances | Preserved in `tests/test_qr_solve.c`: `1e-10` square/analytical/null residual paths, `1.0` loose least-squares/rank-deficient paths, `1e-8` `nos4`/`west0067`/tall solve paths, and `1e-4` `bcsstk04` solve path. |
| Reported versus true residual | Preserved in the focused owner through `assert_qr_solve_true_residual_below` and `qr_solve_rel_residual`. |
| Generated RHS semantics | Preserved as `A * [1, 2, ...]` through `make_qr_solve_exact_rhs`. |
| Mixed reconstruction/solve cases | Preserved for `bcsstk04` and tall synthetic solve through `assert_qr_solve_reconstruction_below`. |
| QR-vs-LU comparison | Preserved as a bounded `nos4` comparison with residual and max-difference checks; no broad direct parity wording was added. |
| Build membership | Makefile and CMake both register the new focused owner. |
| Public API or support surface | No public headers, README, package files, examples, workflows, or API docs were modified. |

## Direct Residual Queue

| Residual candidate | Owner | Status | Rationale |
|---|---|---|---|
| LDLT Matrix Market and KKT fixture helper extraction | Future direct oracle cleanup | Deferred | Good reuse potential, but inertia expectations must remain LDLT-local. Wait until at least one direct split pattern is stable. |
| LDLT cross-backend scenario split | Future direct oracle cleanup | Deferred | Backend telemetry, route selection, linked-list/CSC agreement, and dense-backend environment behavior need a focused backend owner design. |
| LDLT CSC solve scenario split | Future LDLT CSC owner cleanup | Deferred | High value, but relative infinity residuals, AMD behavior, in-place solve, inertia, linked-list agreement, and singular detection are tightly coupled. |
| LDLT CSC external dense-reference split | Future LDLT CSC/external-reference cleanup | Deferred | External process policy, platform skip/error behavior, permutation lifecycle, and analysis-aware state remain too coupled for Sprint 120's first direct split. |
| QR reconstruction or sparse-mode split | Future QR maintainability cleanup | Deferred | Important QR coverage, but less aligned with Sprint 120 generated-RHS/direct-oracle focus than the completed solve-owner split. |
| QR exact-RHS/residual shared helper extraction | Future helper cleanup only if repeated split pressure appears | Deferred | Day 7 deliberately kept helpers local to avoid broad helper semantics. Revisit only if later splits prove duplication is a larger risk than helper coupling. |

## Day 9 Iterative Readiness Checklist

| Requirement | Status for Day 9 |
|---|---|
| Direct split implementation pattern exists | Ready: QR solve owner split provides a bounded new-executable pattern. |
| Build metadata pattern exists | Ready: Makefile and CMake registrations, CTest count proof, and source-list check were exercised. |
| Focused validation pattern exists | Ready: focused old/new executable validation ran before broader quality. |
| Rollback pattern exists | Ready: Day 6 and Day 7 artifacts record rollback steps. |
| Helper policy is clear | Ready: prefer scenario-local helpers for first split; avoid broad shared helper extraction unless concrete duplication warrants it. |
| Iterative target from Day 5 is selected | Ready: block BiCGSTAB scenario owner split is selected for Day 9 design. |
| Iterative behavior constraints are visible | Ready: Day 3 and Day 5 record per-column convergence, result aggregation, preconditioner failure, and block cleanup boundaries. |
| Unsupported claims are blocked | Ready: Day 4, Day 5, and this artifact preserve non-claim wording. |

## No-Claim Notes

The direct split is a maintainability change only. It does not claim:

- new QR functionality;
- broader direct solver parity;
- broader QR-vs-LU equivalence;
- external-oracle completeness;
- package/install support;
- platform expansion;
- public API expansion;
- performance improvement;
- state-of-the-art validation.

## Completion Criteria Check

| Criterion | Status |
|---|---|
| Item 3 is complete or explicitly deferred with evidence | Complete: QR solve scenario owner split is implemented and revalidated; remaining direct candidates are deferred with owners and rationale. |
| Direct residuals are documented | Complete: residual queue records LDLT, LDLT CSC, QR reconstruction/sparse-mode, and helper extraction residuals. |
| No unsupported direct-solver oracle claim is introduced | Complete: public/support surfaces were not modified and non-claims are explicit. |
