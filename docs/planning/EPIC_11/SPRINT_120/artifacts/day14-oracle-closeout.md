# Sprint 120 Day 14 Oracle Closeout

## Purpose

Close Sprint 120 by publishing what split, what stayed in place, what
validation supports the work, which claims were intentionally not made, and
what the next sprint should inherit from the direct/iterative oracle
architecture effort.

## Sprint Outcome Summary

| Area | Outcome | Evidence |
|---|---|---|
| Sprint intake and evidence setup | Completed | Day 1 created the Sprint 120 artifact structure, item map, validation rules, and non-claim expectations. |
| Direct oracle ownership audit | Completed | Day 2 inventoried QR, LDLT, LDLT CSC, LU, and Cholesky oracle owners, helpers, tolerances, and split candidates. |
| Iterative oracle ownership audit | Completed | Day 3 inventoried CG, GMRES, BiCGSTAB, MINRES, block solver, callback, and preconditioner proof owners. |
| Shared fixture architecture | Completed | Day 4 defined where shared fixtures are acceptable and where solver-local tolerances and failure semantics must remain visible. |
| Split candidate ranking | Completed | Day 5 selected a bounded QR solve split and block BiCGSTAB split while deferring higher-risk candidates. |
| Direct split design and implementation | Completed | Days 6-8 split QR solve scenarios into `tests/test_qr_solve.c` and revalidated the old/new QR owners. |
| Iterative split design and implementation | Completed | Days 9-10 split block BiCGSTAB scenarios into `tests/test_bicgstab_block.c` and revalidated scalar/block owners. |
| Cross-solver oracle pilot | Completed | Days 11-12 designed and implemented `tests/test_cross_solver_oracle.c` for a bounded generated-RHS SPD fixture across LU, Cholesky, QR, and CG. |
| Validation package | Completed | Day 13 re-ran focused tests, source-list, CMake/CTest membership, lint, and full tests. |
| Closeout and handoff | Completed | This artifact records residuals, non-claims, and Sprint 121 handoff guidance. |

## Files Changed By Sprint 120

| File | Change |
|---|---|
| `tests/test_qr.c` | Removed QR solve scenario tests that now live in the focused QR solve owner; retained QR factorization, reconstruction, rank, sparse-mode, economy, and refinement proof blocks. |
| `tests/test_qr_solve.c` | Added focused QR solve scenario owner for square, overdetermined, analytical, rank-deficient, SuiteSparse, QR-vs-LU, and synthetic tall solve paths. |
| `tests/test_bicgstab.c` | Removed block BiCGSTAB tests that now live in the focused block owner; retained scalar BiCGSTAB, SuiteSparse, numerical hardening, matrix-free, callback, and adjacent comparison proof blocks. |
| `tests/test_bicgstab_block.c` | Added focused block BiCGSTAB owner for null/shape checks, multi-RHS solves, scalar equivalence, mixed convergence, preconditioned block solve, result aggregation, and preconditioner error propagation. |
| `tests/test_cross_solver_oracle.c` | Added bounded cross-solver oracle pilot for LU, Cholesky, QR, and CG on one generated-RHS SPD fixture. |
| Makefile | Registered `test_qr_solve`, `test_bicgstab_block`, and `test_cross_solver_oracle` in `TEST_SRCS`. |
| `CMakeLists.txt` | Registered `test_qr_solve`, `test_bicgstab_block`, and `test_cross_solver_oracle` in the reviewed CTest surface. |
| `docs/planning/EPIC_11/SPRINT_120/` | Added Sprint 120 plan, working notes, and day-by-day artifacts. |

## Validation Summary

| Validation Lane | Status | Evidence |
|---|---|---|
| Formatting | Pass | `make format` passed. |
| Focused direct tests | Pass | `test_qr`: 63 tests, 0 failed; `test_qr_solve`: 10 tests, 0 failed. |
| Focused iterative tests | Pass | `test_bicgstab`: 49 tests, 0 failed; `test_bicgstab_block`: 12 tests, 0 failed. |
| Focused pilot test | Pass | `test_cross_solver_oracle`: 1 test, 0 failed; LU, Cholesky, QR, and CG residuals were all below `1e-10`. |
| Source-list parity | Pass | `make source-list-check` passed with 49 library sources. |
| CMake build and CTest registration | Pass | Clean reviewed CMake build passed; `ctest -N` reported 57 tests. |
| Static analysis | Pass | `make lint` passed, including strict warnings, clang-tidy, and cppcheck. |
| Full Makefile tests | Pass | `make test` ended with all tests passed. |

## CTest Membership

| Test | Registration |
|---|---|
| `test_qr` | Test #20. |
| `test_qr_solve` | Test #21. |
| `test_bicgstab` | Test #39. |
| `test_bicgstab_block` | Test #40. |
| `test_cross_solver_oracle` | Test #41. |
| Total reviewed CTest count | 57. |

## Residual Queue

| Residual | Why Deferred | Follow-Up Requirement |
|---|---|---|
| LDLT Matrix Market and KKT fixture helper extraction | Good reuse potential, but inertia and fixture expectations need to remain LDLT-local until another direct split proves helper pressure is worth the coupling. | Design an LDLT-local helper contract first; preserve inertia and generated-RHS expectations beside the test owner. |
| LDLT cross-backend scenario split | Backend telemetry, dense/native route selection, linked-list/CSC agreement, and environment behavior are coupled. | Split only with a focused backend owner design, route-selection proof, and CMake/CTest count evidence. |
| LDLT CSC solve scenario split | AMD behavior, in-place solve, inertia, linked-list agreement, singular detection, and relative infinity residuals remain tightly coupled. | Build a dedicated LDLT CSC solve owner checklist before movement. |
| LDLT CSC external dense-reference split | External process policy, platform skip/error behavior, permutation lifecycle, and analysis-aware state are too broad for Sprint 120. | Pair any future split with platform skip policy, external oracle trust boundary, and full quality evidence. |
| QR reconstruction or sparse-mode split | Valuable maintainability work, but less aligned with the Sprint 120 generated-RHS/direct-oracle focus than the completed QR solve split. | Revisit in a QR maintainability sprint if `test_qr.c` remains a priority giant-test owner. |
| QR exact-RHS/residual shared helper extraction | Sprint 120 kept helpers local to avoid generic residual semantics. | Promote only after repeated split pressure proves duplication is worse than helper coupling. |
| Block MINRES split | Lower priority than the selected block BiCGSTAB split and not implemented in Sprint 120. | Design a MINRES-specific owner that preserves LDLT/GMRES comparison semantics and convergence expectations. |
| GMRES SuiteSparse, restart, and right-preconditioner split | Higher coupling across external fixtures, restart policy, and preconditioner behavior. | Split with explicit fixture taxonomy, restart tolerance policy, and focused old/new validation. |
| Matrix-free BiCGSTAB callback split | Callback error propagation and matrix-free behavior remained scalar-owner local. | Extract only if callback-specific ownership is designed and scalar BiCGSTAB non-callback behavior remains stable. |
| Public iterative handle helper movement | Outside the selected Sprint 120 split batch. | Revisit when handle-lifecycle proof owners are being consolidated. |
| Broad CG/GMRES shared fixture extraction | Shared fixture design exists, but Sprint 120 avoided broad helper promotion from limited proof pressure. | Add only after two or more focused owners need the same fixture without hiding solver-local tolerances. |
| External dense-reference oracle expansion | The Day 12 pilot intentionally used a local generated-RHS fixture. | Future external oracle sprint should define trust boundaries, skip/error policy, and platform behavior before adding comparisons. |
| Package/install/platform validation for oracle tests | Sprint 120 did not change install, package, ABI, public headers, or workflows. | Future packaging or CI sprint should decide whether these focused owners need install-consumer or platform count lanes. |

## Non-Claim Register

Sprint 120 is a maintainability and proof-owner sprint. It deliberately does
not create the following claims:

| Non-Claim | Reason |
|---|---|
| No new QR capability claim | QR solve behavior was moved into a focused owner; solver algorithms and public API did not change. |
| No new BiCGSTAB capability claim | Block BiCGSTAB tests moved to a focused owner; block solver behavior was preserved rather than expanded. |
| No broad direct solver parity claim | QR-vs-LU and the pilot are bounded fixtures, not a general direct-solver equivalence statement. |
| No broad direct/iterative parity claim | The cross-solver pilot covers one compatible SPD generated-RHS fixture only. |
| No external-oracle completeness claim | Sprint 120 did not add broad SciPy, LAPACK, SuiteSparse, or external dense-reference coverage. |
| No package, install, platform, or ABI claim | Public headers, packaging files, install metadata, and workflow support surfaces were unchanged. |
| No performance or scalability claim | Benchmarks were not executed; the work reduced ownership risk and preserved behavior. |
| No state-of-the-art claim | The evidence is appropriate for maintainability and focused proof ownership, not broad product positioning. |

## Sprint 121 Handoff

Sprint 121 should inherit Sprint 120 as a proof-owner and validation pattern,
not as permission to make broader solver claims.

| Handoff Area | Guidance |
|---|---|
| Proof-owner movement | Keep future splits small, behavior-preserving, and paired with focused old/new tests before full quality. |
| Shared helper discipline | Use shared fixtures only when they reduce real duplication without hiding solver-specific tolerances, convergence criteria, failure classes, or skip policy. |
| CMake and source-list parity | Any new test owner must be registered in Makefile and CMake, with `ctest -N` count evidence recorded. Any new library source must also update source-list metadata. |
| Direct residuals | LDLT and LDLT CSC candidates need owner-specific design before movement; do not combine external-reference, lifecycle, and backend splits into one batch. |
| Iterative residuals | GMRES, MINRES, matrix-free BiCGSTAB, and handle-lifecycle candidates need solver-local tolerance and failure-mode contracts before movement. |
| Oracle expansion | Treat the Day 12 pilot as a bounded pattern. Broader external-oracle work needs explicit trust boundaries and platform skip/error behavior. |
| Non-claims | Continue recording when a split improves maintainability without changing public capability, performance, platform, or state-of-the-art evidence. |

## Artifact Index

| Day | Artifact | Purpose |
|---|---|---|
| 1 | `day1-sprint-intake.md` | Sprint scope, prerequisites, item owners, validation rules, and non-claims. |
| 2 | `day2-direct-oracle-ownership-audit.md` | Direct solver oracle owner map, tolerances, helpers, and split candidates. |
| 3 | `day3-iterative-oracle-ownership-audit.md` | Iterative solver oracle owner map, progress/callback behavior, and split candidates. |
| 4 | `day4-shared-fixture-architecture.md` | Shared fixture policy and solver-local responsibility boundaries. |
| 5 | `day5-split-ranking-proof-plan.md` | Ranked direct/iterative split candidates and selected implementation batches. |
| 6 | `day6-direct-split-implementation-checklist.md` | Direct split file plan, build impact, validation checklist, and rollback. |
| 7 | `day7-direct-split-implementation.md` | QR solve split implementation evidence. |
| 8 | `day8-direct-validation-consolidation.md` | Direct split revalidation and residual direct queue. |
| 9 | `day9-iterative-split-implementation-checklist.md` | Iterative split file plan, behavior contract, validation checklist, and rollback. |
| 10 | `day10-iterative-split-implementation.md` | Block BiCGSTAB split implementation evidence. |
| 11 | `day11-cross-solver-oracle-pilot-design.md` | Bounded cross-solver oracle pilot design and non-claims. |
| 12 | `day12-cross-solver-oracle-pilot-implementation.md` | Cross-solver oracle pilot implementation and validation evidence. |
| 13 | `day13-validation-package.md` | Focused, source-list, CMake/CTest, lint, and full test validation package. |
| 14 | `day14-oracle-closeout.md` | Closeout, residuals, non-claims, artifact index, and Sprint 121 handoff. |

## Completion Criteria

| Criterion | Status |
|---|---|
| Sprint 120 Item 7 complete | Complete |
| Every split has evidence | Complete |
| Every deferred candidate has an owner or follow-up requirement | Complete |
| Validation outcomes are recorded | Complete |
| Non-claim boundaries are recorded | Complete |
| Sprint 121 handoff risks and prerequisites are recorded | Complete |
