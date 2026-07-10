# Sprint 120 Working Notes

## Sprint Goal

Sprint 120 creates a maintainable direct/iterative oracle architecture and
reduces giant test ownership in the highest-risk direct and iterative proof
files without hiding solver-specific behavior or widening public claims.

## Starting Constraints

- Treat Sprint 118 as the current baseline for evidence templates, validation
  inventory, product truth, hotspot metrics, and public-claim drift guardrails.
- Treat Sprint 119 as the current model for source-boundary discipline:
  movement candidates must be audited, designed, validated, and explicitly
  closed or deferred.
- Do not split direct or iterative proof blocks before audit, shared-fixture
  design, focused proof, source-list/CMake impact, expected CTest count, and
  rollback expectations are documented.
- Preserve solver-specific tolerances, convergence expectations,
  failure-mode interpretation, progress-callback semantics, and lifecycle
  assertions at visible test boundaries.
- Do not claim broad direct/iterative parity, external-oracle completeness,
  state-of-the-art validation, portable performance, package/install support,
  or public API expansion from proof-owner cleanup.
- If `.c` or `.h` files change, run `make format && make lint && make test`.
- If Makefile, CMake, source-list, workflow, package, benchmark, script, or
  install surfaces change, run the relevant focused validation lane and record
  whether it is reviewed, supplemental, or local.
- If documentation only changes, run `git diff --check` and a focused
  trailing-whitespace scan over `docs/planning/EPIC_11/SPRINT_120`.

## Input Artifact Inventory

| Input | Sprint 120 use |
|---|---|
| `docs/planning/EPIC_11/PROJECT_PLAN.md` Sprint 120 section | Authoritative project-plan items, estimates, deliverables, and sprint goal. |
| `docs/planning/EPIC_11/SPRINT_120/PLAN.md` | Day-by-day execution plan and completion criteria. |
| `docs/planning/EPIC_11/SPRINT_118/artifacts/day2-validation-inventory.md` | Validation lane inventory and command expectations. |
| `docs/planning/EPIC_11/SPRINT_118/artifacts/day8-product-truth-map.md` | Product truth and public non-claim boundaries. |
| `docs/planning/EPIC_11/SPRINT_118/artifacts/day9-hotspot-metrics.md` | Source/test hotspot context for giant-test ownership. |
| `docs/planning/EPIC_11/SPRINT_118/artifacts/day10-hotspot-owner-handoff.md` | Hotspot owner handoff and candidate proof-owner work. |
| `docs/planning/EPIC_11/SPRINT_118/artifacts/day11-evidence-template-design.md` | Evidence-template design rules for future sprint artifacts. |
| `docs/planning/EPIC_11/SPRINT_118/artifacts/day12-evidence-template-refresh.md` | Refreshed template usage and validation fields. |
| `docs/planning/EPIC_11/SPRINT_118/artifacts/day13-public-claim-drift-audit.md` | Public-claim drift guardrails and non-claim checks. |
| `docs/planning/EPIC_11/SPRINT_118/artifacts/day14-sprint-closeout-handoff.md` | Sprint 119+ handoff requirements and deferred debt framing. |
| `docs/planning/EPIC_11/SPRINT_118/templates/oracle-expansion-evidence-template.md` | Required oracle expansion fields for direct/iterative pilot work. |
| `docs/planning/EPIC_11/SPRINT_118/templates/source-movement-evidence-template.md` | Required source/test owner movement fields for split work. |
| `docs/planning/EPIC_11/SPRINT_119/artifacts/day13-validation-parity-package.md` | Validation-package pattern for source-list, CMake, CTest, focused tests, and skipped-lane rationale. |
| `docs/planning/EPIC_11/SPRINT_119/artifacts/day14-movement-closeout.md` | Sprint 120 handoff rules for proof-owner movement and non-claims. |
| `docs/planning/EPIC_11/SPRINT_119/RETROSPECTIVE.md` | Recent lessons on safe owner movement, explicit deferrals, and claim boundaries. |

## Day-Level Ownership

| Day | Planned Focus | Project Plan Item |
|---:|---|---|
| 1 | Sprint intake, artifact skeleton, input inventory, validation boundaries, and owner map. | Items 1-7 intake |
| 2 | Direct-solver oracle owner audit for QR, LDLT, LDLT CSC, LU, and Cholesky-adjacent proof blocks. | Item 1 |
| 3 | Iterative-solver oracle owner audit for CG, GMRES, BiCGSTAB, MINRES, block, and callback proof blocks. | Item 1 |
| 4 | Shared fixture architecture design with solver-local tolerance and failure-mode boundaries. | Item 2 |
| 5 | Split candidate ranking, proof plan, focused validation commands, and rollback criteria. | Items 2, 3, 4 |
| 6 | Direct split batch design for exact file/helper/build boundaries. | Item 3 |
| 7 | Direct split implementation and focused proof. | Items 3, 6 |
| 8 | Direct split validation, consolidation, and residual direct oracle queue. | Items 3, 6 |
| 9 | Iterative split batch design for exact file/helper/build boundaries. | Item 4 |
| 10 | Iterative split implementation and focused proof. | Items 4, 6 |
| 11 | Bounded cross-solver oracle pilot design and non-claim framing. | Item 5 |
| 12 | Bounded cross-solver oracle pilot implementation and focused proof. | Items 5, 6 |
| 13 | Full validation package for source-list, focused tests, CMake/CTest, and required quality. | Item 6 |
| 14 | Closeout, residuals, non-claims, artifact index, and Sprint 121 handoff. | Item 7 |

## Validation Expectations

| Touched Surface | Required Checks |
|---|---|
| Documentation-only planning artifacts | `git diff --check`; focused trailing-whitespace scan over `docs/planning/EPIC_11/SPRINT_120`. |
| `.c` or `.h` source/header changes | `make format && make lint && make test`. |
| Source-list or Makefile membership | `make source-list-check` and the relevant focused build target. |
| CMake membership or test-owner changes | CMake configure/build and `ctest -N` count proof as affected. |
| Direct solver behavior | Focused tests for touched QR, LDLT, LDLT CSC, LU, or Cholesky-adjacent proof surfaces. |
| Iterative solver behavior | Focused tests for touched CG, GMRES, BiCGSTAB, MINRES, block-solver, preconditioner, or progress-callback surfaces. |
| Cross-solver oracle pilot | Focused pilot test plus adjacent direct/iterative tests that share fixtures or assertions. |
| Public claim or support wording | Check against Sprint 118 product truth, Sprint 118 public-claim drift audit, and Sprint 119 non-claim register. |

## Oracle/Split Evidence Fields Required

Every split, oracle pilot, or explicit deferral artifact should record:

- scope and touched surfaces;
- baseline owner metrics and current product-truth references;
- behavior boundary and solver-local responsibilities;
- old/new file or helper plan;
- tolerance, residual, convergence, and failure-mode ownership;
- source-list, Makefile, CMake, and CTest impact;
- focused direct, iterative, or pilot proof;
- validation commands and results;
- rollback or defer plan;
- non-claims preserved;
- residual handoff.

## Scope Boundaries

Sprint 120 may inspect, rank, design, split, validate, or explicitly defer the
direct/iterative proof-owner candidates named in the project plan. It should
not:

- broaden direct or iterative solver parity claims;
- hide solver-specific tolerances, convergence rules, or failure behavior
  behind generic helper APIs;
- add broad external solver comparison architecture outside the bounded pilot;
- alter package, ABI, platform, benchmark, adoption, or public API surfaces
  unless required by bounded proof-owner work and explicitly validated;
- perform broad source cleanup outside selected direct/iterative proof owners;
- silently defer high-risk split candidates without residual owners.

## Day 1 Notes

- Created the Sprint 120 working-notes baseline and artifact directory.
- Re-read the Sprint 120 project-plan section and Sprint 120 day-by-day plan.
- Re-read Sprint 118 evidence-template, validation, hotspot, product-truth,
  and public-claim handoff inputs.
- Re-read Sprint 119 validation package, closeout handoff, and retrospective
  for source-boundary discipline, explicit deferral handling, and non-claim
  wording.
- Mapped all Sprint 120 project-plan items to day-level owners.
- Recorded validation expectations for documentation-only, C/header,
  source-list/Makefile, CMake/CTest, direct solver, iterative solver,
  cross-solver pilot, and public-claim touched surfaces.
- Recorded required oracle/split evidence fields before any proof-owner split
  begins.
- Added Day 1 sprint intake artifact:
  `artifacts/day1-sprint-intake.md`.
- Kept Day 1 documentation-only; no C source, header, build, workflow,
  package, benchmark, or test surfaces were modified.

## Day 2 Notes

- Inspected Sprint 120 Day 2 requirements and direct-solver audit scope.
- Measured direct test hotspot sizes:
  - `tests/test_qr.c`: 3234 lines;
  - `tests/test_ldlt.c`: 3006 lines;
  - `tests/test_ldlt_csc.c`: 3915 lines;
  - `tests/test_direct_solver_helpers.h`: 93 lines;
  - `tests/test_solver_helpers.h`: 200 lines.
- Inspected direct oracle owner functions and main test registrations in:
  - `tests/test_qr.c`;
  - `tests/test_ldlt.c`;
  - `tests/test_ldlt_csc.c`.
- Inspected existing helper layers:
  - `tf_relative_residual_l2`;
  - `tf_block_relative_residual_l2`;
  - `tf_read_external_reference_vector`;
  - `tf_sparse_residual_norminf`;
  - `tf_assert_sparse_matrices_equal`.
- Mapped direct proof-owner clusters:
  - QR exact RHS, solve, true residual, SuiteSparse, QR-vs-LU, and refinement;
  - LDLT Matrix Market, KKT, inertia, cross-solver, cross-backend, backend
    telemetry, and dense-backend environment proofs;
  - LDLT CSC KKT fixtures, two-pass factor workflow, external dense reference,
    analysis-aware residuals, solve scenarios, inertia, and singular detection.
- Recorded initial split recommendations for Day 5 ranking:
  - consider QR exact-RHS/residual helper extraction;
  - consider QR solve scenario split;
  - consider LDLT KKT fixture helper extraction;
  - consider LDLT cross-backend scenario split;
  - defer LDLT CSC external dense-reference split unless Day 4 designs a
    state owner;
  - consider LDLT CSC solve split later after shared fixture design.
- Added Day 2 direct audit artifact:
  `artifacts/day2-direct-oracle-ownership-audit.md`.
- Kept Day 2 documentation-only; no C source, header, build, workflow,
  package, benchmark, or test surfaces were modified.

## Day 3 Notes

- Inspected Sprint 120 Day 3 requirements and iterative-solver audit scope.
- Measured iterative test hotspot sizes:
  - `tests/test_iterative.c`: 2,924 lines;
  - `tests/test_bicgstab.c`: 1,826 lines;
  - `tests/test_minres.c`: 1,649 lines;
  - `tests/test_iterative_handle_helpers.h`: 195 lines;
  - `tests/test_solver_helpers.h`: 200 lines.
- Inspected iterative oracle owner functions and main test registrations in:
  - `tests/test_iterative.c`;
  - `tests/test_bicgstab.c`;
  - `tests/test_minres.c`;
  - `tests/test_iterative_handle_helpers.h`.
- Mapped iterative proof-owner clusters:
  - CG generated RHS, residual accuracy, SuiteSparse, Cholesky comparison,
    preconditioner, public handle, and matrix-free proofs;
  - GMRES restart, Arnoldi, SuiteSparse, LU/CG comparison, right-preconditioner,
    public handle, and matrix-free proofs;
  - BiCGSTAB exact RHS, true residual, ILU/ILUT, SuiteSparse, LU/GMRES
    comparison, numerical hardening, block, and matrix-free proofs;
  - MINRES SPD, indefinite, KKT, preconditioner, LDLT/GMRES comparison,
    scaled/ill-conditioned, early-termination, and block proofs.
- Recorded callback and progress-like proof owners:
  - matrix-free matvec callbacks;
  - failing matvec callbacks;
  - preconditioner callbacks;
  - public handle reuse and growth;
  - verbose execution paths;
  - block-solver result aggregation.
- Recorded Day 4 shared-fixture inputs for exact-RHS builders, residual
  measurement helpers, matrix builders, callback fixtures, block RHS fixtures,
  and SuiteSparse/cross-solver reference helpers.
- Recorded initial split recommendations for Day 5 ranking:
  - consider GMRES SuiteSparse/restart/right-preconditioner proof split;
  - consider BiCGSTAB block and matrix-free proof splits;
  - consider MINRES block and preconditioner/direct-comparison proof splits;
  - defer public iterative handle movement until helper placement rules are
    designed;
  - keep solver-specific tolerances, residual interpretation, convergence
    outcomes, and failure modes at named test boundaries.
- Added Day 3 iterative audit artifact:
  `artifacts/day3-iterative-oracle-ownership-audit.md`.
- Kept Day 3 documentation-only; no C source, header, build, workflow,
  package, benchmark, or test surfaces were modified.

## Day 4 Notes

- Inspected Sprint 120 Day 4 requirements and shared-fixture design scope.
- Compared Day 2 direct audit inputs with Day 3 iterative audit inputs across:
  - generated RHS from known solution;
  - matrix construction;
  - residual measurement;
  - external/dense reference comparisons;
  - callback and preconditioner fixtures;
  - cleanup and lifecycle ownership.
- Defined a three-layer fixture architecture:
  - measurement helpers that compute quantities without deciding outcomes;
  - fixture builders that build matrices, RHS vectors, callbacks, and block
    inputs without asserting solver behavior;
  - scenario-local proof owners that retain solver-specific interpretation.
- Defined candidate helper families:
  - exact RHS builders;
  - matrix builders;
  - residual measurement helpers;
  - external/reference solver wrappers;
  - callback fixtures;
  - block RHS fixtures.
- Recorded solver-local responsibility boundaries for:
  - QR;
  - LDLT linked-list;
  - LDLT CSC;
  - CG;
  - GMRES;
  - BiCGSTAB;
  - MINRES;
  - public handles;
  - public claims.
- Recorded helper placement rules for:
  - `tests/test_solver_helpers.h`;
  - `tests/test_direct_solver_helpers.h`;
  - future narrow solver-family helper headers;
  - scenario-local static helpers.
- Added naming rules, source/build/CTest impact rules, rollback instructions,
  and the Day 5 candidate selection checklist.
- Added Day 4 shared fixture architecture artifact:
  `artifacts/day4-shared-fixture-architecture.md`.
- Kept Day 4 documentation-only; no C source, header, build, workflow,
  package, benchmark, or test surfaces were modified.

## Day 5 Notes

- Inspected Sprint 120 Day 5 requirements and split-ranking scope.
- Re-read Day 2 direct, Day 3 iterative, and Day 4 shared-fixture artifacts.
- Checked existing Makefile and CMake test membership for:
  - `test_qr`;
  - `test_iterative`;
  - `test_minres`;
  - `test_bicgstab`;
  - `source-list-check`;
  - CMake `add_sparse_test(...)` registration.
- Ranked direct split candidates by proof value, risk, rollback cost, and Day 4
  solver-local responsibility constraints.
- Selected the direct split batch:
  - QR solve scenario owner split from `tests/test_qr.c`;
  - preferred future target: `tests/test_qr_solve.c` if Day 6 accepts a new
    CTest owner;
  - QR exact RHS, reported residual, fixture-specific tolerances, and QR-vs-LU
    claim boundaries remain scenario-local.
- Ranked iterative split candidates by proof value, risk, rollback cost, and
  callback/block/convergence responsibility constraints.
- Selected the iterative split batch:
  - block BiCGSTAB scenario owner split from `tests/test_bicgstab.c`;
  - preferred future target: `tests/test_bicgstab_block.c` if Day 9 accepts a
    new CTest owner;
  - per-column status, aggregate result semantics, preconditioner failure, and
    block cleanup remain scenario-local.
- Recorded focused validation commands for selected new-executable and
  helper-only paths.
- Recorded source-list, Makefile, CMake, expected CTest-count, full quality,
  rollback, and deferral criteria for selected batches.
- Recorded residual deferred candidate queue for LDLT, LDLT CSC, block MINRES,
  GMRES, matrix-free BiCGSTAB, and public iterative handle movement.
- Added Day 5 split ranking and proof plan artifact:
  `artifacts/day5-split-ranking-proof-plan.md`.
- Kept Day 5 documentation-only; no C source, header, build, workflow,
  package, benchmark, or test surfaces were modified.

## Day 6 Notes

- Inspected Sprint 120 Day 6 requirements and direct split design scope.
- Re-read the Day 5 split ranking and proof plan for the selected direct
  batch.
- Inspected QR solve functions and registrations in `tests/test_qr.c`:
  - `test_qr_solve_square`;
  - `test_qr_solve_overdetermined`;
  - `test_qr_solve_analytical`;
  - `test_qr_solve_rank_deficient`;
  - `test_qr_solve_nos4`;
  - `test_qr_solve_null_residual`;
  - `test_qr_bcsstk04`;
  - `test_qr_west0067`;
  - `test_qr_vs_lu`;
  - `test_qr_tall_synthetic`.
- Inspected QR helper ownership for:
  - `compute_rel_residual`;
  - `assert_qr_true_residual_below`;
  - `make_qr_exact_rhs`;
  - `qr_idx_count_bytes`;
  - `qr_reconstruction_error`;
  - `assert_qr_reconstruction_below`;
  - `make_qr_duplicate_column_4x3`.
- Defined the Day 7 direct split target:
  - new focused owner `tests/test_qr_solve.c`;
  - `tests/test_qr.c` remains owner for non-solve QR coverage.
- Defined required Day 7 build metadata updates:
  - add `$(TESTDIR)/test_qr_solve.c` to Makefile `TEST_SRCS`;
  - add `add_sparse_test(test_qr_solve)` after `test_qr` in `CMakeLists.txt`;
  - expect CTest count to increase by one if the new executable is added.
- Recorded focused direct validation commands, full-quality requirements,
  expected QR solve behavior, and rollback checklist.
- Added Day 6 direct split implementation checklist artifact:
  `artifacts/day6-direct-split-implementation-checklist.md`.
- Kept Day 6 documentation-only; no C source, header, build, workflow,
  package, benchmark, or test surfaces were modified.

## Day 7 Notes

- Implemented the selected direct split batch from Day 6.
- Added focused QR solve owner:
  - `tests/test_qr_solve.c`.
- Moved QR solve scenario tests out of `tests/test_qr.c` and into
  `tests/test_qr_solve.c`:
  - `test_qr_solve_square`;
  - `test_qr_solve_overdetermined`;
  - `test_qr_solve_analytical`;
  - `test_qr_solve_rank_deficient`;
  - `test_qr_solve_nos4`;
  - `test_qr_solve_null_residual`;
  - `test_qr_bcsstk04`;
  - `test_qr_west0067`;
  - `test_qr_vs_lu`;
  - `test_qr_tall_synthetic`.
- Preserved QR solve-local behavior in the new owner:
  - reported residual versus true residual assertions;
  - `A * [1, 2, ...]` generated-RHS semantics;
  - SuiteSparse fixture-specific tolerances;
  - QR-vs-LU bounded comparison assertions;
  - reconstruction assertions for mixed solve/reconstruction cases.
- Updated build metadata:
  - added `$(TESTDIR)/test_qr_solve.c` to Makefile `TEST_SRCS`;
  - added `add_sparse_test(test_qr_solve)` after `test_qr` in
    `CMakeLists.txt`.
- Because C source and build metadata changed, Day 7 requires focused QR proof,
  source-list/CMake membership proof, and `make format && make lint &&
  make test`.
- Ran Day 7 validation:
  - `make format`;
  - `make build/test_qr build/test_qr_solve && ./build/test_qr &&
    ./build/test_qr_solve`;
  - `make source-list-check`;
  - `cmake -S . -B build/quality-review-cmake`;
  - `cmake --build build/quality-review-cmake --parallel 1 --clean-first`;
  - `ctest -N --test-dir build/quality-review-cmake`;
  - `make lint`;
  - `make test`.
- Added Day 7 direct split implementation artifact:
  `artifacts/day7-direct-split-implementation.md`.

## Day 8 Notes

- Inspected Sprint 120 Day 8 requirements and direct validation/consolidation
  scope.
- Re-read the Day 7 implementation artifact and current QR split diff.
- Re-ran focused direct validation:
  - `make build/test_qr build/test_qr_solve && ./build/test_qr &&
    ./build/test_qr_solve`;
  - result: `test_qr` passed 63 tests and `test_qr_solve` passed 10 tests.
- Re-ran source-list proof:
  - `make source-list-check`;
  - result: passed with 49 library sources.
- Re-ran CTest membership inspection:
  - `ctest -N --test-dir build/quality-review-cmake`;
  - result: 55 tests total, with `test_qr_solve` registered as test #21.
- Reviewed direct diff boundaries:
  - QR solve tolerances stayed local to `tests/test_qr_solve.c`;
  - reported-versus-true residual semantics stayed visible;
  - generated RHS remains `A * [1, 2, ...]`;
  - mixed reconstruction/solve checks remain visible for `bcsstk04` and tall
    synthetic solve;
  - QR-vs-LU remains a bounded comparison only;
  - no public headers, README, package files, examples, workflows, or API docs
    were modified.
- Consolidated direct residual queue:
  - LDLT Matrix Market/KKT fixture helper extraction;
  - LDLT cross-backend scenario split;
  - LDLT CSC solve scenario split;
  - LDLT CSC external dense-reference split;
  - QR reconstruction/sparse-mode split;
  - QR exact-RHS/residual shared helper extraction.
- Recorded Day 9 iterative readiness checklist for the selected block
  BiCGSTAB split.
- Added Day 8 direct validation and consolidation artifact:
  `artifacts/day8-direct-validation-consolidation.md`.
- Kept Day 8 implementation surfaces unchanged after Day 7; Day 8 added
  documentation-only consolidation after revalidation.

## Day 9 Notes

- Inspected Sprint 120 Day 9 requirements and the selected Day 5 iterative
  split batch.
- Inspected the block BiCGSTAB region in `tests/test_bicgstab.c`:
  - `test_block_bicgstab_null_inputs`;
  - `test_block_bicgstab_nrhs_zero`;
  - `test_block_bicgstab_nrhs_negative`;
  - `test_block_bicgstab_nonsquare`;
  - `test_block_bicgstab_2rhs`;
  - `test_block_bicgstab_4rhs`;
  - `test_block_bicgstab_matches_single_rhs`;
  - `test_block_bicgstab_mixed_convergence`;
  - `test_block_bicgstab_nrhs_1`;
  - `test_block_bicgstab_preconditioned`;
  - `test_block_bicgstab_result_aggregation`;
  - `test_block_bicgstab_error_propagation`.
- Identified the local block failure helper:
  - `failing_precond`.
- Defined the Day 10 iterative split target:
  - new focused owner `tests/test_bicgstab_block.c`;
  - `tests/test_bicgstab.c` remains owner for scalar BiCGSTAB,
    SuiteSparse, numerical hardening, matrix-free, callback, and adjacent
    comparison proofs.
- Defined helper ownership for Day 10:
  - copy narrow static `build_identity` and `build_unsym_tridiag` helpers into
    the new block owner;
  - keep RHS construction inline in scenario tests;
  - reuse `tf_relative_residual_l2` with local tolerances;
  - keep ILU preconditioner and `failing_precond` behavior scenario-local.
- Defined required Day 10 build metadata updates:
  - add `$(TESTDIR)/test_bicgstab_block.c` after
    `$(TESTDIR)/test_bicgstab.c` in Makefile `TEST_SRCS`;
  - add `add_sparse_test(test_bicgstab_block)` after
    `add_sparse_test(test_bicgstab)` in `CMakeLists.txt`;
  - expect CTest count to increase from 55 to 56 after the new executable is
    added.
- Recorded focused iterative validation commands, full-quality requirements,
  block convergence and aggregation expectations, callback/non-callback
  boundaries, and rollback checklist.
- Added Day 9 iterative split implementation checklist artifact:
  `artifacts/day9-iterative-split-implementation-checklist.md`.
- Kept Day 9 documentation-only; no C source, header, build, workflow,
  package, benchmark, or test surfaces were modified.

## Day 10 Notes

- Implemented the selected iterative split batch from Day 9.
- Added focused block BiCGSTAB owner:
  - `tests/test_bicgstab_block.c`.
- Moved block BiCGSTAB scenario tests out of `tests/test_bicgstab.c` and into
  `tests/test_bicgstab_block.c`:
  - `test_block_bicgstab_null_inputs`;
  - `test_block_bicgstab_nrhs_zero`;
  - `test_block_bicgstab_nrhs_negative`;
  - `test_block_bicgstab_nonsquare`;
  - `test_block_bicgstab_2rhs`;
  - `test_block_bicgstab_4rhs`;
  - `test_block_bicgstab_matches_single_rhs`;
  - `test_block_bicgstab_mixed_convergence`;
  - `test_block_bicgstab_nrhs_1`;
  - `test_block_bicgstab_preconditioned`;
  - `test_block_bicgstab_result_aggregation`;
  - `test_block_bicgstab_error_propagation`.
- Moved the local error-propagation helper:
  - `failing_precond`.
- Preserved block-local behavior in the new owner:
  - null, `nrhs`, and nonsquare argument handling;
  - two-RHS and four-RHS residual thresholds;
  - scalar/block equivalence checks;
  - mixed-convergence aggregate iteration visibility;
  - ILU-preconditioned block solve behavior;
  - aggregate result field semantics;
  - preconditioner error propagation through `SPARSE_ERR_SINGULAR`.
- Left matrix-free callback tests, scalar BiCGSTAB tests, SuiteSparse tests,
  numerical hardening, and adjacent comparison proofs in
  `tests/test_bicgstab.c`.
- Updated build metadata:
  - added `$(TESTDIR)/test_bicgstab_block.c` to Makefile `TEST_SRCS`;
  - added `add_sparse_test(test_bicgstab_block)` after
    `add_sparse_test(test_bicgstab)` in `CMakeLists.txt`.
- Ran Day 10 validation:
  - `make format`;
  - `make build/test_bicgstab build/test_bicgstab_block &&
    ./build/test_bicgstab && ./build/test_bicgstab_block`;
  - `make source-list-check`;
  - `cmake -S . -B build/quality-review-cmake`;
  - `cmake --build build/quality-review-cmake --parallel 1 --clean-first`;
  - `ctest -N --test-dir build/quality-review-cmake`;
  - `make lint`;
  - `make test`.
- Validation results:
  - focused `test_bicgstab` passed 49 tests with 0 failures;
  - focused `test_bicgstab_block` passed 12 tests with 0 failures;
  - source-list check passed with 49 library sources;
  - CTest registered `test_bicgstab_block` as test #40;
  - CTest total increased from 55 to 56 as expected;
  - `make lint` passed;
  - `make test` passed all tests.
- Added Day 10 iterative split implementation artifact:
  `artifacts/day10-iterative-split-implementation.md`.

## Day 11 Notes

- Inspected Sprint 120 Day 11 requirements and the bounded cross-solver oracle
  pilot scope.
- Re-read Day 2 direct audit, Day 3 iterative audit, Day 4 shared fixture
  architecture, Day 5 split ranking, and Day 10 implementation evidence.
- Inspected existing cross-solver comparison owners:
  - QR-vs-LU in `tests/test_qr_solve.c`;
  - CG-vs-Cholesky and GMRES-vs-LU/CG in `tests/test_iterative.c`;
  - BiCGSTAB-vs-LU/GMRES in `tests/test_bicgstab.c`;
  - MINRES-vs-CG/LDLT/GMRES in `tests/test_minres.c`;
  - LDLT-vs-LU/Cholesky in `tests/test_ldlt.c`.
- Selected the bounded Day 12 pilot:
  - new focused owner `tests/test_cross_solver_oracle.c`;
  - one small SPD generated-RHS fixture;
  - solver set limited to LU, Cholesky, QR, and CG;
  - generated exact solution plus residual checks as the oracle;
  - no SuiteSparse, external reference, package, platform, benchmark,
    preconditioner, block-solver, or public API scope.
- Defined pilot tolerances:
  - relative residuals below `1e-10` for LU, Cholesky, QR, and CG;
  - max `|x_solver - x_exact|` below `1e-8`;
  - CG must converge with `max_iter = 100` and `tol = 1e-12`.
- Defined Day 12 build metadata expectations:
  - add `$(TESTDIR)/test_cross_solver_oracle.c` to Makefile `TEST_SRCS`;
  - add `add_sparse_test(test_cross_solver_oracle)` to `CMakeLists.txt`;
  - expect CTest count to increase from 56 to 57 if the new executable is
    added.
- Recorded focused validation commands, full-quality requirements, rollback
  checklist, drift checks, and non-claim wording.
- Added Day 11 cross-solver oracle pilot design artifact:
  `artifacts/day11-cross-solver-oracle-pilot-design.md`.
- Kept Day 11 documentation-only; no C source, header, build, workflow,
  package, benchmark, or test surfaces were modified.

## Day 12 Notes

- Implemented the bounded cross-solver oracle pilot selected on Day 11.
- Added new focused test owner:
  - `tests/test_cross_solver_oracle.c`.
- Implemented one generated-RHS SPD fixture:
  - `8 x 8` tridiagonal matrix;
  - diagonal entries `4.0`;
  - symmetric off-diagonal entries `-1.0`;
  - deterministic exact solution `x_exact[i] = 1.0 + 0.25 * i`;
  - RHS computed locally as `b = A * x_exact`.
- Added one pilot test:
  - `test_spd_generated_rhs_lu_chol_qr_cg_agree`.
- Covered the bounded solver set from the Day 11 design:
  - LU with partial pivoting;
  - Cholesky;
  - QR;
  - CG with `max_iter = 100` and `tol = 1e-12`.
- Preserved local acceptance thresholds:
  - relative residual below `1e-10`;
  - max `|x_solver - x_exact|` below `1e-8`;
  - QR reported residual below `1e-10`;
  - CG must report convergence.
- Updated build metadata:
  - added `$(TESTDIR)/test_cross_solver_oracle.c` to Makefile `TEST_SRCS`;
  - added `add_sparse_test(test_cross_solver_oracle)` to `CMakeLists.txt`.
- Ran focused pilot validation:
  - `make format`;
  - `make build/test_cross_solver_oracle && ./build/test_cross_solver_oracle`.
- Focused pilot output:
  - LU relative residual `2.052e-16`, max solution difference
    `4.441e-16`;
  - Cholesky relative residual `2.163e-16`, max solution difference
    `4.441e-16`;
  - QR relative residual `3.299e-16`, max solution difference
    `4.441e-16`;
  - CG relative residual `2.490e-16`, max solution difference
    `4.441e-16`;
  - 1 test passed, 0 failed, 20 assertions.
- Ran build-surface validation:
  - `make source-list-check`;
  - `cmake -S . -B build/quality-review-cmake`;
  - `cmake --build build/quality-review-cmake --parallel 1 --clean-first`;
  - `ctest -N --test-dir build/quality-review-cmake`.
- CMake/CTest evidence:
  - `test_cross_solver_oracle` registered as CTest test #41;
  - reviewed CTest total increased from 56 to 57;
  - source-list check passed with 49 library sources.
- Ran `make lint`; it passed.
- Ran `make test`; it passed all tests.
- Added Day 12 cross-solver oracle pilot implementation artifact:
  `artifacts/day12-cross-solver-oracle-pilot-implementation.md`.
- Kept public documentation, API headers, package metadata, workflow metadata,
  benchmark surfaces, and platform claims unchanged.

## Day 13 Notes

- Re-ran the Sprint 120 validation package for touched direct, iterative, pilot,
  source-list, CMake, CTest, and quality-gate surfaces.
- Ran `make format`; it passed.
- Ran focused build and test validation:
  - `make build/test_qr build/test_qr_solve build/test_bicgstab
    build/test_bicgstab_block build/test_cross_solver_oracle`;
  - `./build/test_qr`;
  - `./build/test_qr_solve`;
  - `./build/test_bicgstab`;
  - `./build/test_bicgstab_block`;
  - `./build/test_cross_solver_oracle`.
- Focused validation results:
  - `test_qr`: 63 tests passed, 0 failed, 0 skipped;
  - `test_qr_solve`: 10 tests passed, 0 failed, 0 skipped;
  - `test_bicgstab`: 49 tests passed, 0 failed, 0 skipped;
  - `test_bicgstab_block`: 12 tests passed, 0 failed, 0 skipped;
  - `test_cross_solver_oracle`: 1 test passed, 0 failed, 0 skipped.
- Re-ran source-list and reviewed CMake membership validation:
  - `make source-list-check`;
  - `cmake -S . -B build/quality-review-cmake`;
  - `cmake --build build/quality-review-cmake --parallel 1 --clean-first`;
  - `ctest -N --test-dir build/quality-review-cmake`.
- CMake/CTest evidence:
  - source-list check passed with 49 library sources;
  - reviewed CMake build passed;
  - reviewed CTest total remained 57;
  - `test_qr_solve` registered as test #21;
  - `test_bicgstab_block` registered as test #40;
  - `test_cross_solver_oracle` registered as test #41.
- Ran full quality gates required by the branch's `.c` and build metadata
  changes:
  - `make lint`;
  - `make test`.
- Full quality results:
  - `make lint` passed;
  - `make test` passed all tests.
- Recorded skipped supplemental lanes:
  - no local Windows/MSVC lane;
  - no package/install lane because public install, package, ABI, and header
    surfaces were unchanged;
  - no benchmark execution because Sprint 120 does not make performance claims;
  - no external oracle expansion beyond existing focused tests because the Day
    12 pilot intentionally uses a bounded generated-RHS fixture.
- Added Day 13 validation package artifact:
  `artifacts/day13-validation-package.md`.

## Day 14 Notes

- Closed Sprint 120 with a consolidated oracle closeout artifact.
- Summarized all completed Sprint 120 outcomes:
  - sprint intake and evidence setup;
  - direct oracle ownership audit;
  - iterative oracle ownership audit;
  - shared fixture architecture;
  - split candidate ranking;
  - QR solve owner split;
  - block BiCGSTAB owner split;
  - bounded cross-solver oracle pilot;
  - validation package.
- Recorded files changed by the sprint:
  - `tests/test_qr.c`;
  - `tests/test_qr_solve.c`;
  - `tests/test_bicgstab.c`;
  - `tests/test_bicgstab_block.c`;
  - `tests/test_cross_solver_oracle.c`;
  - Makefile;
  - `CMakeLists.txt`;
  - Sprint 120 planning artifacts.
- Published the Day 13 validation summary in closeout form:
  - focused QR tests passed;
  - focused BiCGSTAB tests passed;
  - cross-solver oracle pilot passed;
  - source-list check passed with 49 library sources;
  - reviewed CTest count remained 57;
  - `make format`, `make lint`, and `make test` passed.
- Recorded residual direct/iterative/oracle cleanup queue:
  - LDLT Matrix Market and KKT helper extraction;
  - LDLT cross-backend scenario split;
  - LDLT CSC solve scenario split;
  - LDLT CSC external dense-reference split;
  - QR reconstruction or sparse-mode split;
  - QR exact-RHS/residual helper extraction;
  - block MINRES split;
  - GMRES SuiteSparse/restart/right-preconditioner split;
  - matrix-free BiCGSTAB callback split;
  - public iterative handle helper movement;
  - broad CG/GMRES shared fixture extraction;
  - external dense-reference oracle expansion;
  - package/install/platform validation for oracle tests.
- Recorded non-claims:
  - no new QR capability claim;
  - no new BiCGSTAB capability claim;
  - no broad direct solver parity claim;
  - no broad direct/iterative parity claim;
  - no external-oracle completeness claim;
  - no package, install, platform, or ABI claim;
  - no performance, scalability, or state-of-the-art claim.
- Added Sprint 121 handoff guidance:
  - keep proof-owner movement small and behavior-preserving;
  - preserve solver-local tolerances and failure modes when sharing fixtures;
  - keep Make/CMake/CTest count evidence current for new test owners;
  - design LDLT, LDLT CSC, GMRES, MINRES, matrix-free, and external-oracle
    residual work before movement;
  - continue writing explicit non-claims for maintainability-only work.
- Added Day 14 oracle closeout artifact:
  `artifacts/day14-oracle-closeout.md`.
