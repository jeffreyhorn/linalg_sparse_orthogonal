# Sprint 120 Day 4 Shared Fixture Architecture

## Purpose

Day 4 defines the shared direct/iterative fixture architecture that can be used
to rank and design Sprint 120 proof-owner splits. The architecture compares
the Day 2 direct oracle map and Day 3 iterative oracle map, identifies safe
test-only helper candidates, and records what must remain solver-local so
future source movement does not hide convergence, tolerance, residual,
callback, lifecycle, or publication boundaries.

This is a design artifact only. No C source, header, build, CMake, workflow, or
test membership changes are made by Day 4.

## Cross-Audit Comparison

| Shared need | Direct audit owners | Iterative audit owners | Architecture decision |
|---|---|---|---|
| Generated RHS from known solution | QR `make_qr_exact_rhs`, LDLT Matrix Market `A * ones`, LDLT CSC KKT solve inputs | CG/GMRES exact RHS helpers, BiCGSTAB sequential RHS, MINRES sequential/sine/scaled RHS | Create narrow test fixture helpers only after Day 5 selects a split. Helpers may construct `x_exact` and `b`; callers own tolerance, expected status, and residual interpretation. |
| Matrix construction | QR solve fixtures, LDLT KKT builders, LDLT CSC KKT/scaled KKT builders | Identity, diagonal, SPD tridiagonal, unsymmetric tridiagonal, Laplacian, symmetric indefinite, KKT fixtures | Builders may be grouped by matrix semantics, not by solver. Names must expose SPD, indefinite, unsymmetric, KKT, scaled, and SuiteSparse fixture intent. |
| Residual measurement | QR true residual, LDLT L2 residual, LDLT CSC relative infinity residual, direct helper residuals | CG/GMRES/BiCGSTAB/MINRES true residuals, block residuals, reported-vs-true checks | Shared helpers may compute L2, block L2, infinity, or raw norms. Test cases own pass/fail thresholds and reported-residual meaning. |
| External/dense reference | LDLT CSC external dense reference, QR/LDLT direct comparisons | GMRES/CG/LDLT/GMRES cross-solver comparisons, BiCGSTAB LU/GMRES references, MINRES LDLT/GMRES references | Reference helpers must keep reference solver, fixture name, platform skip behavior, and tolerance at the call site. Do not create broad parity claims. |
| Callback/preconditioner fixtures | Direct tests mostly use lifecycle and backend callbacks indirectly | Matrix-free callbacks, failing matvec callbacks, diagonal/Jacobi/ILU/IC preconditioners, handle reuse | Callback helper families must remain separated by behavior: matvec, failing matvec, preconditioner, handle lifecycle, and block aggregation. |
| Cleanup/lifecycle | QR factor/solve cleanup, LDLT backend/env cleanup, LDLT CSC permutation/analysis/factor cleanup | Public handles, preconditioner setup/free, matrix-free contexts, block result cleanup | Helpers may centralize mechanical cleanup only when ownership is single and obvious. Multi-owner lifecycle stays in scenario-local helpers. |

## Architecture Overview

The fixture architecture has three layers:

| Layer | Responsibility | Allowed examples | Explicit non-goals |
|---|---|---|---|
| Measurement helpers | Compute a numerical quantity without deciding test outcome. | L2 relative residual, block L2 residual, infinity relative residual, vector comparison norm, reported-vs-true residual measurement. | Do not choose tolerances, expected convergence, accepted nonconvergence, solver status, or public support wording. |
| Fixture builders | Build matrices, exact solutions, RHS vectors, callback contexts, or block RHS inputs with clear ownership and cleanup. | Identity, diagonal, SPD tridiagonal, unsymmetric tridiagonal, symmetric indefinite tridiagonal, KKT, Laplacian, `A*x`, `A*ones`, sequential/sine/scaled exact vectors. | Do not assert solver behavior, inertia, restart behavior, iteration-count improvements, platform support, or cross-solver parity. |
| Scenario-local proof owners | Keep solver-specific interpretation, assertions, and result ownership near the named test. | QR solve, LDLT inertia, LDLT CSC two-pass factor, GMRES restart, right-preconditioned GMRES, BiCGSTAB breakdown/nonconvergence, MINRES indefinite/KKT, block solver aggregation. | Do not move behavior meaning into generic helpers or make broad claims from narrow proof blocks. |

## Candidate Helper Families

| Family | Candidate placement | Candidate API shape | Consumers | Design notes |
|---|---|---|---|---|
| Exact RHS builders | Future test-only helper header, likely adjacent to `tests/test_solver_helpers.h` if selected | Build `x_exact` and `b = A*x_exact` from a named vector generator | QR, LDLT, CG, GMRES, BiCGSTAB, MINRES | Support caller-owned vector generators: ones, sequential, sine, scaled sequential, and explicit vector. Return allocation status; caller owns assertions. |
| Matrix builders | Solver-family helper header only after a focused split proves value | Build named matrix fixtures with clear sparse format and cleanup rules | Direct and iterative split candidates | Prefer semantic names such as SPD tridiagonal, unsymmetric tridiagonal, symmetric indefinite tridiagonal, KKT, scaled KKT, Laplacian. Avoid a generic "test matrix" helper. |
| Residual measurement | Existing `tests/test_solver_helpers.h` plus direct-only helper where norm semantics differ | Compute L2 relative, block L2 relative, infinity relative, and optional raw norm | All solver proof owners | Existing helpers already cover part of this. Add only missing norm variants when selected by a split. Never change QR or LDLT CSC residual semantics silently. |
| External/reference solver wrappers | Scenario-local helper first; shared only after repeated safe use | Run/read reference vectors or compare local solver output with named reference solver | LDLT CSC, QR-vs-LU, LDLT-vs-LU/Cholesky, GMRES/BiCGSTAB/MINRES comparisons | Keep matrix name, reference solver, skip/error policy, and tolerance visible in the test. External process behavior must remain explicit. |
| Callback fixtures | Separate iterative callback helper only if Day 5 selects iterative callback split | Create matvec, failing matvec, diagonal/Jacobi preconditioner, and context cleanup fixtures | CG/GMRES/BiCGSTAB/MINRES | Do not merge matvec callbacks with preconditioner callbacks. Failure propagation and residual interpretation stay solver-local. |
| Block RHS fixtures | Iterative block helper if Day 5 ranks block split high | Build block RHS/expected solution columns and zero/mixed-convergence columns | Block BiCGSTAB, block MINRES | Preserve column ordering and per-column expected status at the call site. |

## Solver-Local Responsibility Table

| Solver / area | Must remain solver-local |
|---|---|
| QR | Reported residual interpretation, QR reconstruction semantics, square versus overdetermined residual budgets, rank-deficient behavior, refinement improvement semantics, SuiteSparse matrix-specific tolerance, and QR-vs-LU claim boundary. |
| LDLT linked-list | Inertia expectations, KKT interpretation, backend dispatch telemetry, dense-backend environment/fallback behavior, linked-list versus CSC agreement semantics, and LU/Cholesky comparison boundaries. |
| LDLT CSC | Two-pass factor lifecycle, permutation/unpermutation state, analysis-aware factor ownership, external dense-reference skip/error policy, relative infinity residual meaning, in-place solve behavior, AMD behavior, inertia, near-zero pivot, and singular block failure. |
| CG | SPD requirement, nonsymmetric/indefinite behavior, zero-RHS and exact-initial-guess convergence, maximum-iteration failure, diagonal preconditioner expectations, SuiteSparse tolerances, and Cholesky comparison boundary. |
| GMRES | Restart/unrestarted behavior, lucky breakdown, Arnoldi correctness, relaxed SuiteSparse outcomes, right-preconditioned reported-vs-true residual distinction, left/right residual comparison, and LU/CG comparison boundary. |
| BiCGSTAB | Breakdown/nonconvergence interpretation, true residual, known-solution references, ILU/ILUT iteration expectations, numerical hardening, matrix-free callback failure propagation, block aggregation, and expected small-budget nonconvergence. |
| MINRES | Symmetric-indefinite and KKT interpretation, SPD-vs-CG boundary, scaled/ill-conditioned tolerance behavior, early Lanczos termination, IC/Jacobi/exact preconditioner meaning, LDLT/GMRES comparison boundary, and block aggregation. |
| Public handles | Validation, prepare/reuse/growth behavior, workspace ownership, cleanup ownership, and failure paths. |
| Public claims | Any statement that could imply broad solver parity, external-oracle completeness, platform support, package support, state-of-the-art performance, or public API expansion. |

## Helper Header Boundaries

| Placement | Use when | Boundary |
|---|---|---|
| `tests/test_solver_helpers.h` | A helper is broadly numerical, format-neutral, test-only, and already aligned with existing shared residual/reference helpers. | Measurement and small generic allocation helpers only. No solver status interpretation. |
| `tests/test_direct_solver_helpers.h` | A helper is direct-only and depends on direct sparse solve/factor semantics. | Direct residual or matrix comparison helpers only; no QR/LDLT-specific assertions unless named direct scenario helper remains local. |
| New solver-family helper header | A selected split needs repeated fixture construction across newly split files. | Header must have a narrow owner name, explicit cleanup contract, and no broad "oracle" abstraction. |
| Scenario-local static helper | Behavior is solver-specific, lifecycle-heavy, or has platform/reference-solver policy. | Default placement for LDLT CSC two-pass state, GMRES right preconditioning, block aggregation, and cross-solver comparison setup. |

## Naming Rules

- Name helpers for what they build or measure, not for the outcome they hope to
  prove.
- Include matrix semantics in builder names: `spd`, `unsym`, `indef`, `kkt`,
  `scaled`, `laplacian`, or the SuiteSparse fixture name.
- Include norm semantics in residual helpers: `l2`, `block_l2`, `norminf`, or
  `reported_vs_true`.
- Include callback role in callback helpers: `matvec`, `failing_matvec`,
  `precond`, `handle`, or `block`.
- Avoid names that imply product breadth, such as generic "state of the art",
  "full parity", "complete oracle", or "validated platform" helpers.

## Source, Build, and CTest Impact Rules

| Change type | Required handling |
|---|---|
| Documentation-only design/ranking | `git diff --check` and focused trailing-whitespace scan. |
| New or changed `.c` or `.h` helper/test file | `make format && make lint && make test`. |
| Makefile source-list membership change | `make source-list-check` plus the focused build/test target for the moved owner. |
| CMake test/source membership change | CMake configure/build and `ctest -N` count proof as affected. |
| Split that changes test executable boundaries | Record old/new CTest membership, expected count, focused test command, and rollback path. |
| External reference or platform behavior touched | Record reviewed versus supplemental lane, skip behavior, platform assumptions, and non-claim wording. |

## Rollback Instructions

Every Day 5+ split proposal must include:

1. The exact old owner file and new owner/helper file.
2. The source-list, Makefile, CMake, and CTest membership delta.
3. The focused validation command that proves behavior before full quality.
4. The command or patch path to restore the old owner if validation fails.
5. The residual owner if the split is deferred.
6. The non-claim statement that remains true after rollback or deferral.

If a helper begins to own solver-specific tolerance, status, convergence,
iteration-count, platform, lifecycle, or publication meaning, rollback the
helper and keep the behavior in the scenario-local test.

## Day 5 Selection Checklist

Use this checklist to rank each candidate:

| Question | Required answer before selection |
|---|---|
| What proof value improves? | Name the exact direct or iterative owner, line-count/hotspot reduction, and behavior made easier to maintain. |
| What behavior stays local? | List tolerances, convergence thresholds, residual semantics, failure classes, lifecycle, and callback semantics that remain in named tests. |
| What files change? | Name old files, new files, helper headers, Makefile/source-list/CMake changes, and expected CTest impact. |
| What focused proof runs first? | Name focused test commands before full quality. |
| What is the rollback path? | Name the restoration path and residual owner if validation fails. |
| What claims are not made? | State that the split does not broaden parity, platform, package, public API, performance, or external-oracle claims. |

## Completion Criteria Check

| Criterion | Status |
|---|---|
| Item 2 shared fixture design is complete | Complete: shared fixture families, placement, solver-local responsibilities, build impact, and rollback expectations are documented. |
| Shared helpers have explicit non-goals | Complete: helpers may build or measure, but must not own solver-specific tolerance, convergence, status, lifecycle, callback, platform, or claim meaning. |
| Direct and iterative split candidates can be evaluated against the same architecture | Complete: Day 5 selection checklist applies the same proof-value, behavior-locality, file-impact, validation, rollback, and non-claim criteria to both direct and iterative candidates. |

## Non-Claims

This design does not claim new solver behavior, broader direct/iterative
parity, complete external-oracle coverage, package/install support, platform
support, public API expansion, benchmark improvement, or state-of-the-art
validation. It only defines a test-fixture architecture for bounded Sprint 120
proof-owner cleanup.
