# Sprint 103 Working Notes

## Sprint Context

Sprint 103 implements "Iterative, Eigensolver & SVD External Comparison" from
`docs/planning/EPIC_10/PROJECT_PLAN.md`. The sprint raises evidence quality for
iterative solvers, eigensolvers, and SVD while keeping external-comparison
claims bounded to named fixtures, solver paths, tolerances, validation
commands, and unsupported cases.

## Validation Rules

Validation must scale with the touched surface:

| touched surface | required validation |
|---|---|
| planning documentation only | `git diff --check`; trailing-whitespace scan on touched planning files |
| public documentation only | `git diff --check`; trailing-whitespace scan on touched docs |
| helper script only | focused helper invocation, if executable; docs hygiene |
| test `.c` file | focused test binary; `make format`; `make lint`; `make test` |
| library `.c` or public `.h` file | focused affected tests; `make format`; `make lint`; `make test` |
| build or CMake surface | focused Make/CMake configure or build check plus any code-touch gate |
| workflow or package surface | focused workflow/package command where runnable plus any code-touch gate |

If any `.c` or `.h` file is modified, the full required quality chain is:

```sh
make format && make lint && make test
```

All required checks must pass before closeout or PR creation.

## Claim Boundaries

Sprint 103 may earn only bounded iterative, eigensolver, or SVD evidence claims
tied to named fixtures, helper behavior, solver family, validation commands,
and tolerance rules.

Sprint 103 must not claim:

- broad parity with SciPy, ARPACK, LAPACK, SuiteSparse, or other mature sparse
  linear algebra packages;
- external-oracle coverage for every iterative, eigensolver, or SVD path;
- portable performance superiority from correctness or residual fixtures;
- direct compressed solver API parity unless future implementation explicitly
  adds and validates those paths;
- one convergence or spectral fixture proves state-of-the-art solver quality;
- timing output is a benchmark sentinel unless a later artifact defines a
  machine class, fixture, threshold, and non-claim wording.

## Day 1 - Scope and Comparison Baseline

### Goal

Convert the Sprint 103 project-plan section and Sprint 100/102 handoffs into a
bounded comparison-evidence package with clear workstreams, validation
expectations, and non-claim boundaries.

### Actions

- Re-read the Sprint 103 section of
  `docs/planning/EPIC_10/PROJECT_PLAN.md`.
- Re-read Sprint 100 solver comparison evidence rules:
  - `docs/planning/EPIC_10/SPRINT_100/artifacts/day9-solver-comparison-template.md`
  - `docs/planning/EPIC_10/SPRINT_100/artifacts/templates/solver-comparison-evidence-template.md`
- Re-read Sprint 102 closeout and handoff requirements:
  - `docs/planning/EPIC_10/SPRINT_102/artifacts/day14-closeout-and-handoff.md`
- Created the Sprint 103 artifacts directory.
- Recorded authoritative Day 1 inputs in
  `artifacts/day1-authoritative-inputs.txt`.
- Recorded the Sprint 103 scope baseline, day ownership, validation rules, and
  claim boundaries in `artifacts/day1-scope-baseline.md`.

### Findings

- Sprint 100 requires every solver comparison claim to name fixture set,
  oracle/reference behavior, tolerance or acceptance criteria, validation
  command, unsupported cases, and remaining non-claims.
- Sprint 102 gives Sprint 103 reusable external-reference helper conventions
  and a warning against promoting bounded fixture evidence into broad parity
  claims.
- Sprint 103 should begin with a solver-family audit before fixture design or
  implementation so CG, MINRES, BiCGSTAB, eigen, thick-restart, LOBPCG, and
  SVD paths are ranked by weakness and user impact.
- Fixture taxonomy must precede new oracle expansion so convergence,
  stagnation, restart, preconditioning, residual, and rank behavior have
  explicit expected outcomes.

### Validation Expectations

- Day 1 changes planning documentation only.
- Required checks:
  - `git diff --check`
  - trailing-whitespace scan on `docs/planning/EPIC_10/SPRINT_103`

### Validation Results

- `git diff --check`: passed.
- `rg -n "[ \t]+$" docs/planning/EPIC_10/SPRINT_103`: passed; no matches.

### Day 1 Exit State

Day 1 is complete. Sprint 103 now has working notes, authoritative inputs,
scope baseline, workstream ownership, validation expectations, and preserved
Sprint 100/102 claim boundaries.

## Day 2 - Solver Family Comparison Audit

### Goal

Inventory CG, MINRES, BiCGSTAB, eigen, thick-restart, LOBPCG, and SVD evidence
before fixture, helper, or implementation work begins.

### Actions

- Re-read Sprint 103 Day 2 in
  `docs/planning/EPIC_10/SPRINT_103/PLAN.md`.
- Inventoried iterative, spectral, SVD, stagnation, and integration test
  owners:
  - `tests/test_iterative.c`
  - `tests/test_minres.c`
  - `tests/test_bicgstab.c`
  - `tests/test_stagnation.c`
  - `tests/test_eigs.c`
  - `tests/test_eigs_thick_restart.c`
  - `tests/test_eigs_lobpcg.c`
  - `tests/test_svd.c`
  - `tests/test_sprint13_integration.c`
  - `tests/test_sprint29_integration.c`
- Counted `RUN_TEST` ownership concentration for the target files.
- Classified current evidence as internal consistency, deterministic
  reference, direct-solver cross-check, fixture corpus, property/invariant,
  smoke, or external helper.
- Ranked solver families by user impact, comparison gap, numerical risk, and
  validation cost.
- Recorded the Day 2 audit in
  `artifacts/day2-solver-family-comparison-audit.md`.

### Findings

- CG has high user impact but comparatively strong existing residual,
  SuiteSparse, tolerance, preconditioner, and direct-solver cross-check
  coverage.
- MINRES has broad symmetric-indefinite and block coverage, including LDLT and
  GMRES cross-checks, but lacks a consolidated external/deterministic
  comparison artifact.
- BiCGSTAB has the highest iterative comparison gap because nonsymmetric
  convergence is user-visible and current evidence lacks a bounded external or
  deterministic oracle lane.
- Grow-m eigensolver evidence includes useful closed-form and residual checks,
  but no external ARPACK/SciPy-style oracle.
- Thick-restart evidence is valuable but still leans on grow-m parity.
- LOBPCG has strong deterministic and preconditioning coverage, but residual
  and orthogonality comparison claims need fixture-specific artifact ownership.
- SVD has broad reconstruction, orthogonality, rank, and invariant coverage,
  but no external LAPACK/NumPy/SciPy comparison lane.

### Ranked Queue

1. BiCGSTAB deterministic or external-reference comparison lane.
2. LOBPCG residual and orthogonality comparison lane.
3. Thick-restart independent fixture comparison lane.
4. SVD singular-value/rank/reconstruction follow-through.
5. MINRES consolidated comparison artifact.
6. CG convergence-profile consumer lane.
7. Grow-m eigensolver documentation and residual interpretation.

### Validation Expectations

- Day 2 changes planning documentation only.
- Required checks:
  - `git diff --check`
  - trailing-whitespace scan on `docs/planning/EPIC_10/SPRINT_103`

### Validation Results

- `git diff --check`: passed.
- `rg -n "[ \t]+$" docs/planning/EPIC_10/SPRINT_103`: passed; no matches.

### Day 2 Exit State

Day 2 is complete. All named Sprint 103 solver families are classified and the
Day 3 fixture taxonomy can start from a ranked comparison queue.

## Day 3 - Convergence Fixture Taxonomy

### Goal

Define matrix families, convergence-profile classes, solver-family mappings,
acceptance criteria, and expected skip/failure rules before new Sprint 103
comparison tests or helpers are implemented.

### Actions

- Re-read Sprint 103 Day 3 in
  `docs/planning/EPIC_10/SPRINT_103/PLAN.md`.
- Re-read the Day 2 solver-family comparison audit and ranked expansion queue.
- Reviewed existing generated fixture patterns and Matrix Market corpus inputs
  used by iterative, eigensolver, LOBPCG, thick-restart, and SVD tests.
- Re-read the Sprint 102 fixture taxonomy to preserve the requirement that
  fixture class, expected status, reference behavior, tolerance, validation,
  unsupported cases, and non-claims are declared before implementation.
- Defined matrix family classes for SPD, symmetric-indefinite, nonsymmetric,
  ill-conditioned, rank-deficient, clustered-spectrum, and low-rank behavior.
- Defined convergence-profile classes for fast convergence, slow convergence,
  stagnation, tolerance sensitivity, restart sensitivity, preconditioner
  effectiveness, orthogonality sensitivity, and rank sensitivity.
- Recorded the Day 3 taxonomy in
  `artifacts/day3-convergence-fixture-taxonomy.md`.

### Findings

- Generated diagonal and Laplacian fixtures are the cleanest shared controls
  for exact eigenvalue, residual, and singular-value behavior.
- KKT fixtures are the common bridge between MINRES and shifted/nearest-sigma
  spectral coverage.
- `nos4`, `bcsstk04`, `west0067`, `steam1`, and `orsirr_1` remain useful corpus
  fixtures, but corpus behavior needs fixture-specific residual and convergence
  expectations.
- BiCGSTAB's first comparison lane should start from `nonsym-known-solution`
  and `nonsym-mm-medium` classes, not a broad external-package claim.
- LOBPCG and thick-restart work must separate eigenvalue agreement, eigenpair
  residual, vector orthogonality, restart behavior, and iteration counts.
- SVD work needs rank, singular-value, reconstruction, and orthogonality
  criteria declared before any external dense comparison is added.

### Validation Expectations

- Day 3 changes planning documentation only.
- Required checks:
  - `git diff --check`
  - trailing-whitespace scan on `docs/planning/EPIC_10/SPRINT_103`

### Validation Results

- `git diff --check`: passed.
- `rg -n "[ \t]+$" docs/planning/EPIC_10/SPRINT_103`: passed; no matches.

### Day 3 Exit State

Day 3 is complete. New comparison tests can now cite taxonomy entries before
implementation, and Day 4 can freeze helper/reporting boundaries against
explicit fixture and acceptance criteria.

## Day 4 - Helper and Reporting Boundary

### Goal

Review Sprint 102 helper conventions and current residual, orthogonality,
convergence-profile, and tolerance-reporting patterns before selecting any
Sprint 103 helper reuse or extraction.

### Actions

- Re-read Sprint 103 Day 4 in
  `docs/planning/EPIC_10/SPRINT_103/PLAN.md`.
- Re-read the Day 3 convergence fixture taxonomy.
- Reviewed Sprint 102 helper boundary and extraction artifacts:
  - `docs/planning/EPIC_10/SPRINT_102/artifacts/day4-oracle-helper-boundary.md`
  - `docs/planning/EPIC_10/SPRINT_102/artifacts/day5-oracle-helper-extraction.md`
- Reviewed the current external-reference vector helper in
  `tests/test_solver_helpers.h`.
- Searched iterative, stagnation, eigensolver, LOBPCG, thick-restart, and SVD
  tests for repeated residual, orthogonality, convergence, rank,
  reconstruction, and diagnostic reporting patterns.
- Recorded the helper reuse, reporting contract, skip/error behavior, and Day
  5 validation plan in
  `artifacts/day4-helper-reporting-boundary.md`.

### Findings

- Sprint 102's `tf_read_external_reference_vector(...)` is safe to reuse only
  for vector-valued helper commands that follow the exact `OK n` / `SKIP` /
  `ERROR` contract.
- Residual reporting is common across iterative solvers, but thresholds and
  expected statuses are fixture- and family-specific.
- Ritz residual, orthogonality, rank, and reconstruction helpers are repeated
  concepts, but not yet a safe shared extraction because spectral and SVD
  thresholds are not frozen until later days.
- Day 5 should freeze the BiCGSTAB iterative comparison batch before
  implementation rather than add a generic helper.

### Boundary Decision

- No new C helper extraction is selected for Day 5.
- Keep residual, convergence, Ritz residual, orthogonality, rank, and
  reconstruction semantics family-local.
- Reuse Sprint 102 external-reference status handling only if a future helper
  emits the exact vector-output contract.
- Use the Day 4 reporting contract for all new comparison artifacts.

### Validation Expectations

- Day 4 changes planning documentation only.
- Required checks:
  - `git diff --check`
  - trailing-whitespace scan on `docs/planning/EPIC_10/SPRINT_103`

### Validation Results

- `git diff --check`: passed.
- `rg -n "[ \t]+$" docs/planning/EPIC_10/SPRINT_103`: passed; no matches.

### Day 4 Exit State

Day 4 is complete. Day 5 can begin from a frozen helper/reporting boundary and
does not need a new shared C helper unless the iterative design proves one is
necessary.

## Day 5 - Iterative Oracle Batch Design

### Goal

Select the highest-value iterative solver comparison batch, bind it to Day 3
fixture classes, and freeze tolerances, ownership, validation commands, and
non-claims before implementation.

### Actions

- Re-read Sprint 103 Day 5 in
  `docs/planning/EPIC_10/SPRINT_103/PLAN.md`.
- Re-read Day 2 solver-family ranking, Day 3 fixture taxonomy, and Day 4
  helper/reporting boundary.
- Reviewed current BiCGSTAB coverage in `tests/test_bicgstab.c` and
  BiCGSTAB stagnation/breakdown coverage in `tests/test_stagnation.c`.
- Selected BiCGSTAB as the first iterative comparison family because Day 2
  ranked it highest by comparison gap and user impact.
- Deferred external helper work because Day 4 found no safe helper extraction
  need before the first iterative batch.
- Recorded the selected batch, fixture/tolerance matrix, file ownership,
  validation plan, and deferred follow-ups in
  `artifacts/day5-iterative-oracle-batch-design.md`.

### Selected Batch

1. `bicgstab_nonsym_known_5`: deterministic nonsymmetric known-solution solve,
   compared against LU and `x_true`.
2. `bicgstab_steam1_ilu_vs_gmres30`: corpus residual comparison using
   BiCGSTAB+ILU and GMRES(30)+ILU on `steam1`.
3. `bicgstab_small_budget_unsym_tridiag`: expected non-convergence boundary
   with a deliberately too-small iteration budget.

### Validation Expectations

- Day 5 changes planning documentation only.
- Required checks:
  - `git diff --check`
  - trailing-whitespace scan on `docs/planning/EPIC_10/SPRINT_103`

### Validation Results

- `git diff --check`: passed.
- `rg -n "[ \t]+$" docs/planning/EPIC_10/SPRINT_103`: passed; no matches.

### Day 5 Exit State

Day 5 is complete. Day 6 can implement the selected BiCGSTAB comparison batch
without adding a new shared helper or public API surface.

## Day 6 - Iterative Oracle Batch Implementation

### Goal

Implement the selected BiCGSTAB comparison tests from Day 5, preserve public
solver behavior, and run the required focused and full validation gates for
the touched `.c` file.

### Actions

- Re-read Sprint 103 Day 6 in
  `docs/planning/EPIC_10/SPRINT_103/PLAN.md`.
- Re-read the Day 5 iterative oracle batch design.
- Added a Sprint 103 comparison section to `tests/test_bicgstab.c`.
- Implemented:
  - `test_s103_bicgstab_nonsym_known_5_lu_reference`
  - `test_s103_bicgstab_steam1_ilu_vs_gmres30_reference`
  - `test_s103_bicgstab_small_budget_expected_nonconvergence`
- Registered the new tests in the existing `test_bicgstab` binary.
- Kept all implementation local to `tests/test_bicgstab.c`; no shared helper,
  public API, build-system, or external helper changes were needed.
- Recorded implementation evidence and validation results in
  `artifacts/day6-iterative-oracle-batch-implementation.md`.

### Findings

- The deterministic 5x5 nonsymmetric fixture converged in 5 iterations with
  relative residual `1.136e-16`, matching both `x_true` and LU within the
  declared tolerance.
- The `steam1` corpus comparison converged for both BiCGSTAB+ILU and
  GMRES(30)+ILU, with true residuals below `1e-4`.
- The deliberately small-budget nonsymmetric tridiagonal fixture returned
  `SPARSE_ERR_NOT_CONVERGED` with a finite residual, proving the expected
  non-convergence boundary without treating it as a solver regression.

### Validation Expectations

- Day 6 changed `tests/test_bicgstab.c`.
- Required checks:
  - `make build/test_bicgstab`
  - `./build/test_bicgstab`
  - `make format && make lint && make test`
  - `git diff --check`
  - trailing-whitespace scan on `tests/test_bicgstab.c` and
    `docs/planning/EPIC_10/SPRINT_103`

### Validation Results

- `make build/test_bicgstab`: passed.
- `./build/test_bicgstab`: passed; 61 tests, 0 failures, 0 skips,
  466 assertions.
- `make format && make lint && make test`: passed; `All tests passed.`
- `git diff --check`: passed.
- `rg -n "[ \t]+$" tests/test_bicgstab.c docs/planning/EPIC_10/SPRINT_103`:
  passed; no matches.

### Day 6 Exit State

Day 6 is complete. The selected BiCGSTAB comparison batch is implemented
without helper expansion or public API changes.

## Day 7 - Iterative Batch Closeout and Rerank

### Goal

Validate the iterative comparison batch after implementation, identify any
helper or reporting debt, and rerank the remaining spectral and SVD comparison
work before Day 8 starts.

### Actions

- Re-read Sprint 103 Day 7 in
  `docs/planning/EPIC_10/SPRINT_103/PLAN.md`.
- Re-read the Day 2 ranking, Day 3 fixture taxonomy, Day 5 iterative batch
  design, and Day 6 implementation artifact.
- Re-ran focused and affected-family iterative validation for BiCGSTAB,
  CG/GMRES/MINRES, and stagnation/breakdown coverage.
- Confirmed Day 6 did not introduce shared helper, public API, build-system,
  or external-helper debt.
- Reranked remaining Sprint 103 implementation work after the landed
  BiCGSTAB comparison evidence.
- Recorded the closeout, validation results, updated ranking, residual
  follow-up queue, and Day 8 handoff in
  `artifacts/day7-iterative-closeout-and-rerank.md`.

### Findings

- BiCGSTAB moved from the highest Day 2 comparison gap to a bounded evidence
  lane with three validated fixtures.
- The remaining highest-value Sprint 103 implementation work is now LOBPCG
  residual and orthogonality comparison evidence, followed by thick-restart
  independent fixture evidence.
- SVD remains important but should follow spectral scoping so rank,
  reconstruction, singular-value, and orthogonality claims use consistent
  threshold language.
- Deferred BiCGSTAB external-helper work needs explicit future ownership
  because Day 6 intentionally used constructed solutions, LU, and GMRES(30)
  internal cross-checks rather than an external solver helper.

### Validation Expectations

- Day 7 adds planning documentation only.
- Day 7 also reruns focused iterative validation because it is the closeout
  gate for Day 6 code.
- Required checks:
  - `make build/test_bicgstab build/test_iterative build/test_stagnation`
  - `./build/test_bicgstab`
  - `./build/test_iterative`
  - `./build/test_stagnation`
  - `git diff --check`
  - trailing-whitespace scan on `tests/test_bicgstab.c` and
    `docs/planning/EPIC_10/SPRINT_103`

### Validation Results

- `make build/test_bicgstab build/test_iterative build/test_stagnation`:
  passed; all targets already up to date.
- `./build/test_bicgstab`: passed; 61 tests, 0 failures, 0 skips,
  466 assertions.
- `./build/test_iterative`: passed; 80 tests, 0 failures, 0 skips,
  711 assertions.
- `./build/test_stagnation`: passed; 46 tests, 0 failures, 0 skips,
  308 assertions.

### Day 7 Exit State

Day 7 is complete. The iterative batch is validated and closed, the remaining
work is reranked, and Day 8 can design the LOBPCG/thick-restart spectral batch
without unresolved iterative helper debt.

## Day 8 - Eigensolver Oracle Batch Design

### Goal

Select focused eigen, thick-restart, and LOBPCG comparison cases with bounded
residual, eigenvalue, convergence-status, and orthogonality expectations before
spectral implementation begins.

### Actions

- Re-read Sprint 103 Day 8 in
  `docs/planning/EPIC_10/SPRINT_103/PLAN.md`.
- Re-read the Day 7 rerank and Day 3 fixture taxonomy.
- Reviewed the existing spectral test surface in:
  - `tests/test_eigs_lobpcg.c`
  - `tests/test_eigs_thick_restart.c`
  - `tests/test_eigs.c`
- Selected the Day 9 implementation batch:
  - LOBPCG closed-form Laplacian residual and orthogonality claim.
  - Thick-restart exact diagonal residual and orthogonality claim.
  - Optional LOBPCG `bcsstk04` preconditioned corpus residual enhancement if
    stable with eigenvectors.
- Defined fixture ownership, eigenvalue, residual, orthogonality, convergence,
  skip, and validation expectations.
- Identified SVD overlap opportunities for Day 10 without adding SVD scope to
  Day 9.
- Recorded the design in
  `artifacts/day8-eigensolver-oracle-batch-design.md`.

### Findings

- LOBPCG already has broad diagonal, Laplacian, SuiteSparse, preconditioner,
  nearest-sigma, and cross-backend coverage, but Sprint 103 still benefits
  from a claim-owned residual/orthogonality test tied directly to the Day 3
  taxonomy.
- Thick-restart already has substantial grow-m parity coverage; the most useful
  Day 9 addition is an exact-reference fixture that does not depend on grow-m
  as the oracle.
- The existing `bcsstk04` IC(0) versus LDLT comparison is valuable, but adding
  eigenvector residual checks should be conditional on local stability so Day 9
  does not turn a bounded design into a brittle corpus claim.
- SVD should reuse residual and orthogonality threshold language later, but Day
  9 should not change SVD files.

### Validation Expectations

- Day 8 changes planning documentation only.
- Required checks:
  - `git diff --check`
  - trailing-whitespace scan on `tests/test_bicgstab.c` and
    `docs/planning/EPIC_10/SPRINT_103`

### Validation Results

- `git diff --check`: passed.
- `rg -n "[ \t]+$" tests/test_bicgstab.c docs/planning/EPIC_10/SPRINT_103`:
  passed; no matches.

### Day 8 Exit State

Day 8 is complete when the design artifact is written, hygiene checks pass,
and Day 9 can implement the selected spectral batch without broad external
parity claims.

## Day 9 - Eigensolver Oracle Batch Implementation

### Goal

Implement the Day 8 spectral comparison batch while preserving public
eigensolver API behavior and keeping all claims fixture-specific.

### Actions

- Re-read Sprint 103 Day 9 in
  `docs/planning/EPIC_10/SPRINT_103/PLAN.md`.
- Re-read the Day 8 eigensolver oracle batch design.
- Added a file-local LOBPCG orthogonality helper in
  `tests/test_eigs_lobpcg.c`.
- Implemented `test_s103_lobpcg_laplacian30_smallest4_claim`.
- Enhanced the existing `test_lobpcg_ldlt_beats_ic0_on_bcsstk04` comparison
  to request eigenvectors and assert Ritz residuals and orthogonality for both
  IC(0) and LDLT runs.
- Added file-local thick-restart diagonal, Ritz-residual, and orthogonality
  helpers in `tests/test_eigs_thick_restart.c`.
- Implemented `test_s103_thick_restart_diag12_largest4_claim`.
- Registered the new tests in the existing `test_eigs_lobpcg` and
  `test_eigs_thick_restart` binaries.
- Recorded implementation evidence in
  `artifacts/day9-eigensolver-oracle-batch-implementation.md`.

### Findings

- The new LOBPCG Laplacian claim converges with closed-form eigenvalue,
  residual, and orthogonality evidence.
- The optional `bcsstk04` vector enhancement was stable locally, so the Day 8
  optional corpus residual/orthogonality comparison was kept.
- The new thick-restart diagonal claim uses exact eigenvalues instead of
  grow-m parity as the oracle.
- No public headers, library sources, build files, fixture files, or external
  helper contracts were changed.

### Validation Expectations

- Day 9 changed `.c` test files.
- Required checks:
  - `make build/test_eigs_lobpcg build/test_eigs_thick_restart`
  - `./build/test_eigs_lobpcg`
  - `./build/test_eigs_thick_restart`
  - `make format && make lint && make test`
  - `git diff --check`
  - trailing-whitespace scan on `tests/test_eigs_lobpcg.c`,
    `tests/test_eigs_thick_restart.c`, `tests/test_bicgstab.c`, and
    `docs/planning/EPIC_10/SPRINT_103`

### Validation Results

- `make build/test_eigs_lobpcg build/test_eigs_thick_restart`: passed.
- `./build/test_eigs_lobpcg`: passed; 27 tests, 0 failures, 0 skips,
  247 assertions.
- `./build/test_eigs_thick_restart`: passed; 21 tests, 0 failures, 0 skips,
  285 assertions.
- `make format && make lint && make test`: passed; `All tests passed.`
- `git diff --check`: passed.
- `rg -n "[ \t]+$" tests/test_eigs_lobpcg.c tests/test_eigs_thick_restart.c tests/test_bicgstab.c docs/planning/EPIC_10/SPRINT_103`:
  passed; no matches.

### Day 9 Exit State

Day 9 is complete. The selected spectral comparison batch is implemented,
focused and full validation passed, and the Day 10 spectral closeout can begin
from bounded LOBPCG and thick-restart residual/orthogonality evidence.

## Day 10 - Spectral Closeout and SVD Scope Freeze

### Goal

Validate the Day 9 spectral comparison work, review residual and orthogonality
evidence, and freeze a bounded SVD follow-through scope for Day 11.

### Actions

- Re-read Sprint 103 Day 10 in
  `docs/planning/EPIC_10/SPRINT_103/PLAN.md`.
- Re-read the Day 9 spectral implementation artifact.
- Reviewed current SVD coverage in:
  - `tests/test_svd.c`
  - `tests/test_svd_partial_helpers.h`
  - `tests/test_sprint29_integration.c`
- Ran focused and affected-family validation across LOBPCG, thick-restart,
  SVD, and SVD/eigs integration binaries.
- Rechecked Day 9 residual, orthogonality, convergence, and non-claim
  evidence.
- Froze the Day 11 SVD implementation scope to one claim-owned diagonal,
  rank-sensitive, full-UV test in `tests/test_svd.c`.
- Recorded the closeout, SVD overlap table, selected SVD scope, validation
  commands, deferred follow-ups, and non-claims in
  `artifacts/day10-spectral-closeout-and-svd-scope.md`.

### Findings

- Day 9 spectral evidence remains valid after focused rerun.
- LOBPCG and thick-restart now have bounded residual and orthogonality claims
  tied to named fixtures.
- Existing SVD coverage is broad, so Day 11 should consolidate evidence style
  rather than add a broad new SVD lane.
- The best Day 11 scope is a single diagonal full-UV SVD claim that checks
  singular values, reconstruction residual, U/Vt orthogonality, and explicit
  rank-threshold behavior.
- External SVD helper work, SuiteSparse corpus expansion, and partial SVD
  changes remain deferred.

### Validation Expectations

- Day 10 changes planning documentation only.
- Day 10 also reruns focused spectral and SVD-adjacent validation because it
  is the closeout gate before SVD implementation.
- Required checks:
  - `make build/test_eigs_lobpcg build/test_eigs_thick_restart build/test_svd build/test_sprint29_integration`
  - `./build/test_eigs_lobpcg`
  - `./build/test_eigs_thick_restart`
  - `./build/test_svd`
  - `./build/test_sprint29_integration`
  - `git diff --check`
  - trailing-whitespace scan on touched tests and Sprint 103 docs

### Validation Results

- `make build/test_eigs_lobpcg build/test_eigs_thick_restart build/test_svd build/test_sprint29_integration`:
  passed; all targets already up to date.
- `./build/test_eigs_lobpcg`: passed; 27 tests, 0 failures, 0 skips,
  247 assertions.
- `./build/test_eigs_thick_restart`: passed; 21 tests, 0 failures, 0 skips,
  285 assertions.
- `./build/test_svd`: passed; 97 tests, 0 failures, 0 skips,
  1073 assertions.
- `./build/test_sprint29_integration`: passed; 3 tests, 0 failures, 0 skips,
  25 assertions.
- `git diff --check`: passed.
- `rg -n "[ \t]+$" tests/test_svd.c tests/test_eigs_lobpcg.c tests/test_eigs_thick_restart.c tests/test_bicgstab.c docs/planning/EPIC_10/SPRINT_103`:
  passed; no matches.

### Day 10 Exit State

Day 10 is complete. Day 11 can implement the selected SVD
diagonal/rank/full-UV claim without broad external parity claims.

## Day 11 - SVD Comparison Follow-Through

### Goal

Implement the selected SVD comparison test from Day 10, preserving public SVD
API behavior while checking singular values, reconstruction residuals,
orthogonality, and rank-sensitive behavior against explicit fixture criteria.

### Actions

- Re-read Sprint 103 Day 11 in
  `docs/planning/EPIC_10/SPRINT_103/PLAN.md`.
- Re-read the Day 10 spectral closeout and SVD scope artifact.
- Added `test_s103_svd_diag6_rank_threshold_claim` to `tests/test_svd.c`.
- Registered the new test in the existing `test_svd` binary.
- Kept the implementation local to `tests/test_svd.c`; no public headers,
  library sources, build files, external helpers, fixture files, or
  partial-SVD helpers changed.
- Recorded implementation evidence in
  `artifacts/day11-svd-comparison-follow-through.md`.

### Findings

- The new diagonal SVD claim separates exact singular-value checks,
  full-mode reconstruction residual, U/Vt orthogonality, and rank-threshold
  behavior.
- The selected fixture is deterministic and does not introduce external helper
  availability, versioning, or skip semantics.
- The test uses full-mode UV output to align SVD evidence style with the Day 9
  spectral orthogonality and residual work.

### Validation Expectations

- Day 11 changed `tests/test_svd.c`.
- Required checks:
  - `make build/test_svd`
  - `./build/test_svd`
  - `make format && make lint && make test`
  - `git diff --check`
  - trailing-whitespace scan on `tests/test_svd.c`,
    `tests/test_eigs_lobpcg.c`, `tests/test_eigs_thick_restart.c`,
    `tests/test_bicgstab.c`, and `docs/planning/EPIC_10/SPRINT_103`

### Validation Results

- `make build/test_svd`: passed.
- `./build/test_svd`: passed; 98 tests, 0 failures, 0 skips,
  1093 assertions.
- `make format && make lint && make test`: passed; `All tests passed.`
- `git diff --check`: passed.
- `rg -n "[ \t]+$" tests/test_svd.c tests/test_eigs_lobpcg.c tests/test_eigs_thick_restart.c tests/test_bicgstab.c docs/planning/EPIC_10/SPRINT_103`:
  passed; no matches.

### Day 11 Exit State

Day 11 is complete. The selected SVD comparison follow-through is implemented,
focused and full validation passed, and Day 12 can document the residual,
orthogonality, rank, and comparison-evidence boundaries.

## Day 12 - Reporting and Documentation Update

### Goal

Document Sprint 103 convergence-profile, residual, orthogonality, rank, and
comparison-evidence boundaries so later public or maintainer wording does not
overstate the implemented proof.

### Actions

- Re-read Sprint 103 Day 12 in
  `docs/planning/EPIC_10/SPRINT_103/PLAN.md`.
- Re-read the existing Sprint 102 direct-solver trust-boundary snapshot in
  `docs/maintainer_guide.md`.
- Added `Sprint 103 Iterative, Spectral, and SVD Evidence Boundary Snapshot`
  to `docs/maintainer_guide.md`.
- Documented BiCGSTAB, LOBPCG, thick-restart eigensolver, and SVD maintained
  evidence owners.
- Recorded evidence types as deterministic fixture evidence, internal
  consistency evidence, residual/orthogonality quality evidence, and absent
  external helper-backed parity.
- Added explicit non-claim wording for broad PETSc, SciPy, Trilinos, ARPACK,
  LAPACK, NumPy, and ecosystem parity.
- Recorded the documentation update in
  `artifacts/day12-reporting-and-documentation-update.md`.

### Findings

- Sprint 103 evidence is strongest as named-fixture regression evidence, not as
  broad package parity.
- Direct-solver external dense-reference evidence remains owned by the Sprint
  102 lanes; Sprint 103 did not add external helper-backed iterative,
  eigensolver, or SVD package comparison lanes.
- Residual wording needs family-specific interpretation: iterative solve
  residuals, eigensolver Ritz residuals, and SVD test-computed reconstruction
  residuals are related quality checks but not interchangeable public claims.

### Validation Expectations

- Day 12 only changed documentation files.
- Required checks:
  - `git diff --check`
  - trailing-whitespace scan on `docs/maintainer_guide.md` and
    `docs/planning/EPIC_10/SPRINT_103`

### Validation Results

- `git diff --check`: passed.
- `rg -n "[ \t]+$" docs/maintainer_guide.md docs/planning/EPIC_10/SPRINT_103`:
  passed; no matches.

### Day 12 Exit State

Day 12 is complete. The Sprint 103 documentation boundary now distinguishes
deterministic fixture evidence, internal consistency checks, residual and
orthogonality quality criteria, external dense-reference lanes, and absent
external package parity before Day 13 reconciliation begins.

## Day 13 - Validation and Evidence Reconciliation

### Goal

Run the required branch quality checks for the Sprint 103 C test changes,
reconcile the implemented comparison evidence against the Day 2 ranking, and
freeze remaining gaps before closeout.

### Actions

- Re-read Sprint 103 Day 13 in
  `docs/planning/EPIC_10/SPRINT_103/PLAN.md`.
- Re-read the Day 2 ranking, Day 7 iterative closeout, and Day 10 spectral/SVD
  closeout artifacts.
- Ran the required full validation gate because Sprint 103 modified C test
  files.
- Mapped every implemented comparison lane to its maintained test owner,
  evidence type, artifact trail, and documented claim boundary.
- Confirmed no implemented comparison lacks a matching artifact or documented
  non-claim boundary.
- Recorded remaining gaps and Sprint 104 candidates in
  `artifacts/day13-validation-and-evidence-reconciliation.md`.

### Findings

- The Day 2 implementation priority order was followed: BiCGSTAB first,
  LOBPCG and thick-restart second, and one bounded SVD follow-through after
  spectral closeout.
- Sprint 103 landed deterministic fixture, direct-solver cross-check,
  internal-consistency, residual, orthogonality, and rank-threshold evidence.
- Sprint 103 did not add external package helper lanes for PETSc, SciPy,
  Trilinos, ARPACK, LAPACK, or NumPy.
- MINRES, CG, grow-m eigensolver documentation, and broad external-helper
  parity remain explicit future work rather than implicit claims.

### Validation Results

- `make format && make lint && make test`: passed; final output reported
  `All tests passed.`
- `git diff --check`: passed.
- `rg -n "[ \t]+$" tests/test_bicgstab.c tests/test_eigs_lobpcg.c tests/test_eigs_thick_restart.c tests/test_svd.c docs/maintainer_guide.md docs/planning/EPIC_10/SPRINT_103`:
  passed; no matches.

### Day 13 Exit State

Day 13 is complete. Sprint 103 evidence is reconciled across implemented C
tests, planning artifacts, maintainer documentation, validation results,
remaining gaps, and the Sprint 104 candidate queue. Day 14 can build the final
closeout and handoff package.

## Day 14 - Closeout and Handoff

### Goal

Package the Sprint 103 artifact trail, final validation, and Sprint 104
handoff prerequisites so the sprint can close without unresolved evidence or
claim-boundary ambiguity.

### Actions

- Re-read Sprint 103 Day 14 in
  `docs/planning/EPIC_10/SPRINT_103/PLAN.md`.
- Built the Sprint 103 artifact index covering Day 1 through Day 14.
- Summarized implemented evidence owners for BiCGSTAB, LOBPCG, thick-restart,
  SVD, maintainer documentation, and sprint evidence tracking.
- Confirmed each implemented comparison has a maintained test owner, artifact
  trail, validation record, and claim boundary.
- Identified Sprint 104 prerequisites, deferred work ownership, and risk
  mitigations.
- Recorded closeout and handoff details in
  `artifacts/day14-closeout-and-handoff.md`.

### Findings

- Sprint 103 has a complete trail from audit and ranking through
  implementation, documentation, validation reconciliation, and closeout.
- The remaining work is policy and external-helper driven: helper
  availability, oracle independence, CI role, tolerance ownership, and public
  wording controls should be decided before adding package-parity lanes.
- No Sprint 103 comparison should be described as broad external parity.

### Validation Results

- `make format && make lint && make test`: passed; final output reported
  `All tests passed.`
- `git diff --check`: passed.
- `rg -n "[ \t]+$" tests/test_bicgstab.c tests/test_eigs_lobpcg.c tests/test_eigs_thick_restart.c tests/test_svd.c docs/maintainer_guide.md docs/planning/EPIC_10/SPRINT_103`:
  passed; no matches.

### Day 14 Exit State

Day 14 is complete. Sprint 103 now has a complete artifact trail from audit
through closeout, final validation passed, and Sprint 104 can start from
explicit external-helper prerequisites, deferred-work ownership, and
claim-boundary risks.
