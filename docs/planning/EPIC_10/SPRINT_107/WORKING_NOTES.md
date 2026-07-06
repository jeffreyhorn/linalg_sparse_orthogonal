# Sprint 107 Working Notes

## Sprint Context

Sprint 107 implements "Residual Maintainability Debt & Proof-Owner Cleanup"
from `docs/planning/EPIC_10/PROJECT_PLAN.md`. The sprint exists because
Sprint 106 intentionally left several large proof and source owners untouched
or only partially bounded after completing the higher-priority extraction
work.

The sprint is not a broad rewrite sprint. It must reduce review and failure
localization cost while preserving the constraints that Sprint 106 made
explicit:

- no public API or install-header change;
- no new compiled test helper target;
- no reviewed test-count change unless explicitly approved;
- no broad direct-solver rewrite;
- no broad QR, iterative, eigensolver, or SVD proof-owner cleanup;
- no opportunistic central sparse matrix shell extraction.

## Validation Rules

Validation must scale with the touched surface:

| touched surface | required validation |
|---|---|
| planning documentation only | `git diff --check`; trailing-whitespace scan on touched planning files |
| public documentation only | `git diff --check`; trailing-whitespace scan on touched docs |
| test helper extraction | focused affected test binaries; CTest registration check when registration changes; `make format`; `make lint`; `make test` |
| source extraction in `.c` files | focused affected tests; source-list check if library membership changes; `make format`; `make lint`; `make test` |
| public or internal `.h` changes | focused affected tests; source-list check when library ownership changes; `make format`; `make lint`; `make test` |
| build or CMake surface | source-list checker; focused Make/CMake configure or build check; any code-touch gate |
| mixed source/test/build cleanup | all applicable focused checks plus the full C quality gate |

If any `.c` or `.h` file is modified, the full required quality chain is:

```sh
make format && make lint && make test
```

All required checks must pass before closeout or PR creation.

## Cleanup Boundaries

Sprint 107 may extract helpers only where the call sites remain easier to
review after the move. The sprint must keep proof intent visible in tests that
validate solver behavior, convergence behavior, rank/oracle behavior, or
central matrix API compatibility.

Sprint 107 must preserve:

- current public API and install-header behavior;
- reviewed CTest shape unless a test registration change is explicitly
  planned and counted;
- existing Make/CMake/source-list parity when source ownership changes;
- direct solver, QR, iterative, eigensolver, SVD, and matrix shell evidence
  boundaries from Sprint 100 through Sprint 106.

Sprint 107 must not claim:

- completion of all large-owner cleanup;
- broad solver-family redesign from fixture extraction;
- new public API stability from internal proof-owner cleanup;
- state-of-the-art competitiveness from maintainability work alone.

## Day 1 - Residual Debt Intake

### Goal

Convert Sprint 106 residual deferred debt into a bounded Sprint 107 work
package with clear owner inventory, validation expectations, and a Day 2
boundary re-rank starting point.

### Actions

- Re-read the Sprint 107 section of
  `docs/planning/EPIC_10/PROJECT_PLAN.md`.
- Re-read the Sprint 107 day-by-day plan in
  `docs/planning/EPIC_10/SPRINT_107/PLAN.md`.
- Re-read the Sprint 106 residual deferred debt section in
  `docs/planning/EPIC_10/SPRINT_106/RETROSPECTIVE.md`.
- Reviewed Sprint 106 working notes and artifacts list to distinguish
  completed extraction work from Sprint 107 carry-forward work.
- Created the Sprint 107 artifacts directory.
- Captured residual owner sizes and Day 1 validation expectations in
  `artifacts/day1-residual-debt-intake.md`.

### Findings

- Sprint 107 has six explicit residual owners: `tests/test_ldlt_csc.c`,
  `tests/test_qr.c`, `tests/test_iterative.c`, `tests/test_svd.c`,
  `src/sparse_eigs.c`, and `src/sparse_matrix.c`.
- The six residual owners currently total 15,735 lines, so Day 2 must rank
  by risk reduction and review value rather than line count alone.
- `tests/test_ldlt_csc.c` remains the largest residual proof owner and should
  receive only one narrow helper extraction after a boundary artifact.
- QR, iterative, and SVD work must preserve proof intent at call sites because
  their large tests encode solve, convergence, reconstruction, rank, and oracle
  interpretation.
- `src/sparse_eigs.c` is source-level debt but remains tied to Sprint 103
  comparison evidence, so it needs a fresh boundary before any split.
- `src/sparse_matrix.c` is central API/compatibility territory and should be
  documented as a deferral contract, not split opportunistically.

### Validation Expectations

- Day 1 changes planning documentation only.
- Required checks:
  - `git diff --check`
  - trailing-whitespace scan on `docs/planning/EPIC_10/SPRINT_107`

### Validation Results

- `git diff --check`: passed.
- `rg -n "[ \t]+$" docs/planning/EPIC_10/SPRINT_107 docs/planning/EPIC_10/PROJECT_PLAN.md`:
  passed; no matches.

### Day 1 Exit State

Day 1 is complete when `artifacts/day1-residual-debt-intake.md` exists,
working notes contain the Sprint 107 validation rules and cleanup boundaries,
and documentation hygiene checks pass.

## Day 2 - Residual Boundary Re-rank

### Goal

Rank remaining proof-owner and source-owner debt from live repository evidence
while excluding work already completed in Sprint 106.

### Actions

- Re-read the Day 2 plan section in
  `docs/planning/EPIC_10/SPRINT_107/PLAN.md`.
- Re-read the Day 1 residual intake artifact.
- Captured live line counts for the six residual owners.
- Captured recent churn counts since `2026-06-01` for the residual owners.
- Captured approximate function-like definition counts, assertion/test macro
  hits, and helper/fixture keyword hits.
- Reviewed Sprint 106 closeout and retrospective sections that list completed
  extraction work and explicit residual debt.
- Wrote the Day 2 residual boundary re-rank artifact in
  `artifacts/day2-residual-boundary-rerank.md`.

### Findings

- `tests/test_ldlt_csc.c` remains the largest residual proof owner at 3,884
  lines and has the highest assertion/test macro density among the residual
  test files.
- `tests/test_svd.c` has the highest helper/fixture keyword signal, but rank
  and oracle interpretation make it a later, dedicated cleanup after QR and
  iterative builder patterns are bounded.
- `tests/test_qr.c` is a good first fixture cleanup after LDLT CSC because it
  has broad builders and lower recent churn, provided solve and reconstruction
  assertions remain inline.
- `tests/test_iterative.c` must start with reusable matrix/RHS builders only
  because convergence-sensitive assertions are easy to obscure.
- `src/sparse_eigs.c` needs a Sprint 103 comparison-aware boundary before any
  source split.
- `src/sparse_matrix.c` has the highest recent churn but remains central API
  and compatibility territory, so Sprint 107 should produce a deferral
  contract rather than source extraction.

### Validation Expectations

- Day 2 changes planning documentation only.
- Required checks:
  - `git diff --check`
  - trailing-whitespace scan on `docs/planning/EPIC_10/SPRINT_107`

### Validation Results

- `git diff --check`: passed.
- `rg -n "[ \t]+$" docs/planning/EPIC_10/SPRINT_107 docs/planning/EPIC_10/PROJECT_PLAN.md`:
  passed; no matches.

### Day 2 Exit State

Day 2 is complete when `artifacts/day2-residual-boundary-rerank.md` exists,
all six residual owners have a Sprint 107 disposition, completed Sprint 106
work is excluded from duplicate cleanup, and documentation hygiene checks pass.

## Day 3 - LDLT CSC Proof Boundary

### Goal

Freeze the narrow `tests/test_ldlt_csc.c` proof-helper seam before editing
tests on Day 4.

### Actions

- Re-read the Day 3 plan section in
  `docs/planning/EPIC_10/SPRINT_107/PLAN.md`.
- Inspected `tests/test_ldlt_csc.c` row-adjacency tests, residual helpers,
  dense-reference helpers, and oracle-heavy solve sections.
- Compared candidate helper seams against Sprint 106's instruction to extract
  only one narrow row-adjacency assertion or residual/oracle helper.
- Selected the row-local assertion block inside
  `test_row_adj_matches_reference` as the Day 4 extraction candidate.
- Wrote the LDLT CSC proof-boundary artifact in
  `artifacts/day3-ldlt-csc-proof-boundary.md`.

### Findings

- The small row-adjacency unit tests around allocation, append/growth,
  argument validation, and swap-slot behavior should remain inline.
- The `test_row_adj_matches_reference` loop contains a focused row-local
  expected-count and membership check that can be extracted without hiding the
  larger test intent.
- Residual and external dense-reference helpers already exist and are tied to
  broader solve/oracle semantics, so they are not the right first Sprint 107
  cleanup seam.
- The selected Day 4 candidate is a local static helper named
  `assert_row_adj_matches_l_pattern`.
- No new compiled helper target, public header, install header, Makefile,
  CMake, or source-list update is implied by the selected seam.

### Validation Expectations

- Day 3 changes planning documentation only.
- Required checks:
  - `git diff --check`
  - trailing-whitespace scan on `docs/planning/EPIC_10/SPRINT_107`

### Validation Results

- `git diff --check`: passed.
- `rg -n "[ \t]+$" docs/planning/EPIC_10/SPRINT_107 docs/planning/EPIC_10/PROJECT_PLAN.md`:
  passed; no matches.

### Day 3 Exit State

Day 3 is complete when `artifacts/day3-ldlt-csc-proof-boundary.md` exists,
the Day 4 helper seam is selected, no-new-target rationale is documented, and
documentation hygiene checks pass.

## Day 4 - LDLT CSC Proof Helper Extraction

### Goal

Extract the selected local row-adjacency assertion helper from
`tests/test_ldlt_csc.c` without changing LDLT CSC behavior, test registration,
or build-system surface.

### Actions

- Re-read the Day 4 plan section in
  `docs/planning/EPIC_10/SPRINT_107/PLAN.md`.
- Re-read the Day 3 proof-boundary artifact.
- Added `assert_row_adj_matches_l_pattern` as a `static` helper immediately
  before `test_row_adj_matches_reference`.
- Replaced the inline row-local expected-count and membership assertion block
  in `test_row_adj_matches_reference` with a visible per-row helper call.
- Confirmed no `RUN_TEST`, CMake, Makefile, public-header, or implementation
  source changes were needed.
- Wrote the Day 4 proof-helper extraction artifact in
  `artifacts/day4-ldlt-csc-proof-helper-extraction.md`.

### Findings

- The chosen helper stayed within the Day 3 seam and did not need a shared
  fixture header or compiled helper target.
- The row-adjacency proof now has a named local assertion boundary while the
  test still visibly sweeps every row after native elimination.
- `tests/test_ldlt_csc.c` is 3,885 lines after formatting, a net one-line
  increase from the Day 2 baseline due to the extracted helper shape.
- Reviewed test registration did not drift because no test names or `RUN_TEST`
  entries were changed.

### Validation Expectations

- Day 4 changes a `.c` file, so the required quality gate is:
  - `make format`
  - `make lint`
  - `make test`
- Focused affected test should also pass before full gate confidence:
  - `make build/test_ldlt_csc && ./build/test_ldlt_csc`
- Hygiene checks:
  - `git diff --check`
  - trailing-whitespace scan on Sprint 107 docs, the Epic 10 project plan, and
    `tests/test_ldlt_csc.c`

### Validation Results

- `make build/test_ldlt_csc && ./build/test_ldlt_csc`: passed; focused suite
  ran 100 tests with 0 failures.
- `make format && make lint && make test`: passed.
- `git diff --check`: passed.
- `rg -n "[ \t]+$" docs/planning/EPIC_10/SPRINT_107 docs/planning/EPIC_10/PROJECT_PLAN.md tests/test_ldlt_csc.c`:
  passed; no matches.

### Day 4 Exit State

Day 4 is complete when the helper extraction is in `tests/test_ldlt_csc.c`, the
Day 4 artifact exists, focused and full C quality gates pass, and hygiene
checks pass without whitespace issues.

## Day 5 - QR Fixture Boundary

### Goal

Identify a safe `tests/test_qr.c` fixture-builder cleanup batch that reduces
repeated setup without hiding QR solve, rank, residual, reconstruction,
orthogonality, mode-comparison, or refinement proof intent.

### Actions

- Re-read the Day 5 plan section in
  `docs/planning/EPIC_10/SPRINT_107/PLAN.md`.
- Inspected `tests/test_qr.c` from the forward declarations through the test
  suite registration.
- Inventoried repeated matrix, RHS, reconstruction, residual, rank,
  orthogonality, economy-mode, sparse-mode, and refinement setup patterns.
- Classified repeated patterns as safe builders, already-covered assertion
  helpers, or inline proof logic that should not be extracted.
- Selected a bounded Day 6 builder batch limited to three local static builder
  families and six small-fixture call-site groups.
- Wrote the QR fixture-boundary artifact in
  `artifacts/day5-qr-fixture-boundary.md`.

### Findings

- `tests/test_qr.c` is 3,234 lines and already has local reconstruction and
  true-residual assertion helpers, so Day 6 should not add assertion helpers.
- The safest first cleanup is small fixture construction, not proof behavior.
- The approved Day 6 helpers are:
  - `make_qr_small_banded_4x3(int include_tail)`
  - `make_qr_duplicate_column_4x3(double duplicate_scale)`
  - `make_qr_near_duplicate_4x3(double perturbation)`
- Day 6 should update only the approved call sites and must keep expected rank,
  residual, reconstruction, sparse-vs-dense, QR-vs-LU, and refinement
  assertions inline.
- No new compiled helper target, public header, install header, Makefile,
  CMake, or `RUN_TEST` update is implied by the selected boundary.

### Validation Expectations

- Day 5 changes planning documentation only.
- Required checks:
  - `git diff --check`
  - trailing-whitespace scan on `docs/planning/EPIC_10/SPRINT_107` and
    `docs/planning/EPIC_10/PROJECT_PLAN.md`

### Validation Results

- `git diff --check`: passed.
- `rg -n "[ \t]+$" docs/planning/EPIC_10/SPRINT_107 docs/planning/EPIC_10/PROJECT_PLAN.md`:
  passed; no matches.

### Day 5 Exit State

Day 5 is complete when `artifacts/day5-qr-fixture-boundary.md` exists, the
Day 6 QR builder batch is selected, no-new-target rationale is documented, and
documentation hygiene checks pass.

## Day 6 - QR Fixture Cleanup Batch

### Goal

Implement the selected `tests/test_qr.c` fixture-builder cleanup while keeping
QR proof assertions and expectations visible at the test sites.

### Actions

- Re-read the Day 6 plan section in
  `docs/planning/EPIC_10/SPRINT_107/PLAN.md`.
- Re-read the Day 5 QR fixture-boundary artifact.
- Added three local static builders to `tests/test_qr.c`:
  - `make_qr_small_banded_4x3(int include_tail)`
  - `make_qr_duplicate_column_4x3(double duplicate_scale)`
  - `make_qr_near_duplicate_4x3(double perturbation)`
- Replaced repeated fixture setup in:
  - `test_q_roundtrip`
  - `test_q_apply_multiple`
  - `test_sparse_mode_basic`
  - `test_qr_rank_deficient`
  - `test_qr_solve_rank_deficient`
  - `test_known_nullspace`
  - `test_qr_nearly_singular`
- Left `test_qr_refine_ill_conditioned` inline after confirming its
  near-duplicate fixture uses a different base progression than the selected
  near-duplicate builder.
- Wrote the Day 6 QR fixture-cleanup artifact in
  `artifacts/day6-qr-fixture-cleanup.md`.

### Findings

- `tests/test_qr.c` is 3,210 lines after formatting, down from the Day 5
  baseline of 3,234 lines.
- The cleanup did not add, remove, or rename `RUN_TEST` entries.
- The cleanup did not touch Makefile, CMake, public headers, install headers,
  or compiled helper targets.
- Expected rank, residual, reconstruction, sparse-vs-dense, QR-vs-LU,
  nullspace, and refinement assertions remained inline at the test sites.
- Remaining QR cleanup should focus on generated fixtures, tall/economy
  builders, diagonal/singleton builders, and SuiteSparse exact-RHS setup only
  after a dedicated boundary pass.

### Validation Expectations

- Day 6 changes a `.c` file, so the required quality gate is:
  - `make format`
  - `make lint`
  - `make test`
- Focused affected test should also pass before full gate confidence:
  - `make build/test_qr && ./build/test_qr`
- Hygiene checks:
  - `git diff --check`
  - trailing-whitespace scan on Sprint 107 docs, the Epic 10 project plan,
    `tests/test_qr.c`, and `tests/test_ldlt_csc.c`

### Validation Results

- `make build/test_qr && ./build/test_qr`: passed; focused suite ran 73 tests
  with 0 failures.
- `make format && make lint && make test`: passed after the final
  semantic-preservation edit.
- `git diff --check`: passed.
- `rg -n "[ \t]+$" docs/planning/EPIC_10/SPRINT_107 docs/planning/EPIC_10/PROJECT_PLAN.md tests/test_qr.c tests/test_ldlt_csc.c`:
  passed; no matches.

### Day 6 Exit State

Day 6 is complete when QR fixture builders are extracted, the Day 6 artifact
exists, focused and full C quality gates pass, and hygiene checks pass without
whitespace issues.

## Day 7 - Iterative Fixture Boundary

### Goal

Identify a safe `tests/test_iterative.c` fixture-builder cleanup batch that
reduces repeated setup without hiding iterative solver convergence, residual,
preconditioner, restart, matrix-free, or direct comparison proof intent.

### Actions

- Re-read the Day 7 plan section in
  `docs/planning/EPIC_10/SPRINT_107/PLAN.md`.
- Inspected `tests/test_iterative.c` helper definitions, CG sections, GMRES
  sections, SuiteSparse comparison tests, and matrix-free tests.
- Inventoried repeated matrix, RHS, exact solution, option, preconditioner,
  restart, and result-check setup patterns.
- Classified repeated patterns as safe existing builders, safe literal RHS
  setup, or convergence-sensitive proof logic that should remain inline.
- Selected a bounded Day 8 cleanup batch limited to matrix-free matrix builder
  reuse plus one sequential RHS filler.
- Wrote the iterative fixture-boundary artifact in
  `artifacts/day7-iterative-fixture-boundary.md`.

### Findings

- `tests/test_iterative.c` is 2,841 lines and already has local builders for
  SPD tridiagonal, identity, Laplacian, unsymmetric tridiagonal, and known-RHS
  computation.
- The safest first cleanup is not a broad exact-solution allocator because
  those patterns cross CG, GMRES, SuiteSparse, restart, and preconditioner
  assertions.
- The approved Day 8 helper is:
  - `fill_sequential_rhs(double *b, idx_t n)`
- The approved Day 8 builder reuse is:
  - `test_cg_mf_basic` uses `build_spd_tridiag(n, 4.0, -1.0)`
  - `test_gmres_mf_basic` uses `build_unsym_tridiag(n, 5.0, -2.0, -1.0)`
- The approved Day 8 RHS call sites are:
  - `test_cg_mf_basic`
  - `test_cg_mf_nos4`
  - `test_gmres_mf_basic`
  - `test_gmres_mf_right_precond`
- Solver options, initial guesses, preconditioners, restart loops,
  convergence checks, residual checks, and direct solver comparison
  assertions must remain inline.

### Validation Expectations

- Day 7 changes planning documentation only.
- Required checks:
  - `git diff --check`
  - trailing-whitespace scan on `docs/planning/EPIC_10/SPRINT_107` and
    `docs/planning/EPIC_10/PROJECT_PLAN.md`

### Validation Results

- `git diff --check`: passed.
- `rg -n "[ \t]+$" docs/planning/EPIC_10/SPRINT_107 docs/planning/EPIC_10/PROJECT_PLAN.md`:
  passed; no matches.

### Day 7 Exit State

Day 7 is complete when `artifacts/day7-iterative-fixture-boundary.md` exists,
the Day 8 iterative builder/RHS batch is selected, no-new-target rationale is
documented, and documentation hygiene checks pass.

## Day 8 - Iterative Fixture Cleanup

### Goal

Implement the selected `tests/test_iterative.c` matrix-free fixture cleanup
while keeping iterative solver options, convergence checks, preconditioner
setup, and concrete-versus-matrix-free comparison assertions visible at the
test sites.

### Actions

- Re-read the Day 8 plan section in
  `docs/planning/EPIC_10/SPRINT_107/PLAN.md`.
- Re-read the Day 7 iterative fixture-boundary artifact.
- Added `fill_sequential_rhs` as a local static helper near `compute_rhs`.
- Replaced manual SPD tridiagonal construction in `test_cg_mf_basic` with
  `build_spd_tridiag(n, 4.0, -1.0)`.
- Replaced manual unsymmetric tridiagonal construction in
  `test_gmres_mf_basic` with
  `build_unsym_tridiag(n, 5.0, -2.0, -1.0)`.
- Replaced selected sequential RHS loops with `fill_sequential_rhs` in:
  - `test_cg_mf_basic`
  - `test_cg_mf_nos4`
  - `test_gmres_mf_basic`
  - `test_gmres_mf_right_precond`
- Wrote the Day 8 iterative fixture-cleanup artifact in
  `artifacts/day8-iterative-fixture-cleanup.md`.

### Findings

- The cleanup stayed inside the Day 7 matrix/RHS boundary and did not move
  solver options, preconditioners, restart behavior, result checks, or direct
  comparison assertions.
- The GMRES matrix-free basic fixture preserved the existing upper/lower
  values by using `build_unsym_tridiag(n, 5.0, -2.0, -1.0)`.
- The cleanup did not add, remove, or rename `RUN_TEST` entries.
- The cleanup did not touch Makefile, CMake, public headers, install headers,
  or compiled helper targets.

### Validation Expectations

- Day 8 changes a `.c` file, so the required quality gate is:
  - `make format`
  - `make lint`
  - `make test`
- Focused affected test should also pass before full gate confidence:
  - `make build/test_iterative && ./build/test_iterative`
- Hygiene checks:
  - `git diff --check`
  - trailing-whitespace scan on Sprint 107 docs, the Epic 10 project plan,
    `tests/test_iterative.c`, `tests/test_qr.c`, and `tests/test_ldlt_csc.c`

### Validation Results

- `make build/test_iterative && ./build/test_iterative`: passed; focused
  suite ran 80 tests with 0 failures.
- `make format && make lint && make test`: passed.
- `git diff --check`: passed.
- `rg -n "[ \t]+$" docs/planning/EPIC_10/SPRINT_107 docs/planning/EPIC_10/PROJECT_PLAN.md tests/test_iterative.c tests/test_qr.c tests/test_ldlt_csc.c`:
  passed; no matches.

### Day 8 Exit State

Day 8 is complete when the iterative matrix-free fixture cleanup is in
`tests/test_iterative.c`, the Day 8 artifact exists, focused and full C
quality gates pass, and hygiene checks pass without whitespace issues.

## Day 9 - SVD Proof-Owner Boundary

### Goal

Identify a safe `tests/test_svd.c` cleanup batch that reduces repeated fixture
setup without hiding SVD rank, singular-value, reconstruction, pseudoinverse,
low-rank, partial-SVD, external-oracle, or condition-number proof intent.

### Actions

- Re-read the Day 9 plan section in
  `docs/planning/EPIC_10/SPRINT_107/PLAN.md`.
- Inspected `tests/test_svd.c` helper definitions, Golub-Kahan extraction
  tests, SVD convergence tests, rank tests, pseudoinverse tests, low-rank
  tests, full-mode U/Vt tests, Sprint 103 claim-owned tests, and condition
  number tests.
- Inventoried repeated diagonal, rank-1, rank-deficient, reconstruction,
  partial-SVD, low-rank, and condition-number setup patterns.
- Classified repeated patterns as safe local fixture builders or proof logic
  that should remain inline.
- Selected a bounded Day 10 cleanup batch limited to one diagonal fixture
  builder and one rank-1 row-progression fixture builder.
- Wrote the SVD proof-owner boundary artifact in
  `artifacts/day9-svd-proof-owner-boundary.md`.

### Findings

- `tests/test_svd.c` is 2,879 lines and already has reconstruction and
  partial-SVD helper coverage, so Day 10 should not add assertion helpers.
- The safest first cleanup is local matrix fixture construction, not rank,
  oracle, reconstruction, pseudoinverse, or low-rank proof movement.
- The approved Day 10 helpers are:
  - `make_svd_diag_matrix(idx_t rows, idx_t cols, const double *diag, idx_t diag_len)`
  - `make_svd_rank1_row_progression(idx_t rows, idx_t cols)`
- The approved diagonal call sites are:
  - `test_svd_diagonal_5x5`
  - `test_lowrank_diagonal`
  - `test_lowrank_sparse_diagonal`
  - `test_cond_diagonal`
  - `test_cond_ill_conditioned`
  - `test_cond_rectangular`
- The approved rank-1 call sites are:
  - `test_svd_rank1`
  - `test_svd_rank1_uv`
- Expected singular values, rank thresholds, reconstruction loops,
  pseudoinverse products, low-rank error bounds, condition-number
  expectations, SuiteSparse corpus tolerances, and `RUN_TEST` registration
  must remain inline.

### Validation Expectations

- Day 9 changes planning documentation only.
- Required checks:
  - `git diff --check`
  - trailing-whitespace scan on `docs/planning/EPIC_10/SPRINT_107` and
    `docs/planning/EPIC_10/PROJECT_PLAN.md`

### Validation Results

- `git diff --check`: passed.
- `rg -n "[ \t]+$" docs/planning/EPIC_10/SPRINT_107 docs/planning/EPIC_10/PROJECT_PLAN.md`:
  passed; no matches.

### Day 9 Exit State

Day 9 is complete when `artifacts/day9-svd-proof-owner-boundary.md` exists,
the Day 10 SVD fixture-builder batch is selected, no-new-target rationale is
documented, and documentation hygiene checks pass.

## Day 10 - SVD Proof-Owner Cleanup

### Goal

Implement the selected `tests/test_svd.c` fixture-builder cleanup while
keeping rank, singular-value, reconstruction, pseudoinverse, low-rank,
partial-SVD, external-oracle, and condition-number proof interpretation
visible at the test sites.

### Actions

- Re-read the Day 10 plan section in
  `docs/planning/EPIC_10/SPRINT_107/PLAN.md`.
- Re-read the Day 9 SVD proof-owner boundary artifact.
- Added `make_svd_diag_matrix` as a local static fixture builder near the
  existing SVD helpers.
- Added `make_svd_rank1_row_progression` as a local static fixture builder
  for the repeated row-progression rank-1 fixture.
- Replaced approved diagonal fixture setup in:
  - `test_svd_diagonal_5x5`
  - `test_lowrank_diagonal`
  - `test_lowrank_sparse_diagonal`
  - `test_cond_diagonal`
  - `test_cond_ill_conditioned`
  - `test_cond_rectangular`
- Replaced approved 4x3 rank-1 setup in:
  - `test_svd_rank1`
  - `test_svd_rank1_uv`
- Wrote the Day 10 SVD proof-owner cleanup artifact in
  `artifacts/day10-svd-proof-owner-cleanup.md`.

### Findings

- The cleanup stayed inside the Day 9 fixture-builder boundary and did not
  move rank interpretation, expected singular values, reconstruction loops,
  pseudoinverse checks, low-rank error bounds, or condition-number
  expectations.
- The diagonal builder skips zero entries to preserve sparse fixture shape.
- The rank-1 builder only inserts `A[i,j] = (double)(i + 1)` and leaves the
  expected leading singular value and reconstruction loop inline.
- The cleanup did not add, remove, or rename `RUN_TEST` entries.
- The cleanup did not touch Makefile, CMake, public headers, install headers,
  or compiled helper targets.

### Validation Expectations

- Day 10 changes a `.c` file, so the required quality gate is:
  - `make format`
  - `make lint`
  - `make test`
- Focused affected test should also pass before full gate confidence:
  - `make build/test_svd && ./build/test_svd`
- Hygiene checks:
  - `git diff --check`
  - trailing-whitespace scan on Sprint 107 docs, the Epic 10 project plan,
    `tests/test_svd.c`, `tests/test_iterative.c`, `tests/test_qr.c`, and
    `tests/test_ldlt_csc.c`

### Validation Results

- `make build/test_svd && ./build/test_svd`: passed; focused suite ran 98
  tests with 0 failures.
- `make format && make lint && make test`: passed.
- `git diff --check`: passed.
- `rg -n "[ \t]+$" docs/planning/EPIC_10/SPRINT_107 docs/planning/EPIC_10/PROJECT_PLAN.md tests/test_svd.c tests/test_iterative.c tests/test_qr.c tests/test_ldlt_csc.c`:
  passed; no matches.

### Day 10 Exit State

Day 10 is complete when the SVD fixture builders are in `tests/test_svd.c`,
the Day 10 artifact exists, focused and full C quality gates pass, and hygiene
checks pass without whitespace issues.

## Day 11 - Eigensolver Source Boundary

### Goal

Create a fresh `src/sparse_eigs.c` boundary tied to Sprint 103 comparison
surfaces before deciding whether any Sprint 107 eigensolver source split is
safe.

### Actions

- Re-read the Day 11 plan section in
  `docs/planning/EPIC_10/SPRINT_107/PLAN.md`.
- Inspected `src/sparse_eigs.c`, `src/sparse_eigs_internal.h`, and
  `src/sparse_eigs_workspace_internal.h`.
- Reviewed the existing eigensolver split owners:
  - `src/sparse_eigs_workspace_internal.c`
  - `src/sparse_eigs_thick_restart.c`
  - `src/sparse_eigs_lobpcg.c`
- Reviewed Sprint 103 eigensolver comparison artifacts and closeout notes for
  LOBPCG, thick-restart, grow-m, shift-invert, and residual claim boundaries.
- Mapped shared Lanczos kernels, selection helpers, shift-invert/refinement,
  workspace handle glue, public dispatch helpers, and dense Jacobi ownership.
- Selected no Sprint 107 Day 12 source split because every plausible candidate
  crosses comparison-critical semantics, public dispatch, workspace handle
  behavior, or build-system parity for limited immediate payoff.
- Wrote the Day 11 eigensolver source-boundary artifact in
  `artifacts/day11-eigensolver-source-boundary.md`.

### Findings

- `src/sparse_eigs.c` is large, but the highest-risk backend bodies are
  already split into thick-restart and LOBPCG source owners.
- Shared Lanczos and selection helpers are consumed by grow-m, thick restart,
  LOBPCG, and focused tests, so they are not low-risk Day 12 extraction
  candidates.
- Shift-invert and refinement helpers are tied to grow-m public behavior and
  Sprint 29 residual/refinement evidence.
- Workspace storage is already separated; moving public handle glue now would
  cross the public API behavior boundary.
- Dense Jacobi is the least risky future split candidate, but it is still a
  cross-backend helper used by thick restart and LOBPCG and needs dedicated
  source-list/CMake parity work before movement.
- Day 12 should document an explicit no-split deferral rather than force a
  source edit.

### Validation Expectations

- Day 11 changes planning documentation only.
- Required checks:
  - `git diff --check`
  - trailing-whitespace scan on `docs/planning/EPIC_10/SPRINT_107` and
    `docs/planning/EPIC_10/PROJECT_PLAN.md`

### Validation Results

- `git diff --check`: passed.
- `rg -n "[ \t]+$" docs/planning/EPIC_10/SPRINT_107 docs/planning/EPIC_10/PROJECT_PLAN.md`:
  passed; no matches.

### Day 11 Exit State

Day 11 is complete when
`artifacts/day11-eigensolver-source-boundary.md` exists, the Day 12 no-split
direction is explicit, build-system follow-through is documented for any
future split, and documentation hygiene checks pass.

## Day 12 - Eigensolver Source Deferral

### Goal

Execute the Day 12 deferral path established by the Day 11 eigensolver source
boundary: perform no unsafe `src/sparse_eigs.c` extraction and publish the
future source-split prerequisites.

### Actions

- Re-read the Day 12 plan section in
  `docs/planning/EPIC_10/SPRINT_107/PLAN.md`.
- Re-read the Day 11 eigensolver source-boundary artifact.
- Confirmed Day 11 selected no Sprint 107 low-risk source split.
- Performed no `.c`, `.h`, Makefile, CMake, source-list, public-header, or
  install-header edit for Day 12.
- Wrote the Day 12 eigensolver source-deferral artifact in
  `artifacts/day12-eigensolver-source-deferral.md`.
- Recorded the future eigensolver residual queue:
  - dense Jacobi helper split feasibility;
  - grow-m shift-invert/refinement boundary audit;
  - dispatch and handle boundary audit;
  - cross-backend shared kernel audit.

### Findings

- Deferral is the correct Sprint 107 outcome because the remaining
  eigensolver source groups are shared across grow-m, thick restart, LOBPCG,
  public dispatch, workspace handle behavior, or Sprint 103 comparison
  evidence.
- No build-system follow-through is needed today because no source membership
  changed.
- Future eigensolver source work must start with source-list/CMake parity and
  focused spectral validation, not by moving helper bodies first.
- Day 13 can proceed to the central matrix shell deferral contract with the
  eigensolver workstream closed for Sprint 107.

### Validation Expectations

- Day 12 changes planning documentation only.
- Required checks:
  - `git diff --check`
  - trailing-whitespace scan on `docs/planning/EPIC_10/SPRINT_107` and
    `docs/planning/EPIC_10/PROJECT_PLAN.md`

### Validation Results

- `git diff --check`: passed.
- `rg -n "[ \t]+$" docs/planning/EPIC_10/SPRINT_107 docs/planning/EPIC_10/PROJECT_PLAN.md`:
  passed; no matches.

### Day 12 Exit State

Day 12 is complete when
`artifacts/day12-eigensolver-source-deferral.md` exists, no source extraction
has occurred, the future eigensolver residual queue is explicit, and
documentation hygiene checks pass.

## Day 13 - Central Matrix Shell Deferral Contract

### Goal

Document why `src/sparse_matrix.c` remains central API and compatibility
territory, identify future split preconditions, and prevent opportunistic
source extraction from bypassing public behavior review.

### Actions

- Re-read the Day 13 plan section in
  `docs/planning/EPIC_10/SPRINT_107/PLAN.md`.
- Inspected `src/sparse_matrix.c`, `include/sparse_matrix.h`,
  `src/sparse_matrix_internal.h`, and `src/sparse_matrix_state_internal.h`.
- Re-read Sprint 101 compressed-first notes and artifacts around public
  storage surface, solver-entry paths, compressed-first API design, lifecycle,
  ownership, and compatibility documentation.
- Mapped `src/sparse_matrix.c` responsibilities across lifecycle, mutation,
  access, arithmetic, Matrix Market I/O, printing, permutation accessors, and
  factor compatibility reset behavior.
- Performed no `.c`, `.h`, Makefile, CMake, source-list, public-header, or
  install-header edit for Day 13.
- Wrote the Day 13 central matrix shell deferral contract in
  `artifacts/day13-central-matrix-shell-deferral-contract.md`.

### Findings

- `src/sparse_matrix.c` is 1,359 lines but has high public behavior density;
  it is not just a private implementation bucket.
- The file owns object lifecycle, pool-backed storage, mutation, logical and
  physical access, norm/cache behavior, arithmetic, Matrix Market I/O,
  print/debug helpers, and compatibility state reset behavior.
- Sprint 101 intentionally preserved the mutable `SparseMatrix` shell as the
  public compatibility object even while making CSR/CSC construction a clearer
  compressed-first front door.
- Opportunistic extraction would risk implying direct CSR/CSC solver parity,
  a new public storage model, or install-header expansion that Sprint 101 did
  not approve.
- Future source movement must start with a named behavior boundary,
  private-header dependency plan, Make/CMake/source-list parity plan, and
  focused tests for the moved behavior.

### Validation Expectations

- Day 13 changes planning documentation only.
- Required checks:
  - `git diff --check`
  - trailing-whitespace scan on `docs/planning/EPIC_10/SPRINT_107` and
    `docs/planning/EPIC_10/PROJECT_PLAN.md`

### Validation Results

- `git diff --check`: passed.
- `rg -n "[ \t]+$" docs/planning/EPIC_10/SPRINT_107 docs/planning/EPIC_10/PROJECT_PLAN.md`:
  passed; no matches.

### Day 13 Exit State

Day 13 is complete when
`artifacts/day13-central-matrix-shell-deferral-contract.md` exists,
`src/sparse_matrix.c` non-extraction is explicit, future split prerequisites
are concrete, no public API or install-header change is introduced, and
documentation hygiene checks pass.

## Day 14 - Validation, Metrics & Closeout

### Goal

Validate all Sprint 107 touched surfaces, publish final maintainability
metrics, verify no public API/install-header/test-count drift, and close the
sprint with a residual handoff.

### Actions

- Re-read the Day 14 plan section in
  `docs/planning/EPIC_10/SPRINT_107/PLAN.md`.
- Collected touched-file status, line deltas, artifact inventory, and current
  residual owner sizes.
- Verified the branch has no tracked diff in public headers, internal headers,
  Makefile, CMake files, `.github`, or install metadata.
- Verified no `RUN_TEST` registration lines were added or removed in the
  touched test files.
- Wrote the Day 14 validation, metrics, and closeout artifact in
  `artifacts/day14-validation-metrics-closeout.md`.

### Findings

- Sprint 107 touched planning documentation and four `.c` test proof owners.
- Sprint 107 did not touch implementation sources under `src/`, headers,
  build-system files, install metadata, workflows, CTest registration files,
  or helper-target definitions.
- The tracked test owner line deltas are:
  - `tests/test_ldlt_csc.c`: 3,884 to 3,885 lines.
  - `tests/test_qr.c`: 3,234 to 3,210 lines.
  - `tests/test_iterative.c`: 2,841 to 2,828 lines.
  - `tests/test_svd.c`: 2,879 to 2,885 lines.
- The residual source owners remain documented rather than split:
  - `src/sparse_eigs.c`: 1,538 lines.
  - `src/sparse_matrix.c`: 1,359 lines.

### Validation Expectations

- Sprint 107 changed `.c` test files, so the required closeout gate is:
  - focused tests for the touched test owners;
  - `make format`
  - `make lint`
  - `make test`
- Hygiene checks:
  - `git diff --check`
  - trailing-whitespace scan on Sprint 107 docs, the Epic 10 project plan, and
    touched `.c` files.

### Validation Results

- `make build/test_ldlt_csc && ./build/test_ldlt_csc`: passed; focused
  suite ran 100 tests with 0 failures.
- `make build/test_qr && ./build/test_qr`: passed; focused suite ran 73
  tests with 0 failures.
- `make build/test_iterative && ./build/test_iterative`: passed; focused
  suite ran 80 tests with 0 failures.
- `make build/test_svd && ./build/test_svd`: passed; focused suite ran 98
  tests with 0 failures.
- `make format && make lint && make test`: passed.
- `git diff --check`: passed.
- `rg -n "[ \t]+$" docs/planning/EPIC_10/SPRINT_107 docs/planning/EPIC_10/PROJECT_PLAN.md tests/test_ldlt_csc.c tests/test_qr.c tests/test_iterative.c tests/test_svd.c`:
  passed; no matches.

### Day 14 Exit State

Day 14 is complete when
`artifacts/day14-validation-metrics-closeout.md` exists, all focused and full
quality checks pass, drift checks are documented, Sprint 107 residual debt is
handed forward with rationale, and working notes contain the final validation
results.
