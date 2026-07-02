# Sprint 102 Working Notes

## Sprint Context

Sprint 102 implements "Direct Solver Robustness & External Oracle Expansion"
from `docs/planning/EPIC_10/PROJECT_PLAN.md`. The sprint deepens correctness
evidence for direct solvers with named dense-reference or external-comparison
lanes while keeping proof ownership family-local and claim wording bounded.

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

Sprint 102 may earn only bounded direct-solver evidence claims tied to named
fixtures, helper behavior, solver family, validation commands, and tolerance
rules.

Sprint 102 must not claim:

- direct solver APIs that accept `SparseCsr` or `SparseCsc` directly unless a
  future implementation explicitly adds and validates them;
- broad compressed parity across every solver family;
- portable performance superiority;
- full external-oracle coverage for every direct solver;
- broad state-of-the-art replacement status;
- mutable `SparseMatrix` deprecation.

## Day 1 - Scope and Evidence Baseline

### Goal

Convert the Sprint 102 project-plan section and Sprint 100/101 handoffs into a
bounded direct-solver evidence package with clear workstreams, validation
expectations, and non-claim boundaries.

### Actions

- Re-read the Sprint 102 section of
  `docs/planning/EPIC_10/PROJECT_PLAN.md`.
- Re-read Sprint 100 solver comparison evidence rules:
  - `docs/planning/EPIC_10/SPRINT_100/artifacts/day9-solver-comparison-template.md`
  - `docs/planning/EPIC_10/SPRINT_100/artifacts/templates/solver-comparison-evidence-template.md`
- Re-read Sprint 101 handoff and claim-boundary records:
  - `docs/planning/EPIC_10/SPRINT_101/artifacts/day13-validation-and-reconciliation.md`
  - `docs/planning/EPIC_10/SPRINT_101/artifacts/day14-closeout-and-handoff.md`
- Created the Sprint 102 artifacts directory.
- Recorded authoritative Day 1 inputs in
  `artifacts/day1-authoritative-inputs.txt`.
- Recorded the Sprint 102 scope baseline, day ownership, validation rules, and
  claim boundaries in `artifacts/day1-scope-baseline.md`.

### Findings

- Sprint 100 requires every future solver comparison claim to name fixture
  set, oracle/reference behavior, tolerance or acceptance criteria, validation
  command, unsupported cases, and remaining non-claims.
- Sprint 101 gives Sprint 102 a stable compressed-input-to-`SparseMatrix`
  workflow but does not provide direct CSR/CSC solver APIs or universal
  compressed solver parity.
- Sprint 102 should start by auditing direct-solver evidence depth before
  adding helpers or fixtures.
- Fixture taxonomy must precede new oracle expansion so expected failures do
  not blur into correctness passes.

### Validation Expectations

- Day 1 changes planning documentation only.
- Required checks:
  - `git diff --check`
  - trailing-whitespace scan on `docs/planning/EPIC_10/SPRINT_102`

### Validation Results

- `git diff --check`: passed.
- `rg -n "[ \t]+$" docs/planning/EPIC_10/SPRINT_102`: passed; no matches.

### Day 1 Exit State

Day 1 is complete. Sprint 102 now has working notes, authoritative inputs,
scope baseline, workstream ownership, validation expectations, and preserved
Sprint 101 non-claim boundaries.

## Day 2 - Direct Solver Gap Audit

### Goal

Inventory Cholesky, LDLT, LU, QR, SVD, and direct-dispatch tests before new
oracle, fixture, or helper work begins.

### Actions

- Re-read Sprint 102 Day 2 in
  `docs/planning/EPIC_10/SPRINT_102/PLAN.md`.
- Inventoried direct-solver test, helper, and implementation owners.
- Counted test-owner size and `RUN_TEST` concentration for direct-solver test
  files.
- Located existing external dense-reference lanes:
  - `tests/chol_external_dense_reference.py` with `tests/test_chol_csc.c`
  - `tests/ldlt_external_dense_reference.py` with `tests/test_ldlt_csc.c`
- Reviewed LU, QR, and SVD tests for residual, rank, singular, reconstruction,
  SuiteSparse, and expected-failure coverage.
- Recorded the Day 2 gap audit in
  `artifacts/day2-direct-solver-gap-audit.md`.

### Findings

- Cholesky CSC and LDLT CSC are the only direct-family lanes with external
  dense-reference helpers today.
- LU has high user value and strong residual/failure coverage, but no external
  dense-reference helper lane.
- QR has broad internal invariant coverage and good rank/rectangular cases,
  but no external dense least-squares or rank oracle lane.
- SVD has broad internal reconstruction/rank/condition coverage, but an
  external dense SVD oracle would be heavier and needs taxonomy first.
- Proof-owner concentration is significant in `tests/test_ldlt_csc.c`,
  `tests/test_qr.c`, `tests/test_ldlt.c`, `tests/test_svd.c`, and
  `tests/test_chol_csc.c`.

### Ranked Queue

1. LU external dense-reference solve lane.
2. QR dense-reference least-squares or rank lane.
3. LDLT CSC external fixture expansion.
4. Cholesky CSC external fixture expansion.
5. SVD dense-reference lane.
6. Direct CSC dispatch reporting or oracle consumption lane.

### Validation Expectations

- Day 2 changes planning documentation only.
- Required checks:
  - `git diff --check`
  - trailing-whitespace scan on `docs/planning/EPIC_10/SPRINT_102`

### Validation Results

- `git diff --check`: passed.
- `rg -n "[ \t]+$" docs/planning/EPIC_10/SPRINT_102`: passed; no matches.

### Day 2 Exit State

Day 2 is complete. Cholesky, LDLT, LU, QR, SVD, and direct-dispatch paths are
classified, proof-owner concentration is recorded, and the ranked expansion
queue is ready for Day 3 fixture taxonomy.

## Day 3 - Fixture Taxonomy Design

### Goal

Define solver-neutral and family-local fixture classes, expected outcomes,
expected failures, naming rules, and storage/generation rules before adding
new direct-solver oracle coverage.

### Actions

- Re-read Sprint 102 Day 3 in
  `docs/planning/EPIC_10/SPRINT_102/PLAN.md`.
- Re-read the Day 2 direct-solver gap audit.
- Inventoried existing Matrix Market fixtures under `tests/data` and
  `tests/data/suitesparse`.
- Reviewed existing external helper fixture behavior in:
  - `tests/chol_external_dense_reference.py`
  - `tests/ldlt_external_dense_reference.py`
- Scanned direct solver tests for existing fixture names and expected
  singular, rank-deficient, rectangular, SPD, indefinite, and SuiteSparse
  behavior.
- Recorded the Day 3 fixture taxonomy in
  `artifacts/day3-fixture-taxonomy.md`.

### Findings

- Existing Cholesky external coverage is centered on SPD Matrix Market
  fixtures `nos4` and `bcsstk04`.
- Existing LDLT external coverage is centered on synthetic indefinite KKT
  fixtures `kkt5` and `kkt10`.
- LU already has useful nonsymmetric SuiteSparse residual fixtures
  `orsirr_1` and `steam1`, but a small deterministic nonsymmetric dense
  reference fixture is a better first external-oracle lane.
- QR and SVD already cover many rectangular/rank cases internally; external
  oracle expansion should name whether it targets least-squares, rank,
  singular values, or reconstruction before implementation.
- Expected failures need first-class taxonomy entries so singular,
  indefinite, rectangular, malformed, and unavailable-helper behavior does not
  look like either a correctness pass or an unexpected regression.

### Validation Expectations

- Day 3 changes planning documentation only.
- Required checks:
  - `git diff --check`
  - trailing-whitespace scan on `docs/planning/EPIC_10/SPRINT_102`

### Validation Results

- `git diff --check`: passed.
- `rg -n "[ \t]+$" docs/planning/EPIC_10/SPRINT_102`: passed; no matches.

### Day 3 Exit State

Day 3 is complete. Sprint 102 now has fixture classes, expected-failure
classes, solver-family mapping, naming rules, and storage/generation rules
that Day 4 can use to freeze the oracle helper boundary.

## Day 4 - Oracle Helper Boundary Freeze

### Goal

Decide the smallest dense-reference helper extraction that improves
proof-owner maintainability without widening solver APIs or hiding
family-specific fixture, tolerance, residual, or solver behavior.

### Actions

- Re-read Sprint 102 Day 4 in
  `docs/planning/EPIC_10/SPRINT_102/PLAN.md`.
- Re-read the Day 3 fixture taxonomy.
- Reviewed current Cholesky CSC external dense-reference C harness glue in
  `tests/test_chol_csc.c`.
- Reviewed current LDLT CSC external dense-reference C harness glue in
  `tests/test_ldlt_csc.c`.
- Reviewed Python helper output contracts in:
  - `tests/chol_external_dense_reference.py`
  - `tests/ldlt_external_dense_reference.py`
- Reviewed existing test helper location `tests/test_solver_helpers.h`.
- Recorded the Day 4 helper boundary in
  `artifacts/day4-oracle-helper-boundary.md`.

### Findings

- Cholesky and LDLT duplicate subprocess/vector parsing, status handling,
  dimension checks, parse-failure handling, and pipe-close handling.
- Cholesky and LDLT should keep matrix construction, solver execution,
  permutation handling, tolerances, residual checks, and assertions
  family-local.
- The smallest useful Day 5 extraction is a test-only
  `tf_read_external_reference_vector(...)` helper in
  `tests/test_solver_helpers.h`.
- Day 5 is expected to modify `.c` and `.h` test files, so the full
  `make format && make lint && make test` quality chain will be required.

### Validation Expectations

- Day 4 changes planning documentation only.
- Required checks:
  - `git diff --check`
  - trailing-whitespace scan on `docs/planning/EPIC_10/SPRINT_102`

### Validation Results

- `git diff --check`: passed.
- `rg -n "[ \t]+$" docs/planning/EPIC_10/SPRINT_102`: passed; no matches.

### Day 4 Exit State

Day 4 is complete. The selected Day 5 boundary is a test-only external
reference vector reader in `tests/test_solver_helpers.h`, with Cholesky and
LDLT solver behavior, fixture construction, tolerances, residuals, and claim
boundaries kept family-local.

## Day 5 - Oracle Helper Extraction Batch 1

### Goal

Implement the Day 4 helper extraction without changing direct-solver behavior
or widening public solver APIs.

### Actions

- Re-read Sprint 102 Day 5 in
  `docs/planning/EPIC_10/SPRINT_102/PLAN.md`.
- Re-read the Day 4 oracle helper boundary.
- Added an opt-in `tf_external_reference_status_t` enum and
  `tf_read_external_reference_vector(...)` helper to
  `tests/test_solver_helpers.h`.
- Updated `tests/test_chol_csc.c` to keep command construction local while
  delegating external dense-reference vector parsing to the shared helper.
- Updated `tests/test_ldlt_csc.c` to keep command construction and LDLT
  permutation/solve behavior local while delegating vector parsing to the
  shared helper.
- Recorded the Day 5 implementation evidence in
  `artifacts/day5-oracle-helper-extraction.md`.

### Findings

- The helper extraction removes duplicated `OK`/`SKIP`/`ERROR` parsing and
  vector parsing from Cholesky and LDLT harnesses.
- The helper is opt-in so unrelated users of `tests/test_solver_helpers.h`
  continue to use only the residual helpers.
- The Cholesky CSC and LDLT CSC external dense-reference tests still pass and
  still exercise their Python helpers.
- No solver implementation, public API, build registration, or public
  documentation changed.

### Validation Results

- `make format`: passed.
- `make build/test_chol_csc`: passed.
- `./build/test_chol_csc`: passed; 92 tests, 0 failures, 0 skips, 20844
  assertions.
- `make build/test_ldlt_csc`: passed.
- `./build/test_ldlt_csc`: passed; 98 tests, 0 failures, 0 skips, 2288
  assertions.
- `make lint`: passed.
- `make test`: passed.
- `git diff --check`: passed.
- `rg -n "[ \t]+$" tests/test_solver_helpers.h tests/test_chol_csc.c tests/test_ldlt_csc.c docs/planning/EPIC_10/SPRINT_102`:
  passed; no matches.

### Day 5 Exit State

Day 5 is complete. External dense-reference vector parsing now has one opt-in
test-support helper, and Cholesky/LDLT external oracle behavior is preserved
under focused tests plus the full required quality chain.

## Day 6 - Helper Extraction Closeout and Rerank

### Goal

Validate the helper extraction closeout state and rerank remaining Sprint 102
solver evidence lanes before CSC-family oracle expansion begins.

### Actions

- Re-read Sprint 102 Day 6 in
  `docs/planning/EPIC_10/SPRINT_102/PLAN.md`.
- Re-read the Day 5 helper extraction artifact.
- Re-read Day 2 gap scores and Day 3 fixture taxonomy recommendations.
- Reran focused validation for:
  - `make build/test_chol_csc`
  - `./build/test_chol_csc`
  - `make build/test_ldlt_csc`
  - `./build/test_ldlt_csc`
- Recorded the Day 6 closeout and rerank in
  `artifacts/day6-helper-closeout-and-rerank.md`.

### Findings

- The shared parser extraction is validated and does not change Cholesky or
  LDLT external oracle behavior.
- The extraction removes parser duplication but does not add any new solver
  correctness fixture.
- Within the Day 7-9 CSC-family window, LDLT CSC should consume the next
  boundary slot because Cholesky CSC already has the stronger external SPD
  fixture baseline.
- Within the Day 10-11 general direct-solver window, LU remains the highest
  value missing external dense-reference lane, followed by QR.

### Validation Results

- `make build/test_chol_csc`: passed; target was up to date.
- `./build/test_chol_csc`: passed; 92 tests, 0 failures, 0 skips, 20844
  assertions.
- `make build/test_ldlt_csc`: passed; target was up to date.
- `./build/test_ldlt_csc`: passed; 98 tests, 0 failures, 0 skips, 2288
  assertions.
- `git diff --check`: passed.
- `rg -n "[ \t]+$" tests/test_solver_helpers.h tests/test_chol_csc.c tests/test_ldlt_csc.c docs/planning/EPIC_10/SPRINT_102`:
  passed; no matches.

### Day 6 Exit State

Day 6 is complete. The helper extraction is closed out, focused validation
passed, and Day 7 should freeze the LDLT CSC scaled-KKT external fixture
boundary while preserving LU as the Day 10 general-solver expansion candidate.

## Day 7 - LDLT/Cholesky Oracle Boundary Freeze

### Goal

Freeze the highest-value CSC direct-family oracle expansion before adding new
coverage.

### Actions

- Re-read Sprint 102 Day 7 in
  `docs/planning/EPIC_10/SPRINT_102/PLAN.md`.
- Re-read the Day 3 fixture taxonomy, Day 4 helper boundary, Day 5 helper
  extraction, and Day 6 rerank artifacts.
- Re-read Sprint 98 LDLT CSC external-reference evidence and the Sprint 100
  Cholesky CSC comparison pilot.
- Inspected the existing LDLT external helper and C harness boundary in:
  - `tests/ldlt_external_dense_reference.py`
  - `tests/test_ldlt_csc.c`
- Ran a one-off dense Gaussian-elimination sanity check for the proposed
  scaled KKT construction.
- Recorded the Day 7 CSC oracle boundary in
  `artifacts/day7-csc-oracle-boundary.md`.

### Findings

- LDLT CSC remains the correct CSC-family expansion target because Cholesky
  CSC already owns external SPD proof on `nos4` and `bcsstk04`.
- The selected Day 8 fixture is `ldlt_kkt_scaled_10`, an
  `indef-kkt-scaled` 10x10 synthetic KKT fixture with moderate scale variation
  in both the SPD and coupling blocks.
- The fixture should preserve the existing LDLT external-lane contract:
  `x_true[i] = i + 1`, `b = A*x_true`, LDLT CSC two-pass solve in C, and dense
  Gaussian-elimination reference in Python.
- The proposed fixture recovered `x_true = 1..10` with
  `max|x - x_true| = 8.882e-15` in a dense sanity check, so the Day 8 starting
  tolerance remains `1e-10`.
- Cholesky on the selected indefinite KKT fixture remains unsupported and must
  not be counted as correctness proof.

### Validation Expectations

- Day 7 changes planning documentation only.
- Required checks:
  - `git diff --check`
  - trailing-whitespace scan on `docs/planning/EPIC_10/SPRINT_102`

### Validation Results

- `git diff --check`: passed.
- `rg -n "[ \t]+$" docs/planning/EPIC_10/SPRINT_102`: passed; no matches.

### Day 7 Exit State

Day 7 is complete. The selected CSC-family expansion is
`ldlt_kkt_scaled_10`, with explicit fixture construction, tolerance,
failure-mode, validation, and non-claim boundaries ready for Day 8
implementation.

## Day 8 - LDLT/Cholesky Oracle Expansion Batch

### Goal

Implement the selected CSC direct-family oracle expansion and validate it
without widening direct-solver claims.

### Actions

- Re-read Sprint 102 Day 8 in
  `docs/planning/EPIC_10/SPRINT_102/PLAN.md`.
- Re-read the Day 7 CSC oracle boundary artifact.
- Added `ldlt_kkt_scaled_10` to
  `tests/ldlt_external_dense_reference.py`.
- Added the matching C sparse fixture builder
  `build_kkt_scaled_10x10()` to `tests/test_ldlt_csc.c`.
- Added and registered
  `test_s102_external_dense_reference_scaled_kkt_10x10` next to the existing
  Sprint 98 LDLT CSC external-reference tests.
- Reused the existing LDLT external dense-reference assertion path and Day 5
  `tf_read_external_reference_vector(...)` helper.
- Recorded the Day 8 implementation evidence in
  `artifacts/day8-csc-oracle-expansion-batch.md`.

### Findings

- The Python helper emits `OK 10` for `ldlt_kkt_scaled_10` and recovers
  `x_true = 1..10` to roundoff.
- The new LDLT CSC external lane passes with
  `max|x - x_ref| = 8.882e-15` and `rel_residual = 1.692e-17`.
- The existing `kkt5` and `kkt10` external lanes still pass with their prior
  roundoff-level metrics.
- The Day 7 `1e-10` tolerance remained valid; no relaxation was needed.
- No public headers, library sources, build files, public docs, or Cholesky
  CSC tests changed for Day 8.

### Validation Results

- `make format`: passed.
- `python3 tests/ldlt_external_dense_reference.py ldlt_kkt_scaled_10`:
  passed; emitted `OK 10`.
- `make build/test_ldlt_csc`: passed.
- `./build/test_ldlt_csc`: passed; 99 tests, 0 failures, 0 skips, 2318
  assertions.
- `make lint`: passed.
- `make test`: passed; `All tests passed.`
- `git diff --check`: passed.
- `rg -n "[ \t]+$" tests/ldlt_external_dense_reference.py tests/test_ldlt_csc.c tests/test_chol_csc.c tests/test_solver_helpers.h docs/planning/EPIC_10/SPRINT_102`:
  passed; no matches.

### Day 8 Exit State

Day 8 is complete. Sprint 102 now has a validated LDLT CSC external
dense-reference fixture for `ldlt_kkt_scaled_10`, while public API, Cholesky,
build, and broad direct-solver claims remain unchanged.

## Day 9 - CSC Closeout and LU/QR/SVD Rerank

### Goal

Close the CSC direct-family expansion and select the next general direct-solver
oracle lane before implementation starts.

### Actions

- Re-read Sprint 102 Day 9 in
  `docs/planning/EPIC_10/SPRINT_102/PLAN.md`.
- Re-read the Day 8 CSC oracle expansion artifact.
- Reran the `ldlt_kkt_scaled_10` Python helper.
- Reran the focused LDLT CSC binary.
- Compared Day 8 evidence against the Day 7 boundary criteria.
- Re-read the Day 2 direct-solver gap audit and Day 3 fixture taxonomy.
- Inspected current LU test surfaces in `tests/test_sparse_lu.c` and
  `tests/test_lu_csr.c`.
- Recorded the Day 9 CSC closeout and LU/QR/SVD rerank in
  `artifacts/day9-csc-closeout-and-general-rerank.md`.

### Findings

- The CSC expansion meets every Day 7 acceptance criterion without tolerance
  relaxation.
- `ldlt_kkt_scaled_10` remains stable at `max|x - x_ref| = 8.882e-15` and
  `rel_residual = 1.692e-17`.
- The proof-owner boundary remains correct: LDLT CSC owns fixture construction,
  solve path, permutation mapping, tolerance, and residual checks; the shared
  helper owns only external-reference vector parsing.
- Remaining CSC ideas should be deferred rather than folded into the
  LU/QR/SVD window.
- LU remains the highest-value general direct-solver external-oracle gap.
  Day 10 should select a bounded linked-list LU dense-reference lane unless
  implementation inspection finds that boundary too large.

### Validation Results

- `python3 tests/ldlt_external_dense_reference.py ldlt_kkt_scaled_10`:
  passed; emitted `OK 10`.
- `make build/test_ldlt_csc`: passed; target was up to date.
- `./build/test_ldlt_csc`: passed; 99 tests, 0 failures, 0 skips, 2318
  assertions.
- `git diff --check`: passed.
- `rg -n "[ \t]+$" tests/ldlt_external_dense_reference.py tests/test_ldlt_csc.c tests/test_chol_csc.c tests/test_solver_helpers.h docs/planning/EPIC_10/SPRINT_102`:
  passed; no matches.

### Day 9 Exit State

Day 9 is complete. The CSC-family oracle expansion is closed and the selected
Day 10 general direct-solver boundary target is LU, with QR retained as the
backup lane and SVD deferred.

## Day 10 - LU/QR/SVD Oracle Boundary Freeze

### Goal

Freeze the selected LU, QR, or SVD oracle and failure-mode expansion before
implementation.

### Actions

- Re-read Sprint 102 Day 10 in
  `docs/planning/EPIC_10/SPRINT_102/PLAN.md`.
- Re-read the Day 9 CSC closeout and general solver rerank artifact.
- Re-inspected linked-list LU tests in `tests/test_sparse_lu.c`.
- Re-inspected LU CSR tests in `tests/test_lu_csr.c` as a possible follow-up
  surface.
- Re-read public LU guidance in `include/sparse_lu.h` and `docs/tutorial.md`.
- Rechecked QR and SVD surfaces against the Day 2 and Day 3 rankings.
- Recorded the Day 10 general solver boundary in
  `artifacts/day10-general-solver-oracle-boundary.md`.

### Findings

- Linked-list LU is the right Day 11 owner because it is the primary one-shot
  LU API path and lacks external dense-reference proof.
- LU CSR should stay out of Day 11 so the first LU oracle lane remains bounded.
- The selected positive fixture is `lu_nonsym_square_5`, class
  `nonsym-square-small`, with `x_true[i] = i + 1` and dense-reference solve
  comparison.
- The selected expected-failure fixture is `lu_singular_square_4`, class
  `square-rank-def`, with `SPARSE_ERR_SINGULAR` expected from C LU
  factorization.
- Day 11 should add a small `tests/lu_external_dense_reference.py` helper only
  if it emits the same `OK n` vector contract consumed by
  `tf_read_external_reference_vector(...)`.
- QR remains the backup lane and SVD remains deferred.

### Validation Expectations

- Day 10 changes planning documentation only.
- Required checks:
  - `git diff --check`
  - trailing-whitespace scan on `docs/planning/EPIC_10/SPRINT_102`

### Validation Results

- `git diff --check`: passed.
- `rg -n "[ \t]+$" docs/planning/EPIC_10/SPRINT_102`: passed; no matches.

### Day 10 Exit State

Day 10 is complete. The selected Day 11 implementation lane is linked-list LU
external dense-reference coverage for `lu_nonsym_square_5`, plus deterministic
singular detection for `lu_singular_square_4`.

## Day 11 - General Solver Oracle Expansion Batch

### Goal

Implement the selected linked-list LU oracle and failure-mode expansion while
keeping the proof boundary narrow and family-local.

### Actions

- Re-read Sprint 102 Day 11 in
  `docs/planning/EPIC_10/SPRINT_102/PLAN.md`.
- Re-read the Day 10 general solver oracle boundary.
- Added `tests/lu_external_dense_reference.py` with deterministic positive
  and singular LU fixtures.
- Updated `tests/test_sparse_lu.c` to opt into the shared external-reference
  parser, build the selected sparse fixtures, compare linked-list LU against
  the external dense-reference solution, and assert deterministic singular
  detection.
- Registered the new positive and expected-failure tests in the linked-list LU
  test harness.
- Recorded the Day 11 implementation evidence in
  `artifacts/day11-general-solver-oracle-expansion-batch.md`.

### Findings

- The LU helper emits `OK 5` for `lu_nonsym_square_5` and recovers
  `x_true = 1..5` to roundoff.
- The LU helper emits `ERROR matrix is singular to dense reference tolerance`
  and exits nonzero for `lu_singular_square_4`.
- The new linked-list LU external lane passes with
  `max|x - x_ref| = 8.882e-16` and `residual = 3.553e-15`.
- The singular C fixture returns `SPARSE_ERR_SINGULAR` under
  `SPARSE_PIVOT_COMPLETE`.
- The Day 10 `1e-10` comparison tolerance remained valid; no relaxation was
  needed.
- LU CSR, QR, SVD, direct CSC dispatch, public headers, library sources, build
  files, and public documentation were not changed for Day 11.

### Validation Results

- `python3 tests/lu_external_dense_reference.py lu_nonsym_square_5`: passed;
  emitted `OK 5`.
- `python3 tests/lu_external_dense_reference.py lu_singular_square_4`: passed
  as an expected helper failure; emitted `ERROR matrix is singular to dense
  reference tolerance` and exited with status `1`.
- `make build/test_sparse_lu`: passed.
- `./build/test_sparse_lu`: passed; 39 tests, 0 failures, 0 skips, 144
  assertions.
- `make format`: passed.
- `make lint`: passed.
- `make test`: passed; `All tests passed.`
- `git diff --check`: passed.
- `rg -n "[ \t]+$" tests/lu_external_dense_reference.py tests/test_sparse_lu.c tests/ldlt_external_dense_reference.py tests/test_chol_csc.c tests/test_ldlt_csc.c tests/test_solver_helpers.h docs/planning/EPIC_10/SPRINT_102`:
  passed; no matches.

### Day 11 Exit State

Day 11 is complete. Sprint 102 now has a validated linked-list LU external
dense-reference lane for `lu_nonsym_square_5`, plus deterministic singular
detection for `lu_singular_square_4`, while LU CSR, QR, SVD, direct CSC
dispatch, public APIs, and broad solver claims remain unchanged.

## Day 12 - Direct Solver Guidance Update

### Goal

Update direct-solver selection, capability, failure, and trust-boundary
documentation from validated Sprint 102 evidence without adding unsupported
solver-family-wide claims.

### Actions

- Re-read Sprint 102 Day 12 in
  `docs/planning/EPIC_10/SPRINT_102/PLAN.md`.
- Re-read the Day 11 general solver oracle expansion artifact.
- Reviewed direct-solver guidance in `README.md`, `docs/tutorial.md`,
  `docs/maintainer_guide.md`, and `examples/README.md`.
- Updated `README.md` with bounded solver-selection guidance for LU,
  Cholesky, LDL^T, and QR, plus family-local external evidence wording.
- Updated `docs/tutorial.md` with bounded trust notes for LU, Cholesky, and
  LDL^T direct-solver evidence.
- Updated `docs/maintainer_guide.md` with Sprint 102 proof-owner updates and a
  direct-solver trust-boundary table.
- Recorded the Day 12 documentation evidence in
  `artifacts/day12-direct-solver-guidance-update.md`.

### Findings

- Public docs can safely say LU is the general square one-shot path, Cholesky
  is the SPD path, LDL^T is the symmetric indefinite path, and QR is the
  rectangular/rank-deficient least-squares path.
- External dense-reference confidence must remain tied to named family-local
  test owners and fixtures.
- The new LU external lane supports a bounded linked-list LU claim only; it
  does not support LU CSR, direct compressed LU API, or broad nonsymmetric
  ecosystem claims.
- QR and SVD remain internally tested but did not receive Sprint 102 external
  dense-reference oracle lanes.
- `examples/README.md` already points users toward one-shot direct,
  compressed input, repeated-run direct, and tutorial surfaces, so no Day 12
  example README edit was needed.

### Validation Results

- Day 12 changed documentation only.
- `git diff --check`: passed.
- `rg -n "[ \t]+$" README.md docs/tutorial.md docs/maintainer_guide.md docs/planning/EPIC_10/SPRINT_102`:
  passed; no matches.

### Day 12 Exit State

Day 12 is complete. Public and maintainer direct-solver guidance now reflects
validated Sprint 102 evidence while preserving non-claims around broad solver
superiority, complete external-oracle coverage, and direct compressed solver
APIs.

## Day 13 - Full Validation and Evidence Reconciliation

### Goal

Run final required validation and reconcile Sprint 102 evidence, claims, and
residual follow-up before closeout.

### Actions

- Re-read Sprint 102 Day 13 in
  `docs/planning/EPIC_10/SPRINT_102/PLAN.md`.
- Re-read the Day 3 fixture taxonomy and Sprint 100 solver comparison evidence
  template.
- Ran focused external-helper checks for `ldlt_kkt_scaled_10`,
  `lu_nonsym_square_5`, and `lu_singular_square_4`.
- Ran focused LDLT CSC and linked-list LU test binaries.
- Ran the full required code-touch quality chain:
  `make format && make lint && make test`.
- Reconciled new Sprint 102 evidence against the Day 3 fixture classes and the
  Sprint 100 solver evidence template.
- Recorded earned, deferred, and non-claim states, plus explicit Sprint 103
  dependency notes.
- Wrote the Day 13 validation artifact in
  `artifacts/day13-validation-and-evidence-reconciliation.md`.

### Findings

- `ldlt_kkt_scaled_10` remains a valid `indef-kkt-scaled` LDLT CSC external
  reference lane with `max|x - x_ref| = 8.882e-15` and
  `rel_residual = 1.692e-17`.
- `lu_nonsym_square_5` remains a valid `nonsym-square-small` linked-list LU
  external reference lane with `max|x - x_ref| = 8.882e-16` and
  `residual = 3.553e-15`.
- `lu_singular_square_4` remains a valid `square-rank-def` expected-failure
  lane with `SPARSE_ERR_SINGULAR` in C and helper `ERROR` status.
- Sprint 102 earned bounded Cholesky/LDLT/LU trust-boundary wording, but did
  not earn LU CSR, QR, SVD, direct compressed solver API, or broad
  solver-superiority claims.
- Sprint 103 should start from the maintainer-guide trust-boundary table and
  define any QR, SVD, or LU CSR oracle lane before implementation.

### Validation Results

- `python3 tests/ldlt_external_dense_reference.py ldlt_kkt_scaled_10`:
  passed; emitted `OK 10`.
- `python3 tests/lu_external_dense_reference.py lu_nonsym_square_5`: passed;
  emitted `OK 5`.
- `python3 tests/lu_external_dense_reference.py lu_singular_square_4`: passed
  as an expected helper failure; emitted `ERROR matrix is singular to dense
  reference tolerance` and exited with status `1`.
- `make build/test_ldlt_csc build/test_sparse_lu`: passed; targets were up to
  date.
- `./build/test_ldlt_csc`: passed; 99 tests, 0 failures, 0 skips, 2318
  assertions.
- `./build/test_sparse_lu`: passed; 39 tests, 0 failures, 0 skips, 144
  assertions.
- `make format`: passed.
- `make lint`: passed.
- `make test`: passed; `All tests passed.`
- `git diff --check`: passed.
- `rg -n "[ \t]+$" README.md docs/tutorial.md docs/maintainer_guide.md tests/lu_external_dense_reference.py tests/ldlt_external_dense_reference.py tests/test_chol_csc.c tests/test_ldlt_csc.c tests/test_solver_helpers.h tests/test_sparse_lu.c docs/planning/EPIC_10/SPRINT_102`:
  passed; no matches.

### Day 13 Exit State

Day 13 is complete. All required validation passed, direct-solver claims are
tied to named tests, fixtures, helpers, and validation commands, and Sprint
103 dependencies are explicit.

## Day 14 - Sprint Closeout and Handoff

### Goal

Close Sprint 102 with validated direct-solver oracle evidence and a clear
handoff to Sprint 103.

### Actions

- Re-read Sprint 102 Day 14 in
  `docs/planning/EPIC_10/SPRINT_102/PLAN.md`.
- Reviewed Sprint 102 project-plan items against all Day 1-13 artifacts.
- Reviewed Sprint 100 and Sprint 101 Day 14 closeout/index formats for
  consistency.
- Created the Sprint 102 artifact index in
  `artifacts/day14-artifact-index.md`.
- Created the Sprint 102 closeout and handoff artifact in
  `artifacts/day14-closeout-and-handoff.md`.
- Recorded Sprint 103 handoff requirements, residual direct-solver queues,
  retrospective inputs, final validation notes, and non-claim boundaries.

### Findings

- Every Sprint 102 project-plan item has a corresponding artifact, code
  surface, documentation surface, or explicit deferral.
- Sprint 102 earned bounded external-reference evidence for named LDLT CSC and
  linked-list LU fixtures, plus shared parser reuse across direct-solver test
  lanes.
- QR, SVD, LU CSR, direct compressed solver APIs, and broad solver superiority
  remain deferred or non-claim states.
- Sprint 103 can start from the Day 3 fixture taxonomy, Day 12 trust-boundary
  wording, Day 13 reconciliation artifact, and Day 14 handoff package.

### Validation Results

- Day 14 changed planning documentation only.
- Day 13 already reran the required code-touch validation chain for Sprint
  102's `.c` and `.h` changes:
  - `make format`: passed.
  - `make lint`: passed.
  - `make test`: passed; `All tests passed.`
- Required Day 14 hygiene pending:
- `git diff --check`: passed.
- `rg -n "[ \t]+$" docs/planning/EPIC_10/SPRINT_102`: passed; no matches.

### Day 14 Exit State

Day 14 is complete. Sprint 102 is closed from a complete and hygiene-checked
artifact set, and Sprint 103 can start from named direct-solver oracle
evidence, fixture taxonomy rules, trust-boundary wording, and explicit
comparison prerequisites.
