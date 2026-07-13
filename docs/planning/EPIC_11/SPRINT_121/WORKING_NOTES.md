# Sprint 121 Working Notes

## Sprint Goal

Sprint 121 strengthens SVD, QR, rank, pseudoinverse, and least-squares
evidence with reusable helpers while keeping external library parity and
state-of-the-art claims out of scope.

## Starting Constraints

- Treat Sprint 120 as the current fixture/oracle architecture baseline for
  helper extraction, focused validation, cross-solver pilot boundaries, and
  cleanup discipline.
- Do not start SVD, QR, rank-deficient, least-squares, pseudoinverse, or
  low-rank implementation before audit, taxonomy, helper boundaries, focused
  proof, source-list/CMake impact, expected CTest count, and rollback
  expectations are documented.
- Preserve solver-specific tolerances, residual interpretation,
  reconstruction interpretation, orthogonality thresholds, rank expectations,
  and expected-failure semantics at visible test boundaries.
- Do not claim broad LAPACK, SciPy, SuiteSparse, PETSc, Trilinos, Eigen, or
  state-of-the-art numerical parity from bounded SVD/QR evidence work.
- If `.c` or `.h` files change, run `make format && make lint && make test`.
- If Makefile, CMake, source-list, workflow, package, benchmark, script, or
  install surfaces change, run the relevant focused validation lane and record
  whether it is reviewed, supplemental, or local.
- If documentation only changes, run `git diff --check` and a focused
  trailing-whitespace scan over `docs/planning/EPIC_11/SPRINT_121`.

## Input Artifact Inventory

| Input | Sprint 121 use |
|---|---|
| `docs/planning/EPIC_11/PROJECT_PLAN.md` Sprint 121 section | Authoritative project-plan items, estimates, deliverables, and sprint goal. |
| `docs/planning/EPIC_11/SPRINT_121/PLAN.md` | Day-by-day execution plan and completion criteria. |
| `docs/planning/EPIC_11/SPRINT_120/PLAN.md` | Latest day-by-day pattern for oracle architecture and validation work. |
| `docs/planning/EPIC_11/SPRINT_120/WORKING_NOTES.md` | Current working-notes structure and validation boundary model. |
| `docs/planning/EPIC_11/SPRINT_120/RETROSPECTIVE.md` | Sprint 120 closeout, residual direct/iterative lessons, and oracle pilot handoff. |
| `docs/planning/EPIC_11/SPRINT_120/artifacts/day4-shared-fixture-architecture.md` | Shared fixture architecture pattern and helper-boundary rules. |
| `docs/planning/EPIC_11/SPRINT_120/artifacts/day11-cross-solver-oracle-pilot-design.md` | Bounded pilot design pattern, skip behavior, and non-claim framing. |
| `docs/planning/EPIC_11/SPRINT_120/artifacts/day12-cross-solver-oracle-pilot-implementation.md` | Pilot implementation and focused proof pattern. |
| `docs/planning/EPIC_11/SPRINT_120/artifacts/day13-validation-package.md` | Current focused/full validation package pattern. |
| `docs/planning/EPIC_11/SPRINT_120/artifacts/day14-oracle-closeout.md` | Handoff and residual queue pattern for the next sprint. |
| `docs/planning/EPIC_11/SPRINT_118/artifacts/day2-validation-inventory.md` | Validation lane inventory and command categories. |
| `docs/planning/EPIC_11/SPRINT_118/artifacts/day8-product-truth-map.md` | Product truth and public non-claim boundaries. |
| `docs/planning/EPIC_11/SPRINT_118/artifacts/day11-evidence-template-design.md` | Evidence-template rules for bounded oracle artifacts. |
| `docs/planning/EPIC_11/SPRINT_118/artifacts/day13-public-claim-drift-audit.md` | Claim-drift checks and non-claim wording. |
| `docs/planning/EPIC_11/SPRINT_118/templates/oracle-expansion-evidence-template.md` | Fields for bounded oracle expansion and comparison artifacts. |

## Day-Level Ownership

| Day | Planned Focus | Project Plan Item |
|---:|---|---|
| 1 | Sprint intake, artifact skeleton, input inventory, validation boundaries, and owner map. | Items 1-7 intake |
| 2 | SVD, partial-SVD, low-rank, rank, and pseudoinverse proof audit. | Item 1 |
| 3 | QR, least-squares, rank-deficient, rectangular, and reconstruction proof audit. | Item 1 |
| 4 | Matrix taxonomy design for deterministic rank, conditioning, shape, sparsity, scaling, and expected failures. | Item 2 |
| 5 | Helper extraction plan for SVD and QR proof helpers. | Items 2, 3, 4 |
| 6 | SVD helper extraction for reconstruction, orthogonality, rank, low-rank, and pseudoinverse checks. | Items 3, 6 |
| 7 | QR helper extraction for reconstruction, residual, least-squares, rank-deficient, and generated-RHS checks. | Items 4, 6 |
| 8 | Rank-deficient and near-dependent fixture expansion. | Items 2, 4, 6 |
| 9 | Least-squares, pseudoinverse, and rectangular matrix evidence expansion. | Items 4, 6 |
| 10 | Low-rank and partial-SVD evidence expansion. | Items 3, 6 |
| 11 | Dense-reference or external-process pilot design. | Item 5 |
| 12 | Dense-reference or external-process pilot implementation and focused proof. | Items 5, 6 |
| 13 | Full validation package and trust-boundary documentation updates. | Items 6, 7 |
| 14 | Closeout, residual SVD/QR/rank queue, non-claims, and retrospective inputs. | Item 7 |

## Validation Expectations

| Touched Surface | Required Checks |
|---|---|
| Documentation-only planning artifacts | `git diff --check`; focused trailing-whitespace scan over `docs/planning/EPIC_11/SPRINT_121`. |
| `.c` or `.h` source/header changes | `make format && make lint && make test`. |
| Source-list or Makefile membership | `make source-list-check` and the relevant focused build target. |
| CMake membership or test-owner changes | CMake configure/build and `ctest -N` count proof as affected. |
| SVD behavior | Focused tests for touched full SVD, partial SVD, low-rank, rank, reconstruction, orthogonality, or pseudoinverse paths. |
| QR behavior | Focused tests for touched QR, QR solve, least-squares, rank-deficient, reconstruction, generated-RHS, or QR-vs-reference paths. |
| Dense-reference or external pilot | Focused pilot test plus adjacent SVD/QR tests that share fixtures or helpers. |
| Public claim or support wording | Check against Sprint 118 product truth, Sprint 118 public-claim drift audit, Sprint 120 non-claim framing, and current README/docs wording. |

## Oracle Expansion Evidence Fields Required

Every helper extraction, fixture expansion, dense-reference pilot, or explicit
deferral artifact should record:

- scope and touched surfaces;
- baseline proof owners and current product-truth references;
- fixture taxonomy class and expected behavior;
- tolerance, residual, reconstruction, orthogonality, rank, and
  expected-failure ownership;
- old/new helper or file-boundary plan;
- source-list, Makefile, CMake, and CTest impact;
- focused SVD, QR, or pilot proof;
- validation commands and results;
- rollback or defer plan;
- non-claims preserved;
- residual handoff.

## Scope Boundaries

Sprint 121 may inspect, design, extract, expand, validate, or explicitly defer
the SVD/QR/rank evidence candidates named in the project plan. It should not:

- claim broad LAPACK, SciPy, SuiteSparse, PETSc, Trilinos, Eigen, or external
  solver parity;
- claim state-of-the-art numerical completeness or superiority;
- hide rank, residual, reconstruction, orthogonality, or failure-mode
  interpretation behind generic helper APIs;
- add broad external comparison architecture outside one bounded pilot;
- alter package, ABI, platform, benchmark, adoption, or public API surfaces
  unless required by bounded SVD/QR proof work and explicitly validated;
- silently defer high-risk proof gaps without residual owners.

## Day 1 Notes

- Created the Sprint 121 working-notes baseline and artifact directory.
- Re-read the Sprint 121 project-plan section and Sprint 121 day-by-day plan.
- Reviewed Sprint 120 planning, working notes, retrospective, shared fixture
  architecture, cross-solver pilot design/implementation, validation package,
  and closeout handoff as the current oracle architecture baseline.
- Reviewed Sprint 118 validation, product-truth, evidence-template, and
  public-claim drift inputs for validation and non-claim expectations.
- Mapped all Sprint 121 project-plan items to day-level owners.
- Recorded validation expectations for documentation-only, C/header,
  source-list/Makefile, CMake/CTest, SVD, QR, dense-reference/external pilot,
  and public-claim touched surfaces.
- Recorded required oracle expansion evidence fields before any helper
  extraction, fixture expansion, or pilot implementation begins.
- Added Day 1 sprint intake artifact:
  `artifacts/day1-sprint-intake.md`.
- Kept Day 1 documentation-only; no C source, header, build, workflow,
  package, benchmark, or test surfaces were modified.

## Day 2 Notes

- Audited SVD-facing source, header, tests, example, and benchmark surfaces:
  `include/sparse_svd.h`, `src/sparse_svd.c`,
  `src/sparse_svd_partial.c`, `src/sparse_bidiag.c`,
  `tests/test_svd.c`, `tests/test_svd_partial_helpers.h`,
  `examples/example_svd_lowrank.c`, and `benchmarks/bench_svd.c`.
- Recorded current proof owners for Golub-Kahan extraction, bidiagonal SVD,
  full SVD singular values, full SVD reconstruction/orthogonality, partial
  SVD values/vectors, rank, pseudoinverse, dense low-rank, sparse low-rank,
  and condition-number behavior.
- Identified that partial-SVD evidence mostly uses this library's full SVD as
  the reference oracle. That is useful regression coverage but must remain
  explicitly separate from external dense-library parity claims.
- Identified pseudoinverse gaps: current coverage owns diagonal inversion and
  the first Moore-Penrose identity, but not the remaining identities, wide
  fixtures, or rank-deficient pseudoinverse tolerance behavior.
- Identified low-rank gaps: sparse low-rank has bounded env-on/off equivalence
  evidence, while dense low-rank still needs clearer rectangular and
  rank-deficient taxonomy-backed proof owners.
- Recorded tolerance boundaries so Day 5-6 helper extraction can avoid mixing
  full-SVD `1e-10` reconstruction/orthogonality expectations with looser
  partial-SVD vector residuals and SuiteSparse relative windows.
- Captured Day 4 taxonomy inputs for exact diagonal spectra, thresholded rank,
  duplicate-column rank deficiency, rectangular shape coverage, SuiteSparse
  smoke fixtures, partial internal-reference fixtures, pseudoinverse identity
  fixtures, low-rank drop-tolerance fixtures, and condition-number fixtures.
- Added Day 2 SVD evidence audit artifact:
  `artifacts/day2-svd-evidence-audit.md`.
- Kept Day 2 documentation-only; no C source, header, build, workflow,
  package, benchmark, or test surfaces were modified.

## Day 3 Notes

- Audited QR-facing API, implementation, tests, example, and benchmark
  surfaces: `include/sparse_qr.h`, `src/sparse_qr.c`,
  `tests/test_qr.c`, `tests/test_qr_solve.c`, `tests/test_colamd.c`
  minimum-norm sections, `examples/example_minnorm.c`, and
  `benchmarks/bench_colamd.c`.
- Confirmed QR factorization, solve, rank, nullspace, minimum-norm, and
  refinement routines are centralized in `src/sparse_qr.c`; there is no
  separate `src/sparse_qr_solve.c`.
- Recorded current proof owners for basic QR factorization, reconstruction,
  Q orthogonality/application, rank/nullspace, square solve,
  overdetermined least squares, rank-deficient solve, underdetermined
  minimum-norm solve, iterative refinement, reordering/fill, economy mode,
  and sparse-mode QR.
- Identified ownership drift: minimum-norm QR tests are currently housed in
  `tests/test_colamd.c`, so Sprint 121 helper extraction should either
  document that ownership explicitly or move helper ownership carefully
  without changing reviewed test membership casually.
- Identified rank-deficient and rectangular gaps: inconsistent
  rank-deficient least-squares lacks a named expected-residual owner, and
  overdetermined fixtures need clearer taxonomy separation between
  compatible generated-RHS cases and incompatible true least-squares cases.
- Recorded tolerance boundaries for QR reconstruction, Q orthogonality,
  square solve residuals, least-squares residuals, nullspace residuals,
  minimum-norm norm/residual checks, refinement residual non-increase, and
  sparse/economy backend comparisons.
- Captured Day 4 taxonomy inputs for square exact, overdetermined
  compatible/incompatible, underdetermined minimum-norm, duplicate-column,
  dependent-row, near-rank-deficient, nullspace-owned, economy-mode,
  sparse-mode, reordered, and SuiteSparse smoke fixtures.
- Added Day 3 QR/rank-deficient evidence audit artifact:
  `artifacts/day3-qr-rank-deficient-evidence-audit.md`.
- Kept Day 3 documentation-only; no C source, header, build, workflow,
  package, benchmark, or test surfaces were modified.

## Day 4 Notes

- Combined Day 2 SVD audit inputs and Day 3 QR audit inputs into a shared
  deterministic matrix taxonomy for SVD, QR, rank, pseudoinverse,
  least-squares, low-rank, and expected-failure proof work.
- Defined fixture metadata fields for fixture key, builder name, matrix
  family, dimensions, sparsity, shape, rank model, expected rank,
  conditioning, singular-value shape, R-diagonal shape, RHS construction,
  residual target, reconstruction target, orthogonality target,
  expected-failure/skip behavior, reference boundary, and non-claim notes.
- Defined shared fixture classes covering exact spectra, thresholded rank,
  repeated spectra, outer-product low rank, duplicate columns, dependent
  rows, near dependence, tall compatible/incompatible least-squares, wide
  minimum-norm, bidiagonal explicit, Hilbert-like dense, tridiagonal/banded,
  SuiteSparse smoke, mode equivalence, and API error cases.
- Defined SVD-specific classes for exact diagonal spectra, thresholded rank,
  duplicate-column rank deficiency, low-rank outer-product fixtures, partial
  SVD internal-reference comparisons, Moore-Penrose pseudoinverse identities,
  and condition-number behavior.
- Defined QR-specific classes for square exact solves, overdetermined
  compatible/incompatible least-squares, underdetermined minimum-norm,
  duplicate-column and dependent-row rank deficiency, near-rank-deficient
  thresholds, economy mode, sparse mode, and reordering.
- Defined expected-failure classes for null inputs, invalid rank/k values,
  invalid tolerances, factored/permuted matrix rejection, optional
  SuiteSparse skips, unsupported partial-SVD full-vector mode, and expected
  incompatible least-squares residuals.
- Recorded helper placement guidance so matrix builders may be shared only
  when solver-neutral, while SVD/QR tolerance, residual, reconstruction,
  orthogonality, rank, minimum-norm, and non-claim semantics remain visible
  at scenario boundaries.
- Added Day 4 matrix taxonomy design artifact:
  `artifacts/day4-matrix-taxonomy-design.md`.
- Kept Day 4 documentation-only; no C source, header, build, workflow,
  package, benchmark, or test surfaces were modified.

## Day 5 Notes

- Converted Day 2 SVD helper candidates, Day 3 QR helper candidates, and Day 4
  taxonomy boundaries into a concrete extraction plan for Days 6-7.
- Selected a header-only implementation strategy so existing test executable
  ownership stays stable: `test_svd`, `test_qr`, `test_qr_solve`, and
  optional `test_colamd` remain the proof owners.
- Planned `tests/test_svd_helpers.h` for Day 6 SVD measurement and fixture
  helpers, with `tests/test_svd.c` keeping scenario assertions, tolerances,
  skip policies, and non-claim wording.
- Planned `tests/test_qr_helpers.h` for Day 7 QR reconstruction, residual,
  generated-RHS, and deterministic fixture helpers, with `tests/test_qr.c`
  and `tests/test_qr_solve.c` keeping rank, residual, least-squares,
  economy/sparse-mode, and cross-solver assertion semantics.
- Deferred partial-SVD vector helper extraction to Day 10 because its looser
  residual and internal-reference semantics should be expanded with partial
  SVD evidence rather than folded into full-SVD helpers.
- Deferred minimum-norm helper movement out of `tests/test_colamd.c` unless
  Day 9 needs it, because the current reviewed ownership is historically
  tied to COLAMD/reordering tests.
- Recorded focused validation commands for `test_svd`, `test_qr`,
  `test_qr_solve`, optional `test_colamd`, source-list/CMake impact, and the
  required `make format && make lint && make test` chain for future `.c` or
  `.h` edits.
- Recorded rollback instructions for SVD helper extraction, QR helper
  extraction, hidden tolerance semantics, unexpected build membership changes,
  and unclear minimum-norm ownership.
- Added Day 5 helper extraction plan artifact:
  `artifacts/day5-helper-extraction-plan.md`.
- Kept Day 5 documentation-only; no C source, header, build, workflow,
  package, benchmark, or test surfaces were modified.

## Day 6 Notes

- Implemented the first bounded SVD helper extraction batch in
  `tests/test_svd_helpers.h`.
- Moved reusable SVD fixture builders for diagonal, rank-1 row progression,
  rank-deficient duplicate-column, and deterministic full-UV fixtures out of
  `tests/test_svd.c`.
- Moved reusable SVD measurement helpers for dense-column orthogonality, Vt
  row orthogonality, max-entry reconstruction, relative Frobenius
  reconstruction, first Moore-Penrose pseudoinverse identity, dense low-rank
  Frobenius error, sparse-vs-dense low-rank diffs, and sparse-vs-sparse
  relative Frobenius diffs into the helper header.
- Kept scenario tolerances, assertions, skip behavior, expected-failure
  semantics, condition-number assertion helpers, and Golub-Kahan validation
  ownership in `tests/test_svd.c`.
- Did not change Makefile, CMake, CTest registration, workflow, package,
  benchmark, public API, or production source surfaces.
- Ran focused SVD validation with `make build/test_svd && ./build/test_svd`;
  it passed with 98 tests, 0 failures, 0 skips, and 1580 assertions.
- Ran the required `make format && make lint && make test` chain after the
  C/header edits; it passed.
- Ran `git diff --check` and a focused trailing-whitespace scan over
  `docs/planning/EPIC_11/SPRINT_121`, `tests/test_svd.c`, and
  `tests/test_svd_helpers.h`; both passed.
- Added Day 6 implementation artifact:
  `artifacts/day6-svd-helper-extraction.md`.
- Deferred partial-SVD vector helper extraction to Day 10, QR helper
  extraction to Day 7, minimum-norm helper movement to Day 9 or explicit
  deferral, and GK/bidiagonal helper extraction to future maintainability work.

## Day 7 Notes

- Implemented the first bounded QR helper extraction batch in
  `tests/test_qr_helpers.h`.
- Moved shared QR generated-RHS, checked allocation-size, insert-or-free,
  duplicate-column, near-duplicate, small-banded, tall diagonal-dominant,
  reconstruction max-error, and relative residual helpers out of
  `tests/test_qr.c` and `tests/test_qr_solve.c`.
- Kept QR reconstruction assertion wrappers, QR solve true-residual assertion
  wrappers, rank expectations, least-squares residual interpretation,
  economy/sparse-mode comparisons, QR-vs-LU comparisons, and non-claim
  semantics in the scenario tests.
- Did not change Makefile, CMake, CTest registration, workflow, package,
  benchmark, public API, or production source surfaces.
- Ran focused QR validation with
  `make build/test_qr build/test_qr_solve && ./build/test_qr && ./build/test_qr_solve`;
  `test_qr` passed with 63 tests, 0 failures, 0 skips, and 576 assertions,
  and `test_qr_solve` passed with 10 tests, 0 failures, 0 skips, and 972
  assertions.
- Ran the required `make format && make lint && make test` chain after the
  C/header edits; it passed.
- Ran `git diff --check` and a focused trailing-whitespace scan over
  `docs/planning/EPIC_11/SPRINT_121`, `tests/test_qr.c`,
  `tests/test_qr_solve.c`, and `tests/test_qr_helpers.h`; both passed.
- Added Day 7 implementation artifact:
  `artifacts/day7-qr-helper-extraction.md`.
- Deferred minimum-norm helper movement to Day 9 or explicit closeout
  deferral, new rank-deficient fixture expansion to Day 8, new least-squares
  fixture expansion to Day 9, and assertion-wrapper extraction out of Sprint
  121 scope.

## Day 8 Notes

- Expanded deterministic rank-deficient fixture coverage for QR and SVD.
- Added QR helper fixtures for exact dependent-row rank deficiency and
  diagonal threshold-rank evidence.
- Added SVD helper fixture coverage for the same dependent-row pattern and
  reused the existing SVD diagonal builder for explicit threshold-rank
  evidence.
- Added QR tests for exact rank 2, null-space dimension 1, null-space residual
  below `1e-10`, reconstruction below `1e-10`, and explicit diagonal rank
  cutoffs at `1e-14`, `1e-10`, and `1e-6`.
- Added SVD tests for matching diagonal rank cutoffs and SVD-vs-QR agreement
  on the dependent-row rank fixture.
- Did not change Makefile, CMake, CTest registration, workflow, package,
  benchmark, public API, or production source surfaces.
- Ran focused rank validation with
  `make build/test_qr build/test_svd && ./build/test_qr && ./build/test_svd`;
  `test_qr` passed with 65 tests, 0 failures, 0 skips, and 603 assertions,
  and `test_svd` passed with 100 tests, 0 failures, 0 skips, and 1605
  assertions.
- Ran the required `make format && make lint && make test` chain after the
  C/header edits; it passed.
- Ran `git diff --check` and a focused trailing-whitespace scan over
  `docs/planning/EPIC_11/SPRINT_121`, `tests/test_qr.c`,
  `tests/test_svd.c`, `tests/test_qr_helpers.h`, and
  `tests/test_svd_helpers.h`; both passed.
- Added Day 8 implementation artifact:
  `artifacts/day8-rank-deficient-fixture-expansion.md`.
- Deferred compatible/incompatible/minimum-norm least-squares fixture expansion
  to Day 9, partial-SVD rank/vector fixture decisions to Day 10, and dense
  reference comparison lanes to Days 11-12.

## Day 9 Notes

- Expanded least-squares and pseudoinverse proof coverage with deterministic
  compatible, incompatible, and underdetermined fixtures.
- Added a compatible 4x2 overdetermined QR solve case that checks the known
  solution, reported residual, and independent true residual helper.
- Added an incompatible 4x2 overdetermined QR solve case whose residual vector
  is orthogonal to the column space, pinning the least-squares solution and
  residual norm `sqrt(3)`.
- Added a 2x4 underdetermined QR minimum-norm case that checks the known
  solution `{0.5, 0.5, 0.5, 0.5}`, exact row constraints, and solution norm.
- Added a matching SVD pseudoinverse case that checks `A*A^+*A ~= A`,
  `A^+*b`, exact row constraints, and solution norm for the same 2x4 fixture.
- Recorded non-claims for external-library parity and broad numerical
  optimality; the new assertions remain fixture-local.
- Did not change Makefile, CMake, CTest registration, workflow, package,
  benchmark, public API, or production source surfaces.
- Ran focused validation with
  `make build/test_qr_solve build/test_svd && ./build/test_qr_solve && ./build/test_svd`;
  `test_qr_solve` passed with 13 tests, 0 failures, 0 skips, and 1014
  assertions, and `test_svd` passed with 101 tests, 0 failures, 0 skips, and
  1616 assertions.
- Added Day 9 implementation artifact:
  `artifacts/day9-ls-pinv-expansion.md`.
- Deferred low-rank and partial-SVD proof expansion to Day 10, dense-reference
  comparison lanes to Days 11-12, and broader minimum-norm helper extraction
  unless a later sprint moves those ownership tests out of `tests/test_colamd.c`.

## Day 10 Notes

- Expanded bounded low-rank and partial-SVD evidence with deterministic
  rectangular fixtures.
- Added a 6x4 diagonal rectangular partial-SVD vector test that checks the
  retained singular values `{9, 6}`, `A*v ~= sigma*u` residual below `1e-10`,
  and rank-2 reconstruction error `sqrt(10)` from the omitted spectrum
  `{3, 1}`.
- Added a 5x7 rectangular low-rank dense/sparse consistency test that checks
  dense rank-3 reconstruction error `1.0`, zero-drop sparse-vs-dense
  Frobenius difference `0.0`, output dimensions, and retained/omitted
  diagonal entries.
- Kept the new assertions fixture-local and recorded non-claims for external
  parity and broad numerical optimality.
- Did not change Makefile, CMake, CTest registration, workflow, package,
  benchmark, public API, or production source surfaces.
- Ran focused validation with `make build/test_svd && ./build/test_svd`;
  `test_svd` passed with 103 tests, 0 failures, 0 skips, and 1659 assertions.
- Ran the required full quality gate `make format && make lint && make test`;
  it passed.
- Added Day 10 implementation artifact:
  `artifacts/day10-lowrank-partial-svd-expansion.md`.
- Deferred dense-reference or external comparison lane design to Day 11 and
  broader partial-SVD helper extraction unless future work needs shared helper
  ownership.

## Day 11 Notes

- Designed one bounded external dense-reference comparison lane for Sprint 121
  Day 12 implementation.
- Selected a full-SVD singular-value pilot over a deterministic 6x4 dense
  rectangular full-column-rank fixture with mixed signs and non-diagonal
  structure.
- Chose a pure-Python standard-library reference helper that computes
  eigenvalues of `A^T A` with a bounded symmetric Jacobi routine and returns
  sorted singular values; the design explicitly avoids NumPy, SciPy, LAPACK,
  BLAS, and external package dependencies.
- Planned to keep the pilot inside existing `tests/test_svd.c` ownership so
  Day 12 does not change Makefile, CMake, CTest registration, workflow,
  package, benchmark, public API, or production source surfaces unless the
  implementation needs to deviate from the design.
- Planned to reuse `tf_read_external_reference_vector` from
  `tests/test_solver_helpers.h` with the existing external-reference behavior:
  explicit Windows skip, missing `python3` skip through the helper, and helper
  `ERROR` output as test failure.
- Set the Day 12 acceptance model to compare the four full-SVD singular values
  with max absolute difference below `1e-8`, without checking singular vectors,
  subspace bases, partial-SVD behavior, low-rank optimality, QR behavior,
  performance, platform support, or broad external-library parity.
- Recorded Day 12 focused validation requirements:
  `make format`, `make build/test_svd && ./build/test_svd`, `make lint`,
  `make test`, `git diff --check`, and a focused trailing-whitespace scan over
  Sprint 121 docs, `tests/test_svd.c`, and
  `tests/svd_external_dense_reference.py`.
- Added Day 11 design artifact:
  `artifacts/day11-reference-pilot-design.md`.
- Kept Day 11 documentation-only; no C source, header, build, workflow,
  package, benchmark, or test surfaces were modified.

## Day 12 Notes

- Implemented the Day 11 bounded SVD external dense-reference pilot.
- Added `tests/svd_external_dense_reference.py`, a pure-Python
  standard-library helper that builds `svd_rect_fullrank_6x4`, computes `A^T A`,
  diagonalizes the small symmetric Gram matrix with a bounded Jacobi routine,
  and emits sorted singular values through the existing external-reference
  output protocol.
- Updated `tests/test_svd.c` to enable `tf_read_external_reference_vector`,
  build the 6x4 fixture locally, read the external singular values, run
  `sparse_svd_compute(A, NULL, &svd)`, and compare the four singular values
  with max absolute difference below `1e-8`.
- Registered `test_svd_external_dense_reference_rect_fullrank_6x4` in the
  existing `test_svd` executable, so Makefile, CMake, CTest registration,
  workflow, package, benchmark, public API, and production source surfaces
  were not changed.
- Preserved the Day 11 trust boundary: the helper does not use NumPy, SciPy,
  LAPACK, BLAS, SuiteSparse, or external packages; Windows skips explicitly;
  missing `python3` skips through the existing helper; helper `ERROR` output
  is a test failure.
- Kept singular-vector, subspace, partial-SVD, low-rank, QR, SuiteSparse,
  performance, platform, ABI, package, and state-of-the-art claims out of the
  pilot.
- Ran `python3 tests/svd_external_dense_reference.py svd_rect_fullrank_6x4`; it
  emitted 4 singular values.
- Ran focused validation with `make format && make build/test_svd &&
  ./build/test_svd`; `test_svd` passed with 104 tests, 0 failures, 0 skips,
  and 1685 assertions. The new external-reference check reported max
  `|sigma-sigma_ref| = 6.217e-15`.
- Ran the remaining required quality gate with `make lint && make test`; both
  passed.
- Added Day 12 implementation artifact:
  `artifacts/day12-reference-pilot-implementation.md`.
- Deferred any additional SVD external fixtures, QR external-reference lanes,
  and partial-SVD external parity to future oracle-owner work unless Sprint 121
  closeout explicitly carries them forward.

## Day 13 Notes

- Packaged Sprint 121 validation evidence and trust-boundary guidance for the
  QR, QR solve, rank-deficient, least-squares, pseudoinverse, low-rank,
  partial-SVD, and SVD external-reference proof-owner work.
- Ran focused validation with `python3 tests/svd_external_dense_reference.py
  svd_rect_fullrank_6x4 && make build/test_qr build/test_qr_solve build/test_svd
  && ./build/test_qr && ./build/test_qr_solve && ./build/test_svd`.
- The Python SVD reference helper passed and emitted 4 singular values.
- `test_qr` passed with 65 tests, 0 failures, 0 skips, and 603 assertions.
- `test_qr_solve` passed with 13 tests, 0 failures, 0 skips, and 1014
  assertions.
- `test_svd` passed with 104 tests, 0 failures, 0 skips, and 1685 assertions;
  the external dense-reference pilot reported max `|sigma-sigma_ref| =
  6.217e-15`.
- Ran the required full quality gate `make format && make lint && make test`;
  it passed.
- Did not change Makefile, CMake, CTest registration, workflow, package,
  benchmark, public API, production source, README, solver-selection, or install
  documentation surfaces on Day 13.
- Recorded explicit non-claims for LAPACK/SciPy/NumPy/SuiteSparse/PETSc/
  Trilinos/Eigen parity, singular-vector/subspace parity, partial-SVD external
  parity, QR external parity, global low-rank or pseudoinverse optimality,
  performance, scalability, package, platform, ABI, and state-of-the-art
  behavior.
- Added Day 13 validation artifact:
  `artifacts/day13-validation-package.md`.
- Deferred additional SVD external fixtures, a QR external dense-reference lane,
  partial-SVD external parity, and public solver-selection wording changes to
  future owners unless Sprint 121 closeout carries them forward.

## Day 14 Notes

- Reviewed all Sprint 121 artifacts, working notes, code changes, validation
  outputs, and residual queues.
- Marked all seven Sprint 121 project-plan items complete: evidence audit,
  matrix taxonomy, SVD helper extraction, QR/least-squares expansion,
  external/dense reference pilot, validation, and docs/non-claims.
- Accounted for all sprint deliverables: fixture taxonomy, reusable SVD/QR
  proof helpers, expanded rank-deficient/least-squares/pseudoinverse/low-rank/
  partial-SVD evidence, the bounded SVD external-reference pilot, and
  trust-boundary documentation.
- Confirmed no Makefile, CMake, CTest registration, source-list, workflow,
  package, benchmark, public API, or production source membership changed
  during Sprint 121.
- Preserved Day 13 validation as the closeout quality baseline:
  `make format && make lint && make test` passed after the C/header changes.
- Kept Day 14 documentation-only; no C quality gate was required for the Day
  14 edits.
- Recorded residual queue entries for additional SVD external fixtures, QR
  external dense-reference design, partial-SVD external parity, minimum-norm
  helper ownership migration, Bidiagonal/Golub-Kahan helper extraction, and
  public solver-selection wording after broader evidence lands.
- Added Day 14 closeout artifact:
  `artifacts/day14-sprint-closeout.md`.
- Sprint 121 can close without unresolved validation or non-claim ambiguity.
