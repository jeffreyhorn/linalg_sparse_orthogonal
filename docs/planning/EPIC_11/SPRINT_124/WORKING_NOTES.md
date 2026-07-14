# Sprint 124 Working Notes

## Sprint Goal

Convert Sprint 123's residual QR, partial-SVD, minimum-norm, and
Bidiagonal/Golub-Kahan deferred debt into bounded oracle decisions or explicit
future-owner packages before corpus, reporting, performance, package, and
adoption work consume the oracle truth.

## Starting Constraints

- Treat Sprint 123's residual deferred debt, dependency ordering, and
  non-claim register as the source of truth for Sprint 124 scope.
- Do not reopen completed Sprint 121-123 oracle lanes:
  `svd_rect_fullrank_6x4`, `svd_rankdef_duplicate_5x4`,
  `svd_wide_fullrank_4x6`, `qr_overdetermined_incompatible_4x2`,
  `qr_overdetermined_compatible_5x3`, `partial_svd_diag6_k2`, or
  `partial_svd_tall5x3_k2`.
- Keep QR rank-deficient, QR minimum-norm, QR basis/economy, partial-SVD
  vector/subspace, partial-SVD residual, and helper ownership decisions
  separate because their evidence semantics differ.
- Preserve QR minimum-norm ownership across QR solve, COLAMD,
  SVD-pseudoinverse, fallback, refinement, and optional SuiteSparse scenarios
  unless a future helper movement proves behavior-specific ownership remains
  visible.
- Preserve Bidiagonal/Golub-Kahan helper ownership for wide transpose,
  implicit Householder reconstruction, explicit `U`/`V` reconstruction, wide
  Golub-Kahan skips, and bidiagonal QR iteration semantics unless a dedicated
  helper owner can carry those meanings explicitly.
- Refresh maintainer evidence and solver-selection wording only when the
  evidence supports the claim; otherwise publish an explicit no-update
  rationale.
- Preserve non-claims around broad LAPACK, NumPy, SciPy, SuiteSparse, PETSc,
  Trilinos, Eigen, ARPACK, vendor-backend, dense-library, singular-vector,
  Q-basis, subspace, rank-deficient QR, QR minimum-norm, partial-SVD
  convergence/vector/subspace, low-rank, global optimality, package, ABI,
  platform, performance, scalability, public API, and state-of-the-art parity.
- If any `.c` or `.h` file changes, run `make format && make lint && make test`
  before closeout. Documentation-only changes require `git diff --check` and
  focused whitespace validation.

## Input Artifact Inventory

| Input | Role in Sprint 124 |
| --- | --- |
| `docs/planning/EPIC_11/PROJECT_PLAN.md` Sprint 124 | Defines residual QR, partial-SVD, helper ownership, validation, maintainer-evidence, and claim-gate items. |
| `docs/planning/EPIC_11/SPRINT_124/PLAN.md` | Provides day-level execution order and 166-hour budget. |
| `docs/planning/EPIC_11/SPRINT_123/RETROSPECTIVE.md` | Defines residual deferred debt, non-claims, and completed-work fences. |
| `docs/planning/EPIC_11/SPRINT_123/WORKING_NOTES.md` | Captures bounded SVD, QR, partial-SVD, helper, maintainer-evidence, and solver-selection decisions. |
| Sprint 123 Day 5-8 artifacts | Source for QR compatible/rank-deficient/minimum-norm/Q/economy decisions and completed QR compatible implementation. |
| Sprint 123 Day 9-10 artifacts | Source for partial-SVD top-k implementation and deferred vector/subspace/residual semantics. |
| Sprint 123 Day 11-12 artifacts | Source for minimum-norm and Bidiagonal/Golub-Kahan helper movement deferrals and promotion gates. |
| Sprint 123 Day 13-14 artifacts | Source for maintainer evidence, claim-gate outcome, validation, residual queue, and non-claims. |
| Sprint 122 artifacts | Source for earlier SVD, QR, partial-SVD, helper, and claim-gate ownership boundaries. |
| Sprint 121 artifacts | Source taxonomy for SVD/QR/rank fixtures, matrix families, helper ownership, and oracle evidence classes. |

## Day-Level Ownership

| Day | Owner Focus | Project-Plan Items |
| --- | --- | --- |
| 1 | Sprint intake, residual dependency map, duplicate fence, validation boundary | Items 1-7 |
| 2 | Rank-deficient QR policy design | Item 1 |
| 3 | Rank-deficient QR decision or bounded batch | Item 1 |
| 4 | QR minimum-norm behavior contract | Item 2 |
| 5 | QR minimum-norm decision or bounded batch | Item 2 |
| 6 | QR Q-basis and economy semantics | Item 3 |
| 7 | QR Q-basis/economy decision package | Item 3 |
| 8 | Partial-SVD vector and subspace semantics | Item 4 |
| 9 | Partial-SVD vector/subspace decision or bounded batch | Item 4 |
| 10 | Partial-SVD residual scenario matrix | Item 5 |
| 11 | Partial-SVD residual implementation or deferral package | Item 5 |
| 12 | Minimum-norm and Bidiagonal/Golub-Kahan helper ownership follow-through | Item 6 |
| 13 | Validation, maintainer evidence, and claim gate | Item 7 |
| 14 | Sprint closeout and downstream handoff | Items 1-7 |

## Validation Expectations

| Change Type | Required Validation |
| --- | --- |
| Documentation only | `git diff --check` and focused trailing-whitespace scan of Sprint 124 files. |
| `.c` or `.h` edits | `make format && make lint && make test`. |
| Script or generated helper edits | Focused syntax check plus affected behavior check. |
| Makefile, CMake, or CTest membership edits | Source-list inspection and relevant CMake/CTest proof, including platform count impact if test membership changes. |
| External-reference fixture edits | Focused helper invocation, affected test executable, skip-path proof, and failure interpretation note. |
| Maintainer or public documentation wording edits | Evidence-to-claim traceability, claim-boundary scan, link/path hygiene, and explicit non-claim update. |

## Scope Boundaries

- Sprint 124 may add bounded external oracle evidence only after each lane has
  explicit trust, tolerance, skip, rank, basis, vector/subspace, residual, and
  failure semantics.
- Sprint 124 may explicitly defer work when the future owner, dependency, and
  promotion gate are recorded.
- Sprint 124 must not treat deterministic internal fixtures or already
  completed bounded external lanes as broad external parity.
- Sprint 124 must not use helper extraction to hide scenario-specific QR
  minimum-norm, COLAMD, SVD-pseudoinverse, fallback, refinement, SuiteSparse,
  Bidiagonal, Golub-Kahan, transpose, reconstruction, or iteration semantics.
- Sprint 124 must not expand solver-selection wording unless the evidence
  supports user-facing claims and preserves the non-claim register.

## Day 1 Notes

- Created the Sprint 124 artifact directory.
- Established the working-notes baseline.
- Mapped Sprint 123 residual deferred debt to Sprint 124 day-level owners.
- Recorded duplicate fences for completed Sprint 121-123 oracle lanes and
  helper decisions.
- Set validation expectations for documentation, code, scripts, build
  metadata, external oracle lanes, maintainer evidence, and public wording
  changes.

## Day 2 Notes

- Inventoried current QR rank-deficient, nullspace, diagonal-threshold,
  rank-deficient solve, minimum-norm, and SVD-pseudoinverse evidence owners.
- Kept rank-only, residual-only, nullspace/subspace, and minimum-norm external
  candidates separate because each has different trust and claim boundaries.
- Defined the rank-threshold policy: Sprint 124 must not introduce a global
  numerical-rank threshold; every accepted fixture must pin its own threshold
  and expected rank.
- Defined nullspace policy: nullity can follow a pinned rank, but raw
  nullspace basis equality is not acceptable without sign, ordering, and
  subspace/projection semantics.
- Assigned minimum-norm and pseudoinverse evidence to the Days 4-5
  behavior-specific owner rather than Day 2 rank policy.
- Established Day 3 decision criteria that prefer a structural rank-only
  fixture if accepted, defer minimum-norm/pseudoinverse work, and require
  future-owner gates for any deferral.
- Day 2 changed documentation only; validation used `git diff --check` and a
  focused trailing-whitespace scan over Sprint 124 files.

## Day 3 Notes

- Accepted and implemented one bounded rank-only external QR fixture:
  `qr_rankdef_duplicate_5x4_rank_only`.
- Extended `tests/qr_external_dense_reference.py` with a standard-library
  rank routine and a rank fixture dispatcher that emits `OK 1` followed by the
  expected rank.
- Extended `tests/test_qr_solve.c` with the matching sparse 5x4
  duplicate-column fixture and rank comparison against `sparse_qr_rank(&qr,
  0.0)` and `qr.rank`.
- Updated `docs/maintainer_guide.md` so the QR evidence table names the new
  bounded rank-only fixture while preserving rank-deficient solve, nullspace,
  minimum-norm, Q-basis, economy, sparse-mode, reorder, and broad parity
  non-claims.
- Deferred residual-only rank-deficient QR evidence, nullspace external
  evidence, rank-deficient minimum-norm evidence, near-rank-deficient threshold
  evidence, and SuiteSparse rank-deficient QR evidence with future owners and
  promotion gates.
- Focused helper validation emitted `OK 1` and `3`; focused
  `test_qr_solve` passed with 16 tests, 0 failures, 0 skips, and 1060
  assertions.
- Full required code gate passed: `make format && make lint && make test`.

## Day 4 Notes

- Inventoried current QR minimum-norm evidence across `tests/test_qr_solve.c`,
  `tests/test_colamd.c`, and `tests/test_svd.c`.
- Defined a behavior matrix separating focused QR solve, COLAMD/reorder,
  fallback, rank-deficient, refinement, zero-row, QR-vs-SVD-pseudoinverse,
  SVD pseudoinverse, and optional SuiteSparse submatrix coverage.
- Defined residual, solution, norm, rank, tolerance, and failure-diagnostic
  policy for future minimum-norm external evidence.
- Kept SVD pseudoinverse as a bounded cross-check when explicitly named, not a
  global QR oracle.
- Preserved helper ownership boundaries: no generic `assert_minnorm` helper or
  generic external minimum-norm fixture should hide behavior-specific
  ownership.
- Defined Day 5 decision criteria that prefer a tiny exact underdetermined
  fixture only if expected solution, residual, and norm can be produced without
  dense-library dependencies.
- Day 4 changed documentation only; validation used `git diff --check` and a
  focused trailing-whitespace scan over Sprint 124 files.

## Day 5 Notes

- Accepted and implemented one bounded exact underdetermined minimum-norm QR
  external fixture: `qr_underdetermined_minnorm_2x4`.
- Extended `tests/qr_external_dense_reference.py` with a standard-library
  minimum-norm reference path that emits four solution values, residual norm,
  and solution norm.
- Extended `tests/test_qr_solve.c` with the matching sparse 2x4 fixture and
  explicit solution/residual/norm comparisons against the helper output.
- Updated `docs/maintainer_guide.md` so the QR evidence table names the new
  bounded exact minimum-norm fixture while preserving broad minimum-norm
  non-claims.
- Deferred COLAMD/reordered, fallback, rank-deficient, refinement,
  QR-vs-SVD-pseudoinverse, and SuiteSparse minimum-norm external evidence with
  future owners and promotion gates.
- Focused helper validation emitted `OK 6`, four `0.5` solution values,
  residual `0`, and norm `1`; focused `test_qr_solve` passed with 17 tests, 0
  failures, 0 skips, and 1069 assertions.
- Full required code gate passed: `make format && make lint && make test`.

## Day 6 Notes

- Inventoried current QR Q-basis, Q application, Q orthogonality, economy,
  sparse-mode, wide, square, rank-deficient, and SuiteSparse-smoke evidence in
  `tests/test_qr.c`.
- Defined sign and orientation policy for any future raw Q-column comparison:
  sign normalization, column order, formed-Q layout, degeneracy limits, and
  failure classification must be explicit before implementation.
- Preferred projection, orthogonality, reconstruction, projector-distance, or
  principal-angle metrics over raw basis equality for basis-dependent external
  evidence.
- Defined economy-shape expectations for tall full-rank, tall rank-deficient,
  square, wide, and singleton matrices so Day 7 can classify any accepted lane
  as shape-only, orthogonality, projection, subspace, or raw-basis evidence.
- Preserved owner boundaries: `tests/test_qr.c` remains the Q/economy owner,
  `tests/test_qr_solve.c` remains solve-oriented, and
  `tests/qr_external_dense_reference.py` should not overload least-squares
  output with basis/projection protocol without a Day 7 decision.
- Day 6 changed documentation only; validation used `git diff --check` and a
  focused trailing-whitespace scan over Sprint 124 files.

## Day 7 Notes

- Accepted and implemented one bounded QR economy projector external fixture:
  `qr_economy_projector_5x3`.
- Extended `tests/qr_external_dense_reference.py` with a standard-library
  projector reference path that emits `q_rows`, `q_cols`, `r_rows`, `r_cols`,
  and the dense projector `A (A^T A)^{-1} A^T`.
- Extended `tests/test_qr.c` with the matching sparse 5x3 fixture,
  economy-mode QR factorization, thin-Q formation, `Q Q^T` projector
  comparison, orthogonality comparison, and existing-suite registration.
- Updated `docs/maintainer_guide.md` so the QR evidence table names the new
  bounded economy projector fixture while preserving raw Q-basis,
  Q-sign/orientation, broad economy, sparse-mode, reorder, backend, and
  performance non-claims.
- Deferred raw Q-column, rank-deficient subspace/nullspace, wide economy,
  sparse-mode, and SuiteSparse Q/economy external evidence with future owners
  and promotion gates.
- Focused helper validation emitted `OK 29` with shape values `5`, `3`, `3`,
  `3`; focused `test_qr` passed with 66 tests, 0 failures, 0 skips, and 628
  assertions.
- Full required code gate passed: `make format && make lint && make test`.

## Day 8 Notes

- Inventoried current partial-SVD evidence across external top-k singular
  values, internal value checks, vector availability, orthogonality,
  singular-triplet residuals, rectangular behavior, rank-deficient behavior,
  low-rank reconstruction, and convergence/timing smoke.
- Kept top-k singular-value evidence separate from vector, subspace,
  convergence-budget, rank-deficient, corpus, and low-rank evidence.
- Defined sign-invariant vector policy: raw vector equality is not an external
  pass/fail metric, sign mismatch alone is never a failure, and residuals plus
  orthogonality are the preferred vector evidence.
- Defined projector and principal-angle policy for repeated, clustered,
  rank-deficient, and other basis-ambiguous subspace evidence.
- Defined residual, tolerance, skip, helper-error, Windows-skip, and
  failure-interpretation rules before any Day 9 implementation.
- Identified `partial_svd_vector_residual_diag6_k2` as the lowest-risk Day 9
  candidate if it remains residual-only and preserves sign-invariant
  semantics.
- Day 8 changed documentation only; validation used `git diff --check` and a
  focused trailing-whitespace scan over Sprint 124 files.

## Day 9 Notes

- Accepted one bounded partial-SVD vector-residual lane:
  `partial_svd_vector_residual_diag6_k2`.
- Reused the existing `partial_svd_diag6_k2` external dense-reference helper
  output for singular-value anchoring and kept vector checks in the product
  test owner.
- Added sign-invariant residual metrics for `A v_i - sigma_i u_i` and
  `A^T u_i - sigma_i v_i`, plus `U` and `V` orthogonality checks.
- Avoided raw vector equality and did not introduce subspace, repeated,
  clustered, rank-deficient, convergence-budget, corpus, or low-rank
  optimality claims.
- Updated the maintainer guide to name the bounded partial-SVD
  vector-residual fixture while preserving the broad vector/subspace
  non-claim.
- Deferred repeated-spectrum, clustered-spectrum, rank-deficient subspace,
  rectangular vector-residual, SuiteSparse corpus, low-rank optimality, and
  convergence-budget lanes with future owners and promotion gates.

## Day 10 Notes

- Inventoried remaining partial-SVD residual scenarios after the accepted Day
  9 vector-residual lane: repeated-spectrum subspace, clustered-spectrum
  subspace/convergence, rank-deficient threshold/subspace, rectangular vector
  residuals, SuiteSparse corpus residuals, low-rank optimality, convergence
  budgets, and nonsymmetric rectangular value residuals.
- Deferred all new Day 10 residual implementations because each remaining
  scenario needs metric or ownership semantics beyond the exact square
  diagonal residual lane.
- Defined scenario-specific trust boundaries, required diagnostics, tolerance
  policies, skip policies, and failure-interpretation classes.
- Set Day 11's default handoff to a deferral package unless one exact
  rectangular vector-residual lane can be narrowed without changing helper
  protocol or broadening maintainer claims.
- Preserved broad partial-SVD parity, repeated/clustered correctness,
  rank-deficient subspace, corpus parity, low-rank optimality, convergence,
  platform, performance, and state-of-the-art non-claims.
- Day 10 changed documentation only; validation used `git diff --check` and a
  focused trailing-whitespace scan over Sprint 124 files.

## Day 11 Notes

- Published the partial-SVD residual deferral package because Day 10 accepted
  no new immediate residual scenario beyond the Day 9 exact square diagonal
  vector-residual lane.
- Kept `tests/svd_external_dense_reference.py`, `tests/test_svd.c`,
  `tests/test_svd_partial_helpers.h`, `docs/maintainer_guide.md`, build
  metadata, package metadata, and public examples unchanged.
- Documented why rectangular vector residual, repeated-spectrum subspace,
  clustered-spectrum subspace/convergence, rank-deficient subspace,
  SuiteSparse corpus residual, low-rank optimality, convergence-budget, and
  nonsymmetric rectangular residual lanes remain deferred.
- Added future-owner promotion gates for every deferred residual scenario.
- Confirmed Day 11 does not alter package, ABI, platform, performance, broad
  solver-family, or state-of-the-art claims.
- Day 11 changed documentation only; validation used `git diff --check` and a
  focused trailing-whitespace scan over Sprint 124 files.

## Day 12 Notes

- Revisited Sprint 123's minimum-norm helper migration decision and
  Bidiagonal/Golub-Kahan helper extraction decision after the Sprint 124 QR
  rank, minimum-norm, economy-projector, and partial-SVD vector-residual
  evidence additions.
- Deferred all minimum-norm helper movement because generic helpers would hide
  QR solve, COLAMD, SVD pseudoinverse, fallback, refinement, rank-deficient,
  zero-row, and SuiteSparse scenario ownership.
- Deferred all Bidiagonal/Golub-Kahan helper extraction because current
  helpers own distinct implicit Householder reconstruction, wide transpose,
  explicit `U`/`V`, wide GK skip, and bidiagonal QR iteration semantics.
- Recorded behavior-specific helper names that a future owner may use and
  generic helper names to avoid.
- Left `docs/maintainer_guide.md` unchanged because no helper moved and no new
  evidence was added.
- Day 12 changed documentation only; validation used `git diff --check` and a
  focused trailing-whitespace scan over Sprint 124 files.

## Day 13 Notes

- Rechecked all accepted Sprint 124 implementation lanes: QR rank-only
  `qr_rankdef_duplicate_5x4_rank_only`, QR minimum-norm
  `qr_underdetermined_minnorm_2x4`, QR economy projector
  `qr_economy_projector_5x3`, and partial-SVD vector-residual
  `partial_svd_vector_residual_diag6_k2`.
- Direct helper validation passed for the QR rank-only, QR minimum-norm, QR
  economy-projector, and partial-SVD diagonal singular-value helper protocols.
- Focused executable validation passed:
  `make build/test_qr_solve && ./build/test_qr_solve` with 17 tests, 0
  failures, 0 skips, and 1069 assertions;
  `make build/test_qr && ./build/test_qr` with 66 tests, 0 failures, 0 skips,
  and 628 assertions; and `make build/test_svd && ./build/test_svd` with 109
  tests, 0 failures, 0 skips, and 1803 assertions.
- Full required code quality validation passed:
  `make format && make lint && make test`; the final test phase ended with
  `All tests passed.`
- Post-gate hygiene passed with `git diff --check` and a focused
  trailing-whitespace scan over Sprint 124 documentation plus touched
  maintainer/test/helper files.
- Confirmed `docs/maintainer_guide.md` already names the accepted Sprint 124
  QR and partial-SVD lanes while preserving fixture-scoped trust boundaries
  and broad non-claims.
- Left `docs/solver_selection.md` unchanged because the accepted evidence does
  not support any broader user-facing solver-selection claim.
- Added the Day 13 validation and claim-gate artifact:
  `artifacts/day13-validation-claim-gate.md`.

## Day 14 Notes

- Reviewed all Sprint 124 artifacts against the seven project-plan items:
  rank-deficient QR, QR minimum-norm, QR Q-basis/economy, partial-SVD
  vector/subspace, partial-SVD residual semantics, helper ownership, and
  validation/claim gate.
- Closed Sprint 124 with four accepted bounded implementation lanes:
  `qr_rankdef_duplicate_5x4_rank_only`,
  `qr_underdetermined_minnorm_2x4`, `qr_economy_projector_5x3`, and
  `partial_svd_vector_residual_diag6_k2`.
- Consolidated deferred QR rank/nullspace/threshold/corpus,
  QR minimum-norm/COLAMD/fallback/refinement/SVD/SuiteSparse,
  QR Q-basis/economy/subspace/sparse-mode/corpus, partial-SVD
  rectangular/repeated/clustered/rank-deficient/corpus/low-rank/convergence,
  and helper-ownership work with future owners and promotion gates.
- Preserved the final non-claim register: no broad external package,
  dense-library, QR, SVD, partial-SVD, helper, package, ABI, platform,
  performance, scalability, public API, or state-of-the-art parity claim.
- Prepared the Sprint 125 corpus/report-index handoff around accepted lane
  owners, external-reference script inventory, skip/failure semantics,
  evidence fields, claim boundaries, and future-owner queue.
- Kept Day 14 documentation-only. No source, header, helper script, build,
  package, public API, public solver-selection, or maintainer-guide wording
  changed on Day 14.
- Day 14 relies on the Day 13 full validation baseline:
  `make format && make lint && make test` passed, with the final test phase
  ending in `All tests passed.`
- Added the Day 14 closeout and handoff artifact:
  `artifacts/day14-closeout-and-handoff.md`.
