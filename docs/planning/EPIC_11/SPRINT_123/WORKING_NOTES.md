# Sprint 123 Working Notes

## Sprint Goal

Promote Sprint 122's residual SVD, QR, partial-SVD, helper, and claim debt
into bounded implementation or explicit deferral packages before corpus,
report-index, performance, package, and adoption sprints consume the oracle
truth.

## Starting Constraints

- Treat Sprint 122's residual deferred debt and non-claim register as the
  source of truth for Sprint 123 scope.
- Do not reopen Sprint 122's completed bounded oracle lanes:
  `svd_rankdef_duplicate_5x4`, `qr_overdetermined_incompatible_4x2`, or
  `partial_svd_diag6_k2`.
- Keep SVD, QR, and partial-SVD external evidence separate because their
  fixture, basis, vector, subspace, convergence, and failure semantics differ.
- Preserve minimum-norm ownership across QR, COLAMD, SVD-pseudoinverse,
  refinement, fallback, and SuiteSparse scenarios unless a future helper
  movement proves behavior-specific ownership remains visible.
- Keep Bidiagonal/Golub-Kahan helper extraction in a dedicated semantic owner;
  do not fold wide-transpose, Householder reconstruction, explicit `U`/`V`,
  or bidiagonal QR iteration checks into generic SVD helpers.
- Refresh maintainer and solver-selection documentation only when the evidence
  supports the claim; otherwise publish an explicit no-update rationale.
- Preserve non-claims around broad LAPACK, NumPy, SciPy, SuiteSparse, PETSc,
  Trilinos, Eigen, ARPACK, vendor-backend, ecosystem, platform, package, ABI,
  performance, scalability, public API, and state-of-the-art parity.
- If any `.c` or `.h` file changes, run `make format && make lint && make test`
  before closeout. Documentation-only changes require `git diff --check` and
  focused whitespace validation.

## Input Artifact Inventory

| Input | Role in Sprint 123 |
| --- | --- |
| `docs/planning/EPIC_11/PROJECT_PLAN.md` Sprint 123 | Defines residual SVD/QR oracle, helper, maintainer evidence, and solver-selection claim items. |
| `docs/planning/EPIC_11/SPRINT_123/PLAN.md` | Provides day-level execution order and 166-hour budget. |
| `docs/planning/EPIC_11/SPRINT_122/RETROSPECTIVE.md` | Defines residual deferred debt, non-claims, and completed-work duplicate fences. |
| `docs/planning/EPIC_11/SPRINT_122/WORKING_NOTES.md` | Captures bounded oracle lane decisions, helper-boundary decisions, and validation context. |
| Sprint 122 Day 3-4 artifacts | Source for additional SVD external fixture criteria and completed rank-deficient fixture lane. |
| Sprint 122 Day 5-8 artifacts | Source for QR external lane decisions and partial-SVD top-k external semantics. |
| Sprint 122 Day 9-10 artifacts | Source for minimum-norm and Bidiagonal/Golub-Kahan helper ownership boundaries. |
| Sprint 122 Day 11-12 artifacts | Source for solver-selection evidence gates and no-update rationale. |
| Sprint 122 Day 13-14 artifacts | Source for validation evidence, residual queue, and non-claim register. |
| Sprint 121 artifacts | Source taxonomy for SVD, QR, rank, least-squares, pseudoinverse, low-rank, helper, and matrix-family evidence classes. |

## Day-Level Ownership

| Day | Owner Focus | Project-Plan Items |
| --- | --- | --- |
| 1 | Sprint intake, residual proof map, duplicate fence, validation boundary | Items 1-7 |
| 2 | SVD fixture taxonomy and external reference trust model | Item 1 |
| 3 | SVD external fixture batch decision | Item 1 |
| 4 | SVD external fixture implementation or explicit deferral package | Item 1 |
| 5 | QR external behavior evidence requirements | Item 2 |
| 6 | QR compatible and rank-deficient evidence decision | Item 2 |
| 7 | QR minimum-norm and Q/economy evidence decision | Item 2 |
| 8 | QR evidence implementation or explicit deferral package | Item 2 |
| 9 | Partial-SVD external semantics design | Item 3 |
| 10 | Partial-SVD evidence decision and implementation or deferral package | Item 3 |
| 11 | Minimum-norm helper migration decision | Item 4 |
| 12 | Bidiagonal/Golub-Kahan helper extraction decision | Item 5 |
| 13 | Maintainer evidence-table refresh | Item 6 |
| 14 | Solver-selection claim gate and closeout | Item 7 |

## Validation Expectations

| Change Type | Required Validation |
| --- | --- |
| Documentation only | `git diff --check` and focused trailing-whitespace scan of Sprint 123 files. |
| `.c` or `.h` edits | `make format && make lint && make test`. |
| Script or generated helper edits | Focused syntax check plus affected behavior check. |
| Makefile, CMake, or CTest membership edits | Source-list inspection and relevant CMake/CTest proof, including platform count impact if test membership changes. |
| External-reference fixture edits | Focused helper invocation, affected test executable, skip-path proof, and failure interpretation note. |
| Public documentation wording edits | Evidence-to-claim traceability, claim-boundary scan, and explicit non-claim update. |

## Scope Boundaries

- Sprint 123 may add bounded external oracle evidence only after each lane has
  explicit fixture, trust, tolerance, skip, basis/vector/subspace, and failure
  semantics.
- Sprint 123 may explicitly defer work when the future owner, dependency, and
  promotion gate are recorded.
- Sprint 123 must not treat deterministic internal fixtures as broad external
  parity.
- Sprint 123 must not use helper extraction to hide scenario-specific
  minimum-norm, Bidiagonal, Golub-Kahan, transpose, reconstruction, or
  iteration semantics.
- Sprint 123 must not expand public solver-selection wording unless the
  evidence supports user-facing claims.

## Day 1 Notes

- Created the Sprint 123 artifact directory.
- Established the working-notes baseline.
- Mapped Sprint 122 residual deferred debt to Sprint 123 day-level proof
  owners.
- Recorded duplicate fences for completed Sprint 122 oracle lanes and helper
  decisions.
- Set validation expectations for documentation, code, scripts, build
  metadata, external oracle lanes, and public wording changes.

## Day 2 Notes

- Inventoried current SVD external fixtures:
  `svd_rect_fullrank_6x4`, `svd_rankdef_duplicate_5x4`, and
  `partial_svd_diag6_k2`.
- Classified deterministic SVD coverage for exact diagonal spectra,
  threshold-rank, exact rank deficiency, tall/wide full SVD, repeated spectra,
  pseudoinverse identities, low-rank output, partial-SVD vectors, SuiteSparse
  smoke fixtures, and error paths.
- Defined the Sprint 123 SVD external-reference trust model: small fixed
  fixtures, Python standard-library-only reference arithmetic, explicit output
  shape, singular values only for Day 3 full-SVD candidates, and no NumPy,
  SciPy, LAPACK, BLAS, package, platform, or broad parity assumptions.
- Identified the strongest Day 3 candidate class as a bounded wide full-rank
  singular-value fixture such as `svd_wide_fullrank_4x6`, provided Day 3 pins
  the `min(m,n)` output contract.
- Marked near-dependent threshold, pseudoinverse-threshold, low-rank
  tail-energy, SuiteSparse, and vector/subspace candidates as higher-risk or
  separate-owner work unless Day 3 explicitly narrows their semantics.
- Day 2 changed documentation only; validation used `git diff --check` and a
  focused trailing-whitespace scan over Sprint 123 files.

## Day 3 Notes

- Accepted one bounded full-SVD external fixture batch for Day 4:
  `svd_wide_fullrank_4x6`.
- Pinned the accepted fixture contract to a 4x6 dense full-row-rank matrix,
  singular-value-only comparison, exactly four emitted reference values, and a
  `1e-8` positive singular-value tolerance.
- Identified the Day 4 affected surfaces as
  `tests/svd_external_dense_reference.py` and `tests/test_svd.c`, with no
  expected Makefile, CMake, CTest, public API, public documentation, or
  partial-SVD helper changes.
- Deferred near-dependent threshold, repeated-spectrum, low-rank tail-energy,
  pseudoinverse-threshold, SuiteSparse, and vector/subspace SVD candidates with
  future owners and promotion gates.
- Preserved non-claims around broad external dense-library parity, vector and
  subspace parity, partial-SVD semantics, pseudoinverse/minimum-norm behavior,
  low-rank optimality, rank-threshold policy, package/platform/ABI claims, and
  performance or state-of-the-art behavior.
- Day 3 changed documentation only; validation used `git diff --check` and a
  focused trailing-whitespace scan over Sprint 123 files.

## Day 4 Notes

- Implemented the accepted `svd_wide_fullrank_4x6` external full-SVD fixture.
- Updated `tests/svd_external_dense_reference.py` with the wide fixture and
  `min(m,n)` singular-value emission so wide fixtures do not expose padded
  `A^T A` zero eigenvalue slots as fixture output.
- Updated `tests/test_svd.c` with the matching sparse fixture builder,
  fixture-key allow-list entry, singular-value-only comparison test, and
  existing-suite registration.
- Confirmed no new Makefile, CMake, CTest, public API, public documentation,
  or partial-SVD helper membership was added.
- Focused helper proof emitted `OK 4` for `svd_wide_fullrank_4x6`; focused
  `test_svd` passed with 107 tests, 0 failures, 0 skips, and 1755 assertions.
- Full required code gate passed: `make format && make lint && make test`.
- `git diff --check` and the focused trailing-whitespace scan over Sprint 123
  docs plus touched SVD files passed.

## Day 5 Notes

- Inventoried current QR external evidence and fenced the completed
  `qr_overdetermined_incompatible_4x2` least-squares lane as already owned by
  Sprint 122.
- Classified deterministic QR coverage for reconstruction, Q application,
  compatible tall least-squares, incompatible tall least-squares,
  rank-deficient/nullspace behavior, underdetermined minimum-norm, economy
  mode, sparse mode, refinement, and reordering/fill.
- Defined Day 6 candidate requirements for compatible tall and rank-deficient
  QR external evidence while preserving Day 7 ownership for minimum-norm and
  Q/economy evidence.
- Rejected square QR solve, SuiteSparse external QR, and sparse-mode external
  parity as Sprint 123 external behavior candidates because they duplicate
  deterministic evidence or risk broad platform/backend claims.
- Recorded tolerance, skip, failure-diagnostic, basis, and ownership rules for
  any accepted QR external fixture.
- Day 5 changed documentation only; validation used `git diff --check` and a
  focused trailing-whitespace scan over Sprint 123 files.

## Day 6 Notes

- Accepted one bounded compatible tall QR external fixture for Day 8:
  `qr_overdetermined_compatible_5x3`.
- Pinned the compatible fixture to a 5x3 full-column-rank matrix, expected
  solution `[1.0, -2.0, 0.5]`, right-hand side
  `[2.0, -2.5, 4.0, -0.5, 2.0]`, `OK 4` helper output, solution max-diff below
  `1e-8`, and residual-norm difference below `1e-8`.
- Identified Day 8 affected surfaces as `tests/qr_external_dense_reference.py`
  and `tests/test_qr_solve.c`, with no expected `tests/test_qr.c`, Makefile,
  CMake, CTest, public API, or public documentation changes.
- Deferred rank-deficient QR external evidence because a residual-only fixture
  risks hiding rank-threshold, nullspace, pseudoinverse, and minimum-norm
  policies behind the existing full-rank normal-equation helper pattern.
- Preserved the completed `qr_overdetermined_incompatible_4x2` lane as a
  duplicate fence and kept Q/economy and underdetermined minimum-norm evidence
  for Day 7.
- Day 6 changed documentation only; validation used `git diff --check` and a
  focused trailing-whitespace scan over Sprint 123 files.

## Day 7 Notes

- Deferred underdetermined minimum-norm external QR evidence to a future QR
  solve / minimum-norm oracle owner.
- Deferred Q/economy external evidence to a future QR basis/economy owner that
  can define sign, orientation, projection, subspace, and economy-shape
  semantics before implementation.
- Preserved current minimum-norm ownership across `tests/test_qr_solve.c`,
  `tests/test_colamd.c`, and `tests/test_svd.c` rather than introducing a
  generic external `minnorm` helper.
- Preserved current Q/economy ownership in `tests/test_qr.c`, including Q
  application, orthogonality, economy solve, thin-Q shape, economy R shape,
  wide economy behavior, sparse-mode behavior, and SuiteSparse smoke coverage.
- Limited Day 8 implementation scope to the Day 6 accepted
  `qr_overdetermined_compatible_5x3` fixture unless implementation discovers a
  concrete deferral reason.
- Day 7 changed documentation only; validation used `git diff --check` and a
  focused trailing-whitespace scan over Sprint 123 files.

## Day 8 Notes

- Implemented the accepted `qr_overdetermined_compatible_5x3` external QR
  fixture from Day 6.
- Extended `tests/qr_external_dense_reference.py` with the 5x3 compatible
  fixture and generalized the bounded normal-equation solve from 2x2-only to a
  small dense Gaussian-elimination helper.
- Extended `tests/test_qr_solve.c` with the fixture allow-list entry, sparse
  comparison test, and existing-suite registration.
- Kept rank-deficient QR, underdetermined minimum-norm QR, and Q/economy
  external evidence deferred; no `tests/test_qr.c`, `tests/test_colamd.c`,
  Makefile, CMake, CTest, public API, or public documentation changes were
  added.
- Focused helper proof emitted `OK 4` for
  `qr_overdetermined_compatible_5x3`; focused `test_qr_solve` passed with
  15 tests, 0 failures, 0 skips, and 1042 assertions.
- Full required code gate passed: `make format && make lint && make test`.
- `git diff --check` and the focused trailing-whitespace scan over Sprint 123
  docs plus touched QR/SVD files passed.

## Day 9 Notes

- Inventoried current partial-SVD evidence across top-k singular values,
  vector residuals, rectangular fixtures, rank-deficient fixtures, ordering,
  convergence/timing smoke, and low-rank approximation checks.
- Confirmed the only existing external partial-SVD lane is the bounded
  `partial_svd_diag6_k2` top-k singular-value fixture.
- Separated partial-SVD value semantics from vector, subspace,
  repeated-spectrum, clustered-spectrum, rectangular, rank-deficient,
  convergence-budget, and low-rank semantics.
- Defined sign-invariant vector policy, projection/subspace requirements for
  ambiguous bases, convergence-budget interpretation rules, duplicate fences,
  and non-claims.
- Recommended `partial_svd_tall_diag_8x5_k3` as the lowest-risk Day 10
  value-only implementation candidate if Day 10 chooses to implement rather
  than defer.
- Day 9 changed documentation only; validation used `git diff --check` and a
  focused trailing-whitespace scan over Sprint 123 files.

## Day 10 Notes

- Accepted and implemented the bounded `partial_svd_tall_diag_8x5_k3`
  value-only external partial-SVD lane.
- Extended `tests/svd_external_dense_reference.py` with the 8x5 tall diagonal
  fixture and top-three singular-value emission.
- Added the fixture key to the SVD external-reference allow-list in
  `tests/test_svd.c`.
- Added and registered
  `test_partial_svd_external_dense_reference_tall_diag_8x5_k3` under the
  existing partial-SVD helper owner in `tests/test_svd_partial_helpers.h`.
- Kept vector/subspace, repeated/clustered spectrum, rank-threshold,
  convergence-budget, low-rank optimality, public API, CMake/CTest, package,
  platform, ABI, performance, and broad partial-SVD external parity claims
  explicitly unsupported.
- Focused helper proof emitted `OK 3` for
  `partial_svd_tall_diag_8x5_k3`; focused `test_svd` passed with 108 tests,
  0 failures, 0 skips, and 1769 assertions.
- Full required code gate passed: `make format && make lint && make test`.
- `git diff --check` and the focused trailing-whitespace scan over Sprint 123
  docs plus touched SVD/QR files passed.

## Day 11 Notes

- Inventoried minimum-norm ownership across `tests/test_qr_solve.c`,
  `tests/test_colamd.c`, `tests/test_svd.c`, QR/SVD helper headers, and the
  shared external-reference helper surface.
- Confirmed the current duplication is mostly small measurement or fixture
  setup, while the assertions encode distinct QR solve, COLAMD/reordering,
  fallback, refinement, rank-deficient, SuiteSparse, and SVD pseudoinverse
  behavior.
- Deferred minimum-norm helper migration because moving assertions into
  generic helpers would hide tolerance, owner, and non-claim semantics.
- Defined future behavior-specific helper names and promotion gates for a
  dedicated QR solve / minimum-norm consolidation owner.
- Day 11 changed documentation only; validation used `git diff --check` and a
  focused trailing-whitespace scan over Sprint 123 files.

## Day 12 Notes

- Inventoried Bidiagonal/Golub-Kahan ownership across `tests/test_bidiag.c`,
  `tests/test_svd.c`, `tests/test_svd_helpers.h`, and
  `tests/test_svd_partial_helpers.h`.
- Confirmed the specialized semantics remain too rich for general SVD helper
  absorption: wide internal transpose, implicit Householder reconstruction,
  explicit extracted `U`/`V` reconstruction, wide GK reconstruction skip,
  bidiagonal QR iteration, and scenario-local tolerance ownership.
- Deferred helper extraction to a future dedicated Bidiagonal/GK
  maintainability owner rather than moving code today.
- Defined future helper names, anti-pattern names, promotion gates, and
  validation requirements for any later extraction.
- Day 12 changed documentation only; validation used `git diff --check` and a
  focused trailing-whitespace scan over Sprint 123 files.

## Day 13 Notes

- Inventoried maintainer-guide evidence tables and support docs for stale
  SVD, QR, and partial-SVD external-helper claim boundaries.
- Refreshed `docs/maintainer_guide.md` QR and SVD rows so the bounded Sprint
  121-123 external dense-reference lanes are tied to their family-local test
  and helper owners.
- Fixed stale Sprint 103 interpretation that described external
  dense-reference evidence as limited to Sprint 102 direct-solver lanes.
- Cross-checked `README.md`, `docs/solver_selection.md`, and
  `docs/algorithm.md`; no unsupported public claim drift or edits were needed.
- Drafted the residual queue for the Day 14 solver-selection claim gate.
- Day 13 changed documentation only; validation used `git diff --check`, a
  focused trailing-whitespace scan, and a stale-phrase scan over the maintainer
  guide.

## Day 14 Notes

- Reviewed Sprint 123 implementation and deferral outcomes across SVD, QR,
  partial-SVD, minimum-norm helpers, Bidiagonal/Golub-Kahan helpers, and the
  maintainer evidence refresh.
- Decided not to update `docs/solver_selection.md` because the current public
  wording already stays at workflow guidance level and does not claim broad
  external parity or fixture-level proof.
- Published the no-update rationale, final non-claim register,
  dependency-ordered residual deferred debt, validation summary, and
  retrospective inputs in the Day 14 closeout artifact.
- Confirmed all Sprint 123 project-plan items are complete through bounded
  implementation, explicit deferral, maintainer evidence refresh, or public
  claim no-update rationale.
- Day 14 changed documentation only; validation used `git diff --check`, a
  focused trailing-whitespace scan, and a public/support claim-boundary scan.
