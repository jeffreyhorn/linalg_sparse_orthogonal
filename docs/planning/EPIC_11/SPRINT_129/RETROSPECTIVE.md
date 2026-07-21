# Sprint 129 Retrospective

**Sprint:** 129 - QR Q-Basis, Economy & Helper Ownership Follow-Through
**Duration:** 14 days
**Status:** Complete

## Definition Of Done Checklist

- [x] Created Sprint 129 day-by-day plan, working notes, and artifact
      directory.
- [x] Mapped Sprint 129 project-plan items to Q-basis, economy, sparse-mode,
      SuiteSparse Q/economy, minimum-norm helper, and Bidiagonal/Golub-Kahan
      helper owners.
- [x] Preserved the Sprint 128 residual no-reopen boundary so residual QR debt
      stayed in the end-of-epic queue unless a Q/economy/helper candidate had
      a distinct behavior-specific claim and satisfied its promotion gate.
- [x] Refreshed raw Q-basis and economy policy with sign, orientation,
      projection, economy-shape, sparse-mode, SuiteSparse, support-tier, skip,
      runtime, and non-claim rules.
- [x] Explicitly deferred raw Q-column equality because no candidate added
      enough trust beyond shape, orthogonality, reconstruction, projector,
      projection, and Q-application metrics.
- [x] Refreshed rank-deficient Q/nullspace policy against Sprint 125-128
      projector, threshold, SuiteSparse, optional-large, and minimum-norm
      evidence.
- [x] Implemented the bounded
      `test_qr_dependent_row_q_transpose_column_space_rhs` Q-application
      evidence lane.
- [x] Refreshed wide economy and sparse-mode output policy with shape, product,
      tolerance, diagnostics, and non-claim requirements.
- [x] Implemented the bounded `test_sparse_mode_economy_tall_q_shape`
      sparse-mode economy evidence lane.
- [x] Re-ran the SuiteSparse Q/economy support-tier gate and accepted only the
      checked-in small-control `nos4` sparse-mode economy Q orthogonality lane.
- [x] Implemented
      `test_suitesparse_nos4_sparse_mode_economy_q_orthogonality`.
- [x] Re-ran the minimum-norm helper ownership gate and explicitly deferred
      helper movement because QR solve, COLAMD/minimum-norm, and SVD
      pseudoinverse fixtures differ in owner semantics.
- [x] Re-ran the Bidiagonal/Golub-Kahan helper gate and extracted exactly one
      Bidiagonal-owned helper while preserving Golub-Kahan/SVD ownership.
- [x] Added `tests/test_bidiag_helpers.h` with
      `tf_bidiag_reconstruction_max_error` and updated `tests/test_bidiag.c`
      call sites.
- [x] Left public API, public solver-selection wording, README, package
      metadata, Matrix Market fixtures, Makefile, CMake, and CTest membership
      unchanged.
- [x] Ran focused owner checks for every accepted Sprint 129 code lane.
- [x] Ran the required full C quality gate after Sprint 129 code changes:
      `make format && make lint && make test`.
- [x] Published final evidence, deferral, validation, non-claim, and Sprint
      130 handoff artifacts.
- [x] Finalized this retrospective and ran final documentation hygiene.

## What Went Well

1. **The sprint did not reopen Sprint 128 residual debt.** Sprint 129 stayed
   focused on Q-basis, economy, sparse-mode, SuiteSparse Q/economy, and helper
   ownership instead of continuing the residual-debt churn.

2. **Raw Q equality stayed out of the required suite.** The sprint preserved
   durable, basis-invariant metrics and rejected raw Q-column checks that
   would mostly pin sign, orientation, or Householder implementation details.

3. **Rank-deficient Q evidence landed as a Q-application claim.** The
   dependent-row `Q^T b` fixture checks a column-space RHS and round-trip
   product behavior without becoming a new nullspace projector, residual-only,
   minimum-norm, raw-basis, economy, sparse-mode, or SuiteSparse claim.

4. **Sparse-mode economy behavior gained bounded shape coverage.** The tall
   sparse-mode economy fixture records Q/R shape, orthogonality, and dense
   economy versus sparse economy solve/residual agreement without claiming
   backend, platform, performance, or broad sparse QR parity.

5. **SuiteSparse Q/economy evidence stayed support-tier aware.** Sprint 129
   accepted only the checked-in `nos4` sparse-mode economy Q orthogonality lane
   and kept `west0067`, `bcsstk04`, larger checked-in matrices, raw
   SuiteSparse Q, rank-deficient corpus, and minimum-norm corpus work behind
   future gates.

6. **Minimum-norm helper movement was usefully rejected.** The sprint found
   that QR solve/SVD and COLAMD fixtures use different 2 x 4 layouts, so a
   generic helper would hide owner semantics instead of paying down real debt.

7. **Bidiagonal helper ownership became clearer.** The extracted helper is
   Bidiagonal-owned, test-only, header-local, and keeps implicit Householder
   reconstruction separate from explicit Golub-Kahan `U`/`V` reconstruction.

## What Did Not Go Well

1. **A lot of Q/economy work still ends as gatekeeping.** Raw Q-column,
   wide/economy, near-threshold, and larger SuiteSparse candidates still need
   more metadata before implementation is defensible.

2. **SuiteSparse rank-deficient QR remains blocked.** Checked-in matrices
   remain controls, not rank-deficient evidence, until independent rank/nullity
   metadata and threshold semantics exist.

3. **Large SuiteSparse Q/economy lanes are still report-only candidates.**
   They need explicit runtime, memory, skip/report, support-tier, and
   failure-interpretation policies before required test registration.

4. **Minimum-norm helper consolidation remains fragmented.** This is the right
   outcome for owner clarity, but QR solve, COLAMD, SVD pseudoinverse, and
   SuiteSparse submatrix helpers still need owner-specific gates if future
   duplication becomes costly.

5. **Golub-Kahan helper movement did not happen.** It stayed deferred because
   current reuse is not strong enough to justify moving explicit `U`/`V`
   reconstruction out of `tests/test_svd.c`.

6. **Public solver-selection wording still has no new claim.** Sprint 129
   improved internal evidence and helper ownership, but not enough to justify
   user-facing solver-selection expansion.

## Final Metrics

### Validation

| Metric | Sprint 129 close state |
|---|---:|
| focused QR tests after Day 5/7/9 code lanes | passed |
| focused Bidiagonal tests | 12 passed, 0 failed, 0 skipped |
| focused Bidiagonal assertions | 164 |
| focused SVD/Golub-Kahan tests | 109 passed, 0 failed, 0 skipped |
| focused SVD/Golub-Kahan assertions | 1802 |
| required full Make formatting | `make format` passed |
| required full Make lint | `make lint` passed |
| required full Make tests | `make test` passed |
| full Make test final result | `All tests passed.` |
| diff hygiene | `git diff --check` passed |
| trailing-whitespace scan | passed on Sprint 129 docs and touched files |
| public docs wording expansion | 0 |

### Sprint Artifact Package

| Metric | Sprint 129 close state |
|---|---:|
| artifact files under `SPRINT_129/artifacts/` | 14 |
| sprint plan files | 1 |
| working notes files | 1 |
| retrospective files | 1 |
| new test helper headers | 1 |
| modified existing test files | 2 |
| maintainer guide updates | 0 |
| public solver-selection wording updates | 0 |
| README/public-header wording updates | 0 |
| Makefile/CMake/CTest registration changes | 0 |

## Movement And Claim Outcomes

| Area | Outcome |
|---|---|
| Sprint intake | Completed Q/economy/helper owner map, duplicate fence, validation boundary, and Sprint 128 residual no-reopen rule. |
| Q-basis/economy policy | Refreshed sign, orientation, projection, economy-shape, sparse-mode, SuiteSparse, support-tier, skip, runtime, and non-claim rules. |
| Raw Q-column evidence | Deferred; no raw equality candidate proved distinct trust beyond existing product, projector, orthogonality, reconstruction, or Q-application evidence. |
| Rank-deficient Q/nullspace policy | Refreshed projector, projection, rank/nullity, threshold, SuiteSparse, sparse/economy, and raw-basis non-claim rules. |
| Rank-deficient Q evidence | Added bounded `test_qr_dependent_row_q_transpose_column_space_rhs` Q-application evidence. |
| Wide economy/sparse-mode policy | Refreshed output-shape and metric rules for tall, square, wide, sparse-mode, and SuiteSparse Q/economy surfaces. |
| Sparse-mode economy evidence | Added bounded `test_sparse_mode_economy_tall_q_shape`. |
| SuiteSparse Q/economy gate | Accepted only the checked-in `nos4` sparse-mode economy Q orthogonality lane. |
| SuiteSparse Q/economy evidence | Added bounded `test_suitesparse_nos4_sparse_mode_economy_q_orthogonality`. |
| Minimum-norm helper movement | Deferred; fixture topology differs across QR solve/SVD and COLAMD owners. |
| Bidiagonal/Golub-Kahan helper gate | Accepted one Bidiagonal-owned helper movement and left Golub-Kahan/SVD helpers in place. |
| Bidiagonal helper movement | Added `tests/test_bidiag_helpers.h` with `tf_bidiag_reconstruction_max_error`. |
| Maintainer evidence | No update; Sprint 129 did not create a new maintainer-guide evidence row. |
| Public docs | No public wording expansion; README, solver-selection, public headers, package metadata, and API wording remain unchanged. |
| Public API | Unchanged. |
| Build registration | Unchanged; no new executable, library source, Makefile entry, CMake entry, or CTest member was added. |
| External-library parity | Not claimed. |

## Residual Deferred Debt

Most important carry-forward work:

- Add raw QR Q-column evidence only if a future fixture has explicit sign
  normalization, column order, storage layout, permutation interpretation,
  tolerance, diagnostics, and distinct trust value beyond product metrics.
- Add wide economy/nullspace interaction only after explicit wide output
  shape, underdetermined solution semantics, projection metric, tolerance, and
  non-minimum-norm wording are pinned.
- Add near-threshold Q/nullspace or subspace lanes only after expected rank,
  nullity, threshold semantics, projector or two-way projection metric,
  diagnostics, and failure interpretation are pinned.
- Add additional duplicate-column or rank-1/nullity projector lanes only for a
  distinct claim not already covered by Sprint 125-128 projector evidence.
- Add `west0067`, `bcsstk04`, or larger SuiteSparse Q/economy lanes only after
  runtime budget, support tier, skip/report policy, independent metric,
  diagnostics, and failure interpretation are explicit.
- Add SuiteSparse rank-deficient QR corpus evidence only after independent
  expected-rank and rank/nullity metadata, threshold semantics, support tier,
  diagnostics, skip behavior, runtime budget, and validation are explicit.
- Add SuiteSparse or optional-large minimum-norm evidence only after
  extraction rule, shape, nnz, RHS, rank/nullity if claimed, residual/norm
  metrics, support tier, runtime, and skip behavior are pinned.
- Add QR-solve-local, COLAMD-local, SVD-local, or SVD pseudoinverse helper
  movement only under owner-specific gates with behavior-specific helper
  names, visible expected values, tolerances, layouts, and diagnostics.
- Add Golub-Kahan reconstruction helper movement only if future cross-file
  reuse justifies a dedicated SVD/GK helper owner while preserving explicit
  `U`/`V`, `diag`, `superdiag`, and wide-skip semantics.
- Move partial-SVD helpers only through Sprint 130 partial-SVD residual and
  subspace metric policy, not through Sprint 129 Q/economy/helper ownership.
- Continue partial-SVD residual expansion and solver-selection claim gating in
  Sprint 130 without reopening Sprint 129 Q/economy/helper boundaries.

Still consciously constrained rather than silently solved:

- no LAPACK parity claim;
- no SciPy or NumPy parity claim;
- no BLAS, SuiteSparse, PETSc, Trilinos, Eigen, ARPACK, or vendor-backend
  parity claim;
- no broad external dense-library, external package, or ecosystem parity
  claim;
- no broad QR factorization, QR solve, compatible solve, wide solve,
  rank-deficient solve, nullspace, minimum-norm, Q-basis, economy,
  sparse-mode, reorder, backend, corpus, optional-data, platform, or
  performance parity claim;
- no global QR rank-threshold, default-threshold, or numerical-rank policy;
- no raw Q-basis equality, Q-sign, Q-orientation, unique-basis, raw nullspace
  basis, or broad projection/subspace parity claim;
- no broad SVD-pseudoinverse oracle claim;
- no broad SuiteSparse corpus, optional-data, platform, runtime, or
  performance claim;
- no generic helper API or helper consolidation claim;
- no Golub-Kahan helper ownership claim from Bidiagonal helper movement;
- no partial-SVD helper movement claim;
- no package-manager distribution claim;
- no shared-library or dynamic ABI stability claim;
- no equal Linux/macOS/Windows reviewed-support claim;
- no public API, install-header, package, CMake, Makefile, CI, or CTest
  expansion claim;
- no public solver-selection wording readiness claim;
- no portable performance, scalability, memory, or state-of-the-art claim.

Not carried forward as unresolved Sprint 129 debt:

- Sprint 129 intake, owner map, validation boundary, and no-reopen rule;
- Q-basis/economy policy refresh;
- raw Q-column evidence decision and explicit deferral;
- rank-deficient Q/nullspace policy gate;
- `test_qr_dependent_row_q_transpose_column_space_rhs` implementation;
- wide economy and sparse-mode policy gate;
- `test_sparse_mode_economy_tall_q_shape` implementation;
- SuiteSparse Q/economy support-tier gate;
- `test_suitesparse_nos4_sparse_mode_economy_q_orthogonality`
  implementation;
- minimum-norm helper ownership gate and explicit movement deferral;
- Bidiagonal/Golub-Kahan helper gate;
- Bidiagonal reconstruction helper extraction into
  `tests/test_bidiag_helpers.h`;
- final validation package and Sprint 130 handoff evidence.

## Key Deliverables

- [PLAN.md](./PLAN.md)
- [WORKING_NOTES.md](./WORKING_NOTES.md)
- [day1-intake-no-reopen-boundary.md](./artifacts/day1-intake-no-reopen-boundary.md)
- [day2-q-basis-economy-policy-refresh.md](./artifacts/day2-q-basis-economy-policy-refresh.md)
- [day3-raw-q-column-evidence-decision.md](./artifacts/day3-raw-q-column-evidence-decision.md)
- [day4-rankdef-q-nullspace-policy-gate.md](./artifacts/day4-rankdef-q-nullspace-policy-gate.md)
- [day5-rankdef-q-nullspace-evidence.md](./artifacts/day5-rankdef-q-nullspace-evidence.md)
- [day6-wide-economy-sparse-mode-policy.md](./artifacts/day6-wide-economy-sparse-mode-policy.md)
- [day7-wide-economy-sparse-mode-evidence.md](./artifacts/day7-wide-economy-sparse-mode-evidence.md)
- [day8-suitesparse-q-economy-gate.md](./artifacts/day8-suitesparse-q-economy-gate.md)
- [day9-suitesparse-q-economy-evidence.md](./artifacts/day9-suitesparse-q-economy-evidence.md)
- [day10-minnorm-helper-ownership-gate.md](./artifacts/day10-minnorm-helper-ownership-gate.md)
- [day11-minnorm-helper-movement-decision.md](./artifacts/day11-minnorm-helper-movement-decision.md)
- [day12-bidiag-golub-kahan-helper-gate.md](./artifacts/day12-bidiag-golub-kahan-helper-gate.md)
- [day13-bidiag-golub-kahan-helper-decision.md](./artifacts/day13-bidiag-golub-kahan-helper-decision.md)
- [day14-sprint-closeout-handoff.md](./artifacts/day14-sprint-closeout-handoff.md)
