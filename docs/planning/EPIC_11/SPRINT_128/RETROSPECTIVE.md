# Sprint 128 Retrospective

**Sprint:** 128 - QR Residual Claim-Gate Closure & Corpus Semantics
**Duration:** 14 days
**Status:** Complete

## Definition Of Done Checklist

- [x] Created Sprint 128 day-by-day plan, working notes, and artifact
      directory.
- [x] Mapped Sprint 127 residual deferred debt against completed Sprint
      121-127 QR residual, nullspace/subspace, threshold, SuiteSparse,
      optional-large, minimum-norm, QR-vs-SVD, and helper evidence without
      duplicating completed fixtures.
- [x] Defined compatible zero-residual and wide residual-only semantics with
      explicit underdetermined output, solution-selection, nullspace,
      minimum-norm, Q/economy, and sparse-mode boundaries.
- [x] Explicitly deferred compatible zero-residual and wide residual-only
      evidence until distinct trust value and output semantics are pinned.
- [x] Refreshed wide, near-threshold, and SuiteSparse nullspace/subspace policy
      with rank/nullity, projector, two-way projection, support-tier,
      skip-behavior, and raw-basis non-claim gates.
- [x] Implemented the bounded
      `qr_rankdef_wide_3x5_nullspace_subspace` projector fixture.
- [x] Refreshed remaining threshold-family policy for dependent-row, wide,
      default-threshold, SuiteSparse, and near-threshold candidates.
- [x] Implemented the bounded
      `qr_rank_threshold_dependent_row_4x3_perturbed_family` threshold fixture.
- [x] Re-ran the SuiteSparse rank-deficient QR corpus gate and explicitly
      deferred additional corpus evidence until independent expected-rank
      metadata, threshold semantics, support tier, diagnostics, skip behavior,
      runtime, and validation are pinned.
- [x] Re-ran the SuiteSparse and optional-large minimum-norm gate and preserved
      the Sprint 125 `west0067` 30 x 67 submatrix smoke as the only accepted
      checked-in SuiteSparse minimum-norm corpus baseline.
- [x] Preserved Sprint 125-127 exact minimum-norm and QR-vs-SVD evidence:
      `qr_underdetermined_minnorm_2x4`, `qr_minnorm_5x10_exact_values`,
      `qr_minnorm_3x6_exact_values`, and the bounded 2 x 4 QR-vs-SVD
      cross-check.
- [x] Explicitly deferred duplicate exact minimum-norm lanes, additional
      QR-vs-SVD cross-checks, SuiteSparse-derived exact/cross-check lanes, and
      generic helper movement.
- [x] Updated `docs/maintainer_guide.md` with the accepted Sprint 128 QR
      evidence while preserving broad non-claims.
- [x] Ran focused helper and owner-test checks for every accepted Sprint 128
      evidence lane.
- [x] Ran the required full C quality gate after Sprint 128 code changes:
      `make format && make lint && make test`.
- [x] Published final evidence, deferral, validation, non-claim, and Sprint
      129 handoff artifacts.
- [x] Finalized this retrospective and ran final documentation hygiene.

## What Went Well

1. **The sprint stayed deduplicated.** Sprint 128 reused completed Sprint
   121-127 residual, projector, threshold, SuiteSparse, minimum-norm, and
   QR-vs-SVD lanes as baselines instead of relabeling them as broader proof.

2. **Wide nullspace evidence landed with a bounded metric.** The new 3 x 5
   rank-deficient fixture compares a locally orthonormalized product
   nullspace projector to a standard-library reference while preserving raw
   basis, ordering, sign, unique-basis, Q/economy, and minimum-norm
   non-claims.

3. **Threshold evidence gained a dependent-row perturbation family.** The new
   4 x 3 fixture records the perturbation, thresholds, expected ranks,
   absolute thresholds, pivot ratio, and R diagonal diagnostics before making
   a fixture-local threshold claim.

4. **Compatible and wide residual candidates did not overclaim.** The sprint
   explicitly deferred zero-residual and wide residual-only lanes that could
   not prove distinct trust value or clean underdetermined output semantics.

5. **SuiteSparse decisions stayed metadata-driven.** Rank-deficient QR corpus
   and additional minimum-norm corpus lanes remained blocked until independent
   expected-rank, extraction, rank/nullity, support-tier, skip, runtime, and
   validation metadata exist.

6. **The exact minimum-norm and QR-vs-SVD boundary stayed tight.** Sprint 128
   preserved the accepted 2 x 4, 3 x 6, and 5 x 10 exact/cross-check lanes and
   rejected duplicate work that would have turned SVD-pseudoinverse into a
   broader oracle claim.

7. **The Sprint 129 handoff is explicit.** Q-basis, economy, and helper
   ownership work can start from named claim boundaries instead of reopening
   Sprint 128 residual, corpus, and minimum-norm decisions.

## What Did Not Go Well

1. **Most remaining work is still blocked by metadata, not code mechanics.**
   SuiteSparse rank-deficient QR, optional-large minimum-norm, near-threshold
   subspace, and corpus cross-check lanes still need independent references
   before implementation is defensible.

2. **Compatible zero-residual evidence remains deferred.** The candidate lane
   still needs proof that it adds trust beyond existing compatible solve,
   rank-only, residual-only, and rank-deficient solve-smoke coverage.

3. **Wide residual-only evidence remains deferred.** Underdetermined output
   semantics, solution-selection wording, Q/economy boundaries, and
   residual-only proof value remain future-owner work.

4. **SuiteSparse rank-deficient QR evidence remains unavailable.** Checked-in
   matrices continue to serve as controls, not rank-deficient evidence, because
   expected-rank and threshold metadata are not independently pinned.

5. **Additional minimum-norm corpus evidence remains constrained.** The
   `west0067` smoke stays useful, but new SuiteSparse or optional-large lanes
   need extraction, RHS, rank/nullity, residual, norm, skip, runtime, and
   support-tier metadata.

6. **Helper consolidation remains deferred.** This protects owner clarity, but
   Sprint 129 still needs behavior-specific helper ownership decisions before
   Q/economy work grows.

## Final Metrics

### Validation

| Metric | Sprint 128 close state |
|---|---:|
| wide nullspace helper | passed, emitted `OK 29` |
| dependent-row threshold helper | passed, emitted `OK 9` |
| focused QR tests | 74 passed, 0 failed, 0 skipped |
| focused QR assertions | 885 |
| focused QR solve tests | 19 passed, 0 failed, 0 skipped |
| focused QR solve assertions | 1104 |
| focused COLAMD/minimum-norm tests | 70 passed, 0 failed, 0 skipped |
| focused COLAMD/minimum-norm assertions | 317 |
| required full Make formatting | `make format` passed |
| required full Make lint | `make lint` passed |
| required full Make tests | `make test` passed |
| full Make test final result | `All tests passed.` |
| diff hygiene | `git diff --check` passed |
| trailing-whitespace scan | passed on Sprint 128 docs and touched files |
| Python cache cleanup | no generated cache kept in the worktree |
| public docs wording expansion | 0 |

### Sprint Artifact Package

| Metric | Sprint 128 close state |
|---|---:|
| artifact files under `SPRINT_128/artifacts/` | 14 |
| sprint plan files | 1 |
| working notes files | 1 |
| retrospective files | 1 |
| modified external-reference helper scripts | 1 |
| modified existing test files | 1 |
| maintainer guide updates | 1 |
| public solver-selection wording updates | 0 |
| README/public-header wording updates | 0 |
| Makefile/CMake/CTest registration changes | 0 |

## Movement And Claim Outcomes

| Area | Outcome |
|---|---|
| Sprint intake | Completed Sprint 127 residual dedupe map, duplicate fence, validation boundary, and day-level owner map. |
| Compatible and wide residual semantics | Defined trust rules and deferred compatible zero-residual and wide residual-only lanes until promotion metadata is explicit. |
| Compatible and wide residual evidence | Deferred; no candidate proved distinct trust value beyond existing bounded residual and solve fixtures. |
| Wide/near-threshold subspace policy | Refreshed projector, two-way projection, rank/nullity, support-tier, output-semantics, and raw-basis non-claim rules. |
| Wide subspace evidence | Added bounded `qr_rankdef_wide_3x5_nullspace_subspace` projector evidence. |
| Near-threshold and SuiteSparse subspace evidence | Deferred behind rank/nullity, projection metric, support-tier, skip, runtime, and output-semantics gates. |
| Threshold-family policy | Refreshed dependent-row, wide, default-threshold, SuiteSparse, diagnostics, support-tier, and failure-interpretation gates. |
| Dependent-row threshold evidence | Added bounded `qr_rank_threshold_dependent_row_4x3_perturbed_family` rank-threshold evidence. |
| Wide/default/SuiteSparse threshold evidence | Deferred until expected ranks, threshold semantics, diagnostics, support tier, skip behavior, and validation are explicit. |
| SuiteSparse rank-deficient QR evidence | Deferred; current checked-in QR SuiteSparse matrices remain controls until independent expected-rank metadata exists. |
| SuiteSparse and optional-large minimum-norm evidence | Deferred additional corpus lanes; Sprint 125 `west0067` submatrix smoke remains the checked-in baseline. |
| Exact minimum-norm evidence | Preserved completed Sprint 125-127 exact-value lanes; no duplicate Sprint 128 exact fixture was accepted. |
| QR-vs-SVD minimum-norm evidence | Deferred additional cross-checks; Sprint 125 2 x 4 cross-check remains the bounded baseline. |
| Helper movement | Deferred generic QR/SVD/minimum-norm helper consolidation; behavior-specific names and call-site tolerances remain required. |
| Maintainer evidence | Updated `docs/maintainer_guide.md` with Sprint 128 bounded wide subspace and dependent-row threshold evidence while preserving non-claims. |
| Public docs | No public wording expansion; README, solver-selection, public headers, package metadata, and API wording remain unchanged for Sprint 128. |
| Public API | Unchanged. |
| Build registration | Unchanged; no new executable, library source, Makefile entry, CMake entry, or CTest member was added. |
| External-library parity | Not claimed. |

## Residual Deferred Debt

Most important carry-forward work:

- Add compatible zero-residual rank-deficient QR residual evidence only after
  proving distinct trust value beyond existing compatible solve, rank-only,
  residual-only, and rank-deficient solve-smoke coverage.
- Add wide residual-only QR evidence only after underdetermined output
  semantics, solution-selection policy, Q/economy boundaries, sparse-mode
  boundaries, and residual-only proof value are pinned.
- Add near-threshold nullspace/subspace evidence only after rank/nullity,
  projector or two-way projection metric, threshold semantics, tolerance,
  diagnostics, and failure interpretation are pinned.
- Add SuiteSparse QR nullspace/subspace evidence only after independent
  rank/nullity metadata, support tier, optional-data behavior, runtime budget,
  skip behavior, and diagnostics are explicit.
- Add wide, default-threshold, or SuiteSparse threshold families only after
  expected ranks, threshold semantics, support tier, skip behavior, diagnostics,
  and failure interpretation are pinned.
- Add SuiteSparse rank-deficient QR corpus evidence only after independent
  expected-rank metadata, threshold semantics, support tier, diagnostics, skip
  behavior, runtime budget, and validation are explicit.
- Add additional SuiteSparse minimum-norm evidence only after extraction rule,
  shape, nnz, RHS, expected rank/nullity when claimed, residual/norm metrics,
  skip behavior, runtime expectations, and support tier are pinned.
- Add optional-large SuiteSparse QR or minimum-norm evidence only through the
  optional-large gate with missing-data skip behavior and runtime/platform
  expectations recorded before default test registration.
- Add additional exact underdetermined minimum-norm lanes only for
  non-duplicate shapes with closed-form expected values, exact norm, residual
  tolerance, value tolerance, norm tolerance, diagnostics, and owner-local
  placement.
- Add additional QR-vs-SVD minimum-norm fixtures only as bounded cross-checks
  with fixture keys, QR residual and norm metrics, SVD tolerance, and
  non-oracle wording per fixture.
- Revisit generic QR/SVD/minimum-norm helper movement only with
  behavior-specific helper names, visible owner call-site tolerances, focused
  QR solve/COLAMD/SVD validation, and the full quality gate.
- Continue Q-basis, economy, sparse-mode Q/economy, and helper ownership
  follow-through in Sprint 129 without reopening Sprint 128 residual, corpus,
  threshold, or minimum-norm claim gates unless a fixture has a distinct
  behavior-specific claim.

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
- no package-manager distribution claim;
- no shared-library or dynamic ABI stability claim;
- no equal Linux/macOS/Windows reviewed-support claim;
- no public API, install-header, package, CMake, Makefile, CI, or CTest
  expansion claim;
- no portable performance, scalability, memory, or state-of-the-art claim.

Not carried forward as unresolved Sprint 128 debt:

- Sprint 128 intake, residual dependency map, and duplicate fencing;
- compatible/wide residual semantics policy and explicit evidence decision;
- wide/near-threshold subspace policy refresh;
- `qr_rankdef_wide_3x5_nullspace_subspace` implementation;
- remaining threshold-family policy refresh;
- `qr_rank_threshold_dependent_row_4x3_perturbed_family` implementation;
- SuiteSparse rank-deficient QR corpus gate and explicit deferral;
- SuiteSparse and optional-large minimum-norm gate and explicit deferral;
- exact minimum-norm and QR-vs-SVD/helper gate;
- additional exact minimum-norm and QR-vs-SVD explicit deferral;
- helper movement explicit deferral;
- final validation package and Sprint 129 handoff evidence.

## Key Deliverables

- [PLAN.md](./PLAN.md)
- [WORKING_NOTES.md](./WORKING_NOTES.md)
- [day1-residual-dedupe-baseline.md](./artifacts/day1-residual-dedupe-baseline.md)
- [day2-compatible-wide-residual-semantics-policy.md](./artifacts/day2-compatible-wide-residual-semantics-policy.md)
- [day3-compatible-wide-residual-evidence-decision.md](./artifacts/day3-compatible-wide-residual-evidence-decision.md)
- [day4-wide-near-threshold-subspace-policy.md](./artifacts/day4-wide-near-threshold-subspace-policy.md)
- [day5-wide-near-threshold-subspace-evidence.md](./artifacts/day5-wide-near-threshold-subspace-evidence.md)
- [day6-remaining-threshold-family-policy.md](./artifacts/day6-remaining-threshold-family-policy.md)
- [day7-remaining-threshold-family-evidence.md](./artifacts/day7-remaining-threshold-family-evidence.md)
- [day8-suitesparse-rankdef-qr-corpus-gate.md](./artifacts/day8-suitesparse-rankdef-qr-corpus-gate.md)
- [day9-suitesparse-rankdef-qr-evidence-decision.md](./artifacts/day9-suitesparse-rankdef-qr-evidence-decision.md)
- [day10-suitesparse-optional-large-minnorm-policy.md](./artifacts/day10-suitesparse-optional-large-minnorm-policy.md)
- [day11-suitesparse-optional-large-minnorm-evidence-decision.md](./artifacts/day11-suitesparse-optional-large-minnorm-evidence-decision.md)
- [day12-exact-minnorm-crosscheck-helper-gate.md](./artifacts/day12-exact-minnorm-crosscheck-helper-gate.md)
- [day13-crosscheck-helper-integrated-validation.md](./artifacts/day13-crosscheck-helper-integrated-validation.md)
- [day14-sprint-closeout-handoff.md](./artifacts/day14-sprint-closeout-handoff.md)
