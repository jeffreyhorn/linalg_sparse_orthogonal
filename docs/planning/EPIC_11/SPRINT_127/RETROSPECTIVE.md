# Sprint 127 Retrospective

**Sprint:** 127 - QR Deferred Evidence Semantics & Corpus Follow-Through
**Duration:** 14 days
**Status:** Complete

## Definition Of Done Checklist

- [x] Created Sprint 127 day-by-day plan, working notes, and artifact
      directory.
- [x] Mapped Sprint 126 deferred residual debt to existing Sprint 121-126 QR
      residual, nullspace/subspace, threshold, SuiteSparse, minimum-norm,
      SVD-pseudoinverse, Q/economy, and helper evidence without duplicating
      completed fixtures.
- [x] Defined compatible zero-residual and wide residual-only semantics with
      explicit output-semantics, solution-selection, nullspace, minimum-norm,
      and Q/economy boundaries.
- [x] Explicitly deferred compatible zero-residual and wide residual-only
      evidence until distinct proof value and output semantics are pinned.
- [x] Refreshed nullspace/subspace policy for dependent-row, wide-shape,
      near-threshold, sparse/economy, and SuiteSparse candidates.
- [x] Implemented the bounded
      `qr_rankdef_dependent_row_4x3_nullspace_projector` projector fixture.
- [x] Refreshed threshold-family policy for perturbed duplicate-column,
      dependent-row, wide, default-threshold, and SuiteSparse candidates.
- [x] Implemented the bounded
      `qr_rank_threshold_duplicate_5x4_perturbed_family` threshold fixture.
- [x] Re-ran the SuiteSparse rank-deficient QR corpus gate and explicitly
      deferred additional corpus evidence until independent expected-rank
      metadata, threshold semantics, support tier, diagnostics, skip behavior,
      runtime, and validation are pinned.
- [x] Re-ran the SuiteSparse and optional-large minimum-norm gate and preserved
      the Sprint 125 `west0067` 30 x 67 submatrix smoke as the only accepted
      SuiteSparse minimum-norm corpus evidence.
- [x] Accepted the 3 x 6 underdetermined exact-value minimum-norm lane with
      closed-form solution values and `sqrt(8.4)` norm.
- [x] Implemented `qr_minnorm_3x6_exact_values` in the owner-local COLAMD test
      surface.
- [x] Explicitly deferred additional QR-vs-SVD minimum-norm cross-checks so
      SVD pseudoinverse remains a bounded cross-check, not a global QR oracle.
- [x] Kept generic QR/SVD helper movement deferred behind behavior-specific
      names, call-site tolerances, and focused owner validation.
- [x] Refreshed `docs/maintainer_guide.md` with the accepted Sprint 127 exact
      minimum-norm evidence and preserved broad non-claims.
- [x] Audited public solver-selection, README, public headers, package
      metadata, build metadata, CMake/CTest membership, optional-data, and API
      wording and made no public wording expansion.
- [x] Ran focused helper and owner-test checks for every accepted Sprint 127
      evidence lane.
- [x] Ran the required full C quality gate:
      `make format && make lint && make test`.
- [x] Published final validation, claim-gate, closeout, residual, non-claim,
      and Sprint 128 handoff artifacts.
- [x] Finalized this retrospective and ran final documentation hygiene.

## What Went Well

1. **Sprint 126 evidence stayed deduplicated.** Sprint 127 reused the
   completed dependent-row residual, rank-1/nullity-2 projector, scaled
   threshold, SuiteSparse deferrals, `west0067` minimum-norm smoke, and 5 x 10
   exact minimum-norm lanes as baselines instead of relabeling them as new
   evidence.

2. **The compatible/wide residual lane stayed honest.** The sprint resisted
   adding zero-residual or wide residual-only tests without distinct proof
   value, explicit underdetermined output semantics, and a clear separation
   from minimum-norm or Q/economy behavior.

3. **Dependent-row subspace evidence landed without raw-basis claims.** The
   accepted 4 x 3 projector fixture compares a normalized basis projector
   against a standard-library reference while preserving sign, orientation,
   ordering, sparse/economy, and unique-basis boundaries.

4. **Threshold evidence expanded into perturbation behavior.** The duplicate
   5 x 4 fixture now has a near-threshold perturbation family with expected
   ranks at two explicit thresholds and diagnostics that stay fixture-local.

5. **SuiteSparse gates remained metadata-driven.** The sprint did not force
   rank-deficient QR or minimum-norm SuiteSparse evidence out of matrices that
   lack independent rank, extraction, norm, support-tier, skip, or runtime
   metadata.

6. **The underdetermined exact-value lane gained a non-duplicate shape.** The
   3 x 6 fixture now asserts a closed-form solution vector, residual, and
   exact `sqrt(8.4)` norm without reusing the completed 2 x 4 or 5 x 10 lanes.

7. **The claim gate stayed bounded.** Maintainer evidence was refreshed only
   for the new exact fixture; public solver-selection, README, headers,
   packages, build metadata, and API wording stayed unchanged.

## What Did Not Go Well

1. **The evidence package remains intentionally fixture-scoped.** Sprint 127
   strengthened several named lanes, but broad QR, nullspace, minimum-norm,
   SuiteSparse, dense-library, platform, and performance parity remain
   non-claims.

2. **Compatible and wide residual evidence remain deferred.** Compatible
   zero-residual behavior still needs proof that it adds trust beyond existing
   deterministic compatible solve tests. Wide residual-only behavior still
   depends on underdetermined output and solution-selection semantics.

3. **SuiteSparse rank-deficient QR evidence remains blocked on metadata.**
   Checked-in QR SuiteSparse matrices continue to serve as full-rank controls,
   and optional/report-only matrices still lack support-tier and expected-rank
   proof.

4. **SuiteSparse minimum-norm expansion remains blocked.** Additional
   `west0067`, `steam1`, optional-large, report-only, and rank-deficient
   corpus candidates still need extraction, rank/nullity, residual, norm,
   skip, runtime, and support-tier metadata before registration.

5. **QR-vs-SVD minimum-norm coverage did not expand.** This preserved the
   non-oracle boundary, but broader cross-check confidence remains future work
   with fixture-specific wording and tolerances.

6. **Helper consolidation remains deferred.** The sprint intentionally kept
   behavior-specific owners visible, but Sprint 128 still needs a careful
   helper ownership decision before Q/economy and helper extraction work
   grows.

## Final Metrics

### Validation

| Metric | Sprint 127 close state |
|---|---:|
| dependent-row nullspace helper | passed, emitted `OK 13` |
| perturbed threshold-family helper | passed, emitted `OK 6` |
| focused QR tests | 72 passed, 0 failed, 0 skipped |
| focused QR assertions | 825 |
| focused QR solve tests | 19 passed, 0 failed, 0 skipped |
| focused QR solve assertions | 1104 |
| focused COLAMD/minimum-norm tests | 70 passed, 0 failed, 0 skipped |
| focused COLAMD/minimum-norm assertions | 317 |
| required full Make formatting | `make format` passed |
| required full Make lint | `make lint` passed |
| required full Make tests | `make test` passed |
| full Make test final result | `All tests passed.` |
| diff hygiene | `git diff --check` passed |
| trailing-whitespace scan | passed on Sprint 127 docs and touched files |
| Python cache scan | no `__pycache__` or `.pyc` artifacts found |
| public docs wording expansion | 0 |

### Sprint Artifact Package

| Metric | Sprint 127 close state |
|---|---:|
| artifact files under `SPRINT_127/artifacts/` | 14 |
| sprint plan files | 1 |
| working notes files | 1 |
| retrospective files | 1 |
| modified external-reference helper scripts | 1 |
| modified existing test files | 2 |
| maintainer guide updates | 1 |
| public solver-selection wording updates | 0 |
| README/public-header wording updates | 0 |
| Makefile/CMake/CTest registration changes | 0 |

## Movement And Claim Outcomes

| Area | Outcome |
|---|---|
| Sprint intake | Completed Sprint 126 residual dedupe map, duplicate fence, validation boundary, and day-level owner map. |
| Compatible and wide residual semantics | Defined trust rules and deferred compatible zero-residual and wide residual-only lanes until promotion metadata is explicit. |
| Nullspace/subspace policy | Refreshed projector, two-way projection, rank/nullity, sparse/economy, support-tier, and raw-basis non-claim rules. |
| Dependent-row nullspace evidence | Added bounded `qr_rankdef_dependent_row_4x3_nullspace_projector` projector evidence. |
| Threshold-family policy | Refreshed perturbed, dependent-row, wide, default-threshold, and SuiteSparse threshold gates. |
| Perturbed threshold evidence | Added bounded `qr_rank_threshold_duplicate_5x4_perturbed_family` rank-threshold evidence. |
| SuiteSparse rank-deficient QR evidence | Deferred; current checked-in QR SuiteSparse matrices remain controls until independent expected-rank metadata exists. |
| SuiteSparse and optional-large minimum-norm evidence | Deferred additional corpus lanes; Sprint 125 `west0067` submatrix smoke remains the checked-in baseline. |
| Underdetermined minimum-norm evidence | Added bounded `qr_minnorm_3x6_exact_values` exact-value and exact-norm evidence. |
| QR-vs-SVD minimum-norm evidence | Deferred additional cross-checks; Sprint 125 2 x 4 cross-check remains the bounded baseline. |
| Helper movement | Deferred generic QR/SVD helper movement; behavior-specific names and call-site tolerances remain required. |
| Maintainer evidence | Updated `docs/maintainer_guide.md` with Sprint 127 exact minimum-norm evidence and preserved non-claims. |
| Public docs | No public wording expansion; README, solver-selection, public headers, package metadata, and API wording remain unchanged for Sprint 127. |
| Public API | Unchanged. |
| Build registration | Unchanged; no new executable, library source, Makefile entry, CMake entry, or CTest member was added. |
| External-library parity | Not claimed. |

## Residual Deferred Debt

Most important carry-forward work:

- Add compatible zero-residual rank-deficient QR residual evidence only after
  proving the zero-residual lane adds trust beyond deterministic compatible
  solve behavior and cannot be misread as minimum-norm evidence.
- Add wide residual-only QR evidence only after Sprint 128 or a later owner
  defines underdetermined output semantics, solution-selection policy,
  Q/economy boundaries, and residual-only proof value.
- Add wide-shape nullspace/subspace evidence only after rank/nullity,
  projection metric, tolerance, and sparse/economy output semantics are pinned.
- Add near-threshold or SuiteSparse nullspace/subspace evidence only after
  projector or two-way projection residual metrics, expected rank/nullity,
  threshold semantics, and support tier are explicit.
- Add dependent-row, wide, default-threshold, or SuiteSparse threshold
  families only after primary claim, expected ranks, threshold semantics,
  support tier, diagnostics, and failure interpretation are pinned.
- Add SuiteSparse rank-deficient QR corpus evidence only after independent
  expected-rank metadata, threshold semantics, support tier, diagnostics, skip
  behavior, runtime budget, and validation are explicit.
- Add additional SuiteSparse minimum-norm evidence only after extraction rule,
  shape, nnz, RHS, rank/nullity if claimed, residual/norm metrics, skip
  behavior, and support tier are pinned.
- Add optional-large SuiteSparse QR or minimum-norm evidence only through the
  optional-large gate with missing-data skip behavior and runtime/platform
  expectations recorded before default test registration.
- Add additional QR-vs-SVD minimum-norm fixtures only as bounded cross-checks
  with fixture keys, QR residual and norm metrics, SVD tolerance, and
  non-oracle wording per fixture.
- Add larger exact underdetermined minimum-norm lanes only for non-duplicate
  shapes with closed-form expected values and explicit residual, value, and
  norm tolerances.
- Revisit generic QR/SVD helper movement only with behavior-specific helper
  names, call-site tolerances, focused QR solve/COLAMD/SVD validation, and the
  full quality gate.
- Continue raw Q-column, wide economy, sparse-mode Q/economy, and SuiteSparse
  Q/economy follow-through through the Sprint 124 projector policy,
  Sprint 125-127 corpus support rules, and named output-shape semantics.

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

Not carried forward as unresolved Sprint 127 debt:

- Sprint 127 intake, residual dependency map, and duplicate fencing;
- compatible/wide residual semantics policy and evidence decision;
- nullspace/subspace expansion policy refresh;
- `qr_rankdef_dependent_row_4x3_nullspace_projector` implementation;
- threshold-family follow-through policy;
- `qr_rank_threshold_duplicate_5x4_perturbed_family` implementation;
- SuiteSparse rank-deficient QR corpus gate and explicit deferral;
- SuiteSparse and optional-large minimum-norm gate and explicit deferral;
- exact minimum-norm and QR-vs-SVD cross-check/helper gate;
- `qr_minnorm_3x6_exact_values` implementation;
- QR-vs-SVD minimum-norm explicit deferral;
- maintainer evidence and public claim gate;
- final validation package and Sprint 128 handoff evidence.

## Key Deliverables

- [PLAN.md](./PLAN.md)
- [WORKING_NOTES.md](./WORKING_NOTES.md)
- [day1-deferred-dedupe-baseline.md](./artifacts/day1-deferred-dedupe-baseline.md)
- [day2-compatible-wide-residual-semantics-policy.md](./artifacts/day2-compatible-wide-residual-semantics-policy.md)
- [day3-compatible-wide-residual-evidence-decision.md](./artifacts/day3-compatible-wide-residual-evidence-decision.md)
- [day4-nullspace-subspace-expansion-policy.md](./artifacts/day4-nullspace-subspace-expansion-policy.md)
- [day5-nullspace-subspace-evidence.md](./artifacts/day5-nullspace-subspace-evidence.md)
- [day6-threshold-family-follow-through-policy.md](./artifacts/day6-threshold-family-follow-through-policy.md)
- [day7-threshold-family-evidence.md](./artifacts/day7-threshold-family-evidence.md)
- [day8-suitesparse-rankdef-qr-corpus-gate.md](./artifacts/day8-suitesparse-rankdef-qr-corpus-gate.md)
- [day9-suitesparse-rankdef-qr-evidence-decision.md](./artifacts/day9-suitesparse-rankdef-qr-evidence-decision.md)
- [day10-suitesparse-optional-large-minnorm-gate.md](./artifacts/day10-suitesparse-optional-large-minnorm-gate.md)
- [day11-suitesparse-optional-large-minnorm-evidence-decision.md](./artifacts/day11-suitesparse-optional-large-minnorm-evidence-decision.md)
- [day12-exact-minnorm-crosscheck-helper-gate.md](./artifacts/day12-exact-minnorm-crosscheck-helper-gate.md)
- [day13-exact-minnorm-crosscheck-evidence.md](./artifacts/day13-exact-minnorm-crosscheck-evidence.md)
- [day14-validation-claim-gate-handoff.md](./artifacts/day14-validation-claim-gate-handoff.md)

## Final Status

Sprint 127 is complete. It converted Sprint 126's remaining rank-deficient QR
residual, nullspace/subspace, threshold-family, SuiteSparse corpus,
optional-large, minimum-norm, QR-vs-SVD, and helper debt into bounded
dependent-row nullspace projector, perturbed threshold-family, and exact 3 x 6
minimum-norm evidence lanes, explicit compatible/wide residual,
SuiteSparse/corpus, QR-vs-SVD, and helper future-owner packages, a validated
maintainer evidence refresh, and a stable Sprint 128 Q/economy/helper handoff.
