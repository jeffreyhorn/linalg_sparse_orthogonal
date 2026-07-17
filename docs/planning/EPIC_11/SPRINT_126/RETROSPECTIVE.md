# Sprint 126 Retrospective

**Sprint:** 126 - Rank-Deficient QR Residual Corpus & Minimum-Norm Follow-Through
**Duration:** 14 days
**Status:** Complete

## Definition Of Done Checklist

- [x] Created Sprint 126 day-by-day plan, working notes, and artifact
      directory.
- [x] Mapped Sprint 125 deferred residual debt to existing Sprint 121-125 QR,
      nullspace, threshold, SuiteSparse, minimum-norm, QR-vs-SVD, Q/economy,
      and helper evidence without duplicating completed fixtures.
- [x] Defined compatible, dependent-row, and wide residual fixture trust
      policy with explicit residual-only proof boundaries.
- [x] Implemented the bounded
      `qr_rankdef_dependent_row_4x3_residual_only` external residual fixture.
- [x] Refreshed nullspace/subspace policy for multi-dimensional, wide,
      dependent-row, near-threshold, and SuiteSparse candidates.
- [x] Implemented the bounded `qr_rank1_4x3_nullspace_projector` projector
      fixture.
- [x] Refreshed threshold-family policy for scaled, perturbed, dependent-row,
      wide, default-threshold, and SuiteSparse candidates.
- [x] Implemented the bounded `qr_rank_threshold_diag4_scaled_family`
      threshold-rank fixture.
- [x] Re-ran the SuiteSparse rank-deficient QR corpus gate and explicitly
      deferred additional corpus evidence until expected-rank metadata,
      threshold semantics, support tier, diagnostics, skip behavior, runtime,
      and validation are pinned.
- [x] Re-ran the SuiteSparse minimum-norm corpus gate and preserved the Sprint
      125 `west0067` 30 x 67 submatrix smoke as the only accepted default
      checked-in SuiteSparse minimum-norm corpus evidence.
- [x] Accepted the 5 x 10 underdetermined exact-value minimum-norm lane with
      closed-form solution values and `sqrt(11)` norm.
- [x] Implemented `qr_minnorm_5x10_exact_values` in the owner-local COLAMD
      test surface.
- [x] Explicitly deferred additional QR-vs-SVD minimum-norm cross-checks so
      SVD pseudoinverse remains a bounded cross-check, not a global QR oracle.
- [x] Refreshed `docs/maintainer_guide.md` with accepted Sprint 126 QR
      evidence and preserved broad non-claims.
- [x] Audited public solver-selection, README, public headers, package
      metadata, and API wording and made no public wording expansion.
- [x] Ran focused helper and owner-test checks for every accepted Sprint 126
      evidence lane.
- [x] Ran the required full C quality gate:
      `make format && make lint && make test`.
- [x] Published final validation, claim-gate, closeout, residual, non-claim,
      and Sprint 127 handoff artifacts.
- [x] Finalized this retrospective and ran final documentation hygiene.

## What Went Well

1. **Sprint 125 debt stayed deduplicated.** Sprint 126 reused the completed
   Sprint 125 residual-only, duplicate-column nullspace, unscaled threshold,
   minimum-norm, QR-vs-SVD, and `west0067` lanes as baselines instead of
   relabeling them as new evidence.

2. **Residual-only evidence added a second structural family.** The accepted
   dependent-row 4 x 3 fixture adds a non-zero residual comparison against a
   standard-library projection reference while avoiding solution-vector,
   rank, nullspace, and minimum-norm assertions.

3. **Nullspace evidence expanded to nullity two without raw-basis claims.**
   The `qr_rank1_4x3_nullspace_projector` lane uses projector comparison and
   local orthonormalization, preserving sign, orientation, ordering, and
   unique-basis boundaries.

4. **Threshold evidence now covers scale without becoming a global policy.**
   The scaled diagonal ladder checks three scale values and three relative
   thresholds while keeping the result fixture-local and diagnostic-rich.

5. **SuiteSparse corpus gates stayed honest.** The sprint did not force
   rank-deficient or minimum-norm SuiteSparse evidence out of matrices that
   lack independent rank, extraction, residual, norm, support-tier, or skip
   metadata.

6. **The underdetermined minimum-norm lane gained exact values.** The 5 x 10
   fixture now asserts a closed-form solution vector, residual, and exact
   `sqrt(11)` norm instead of only relying on residual smoke coverage.

7. **The claim gate remained bounded.** Maintainer evidence was refreshed for
   the new Sprint 126 fixtures, while public solver-selection, README,
   headers, package metadata, and API wording stayed unchanged.

## What Did Not Go Well

1. **The evidence package remains intentionally fixture-scoped.** Sprint 126
   strengthened several named lanes, but broad QR, nullspace, minimum-norm,
   SuiteSparse, dense-library, platform, and performance parity remain
   non-claims.

2. **Compatible and wide residual evidence remain deferred.** Compatible
   zero-residual behavior still needs proof that it adds trust beyond existing
   deterministic compatible solve tests. Wide residual-only behavior still
   depends on underdetermined output and solution-selection semantics.

3. **SuiteSparse rank-deficient QR evidence remains blocked on metadata.**
   The checked-in QR SuiteSparse matrices continue to serve as full-rank
   controls, and optional/report-only matrices still lack support-tier and
   expected-rank proof.

4. **SuiteSparse minimum-norm expansion remains blocked.** Additional
   `west0067`, `steam1`, optional-large, report-only, and rank-deficient
   corpus candidates still need extraction, rank/nullity, residual, norm,
   skip, and support-tier metadata before registration.

5. **QR-vs-SVD minimum-norm coverage did not expand.** This preserved the
   non-oracle boundary, but broader cross-check confidence remains future
   work with fixture-specific wording and tolerances.

6. **Helper consolidation remains deferred.** The sprint intentionally kept
   behavior-specific owners visible, but Sprint 127 still needs a careful
   helper ownership decision before Q/economy and cross-check work grows.

## Final Metrics

### Validation

| Metric | Sprint 126 close state |
|---|---:|
| dependent-row residual helper | passed, emitted `OK 1` and residual `4.2840332837724997` |
| multi-dimensional nullspace helper | passed, emitted `OK 13` |
| scaled threshold-family helper | passed, emitted `OK 27` |
| focused QR tests | 70 passed, 0 failed, 0 skipped |
| focused QR assertions | 760 |
| focused QR solve tests | 19 passed, 0 failed, 0 skipped |
| focused QR solve assertions | 1104 |
| focused COLAMD/minimum-norm tests | 70 passed, 0 failed, 0 skipped |
| focused COLAMD/minimum-norm assertions | 310 |
| required full Make formatting | `make format` passed |
| required full Make lint | `make lint` passed |
| required full Make tests | `make test` passed |
| full Make test final result | `All tests passed.` |
| diff hygiene | `git diff --check` passed |
| trailing-whitespace scan | passed on Sprint 126 docs and touched files |
| Python cache scan | no `__pycache__` or `.pyc` artifacts found |
| public docs wording expansion | 0 |

### Sprint Artifact Package

| Metric | Sprint 126 close state |
|---|---:|
| artifact files under `SPRINT_126/artifacts/` | 14 |
| sprint plan files | 1 |
| working notes files | 1 |
| retrospective files | 1 |
| modified external-reference helper scripts | 1 |
| modified existing test files | 3 |
| maintainer guide updates | 1 |
| public solver-selection wording updates | 0 |
| README/public-header wording updates | 0 |
| Makefile/CMake/CTest registration changes | 0 |

## Movement And Claim Outcomes

| Area | Outcome |
|---|---|
| Sprint intake | Completed Sprint 125 residual dedupe map, duplicate fence, validation boundary, and day-level owner map. |
| Compatible and wide residual policy | Defined trust rules and deferred compatible zero-residual and wide residual-only lanes until promotion metadata is explicit. |
| Dependent-row residual evidence | Added bounded `qr_rankdef_dependent_row_4x3_residual_only` residual-only evidence. |
| Nullspace/subspace policy | Refreshed projector, two-way projection, rank/nullity, tolerance, and raw-basis non-claim rules for expanded candidate families. |
| Multi-dimensional nullspace evidence | Added bounded `qr_rank1_4x3_nullspace_projector` projector evidence. |
| Threshold-family policy | Refreshed scaled, perturbed, dependent-row, wide, default-threshold, and SuiteSparse threshold gates. |
| Scaled threshold evidence | Added bounded `qr_rank_threshold_diag4_scaled_family` rank-threshold evidence. |
| SuiteSparse rank-deficient QR evidence | Deferred; current checked-in QR SuiteSparse matrices remain full-rank controls. |
| SuiteSparse minimum-norm evidence | Deferred additional corpus lanes; Sprint 125 `west0067` submatrix smoke remains the default checked-in baseline. |
| Underdetermined minimum-norm evidence | Added bounded `qr_minnorm_5x10_exact_values` exact-value and exact-norm evidence. |
| QR-vs-SVD minimum-norm evidence | Deferred additional cross-checks; Sprint 125 2 x 4 cross-check remains the bounded baseline. |
| Maintainer evidence | Updated `docs/maintainer_guide.md` with Sprint 126 QR evidence and non-claims. |
| Public docs | No public wording expansion; README, solver-selection, public headers, package metadata, and API wording remain unchanged for Sprint 126. |
| Public API | Unchanged. |
| Build registration | Unchanged; no new executable, library source, Makefile entry, CMake entry, or CTest member was added. |
| External-library parity | Not claimed. |

## Residual Deferred Debt

Most important carry-forward work:

- Add compatible zero-residual rank-deficient QR residual evidence only after
  proving the zero-residual lane adds trust beyond deterministic compatible
  solve behavior and cannot be misread as minimum-norm evidence.
- Add wide residual-only QR evidence only after Sprint 127 or a later owner
  defines underdetermined output semantics, solution-selection policy,
  Q/economy boundaries, and residual-only proof value.
- Add wide-shape nullspace/subspace evidence only after rank/nullity,
  projection metric, tolerance, and sparse/economy output semantics are pinned.
- Add dependent-row, near-threshold, or SuiteSparse nullspace/subspace
  evidence only after projector or two-way projection residual metrics,
  expected rank/nullity, threshold semantics, and support tier are explicit.
- Add perturbed duplicate-column threshold evidence only after perturbation
  sizes are separated from thresholds by at least two orders of magnitude and
  default-threshold claims remain fenced.
- Add dependent-row, wide, default-threshold, or SuiteSparse threshold families
  only after primary claim, expected ranks, threshold semantics, support tier,
  diagnostics, and failure interpretation are pinned.
- Add SuiteSparse rank-deficient QR corpus evidence only after expected-rank
  metadata, threshold semantics, support tier, diagnostics, skip behavior,
  runtime budget, and validation are explicit.
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
  Sprint 125/126 corpus support rules, and named output-shape semantics.

Still consciously constrained rather than silently solved:

- no LAPACK parity claim;
- no SciPy or NumPy parity claim;
- no BLAS, SuiteSparse, PETSc, Trilinos, Eigen, ARPACK, or vendor-backend
  parity claim;
- no broad external dense-library, external package, or ecosystem parity
  claim;
- no broad QR factorization, QR solve, compatible solve, wide solve,
  rank-deficient solve, nullspace, minimum-norm, Q-basis, economy,
  sparse-mode, reorder, backend, corpus, or performance parity claim;
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

Not carried forward as unresolved Sprint 126 debt:

- Sprint 126 intake, residual dependency map, and duplicate fencing;
- compatible/dependent-row/wide residual fixture trust policy;
- `qr_rankdef_dependent_row_4x3_residual_only` implementation;
- expanded nullspace/subspace policy refresh;
- `qr_rank1_4x3_nullspace_projector` implementation;
- threshold-family expansion policy;
- `qr_rank_threshold_diag4_scaled_family` implementation;
- SuiteSparse rank-deficient QR corpus gate and explicit deferral;
- SuiteSparse minimum-norm corpus gate and explicit deferral;
- larger underdetermined exact-value and QR-vs-SVD cross-check gate;
- `qr_minnorm_5x10_exact_values` implementation;
- QR-vs-SVD minimum-norm explicit deferral;
- maintainer evidence and public claim gate;
- final validation package and Sprint 127 handoff evidence.

## Key Deliverables

- [PLAN.md](./PLAN.md)
- [WORKING_NOTES.md](./WORKING_NOTES.md)
- [day1-residual-dedupe-baseline.md](./artifacts/day1-residual-dedupe-baseline.md)
- [day2-residual-fixture-trust-policy.md](./artifacts/day2-residual-fixture-trust-policy.md)
- [day3-residual-evidence.md](./artifacts/day3-residual-evidence.md)
- [day4-nullspace-subspace-expansion-policy.md](./artifacts/day4-nullspace-subspace-expansion-policy.md)
- [day5-nullspace-subspace-evidence.md](./artifacts/day5-nullspace-subspace-evidence.md)
- [day6-threshold-family-expansion-policy.md](./artifacts/day6-threshold-family-expansion-policy.md)
- [day7-threshold-family-evidence.md](./artifacts/day7-threshold-family-evidence.md)
- [day8-suitesparse-rankdef-qr-corpus-gate.md](./artifacts/day8-suitesparse-rankdef-qr-corpus-gate.md)
- [day9-suitesparse-rankdef-qr-evidence-decision.md](./artifacts/day9-suitesparse-rankdef-qr-evidence-decision.md)
- [day10-suitesparse-minnorm-corpus-gate.md](./artifacts/day10-suitesparse-minnorm-corpus-gate.md)
- [day11-suitesparse-minnorm-evidence-decision.md](./artifacts/day11-suitesparse-minnorm-evidence-decision.md)
- [day12-underdetermined-crosscheck-gate.md](./artifacts/day12-underdetermined-crosscheck-gate.md)
- [day13-underdetermined-minnorm-evidence.md](./artifacts/day13-underdetermined-minnorm-evidence.md)
- [day14-validation-claim-gate-handoff.md](./artifacts/day14-validation-claim-gate-handoff.md)

## Final Status

Sprint 126 is complete. It converted Sprint 125's remaining
rank-deficient QR, nullspace, threshold, SuiteSparse, and minimum-norm
residual debt into bounded dependent-row residual, multi-dimensional
nullspace projector, scaled threshold-family, and exact 5 x 10 minimum-norm
evidence lanes, explicit SuiteSparse and QR-vs-SVD future-owner packages, a
validated maintainer evidence refresh, and a stable Sprint 127
Q/economy/helper handoff.
