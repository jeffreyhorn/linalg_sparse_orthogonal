# Sprint 125 Retrospective

**Sprint:** 125 - Rank-Deficient QR & Minimum-Norm Residual Evidence
**Duration:** 14 days
**Status:** Complete

## Definition Of Done Checklist

- [x] Created Sprint 125 day-by-day plan, working notes, and artifact
      directory.
- [x] Mapped Sprint 124 deferred rank-deficient QR and minimum-norm debt to
      existing Sprint 121-124 evidence without duplicating completed fixtures.
- [x] Defined the residual-only trust gate for rank-deficient QR evidence.
- [x] Implemented the bounded
      `qr_rankdef_duplicate_5x4_residual_only` external residual fixture.
- [x] Defined nullspace/subspace sign, ordering, nullity, projection, and
      fixture-local tolerance policy.
- [x] Implemented the bounded
      `qr_rankdef_duplicate_5x4_nullspace_projector` projector fixture.
- [x] Defined near-rank-deficient threshold-family policy and non-global
      rank-threshold boundaries.
- [x] Implemented the bounded `qr_rank_threshold_diag4_family` threshold-rank
      fixture.
- [x] Defined SuiteSparse rank-deficient QR corpus support-tier, skip,
      diagnostics, and non-claim policy.
- [x] Explicitly deferred SuiteSparse rank-deficient QR evidence because the
      checked-in QR corpus currently serves as full-rank controls.
- [x] Split QR minimum-norm work into behavior-specific owners across QR solve,
      COLAMD, fallback, rank-deficient, refinement, zero-row, QR-vs-SVD, and
      corpus scenarios.
- [x] Strengthened owner-local minimum-norm evidence in `tests/test_colamd.c`
      for COLAMD, fallback, rank-deficient, refinement, and zero-row behavior.
- [x] Accepted a bounded QR-vs-SVD-pseudoinverse cross-check without turning
      SVD into a global QR oracle.
- [x] Accepted one default checked-in `west0067` minimum-norm submatrix smoke.
- [x] Refreshed `docs/maintainer_guide.md` with accepted Sprint 125 QR
      evidence and preserved broad non-claims.
- [x] Audited public solver-selection, README, and public-header wording and
      made no public wording expansion.
- [x] Ran focused helper and owner-test checks for every accepted Sprint 125
      evidence lane.
- [x] Ran the required full C quality gate:
      `make format && make lint && make test`.
- [x] Published final validation, claim-gate, closeout, residual, non-claim,
      and Sprint 126 handoff artifacts.
- [x] Finalized this retrospective and ran final documentation hygiene.

## What Went Well

1. **Sprint 124 residual debt became explicit evidence or explicit deferral.**
   The sprint avoided reopening completed Sprint 124 policy and fixtures while
   adding new residual, nullspace, threshold, and minimum-norm lanes only where
   the proof boundary was clear.

2. **Rank-deficient QR stayed behavior-specific.** Residual-only evidence,
   nullspace projector evidence, and threshold-rank evidence were split into
   separate fixture keys and artifacts, preventing residual checks from
   becoming hidden nullspace or minimum-norm claims.

3. **The nullspace lane used projector semantics.** The accepted
   `qr_rankdef_duplicate_5x4_nullspace_projector` fixture compares projectors
   rather than raw basis vectors, preserving sign, orientation, and ordering
   boundaries.

4. **Threshold evidence remained fixture-local.** The diagonal ladder proves
   rank outcomes for named tolerances only. It does not become a public global
   rank-threshold policy.

5. **SuiteSparse QR rank-deficient evidence was not forced.** The sprint
   explicitly preserved `nos4`, `bcsstk04`, and `west0067` as full-rank
   controls rather than relabeling them as rank-deficient evidence.

6. **Minimum-norm evidence kept owners visible.** COLAMD, fallback,
   rank-deficient, refinement, zero-row, QR-vs-SVD, and `west0067` submatrix
   behavior stayed in named tests instead of being hidden behind generic
   helpers.

7. **The validation gate was strong.** Day 13 reran focused helper and
   executable checks plus the full `make format && make lint && make test`
   chain, then refreshed maintainer evidence without widening public claims.

## What Did Not Go Well

1. **The evidence package is still intentionally narrow.** Sprint 125 added
   useful named lanes, but broad QR, nullspace, minimum-norm, SuiteSparse,
   dense-library, platform, and performance parity remain non-claims.

2. **SuiteSparse rank-deficient QR remains deferred.** The checked-in corpus
   did not have a documented small rank-deficient QR fixture with pinned rank,
   threshold, nullity, residual semantics, and support-tier behavior.

3. **Nullspace evidence still covers only nullity one.** Multi-dimensional,
   wide-shape, dependent-row, near-threshold, and SuiteSparse subspace
   evidence still need projector or two-way projection metrics and promotion
   gates.

4. **Threshold evidence covers only the unscaled diagonal ladder.** Scaled,
   perturbed duplicate-column, dependent-row, wide, and corpus threshold
   families remain future work.

5. **Minimum-norm helper consolidation remains deferred.** This preserved
   behavior ownership, but future maintainability work still needs
   behavior-specific helper names and validation.

6. **Public docs did not gain a broader user-facing claim.** This was the
   correct result, but it means the new evidence is primarily maintainer and
   regression evidence until future sprints broaden corpus and claim surfaces.

## Final Metrics

### Validation

| Metric | Sprint 125 close state |
|---|---:|
| QR residual-only helper | passed, emitted `OK 1` and residual `3.7886027630095733` |
| QR nullspace-projector helper | passed, emitted `OK 20` |
| QR threshold-family helper | passed, emitted `OK 6` and ranks `3`, `2`, `1` |
| focused QR tests | 68 passed, 0 failed, 0 skipped |
| focused QR assertions | 669 |
| focused QR solve tests | 18 passed, 0 failed, 0 skipped |
| focused QR solve assertions | 1089 |
| focused COLAMD/minimum-norm tests | 70 passed, 0 failed, 0 skipped |
| focused COLAMD/minimum-norm assertions | 299 |
| focused SVD tests | 109 passed, 0 failed, 0 skipped |
| focused SVD assertions | 1802 |
| required full Make formatting | `make format` passed |
| required full Make lint | `make lint` passed |
| required full Make tests | `make test` passed |
| full Make test final result | `All tests passed.` |
| diff hygiene | `git diff --check` passed |
| trailing-whitespace scan | passed on Sprint 125 docs and touched files |
| Day 14 C quality rerun | not required; documentation-only closeout |

### Sprint Artifact Package

| Metric | Sprint 125 close state |
|---|---:|
| artifact files under `SPRINT_125/artifacts/` | 14 |
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
| Sprint intake | Completed deferred QR/minimum-norm dedupe map, duplicate fence, validation boundary, and day-level owner map. |
| Rank-deficient QR residual evidence | Added bounded `qr_rankdef_duplicate_5x4_residual_only` residual-only evidence. |
| Nullspace/subspace policy | Defined sign, ordering, nullity, projector/subspace, tolerance, and non-claim rules. |
| Rank-deficient QR nullspace evidence | Added bounded `qr_rankdef_duplicate_5x4_nullspace_projector` projector evidence. |
| Near-rank-deficient threshold evidence | Added bounded `qr_rank_threshold_diag4_family` threshold-rank evidence. |
| SuiteSparse rank-deficient QR evidence | Deferred; current checked-in QR SuiteSparse matrices remain full-rank controls. |
| Minimum-norm owner map | Split QR solve, COLAMD, fallback, rank-deficient, refinement, zero-row, QR-vs-SVD, and SuiteSparse behavior owners. |
| Core QR minimum-norm evidence | Strengthened COLAMD, fallback, rank-deficient, refinement, and zero-row owner-local tests. |
| QR-vs-SVD minimum-norm evidence | Accepted one bounded cross-check; SVD pseudoinverse remains non-global. |
| SuiteSparse minimum-norm evidence | Accepted one default checked-in `west0067` 30 x 67 submatrix smoke. |
| Maintainer evidence | Updated `docs/maintainer_guide.md` with Sprint 125 QR evidence and non-claims. |
| Public docs | No public wording expansion; README, solver-selection, and public headers remain unchanged for Sprint 125. |
| Public API | Unchanged. |
| Build registration | Unchanged; no new executable, library source, Makefile entry, CMake entry, or CTest member was added. |
| External-library parity | Not claimed. |

## Residual Deferred Debt

Most important carry-forward work:

- Add compatible zero-residual, dependent-row, and wide rank-deficient QR
  residual fixtures only after proving distinct trust value and preserving
  nullspace/minimum-norm non-claims.
- Add multi-dimensional, wide-shape, near-threshold, dependent-row, and
  SuiteSparse nullspace/subspace evidence only with projector or two-way
  projection metrics and pinned rank/nullity metadata.
- Add scaled diagonal, perturbed duplicate-column, dependent-row, wide, and
  SuiteSparse threshold families only with fixture-local expected ranks,
  diagnostics, and non-global interpretation.
- Add SuiteSparse rank-deficient QR corpus evidence only after expected-rank
  metadata, support tier, diagnostics, skip behavior, and validation are
  explicit.
- Add optional-large SuiteSparse and rank-deficient SuiteSparse minimum-norm
  evidence only after support-tier, residual, norm, rank, nullity, and corpus
  metadata are pinned.
- Add additional QR-vs-SVD minimum-norm fixtures only as bounded cross-checks
  with explicit fixture keys, tolerances, and non-oracle wording.
- Add larger underdetermined minimum-norm expected-value lanes only after
  deciding which existing shape controls deserve exact-value contracts.
- Revisit generic QR/SVD minimum-norm helper movement only with
  behavior-specific names and focused QR solve, COLAMD, SVD, and full quality
  validation.
- Continue raw Q-column, wide economy, sparse-mode Q/economy, and SuiteSparse
  Q/economy follow-through only through the Sprint 124 projector policy and
  Sprint 125 corpus support rules.

Still consciously constrained rather than silently solved:

- no LAPACK parity claim;
- no SciPy or NumPy parity claim;
- no BLAS, SuiteSparse, PETSc, Trilinos, Eigen, ARPACK, or vendor-backend
  parity claim;
- no broad external dense-library or ecosystem parity claim;
- no broad QR factorization, QR solve, rank-deficient solve, nullspace,
  minimum-norm, Q-basis, economy, sparse-mode, reorder, backend, corpus, or
  performance parity claim;
- no global QR rank-threshold policy;
- no raw Q-basis equality, Q-sign, Q-orientation, unique-basis, raw nullspace
  basis, or subspace external parity claim;
- no broad SVD-pseudoinverse oracle claim;
- no broad SuiteSparse corpus, optional-data, platform, or performance claim;
- no generic helper API or helper consolidation claim;
- no package-manager distribution claim;
- no shared-library or dynamic ABI stability claim;
- no equal Linux/macOS/Windows reviewed-support claim;
- no public API, install-header, package, CMake, Makefile, CI, or CTest
  expansion claim;
- no portable performance, scalability, memory, or state-of-the-art claim.

Not carried forward as unresolved Sprint 125 debt:

- Sprint 125 intake, residual dependency map, and duplicate fencing;
- residual-only rank-deficient QR trust gate;
- `qr_rankdef_duplicate_5x4_residual_only` implementation;
- nullspace/subspace policy design;
- `qr_rankdef_duplicate_5x4_nullspace_projector` implementation;
- near-rank-deficient threshold-family policy;
- `qr_rank_threshold_diag4_family` implementation;
- SuiteSparse rank-deficient QR corpus policy and explicit deferral;
- QR minimum-norm behavior owner map;
- COLAMD, fallback, rank-deficient, refinement, and zero-row minimum-norm
  owner-local evidence;
- QR-vs-SVD-pseudoinverse bounded cross-check decision;
- `west0067` minimum-norm submatrix smoke;
- maintainer evidence and public claim gate;
- final validation package and closeout evidence.

## Key Deliverables

- [PLAN.md](./PLAN.md)
- [WORKING_NOTES.md](./WORKING_NOTES.md)
- [day1-deferred-qr-dedupe.md](./artifacts/day1-deferred-qr-dedupe.md)
- [day2-residual-trust-gate.md](./artifacts/day2-residual-trust-gate.md)
- [day3-rankdef-residual-evidence.md](./artifacts/day3-rankdef-residual-evidence.md)
- [day4-nullspace-subspace-policy.md](./artifacts/day4-nullspace-subspace-policy.md)
- [day5-nullspace-subspace-decision.md](./artifacts/day5-nullspace-subspace-decision.md)
- [day6-near-rank-threshold-families.md](./artifacts/day6-near-rank-threshold-families.md)
- [day7-near-rank-threshold-evidence.md](./artifacts/day7-near-rank-threshold-evidence.md)
- [day8-suitesparse-rankdef-qr-policy.md](./artifacts/day8-suitesparse-rankdef-qr-policy.md)
- [day9-suitesparse-rankdef-qr-decision.md](./artifacts/day9-suitesparse-rankdef-qr-decision.md)
- [day10-minnorm-behavior-owner-map.md](./artifacts/day10-minnorm-behavior-owner-map.md)
- [day11-core-minnorm-evidence.md](./artifacts/day11-core-minnorm-evidence.md)
- [day12-oracle-corpus-minnorm-decision.md](./artifacts/day12-oracle-corpus-minnorm-decision.md)
- [day13-validation-claim-gate.md](./artifacts/day13-validation-claim-gate.md)
- [day14-closeout-and-handoff.md](./artifacts/day14-closeout-and-handoff.md)

## Final Status

Sprint 125 is complete. It converted Sprint 124's rank-deficient QR and
minimum-norm residual deferred debt into bounded residual, nullspace,
threshold, minimum-norm, QR-vs-SVD, and corpus-smoke evidence lanes, explicit
future-owner packages, a validated maintainer evidence and claim gate, and a
stable Sprint 126 handoff.
