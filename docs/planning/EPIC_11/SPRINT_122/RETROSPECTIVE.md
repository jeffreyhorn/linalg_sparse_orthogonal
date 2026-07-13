# Sprint 122 Retrospective

**Sprint:** 122 - SVD/QR External Oracle Residual Follow-Through
**Duration:** 14 days
**Status:** Complete

## Definition Of Done Checklist

- [x] Created Sprint 122 day-by-day plan, working notes, and artifact
      directory.
- [x] Re-read Epic 11 Sprint 122 scope and Sprint 121 residual deferred debt.
- [x] Converted residual SVD, QR, partial-SVD, helper-ownership, and
      solver-selection items into explicit proof owners.
- [x] Rejected completed Sprint 121 audit, taxonomy, helper extraction, and
      first SVD external-reference work as duplicates.
- [x] Designed and implemented the bounded `svd_rankdef_duplicate_5x4`
      external dense-reference fixture.
- [x] Designed and implemented the bounded
      `qr_overdetermined_incompatible_4x2` external least-squares reference
      lane.
- [x] Designed and implemented the bounded `partial_svd_diag6_k2` external
      top-k singular-value lane.
- [x] Preserved minimum-norm helper ownership at the scenario level and
      deferred migration with explicit future boundaries.
- [x] Preserved Bidiagonal/Golub-Kahan ownership outside the general SVD helper
      layer and deferred extraction with explicit future boundaries.
- [x] Audited solver-selection and public/support wording against the sprint's
      evidence.
- [x] Published a no-update rationale for public solver-selection wording.
- [x] Preserved broad external parity, platform, package, ABI, performance,
      public API, and state-of-the-art non-claims.
- [x] Ran focused external-helper and test-owner checks for the new oracle
      lanes.
- [x] Ran the required full C quality gate after Sprint 122 `.c` and `.h`
      changes: `make format`, `make lint`, and `make test`.
- [x] Published validation, non-claim, residual, and closeout artifacts.
- [x] Finalized this retrospective and ran final diff hygiene.

## What Went Well

1. **Residual scope became concrete proof ownership.** Sprint 122 started from
   Sprint 121's deferred debt and converted each residual into a named owner,
   duplicate fence, dependency order, and promotion gate before implementation.

2. **External oracle coverage expanded without overclaiming.** The sprint added
   one bounded rank-deficient SVD fixture, one bounded QR least-squares fixture,
   and one bounded partial-SVD top-k fixture while preserving the distinction
   between fixture-local evidence and broad dense-library parity.

3. **Helper decisions favored visible numerical semantics.** Minimum-norm and
   Bidiagonal/Golub-Kahan checks stayed with their scenario owners because the
   current duplication still carries behavior-specific tolerance, fallback,
   transpose, reconstruction, and iteration meaning.

4. **Public wording stayed evidence-led.** The sprint audited README,
   solver-selection, tutorial, examples, benchmark, install, and maintainer
   surfaces, then chose not to expand public wording because the new oracle
   lanes were bounded validation, not support-level promotion.

5. **Validation was staged around actual change risk.** Full C quality ran
   after the last C/header change. Later documentation-only days used diff and
   whitespace checks while preserving the earlier full gate evidence.

## What Did Not Go Well

1. **The external oracle matrix is still intentionally narrow.** The new lanes
   improve SVD, QR, and partial-SVD evidence, but they do not cover broad
   matrix families, vector/subspace parity, LAPACK/NumPy/SciPy parity, or
   ecosystem-level correctness.

2. **Partial-SVD external semantics remain only partly proven.** Sprint 122
   added top-k singular-value evidence, but vector orientation, subspace
   angles, repeated spectra, clustered spectra, convergence budgets, and
   rectangular/rank-deficient behavior still need separate owners.

3. **Helper consolidation debt remains visible.** Deferring minimum-norm and
   Bidiagonal/Golub-Kahan migration was the right choice, but it means future
   maintainability work still needs behavior-specific extraction plans.

4. **Maintainer and public documentation did not gain a new evidence table.**
   The sprint produced claim gates and closeout evidence, but the maintainer
   guide and public solver-selection wording need a future update once the
   evidence is broad enough to justify it.

5. **Windows and package-surface implications remain non-claims.** The new
   lanes use existing helper/test owners and skip-policy boundaries, but Sprint
   122 did not expand reviewed platform, package, ABI, or CTest support.

## Final Metrics

### Validation

| Metric | Sprint 122 close state |
|---|---:|
| SVD rank-deficient external helper | passed, emitted 4 singular values |
| partial-SVD top-k external helper | passed, emitted 2 singular values |
| QR least-squares external helper | passed, emitted solution and residual |
| focused QR solve tests | 14 passed, 0 failed, 0 skipped |
| focused QR solve assertions | 1025 |
| focused SVD tests | 106 passed, 0 failed, 0 skipped |
| focused SVD assertions | 1729 |
| required full Make formatting | `make format` passed |
| required full Make lint | `make lint` passed |
| required full Make tests | `make test` passed |
| diff hygiene | `git diff --check` passed |
| trailing-whitespace scan | passed on Sprint 122 docs |
| Day 14 C quality rerun | not required; documentation-only closeout |

### Sprint Artifact Package

| Metric | Sprint 122 close state |
|---|---:|
| artifact files under `SPRINT_122/artifacts/` | 14 |
| sprint plan files | 1 |
| working notes files | 1 |
| retrospective files | 1 |
| new external-reference helper scripts | 1 |
| modified existing external-reference helper scripts | 1 |
| modified existing test files | 3 |
| modified existing test helper headers | 1 |
| Makefile/CMake/CTest registration changes | 0 |

## Movement And Claim Outcomes

| Area | Outcome |
|---|---|
| Residual owner map | Completed dependency-ordered owner map and duplicate fence for Sprint 121 residuals. |
| SVD external fixture | Added bounded `svd_rankdef_duplicate_5x4` singular-value evidence. |
| QR external lane | Added bounded `qr_overdetermined_incompatible_4x2` least-squares solution and residual evidence. |
| Partial-SVD external lane | Added bounded `partial_svd_diag6_k2` top-k singular-value evidence. |
| Minimum-norm helper ownership | Deferred migration; preserved QR/COLAMD/SVD-pinv/refinement/fallback/SuiteSparse scenario ownership. |
| Bidiagonal/Golub-Kahan ownership | Deferred generic consolidation; preserved specialized reconstruction and iteration semantics. |
| Solver-selection wording | No public wording expansion; future claim gates documented. |
| Build registration | Unchanged; no new test executable or library source was added. |
| Public API | Unchanged. |
| Public documentation claims | Unchanged; closeout records non-claims and no-update rationale. |
| Benchmarks/performance | Not claimed and not refreshed. |
| External-library parity | Not claimed. |

## Residual Deferred Debt

Most important carry-forward work:

- Broaden the SVD external fixture matrix only after fixture taxonomy,
  reference trust model, vector/rank/pseudoinverse/low-rank semantics,
  tolerance policy, skip handling, and failure interpretation are explicit.
- Add QR external compatible, rank-deficient, underdetermined/minimum-norm, and
  Q/economy evidence only behind behavior-specific fixtures and basis/tolerance
  rules.
- Expand partial-SVD external semantics beyond top-k singular values to vector,
  subspace, convergence-budget, repeated/clustered spectrum, and
  rectangular/rank-deficient behavior.
- Revisit minimum-norm helper migration with behavior-specific helper names and
  unchanged QR/COLAMD/SVD-pinv/refinement/fallback/SuiteSparse scenario
  ownership.
- Extract Bidiagonal/Golub-Kahan helpers only into a dedicated owner that
  preserves wide-transpose, implicit Householder reconstruction, explicit
  `U`/`V` reconstruction, and bidiagonal QR iteration semantics.
- Refresh maintainer evidence tables with Sprint 122 oracle lane ownership,
  trust boundaries, validation commands, and non-claims.
- Refresh public solver-selection wording only when broader evidence supports
  a user-facing claim.

Still consciously constrained rather than silently solved:

- no LAPACK parity claim;
- no SciPy or NumPy parity claim;
- no SuiteSparse, PETSc, Trilinos, Eigen, ARPACK, or vendor-backend parity
  claim;
- no broad external dense-library parity claim;
- no singular-vector, Q-basis, Ritz-vector, or subspace external parity claim;
- no broad partial-SVD convergence or vector/subspace claim;
- no broad low-rank or minimum-norm global optimality claim;
- no broad cross-solver equivalence or solver-superiority claim;
- no package-manager distribution claim;
- no shared-library or dynamic ABI stability claim;
- no equal Linux/macOS/Windows reviewed-support claim;
- no public API, install-header, package, CMake, Makefile, CI, or CTest
  expansion claim;
- no portable performance, scalability, memory, or state-of-the-art claim.

Not carried forward as unresolved Sprint 122 debt:

- Sprint 121 residual owner dedupe;
- additional bounded SVD external fixture decision and implementation;
- QR external dense-reference lane decision and implementation;
- partial-SVD external top-k lane decision and implementation;
- minimum-norm helper ownership decision;
- Bidiagonal/Golub-Kahan helper boundary decision;
- solver-selection claim inventory and no-update rationale;
- final validation package and closeout evidence.

## Key Deliverables

- [PLAN.md](./PLAN.md)
- [WORKING_NOTES.md](./WORKING_NOTES.md)
- [day1-sprint-intake.md](./artifacts/day1-sprint-intake.md)
- [day2-residual-owner-map.md](./artifacts/day2-residual-owner-map.md)
- [day3-svd-fixture-inventory.md](./artifacts/day3-svd-fixture-inventory.md)
- [day4-svd-fixture-decision.md](./artifacts/day4-svd-fixture-decision.md)
- [day5-qr-external-lane-requirements.md](./artifacts/day5-qr-external-lane-requirements.md)
- [day6-qr-external-lane-design.md](./artifacts/day6-qr-external-lane-design.md)
- [day7-partial-svd-semantics.md](./artifacts/day7-partial-svd-semantics.md)
- [day8-partial-svd-external-design.md](./artifacts/day8-partial-svd-external-design.md)
- [day9-minnorm-helper-ownership.md](./artifacts/day9-minnorm-helper-ownership.md)
- [day10-bidiag-gk-helper-boundary.md](./artifacts/day10-bidiag-gk-helper-boundary.md)
- [day11-solver-selection-claim-gate-inventory.md](./artifacts/day11-solver-selection-claim-gate-inventory.md)
- [day12-solver-selection-claim-gate-decision.md](./artifacts/day12-solver-selection-claim-gate-decision.md)
- [day13-validation-package.md](./artifacts/day13-validation-package.md)
- [day14-sprint-closeout.md](./artifacts/day14-sprint-closeout.md)

## Final Status

Sprint 122 is complete. It added bounded external oracle follow-through for
SVD, QR, and partial SVD, kept helper and public-claim semantics visible, and
left future work dependency-ordered for later Epic 11 sprints.
