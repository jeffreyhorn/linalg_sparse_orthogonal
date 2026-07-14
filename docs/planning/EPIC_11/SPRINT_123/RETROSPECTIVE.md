# Sprint 123 Retrospective

**Sprint:** 123 - Residual SVD/QR Oracle, Helper & Claim Evidence Follow-Through
**Duration:** 14 days
**Status:** Complete

## Definition Of Done Checklist

- [x] Created Sprint 123 day-by-day plan, working notes, and artifact
      directory.
- [x] Re-read Epic 11 Sprint 123 scope, Sprint 122 residual deferred debt, and
      Sprint 121 taxonomy inputs.
- [x] Established duplicate fences around completed Sprint 122 SVD, QR, and
      partial-SVD oracle lanes.
- [x] Defined the Sprint 123 SVD external-reference trust model, fixture
      taxonomy, candidate criteria, and non-claims.
- [x] Implemented the bounded `svd_wide_fullrank_4x6` full-SVD external
      singular-value fixture.
- [x] Defined QR compatible, rank-deficient, minimum-norm, and Q/economy
      external evidence requirements and deferral boundaries.
- [x] Implemented the bounded `qr_overdetermined_compatible_5x3` external
      least-squares fixture.
- [x] Defined partial-SVD value, vector, subspace, repeated-spectrum,
      convergence-budget, rank-deficient, and low-rank semantics.
- [x] Implemented the bounded `partial_svd_tall_diag_8x5_k3` external top-k
      singular-value fixture.
- [x] Revisited minimum-norm helper migration and explicitly deferred generic
      migration to preserve behavior-specific ownership.
- [x] Revisited Bidiagonal/Golub-Kahan helper extraction and explicitly
      deferred generic extraction to preserve specialized semantics.
- [x] Refreshed `docs/maintainer_guide.md` evidence tables for Sprint 121-123
      QR, SVD, and partial-SVD oracle ownership and non-claims.
- [x] Audited public/support wording and published a no-update rationale for
      `docs/solver_selection.md`.
- [x] Published final non-claim, residual deferred debt, validation, and
      closeout artifacts.
- [x] Ran focused helper and owner-test checks for the new oracle lanes.
- [x] Ran the required full C quality gate after Sprint 123 `.c` and `.h`
      changes: `make format`, `make lint`, and `make test`.
- [x] Finalized this retrospective and ran final diff hygiene.

## What Went Well

1. **Residual debt turned into bounded proof.** Sprint 123 started from the
   Sprint 122 residual queue and converted the lowest-risk SVD, QR, and
   partial-SVD candidates into explicit fixture contracts before implementation.

2. **The new oracle lanes stayed narrow.** The sprint added one wide SVD
   fixture, one compatible QR least-squares fixture, and one tall partial-SVD
   fixture without expanding Makefile, CMake, CTest, public API, package,
   platform, ABI, performance, or broad external parity claims.

3. **Helper ownership stayed behavior-specific.** Minimum-norm and
   Bidiagonal/Golub-Kahan consolidation were evaluated, but the sprint chose
   explicit deferral because the current tests still encode scenario-local
   tolerance, fallback, transpose, reconstruction, and iteration semantics.

4. **Maintainer evidence caught up with implementation.** The maintainer guide
   now maps the bounded Sprint 121-123 QR, SVD, and partial-SVD helper lanes to
   their family-local owners and non-claims instead of saying those lanes do
   not exist.

5. **Public wording remained evidence-led.** The solver-selection guide already
   described user workflows without claiming external parity. The sprint
   documented a no-update rationale instead of widening user-facing claims.

## What Did Not Go Well

1. **The oracle matrix remains intentionally sparse.** Sprint 123 improved
   confidence on three named fixtures, but the library still lacks broad SVD,
   QR, partial-SVD, vector/subspace, package, platform, and dense-library
   comparison coverage.

2. **Rank-deficient QR remains deferred.** The sprint rejected a quick
   rank-deficient QR external lane because it would need explicit
   rank-threshold, nullspace, pseudoinverse, and minimum-norm semantics before
   helper-backed comparison is meaningful.

3. **Partial-SVD evidence remains value-only.** The new tall fixture proves
   top-k singular values for one case, but vector orientation, subspace angles,
   repeated or clustered spectra, convergence budgets, rank-deficient behavior,
   and low-rank optimality still need separate proof owners.

4. **Helper maintainability debt remains visible.** Deferring minimum-norm and
   Bidiagonal/Golub-Kahan helper movement avoided hiding semantics, but future
   maintainability work still needs dedicated behavior-specific extraction.

5. **Public docs did not gain a new user-facing claim.** This was the right
   outcome, but it means adoption-facing wording still cannot advertise the new
   oracle lanes beyond maintainer evidence.

## Final Metrics

### Validation

| Metric | Sprint 123 close state |
|---|---:|
| SVD wide external helper | passed, emitted 4 singular values |
| QR compatible external helper | passed, emitted 3 solution entries and 1 residual |
| partial-SVD tall external helper | passed, emitted 3 singular values |
| focused QR solve tests | 15 passed, 0 failed, 0 skipped |
| focused QR solve assertions | 1042 |
| focused SVD tests after wide fixture | 107 passed, 0 failed, 0 skipped |
| focused SVD assertions after wide fixture | 1755 |
| focused SVD tests after partial-SVD fixture | 108 passed, 0 failed, 0 skipped |
| focused SVD assertions after partial-SVD fixture | 1769 |
| required full Make formatting | `make format` passed |
| required full Make lint | `make lint` passed |
| required full Make tests | `make test` passed |
| diff hygiene | `git diff --check` passed |
| trailing-whitespace scan | passed on Sprint 123 docs and touched files |
| Day 14 C quality rerun | not required; documentation-only closeout |

### Sprint Artifact Package

| Metric | Sprint 123 close state |
|---|---:|
| artifact files under `SPRINT_123/artifacts/` | 14 |
| sprint plan files | 1 |
| working notes files | 1 |
| retrospective files | 1 |
| modified external-reference helper scripts | 2 |
| modified existing test files | 2 |
| modified existing test helper headers | 1 |
| maintainer guide updates | 1 |
| solver-selection public wording updates | 0 |
| Makefile/CMake/CTest registration changes | 0 |

## Movement And Claim Outcomes

| Area | Outcome |
|---|---|
| Sprint intake | Completed residual map, duplicate fence, validation boundary, and day-level owner map. |
| SVD external fixture | Added bounded `svd_wide_fullrank_4x6` singular-value evidence. |
| QR external lane | Added bounded `qr_overdetermined_compatible_5x3` compatible least-squares solution and residual evidence. |
| Partial-SVD external lane | Added bounded `partial_svd_tall_diag_8x5_k3` top-k singular-value evidence. |
| Rank-deficient QR evidence | Deferred until rank-threshold, nullspace, pseudoinverse, and minimum-norm semantics are explicit. |
| QR minimum-norm evidence | Deferred to a future QR solve / minimum-norm oracle owner. |
| QR Q/economy evidence | Deferred to a future QR basis/economy owner. |
| Partial-SVD vector/subspace evidence | Deferred to a future partial-SVD semantic owner. |
| Minimum-norm helper migration | Deferred; preserved QR/COLAMD/SVD-pinv/refinement/fallback/SuiteSparse scenario ownership. |
| Bidiagonal/Golub-Kahan helper extraction | Deferred; preserved wide transpose, Householder reconstruction, explicit `U`/`V`, wide GK skip, and bidiagonal QR iteration ownership. |
| Maintainer evidence | Refreshed `docs/maintainer_guide.md` with bounded owners, trust boundaries, validation commands, and non-claims. |
| Solver-selection wording | No public wording expansion; no-update rationale published. |
| Public API | Unchanged. |
| Build registration | Unchanged; no new test executable or library source was added. |
| External-library parity | Not claimed. |

## Residual Deferred Debt

Most important carry-forward work:

- Design rank-deficient QR external oracle evidence only after explicit
  rank-threshold, nullspace, pseudoinverse, and minimum-norm policy is chosen.
- Add QR minimum-norm external oracle evidence only under a behavior-specific
  owner spanning QR solve, COLAMD, SVD-pseudoinverse, fallback, refinement, and
  SuiteSparse paths.
- Add QR Q-basis and economy external evidence only after sign, orientation,
  projection, subspace, and economy-shape semantics are defined.
- Expand partial-SVD external semantics beyond top-k values to vector,
  subspace, repeated-spectrum, clustered-spectrum, rank-deficient,
  convergence-budget, and low-rank optimality behavior.
- Revisit minimum-norm helper migration with behavior-specific helper names and
  promotion gates that keep scenario-local assertions visible.
- Extract Bidiagonal/Golub-Kahan helpers only into a dedicated owner that
  preserves wide transpose, implicit Householder reconstruction, explicit
  `U`/`V` reconstruction, wide GK skips, and bidiagonal QR iteration semantics.
- Refresh public solver-selection wording only when future evidence supports a
  user-facing claim beyond the current workflow guidance.

Still consciously constrained rather than silently solved:

- no LAPACK parity claim;
- no SciPy or NumPy parity claim;
- no SuiteSparse, PETSc, Trilinos, Eigen, ARPACK, or vendor-backend parity
  claim;
- no broad external dense-library parity claim;
- no singular-vector, Q-basis, Ritz-vector, or subspace external parity claim;
- no rank-deficient QR external oracle claim;
- no broad QR minimum-norm external oracle claim;
- no broad partial-SVD convergence or vector/subspace claim;
- no broad low-rank or minimum-norm global optimality claim;
- no broad cross-solver equivalence or solver-superiority claim;
- no package-manager distribution claim;
- no shared-library or dynamic ABI stability claim;
- no equal Linux/macOS/Windows reviewed-support claim;
- no public API, install-header, package, CMake, Makefile, CI, or CTest
  expansion claim;
- no portable performance, scalability, memory, or state-of-the-art claim.

Not carried forward as unresolved Sprint 123 debt:

- Sprint 123 intake and duplicate fencing;
- SVD fixture taxonomy and trust-model decision;
- `svd_wide_fullrank_4x6` implementation;
- QR compatible fixture decision and implementation;
- partial-SVD top-k tall fixture decision and implementation;
- minimum-norm helper migration decision;
- Bidiagonal/Golub-Kahan helper extraction decision;
- maintainer evidence table refresh;
- solver-selection claim no-update rationale;
- final validation package and closeout evidence.

## Key Deliverables

- [PLAN.md](./PLAN.md)
- [WORKING_NOTES.md](./WORKING_NOTES.md)
- [day1-sprint-intake.md](./artifacts/day1-sprint-intake.md)
- [day2-svd-fixture-taxonomy-trust-model.md](./artifacts/day2-svd-fixture-taxonomy-trust-model.md)
- [day3-svd-external-fixture-batch-decision.md](./artifacts/day3-svd-external-fixture-batch-decision.md)
- [day4-svd-fixture-implementation.md](./artifacts/day4-svd-fixture-implementation.md)
- [day5-qr-external-behavior-requirements.md](./artifacts/day5-qr-external-behavior-requirements.md)
- [day6-qr-compatible-rankdef-decision.md](./artifacts/day6-qr-compatible-rankdef-decision.md)
- [day7-qr-minnorm-q-economy-decision.md](./artifacts/day7-qr-minnorm-q-economy-decision.md)
- [day8-qr-evidence-implementation.md](./artifacts/day8-qr-evidence-implementation.md)
- [day9-partial-svd-external-semantics-design.md](./artifacts/day9-partial-svd-external-semantics-design.md)
- [day10-partial-svd-evidence-package.md](./artifacts/day10-partial-svd-evidence-package.md)
- [day11-minnorm-helper-migration-decision.md](./artifacts/day11-minnorm-helper-migration-decision.md)
- [day12-bidiag-gk-helper-decision.md](./artifacts/day12-bidiag-gk-helper-decision.md)
- [day13-maintainer-evidence-refresh.md](./artifacts/day13-maintainer-evidence-refresh.md)
- [day14-solver-selection-claim-closeout.md](./artifacts/day14-solver-selection-claim-closeout.md)

## Final Status

Sprint 123 is complete. It added bounded SVD, QR, and partial-SVD external
oracle follow-through, refreshed maintainer evidence for the new proof owners,
kept public solver-selection claims unchanged, and left the remaining
QR/partial-SVD/helper debt dependency-ordered for later Epic 11 sprints.
