# Sprint 124 Retrospective

**Sprint:** 124 - Residual QR, Partial-SVD & Helper Oracle Follow-Through
**Duration:** 14 days
**Status:** Complete

## Definition Of Done Checklist

- [x] Created Sprint 124 day-by-day plan, working notes, and artifact
      directory.
- [x] Re-read Epic 11 Sprint 124 scope, Sprint 123 residual deferred debt, and
      Sprint 121-123 oracle taxonomy inputs.
- [x] Established duplicate fences around completed Sprint 121-123 SVD, QR,
      partial-SVD, helper, maintainer-evidence, and solver-selection work.
- [x] Defined rank-deficient QR policy for rank threshold, nullspace,
      pseudoinverse, tolerance, skip behavior, and failure interpretation.
- [x] Implemented the bounded `qr_rankdef_duplicate_5x4_rank_only` external
      rank-only QR fixture.
- [x] Defined QR minimum-norm behavior boundaries across QR solve, COLAMD,
      SVD-pseudoinverse, fallback, refinement, rank-deficient, zero-row, and
      SuiteSparse scenarios.
- [x] Implemented the bounded `qr_underdetermined_minnorm_2x4` exact
      minimum-norm QR fixture.
- [x] Defined QR Q-basis/economy sign, orientation, projection, subspace, and
      economy-shape semantics.
- [x] Implemented the bounded `qr_economy_projector_5x3` economy-Q projector
      fixture.
- [x] Defined partial-SVD vector/subspace residual, sign-invariance,
      projection, tolerance, skip, and failure-interpretation semantics.
- [x] Implemented the bounded `partial_svd_vector_residual_diag6_k2`
      vector-residual fixture.
- [x] Published the partial-SVD residual scenario matrix and explicit
      deferral package for repeated, clustered, rank-deficient, rectangular,
      SuiteSparse corpus, low-rank optimality, convergence-budget, and
      nonsymmetric rectangular lanes.
- [x] Revisited minimum-norm helper migration and Bidiagonal/Golub-Kahan
      helper extraction, preserving behavior-specific ownership through
      explicit deferrals.
- [x] Confirmed `docs/maintainer_guide.md` names the accepted Sprint 124
      bounded QR and partial-SVD lanes while preserving broad non-claims.
- [x] Audited `docs/solver_selection.md` and published a no-update rationale
      because the new evidence remains fixture-scoped.
- [x] Ran focused helper and owner-test checks for every accepted Sprint 124
      oracle lane.
- [x] Ran the required full C quality gate after Sprint 124 `.c` and `.h`
      changes: `make format`, `make lint`, and `make test`.
- [x] Published final validation, claim-gate, closeout, non-claim, residual,
      and Sprint 125 handoff artifacts.
- [x] Finalized this retrospective and ran final documentation hygiene.

## What Went Well

1. **Residual QR debt became bounded evidence instead of broad claims.**
   Sprint 124 separated rank-only, minimum-norm, and Q/economy QR evidence
   before implementation. That produced three named external-reference lanes
   without turning them into broad QR, LAPACK, NumPy, SciPy, SuiteSparse,
   basis, nullspace, or performance parity claims.

2. **Minimum-norm stayed behavior-specific.** The sprint added one exact 2x4
   QR solve proof, but kept COLAMD, fallback, refinement, rank-deficient,
   QR-vs-SVD-pseudoinverse, and SuiteSparse minimum-norm evidence explicitly
   deferred to scenario owners.

3. **Q/economy evidence avoided raw basis traps.** The accepted QR economy
   lane compares projectors and orthogonality instead of raw Q columns. That
   gives useful economy-mode evidence while preserving sign, orientation, and
   basis-ambiguity boundaries.

4. **Partial-SVD vector evidence used sign-invariant checks.** The new
   partial-SVD lane validates singular-triplet residuals and U/V
   orthogonality for one exact diagonal fixture without comparing raw vector
   components or claiming broad vector/subspace parity.

5. **The claim gate stayed disciplined.** The maintainer guide records the
   bounded Sprint 124 lanes, while solver-selection wording remains unchanged
   because the evidence does not justify broader user-facing claims.

6. **Closeout produced a usable Sprint 125 handoff.** Day 14 consolidated the
   accepted lane owners, external-reference script inventory, skip/failure
   semantics, report-index inputs, non-claims, and future-owner queue.

## What Did Not Go Well

1. **The oracle matrix remains intentionally narrow.** Sprint 124 added four
   useful named fixtures, but broad QR, SVD, partial-SVD, corpus,
   dense-library, platform, and performance parity remain non-claims.

2. **Rank-deficient QR is still mostly deferred.** The sprint added rank-only
   evidence, but residual-only rank-deficient solves, nullspace/subspace,
   near-threshold behavior, and SuiteSparse rank-deficient QR still require
   separate owners and promotion gates.

3. **Minimum-norm coverage is still one exact solve lane.** COLAMD/reordered,
   fallback, refinement, rank-deficient, SVD-pseudoinverse, and SuiteSparse
   minimum-norm behavior remain unresolved because each needs its own failure
   semantics and claim boundary.

4. **Partial-SVD residual expansion stopped at one exact diagonal vector
   lane.** Repeated-spectrum, clustered-spectrum, rank-deficient subspace,
   rectangular vector residual, SuiteSparse corpus, low-rank optimality, and
   convergence-budget evidence were explicitly deferred.

5. **Helper consolidation debt remains visible.** Deferring generic
   minimum-norm and Bidiagonal/Golub-Kahan helper movement preserved numerical
   meaning, but future maintainability work still needs dedicated
   behavior-specific helper extraction.

6. **Public docs did not gain a broader user-facing claim.** This was the
   correct outcome, but it means the new evidence is primarily maintainer
   evidence until later sprints broaden the corpus and claim surface.

## Final Metrics

### Validation

| Metric | Sprint 124 close state |
|---|---:|
| QR rank-only external helper | passed, emitted `OK 1` and rank `3` |
| QR minimum-norm external helper | passed, emitted `OK 6`, four solution entries, residual, and norm |
| QR economy-projector external helper | passed, emitted `OK 29`, shape values, and projector entries |
| partial-SVD singular-value helper for vector-residual lane | passed, emitted `OK 2` and 2 singular values |
| focused QR solve tests | 17 passed, 0 failed, 0 skipped |
| focused QR solve assertions | 1069 |
| focused QR tests | 66 passed, 0 failed, 0 skipped |
| focused QR assertions | 628 |
| focused SVD tests | 109 passed, 0 failed, 0 skipped |
| focused SVD assertions | 1803 |
| required full Make formatting | `make format` passed |
| required full Make lint | `make lint` passed |
| required full Make tests | `make test` passed |
| full Make test final result | `All tests passed.` |
| diff hygiene | `git diff --check` passed |
| trailing-whitespace scan | passed on Sprint 124 docs and touched files |
| Day 14 C quality rerun | not required; documentation-only closeout |

### Sprint Artifact Package

| Metric | Sprint 124 close state |
|---|---:|
| artifact files under `SPRINT_124/artifacts/` | 14 |
| sprint plan files | 1 |
| working notes files | 1 |
| retrospective files | 1 |
| modified external-reference helper scripts | 1 |
| modified existing test files | 3 |
| modified existing test helper headers | 1 |
| maintainer guide updates | 1 |
| solver-selection public wording updates | 0 |
| Makefile/CMake/CTest registration changes | 0 |

## Movement And Claim Outcomes

| Area | Outcome |
|---|---|
| Sprint intake | Completed residual dependency map, duplicate fence, validation boundary, and day-level owner map. |
| QR rank-deficient evidence | Added bounded `qr_rankdef_duplicate_5x4_rank_only` rank-only evidence. |
| QR minimum-norm evidence | Added bounded `qr_underdetermined_minnorm_2x4` exact solution/residual/norm evidence. |
| QR Q/economy evidence | Added bounded `qr_economy_projector_5x3` projector and orthogonality evidence. |
| Partial-SVD vector/subspace evidence | Added bounded `partial_svd_vector_residual_diag6_k2` residual and orthogonality evidence. |
| Partial-SVD residual scenarios | Deferred repeated, clustered, rank-deficient, rectangular, corpus, low-rank, convergence-budget, and nonsymmetric rectangular lanes with promotion gates. |
| Minimum-norm helper migration | Deferred; preserved QR solve/COLAMD/SVD-pinv/fallback/refinement/rank-deficient/zero-row/SuiteSparse scenario ownership. |
| Bidiagonal/Golub-Kahan helper extraction | Deferred; preserved implicit Householder reconstruction, wide transpose, explicit `U`/`V`, wide GK skip, and bidiagonal QR iteration ownership. |
| Maintainer evidence | Confirmed `docs/maintainer_guide.md` names the Sprint 124 bounded lanes and preserves fixture-scoped trust boundaries. |
| Solver-selection wording | No public wording expansion; no-update rationale published. |
| Public API | Unchanged. |
| Build registration | Unchanged; no new test executable, library source, Makefile entry, CMake entry, or CTest member was added. |
| External-library parity | Not claimed. |

## Residual Deferred Debt

Most important carry-forward work:

- Add rank-deficient QR residual-only evidence only after it proves trust
  beyond deterministic tests without implying nullspace or minimum-norm
  behavior.
- Add rank-deficient QR nullspace/subspace evidence only after sign, ordering,
  nullity, projection/subspace metric, and fixture-local tolerance policies
  are explicit.
- Add near-rank-deficient QR threshold evidence only after threshold family,
  expected ranks, stability policy, and non-global interpretation are defined.
- Add SuiteSparse rank-deficient QR evidence only after optional corpus,
  platform skip behavior, support tier, diagnostics, and claim boundaries are
  explicit.
- Add QR minimum-norm COLAMD, fallback, rank-deficient, refinement,
  QR-vs-SVD-pseudoinverse, and SuiteSparse evidence only under
  behavior-specific owners.
- Add raw QR Q-column, rank-deficient Q/nullspace subspace, wide economy,
  sparse-mode Q/economy, and SuiteSparse Q/economy evidence only after the
  appropriate basis, shape, projection, skip, and corpus policies are defined.
- Expand partial-SVD residual evidence to rectangular, repeated-spectrum,
  clustered-spectrum, rank-deficient subspace, SuiteSparse corpus, low-rank
  optimality, convergence-budget, and nonsymmetric rectangular cases only
  with dedicated owners and metric policies.
- Revisit minimum-norm helper movement only with behavior-specific helper
  names and focused QR solve, COLAMD, SVD, and full quality validation.
- Extract Bidiagonal/Golub-Kahan helpers only into a dedicated owner that
  preserves transpose, reconstruction, explicit `U`/`V`, wide skip, and
  QR-iteration semantics.
- Refresh public solver-selection wording only when future evidence supports a
  user-facing claim beyond the current workflow guidance.

Still consciously constrained rather than silently solved:

- no LAPACK parity claim;
- no SciPy or NumPy parity claim;
- no BLAS, SuiteSparse, PETSc, Trilinos, Eigen, ARPACK, or vendor-backend
  parity claim;
- no broad external dense-library or ecosystem parity claim;
- no broad QR factorization, QR solve, rank-deficient solve, nullspace,
  minimum-norm, Q-basis, economy, sparse-mode, reorder, backend, corpus, or
  performance parity claim;
- no raw Q-basis equality, Q-sign, Q-orientation, unique-basis, singular-vector,
  or subspace external parity claim;
- no broad SVD or partial-SVD external parity claim;
- no repeated-spectrum, clustered-spectrum, rank-deficient subspace,
  low-rank optimality, convergence-budget, or corpus parity claim;
- no QR/SVD pseudoinverse oracle parity claim beyond explicitly named bounded
  checks;
- no generic helper API or helper consolidation claim;
- no package-manager distribution claim;
- no shared-library or dynamic ABI stability claim;
- no equal Linux/macOS/Windows reviewed-support claim;
- no public API, install-header, package, CMake, Makefile, CI, or CTest
  expansion claim;
- no portable performance, scalability, memory, or state-of-the-art claim.

Not carried forward as unresolved Sprint 124 debt:

- Sprint 124 intake, residual dependency map, and duplicate fencing;
- rank-deficient QR policy design;
- `qr_rankdef_duplicate_5x4_rank_only` implementation;
- QR minimum-norm behavior contract;
- `qr_underdetermined_minnorm_2x4` implementation;
- QR Q-basis/economy semantic design;
- `qr_economy_projector_5x3` implementation;
- partial-SVD vector/subspace semantic design;
- `partial_svd_vector_residual_diag6_k2` implementation;
- partial-SVD residual scenario matrix and explicit deferral package;
- minimum-norm helper migration decision;
- Bidiagonal/Golub-Kahan helper extraction decision;
- maintainer evidence and solver-selection claim gate;
- final validation package and closeout evidence.

## Key Deliverables

- [PLAN.md](./PLAN.md)
- [WORKING_NOTES.md](./WORKING_NOTES.md)
- [day1-sprint-intake.md](./artifacts/day1-sprint-intake.md)
- [day2-qr-rank-policy.md](./artifacts/day2-qr-rank-policy.md)
- [day3-qr-rank-decision.md](./artifacts/day3-qr-rank-decision.md)
- [day4-qr-minnorm-behavior-contract.md](./artifacts/day4-qr-minnorm-behavior-contract.md)
- [day5-qr-minnorm-decision.md](./artifacts/day5-qr-minnorm-decision.md)
- [day6-qr-basis-economy-semantics.md](./artifacts/day6-qr-basis-economy-semantics.md)
- [day7-qr-basis-economy-decision.md](./artifacts/day7-qr-basis-economy-decision.md)
- [day8-partial-svd-vector-subspace-semantics.md](./artifacts/day8-partial-svd-vector-subspace-semantics.md)
- [day9-partial-svd-vector-subspace-decision.md](./artifacts/day9-partial-svd-vector-subspace-decision.md)
- [day10-partial-svd-residual-scenario-matrix.md](./artifacts/day10-partial-svd-residual-scenario-matrix.md)
- [day11-partial-svd-residual-deferral-package.md](./artifacts/day11-partial-svd-residual-deferral-package.md)
- [day12-helper-ownership-follow-through.md](./artifacts/day12-helper-ownership-follow-through.md)
- [day13-validation-claim-gate.md](./artifacts/day13-validation-claim-gate.md)
- [day14-closeout-and-handoff.md](./artifacts/day14-closeout-and-handoff.md)

## Final Status

Sprint 124 is complete. It converted Sprint 123's residual QR,
partial-SVD, minimum-norm, and Bidiagonal/Golub-Kahan deferred debt into four
bounded oracle evidence lanes, explicit future-owner packages, a validated
maintainer evidence and claim gate, and a stable Sprint 125 corpus/report
handoff.
