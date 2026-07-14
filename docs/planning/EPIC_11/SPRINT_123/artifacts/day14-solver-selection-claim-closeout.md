# Day 14 Solver-Selection Claim Closeout

## Purpose

Close Sprint 123 by deciding whether the bounded SVD, QR, partial-SVD, helper,
and maintainer evidence outcomes justify any public solver-selection wording
change.

## Claim Gate Decision

No public solver-selection wording update was made.

The Sprint 123 implementation outcomes improve maintainer confidence for
specific named fixtures, but they do not create a broader user-facing claim
than `docs/solver_selection.md` already makes:

- QR remains the first public workflow for rectangular, least-squares,
  minimum-norm, and rank-sensitive workflows.
- SVD remains the public workflow for singular values, numerical rank,
  condition estimates, pseudoinverse behavior, and low-rank approximations.
- Benchmarks remain local measurement artifacts, not portable timing
  guarantees.
- The guide does not claim LAPACK, NumPy, SciPy, package, platform, ABI,
  performance, vector/subspace, rank-deficient QR, or broad external oracle
  parity.

The earned Sprint 123 evidence is maintainer-facing and fixture-bounded. The
right public action is therefore no wording expansion.

## Completed Evidence

| Area | Sprint 123 outcome | Public claim impact |
| --- | --- | --- |
| SVD external fixtures | Added `svd_wide_fullrank_4x6` as a bounded singular-value-only external dense-reference fixture. | no public wording change; fixture proves one additional wide singular-value lane only |
| QR external fixtures | Added `qr_overdetermined_compatible_5x3` as a bounded full-column-rank compatible least-squares fixture. | no public wording change; fixture proves one additional compatible least-squares lane only |
| Partial SVD external fixtures | Added `partial_svd_tall_diag_8x5_k3` as a bounded top-three singular-value fixture. | no public wording change; fixture proves value-only top-k behavior for one tall diagonal case |
| Minimum-norm helper migration | Explicitly deferred generic helper movement to preserve behavior-specific ownership. | no public wording change |
| Bidiagonal/Golub-Kahan helper extraction | Explicitly deferred helper extraction to preserve specialized semantic ownership. | no public wording change |
| Maintainer evidence table | Updated `docs/maintainer_guide.md` with bounded owners, trust boundaries, validation commands, and non-claims. | maintainer-only proof map; no user-facing expansion |

## Final Non-Claim Register

Sprint 123 does not claim:

- LAPACK, NumPy, SciPy, Eigen, SuiteSparse, PETSc, Trilinos, ARPACK, or vendor
  backend parity.
- Broad QR, SVD, partial-SVD, direct-solver, iterative-solver, eigensolver,
  package, platform, ABI, performance, scalability, or state-of-the-art parity.
- Rank-deficient QR external oracle parity.
- QR minimum-norm external oracle parity.
- QR Q-basis, economy-mode, sparse-mode, reorder, or fill external parity.
- SVD vector, subspace, pseudoinverse, minimum-norm, low-rank optimality, or
  rank-threshold external parity.
- Partial-SVD vector, subspace, repeated-spectrum, clustered-spectrum,
  rank-deficient, convergence-budget, or low-rank optimality external parity.
- That helper extraction has made minimum-norm or Bidiagonal/Golub-Kahan
  behavior generic across owners.
- That examples, benchmarks, maintainer tables, or deterministic internal
  fixtures replace family-local oracle owners.

## Dependency-Ordered Residual Deferred Debt

1. Rank-deficient QR external oracle design.
   - Dependency: explicit rank-threshold, nullspace, pseudoinverse, and
     minimum-norm policy before helper-backed comparison.
   - Future owner: QR solve oracle sprint.
2. QR minimum-norm external oracle design.
   - Dependency: behavior-specific ownership across QR solve, COLAMD,
     SVD-pseudoinverse, fallback, refinement, and SuiteSparse paths.
   - Future owner: QR solve / minimum-norm sprint.
3. QR Q-basis and economy external oracle design.
   - Dependency: sign, orientation, projection, subspace, and economy-shape
     semantics.
   - Future owner: QR basis/economy sprint.
4. Partial-SVD vector and subspace oracle design.
   - Dependency: sign-invariant vector rules and projection/subspace criteria.
   - Future owner: partial-SVD semantic sprint.
5. Partial-SVD repeated-spectrum, clustered-spectrum, rank-deficient,
   convergence-budget, and low-rank optimality oracle design.
   - Dependency: ambiguity, tolerance, convergence, and optimality policy for
     each class.
   - Future owner: partial-SVD oracle sprint.
6. Minimum-norm helper migration.
   - Dependency: behavior-specific helper names and promotion gates that keep
     scenario-local assertions visible.
   - Future owner: maintainability/helper sprint.
7. Bidiagonal/Golub-Kahan helper extraction.
   - Dependency: dedicated owner preserving wide transpose, Householder
     reconstruction, explicit `U`/`V`, wide GK skip, and bidiagonal QR
     iteration semantics.
   - Future owner: Bidiagonal/GK maintainability sprint.

## Retrospective Inputs

- Sprint 123 should be recorded as a mixed implementation/deferral sprint.
- The sprint successfully converted three bounded oracle candidates into code:
  `svd_wide_fullrank_4x6`, `qr_overdetermined_compatible_5x3`, and
  `partial_svd_tall_diag_8x5_k3`.
- The sprint intentionally preserved helper-local ownership for minimum-norm
  and Bidiagonal/Golub-Kahan behaviors instead of flattening scenario-specific
  assertions.
- The maintainer evidence table was the right surface for new proof detail;
  `docs/solver_selection.md` did not need public claim expansion.
- Future sprints should promote the residual queue only after each owner
  defines basis, ambiguity, tolerance, skip, and failure semantics.

## Validation

- `git diff --check`
- `rg -n "[ \t]$" docs/solver_selection.md docs/maintainer_guide.md docs/planning/EPIC_11/SPRINT_123`
- `rg -n "external parity|ecosystem parity|state.of.the.art|LAPACK|NumPy|SciPy|minimum-norm|rank-deficient|partial-SVD|SVD|QR" README.md docs/solver_selection.md docs/algorithm.md docs/maintainer_guide.md`

## Completion Criteria

- Item 7 is complete.
- All Sprint 123 items are complete or explicitly deferred.
- Every day remains at or below 12 hours and the total estimate remains 166
  hours.
- No unsupported public, support-level, package, platform, ABI, performance, or
  state-of-the-art claim was introduced.
