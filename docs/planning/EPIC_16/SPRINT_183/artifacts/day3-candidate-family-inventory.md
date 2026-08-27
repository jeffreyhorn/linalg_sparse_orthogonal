# Sprint 183 Day 3: Candidate Family Inventory

## Purpose

Inventory bounded external comparison families that are not already selected,
score them against Sprint 183 selection criteria, reject broad or unstable
candidates early, and hand one or two feasible candidates to Day 4.

## Existing Selected Coverage

Sprint 183 starts with these selected comparison families already covered:

| Target | Family | Claim shape |
| --- | --- | --- |
| `qr-minnorm` | QR minimum-norm solve | Fixture-local solve comparison against the selected dense QR helper. |
| `qr-compatible-ls` | QR compatible least-squares solve | Fixture-local solve comparison against the selected dense QR helper. |
| `partial-svd-diag6-k2` | Partial-SVD diagonal top-k | Fixture-local singular-value and residual comparison against the selected dense SVD helper. |
| `lu-nonsym-square-5` | Linked-list LU square solve | Fixture-local solve comparison against the selected dense LU helper. |

The additional family should therefore add one new solver/factorization
question without widening claims for QR, partial-SVD, or LU.

## Candidate Inventory

| Candidate | Candidate fixture | Helper availability | Expected metrics | Optional dependencies | C coverage | Score | Decision |
| --- | --- | --- | --- | --- | --- | ---: | --- |
| Cholesky SPD solve | 5x5 SPD tridiagonal Matrix Market fixture or equivalent generated entries | `tests/chol_external_dense_reference.py` already provides a source-controlled dense Cholesky reference. | Status, residual norm, solution norm, solution values, max project-vs-baseline delta. | NumPy/SciPy deferred only; not pass evidence. | `tests/test_cholesky.c` covers 5x5 tridiagonal solve, AMD/RCM variants, SuiteSparse SPD fixtures, and nearly singular SPD input. | 27/30 | Shortlist |
| LDLT symmetric-indefinite KKT solve | `ldlt_kkt_scaled_10` from `tests/ldlt_external_dense_reference.py` | `tests/ldlt_external_dense_reference.py` already provides source-controlled dense Gaussian solve rows. | Status, residual norm, solution norm, solution values, max delta; inertia only if Day 4 accepts added scope. | NumPy/SciPy deferred only; not pass evidence. | `tests/test_ldlt.c`, `tests/test_ldlt_backend_dispatch.c`, and `tests/test_ldlt_csc.c` cover indefinite solves, KKT-style fixtures, mixed pivots, inertia, and backend dispatch. | 25/30 | Shortlist |
| Cholesky reordered SPD solve | 5x5 or SuiteSparse SPD with AMD/RCM | Same Cholesky helper can solve original matrix. | Solve metrics plus reorder telemetry if added. | NumPy/SciPy deferred only. | Cholesky reordering tests already exist. | 20/30 | Reject for Sprint 183 |
| Eigensolver diagonal top-k | Small diagonal or tridiagonal symmetric matrix | No current source-controlled external eigensolver helper in the selected comparison runner shape. | Eigenvalue status, eigenvalue deltas, residuals, optional vector residuals. | External package baselines would need to stay deferred. | `tests/test_eigs.c` is broad and deep, including shift-invert and repeated-run behavior. | 18/30 | Reject for Sprint 183 |
| CG SPD solve | Small SPD tridiagonal or diagonal matrix | Could reuse a simple dense Gaussian helper, but no selected comparison helper exists today. | Status, residual norm, solution values, iterations, convergence reason. | Optional package rows deferred. | Iterative and IC tests exist, but convergence budget semantics are solver-specific. | 17/30 | Reject for Sprint 183 |
| GMRES/BiCGSTAB nonsymmetric solve | Small nonsymmetric square fixture | Could reuse dense Gaussian reference, but overlaps LU solve evidence. | Status, residual norm, solution values, iterations, restart/preconditioner diagnostics. | Optional package rows deferred. | Existing iterative tests are broad, but claims are convergence-budget sensitive. | 15/30 | Reject for Sprint 183 |
| Backend, dispatch, or performance comparison | Cholesky/LDLT CSC vs linked-list or timing fixture | Internal project paths, not an external dense reference comparison. | Backend telemetry, fill, time, or layout rows. | None required, but external comparison framing is wrong. | Backend tests exist. | 10/30 | Reject for Sprint 183 |

Scoring uses five criteria from Day 1, each scored from 1 to 5:
user value, fixture stability, comparator availability, implementation size,
maintenance cost, and claim risk. Higher scores indicate lower sprint risk and
better fit.

## Shortlisted Candidates

### Cholesky SPD 5x5 Tridiagonal Solve

This candidate would add a fixture-local selected comparison for the one-shot
Cholesky SPD solve path. It fits the existing six-row solve comparison shape:

- `project_status`;
- `baseline_status`;
- `residual_norm`;
- `solution_norm`;
- `solution_values`;
- `project_vs_baseline_max_abs_delta`.

It has the lowest implementation risk because the dense helper already exists,
the fixture can be deterministic and small, and the claim can avoid backend,
reordering, fill, performance, package, ABI, and broad SPD correctness claims.

### LDLT Scaled KKT 10x10 Solve

This candidate would add a fixture-local selected comparison for a
symmetric-indefinite KKT-style solve. It has strong user value because KKT and
saddle-point systems are not represented by the selected QR, SVD, or LU rows.

The risk is higher than Cholesky because LDLT has pivot-block and backend
semantics that can tempt broader claims. Day 4 should only select this family
if the claim remains a single fixture-local solve comparison. Inertia rows are
useful but should be deferred unless the sprint deliberately accepts a larger
row contract.

## Explicit Rejections

| Rejected candidate | Reason |
| --- | --- |
| Cholesky reordered SPD solve | Reordering-specific evidence risks backend and layout claims; a first Cholesky selected row should stay on the plain solve contract. |
| Eigensolver diagonal top-k | Eigenvector sign, ordering, subspace, shift-invert, and convergence semantics need a different contract from the existing solve-shaped selected rows. |
| CG SPD solve | Iterative convergence budgets and iteration counts create portable-performance and convergence-rate claim risk. |
| GMRES/BiCGSTAB nonsymmetric solve | Overlaps the existing LU nonsymmetric selected row while adding restart and preconditioner semantics. |
| Backend, dispatch, fill, or timing comparison | These are internal or performance comparisons, not external dense-reference comparisons. |
| Optional package comparison | NumPy, SciPy, LAPACK, SuiteSparse, and Eigen rows must remain deferred context, not selected pass evidence. |

## Claim Boundaries For Shortlist

The shortlisted candidates must preserve these boundaries:

- no broad solver correctness;
- no external-library ecosystem parity;
- no package-manager proof;
- no shared-library ABI proof;
- no Windows report freshness;
- no portable performance claim;
- no backend-layout identity claim;
- no release readiness claim;
- no state-of-the-art claim.

Cholesky-specific non-claims should also reject broad SPD coverage, broad
reordering coverage, fill superiority, and CSC-vs-linked-list parity. LDLT
non-claims should also reject broad symmetric-indefinite coverage, broad KKT
coverage, pivot-pattern identity, inertia generality unless specifically
selected, and sparse-direct solver parity.

## Day 4 Handoff

Day 4 should select exactly one family. Recommended order for final selection:

1. Cholesky SPD 5x5 tridiagonal solve.
2. LDLT scaled KKT 10x10 solve.

The Cholesky candidate is recommended for lowest implementation and claim risk.
The LDLT candidate is defensible if Sprint 183 prioritizes adding
symmetric-indefinite selected comparison coverage over implementation
simplicity.

## Validation

Day 3 changes planning artifacts only. Validation:

- `git diff --check`

## Completion Criteria Review

| Criterion | Status | Notes |
| --- | --- | --- |
| At least one feasible bounded family is shortlisted. | Complete | Cholesky SPD solve and LDLT KKT solve are shortlisted. |
| Broad or unstable candidates have rejection reasons. | Complete | Eigensolver, iterative, backend, performance, and optional-package candidates are rejected explicitly. |
| Family selection remains tied to defensible evidence. | Complete | Shortlist references existing helpers, C coverage, metrics, and non-claims. |
