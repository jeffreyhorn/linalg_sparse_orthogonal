# Sprint 191 Day 2: Candidate Family Audit

## Purpose

Score the Day 1 candidate families and select exactly one bounded external
comparison family for Sprint 191 implementation.

## Selection Decision

Selected candidate:

| Field | Decision |
| --- | --- |
| Target key | `qr-incompatible-ls` |
| Fixture key | `qr_overdetermined_incompatible_4x2` |
| Solver family | QR |
| Subfamily | `qr_incompatible_ls` |
| Operation | `least_squares_solve` |
| Reference path | Source-controlled dense QR reference helper in `tests/qr_external_dense_reference.py` |
| Dependency policy | Required source-controlled Python helper; no external package dependency and no unavailable dependency counted as pass evidence. |
| Expected metric shape | Solve-style six-row comparison: project status, baseline status, residual norm, solution norm, solution values, and project-vs-baseline max absolute delta. |
| Initial tolerance posture | Reuse the selected comparison defaults where feasible: `1e-10` residual and solution tolerances, with Day 3 confirming residual-delta semantics for the incompatible least-squares fixture. |
| Claim scope | Fixture-local QR incompatible least-squares comparison against the selected source-controlled dense reference helper. |
| Required non-claims | No broad QR parity, no broad least-squares parity, no global rank-threshold policy, no broad rank-deficient solve claim, no NumPy/SciPy/LAPACK/SuiteSparse/Eigen parity, no Windows report freshness expansion, no package-manager proof, no shared-library ABI proof, no performance superiority, and no state-of-the-art claim. |

This candidate is selected because it adds residual-bearing incompatible
least-squares evidence that is materially different from the existing QR
minimum-norm and compatible least-squares selected rows, while reusing a
source-controlled fixture/reference path already exercised by C tests.

## Scoring Rubric

Scores use `1` for weak/unacceptable and `5` for strong/low-risk.

| Candidate | Evidence value | Reference availability | Determinism | CI/runtime cost | Claim safety | Implementation fit | Total |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| QR incompatible least-squares comparison | 5 | 5 | 5 | 5 | 4 | 5 | 29 |
| QR rank-threshold solve comparison | 4 | 5 | 5 | 5 | 3 | 4 | 26 |
| Partial-SVD nonsymmetric rectangular top-k comparison | 5 | 4 | 4 | 4 | 3 | 3 | 23 |
| Cholesky non-tridiagonal SPD comparison | 3 | 4 | 5 | 5 | 4 | 4 | 25 |
| LU singular expected-failure comparison | 4 | 4 | 5 | 5 | 3 | 3 | 24 |
| Sparse matrix-vector product comparison | 2 | 5 | 5 | 5 | 5 | 3 | 25 |

## Candidate Findings

### QR Incompatible Least-Squares Comparison

The fixture already exists in `tests/qr_external_dense_reference.py` as
`qr_overdetermined_incompatible_4x2`. It uses a deterministic 4-by-2 system
with exact least-squares solution `[2.0, -1.0]` and a known nonzero residual.
`tests/test_qr_solve.c` already compares project behavior against the
source-controlled dense helper for solution and residual agreement.

Strengths:

- improves QR selected comparison coverage from exact compatible and
  minimum-norm fixtures to an inconsistent residual-minimizing solve;
- requires no external package, data download, or broad dependency policy;
- can reuse the solve-style runner output pattern and six expected rows;
- likely avoids `.c` or `.h` changes because the fixture and public solver
  behavior already exist;
- claim wording can stay fixture-local and residual-specific.

Risks:

- documentation must avoid implying broad least-squares parity;
- Day 3 must confirm whether `residual_norm` should compare absolute residual
  value, project-baseline residual delta, or both;
- target naming must clearly distinguish incompatible least-squares from the
  existing compatible least-squares selected row.

Disposition: selected.

### QR Rank-Threshold Solve Comparison

Rank-threshold fixtures and dense reference helpers already exist, including
diagonal, scaled diagonal, duplicate-row perturbed, and dependent-row perturbed
families. This candidate would add useful rank-policy evidence, but the claim
surface is more delicate because rank-threshold behavior can be mistaken for a
global numerical-rank policy.

Disposition: defer. It remains a good future candidate after Sprint 191 proves
another QR selected comparison path cleanly.

### Partial-SVD Nonsymmetric Rectangular Top-K Comparison

The project has existing nonsymmetric partial-SVD fixture coverage, and this
candidate would add higher-value evidence beyond the diagonal-only selected
partial-SVD row. The implementation surface is larger because partial-SVD
study rows include singular values, residual, orthogonality, and projector
diagnostics, and vector orientation/sign caveats are easier to misstate.

Disposition: defer. It has high evidence value but too much review surface for
the next bounded family after the Windows freshness work.

### Cholesky Non-Tridiagonal SPD Comparison

This candidate would expand Cholesky selected comparison evidence beyond the
current tridiagonal SPD fixture. It is technically bounded and likely
deterministic, but the value is incremental because Sprint 183 and Sprint 190
already focus heavily on Cholesky selected comparison behavior.

Disposition: reject for Sprint 191 selection. Keep as a fallback only if QR
implementation uncovers an unexpected blocker.

### LU Singular Expected-Failure Comparison

Singular LU failure behavior has existing test coverage, but modeling an
expected failure as selected comparison pass evidence would require careful row
semantics. It may need a different status convention than the current
solve-style selected comparison row contract.

Disposition: defer. The failure-mode evidence is valuable, but Day 2 does not
select it because it risks changing comparison status semantics before the new
positive comparison family is complete.

### Sparse Matrix-Vector Product Comparison

This candidate is simple and deterministic, but it is less aligned with the
Sprint 191 wording around solver-family comparison evidence. It would add a
new operation family rather than a stronger selected solver comparison.

Disposition: reject for Sprint 191 selection. It is too low-value relative to
the selected QR incompatible least-squares candidate.

## Implementation Boundaries For Day 3

Day 3 should define the exact fixture and metric contract for
`qr-incompatible-ls`:

- fixture entries for the 4-by-2 matrix;
- right-hand side `[1.0, -2.0, 2.0, 5.0]`;
- expected solution `[2.0, -1.0]`;
- expected solution norm `sqrt(5)`;
- expected baseline residual `sqrt(3)`;
- row IDs for six solve-style generated comparison rows;
- residual tolerance and solution tolerance;
- exact claim and non-claim wording;
- whether Windows selected comparison metadata remains unchanged.

## Day 2 Validation

Read-only/source checks:

```sh
git status --short --branch --ahead-behind
sed -n '55,100p' docs/planning/EPIC_17/SPRINT_191/PLAN.md
sed -n '1,170p' docs/planning/EPIC_17/SPRINT_187/artifacts/day9-comparison-performance-gates.md
sed -n '1,220p' tests/qr_external_dense_reference.py
sed -n '230,380p' tests/test_qr_solve.c
rg -n "qr_overdetermined|rank_threshold|incompatible|partial_svd_nonsym|partial_svd_tall|lu_singular|cholesky" tests scripts include src docs/maintainer_guide.md
git diff --check
```

No `.c` or `.h` files were changed on Day 2, so `make format && make lint &&
make test` is not required.
