# Sprint 183 Day 4: Family Selection

## Purpose

Select exactly one bounded external comparison family for Sprint 183 and define
the closed claim, non-claims, support boundary, and implementation surface.

## Decision

Selected family: Cholesky SPD tridiagonal solve.

Proposed target metadata:

| Field | Value |
| --- | --- |
| Target key | `cholesky-spd-tridiag-5` |
| Report family | `comparison` |
| Subfamily | `cholesky_spd_tridiag_5` |
| Fixture key | `cholesky_spd_tridiag_5` |
| Operation | `cholesky_spd_solve` |
| Output directory | `build/comparison/cholesky_spd_tridiag_5/` |
| Required helper | `tests/chol_external_dense_reference.py` |
| Support tier | `local_only` |
| Workflow platforms | `linux;macos` |
| Windows status | Deferred; no Sprint 183 Windows report freshness promotion |

## Selection Rationale

| Criterion | Cholesky SPD solve | LDLT KKT solve | Decision impact |
| --- | --- | --- | --- |
| User value | Adds direct SPD solve evidence, a common solver-selection branch not selected today. | Adds symmetric-indefinite KKT evidence, also valuable. | Both useful. |
| Fixture stability | A 5x5 SPD tridiagonal matrix is deterministic, well-conditioned, and small. | `ldlt_kkt_scaled_10` is deterministic but exercises pivot semantics. | Cholesky lower risk. |
| Comparator availability | Source-controlled dense Cholesky helper already exists. | Source-controlled dense Gaussian helper already exists. | Both feasible. |
| Implementation size | Fits the existing six-row solve comparison shape with one new project probe mode. | Fits the solve shape, but needs stricter LDLT-specific semantics. | Cholesky lower risk. |
| Maintenance cost | SPD-only fixture and Cholesky helper are easy to audit. | KKT/pivot behavior requires more care in future reviews. | Cholesky lower risk. |
| Claim risk | Closed fixture-local SPD solve claim is straightforward. | Indefinite, pivot-pattern, inertia, and backend wording carry more risk. | Cholesky selected. |
| Manifest/workflow fit | Exact six-file selected comparison contract can be reused. | Exact six-file contract can be reused if inertia is deferred. | Both feasible. |

The LDLT KKT candidate remains a good future candidate, but Sprint 183 should
select Cholesky because it gives useful new family coverage with the lowest
remaining-sprint implementation and claim risk.

## Closed Claim

For the selected 5x5 symmetric positive-definite tridiagonal fixture with
diagonal value 4 and off-diagonal value -1, the project one-shot Cholesky path
and the source-controlled dense Cholesky helper both solve the same generated
linear system successfully, and their selected output metrics agree within the
declared tolerances.

This claim is valid only for the generated comparison row context:

- the named fixture;
- the named project command;
- the named baseline helper;
- the recorded source commit and branch;
- the recorded platform, compiler, and configuration;
- the selected generated output files.

## Non-Claims

The selected family must explicitly reject:

- broad Cholesky correctness;
- broad SPD matrix coverage;
- broad reordering coverage;
- CSC-vs-linked-list parity;
- factor layout identity;
- fill superiority;
- external-library ecosystem parity;
- NumPy, SciPy, LAPACK, SuiteSparse, or Eigen parity;
- package-manager proof;
- shared-library ABI proof;
- Windows report freshness;
- broad platform portability;
- release readiness;
- portable performance;
- state-of-the-art status.

## Support Boundary

The selected row should use `support_tier=local_only`, matching existing
comparison rows. Linux and macOS selected workflow uploads may promote the
generated artifacts as reviewed selected evidence, but the generated row
metadata itself must not claim hosted CI proof, Windows proof, package proof,
ABI proof, release proof, or broad platform support.

## Fixture Direction

Day 5 should define an exact fixture contract for `cholesky_spd_tridiag_5`.
The preferred matrix is:

```text
A = [[ 4, -1,  0,  0,  0],
     [-1,  4, -1,  0,  0],
     [ 0, -1,  4, -1,  0],
     [ 0,  0, -1,  4, -1],
     [ 0,  0,  0, -1,  4]]
```

Use `x_expected = [1, 2, 3, 4, 5]` and derive `rhs = A * x_expected` from the
fixture. Day 5 should finalize whether this is represented as inline generated
entries, a `GENERATED_FIXTURES` row, or a source-controlled Matrix Market file.
The runner already supports inline entries for solve-shaped targets, so no
stored file is required unless the helper contract is simpler with one.

## Expected Rows

The selected report should emit the standard six solve rows:

| Metric | Row intent |
| --- | --- |
| `project_status` | Project Cholesky factor/solve status. |
| `baseline_status` | Source-controlled dense helper status. |
| `residual_norm` | Project residual for `A*x - rhs`. |
| `solution_norm` | Project solution norm. |
| `solution_values` | Project solution vector compared to baseline vector. |
| `project_vs_baseline_max_abs_delta` | Maximum absolute project-vs-baseline solution delta. |

Day 5 should set exact row IDs using the current pattern
`comparison_<fixture_key>_<metric>_v1`.

## Implementation Surface Map

| Surface | Required work |
| --- | --- |
| `scripts/run_external_comparison.py` target registry | Add one Cholesky target with exact fixture, RHS, expected solution, tolerances, summary text, success message, and non-claims. |
| Project probe | Add `cholesky_spd_solve` handling that includes `sparse_cholesky.h`, factors with `sparse_cholesky_factor`, solves with `sparse_cholesky_solve`, and emits the existing solve fields. |
| Baseline helper dispatch | Add Cholesky-specific helper routing to `tests/chol_external_dense_reference.py`, plus Cholesky baseline name/version/configuration/dependency rows. |
| Python runner tests | Extend target expectations, unsupported-target diagnostics, required helper path, expected row IDs, and report-family metadata checks. |
| C validation | Reuse `tests/test_cholesky.c` 5x5 tridiagonal solve proof; add only focused C coverage if the implementation changes production Cholesky behavior. |
| Report-family manifest | Add `comparison/cholesky_spd_tridiag_5` with generated-local origin, generated-compare freshness policy, exact generator command, exact artifact pattern, and non-claims. |
| Selected target manifest | Add one selected comparison row with Linux/macOS workflow metadata and exact required files. |
| Makefile freshness target | Generate the Cholesky comparison before running selected comparison freshness normalization. |
| Workflow guards | Add the Cholesky output directory to exact selected uploads and keep broad upload rejection intact. |
| Documentation | Add bounded Cholesky selected comparison language to README, solver-selection, maintainer guide, corpus README, and report-index schema docs. |

## Deferred Candidate

LDLT scaled KKT 10x10 solve is deferred. It should remain available for a
future sprint if symmetric-indefinite selected comparison coverage becomes the
next priority. That future selection should decide explicitly whether inertia
is in scope, because adding inertia changes the row count and claim surface.

## Day 5 Handoff

Day 5 should convert this decision into the exact fixture and metric contract:

- fixture representation and provenance;
- target key, fixture key, subfamily, operation, and output directory;
- RHS and expected solution values;
- row IDs and tolerances;
- dependency defer rows;
- report-family and selected-target manifest draft rows;
- exact non-claim wording.

## Validation

Day 4 changes planning artifacts only. Validation:

- `git diff --check`

## Completion Criteria Review

| Criterion | Status | Notes |
| --- | --- | --- |
| Exactly one family is selected. | Complete | Cholesky SPD tridiagonal solve is selected. |
| Selected family claim is bounded and testable. | Complete | Claim is fixture-local, solve-shaped, and tied to six selected metrics. |
| Implementation scope is narrow enough for the remaining sprint. | Complete | Required work is one runner target, one probe mode, manifest/report/docs updates, and focused tests. |
