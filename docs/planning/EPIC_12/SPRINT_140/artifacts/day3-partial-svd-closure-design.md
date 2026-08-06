# Day 3 Partial-SVD Closure Design

## Selected Residual

Working fixture key: `partial_svd_clustered_repeated_diag8x6_k3_v1`.

The selected Sprint 140 residual is fixture-local partial-SVD behavior on a
deterministic rectangular sparse diagonal matrix whose leading singular values
include an exact repeat and a tight cluster. The closure proves that the solver
returns the correct top-k singular subspace under valid basis ambiguity and
that convergence-budget handling fails closed before recovering with the
default budget.

This design supersedes the Day 2 working name
`partial_svd_clustered_repeated_subspace_budget_v1` with a concrete fixture key.

## Fixture Class

| Field | Design |
| --- | --- |
| Fixture key | `partial_svd_clustered_repeated_diag8x6_k3_v1` |
| Fixture family | `partial_svd_clustered_repeated` |
| Storage kind | generated sparse COO |
| Dimensions | 8 rows x 6 columns |
| Requested rank | `k=3` |
| Nonzeros | 5 diagonal entries |
| Singular values | `{10.0, 10.0, 9.999999, 4.0, 1.0, 0.0}` |
| Rank profile | rank 5 with one structural zero singular value |
| Rectangularity | tall rectangular |
| Conditioning class | clustered leading spectrum with moderate trailing gap |
| Scale class | unit-to-10 scale |
| Sparsity class | structured sparse diagonal |
| Generator | fixed generated diagonal entries, no random seed |
| Primary gap | repeated/clustered top-k subspace behavior plus budget failure/recovery |

The top three singular values are deliberately close enough to make raw vector
identity the wrong proof target. The gap between `9.999999` and `4.0` keeps the
top-k subspace separable for projector/subspace validation.

## Success Contract

The fixture closes only when the default-budget solver path satisfies all
fixture-local rows below.

| Row | Operation | Comparison kind | Expected result | Tolerance |
| --- | --- | --- | --- | --- |
| `partial_svd_clustered_repeated_diag8x6_k3_v1_singular_values` | `singular_values` | `value` | ordered top-k singular values `{10.0, 10.0, 9.999999}` | absolute `1e-8` |
| `partial_svd_clustered_repeated_diag8x6_k3_v1_left_subspace` | `singular_subspace` | `subspace_distance` | left top-k projector distance `<= 1e-8` | projector `1e-8` |
| `partial_svd_clustered_repeated_diag8x6_k3_v1_right_subspace` | `singular_subspace` | `subspace_distance` | right top-k projector distance `<= 1e-8` | projector `1e-8` |
| `partial_svd_clustered_repeated_diag8x6_k3_v1_vector_residual` | `vector_residuals` | `residual_norm` | max of `||A*v_i - sigma_i*u_i||` and `||A^T*u_i - sigma_i*v_i||` is `<= 1e-8` | absolute `1e-8` |
| `partial_svd_clustered_repeated_diag8x6_k3_v1_orthogonality` | `orthogonality` | `residual_norm` | max U/V orthogonality residual `<= 1e-8` | absolute `1e-8` |
| `partial_svd_clustered_repeated_diag8x6_k3_v1_default_status` | `convergence_budget` | `status` | `SPARSE_SUCCESS` | status only |

## Diagnostic Failure Contract

A tight-budget run on the same fixture should fail closed.

| Row | Operation | Comparison kind | Expected result | Tolerance |
| --- | --- | --- | --- | --- |
| `partial_svd_clustered_repeated_diag8x6_k3_v1_tight_budget_status` | `convergence_budget` | `status` | `SPARSE_ERR_NOT_CONVERGED` | status only |
| `partial_svd_clustered_repeated_diag8x6_k3_v1_tight_budget_no_partial_arrays` | `diagnostic` | `diagnostic` | no `sigma`, `U`, or `Vt` arrays are published on failure | not applicable |

The tight-budget row is diagnostic correctness evidence only. It is not a
convergence-rate, performance, or partial-result guarantee.

## Comparison Semantics

| Behavior | Interpretation |
| --- | --- |
| Singular values | Compare the requested top-k values in descending order. The repeated pair may appear in either basis orientation, but the value sequence must match the expected top-k spectrum. |
| Left/right vectors | Do not compare raw vector columns. Sign flips and rotations inside the repeated leading singular subspace are valid. |
| Left/right subspaces | Compare projectors for the full top-k left and right subspaces. A valid solver result may use any orthonormal basis spanning the same top-k subspace. |
| Vector residuals | Compute residuals for each returned triplet and compare the maximum residual against the row tolerance. This catches mismatched singular triplets without requiring basis identity. |
| Orthogonality | Check U and V basis orthogonality independently from residuals. |
| Basis rotation | Accepted inside the repeated singular-value block when projector distance and residual rows pass. |
| Clustered values | The `9.999999` value is distinct but near the repeated pair; value tolerance must be tight enough to prevent accidental collapse to a triple repeat. |
| Partial convergence | A non-converged run must return the documented status and must not be counted as solver pass evidence for value, residual, or subspace rows. |

## Tolerance Model

| Metric | Initial tolerance | Rationale |
| --- | --- | --- |
| Singular-value absolute error | `1e-8` | Fixture values are exactly generated and moderate in scale. |
| Projector/subspace distance | `1e-8` | Subspace comparison should tolerate floating-point basis choice while catching wrong top-k space. |
| Vector residual max norm | `1e-8` | Matches existing helper-scale residual checks. |
| Orthogonality max residual | `1e-8` | Matches existing U/V orthogonality expectations. |
| Status comparisons | exact/status only | Budget behavior is categorical. |

If implementation reveals stable solver accuracy is looser on this fixture, the
first choice is to revisit fixture clustering and validation design. Raising
tolerances should require an explicit note in the implementation artifact.

## Convergence-Budget Rules

| Budget mode | Proposed options | Required interpretation |
| --- | --- | --- |
| Default recovery | `compute_uv=1`, `economy=1`, default `max_iter`, default tolerance unless a fixture-specific tolerance is required | Must return `SPARSE_SUCCESS` and pass all value, residual, subspace, and orthogonality rows. |
| Tight failure | `compute_uv=1`, `economy=1`, `max_iter=1`, strict tolerance | Must return `SPARSE_ERR_NOT_CONVERGED` and must not publish partial `sigma`, `U`, or `Vt` arrays. |
| No-vector variant | Deferred | Not needed to close the selected subspace residual. |
| Partial-result variant | Deferred | Public API does not promise usable partial results after non-convergence. |

The default-budget and tight-budget runs must share the same fixture so the
diagnostic failure is tied to the selected residual rather than to an unrelated
diagonal smoke test.

## Proof Owner Boundary

| Surface | Day 3 ownership decision |
| --- | --- |
| Source-controlled corpus metadata | Day 4 should add the fixture row and generator row under `tests/corpus/`. |
| Expected rows | Day 5 should add expected rows under `tests/corpus/expected/`. |
| Solver-backed proof | Day 8 or Day 9 should add a focused partial-SVD corpus/proof owner, either a new `tests/test_svd_partial_corpus.c` or a narrowly scoped section in the existing SVD owner. |
| Helper ownership | Keep reusable projector, residual, and orthogonality helpers in a focused partial-SVD helper surface. Do not grow unrelated full-SVD helper code. |
| Oracle/report runner | Extend `scripts/run_corpus_oracle.py` only if needed for generated or solver-backed partial-SVD rows; keep generated outputs under `build/`. |
| Public docs | Update only after passing fixture-local proof exists. |

Preferred proof owner for implementation planning:
`tests/test_svd_partial_corpus.c`, with shared helpers extracted only if that
keeps `tests/test_svd.c` and `tests/test_svd_partial_helpers.h` readable.

## Non-Claims

The selected closure does not support these claims:

- broad partial-SVD correctness;
- broad repeated or clustered singular-value coverage;
- raw singular-vector identity;
- broad rectangular or nonsymmetric partial-SVD behavior;
- near-zero rank-threshold policy;
- low-rank approximation product guarantees;
- partial-result availability after non-convergence;
- convergence-rate or performance behavior;
- LAPACK, NumPy, SciPy, SuiteSparse, or broad external-library parity;
- platform, package, ABI, or state-of-the-art claims.

## Day 4 Handoff

Day 4 should convert this contract into source-controlled fixture metadata and
generator metadata. The implementation should preserve the fixture key, row
names, tolerances, claim scope, and non-claims unless a documented blocker is
found while adding the corpus rows.
