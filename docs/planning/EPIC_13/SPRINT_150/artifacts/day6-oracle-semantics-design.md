# Sprint 150 Day 6: QR Oracle Semantics Design

## Purpose

Define the numerical oracle semantics for the Sprint 150 QR corpus fixtures
before implementing or extending generated oracle rows. The design keeps
comparisons fixture-local, subspace-safe, and bounded to the source-controlled
expected rows from Day 5.

## Current Oracle Baseline

`scripts/run_corpus_oracle.py` currently supports:

- generated-reference rows for `qr_rank_deficient_6x4_nullspace_v1`;
- optional solver-backed QR rows for the same 6x4 fixture through a temporary
  C probe when `--include-solver-qr` is used;
- generated-reference partial-SVD rows for the Sprint 140 fixture when
  `--include-partial-svd` is used;
- report/index rows with command, commit, branch, timestamp, platform,
  compiler, configuration, support tier, claim scope, and non-claims.

The current QR solver-backed path is hard-coded to one fixture and one
nullspace residual metric. The generic `value` comparison helper is currently
specialized for partial-SVD `top_k` rows, so Day 7 must extend it before
minimum-norm scalar/vector rows can become executable oracle comparisons.

## Required Oracle Row Families

### Rank-Deficient Rectangular QR

Fixtures:

- `qr_rankdef_duplicate_5x4_v1`
- `qr_rankdef_dependent_row_4x3_v1`

Rows per fixture:

| Row Suffix | Operation | Comparison | Expected Encoding | Observed Encoding | Tolerance |
| --- | --- | --- | --- | --- | --- |
| `_rank` | `rank_info` | exact rank | integer rank | integer rank | exact `0` |
| `_nullity` | `rank_info` | exact nullity | integer nullity | integer nullity | exact `0` |
| `_nullspace_residual` | `nullspace` | normalized residual | `normalized_null_vector_residual<=1e-10` | `normalized_null_vector_residual=<value>` | absolute `1e-10` |
| `_nullspace_subspace` | `nullspace` | projector max-abs distance | `projector_distance<=1e-8` | `projector_distance=<value>` | projector `1e-8` |

Semantics:

- `rank` comes from `sparse_qr_rank(&qr, 0.0)`.
- `nullity` comes from `sparse_qr_nullspace(&qr, 0.0, NULL, &nullity)`.
- nullspace residual is `||A Z||_F / max(||Z||_F, tiny)`, where `Z` is the
  solver-produced nullspace basis. For nullity `1`, this is equivalent to the
  normalized vector residual already used by `test_qr_corpus.c`.
- projector distance is `max_abs(Z Z^T - Z_ref Z_ref^T)` after both bases are
  column-normalized and treated as a subspace. The comparison must not inspect
  raw basis-vector signs, ordering, orientation, or scale.
- reference projectors must be generated from exact deterministic fixture
  matrices, preferably by rational row-reduction followed by deterministic
  orthonormalization in the oracle code. Existing dense-reference test helpers
  may be used as a cross-check, but the maintained corpus oracle should not
  require optional external data.

Failure classes:

- `fail_rank_mismatch`
- `fail_nullity_mismatch`
- `fail_nullspace_residual`
- `fail_nullspace_projector_distance`
- `fail_qr_probe_compile`
- `fail_qr_probe_runtime`

### Underdetermined Minimum-Norm QR

Fixtures:

- `qr_underdetermined_minnorm_2x4`
- `qr_minnorm_3x6_exact_values`
- `qr_minnorm_5x10_exact_values`

Rows per fixture:

| Row Suffix | Operation | Comparison | Expected Encoding | Observed Encoding | Tolerance |
| --- | --- | --- | --- | --- | --- |
| `_status` | `minnorm_solve` | status | `SPARSE_SUCCESS` | `SPARSE_SUCCESS` or error symbol | status only |
| `_residual` | `minnorm_solve` | residual norm | `residual_norm<=1e-10` | `residual_norm=<value>` | absolute `1e-10` |
| `_solution_norm` | `minnorm_solve` | scalar value | `solution_norm=<numeric>` | `solution_norm=<value>` | absolute `1e-10` |
| `_solution_values` | `minnorm_solve` | vector max-abs error | `solution_values=<comma-vector>` | `solution_values=<comma-vector>;max_abs_error=<value>` | absolute `1e-10` |

Semantics:

- status comes from `sparse_qr_solve_minnorm(A, b, x, NULL)`.
- residual is `||A x - b||_2`.
- solution norm is `||x||_2`.
- exact solution values are compared by maximum absolute component error.
- exact-value rows are allowed only for the selected deterministic full-row-rank
  fixtures and must not be generalized to rank-deficient or inconsistent
  systems.

Day 7 normalization:

- replace free-form expected norm strings such as `sqrt(8.4)` and
  `sqrt(11.0)` with numeric key/value encodings before executable oracle
  comparison, for example `solution_norm=2.8982753492378879`;
- replace bare scalar `1.0` with `solution_norm=1.0`;
- replace bare comma-vector exact values with
  `solution_values=0.5,0.5,0.5,0.5` style key/value encodings;
- update the generic `value` comparison helper so it handles scalar and vector
  QR value rows without breaking existing partial-SVD `top_k` comparisons.

Failure classes:

- `fail_minnorm_status`
- `fail_minnorm_residual`
- `fail_minnorm_solution_norm`
- `fail_minnorm_solution_values`
- `fail_qr_probe_compile`
- `fail_qr_probe_runtime`

## Probe Implementation Rules

Day 7 should generalize the current temporary C probe instead of creating a
separate one-off path for each fixture.

Required probe inputs:

- fixture key;
- generated matrix entries from `GENERATED_FIXTURES`;
- fixture rows/columns;
- operation family: `rankdef_nullspace` or `minnorm_solve`;
- explicit RHS for minimum-norm fixtures.

Required probe outputs:

- `rank=<integer>` and `nullity=<integer>` for rank-deficient rows;
- `normalized_null_vector_residual=<float>` for rank-deficient rows;
- `projector_distance=<float>` for subspace rows;
- `status=<SPARSE_* symbol>`, `residual_norm=<float>`,
  `solution_norm=<float>`, and `solution_values=<comma-vector>` for
  minimum-norm rows.

The oracle should preserve the existing report metadata fields:

- command;
- source commit and branch;
- generated timestamp;
- platform and compiler;
- static-default configuration;
- fixture structure/value hashes;
- support tier;
- claim scope and non-claims.

## Tolerance Rationale

| Metric | Tolerance | Rationale |
| --- | --- | --- |
| Rank/nullity | exact `0` | Expected integer metadata for exact small generated fixtures. |
| Nullspace residual | absolute `1e-10` | Matches existing QR corpus residual contract and leaves room for QR roundoff. |
| Projector distance | projector `1e-8` | Subspace comparison is sign/order/orientation invariant and matches existing dense-reference projector tests. |
| Minimum-norm residual | absolute `1e-10` | Existing owner-local C tests assert exact small-system residuals at this scale. |
| Solution norm | absolute `1e-10` | Deterministic small fixtures have analytic norms; tolerance covers floating QR solve roundoff. |
| Solution values | absolute `1e-10` max component error | Exact rows are limited to stable deterministic fixtures already covered by owner-local tests. |

Downgrade rule: if Day 8-9 proof-owner tests show exact solution values are
platform-sensitive while residual and norm are stable, remove or defer the
`_solution_values` rows for that fixture before closeout.

## Claim Boundary

Allowed fixture-local claims:

- selected fixture shape, `nnz`, rank, nullity, and RHS metadata;
- selected fixture QR factorization success;
- selected fixture nullspace residual and projector distance;
- selected fixture minimum-norm solve status, residual, solution norm, and
  exact values where explicitly owned.

Rejected claims:

- raw Q/R basis equality;
- raw nullspace basis equality;
- sign, orientation, scale, or column-order parity;
- global rank-threshold policy;
- broad QR correctness;
- broad minimum-norm or least-squares guarantee;
- rank-deficient minimum-norm recovery;
- inconsistent-system behavior;
- external-library parity;
- platform, package, ABI, performance, or state-of-the-art claims.

## Day 7 Implementation Checklist

1. Add fixture-key tables for the five Sprint 150 QR fixtures.
2. Generalize solver QR probe generation so it can emit rank/nullity,
   nullspace residual, projector distance, minimum-norm status, residual,
   norm, and vector observations.
3. Extend `compare()` to support QR scalar/vector `value` rows while preserving
   partial-SVD `top_k` behavior.
4. Normalize minimum-norm expected rows to key/value encodings.
5. Add solver-backed oracle row-id mappings for all selected fixtures.
6. Run `python3 scripts/validate_corpus_schema.py`.
7. If the static library is available, run
   `python3 scripts/run_corpus_oracle.py --include-solver-qr`; otherwise record
   the build prerequisite in the Day 7 artifact.

## Completion Criteria Status

| Completion Criteria | Status | Evidence |
| --- | --- | --- |
| Oracle semantics are numerically meaningful and bounded. | Complete | Rank/nullity, residual, projector, status, solution-norm, and exact-value semantics are fixture-local with explicit tolerances. |
| Raw-basis identity claims are explicitly rejected. | Complete | Claim boundary rejects raw Q/R and nullspace basis equality, sign, orientation, scale, and column-order parity. |
| Each selected family has a concrete oracle rule and tolerance. | Complete | Rank-deficient rectangular and underdetermined minimum-norm fixtures have row-level semantics and tolerances. |
