# Sprint 139 Day 4: Fixture Batch Design

## Purpose

Day 4 specifies the deterministic QR fixture batch for the selected Sprint 139
closure. The design keeps the first-class source-controlled evidence narrow:
`qr_rank_deficient_6x4_nullspace_v1` remains the maintained fixture to close.
Success, diagnostic failure, and tolerance-boundary behavior are mapped without
silently adding broad QR, SuiteSparse, minimum-norm, least-squares, or
rank-threshold claims.

This is a design artifact. It does not change corpus rows, QR source, tests,
oracle commands, generated outputs, public documentation, or support tiers.

## Batch Decision

Sprint 139 uses one first-class source-controlled fixture for closure:

`qr_rank_deficient_6x4_nullspace_v1`

Rationale:

- Sprint 138 already created this fixture, generator metadata, expected rows,
  schema validation, oracle/report command, and handoff documentation.
- Day 2 selected this residual because it can be closed fully without broad
  claim widening.
- Adding several new source-controlled QR fixtures before the first closure
  would dilute the sprint objective and risk partial progress across more
  gaps.

Failure and tolerance-boundary behavior should be represented by focused test
assertions and oracle failure semantics around this same fixture unless a later
day finds a concrete implementation blocker.

## First-Class Fixture Row

| Field | Planned value |
| --- | --- |
| `fixture_key` | `qr_rank_deficient_6x4_nullspace_v1` |
| `fixture_family` | `qr_rank_deficient` |
| `storage_kind` | `generated` |
| `matrix_path` | empty |
| `generator_key` | `qr_rank_deficient_6x4_nullspace_generator_v1` |
| `rows` | `6` |
| `cols` | `4` |
| `nnz` | `14` |
| `symmetry` | `none` |
| `definiteness` | `rectangular` |
| `rank_status` | `rank_deficient` |
| `expected_rank` | `3` |
| `nullity` | `1` |
| `conditioning_class` | `moderate` |
| `scale_class` | `unit` |
| `sparsity_class` | `structured_sparse` |
| `rhs_policy` | `generated_rhs` in the existing row; Sprint 139 closure uses matrix/nullspace behavior only |
| `expected_behavior` | `success` |
| `support_tier` | `local_only` until reviewed evidence promotes it |
| `validation_command` | `python3 scripts/run_corpus_oracle.py` plus the focused QR proof command once implemented |

Day 5 should not create a second first-class source-controlled QR fixture
unless Day 5 discovers that the existing row contradicts the selected closure.

## Generator Row

| Field | Planned value |
| --- | --- |
| `generator_key` | `qr_rank_deficient_6x4_nullspace_generator_v1` |
| `generator_version` | `1` |
| `algorithm` | `fixed_columns_c3_equals_c0_plus_c1` |
| `seed` | `none` |
| `parameters` | `rows=6;cols=4;expected_rank=3;nullity=1;dependency=c3-c0-c1` |
| `expected_structure_hash` | existing Sprint 138 hash |
| `expected_value_hash` | existing Sprint 138 hash |
| `canonical_format` | `coo_zero_based_row_col_value_f64_text_v1` |
| `floating_policy` | exact integer structure/values for generation; exact rank/nullity; normalized null-vector residual tolerance `1e-10` |
| `regeneration_command` | `python3 scripts/run_corpus_oracle.py` |
| `change_policy` | update generator version, fixture metadata, expected results, oracle rows, validation command, and docs together |

The Day 9 C fixture helper should mirror this generator exactly. It should not
be treated as an independent generator with separate semantics.

## Expected Matrix Shape

Columns:

```text
c0 = [1, 0, 0, 1, 0, 1]^T
c1 = [0, 1, 0, 1, 1, 0]^T
c2 = [0, 0, 1, 0, 1, 1]^T
c3 = [1, 1, 0, 2, 1, 1]^T = c0 + c1
```

Expected consequences:

- `c0`, `c1`, and `c2` are independent.
- `c3 = c0 + c1`.
- rank is exactly `3`.
- nullity is exactly `1`.
- `[-1, -1, 0, 1]` is a valid right-nullspace vector.

## Expected-Result Row Plan

| Row ID | Short form | Operation | Comparison kind | Expected result | Tolerance | Claim scope |
| --- | --- | --- | --- | --- | --- | --- |
| `qr_rank_deficient_6x4_nullspace_v1_rank` | `rank` | `rank_info` | `rank` | `3` | `exact=0` | fixture-local generated reference rank metadata; later solver-backed QR rank evidence when a QR-owned row emits the observation |
| `qr_rank_deficient_6x4_nullspace_v1_nullity` | `nullity` | `rank_info` | `nullity` | `1` | `exact=0` | fixture-local generated reference nullity metadata; later solver-backed QR nullity evidence when a QR-owned row emits the observation |
| `qr_rank_deficient_6x4_nullspace_v1_projector_residual` | `projector_residual` | `nullspace` | `residual_norm` | `normalized_null_vector_residual<=1e-10` | `absolute=1e-10` | fixture-local normalized null-vector residual evidence |

The row IDs are already unique and follow the current oracle schema convention:
use `<fixture_key>_<comparison_kind>` when unambiguous and include operation
detail when needed to disambiguate.

## Behavior Coverage Map

| Behavior | Source-controlled fixture? | Planned mechanism | Pass/fail interpretation |
| --- | --- | --- | --- |
| Success | Yes, `qr_rank_deficient_6x4_nullspace_v1` | Focused QR proof builds the fixture, factors it, checks rank, nullity, and normalized residual. | Passing evidence closes only the selected fixture-local QR residual. |
| Diagnostic failure | No new fixture row | Focused test/oracle implementation should fail closed if factorization, nullity query, basis extraction, zero-norm basis, or malformed expected row occurs. | Failure is a test/oracle failure, not unsupported pass evidence. |
| Tolerance boundary | Same fixture row | Residual tolerance remains `absolute=1e-10`; the focused proof should compute and print the observed residual. | Residual above tolerance fails the selected closure. |
| Raw-basis variation | Same fixture row | Do not compare the raw vector to `[-1, -1, 0, 1]`; compare residual and nullity. | Equivalent basis choices pass when residual criteria pass. |
| Optional SuiteSparse external data | No source-controlled pass fixture | Existing optional-data row stays disabled/skipped. | Skip/defer remains policy evidence only. |

## Staged and Deferred Fixture Candidates

| Candidate | Status | Reason |
| --- | --- | --- |
| `qr_rankdef_duplicate_5x4_nullspace_projector` | Backup only | Existing bounded projector evidence can guide proof shape but is not the maintained Sprint 138 corpus fixture. |
| near-dependent rank-threshold variants | Deferred | Would reopen global rank-threshold policy, which Day 2 deferred. |
| rectangular least-squares fixtures | Deferred | Existing solve evidence is bounded and outside the selected nullspace closure. |
| minimum-norm fixtures | Deferred | Minimum-norm correctness remains a separate bounded solve claim, not this fixture-local nullspace closure. |
| COLAMD/reordered QR fixture | Deferred | Adds reorder/permutation semantics outside the selected residual. |
| SuiteSparse rank-deficient QR subset | Deferred/optional | Optional external data remains disabled by default and lacks reviewed evidence. |

Day 5 should implement only the first-class selected fixture batch unless a
blocking contradiction is discovered and recorded.

## Claim and Non-Claim Table

| Evidence row or mechanism | May claim after passing solver-backed proof | Must not claim |
| --- | --- | --- |
| rank row | QR reports rank `3` on `qr_rank_deficient_6x4_nullspace_v1`. | global rank-threshold policy or broad QR rank correctness |
| nullity row | QR reports nullity `1` on `qr_rank_deficient_6x4_nullspace_v1`. | broad nullspace correctness or raw-basis parity |
| residual row | QR produces a nonzero nullspace vector whose normalized residual is `<= 1e-10` on the fixture. | broad rank-deficient solve, least-squares, minimum-norm, projector parity, SuiteSparse parity, or external-library parity |
| skipped optional-data row | optional SuiteSparse data is disabled/unavailable/deferred. | solver pass evidence |

## Day 5 Implementation Instructions

Day 5 should:

1. Re-run `python3 scripts/validate_corpus_schema.py`.
2. Confirm the existing fixture, generator, and expected-result rows already
   match this Day 4 design.
3. Avoid adding new source-controlled QR fixture rows unless there is a
   documented mismatch.
4. If row wording changes are needed, keep claim scopes fixture-local and
   preserve all broad QR/SuiteSparse/minimum-norm non-claims.
5. Record whether Day 5 is a no-row-change confirmation or a narrow metadata
   correction.

## Day 4 Completion Criteria

| Criterion | Status | Evidence |
| --- | --- | --- |
| Every planned fixture has a precise QR behavior and claim scope. | Complete | The first-class fixture row, generator row, expected-result row plan, behavior coverage map, and claim table define the selected behavior. |
| Row IDs can be validated without ambiguity. | Complete | The expected-result row table preserves unique row IDs and short forms. |
| No fixture implies broad QR, SuiteSparse, or corpus completeness. | Complete | Staged/deferred fixture notes and non-claim table fence broader QR, optional-data, and parity claims. |
