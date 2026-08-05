# Oracle Row Field Schema

Observed oracle rows record one generated comparison for one maintained corpus
fixture. They are generated evidence and should live under ignored
`build/corpus/oracle/`, not in this source-controlled schema directory.

Expected-result rows live in `tests/corpus/expected/` and define target values
or conditions before a run. A passing observed oracle row may support only the
row's fixture-local `claim_scope` and must preserve its `non_claims`.

## Observed Oracle TSV Fields

| Field | Required | Allowed values or format | Meaning |
| --- | --- | --- | --- |
| `oracle_row_id` | Yes | Stable lowercase snake case | Unique row ID, usually `<fixture_key>_<operation>_<comparison_kind>`. |
| `fixture_key` | Yes | Existing fixture manifest key | Connects the observed row to `tests/corpus/manifests/fixtures.tsv`. |
| `solver_family` | Yes | `qr`, `partial_svd`, `lu`, `ldlt`, `cholesky`, `iterative`, `eigs`, `runtime`, `package`, or `unknown` | Solver or evidence family that owns the comparison. |
| `operation` | Yes | Lowercase snake case | Operation under test, such as `rank_info`, `nullspace`, `solve`, `singular_values`, `convergence_budget`, `diagnostic`, or `optional_data_check`. |
| `comparison_kind` | Yes | `value`, `residual_norm`, `rank`, `nullity`, `subspace_distance`, `status`, `diagnostic`, or `local_measurement` | Machine-readable comparison semantics. |
| `command` | Yes | Exact shell command | Command that produced the observed row. |
| `source_commit` | Yes | Git commit SHA or `unknown` | Commit used for the observed row. |
| `source_branch` | Yes | Branch, tag, `detached`, or `unknown` | Branch context for report freshness. |
| `generated_at_utc` | Yes | UTC timestamp or `unknown` | Time the observed row was generated. |
| `platform` | Yes | OS and architecture or `unknown` | Platform context for support-tier and freshness. |
| `compiler` | Conditional | Compiler/version or `not_applicable` | Required for compiled solver rows; `not_applicable` for pure metadata rows. |
| `configuration` | Yes | Semicolon-separated stable key/value text | Build flags, optional-data state, backend state, and runtime options. |
| `support_tier` | Yes | `reviewed_linux`, `reviewed_cross_platform`, `supplemental_macos`, `supplemental_windows`, `local_only`, `optional_data`, or `staged` | Support meaning of the row. |
| `expected_result_kind` | Yes | `value`, `residual_norm`, `rank`, `nullity`, `subspace_distance`, `status`, `diagnostic`, or `performance_local` | Type of expected result. |
| `expected_result` | Yes | Scalar, range, status, vector summary, or structured text | Expected value or condition. |
| `observed_result` | Yes | Scalar, range, status, vector summary, or structured text | Observed value or condition. |
| `tolerance_kind` | Yes | `exact`, `absolute`, `relative`, `mixed`, `projector`, `status_only`, or `not_applicable` | How expected and observed values are compared. |
| `tolerance_value` | Conditional | Numeric value, stable key/value fields, or empty | Required for numeric tolerances; empty for `status_only` and `not_applicable`. |
| `comparison_status` | Yes | `pass`, `fail`, `skip`, `defer`, `unsupported`, or `xfail` | Outcome after comparison. |
| `failure_class` | Conditional | Allowed failure class or empty | Required unless `comparison_status=pass`. |
| `skip_or_defer_reason` | Conditional | Stable reason or empty | Required for `skip`, `defer`, and `unsupported`. |
| `claim_scope` | Yes | One-sentence fixture-local claim | What a passing row may support. |
| `non_claims` | Yes | Semicolon-separated boundaries | Claims this row does not support. |

## Comparison Status

| Status | Meaning | Pass-count handling |
| --- | --- | --- |
| `pass` | Observed result satisfied expected result under the row tolerance and support tier. | May count only for the row's fixture-local claim scope. |
| `fail` | Observed result did not satisfy expected result or row integrity failed. | Failing validation result. |
| `skip` | Row was intentionally skipped because optional data or a prerequisite is unavailable. | Never solver pass evidence. |
| `defer` | Row is intentionally not implemented yet. | Never solver pass evidence. |
| `unsupported` | Fixture, platform, feature, or data source is outside the current support tier. | Never solver pass evidence. |
| `xfail` | Known residual is expected to fail until a tracked owner closes it. | Never solver pass evidence. |

## Failure Classes

| Failure class | Required status | Meaning |
| --- | --- | --- |
| `fail_oracle_mismatch` | `fail` | Observed numerical value, residual, rank, nullity, status, or subspace metric did not satisfy the expected row. |
| `fail_generator_mismatch` | `fail` | Regenerated fixture structure or values do not match expected hashes. |
| `fail_report_stale` | `fail` | Commit, command, platform, compiler, configuration, or generated timestamp is stale or incompatible. |
| `fail_malformed_row` | `fail` | Required fields are missing, invalid, or have inconsistent TSV width. |
| `skip_optional_unavailable` | `skip` | Optional data is disabled, unavailable, unlicensed, unsupported, or unconfigured. |
| `defer_not_implemented` | `defer` | Fixture or oracle row is intentionally defined but not implemented. |
| `unsupported_platform` | `unsupported` | Platform or support tier is not supported by the row. |
| `xfail_known_residual` | `xfail` | Known residual is expected to fail until a tracked owner removes the xfail. |

## First-Lane Expected Oracle Rows

The first durable lane reserves these row IDs:

| Oracle row ID | Operation | Comparison kind | Expected result |
| --- | --- | --- | --- |
| `qr_rank_deficient_6x4_nullspace_v1_rank` | `rank_info` | `rank` | `3` |
| `qr_rank_deficient_6x4_nullspace_v1_nullity` | `rank_info` | `nullity` | `1` |
| `qr_rank_deficient_6x4_nullspace_v1_projector_residual` | `nullspace` | `subspace_distance` | `projector_distance<=1e-10` |

Raw QR basis vectors are not a primary expected-result artifact for this lane.

## Maintained Command

Run the first maintained corpus/oracle command with:

```sh
python3 scripts/run_corpus_oracle.py
```

It validates the source-controlled corpus metadata, writes observed oracle rows
to `build/corpus/oracle/`, and writes a report index to
`build/corpus-reports/`. These generated outputs remain local evidence and are
not committed.
