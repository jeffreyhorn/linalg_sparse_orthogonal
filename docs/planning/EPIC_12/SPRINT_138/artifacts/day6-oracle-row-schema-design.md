# Sprint 138 Day 6 - Oracle Row Schema Design

## Purpose

Day 6 defines the oracle row schema, comparison kinds, tolerance policy,
failure classes, serialization rules, and validation expectations for the
Sprint 138 maintained corpus lane.

This is a design artifact. It does not add the maintained oracle schema file,
validation helper, generator, observed oracle rows, report command, tests, or
solver behavior changes. Day 7 owns the initial schema implementation.

## Row Families

The corpus/oracle lane uses three distinct row families:

| Row family | Maintained path | Generated path | Meaning |
| --- | --- | --- | --- |
| Fixture manifest row | `tests/corpus/manifests/fixtures.tsv` | None | Defines fixture identity, matrix class, expected behavior, support tier, validation command, claim scope, and non-claims. |
| Expected-result row | `tests/corpus/expected/<fixture_key>.tsv` | None | Defines expected fixture-local values, statuses, or tolerances before a run. |
| Observed oracle row | Future schema under `tests/corpus/schemas/oracle_fields.md` | `build/corpus/oracle/<fixture_key>.oracle.tsv` | Records one command's observed result, comparison status, failure class, support tier, command, commit, platform, and claim boundary. |

Expected-result rows are source-controlled target metadata. Observed oracle
rows are generated evidence and must stay under ignored `build/` unless a
later sprint explicitly promotes a stable example artifact.

## Oracle Row Schema

Observed oracle rows should use TSV serialization with one header row and one
row per comparison.

| Field | Required | Allowed values or format | Meaning |
| --- | --- | --- | --- |
| `oracle_row_id` | Yes | Stable lowercase snake case | Unique row ID, usually `<fixture_key>_<operation>_<comparison_kind>`. |
| `fixture_key` | Yes | Existing fixture manifest key | Connects the observed row to `tests/corpus/manifests/fixtures.tsv`. |
| `solver_family` | Yes | `qr`, `partial_svd`, `lu`, `ldlt`, `cholesky`, `iterative`, `eigs`, `runtime`, `package`, `unknown` | Solver or evidence family that owns the comparison. |
| `operation` | Yes | Lowercase snake case | Operation under test, such as `rank_info`, `nullspace`, `solve`, `singular_values`, `convergence_budget`, `diagnostic`, or `optional_data_check`. |
| `comparison_kind` | Yes | See comparison-kind table | Machine-readable comparison semantics. |
| `command` | Yes | Exact shell command | Command that produced the observed row. |
| `source_commit` | Yes | Git commit SHA or `unknown` | Commit used for the observed row. |
| `source_branch` | Yes | Branch, tag, `detached`, or `unknown` | Branch context for report freshness. |
| `generated_at_utc` | Yes | ISO-like UTC timestamp or `unknown` | Time the observed row was generated. |
| `platform` | Yes | OS and architecture or `unknown` | Platform context for support-tier and freshness. |
| `compiler` | Conditional | Compiler and version or `not_applicable` | Required for compiled solver rows; `not_applicable` for pure metadata rows. |
| `configuration` | Yes | Semicolon-separated stable key/value text | Build flags, optional-data state, backend state, and relevant runtime options. |
| `support_tier` | Yes | `reviewed_linux`, `reviewed_cross_platform`, `supplemental_macos`, `supplemental_windows`, `local_only`, `optional_data`, `staged` | Support meaning of the row. |
| `expected_result_kind` | Yes | See comparison-kind table | Type of expected result. |
| `expected_result` | Yes | Scalar, range, status, vector summary, or structured text | Expected value or condition. |
| `observed_result` | Yes | Scalar, range, status, vector summary, or structured text | Observed value or condition. |
| `tolerance_kind` | Yes | See tolerance policy | How to compare expected and observed values. |
| `tolerance_value` | Conditional | Numeric value, semicolon-separated numeric fields, or empty | Required for numeric tolerances; empty only for `status_only` or `not_applicable`. |
| `comparison_status` | Yes | `pass`, `fail`, `skip`, `defer`, `unsupported`, `xfail` | Outcome after comparison. |
| `failure_class` | Conditional | See failure-class table or empty | Required unless `comparison_status=pass`. |
| `skip_or_defer_reason` | Conditional | Short stable reason or empty | Required for `skip`, `defer`, and `unsupported`. |
| `claim_scope` | Yes | One-sentence fixture-local claim | What a passing row may support. |
| `non_claims` | Yes | Semicolon-separated boundaries | Claims this row does not support. |

## Comparison Kinds

| Comparison kind | Expected result kind | Typical operation | Meaning | First-lane use |
| --- | --- | --- | --- | --- |
| `value` | `value` | Generic scalar or vector check | Observed value must match expected value under the tolerance policy. | Not primary for Day 6 first lane. |
| `residual_norm` | `residual_norm` | `solve`, `factor`, `reconstruction` | Observed residual norm must be within absolute, relative, or mixed tolerance. | Possible later QR reconstruction or solve residual row. |
| `rank` | `rank` | `rank_info`, `factor` | Observed rank must match the expected integer exactly unless a future tolerance policy says otherwise. | Required for `qr_rank_deficient_6x4_nullspace_v1`. |
| `nullity` | `nullity` | `nullspace`, `rank_info` | Observed nullity must match the expected integer exactly. | Required for `qr_rank_deficient_6x4_nullspace_v1`. |
| `subspace_distance` | `subspace_distance` | `nullspace`, `range`, `partial_svd_vectors` | Observed projector or two-way projection distance must be within tolerance. Raw basis equality is not a valid primary comparison. | Required for future QR nullspace/subspace row. |
| `status` | `status` | `diagnostic`, `optional_data_check`, `unsupported_feature` | Observed status must equal expected status. | Used for skip/defer/unsupported rows. |
| `diagnostic` | `diagnostic` | Expected failure or error classification | Observed diagnostic status and failure class must match expected diagnostic. | Future direct-solver or parser residuals. |
| `local_measurement` | `performance_local` | Runtime/backend or benchmark context | Measurement row with local-only meaning; pass/fail only when an explicit threshold exists. | Not used by first QR lane. |

## Tolerance Policy

| Tolerance kind | Required `tolerance_value` | Use |
| --- | --- | --- |
| `exact` | Required; use `0` for exact integer equality | Rank, nullity, enum, and exact count comparisons. |
| `absolute` | Required numeric scalar | Absolute scalar/vector/residual comparisons. |
| `relative` | Required numeric scalar | Relative error comparisons when expected magnitude is meaningful. |
| `mixed` | Required as `absolute=<value>;relative=<value>` | Numeric comparisons needing both floor and scale-aware tolerance. |
| `projector` | Required numeric scalar | Subspace comparisons using projector distance or equivalent two-way projection metric. |
| `status_only` | Empty | Status, skip, defer, unsupported, and diagnostic rows where no numeric tolerance applies. |
| `not_applicable` | Empty | Metadata-only rows excluded from numeric comparison. |

Tolerance values must be fixture-local. A tolerance in an oracle row does not
define a global solver accuracy claim.

## Comparison Status Values

| Status | Meaning | Pass-count handling |
| --- | --- | --- |
| `pass` | Observed result satisfied expected result under the row tolerance and support tier. | May count only for the row's fixture-local claim scope. |
| `fail` | Observed result did not satisfy expected result or row integrity failed. | Failing validation result. |
| `skip` | Row was intentionally skipped, usually because optional data or a support-tier prerequisite is unavailable. | Never solver pass evidence. |
| `defer` | Row is intentionally not implemented yet. | Never solver pass evidence; must carry owner/prerequisite in reason text. |
| `unsupported` | Fixture, platform, feature, or data source is outside the current support tier. | Never solver pass evidence. |
| `xfail` | Known residual is expected to fail until a named sprint or owner closes it. | Never pass evidence; must have removal condition. |

## Failure Classes

| Failure class | Required status | Meaning | Required handling |
| --- | --- | --- | --- |
| `fail_oracle_mismatch` | `fail` | Observed numerical value, residual, rank, nullity, status, or subspace metric did not satisfy the expected row. | Treat as validation failure; do not update expected rows without implementation or tolerance review. |
| `fail_generator_mismatch` | `fail` | Regenerated fixture structure or values do not match expected hashes. | Treat as corpus integrity failure; update generator version and dependent rows together. |
| `fail_report_stale` | `fail` | Commit, command, platform, compiler, configuration, or generated timestamp is stale or incompatible. | Regenerate before using evidence for claims. |
| `fail_malformed_row` | `fail` | Required fields are missing, invalid, or have inconsistent TSV width. | Fix schema/data before interpreting the row. |
| `skip_optional_unavailable` | `skip` | Optional data is disabled, unavailable, unlicensed, unsupported, or unconfigured. | Record skip-policy evidence only. |
| `defer_not_implemented` | `defer` | Fixture or oracle row is intentionally defined but not implemented. | Keep out of pass counts; list owner and prerequisite. |
| `unsupported_platform` | `unsupported` | Platform or support tier is not supported by the row. | Preserve support boundary; do not promote platform claims. |
| `xfail_known_residual` | `xfail` | Known residual is expected to fail until a tracked owner removes the xfail. | Require owner, residual reference, and removal condition. |

## First-Lane Oracle Design

The first lane, `qr_rank_deficient_6x4_nullspace_v1`, needs at least these
oracle rows once implementation begins:

| Oracle row ID | Operation | Comparison kind | Expected result | Tolerance | Claim scope |
| --- | --- | --- | --- | --- | --- |
| `qr_rank_deficient_6x4_nullspace_v1_rank` | `rank_info` | `rank` | `3` | `exact=0` | Fixture-local QR rank behavior. |
| `qr_rank_deficient_6x4_nullspace_v1_nullity` | `rank_info` | `nullity` | `1` | `exact=0` | Fixture-local QR nullity behavior. |
| `qr_rank_deficient_6x4_nullspace_v1_projector_residual` | `nullspace` | `subspace_distance` | Projector or two-way projection distance below fixture-local tolerance. | `projector=TBD_DAY8_OR_DAY9` | Fixture-local QR nullspace/subspace residual behavior. |

Raw QR basis vectors should not be a primary expected result because QR basis
orientation and signs are not stable enough for a durable corpus contract.

## Serialization Rules

| Rule | Requirement |
| --- | --- |
| Format | TSV with one header row. |
| Encoding | UTF-8-compatible ASCII content unless a future artifact explicitly justifies non-ASCII. |
| Missing optional value | Empty field, not `NULL`. |
| Multiple values | Semicolon-separated stable key/value text when a structured value is needed. |
| Numeric formatting | Use enough precision to make fixture-local tolerance meaningful; avoid locale-dependent formatting. |
| Stable ordering | Sort observed rows by `oracle_row_id` unless a command's documented output order is part of the evidence. |
| Generated output path | `build/corpus/oracle/<fixture_key>.oracle.tsv`. |
| Expected-result path | `tests/corpus/expected/<fixture_key>.tsv`. |
| Schema path | `tests/corpus/schemas/oracle_fields.md` after Day 7 implementation. |

## Validation Expectations

Day 7 or later validation should check:

1. Every observed oracle row has the same number of TSV fields as the header.
2. Required fields are non-empty.
3. `fixture_key` exists in `tests/corpus/manifests/fixtures.tsv`.
4. `comparison_kind`, `expected_result_kind`, `tolerance_kind`,
   `comparison_status`, `failure_class`, and `support_tier` values are valid.
5. Numeric tolerance rows have non-empty `tolerance_value`; `status_only` and
   `not_applicable` rows have empty `tolerance_value`.
6. Non-pass rows have an allowed `failure_class`.
7. `skip`, `defer`, and `unsupported` rows have `skip_or_defer_reason`.
8. `pass` rows have empty `failure_class` and empty `skip_or_defer_reason`.
9. `claim_scope` and `non_claims` are non-empty for every row.
10. Skipped, deferred, unsupported, and xfail rows are excluded from pass
    counts and cannot promote solver behavior claims.

## Day 7 Implementation Handoff

| Task | Day 6 requirement |
| --- | --- |
| Add oracle schema file | Create `tests/corpus/schemas/oracle_fields.md` from this design. |
| Update expected placeholders | Replace `TBD_DAY6_ORACLE_SCHEMA` and `TBD_DAY6` in first-lane expected rows with schema-aligned placeholder values where possible. |
| Add validation helper or documented path | Implement or document a lightweight TSV/schema check that can validate row width and basic enum requirements. |
| Preserve generated-output boundary | Keep observed oracle rows under `build/corpus/oracle/` and out of source control. |
| Preserve non-claims | Keep fixture-local claim scope and non-claims required on every expected and observed row. |

## Day 6 Completion Criteria

| Criterion | Status | Evidence |
| --- | --- | --- |
| Oracle rows can represent the first corpus lane without ambiguity. | Complete | First-lane oracle design lists rank, nullity, and projector/subspace residual rows with comparison kinds and tolerance policies. |
| Skip, defer, and unsupported statuses are distinct from pass/fail. | Complete | Status table and failure-class table keep skip/defer/unsupported rows out of pass evidence. |
| Row fields preserve fixture-local claim boundaries. | Complete | Schema requires `claim_scope` and `non_claims` on every observed oracle row. |
