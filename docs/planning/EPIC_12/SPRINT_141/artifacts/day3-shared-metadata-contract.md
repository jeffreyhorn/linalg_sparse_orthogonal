# Day 3 Shared Metadata Contract

## Purpose

Day 3 defines the shared report metadata contract for Sprint 141 before any
generator or freshness-gate implementation begins. The contract must preserve
existing Sprint 138-140 corpus/oracle row meaning while making benchmark,
sentinel, guardrail, coverage, dead-code, package, install, and CI rows
discoverable without overclaiming their evidence.

This artifact is a design contract. It does not change schemas, scripts,
generated reports, CI workflows, tests, package behavior, or public claims.

## Contract Principles

1. Preserve native row meaning first. A normalized index may add common fields,
   but it must not collapse materially different row semantics into one
   pass/fail interpretation.
2. Keep native row IDs. The normalized index gets its own `row_id`, while
   family-specific identity is preserved in `native_row_id`.
3. Separate source-controlled metadata from generated evidence. Committed
   manifests, expected rows, workflow YAML, package templates, and docs define
   eligible evidence; generated reports record observed evidence for a command,
   commit, platform, compiler, configuration, and support tier.
4. Treat local measurements as local measurements. Benchmark, sentinel,
   coverage, and dead-code outputs need freshness and context fields, not
   portable release claims.
5. Prefer explicit `skip`, `defer`, and `unsupported` rows over missing or
   fabricated pass rows.
6. Carry runtime/backend fields as metadata in Sprint 141, but hand broad
   runtime/backend governance to Sprint 142.

## Common Fields

These fields form the normalized index contract.

| Field | Required for normalized row | Meaning |
| --- | --- | --- |
| `row_id` | Yes | Stable normalized row ID, unique within the normalized index. |
| `report_family` | Yes | Top-level enum: `corpus`, `oracle`, `benchmark`, `sentinel`, `guardrail`, `coverage`, `deadcode`, `package`, `install`, `ci`, `documentation`. |
| `subfamily` | Yes | Family-specific grouping such as `fixture_manifest`, `expected_result`, `canonical`, `wall_check`, `large_matrix`, `lcov`, `gcovr`, `pkg_config`, `cmake_export`, or `workflow_lane`. |
| `native_row_id` | Yes | Original family identity, such as `fixture_key`, `oracle_row_id`, `artifact`, `sentinel_id:metric`, `lane_id`, package assertion name, or workflow job name. |
| `row_origin` | Yes | `source_controlled`, `generated_local`, `generated_ci`, `hosted_ci`, `external_optional`, or `documentation`. |
| `row_meaning` | Yes | Controlled row-meaning value from the taxonomy below. |
| `status` | Yes | `pass`, `fail`, `report`, `skip`, `defer`, `unsupported`, `xfail`, `current`, `stale`, or `unknown`. |
| `status_reason` | Conditional | Required for non-pass, stale, skip, defer, unsupported, xfail, or unknown rows. |
| `support_tier` | Yes | Existing support-tier vocabulary plus lane-specific values when needed: `reviewed_linux`, `reviewed_cross_platform`, `supplemental_macos`, `supplemental_windows`, `local_only`, `optional_data`, `staged`, `advisory`. |
| `claim_scope` | Yes | What the row may support when interpreted as current and passing. |
| `non_claims` | Yes | Semicolon-separated boundaries the row must not be used to claim. |
| `generator_command` | Conditional | Required for generated rows; for source-controlled rows use the validation or regeneration command when one exists. |
| `source_commit` | Conditional | Required for generated rows; optional for source-controlled rows. |
| `source_branch` | Conditional | Required for generated rows; optional for source-controlled rows. |
| `generated_at_utc` | Conditional | Required for generated rows; empty or `not_applicable` for source-controlled rows. |
| `platform` | Conditional | Required for generated compiled/tool rows; `not_applicable` for pure source metadata. |
| `compiler` | Conditional | Required for compiled solver/package/coverage/benchmark rows; `not_applicable` for pure metadata rows. |
| `configuration` | Yes | Stable key/value text for relevant build flags, backend settings, optional-data state, package mode, or report mode. |
| `artifact_path` | Conditional | Required when the row points to a generated or source-controlled artifact. |
| `freshness_status` | Yes | `source_controlled`, `fresh`, `stale`, `not_generated`, `unknown`, `not_applicable`, or `deferred`. |
| `freshness_reason` | Yes | Short explanation of the freshness state. |
| `skip_or_defer_reason` | Conditional | Required for `skip`, `defer`, and `unsupported`. |

## Row-Meaning Taxonomy

| Row meaning | Applies to | Interpretation |
| --- | --- | --- |
| `fixture_metadata` | corpus fixture manifests | Defines an eligible fixture lane; not observed pass evidence. |
| `generator_metadata` | corpus generator manifests | Defines deterministic generator metadata and hashes; not observed solver evidence. |
| `expected_result` | corpus expected TSVs | Defines target comparison rows; not observed pass evidence. |
| `optional_data_policy` | optional-data manifest and generated skip rows | Records availability, skip/defer policy, and claim boundaries; never solver pass evidence. |
| `observed_oracle_comparison` | generated oracle rows | Local observed comparison tied to command, commit, platform, compiler, configuration, support tier, and claim scope. |
| `solver_backed_fixture_proof` | opt-in QR/partial-SVD oracle rows and focused proof owners | Fixture-local solver evidence only. |
| `benchmark_measurement` | canonical benchmark CSV/index rows | Threshold-free local measurement snapshot, not portable performance proof. |
| `sentinel_hard_gate` | performance sentinel `S5` wall-check rows | Enforced local gate row with threshold semantics. |
| `sentinel_advisory_measurement` | performance sentinel `S2` rows | Threshold-free local report context. |
| `guardrail_reviewed_lane` | large-matrix `G1`-`G4` rows | Reviewed structural or CSV-shape guardrail evidence. |
| `guardrail_supplemental_lane` | large-matrix `S1`/`S2` rows | Optional threshold-free context; skip unless enabled. |
| `deadcode_classification` | dead-code report rows | Static-analysis classification/completeness evidence, not zero-findings or removal-ready proof. |
| `coverage_summary` | lcov/gcovr summaries | Backend/tool/platform-specific coverage signal. |
| `package_contract_proof` | static package deferral and package metadata rows | Static-first package-contract evidence, scoped to command/platform. |
| `install_consumer_proof` | Make/CMake install tests and CI snippets | Installed static library/package consumer proof, scoped to platform and toolchain. |
| `ci_lane_definition` | workflow YAML job definitions | Source-controlled lane intent and support-tier split; not a job-result row by itself. |
| `documentation_advisory` | README, maintainer, cookbook, benchmark, install docs | Interpretation guidance and non-claim wording. |

## Field Requirement Matrix

| Field group | Required fields | Conditional or optional fields |
| --- | --- | --- |
| Common all rows | `row_id`, `report_family`, `subfamily`, `native_row_id`, `row_origin`, `row_meaning`, `status`, `support_tier`, `claim_scope`, `non_claims`, `configuration`, `freshness_status`, `freshness_reason` | `status_reason`, `skip_or_defer_reason`, `artifact_path` |
| Generated rows | all common fields plus `generator_command`, `source_commit`, `source_branch`, `generated_at_utc`, `platform` | `compiler`, input hashes, row count, artifact hash, exit code |
| Source-controlled rows | all common fields | validation command, owning file, source path, expected hash, introduced sprint |
| Compiled/tool rows | all common fields plus `platform`, `compiler`, `configuration` | build mode, OpenMP state, backend request/selected/fallback |
| Measurement rows | all generated fields plus metric name/value/unit and measurement context | baseline, threshold, repeat count, fixture/matrix, wall-time notes |
| Package/install rows | command, platform, package surface, static/shared scope, artifact path or assertion name | install prefix, tool version, package version, exact-version behavior |
| CI lane rows | workflow path, job name, runner OS, reviewed/supplemental scope | latest run URL/status only if a generated source exists |

## Freshness Semantics

| Freshness status | Meaning | Gate behavior |
| --- | --- | --- |
| `source_controlled` | Row is committed metadata, schema, workflow, script, or documentation. | Validate shape and references; freshness is source revision itself. |
| `fresh` | Generated row matches current command, source commit or accepted source context, expected input hashes, platform/toolchain fields, and support tier. | May pass freshness checks if row meaning supports it. |
| `stale` | Generated row predates or mismatches source, command, fixture, expected row, generator, package metadata, platform, compiler, configuration, or support tier requirements. | Error for enforced/generated proof rows; warning for local/advisory measurement rows unless explicitly promoted. |
| `not_generated` | Source row exists but generated output has not been produced. | Warning, skip, or defer depending on family and required lane. |
| `unknown` | Required freshness context is missing. | Error for enforced proof rows; warning for advisory rows. |
| `not_applicable` | Freshness does not apply, usually pure source metadata. | No freshness failure. |
| `deferred` | Freshness cannot be evaluated until a future sprint or external input exists. | Defer row with owner/reason. |

### Freshness Inputs

Generated rows should preserve the smallest honest set of freshness inputs:

- generator command and arguments;
- source commit and branch;
- generated timestamp;
- platform and architecture;
- compiler and relevant tool versions;
- build configuration and backend/runtime settings;
- source-controlled manifest, expected-row, schema, generator, package, or
  workflow inputs;
- support tier and row meaning;
- optional-data availability state;
- artifact path and row count when applicable.

Sprint 141 should not require generated local measurement outputs to be
committed. The freshness gate should compare current source-controlled inputs
against generated rows when rows exist, and produce `not_generated`, `skip`, or
`defer` states when local rows do not exist.

## Validation Severity Model

| Severity | Use when | Example |
| --- | --- | --- |
| `error` | The row is malformed, contradicts source-controlled metadata, fails an enforced proof, or stale state invalidates a required proof row. | Corpus solver-backed row for current fixture has wrong expected hash or non-pass status. |
| `warning` | The row is advisory, local, threshold-free, or missing freshness context that should not fail the current reviewed path. | Canonical benchmark snapshot is absent or stale. |
| `defer` | The row is intentionally not implemented or is assigned to a later sprint. | Runtime/backend governance row awaiting Sprint 142. |
| `skip` | Optional data, fixture, binary, tool, or platform prerequisite is unavailable. | Optional SuiteSparse corpus payload absent. |
| `unsupported` | Platform, feature, package mode, or report family is outside the current support tier. | Windows Make install validation lane. |
| `advisory` | Documentation or local interpretation row that guides users but should not be counted as proof. | Benchmark README report-index handoff paragraph. |

## Answers To Day 2 Contract Questions

1. Use a small `report_family` enum plus `subfamily`; do not encode all
   meaning in one family string.
2. Use normalized `row_id` and preserve native identity in `native_row_id`.
3. Require command, source commit, branch, generated time, platform, compiler,
   configuration, support tier, status, row meaning, claim scope, non-claims,
   freshness status, and freshness reason for generated rows.
4. Compare source-controlled input hashes and metadata to generated rows when
   generated rows exist; do not require local measurement rows to be committed.
5. Preserve optional data as `skip` when unavailable/disabled and `defer` when
   intentionally unimplemented or policy-blocked.
6. Fail staleness for enforced proof rows; warn for local measurement and
   supplemental report rows unless a reviewed lane explicitly makes them
   required.
7. Carry runtime/backend request, selected backend, fallback, dense-kernel, and
   build-mode fields as family-specific metadata while deferring policy
   interpretation to Sprint 142.

## Non-Normalizable Defer Path

Rows that cannot preserve meaning under the common contract must still appear
as explicit normalized rows:

- `status=defer`;
- `freshness_status=deferred`;
- `support_tier=staged` or `advisory`;
- `row_meaning` set to the closest honest taxonomy value;
- `skip_or_defer_reason` naming the missing policy, data, platform, or
  implementation owner;
- `claim_scope` limited to "deferred report-family metadata only";
- `non_claims` stating that the defer row is not pass evidence.

This prevents non-normalizable families from silently disappearing from the
index while still preventing false pass evidence.

## Day 3 Completion Criteria

| Criterion | Status | Evidence |
| --- | --- | --- |
| The contract can represent Sprint 138-140 corpus/oracle rows without losing meaning. | Complete | Common fields preserve native row IDs, support tier, claim scope, non-claims, generated command, and freshness metadata already present in corpus/oracle rows. |
| Benchmark and sentinel measurements are not framed as release proof. | Complete | Measurement rows have local/advisory row meanings and warning-oriented freshness semantics unless an enforced gate owns them. |
| Non-normalizable families have an explicit defer path. | Complete | Defer path defines status, freshness, support tier, reason, claim scope, and non-claims for rows that cannot yet be normalized honestly. |
