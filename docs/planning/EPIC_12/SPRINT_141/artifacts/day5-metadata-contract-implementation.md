# Day 5 Metadata Contract Implementation

## Purpose

Day 5 implements the source-controlled metadata contract needed by the
normalized report index. The implementation gives Sprint 141 a mechanically
validated vocabulary for report families, row meanings, support tiers,
freshness policies, commands, artifact patterns, claim scopes, and non-claim
boundaries before the Day 6 generator starts emitting normalized rows.

## Implemented Surfaces

| Surface | Change | Purpose |
| --- | --- | --- |
| `tests/corpus/manifests/report_families.tsv` | Added report-family contract rows. | Defines normalizable report families and row meanings without generating evidence. |
| `tests/corpus/schemas/report_index_fields.md` | Added field and freshness-policy schema documentation. | Documents the metadata contract consumed by the future normalized index generator. |
| `scripts/validate_corpus_schema.py` | Added required fields, controlled vocabularies, duplicate-key checks, enum validation, snake-case checks, and false-pass guardrails. | Makes the metadata contract mechanically checkable with the existing corpus validation command. |
| `tests/corpus/README.md` | Added layout, ownership, and interpretation notes for report-family contract rows. | Keeps maintainer documentation aligned with the new contract surface. |

## Report-Family Coverage

| Report family | Implemented row meanings | Default interpretation |
| --- | --- | --- |
| `corpus` | `fixture_metadata`, `generator_metadata`, `optional_data_policy`, `expected_result` | Source-controlled metadata and target rows; no observed pass evidence. |
| `oracle` | `observed_oracle_comparison`, `solver_backed_fixture_proof` | Local generated comparison rows; fixture-local only when generated and valid. |
| `benchmark` | `benchmark_measurement` | Local advisory measurement rows. |
| `sentinel` | `sentinel_hard_gate`, `sentinel_advisory_measurement` | Bounded local hard gate or advisory measurements, depending on subfamily. |
| `guardrail` | `guardrail_lane` | Local structural or bounded large-matrix guardrail rows. |
| `deadcode` | `deadcode_classification` | Local maintainer classification output, not a zero-findings guarantee. |
| `coverage` | `coverage_summary` | Local tool output for the current runner/backend only. |
| `package` | `package_install_proof_owner` | Source-controlled static-first proof-owner command metadata. |
| `ci` | `ci_lane_definition` | Source-controlled hosted-lane definitions; logs remain external. |
| `documentation` | `documentation_advisory` | Maintained interpretation anchors, not executable proof. |
| `report_index` | `not_generated` | Explicit missing-generated-row semantics. |
| `runtime_backend` | `deferred_governance` | Sprint 142 handoff for runtime/backend policy decisions. |

## Controlled Vocabularies

The validator now checks these Day 3 contract vocabularies:

| Vocabulary | Values |
| --- | --- |
| Row origins | `source_controlled`, `generated_local`, `generated_ci`, `external_optional`, `documentation` |
| Statuses | `pass`, `fail`, `skip`, `defer`, `unsupported`, `xfail`, `unknown`, `advisory` |
| Freshness policies | `source_controlled`, `generated_compare_inputs`, `generated_local_advisory`, `hosted_ci_external`, `optional_data_skip`, `deferred_governance` |
| Support tiers | Existing corpus support tiers: `reviewed_linux`, `reviewed_cross_platform`, `supplemental_macos`, `supplemental_windows`, `local_only`, `optional_data`, and `staged` |

Contract rows may include the full status vocabulary for future generated-row
compatibility, but the source-controlled manifest rejects `status=pass`.
Observed generated rows own pass/fail status.

## Mechanical Validation

`scripts/validate_corpus_schema.py` now validates `report_families.tsv` by:

- requiring all contract fields;
- checking TSV width and duplicate headers through the existing TSV reader;
- enforcing unique `(report_family, subfamily, row_meaning)` identities;
- enforcing lowercase snake-case identity fields;
- enforcing row-origin, row-meaning, status, support-tier, and
  freshness-policy enums;
- rejecting source-controlled contract rows that use `status=pass`;
- rejecting claim scopes that assert state-of-the-art status;
- requiring non-claim boundaries.

## Claim Boundaries Preserved

The implemented rows explicitly preserve the Sprint 141 non-claims:

- no source-controlled contract row is pass evidence;
- missing generated rows become explicit `not_generated` or `unknown`
  semantics rather than implicit success;
- benchmark, sentinel, coverage, and dead-code rows remain local or advisory
  unless a reviewed gate later promotes them;
- package/install rows remain static-first proof-owner metadata and do not
  claim package-manager, shared-library ABI, or broad platform support;
- CI rows identify source-controlled lane definitions but not external hosted
  logs;
- runtime/backend governance remains a Sprint 142 handoff.

## Validation Evidence

Commands run:

```sh
python3 -m py_compile scripts/validate_corpus_schema.py
python3 scripts/validate_corpus_schema.py
```

Both commands passed.

## Day 6 Handoff

The Day 6 normalized index generator should read
`tests/corpus/manifests/report_families.tsv` as the canonical source for
family definitions. Generated row ingestion should map native report rows onto
these contract definitions and should fail if it would need a row meaning,
status, support tier, or freshness policy outside the validated vocabulary.

## Completion Criteria

| Criterion | Status | Evidence |
| --- | --- | --- |
| The metadata contract can be validated mechanically. | Complete | `scripts/validate_corpus_schema.py` validates `report_families.tsv` fields, enums, identity, and false-pass guardrails. |
| Existing corpus/oracle checks continue to pass. | Complete | `python3 scripts/validate_corpus_schema.py` passed with existing fixture, generator, optional-data, expected-result, and new report-family rows. |
| No field implies unsupported performance, package, or platform claims. | Complete | Manifest rows use advisory/unknown/defer statuses and explicit non-claim boundaries; source-controlled `status=pass` is rejected. |
