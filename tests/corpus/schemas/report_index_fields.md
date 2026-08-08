# Report Index Metadata Contract

This schema documents the source-controlled report-family contract used by
the normalized report index work. The contract describes row meanings,
freshness policy, support tier, generator command, and non-claim boundaries.
It does not create generated evidence and does not turn local reports into
release proof.

## Manifest

Rows live in `tests/corpus/manifests/report_families.tsv` and are validated by
`scripts/validate_corpus_schema.py`.

## Fields

| Field | Required | Meaning |
| --- | --- | --- |
| `report_family` | Yes | Stable lowercase snake-case family, such as `corpus`, `oracle`, `benchmark`, `sentinel`, `guardrail`, `deadcode`, `coverage`, `package`, `ci`, `documentation`, `report_index`, or `runtime_backend`. |
| `subfamily` | Yes | Stable lowercase snake-case subdivision within the family. |
| `row_meaning` | Yes | Controlled row-meaning vocabulary that the normalized index can preserve honestly. |
| `row_origin` | Yes | One of `source_controlled`, `generated_local`, `generated_ci`, `external_optional`, or `documentation`. |
| `status` | Yes | Default contract status. Contract rows must not use `pass`; observed generated rows own pass/fail status. |
| `support_tier` | Yes | Existing corpus support tier vocabulary. |
| `freshness_policy` | Yes | How freshness should be interpreted for the row family. |
| `generator_command` | Yes | Maintained command or owner that can generate or validate the row family. |
| `artifact_pattern` | Yes | Source-controlled or generated path pattern indexed by the row family. |
| `claim_scope` | Yes | The narrow claim a row in this family may support when valid. |
| `non_claims` | Yes | Semicolon-separated boundaries that the normalized index must preserve. |
| `owner` | Yes | Maintainer or subject-matter owner for the family. |
| `introduced_in` | Yes | Planning provenance for the contract row. |

## Freshness Policies

| Policy | Meaning |
| --- | --- |
| `source_controlled` | Freshness is governed by source-control state and schema validation. |
| `generated_compare_inputs` | Generated rows must be compared with recorded command, commit, platform, compiler, configuration, and source inputs. |
| `generated_local_advisory` | Local generated rows can be stale or absent without failing release checks unless a later gate explicitly requires them. |
| `hosted_ci_external` | The source-controlled row identifies the lane; hosted logs remain external evidence. |
| `optional_data_skip` | Skip/defer interpretation depends on optional-data availability and must not count as pass evidence. |
| `deferred_governance` | Row meaning is acknowledged but policy closure belongs to a later sprint. |

## Guardrails

- Source-controlled contract rows are advisory or deferred; they are not pass
  evidence.
- Missing generated rows must be represented as `not_generated` or
  `unknown`, not silently omitted when the family is selected.
- Benchmark, sentinel, coverage, and dead-code rows remain local or advisory
  unless a reviewed lane explicitly promotes them.
- Package/install rows identify maintained proof owners and static-first
  scope; they do not claim package-manager availability, shared-library ABI,
  or broad platform support.
- Runtime/backend governance rows stay deferred to Sprint 142 when policy
  decisions are needed.
