# Report Index Metadata Contract

This schema documents the source-controlled report-family contract used by
the normalized report index work. The contract describes row meanings,
freshness policy, support tier, generator command, and non-claim boundaries.
It does not create generated evidence and does not turn local reports into
release proof.

## Manifests

Report-family rows live in `tests/corpus/manifests/report_families.tsv`.
Selected target rows live in
`tests/corpus/manifests/selected_report_targets.tsv`. Both manifests are
validated by `scripts/validate_corpus_schema.py`.

`report_families.tsv` remains the broad family authority for row meanings,
freshness-policy vocabulary, default support tiers, family-level claims, and
family-level non-claims. `selected_report_targets.tsv` is the source of truth
for the selected oracle, comparison, and benchmark target list consumed by
report and workflow guards. Selected target rows narrow existing report-family
semantics to named targets; they do not widen support claims or turn
unselected families into generated pass evidence.

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
| `runtime_backend_governance_policy` | Source-controlled runtime/backend rows identify maintained control-boundary policy; generated sentinel measurements stay under the `sentinel` family. |
| `deferred_governance` | Row meaning is acknowledged but policy closure belongs to a later sprint. |

## Selected Target Manifest

The selected-target manifest owns the target-specific fields that were
previously repeated across docs, workflow guards, and freshness checks:

| Field group | Manifest fields |
| --- | --- |
| Identity | `target_id`, `family`, `subfamily`, `target_key`, `row_meaning` |
| Selection and policy | `selection_scope`, `support_tier`, `freshness_policy` |
| Generation and artifacts | `generator_command`, `artifact_pattern`, `required_files`, `expected_rows`, `expected_row_ids` |
| Hosted workflow scope | `workflow_file`, `workflow_job`, `workflow_artifact`, `workflow_platforms` |
| Claims and ownership | `claim_scope`, `non_claims`, `owner`, `introduced_in` |

Use the manifest when changing selected targets, selected expected row counts,
required generated files, hosted upload artifact names, support tiers,
freshness policies, claim scopes, or owner metadata. Update docs and tests to
reference the manifest instead of copying a second authoritative target list.

## Selected Oracle Freshness Gate

Sprint 152 selects the local oracle family for required freshness through:

```sh
make report-index-oracle-freshness
```

That target regenerates the combined local oracle report and then runs:

```sh
python3 scripts/normalize_report_index.py --family oracle --require-generated oracle --check-freshness
```

The selected oracle gate reads selected target identity, expected row count,
selected fixture keys, required artifacts, support tier, freshness policy,
workflow artifact name, claim scope, non-claims, and owner metadata from
`SRT-ORACLE-QR-PSVD-LOCAL` in `selected_report_targets.tsv`. Generated-local
rows must live under `build/corpus/oracle/*.tsv`, match the current source
commit, and satisfy the selected combined row family. Per-solver-family bucket
counts remain a normalizer compatibility check because the selected-target
schema owns the total selected row count and fixture-key set, not bucketed
solver-family counts.

Required selected oracle freshness fails missing artifacts, stale commits,
generated comparison failures, row-count mismatches, missing solver families,
and missing fixture keys. The diagnostics are part of the gate contract:

- `oracle_selected_row_count`
- `oracle_selected_solver_families`
- `oracle_selected_fixture_keys`

Generated oracle, corpus-report, and report-index outputs stay under ignored
`build/` paths. They are local evidence and are not hosted CI proof, release
proof, package proof, ABI proof, platform proof, performance proof,
external-library parity, broad QR correctness, broad partial-SVD correctness,
or state-of-the-art evidence.

## Selected Comparison Freshness Gate

Sprint 174 extends the selected comparison family for required freshness
through:

```sh
make report-index-comparison-freshness
```

That target regenerates the manifest-selected QR, partial-SVD, and LU
comparison reports and then runs:

```sh
python3 scripts/normalize_report_index.py --family comparison --require-generated comparison --check-freshness
```

The selected comparison gate reads selected target keys, expected row counts,
expected row IDs, generated commands, artifact patterns, required files,
workflow upload artifacts, support tiers, freshness policies, claim scopes,
non-claims, and owners from the comparison rows in
`selected_report_targets.tsv`. Generated comparison rows must live under the
manifest-owned `build/comparison/*/study.tsv` artifact patterns and match the
current source commit.

Required selected comparison freshness fails missing artifacts, stale commits,
generated comparison failures, skipped or deferred selected rows, duplicate
rows, unexpected rows, row-count mismatches, and missing selected families.
The generated rows are local fixture evidence by default. The reviewed Linux
hosted report-freshness lane promotes only this selected comparison gate and
its uploaded selected artifacts after hosted CI passes. They are not broad QR,
LU, nonsymmetric solve, SVD, or partial-SVD correctness; LU CSR parity; raw QR
basis identity; raw singular-vector identity; vector sign/orientation identity;
external-library parity; platform proof; package proof; ABI proof; performance
proof; release proof; or state-of-the-art evidence.

## Guardrails

- Source-controlled contract rows are advisory or deferred; they are not pass
  evidence.
- Missing generated rows must be represented as `not_generated` or
  `unknown`, not silently omitted when the family is selected.
- Benchmark selected freshness is represented by the benchmark row in
  `selected_report_targets.tsv`; sentinel, coverage, dead-code, package, CI,
  and documentation rows remain local, source-controlled, or advisory unless a
  later reviewed manifest row explicitly promotes them.
- Sprint 163 benchmark rows may expose methodology fields such as
  `support_tier`, `claim_boundary`, `repeat_semantics`, `warmup`, `variance`,
  `baseline`, `threshold`, and `methodology_notes` through generated report
  indexes or normalized `configuration` text. These fields preserve local
  context; they do not create pass/fail benchmark proof.
- Sprint 163 sentinel rows may expose `baseline_provenance`,
  `repeat_semantics`, `warmup`, `variance`, and `methodology_notes`. S5 remains
  the local wall-check hard gate; S2 and S3 remain threshold-free
  backend-context rows.
- Package/install rows identify maintained proof owners and static-first
  scope; they do not claim package-manager availability, shared-library ABI,
  or broad platform support.
- Runtime/backend governance rows should remain deferred only where policy
  decisions are still open. Source-controlled policy rows identify the
  maintained control boundary, while selected Sprint 142 sentinel rows belong
  under the `sentinel` family and keep local-only claim boundaries.
