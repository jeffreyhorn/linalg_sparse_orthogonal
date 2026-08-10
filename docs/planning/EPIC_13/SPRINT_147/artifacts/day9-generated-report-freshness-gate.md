# Sprint 147 Day 9 Generated Report Freshness Gate

## Purpose

Day 9 defines the generated report freshness gate for Sprint 152. The gate
identifies which generated report families are relevant to Epic 13 claims,
which can become required-generated checks, which remain advisory, and what
metadata, failure semantics, and CI artifact policy are required before a
freshness claim is promoted.

Generated reports are local evidence unless a hosted artifact policy records
workflow run IDs, commit SHA, job names, conclusions, and retained artifacts.

## Current Generated Report Families

| Family/Subfamily | Current Artifact Pattern | Freshness Policy | Epic 13 Relevance | Default Day 9 Decision |
| --- | --- | --- | --- | --- |
| `oracle/generated_reference` | `build/corpus/oracle/*.tsv` | `generated_compare_inputs` | Supports QR and partial-SVD corpus rows. | Candidate required-generated family after Sprints 150-151 add claim-bearing rows. |
| `oracle/solver_backed` | `build/corpus/oracle/*.tsv` | `generated_compare_inputs` | Supports solver-backed QR and partial-SVD corpus proof. | Candidate required-generated family after proof-owner tests and oracle rows land. |
| `benchmark/canonical` | `build/bench-reports/canonical/index.tsv` | `generated_local_advisory` | Useful for context, but no portable performance claim is selected. | Advisory. |
| `sentinel/runtime` | `build/bench-reports/sentinels/sentinels.tsv` | `generated_compare_inputs` | Relevant only if Sprint 152 selects a hard-gate sentinel tied to a claim. | Conditional; advisory unless selected. |
| `sentinel/advisory` | `build/bench-reports/sentinels/*.tsv` | `generated_local_advisory` | Local maintainer inspection only. | Advisory. |
| `guardrail/large_matrix` | `build/bench-reports/large-matrix-guardrails/index.tsv` | `generated_compare_inputs` | Could support bounded structural guardrail claims, but no such claim is selected yet. | Conditional; advisory unless selected. |
| `deadcode/report` | `build/deadcode/report.tsv` | `generated_local_advisory` | Maintainer classification context. | Advisory. |
| `coverage/src` | `coverage/coverage-src.info` | `generated_local_advisory` | Test visibility context, not coverage-completeness proof. | Advisory. |
| `report_index/missing_generated` | `build/report-index/normalized-index.tsv` | `generated_local_advisory` | Makes absent generated artifacts visible. | Advisory absence signal, never pass evidence. |

Source-controlled report families such as `corpus/*`, `package/static_install`,
`ci/reviewed_lanes`, `documentation/report_guidance`, and
`runtime_backend/governance` define row meaning and owners. They are not
generated report freshness evidence.

## Required-Generated Decision Table

| Candidate Requirement | Selected By Default? | Sprint Owner | Required Command | Promotion Rule |
| --- | --- | --- | --- | --- |
| Require generated `oracle` rows for selected QR and partial-SVD corpus families. | Yes, after Sprints 150-151 land rows. | Sprint 152 | `python3 scripts/normalize_report_index.py --family oracle --require-generated oracle --check-freshness` | Promote only when generated oracle rows exist, match current inputs, and all selected corpus proof rows are pass/expected skip/defer. |
| Require generated `corpus` source-controlled rows. | No. | Corpus/report owner | `python3 scripts/normalize_report_index.py --family corpus --check` | Source-controlled corpus rows are validated by schema and Git review, not generated freshness. |
| Require generated `benchmark` rows. | No. | Benchmark owner | `make bench-canonical-report` plus normalized check if selected. | Keep advisory unless Sprint 152 selects a concrete benchmark claim and rejects portable performance wording. |
| Require generated `sentinel` hard-gate rows. | Conditional. | Runtime/benchmark owner | `make performance-sentinels` plus normalized freshness check. | Require only if a selected hard gate owns a claim; advisory sentinel rows remain advisory. |
| Require generated `guardrail` rows. | Conditional. | Benchmark owner | `make large-matrix-guardrails` plus normalized freshness check. | Require only if a selected structural guardrail claim names the lane and support tier. |
| Require generated `coverage` rows. | No. | Maintainer | `make coverage` plus normalized check if selected. | Coverage rows remain local context; no coverage-completeness claim. |
| Require generated `deadcode` rows. | No. | Maintainer | `make deadcode-report` plus normalized check if selected. | Dead-code rows remain maintainer classification context; no zero-dead-code claim. |
| Require generated external comparison rows. | Conditional after Sprint 154. | Numerical lead/report owner | Sprint 154 comparison command plus normalized check. | Require only for the first narrow external comparison study if report integration exists. |

## Freshness Metadata Requirements

Every generated row selected for freshness gating must include or map to:

- report family and subfamily;
- native row ID and normalized row ID;
- exact generator command;
- artifact path;
- source commit;
- source branch;
- generated timestamp;
- platform and architecture;
- compiler or `not_applicable`;
- configuration key/value fields;
- support tier;
- status and comparison status where applicable;
- freshness status and freshness reason;
- claim scope;
- non-claims;
- skip or defer reason when applicable.

Generated rows with `source_commit=unknown`, missing command, missing platform,
or missing support tier may be indexed for navigation, but they must not be
used for required-generated promotion.

## Failure Semantics

| State | Meaning | Required-Family Behavior | Advisory-Family Behavior |
| --- | --- | --- | --- |
| `fresh` | Generated row matches current commit or selected freshness comparison. | Passes freshness requirement if row status also supports the selected claim. | Advisory pass context only. |
| `generated_present_unchecked` | Artifact exists but strict freshness comparison is not complete. | Error unless Sprint 152 documents why unchecked freshness is acceptable for that family. | Advisory or warning depending on policy. |
| `stale` | Generated row source commit does not match current `HEAD` or selected input state. | Error. | Advisory or warning unless strict mode is selected. |
| `not_generated` | Required generated artifact is absent. | Error when `--require-generated <family>` selects the family. | Advisory absence signal. |
| `optional_data_skip` | Optional data is unavailable, disabled, or unconfigured. | Never pass evidence; may be accepted if the selected claim excludes optional data. | Skip-policy evidence only. |
| `deferred` | Row family or governance decision is intentionally not implemented. | Error if the deferred row is selected as required. | Defer signal only. |
| `unsupported` | Platform, feature, or data source is outside support. | Error if selected as required. | Unsupported signal only. |
| `fail` | Generated hard-gate, guardrail, or comparison row reports failure. | Error. | Error for hard-gate/guardrail rows; otherwise advisory only if explicitly classified that way. |

Skip, defer, unsupported, missing-generated, and advisory rows must not be
counted as solver, package, platform, comparison, performance, or release pass
evidence.

## Sprint 152 Implementation Target

The default Sprint 152 implementation target is narrow:

1. Require generated `oracle` rows after Sprint 150 and Sprint 151 add broader
   QR and partial-SVD corpus families.
2. Keep benchmark, advisory sentinel, coverage, dead-code, and missing-report
   rows advisory unless a selected claim explicitly needs them.
3. Decide separately whether `sentinel/runtime`, `guardrail/large_matrix`, or
   external comparison rows become required in later sprints.
4. Preserve generated outputs under ignored `build/` or `coverage/` paths
   unless a source-controlled snapshot policy is explicitly adopted.

Recommended Sprint 152 baseline commands:

```sh
python3 scripts/validate_corpus_schema.py
python3 tests/test_normalize_report_index.py
python3 scripts/normalize_report_index.py --no-generated --check
python3 scripts/run_corpus_oracle.py --include-solver-qr --include-partial-svd
python3 scripts/normalize_report_index.py --family oracle --require-generated oracle --check-freshness
```

Optional advisory commands:

```sh
python3 scripts/normalize_report_index.py --check-freshness
python3 scripts/normalize_report_index.py --strict-generated --advisory-ok --check-freshness
python3 scripts/normalize_report_index.py --family coverage --family deadcode --family package --check-freshness
```

## CI Artifact Policy Draft

Default policy:

- Generated local reports remain ignored build outputs.
- CI may run freshness checks for selected families, but source-controlled
  report rows alone are not hosted proof.
- Hosted proof requires recording workflow run ID, URL, commit SHA, job name,
  conclusion, runner image, command, and artifact retention policy.
- CI artifacts may be uploaded for inspection if a sprint explicitly defines
  retention name, path, and expiration.
- Public docs may cite hosted generated evidence only when the hosted run and
  artifact metadata are recorded.

Do not commit generated report outputs unless a later sprint adopts a
source-controlled snapshot policy with update rules, stale detection, and
review requirements.

## Source-Control Boundaries

Source-controlled:

- manifest rows;
- schemas;
- expected-result rows;
- generator/oracle/normalizer scripts;
- tests and proof owners;
- documentation explaining row meaning.

Ignored/generated:

- `build/corpus/oracle/*.tsv`;
- `build/corpus-reports/*.tsv`;
- `build/corpus-reports/manifest.txt`;
- `build/report-index/normalized-index.tsv`;
- `build/bench-reports/**`;
- `build/deadcode/report.tsv`;
- `coverage/coverage-src.info`.

## Stop Conditions

- A required-generated family is selected without a stable command and artifact
  path.
- A generated row lacks commit, branch, platform, compiler/configuration, or
  support-tier context.
- Missing, skipped, deferred, unsupported, or advisory rows are treated as pass
  evidence.
- Benchmark or sentinel rows are used for portable performance or
  state-of-the-art claims.
- Coverage rows are used for coverage-completeness claims.
- Dead-code rows are used for zero-dead-code claims.
- CI workflow rows are used as hosted proof without hosted log metadata.
- Generated outputs are committed without an explicit snapshot policy.

## Day 10 Handoff

Day 10 should define the ABI/package evidence gate. It should reuse the Day 9
boundary that package proof-owner rows are source-controlled metadata, while
package support claims require executable install/export/downstream proof and
shared-library claims require a product decision plus loader/package evidence.

## Completion Criteria

| Criterion | Status | Evidence |
| --- | --- | --- |
| Sprint 152 has a concrete freshness implementation target. | Complete | Default target requires generated `oracle` rows for selected QR/partial-SVD corpus claims and leaves other families advisory unless selected. |
| Advisory rows are not treated as pass evidence. | Complete | Failure semantics and stop conditions keep advisory, missing, skip, defer, and unsupported rows out of pass evidence. |
| Generated reports have clear source-control and CI boundaries. | Complete | Source-control boundary and CI artifact policy separate ignored local outputs from hosted proof. |
