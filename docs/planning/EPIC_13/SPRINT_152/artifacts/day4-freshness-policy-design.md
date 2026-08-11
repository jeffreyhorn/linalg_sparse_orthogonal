# Sprint 152 Day 4 Freshness Policy Design

## Purpose

Day 4 defines the generated report freshness contract for the selected Sprint
152 families before command, metadata, or gate implementation changes. The
selected surface is deliberately narrow: `oracle/generated_reference`,
`oracle/solver_backed`, and supporting `report_index/missing_generated` rows.

## Current Behavior Reviewed

`scripts/normalize_report_index.py` currently supports:

- `--family <family>` to select report families;
- `--require-generated <family>` to fail `--check` mode when selected
  generated rows are missing;
- `--check-freshness` to evaluate row freshness diagnostics;
- `--strict-generated` to make stale or unchecked generated rows stricter;
- `--advisory-ok` to keep advisory generated families from failing strict
  freshness.

Current generated oracle rows are loaded with:

- `freshness_status=generated_present_unchecked`;
- `freshness_reason=oracle_row_loaded;stale_rules_deferred_to_days10_11`;
- `freshness_policy=generated_compare_inputs` by family policy;
- row-level metadata from `scripts/run_corpus_oracle.py`.

## Selected Freshness State Model

| State | Applies To | Meaning | Default Severity | Required Local Severity | Strict Local Severity |
| --- | --- | --- | --- | --- | --- |
| `source_controlled` | Corpus prerequisites and report-family contracts | Governed by Git and schema validation. | advisory | advisory | advisory |
| `not_generated` | Selected generated family with no local artifact | No local generated artifact exists; no pass evidence is manufactured. | advisory or warning by policy | error | error if selected and strict |
| `generated_present_unchecked` | Generated row loaded but not fully compared | Row exists; command/source metadata is present but selected strict checks are not yet fully promoted. | warning for `generated_compare_inputs`; advisory for advisory families | warning unless source/row mismatch makes it error | error after selected strict policy is implemented |
| `fresh` | Generated row source metadata matches current policy | Row exists and selected freshness inputs match current repository state and selected command/path policy. | advisory success | advisory success | advisory success |
| `stale` | Generated row source metadata differs from current policy | Row exists but recorded source commit, command, path, or selected metadata does not match current policy. | warning for strict policies | error | error |
| `optional_data_skip` | Optional-data rows | Optional data unavailable, disabled, or deferred. | skip | skip | skip |
| `deferred` | Governance-deferred rows | Row family is acknowledged but intentionally not closed. | defer | defer unless explicitly selected | defer |
| `unsupported` | Unsupported context rows | Row is unsupported for the current context. | unsupported | error if required | error if selected and strict |

## Selected Oracle Policy

### Local Required Policy

Sprint 152 should make oracle freshness actionable locally, not hosted-CI
release proof.

Required local command shape:

```sh
python3 scripts/run_corpus_oracle.py --include-solver-qr --include-partial-svd
python3 scripts/normalize_report_index.py --family corpus --family oracle --check
python3 scripts/normalize_report_index.py --family oracle --require-generated oracle --check-freshness
```

The required oracle family should fail locally when:

- no generated oracle artifact exists for selected oracle rows;
- an oracle row is stale relative to current `HEAD`;
- selected expected row counts are missing from generated oracle output;
- selected solver families or fixture keys are missing from the manifest or
  normalized rows;
- a selected oracle row has `comparison_status=fail`;
- duplicate normalized oracle row IDs are generated;
- command/path metadata does not match the selected policy after Day 5/Day 6
  stabilization.

### Strict Local Policy

Strict generated freshness should become a local closeout check only after the
selected oracle rows can be regenerated deterministically and compared against
stable command/path/metadata rules.

Candidate strict command shape:

```sh
python3 scripts/normalize_report_index.py --family oracle --require-generated oracle --strict-generated --check-freshness
```

Strict mode should fail for selected oracle rows when:

- `source_commit` differs from current `HEAD`;
- command metadata does not match a selected canonical command;
- artifact path differs from the selected generated output path;
- row-count metadata differs from selected QR/partial-SVD expectations;
- selected solver families or fixture keys are missing;
- the row remains `generated_present_unchecked` after strict comparison is
  implemented and no advisory exception applies.

Strict mode should not fail for:

- source-controlled contract rows;
- optional-data skip/defer rows;
- non-selected benchmark, sentinel advisory, guardrail, dead-code, coverage,
  package, CI, documentation, or runtime-backend rows;
- absent non-selected generated families.

## Required Metadata Contract

Selected oracle generated rows must carry these fields before promotion:

| Field | Source | Required Meaning |
| --- | --- | --- |
| `generator_command` / `command` | `scripts/run_corpus_oracle.py` and normalized row | Exact command used to generate the artifact. |
| `source_commit` | Git at generation time | Commit that generated the local row. |
| `source_branch` | Git branch at generation time | Branch context for maintainer interpretation. |
| `generated_at_utc` | Generator timestamp | Trace timestamp, not freshness proof by itself. |
| `platform` | Generator runtime | Local platform context. |
| `compiler` | Generator/probe runtime | Compiler or `not_applicable`, depending on row. |
| `configuration` | Generator and normalizer | Solver family, fixture key, operation, comparison kind, support details, hashes, tolerances, and policy metadata. |
| `support_tier` | Oracle row | `local_only` unless separately promoted. |
| `artifact_path` | Normalized report index | Generated local artifact path under ignored build output. |
| `comparison_status` / `status` | Oracle comparison | Selected rows must be pass or expected non-pass status according to their expected-result contract. |
| `claim_scope` | Expected/oracle row | Fixture-local claim only. |
| `non_claims` | Expected/oracle row | Boundaries that prevent broad release claims. |

Manifest-level metadata should include:

- selected command;
- oracle row count;
- solver family row counts;
- selected fixture keys;
- source commit and branch;
- generated timestamp;
- platform;
- path to generated oracle output.

## Local/CI Failure Policy Matrix

| Mismatch Class | Local Default | Local Required Oracle | Local Strict Oracle | Hosted CI |
| --- | --- | --- | --- | --- |
| Missing selected oracle artifact | advisory/warning | error | error | no CI claim unless Day 9 selects one |
| Stale `source_commit` | warning for strict policy | error | error | no CI claim unless Day 9 selects one |
| Generated row present but unchecked | warning | warning until strict comparison implemented | error after strict promotion | no CI claim unless Day 9 selects one |
| Oracle row comparison failure | error | error | error | no CI claim unless Day 9 selects one |
| Missing selected row count | error after Day 8 implementation | error | error | no CI claim unless Day 9 selects one |
| Missing selected fixture key | error after Day 8 implementation | error | error | no CI claim unless Day 9 selects one |
| Optional-data skip/defer | skip/defer | skip/defer | skip/defer | external optional policy only |
| Benchmark/sentinel/coverage/dead-code missing | advisory | advisory unless separately selected | advisory/defer | external artifact logs only |

## Generated-Local Versus Release Proof Boundary

Selected oracle freshness promotion remains local-only. It does not make
generated `build/` artifacts source-controlled release proof and does not imply
hosted CI coverage, package support, ABI support, platform support,
performance, external-library parity, or state-of-the-art status.

Hosted CI policy remains deferred to Days 9-10. Until then, selected oracle
freshness commands are local validation and retrospective evidence only.

## Day 5 Handoff

Day 5 should design the concrete stabilization needed to implement this policy:

- canonical oracle generation command or accepted command set;
- selected artifact paths;
- row-count and fixture-key metadata checks;
- required-family failure messages with regeneration commands;
- strict freshness diagnostics that distinguish missing, stale, unchecked, and
  advisory rows;
- documentation wording that keeps generated-local rows local-only.
