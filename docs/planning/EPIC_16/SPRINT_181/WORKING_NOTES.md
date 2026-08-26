# Sprint 181 Working Notes

**Sprint:** 181 - Selected Report Target Manifest
**Epic source:** `docs/planning/EPIC_16/PROJECT_PLAN.md`
**Sprint path:** `docs/planning/EPIC_16/SPRINT_181/`
**Status:** Complete

## Source Artifact Note

The Sprint 181 source section lives in
`docs/planning/EPIC_16/PROJECT_PLAN.md` under "Sprint 181: Selected Report
Target Manifest". Sprint 181 artifacts in this directory follow the Epic 16
scope.

## Sprint Goal

Centralize selected report target metadata so workflows, guards, docs, and
freshness checks stop duplicating target lists by hand.

## Baseline Inputs

- `docs/planning/EPIC_16/PROJECT_PLAN.md`
- `docs/planning/EPIC_16/SPRINT_181/PLAN.md`
- `docs/planning/EPIC_16/SPRINT_177/artifacts/day2-residual-audit.md`
- `docs/planning/EPIC_16/SPRINT_177/artifacts/day5-matrix-schema.md`
- `docs/planning/EPIC_16/SPRINT_177/artifacts/day6-populated-matrix.md`
- `docs/planning/EPIC_16/SPRINT_177/artifacts/day7-target-selection.md`
- `docs/planning/EPIC_16/SPRINT_177/artifacts/day8-gate-templates.md`
- `docs/planning/EPIC_16/SPRINT_177/artifacts/day10-quality-surface-map.md`
- `docs/planning/EPIC_16/SPRINT_177/artifacts/day11-claim-boundary-freeze.md`
- `docs/planning/EPIC_16/SPRINT_180/artifacts/day14-integrated-validation-and-closeout.md`
- `docs/planning/EPIC_16/SPRINT_180/RETROSPECTIVE.md`
- `tests/corpus/manifests/report_families.tsv`
- `scripts/normalize_report_index.py`
- `scripts/run_corpus_oracle.py`
- `scripts/run_external_comparison.py`
- `scripts/bench_canonical_report.sh`
- `tests/test_normalize_report_index.py`
- `tests/test_selected_comparison_workflow.py`
- `.github/workflows/*.yml`
- `README.md`
- `docs/maintainer_guide.md`
- `benchmarks/README.md`
- `INSTALL.md`

## Starting Branch Snapshot

- Branch: `sprint-181`
- Starting commit: `398dc9221ad2223a0e500d2dcfb944bdcb6109c5`
- Recent base context:
  - `398dc922` Merge pull request #200 from `sprint-180`
  - `4d92545a` Address PR #200 review comments
  - `35ad506a` Address PR #200 review comments
  - `58c75bdb` Address PR #200 review comments
  - `6f8bf17b` Address PR #200 review comments

## Sprint 181 Project-Plan Items

| Item | Name | Status | Notes |
| --- | --- | --- | --- |
| 181.1 | Target Inventory | Complete | Day 1 defines report-target intake fields. Day 2 inventories report-family rows, selected target surfaces, generated paths, freshness commands, workflow upload scopes, docs claim surfaces, and duplicate target-list owners. Day 3 completes the workflow/guard duplication audit and classifies manifest-owned versus guard-owned fields. |
| 181.2 | Manifest Schema | Complete | Day 4 chooses `tests/corpus/manifests/selected_report_targets.tsv`, defines required schema fields, duplicate detection, diagnostics, and mapping rules against `report_families.tsv`. Day 5 adds the first six-row prototype. Day 6 adds parser/schema validation and focused malformed-row tests. |
| 181.3 | Guard Refactor | Complete | Day 7 scopes the normalizer refactor and maps selected oracle/comparison constants to manifest fields. Day 8 moves selected oracle/comparison normalizer freshness expectations behind manifest helpers. Day 9 moves selected benchmark artifact/support expectations behind the manifest and explicitly preserves non-promoted advisory families. |
| 181.4 | Workflow Scope Checks | Complete | Day 10 refactors workflow guards to consume selected-target workflow fields while keeping exact job and upload-block checks guard-owned. |
| 181.5 | Documentation Alignment | Complete | Day 11 updates maintainer, README, benchmark, and report-index docs to name `selected_report_targets.tsv` as selected target authority while preserving claim boundaries. |
| 181.6 | Validation | Complete | Day 12 hardens selected-target diagnostics and workflow drift regressions. Day 13 completes the integrated validation sweep. Day 14 reconciles deliverables and records Sprint 182 handoff inputs. |

## Inherited Sprint 177 Gate

Sprint 181 implements Sprint 177 Day 8 Gate 4:

| Field | Day 1 baseline |
| --- | --- |
| Target | Selected report target manifest. |
| Residual | S177-R10 workflow and selected report target-list duplication. |
| Matrix rows | ESM-006 selected oracle freshness, ESM-007 selected comparison freshness, ESM-009 selected performance, and ESM-013 registration/workflow drift. |
| Owner files | New selected-target manifest, `Makefile`, `scripts/normalize_report_index.py`, `scripts/run_corpus_oracle.py`, `scripts/run_external_comparison.py`, benchmark report scripts, workflow guard tests, `.github/workflows/*.yml`, README, maintainer guide, and benchmark docs. |
| Required evidence | A source-controlled manifest owns selected oracle, comparison, performance, artifact, expected-row, support-tier, and workflow upload metadata with duplicate detection. |
| Validation commands | `python3 tests/test_selected_comparison_workflow.py`, `python3 tests/test_normalize_report_index.py`, selected freshness Make targets, Python compile checks for changed scripts, and `git diff --check`. |
| Pass definition | Workflows, guards, and docs read or validate against manifest-owned expectations; duplicates fail clearly; upload blocks remain fail-closed; selected target changes require one manifest update. |
| Fail definition | Target lists remain hand-duplicated in several owners, missing or duplicate rows pass silently, workflow upload checks are broad substring checks, or selected target support tiers can drift. |
| Claim boundary | Selected report target metadata has one canonical reviewed owner used by local and workflow guards. |
| Protected non-claims | No broad report-index freshness, unselected oracle/comparison/performance freshness, release proof, package/ABI support, platform parity, or state-of-the-art claim. |

## Inherited Evidence Baseline

| Evidence row | Current meaning for Sprint 181 |
| --- | --- |
| ESM-006 selected oracle freshness | Local selected QR/partial-SVD oracle freshness exists and is mirrored by reviewed Linux hosted report-freshness evidence. MacOS and Windows selected oracle freshness remain non-claims. |
| ESM-007 selected comparison freshness | Local selected QR, partial-SVD, and LU comparison freshness exists and is mirrored by reviewed Linux/macOS hosted selected comparison lanes. Windows report freshness and unselected comparison families remain non-claims. |
| ESM-009 selected performance | Selected performance/report targets remain local or reviewed as explicitly scoped by current report metadata; no portable performance or state-of-the-art claim is implied. |
| ESM-013 registration and workflow drift | Some guard coverage exists, but selected report/workflow target lists remain repeated across Makefile, scripts, tests, workflows, and docs. |

## Sprint 180 Handoff

Sprint 180 leaves package-manager posture stable for Sprint 181:

- Homebrew local formula/tap proof is selected but remains unclaimed.
- Package-manager support is still unavailable in public docs.
- Sprint 181 should not widen package-manager, ABI, platform, performance, or
  state-of-the-art claims while centralizing report-target metadata.

## Selected Target Inventory Fields

Day 1 defines these fields as the starting point for Day 2 inventory and Day 4
manifest-schema design:

| Field | Purpose |
| --- | --- |
| `family` | Report family such as oracle, comparison, benchmark, sentinel, guardrail, deadcode, coverage, package, CI, documentation, or report-index. |
| `subfamily` | Existing report-family subdivision or selected target grouping. |
| `target_key` | Stable unique key for one selected report target row or workflow scope. |
| `row_meaning` | Human-readable row purpose, aligned with report index semantics. |
| `row_origin` | Source-controlled, generated-local, hosted-CI, documentation, or advisory origin. |
| `support_tier` | Local-only, reviewed hosted, reviewed cross-platform, optional-data, or another allowed support tier. |
| `freshness_policy` | Source-controlled, generated compare-inputs, generated local advisory, hosted external, optional-data skip, or similar policy. |
| `generator_command` | Command that creates or validates the target when applicable. |
| `artifact_pattern` | Required generated or source-controlled artifact path or glob. |
| `expected_rows` | Expected generated row count where the target has a count contract. |
| `workflow_job` | Exact workflow job that owns hosted validation, if any. |
| `workflow_artifact` | Exact artifact upload name/path block expected for hosted evidence, if any. |
| `claim_scope` | Positive evidence claim allowed by the selected target. |
| `non_claims` | Unsupported interpretations that must remain explicit. |
| `owner` | Maintainer role or owner file family responsible for updates. |
| `introduced_in` | Sprint/day or artifact that introduced the selected target. |

## Current Report Surface Inventory

Day 2 inventories the current report surface before schema design:

| Surface | Current state |
| --- | --- |
| Report-family manifest | `tests/corpus/manifests/report_families.tsv` contains `21` rows across source-controlled, generated-local, and documentation origins. |
| Selected oracle | `make report-index-oracle-freshness` regenerates selected QR/partial-SVD oracle output and checks required oracle freshness. Current expected generated total is `52` rows. |
| Selected comparison | `make report-index-comparison-freshness` regenerates `qr-minnorm`, `qr-compatible-ls`, `partial-svd-diag6-k2`, and `lu-nonsym-square-5`; expected rows are `6`, `6`, `10`, and `6`. |
| Selected performance | `make bench-canonical-report-freshness` owns selected canonical benchmark freshness for `bench_refactor_csc` on `nos4.mtx --repeat 1`. |
| Generated artifact roots | `build/corpus/oracle/`, `build/corpus-reports/`, `build/comparison/`, `build/bench-reports/`, `build/deadcode/`, `build/report-index/`, and `coverage/`. |
| Workflow scopes | Linux runs selected oracle and comparison freshness; macOS runs selected comparison freshness; Windows report freshness remains a non-claim. |
| Documentation surfaces | README, maintainer guide, benchmark docs, INSTALL, and report-index schema docs repeat selected commands, target names, expected rows, artifact paths, support tiers, and non-claims. |

## Duplicate Target-List Running Inventory

Day 2 identifies these current duplication owners for Day 3's focused audit:

- `Makefile`
- `scripts/normalize_report_index.py`
- `tests/test_normalize_report_index.py`
- `tests/test_selected_comparison_workflow.py`
- `.github/workflows/ci.yml`
- `.github/workflows/macos-ci.yml`
- `.github/workflows/windows-ci.yml`
- README
- `docs/maintainer_guide.md`
- `benchmarks/README.md`
- `INSTALL.md`
- `tests/corpus/schemas/report_index_fields.md`

## Day 3 Workflow And Guard Duplication Summary

Day 3 identifies the highest-risk duplicated selected report target facts:

| Area | Duplicated facts | Current owners |
| --- | --- | --- |
| Selected oracle | command, generator command, expected total rows, solver-family row counts, selected fixture keys, artifact paths | `Makefile`, `scripts/normalize_report_index.py`, `tests/test_normalize_report_index.py`, README, maintainer guide, report-index schema docs, Linux workflow |
| Selected comparison | target keys, subfamily/path mapping, expected rows, selected row IDs, required uploaded files, support tiers, and non-claims | `Makefile`, `scripts/normalize_report_index.py`, `tests/test_normalize_report_index.py`, `tests/test_selected_comparison_workflow.py`, Linux/macOS workflows, README, maintainer guide, benchmark docs, report-index schema docs |
| Selected performance | selected artifact, command/workload, relative path, methodology fields, hosted support tier, and claim boundary | `scripts/check_bench_canonical_freshness.py`, `tests/test_bench_canonical_freshness.py`, Linux workflow, README, maintainer guide, benchmark docs |
| Workflow upload scopes | exact jobs, upload artifact names, upload paths, fail-closed upload behavior, and platform non-claims | `.github/workflows/ci.yml`, `.github/workflows/macos-ci.yml`, `.github/workflows/windows-ci.yml`, `tests/test_selected_comparison_workflow.py` |

Day 3 classifies target identity, expected counts, artifact paths, required
files, generator commands, support tiers, claim scopes, non-claims, and hosted
platform metadata as manifest candidates. Exact YAML job placement, upload
step boundaries, `actions/upload-artifact@v4`, `if-no-files-found: error`, and
absence of Windows report freshness remain guard-owned structure.

## Day 4 Manifest Schema Design Summary

Day 4 selects the canonical source-controlled manifest path:

`tests/corpus/manifests/selected_report_targets.tsv`

The manifest will use TSV to match existing corpus/report metadata files and
will own selected target instances rather than broad report-family semantics.

Required schema fields:

| Field group | Fields |
| --- | --- |
| Identity | `target_id`, `family`, `subfamily`, `target_key`, `row_meaning` |
| Selection and policy | `selection_scope`, `support_tier`, `freshness_policy` |
| Generation and artifacts | `generator_command`, `artifact_pattern`, `required_files`, `expected_rows`, `expected_row_ids` |
| Hosted workflow metadata | `workflow_file`, `workflow_job`, `workflow_artifact`, `workflow_platforms` |
| Claims and ownership | `claim_scope`, `non_claims`, `owner`, `introduced_in` |

`report_families.tsv` remains the broad family authority for origin class,
default policy vocabulary, family-level claims, and family-level non-claims.
The new selected-target manifest owns the exact selected oracle, comparison,
and performance target rows, counts, required files, commands, and hosted
artifact names. Selected target rows may narrow a report-family claim but must
not widen it.

Day 4 defines validation failures for duplicate target IDs, duplicate
family/subfamily/target-key tuples, missing required cells, unsupported
support tiers, unsupported freshness policies, missing generated rows, stale
generated rows, wrong generated commands, expected-row-count mismatches, and
missing required artifact files.

## Day 5 Manifest Prototype Summary

Day 5 adds the first selected report target manifest:

`tests/corpus/manifests/selected_report_targets.tsv`

Prototype row coverage:

| Category | Rows | Notes |
| --- | ---: | --- |
| Selected oracle | 1 | `SRT-ORACLE-QR-PSVD-LOCAL` captures selected QR/partial-SVD oracle freshness, `52` expected rows, selected fixture keys, Linux hosted upload metadata, and oracle non-claims. |
| Selected comparison | 4 | One row each for `qr-minnorm`, `qr-compatible-ls`, `partial-svd-diag6-k2`, and `lu-nonsym-square-5`, with expected rows, row IDs, six required files, Linux/macOS upload metadata, and comparison non-claims. |
| Selected performance | 1 | `SRT-BENCH-REFACTOR-CSC-NOS4` captures the selected canonical benchmark target, `bench_refactor_csc.csv`, required index/manifest files, Linux hosted upload metadata, and threshold-free performance non-claims. |

Day 5 deliberately does not add separate selected rows for package, CI-only,
documentation, sentinel, guardrail, dead-code, or coverage surfaces. CI upload
metadata is represented on selected oracle, comparison, and performance rows;
the other surfaces remain report-family metadata or advisory/generated-local
evidence rather than selected Sprint 181 proof.

## Day 6 Parser And Schema Checks Summary

Day 6 extends `scripts/validate_corpus_schema.py` with selected report target
manifest validation:

| Validation area | Coverage |
| --- | --- |
| Required schema | `SELECTED_REPORT_TARGET_REQUIRED` defines the Day 4 header and `validate()` loads `selected_report_targets.tsv`. |
| Duplicate detection | Duplicate `target_id`, duplicate `family`/`subfamily`/`target_key`, artifact/command/count collisions, and cross-family hosted artifact collisions fail clearly. |
| Required values | Generated targets require commands and files; countable targets require positive expected rows and row IDs; hosted targets require workflow file, job, artifact, and platform metadata. |
| Allowed enums | Selection scope, support tier, and freshness policy values are validated. `hosted_selected` is accepted only through selected-target support-tier validation, not the broad report-family enum. |
| Family mapping | Selected rows must map to `report_families.tsv` family/subfamily pairs. |

Added `tests/test_selected_report_targets_manifest.py` for malformed-row and
unsupported-value diagnostics. `tests/corpus/README.md` now lists the selected
target manifest in the layout, ownership table, and row-interpretation
guidance.

## Day 7 Normalizer Refactor Design Summary

Day 7 scopes the normalizer refactor before implementation:

| Current normalizer expectation | Manifest migration path |
| --- | --- |
| `SELECTED_ORACLE_TOTAL_ROWS` | Read `expected_rows` from `SRT-ORACLE-QR-PSVD-LOCAL`. |
| `SELECTED_ORACLE_FIXTURE_KEYS` | Read `expected_row_ids` from the oracle selected target row and interpret them as fixture keys for oracle checks. |
| `SELECTED_ORACLE_ROW_COUNTS` | Keep as a Day 8 compatibility helper unless a later schema revision owns solver-family buckets explicitly. |
| `SELECTED_COMPARISON_ROW_IDS` | Read and union `expected_row_ids` from all `SRT-COMP-*` rows. |
| `SELECTED_COMPARISON_ARTIFACTS` | Read `artifact_pattern` from all `SRT-COMP-*` rows. |
| Comparison expected row count | Sum `expected_rows` from all selected comparison rows. |

Day 7 keeps `--require-generated <family>` as the CLI boundary for selected
freshness enforcement. No target-key or support-tier CLI flags are needed for
Day 8. Missing selected manifest rows should fail selected freshness checks
with manifest-specific diagnostics, while generic index generation and
unselected/advisory families keep their existing behavior.

## Day 8 Report Guard Refactor Batch 1 Summary

Day 8 updates `scripts/normalize_report_index.py` so selected oracle and
selected comparison freshness checks consume manifest-owned expectations:

| Manifest-backed helper | Consumed fields |
| --- | --- |
| `selected_oracle_expected_rows` | `SRT-ORACLE-QR-PSVD-LOCAL.expected_rows` |
| `selected_oracle_fixture_keys` | `SRT-ORACLE-QR-PSVD-LOCAL.expected_row_ids` |
| `selected_comparison_row_ids` | union of `SRT-COMP-* expected_row_ids` |
| `selected_comparison_expected_rows` | sum of `SRT-COMP-* expected_rows` |
| `selected_comparison_artifact_diagnostic` | `SRT-COMP-* artifact_pattern` values |

The refactor preserves existing diagnostic identifiers while adding manifest
context such as selected target IDs. `SELECTED_ORACLE_ROW_COUNTS` remains a
local compatibility helper because the Day 5 manifest does not yet model
per-solver-family bucket counts.

`tests/test_normalize_report_index.py` now verifies that manifest-derived
oracle and comparison expectations match the legacy constants used by the
existing synthetic freshness fixtures. The selected freshness Make targets
remain runnable and passing.

## Day 9 Report Guard Refactor Batch 2 Summary

Day 9 updates `scripts/check_bench_canonical_freshness.py` so the selected
benchmark guard reads `SRT-BENCH-REFACTOR-CSC-NOS4` before validating generated
benchmark artifacts.

| Manifest-backed benchmark field | Guard coverage |
| --- | --- |
| `target_id` | Requires exactly one selected benchmark contract. |
| `family` | Validates generated benchmark `report_family`. |
| `target_key` and `expected_row_ids` | Select the generated benchmark row by artifact identity. |
| `artifact_pattern` | Derives the selected CSV relative path. |
| `required_files` | Drives required artifact presence checks. |
| `support_tier` | Drives hosted selected support-tier expectations. |
| `freshness_policy` | Covered by selected benchmark manifest regression tests. |

Benchmark workload and methodology fields remain guard-owned because the
Sprint 181 manifest schema does not model matrix size, repeat semantics,
warmup, variance, baseline, threshold, or methodology notes as typed fields.

`scripts/validate_corpus_schema.py` now rejects selected target rows with
`artifact_pattern=none`. `tests/test_selected_report_targets_manifest.py`
adds that regression and asserts that package, CI, documentation, sentinel,
guardrail, dead-code, and coverage are not promoted to selected target rows.
These surfaces remain report-family metadata, advisory generated-local checks,
or documentation/source-controlled evidence rather than generated pass
evidence.

## Day 10 Workflow Scope Checks Summary

Day 10 updates `tests/test_selected_comparison_workflow.py` from a
comparison-only guard into a selected report freshness workflow guard.

| Workflow | Job | Manifest-backed selected families |
| --- | --- | --- |
| `.github/workflows/ci.yml` | `generated-report-freshness` | oracle and comparison |
| `.github/workflows/ci.yml` | `hosted-performance-freshness` | benchmark |
| `.github/workflows/macos-ci.yml` | `selected-comparison-freshness` | comparison |

The guard now reads selected workflow expectations from
`selected_report_targets.tsv`: `workflow_file`, `workflow_job`,
`workflow_artifact`, `workflow_platforms`, `target_key`, `artifact_pattern`,
`required_files`, `expected_rows`, `expected_row_ids`, and
`generator_command`.

YAML structure remains guard-owned. The test extracts exact job blocks and
exact upload blocks before checking commands, upload artifact names,
`actions/upload-artifact@v4`, `if-no-files-found: error`, selected artifact
paths, and broad-upload rejections. Windows remains an explicit non-claim and
must not run or upload selected oracle, comparison, or benchmark freshness.

## Day 11 Documentation Alignment Summary

Day 11 updates report-index and maintainer-facing documentation so
`tests/corpus/manifests/selected_report_targets.tsv` is the documented
authority for selected oracle, comparison, and benchmark target metadata.

Updated docs:

- `tests/corpus/schemas/report_index_fields.md` now documents the selected
  target manifest fields and separates broad report-family authority from
  selected target authority.
- `docs/maintainer_guide.md` now points selected oracle, comparison, and
  canonical benchmark freshness sections at the manifest rows instead of
  treating duplicated target lists as authoritative.
- `README.md` now tells maintainers that selected target lists, expected rows,
  required artifacts, workflow uploads, support tiers, freshness policies,
  claim scopes, and non-claims live in the selected-target manifest.
- `benchmarks/README.md` now points selected benchmark and comparison handoff
  wording at the manifest.

Remaining duplicated documentation is limited to user-facing command names,
high-level non-claim wording, generated artifact inspection examples,
benchmark methodology hints not modeled by the current schema, and generated
row-name summaries used to interpret `study.tsv`.

## Day 12 Diagnostics And Drift Tests Summary

Day 12 hardens failure diagnostics around the selected-target manifest and
workflow scope guard.

Schema diagnostics now include `target_id` for selected-target enum failures,
invalid expected rows, missing countable row IDs, missing generated commands,
missing required files, missing hosted workflow metadata, and missing or
unsupported artifact patterns. Artifact patterns must be repo-relative and
must not use parent traversal.

Regression coverage added:

- unsupported support tier and freshness policy rows;
- missing or parent-traversing artifact patterns;
- missing expected row IDs;
- missing generated required files;
- missing exact workflow job IDs;
- wrong selected upload artifact names;
- missing `if-no-files-found: error` in the exact selected upload block;
- broad comparison upload globs.

Existing normalizer tests already cover stale generated selected rows, missing
generated selected rows, selected expected row mismatches, duplicate selected
comparison rows, unexpected selected comparison rows, and selected comparison
failed/skipped/deferred rows.

## Day 13 Integrated Validation Summary

Day 13 runs the integrated Sprint 181 validation sweep:

| Area | Command | Result |
| --- | --- | --- |
| Corpus schema and selected-target manifest | `python3 scripts/validate_corpus_schema.py` | Pass |
| Selected-target malformed-row regressions | `python3 tests/test_selected_report_targets_manifest.py` | Pass |
| Selected workflow guard and drift tests | `python3 tests/test_selected_comparison_workflow.py` | Pass |
| Normalizer manifest/freshness regressions | `python3 tests/test_normalize_report_index.py` | Pass |
| Selected oracle freshness | `make report-index-oracle-freshness` | Pass |
| Selected comparison freshness | `make report-index-comparison-freshness` | Pass |
| Selected benchmark freshness | `make bench-canonical-report-freshness` | Pass |
| Benchmark freshness regression tests | `python3 tests/test_bench_canonical_freshness.py` | Pass |
| Static package/support deferral guard | `bash scripts/static_package_deferral_check.sh` | Pass |
| Python compile checks | `python3 -m py_compile ...` | Pass |
| Whitespace | `git diff --check` | Pass |

Benchmark freshness checks that write `build/bench-reports/canonical/` should
run sequentially. A parallel attempt raced on that shared output directory;
the selected benchmark Make target passed when rerun by itself.

## Day 14 Closeout And Handoff Summary

Day 14 reconciles Sprint 181 against the project plan:

| Item | Final status |
| --- | --- |
| 181.1 Target Inventory | Complete |
| 181.2 Manifest Schema | Complete |
| 181.3 Guard Refactor | Complete |
| 181.4 Workflow Scope Checks | Complete |
| 181.5 Documentation Alignment | Complete |
| 181.6 Validation | Complete |

Sprint 181 final authority model:

- `tests/corpus/manifests/report_families.tsv` remains the broad report-family
  authority.
- `tests/corpus/manifests/selected_report_targets.tsv` is the selected target
  authority for selected oracle, comparison, and benchmark target identity,
  commands, artifacts, expected rows, workflow metadata, support tiers,
  freshness policies, claim scopes, non-claims, owners, and provenance.

Sprint 182 handoff:

- Windows report freshness remains explicitly unselected.
- The Windows workflow remains CMake-first and package/static install scoped.
- The selected workflow guard rejects selected report freshness commands and
  selected upload artifacts in the Windows workflow.
- Sprint 182 should either add explicit Windows selected-target workflow
  metadata for one promoted Windows-safe report freshness path or formalize
  Windows report freshness as a deferred product decision.

## Day 1 Decisions

- Treat `docs/planning/EPIC_16/PROJECT_PLAN.md` as the Sprint 181 source
  authority.
- Treat Sprint 177 Gate 4 as the sprint acceptance gate.
- Treat Sprint 180 as a stable package-manager boundary, not an input that
  widens report-target scope.
- Use existing manifest conventions before inventing a new format.
- Separate selected target metadata from generated report evidence: the
  manifest can own expected targets, but generated rows still prove only the
  current local or hosted run they describe.
- Preserve existing non-claims for broad report-index freshness, unselected
  oracle/comparison/performance targets, Windows report freshness, package/ABI
  support, platform parity, release proof, and state-of-the-art claims.

## Daily Log

### Day 1 - Report Target Intake

Status: Complete

Completed:

- Re-read the Sprint 181 project-plan section and Sprint 181 plan.
- Reviewed Sprint 177 Gate 4, quality surface map, evidence matrix, target
  selection, residual audit, and claim-boundary freeze.
- Reviewed Sprint 180 closeout and retrospective handoff to keep
  package-manager posture out of Sprint 181 report-target scope.
- Created Sprint 181 working notes and artifact directory structure.
- Defined initial selected target inventory fields for family, subfamily,
  target key, generator command, artifact pattern, expected row count, support
  tier, freshness policy, owner, claim scope, non-claims, and workflow upload
  metadata.
- Created the Day 1 report-target-intake artifact.

Validation:

- `git diff --check`

### Day 2 - Current Report Surface Inventory

Status: Complete

Completed:

- Inspected `tests/corpus/manifests/report_families.tsv` and classified
  current rows by source-controlled, generated-local, documentation, and
  hosted-CI metadata policy origins.
- Inventoried selected oracle, comparison, performance, package, CI,
  documentation, benchmark, sentinel, guardrail, dead-code, coverage, and
  report-index surfaces.
- Inspected `scripts/normalize_report_index.py`,
  `tests/test_normalize_report_index.py`,
  `tests/test_selected_comparison_workflow.py`, Makefile targets, workflow
  references, README, maintainer guide, benchmark docs, INSTALL, and
  report-index schema docs for selected target ownership.
- Recorded current generated artifact paths under `build/` and `coverage/`.
- Identified duplicate selected target-list owners for Day 3.
- Created the Day 2 report-surface-inventory artifact.

Validation:

- `git diff --check`

### Day 3 - Workflow And Guard Duplication Audit

Status: Complete

Completed:

- Inspected selected workflow guard tests, including
  `tests/test_selected_comparison_workflow.py`,
  `tests/test_normalize_report_index.py`, and
  `tests/test_bench_canonical_freshness.py`.
- Inspected CI workflow YAML for Linux selected oracle/comparison freshness,
  Linux selected performance freshness, macOS selected comparison freshness,
  and Windows report freshness non-claims.
- Inspected embedded selected target constants in
  `scripts/normalize_report_index.py` and
  `scripts/check_bench_canonical_freshness.py`.
- Compared duplicated target facts against `report_families.tsv`, maintainer
  guide sections, README, benchmark docs, and report-index schema docs.
- Classified duplication into target identity, expected count, artifact path,
  required file, generator command, support tier, claim scope, non-claim,
  workflow scope, and diagnostic categories.
- Corrected the Day 2 working-notes baseline to `21` report-family rows.
- Created the Day 3 workflow-and-guard-duplication artifact.

Validation:

- `git diff --check`

### Day 4 - Manifest Schema Design

Status: Complete

Completed:

- Chose `tests/corpus/manifests/selected_report_targets.tsv` as the canonical
  source-controlled manifest path.
- Defined required fields for target identity, selection scope, support tier,
  freshness policy, generator command, artifact pattern, required files,
  expected rows, workflow metadata, claim scope, non-claims, owner, and
  introduction source.
- Defined duplicate detection for `target_id`,
  `family`/`subfamily`/`target_key`, artifact/command/count collisions, and
  hosted workflow artifact conflicts.
- Defined diagnostics for missing required files, missing generated rows,
  stale generated rows, wrong commands, expected-row-count mismatches,
  unsupported policy values, and duplicate keys.
- Defined mapping rules that keep `report_families.tsv` as the broad
  report-family authority while the new manifest owns selected target
  instances.
- Created the Day 4 manifest-schema-design artifact.

Validation:

- `git diff --check`

### Day 5 - Manifest Prototype

Status: Complete

Completed:

- Added `tests/corpus/manifests/selected_report_targets.tsv`.
- Populated the selected oracle row with command, artifact paths, expected
  rows, selected fixture keys, Linux workflow upload metadata, owner, claim
  scope, and non-claims.
- Populated four selected comparison rows for QR minimum-norm, QR compatible
  least-squares, partial-SVD diagonal top-k, and LU nonsymmetric square-solve
  freshness.
- Populated the selected canonical performance row for `bench_refactor_csc` on
  `tests/data/suitesparse/nos4.mtx --repeat 1`.
- Represented hosted CI upload metadata on selected target rows without
  promoting CI-only metadata to standalone generated proof.
- Documented explicit non-promotions for package, documentation, sentinel,
  guardrail, dead-code, and coverage report-family surfaces.
- Created the Day 5 manifest-prototype artifact.

Validation:

- `awk -F '\t' 'NR==1 {w=NF; print "header_fields=" w; next} NF!=w {print FILENAME ":" NR ": expected " w " fields, got " NF; bad=1} END {if (bad) exit 1; print "rows=" NR-1}' tests/corpus/manifests/selected_report_targets.tsv`
- `git diff --check`

### Day 6 - Parser And Schema Checks

Status: Complete

Completed:

- Extended `scripts/validate_corpus_schema.py` to load and validate
  `tests/corpus/manifests/selected_report_targets.tsv`.
- Added selected-target schema constants for required fields, selection
  scopes, support tiers, and freshness policies.
- Added duplicate detection for target IDs, target-key tuples,
  artifact/generator/count collisions, and cross-family hosted upload
  collisions.
- Added generated-row, hosted-row, expected-row, and report-family mapping
  validation.
- Added `tests/test_selected_report_targets_manifest.py` with focused
  malformed-row and unsupported-value tests.
- Updated `tests/corpus/README.md` so maintainers can find and interpret the
  selected target manifest.
- Created the Day 6 parser-and-schema-checks artifact.

Validation:

- `python3 scripts/validate_corpus_schema.py`
- `python3 tests/test_selected_report_targets_manifest.py`
- `python3 -m py_compile scripts/validate_corpus_schema.py tests/test_selected_report_targets_manifest.py`
- `make format && make lint && make test`
- `git diff --check`

### Day 7 - Normalizer Refactor Design

Status: Complete

Completed:

- Mapped current report normalizer family filtering and selected freshness
  checks to manifest-owned selected target rows.
- Identified embedded oracle and comparison constants that should move behind
  manifest access during Day 8.
- Defined compatibility behavior for advisory rows, unselected generated
  report families, and `--require-generated` family gating.
- Defined diagnostic names and message shapes for missing manifest rows,
  duplicate manifest-backed rows, row-count mismatches, missing fixture keys,
  comparison row-set mismatches, and non-pass selected comparison rows.
- Decided not to add new target-key or support-tier CLI flags for Day 8.
- Created the Day 7 normalizer-refactor-design artifact.

Validation:

- `git diff --check`

### Day 8 - Report Guard Refactor Batch 1

Status: Complete

Completed:

- Added selected report target manifest loading helpers to
  `scripts/normalize_report_index.py`.
- Refactored selected oracle freshness diagnostics to read expected total rows
  and selected fixture keys from `selected_report_targets.tsv`.
- Refactored selected comparison freshness diagnostics to read selected row
  IDs, expected row totals, and artifact diagnostics from
  `selected_report_targets.tsv`.
- Kept `--require-generated <family>` as the selected freshness CLI boundary.
- Preserved advisory and unselected report-family behavior.
- Added normalizer tests proving manifest-derived expectations match the
  former embedded selected oracle/comparison constants.
- Verified existing selected oracle and comparison freshness Make targets.
- Created the Day 8 report-guard-refactor-batch-1 artifact.

Validation:

- `python3 scripts/validate_corpus_schema.py`
- `python3 tests/test_selected_report_targets_manifest.py`
- `python3 tests/test_normalize_report_index.py`
- `python3 -m py_compile scripts/normalize_report_index.py scripts/validate_corpus_schema.py tests/test_normalize_report_index.py tests/test_selected_report_targets_manifest.py`
- `make report-index-oracle-freshness`
- `make report-index-comparison-freshness`
- `git diff --check`

### Day 9 - Report Guard Refactor Batch 2

Status: Complete

Completed:

- Refactored `scripts/check_bench_canonical_freshness.py` to load the selected
  benchmark target contract from `selected_report_targets.tsv`.
- Moved benchmark selected artifact identity, selected CSV relative path,
  required artifact list, report family, and hosted support-tier expectation
  behind `SRT-BENCH-REFACTOR-CSC-NOS4`.
- Kept benchmark workload and methodology assertions guard-owned because the
  manifest schema does not yet model those fields.
- Added schema validation that selected target rows must have an
  `artifact_pattern`.
- Added manifest regression tests for missing artifact patterns and explicit
  non-promotion of package, CI, documentation, sentinel, guardrail, dead-code,
  and coverage families.
- Added a benchmark freshness regression that checks the selected benchmark
  manifest row against the checker contract.
- Created the Day 9 report-guard-refactor-batch-2 artifact.

Validation:

- `python3 scripts/validate_corpus_schema.py`
- `python3 tests/test_selected_report_targets_manifest.py`
- `python3 tests/test_bench_canonical_freshness.py`
- `python3 -m py_compile scripts/check_bench_canonical_freshness.py scripts/validate_corpus_schema.py tests/test_bench_canonical_freshness.py tests/test_selected_report_targets_manifest.py`

### Day 10 - Workflow Scope Checks

Status: Complete

Completed:

- Refactored `tests/test_selected_comparison_workflow.py` to load workflow
  expectations from `selected_report_targets.tsv`.
- Added manifest-backed checks for Linux selected oracle/comparison freshness,
  Linux selected benchmark freshness, and macOS selected comparison freshness.
- Kept YAML parsing scoped to exact workflow job blocks and exact upload
  blocks.
- Added fail-closed checks for selected upload artifact names,
  `actions/upload-artifact@v4`, `if-no-files-found: error`, required selected
  paths, and broad generated-report upload patterns.
- Preserved the Linux workflow guard placement outside the
  `generated-report-freshness` job.
- Added Windows non-claim checks that reject selected report freshness
  commands and selected artifact uploads in the Windows workflow.
- Created the Day 10 workflow-scope-checks artifact.

Validation:

- `python3 tests/test_selected_comparison_workflow.py`
- `python3 -m py_compile tests/test_selected_comparison_workflow.py`

### Day 11 - Documentation Alignment

Status: Complete

Completed:

- Updated `tests/corpus/schemas/report_index_fields.md` to describe
  `selected_report_targets.tsv`, its field groups, and its authority relative
  to `report_families.tsv`.
- Updated selected oracle, selected comparison, and selected benchmark
  freshness sections in `docs/maintainer_guide.md` to point at manifest rows.
- Updated README report-freshness guidance to name the selected-target
  manifest as the selected target-list, expected-row, artifact, workflow,
  support-tier, freshness-policy, claim-scope, and non-claim authority.
- Updated `benchmarks/README.md` report-index handoff wording so selected
  benchmark and comparison metadata points at the manifest.
- Preserved claim boundaries for local-only generated reports, hosted selected
  lanes, Windows report freshness, package-manager support, performance, and
  external-library parity.
- Created the Day 11 documentation-alignment artifact.

Validation:

- `python3 scripts/validate_corpus_schema.py`
- `python3 tests/test_selected_report_targets_manifest.py`
- `python3 tests/test_selected_comparison_workflow.py`
- `git diff --check`

### Day 12 - Diagnostics And Drift Tests

Status: Complete

Completed:

- Added selected-target enum diagnostics that include `target_id`.
- Added selected-target artifact-pattern validation for missing, absolute, and
  parent-traversing artifact paths.
- Added target-aware diagnostics for invalid expected rows, missing expected
  row IDs, missing generated required files, and missing hosted workflow
  metadata.
- Added manifest regression tests for unsupported freshness policies,
  unsupported artifact patterns, missing expected row IDs, and missing
  generated required files.
- Added workflow drift regression tests for missing exact job IDs, wrong
  selected upload artifact names, missing fail-closed upload settings, and
  broad comparison upload globs.
- Created the Day 12 diagnostics-and-drift-tests artifact.

Validation:

- `python3 scripts/validate_corpus_schema.py`
- `python3 tests/test_selected_report_targets_manifest.py`
- `python3 tests/test_selected_comparison_workflow.py`
- `python3 -m py_compile scripts/validate_corpus_schema.py tests/test_selected_report_targets_manifest.py tests/test_selected_comparison_workflow.py`

### Day 13 - Integrated Validation

Status: Complete

Completed:

- Ran selected-target schema and malformed-row validation.
- Ran selected workflow guard and workflow drift validation.
- Ran normalizer manifest/freshness regression validation.
- Ran selected oracle, comparison, and benchmark freshness Make targets.
- Ran benchmark freshness regression tests.
- Ran the static package/support deferral guard because Day 11 touched support
  and package-boundary documentation.
- Ran Python compile checks for changed Python tooling and tests.
- Ran whitespace validation.
- Created the Day 13 integrated-validation artifact.

Validation:

- `python3 scripts/validate_corpus_schema.py`
- `python3 tests/test_selected_report_targets_manifest.py`
- `python3 tests/test_selected_comparison_workflow.py`
- `python3 tests/test_normalize_report_index.py`
- `make report-index-oracle-freshness`
- `make report-index-comparison-freshness`
- `make bench-canonical-report-freshness`
- `python3 tests/test_bench_canonical_freshness.py`
- `bash scripts/static_package_deferral_check.sh`
- `python3 -m py_compile scripts/normalize_report_index.py scripts/validate_corpus_schema.py scripts/check_bench_canonical_freshness.py tests/test_normalize_report_index.py tests/test_selected_report_targets_manifest.py tests/test_selected_comparison_workflow.py tests/test_bench_canonical_freshness.py`
- `git diff --check`

### Day 14 - Closeout And Handoff

Status: Complete

Completed:

- Reconciled all Sprint 181 project-plan items against artifacts and changed
  files.
- Recorded the selected target manifest authority model.
- Recorded intentionally retained duplication exceptions.
- Preserved unsupported report, platform, package, ABI, performance, release,
  external-parity, and state-of-the-art claim boundaries.
- Prepared Sprint 182 Windows report freshness handoff inputs.
- Recorded retrospective inputs for manifest design, guard migration, and
  benchmark validation sequencing.
- Created the Day 14 closeout-and-handoff artifact.

Validation:

- `python3 scripts/validate_corpus_schema.py`
- `python3 tests/test_selected_report_targets_manifest.py`
- `python3 tests/test_selected_comparison_workflow.py`
- `git diff --check`
