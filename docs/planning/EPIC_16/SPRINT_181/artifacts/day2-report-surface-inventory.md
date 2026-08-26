# Sprint 181 Day 2: Current Report Surface Inventory

## Purpose

Day 2 inventories current report-family rows, selected report targets,
generated artifact paths, freshness commands, docs claim surfaces, and
duplicated target-list owners before Sprint 181 designs the selected report
target manifest.

## Report-Family Row Inventory

`tests/corpus/manifests/report_families.tsv` currently defines `21`
source-controlled report-family rows.

| Origin class | Rows | Families |
| --- | ---: | --- |
| Source-controlled | 6 | `corpus`, `package`, `ci` |
| Generated local | 13 | `oracle`, `benchmark`, `sentinel`, `guardrail`, `deadcode`, `coverage`, `report_index`, `comparison` |
| Documentation | 2 | `documentation`, `runtime_backend` |
| Hosted-CI metadata policy | included above | `ci/reviewed_lanes` uses `freshness_policy=hosted_ci_external` while remaining source-controlled metadata |

Current row-origin split:

- `source_controlled`: corpus fixtures, generators, optional-data policy,
  expected rows, package proof owner, and CI lane definitions.
- `generated_local`: selected oracle rows, selected comparison rows, benchmark
  reports, sentinels, guardrails, dead-code reports, coverage, and missing
  generated report-index rows.
- `documentation`: report guidance and runtime/backend governance policy.

## Selected Target Inventory

| Category | Current selected surface | Current owner files |
| --- | --- | --- |
| Oracle | Selected QR and partial-SVD local oracle freshness, with `52` expected generated rows split across partial-SVD, QR, and unknown solver-family rows. | `Makefile`, `scripts/normalize_report_index.py`, `scripts/run_corpus_oracle.py`, `tests/test_normalize_report_index.py`, README, maintainer guide, Linux workflow. |
| Comparison | Four selected comparison targets: `qr-minnorm`, `qr-compatible-ls`, `partial-svd-diag6-k2`, and `lu-nonsym-square-5`; expected rows are `6`, `6`, `10`, and `6`. | `Makefile`, `scripts/normalize_report_index.py`, `scripts/run_external_comparison.py`, `tests/test_normalize_report_index.py`, `tests/test_selected_comparison_workflow.py`, README, maintainer guide, benchmark docs, Linux/macOS workflows. |
| Performance | Selected canonical benchmark freshness for `bench_refactor_csc` on `nos4.mtx --repeat 1`. | `Makefile`, `scripts/bench_canonical_report.sh`, `scripts/check_bench_canonical_freshness.py`, `tests/test_bench_canonical_freshness.py`, README, maintainer guide, benchmark docs, Linux workflow. |
| Package | Static install proof-owner row, not package-manager evidence. | `tests/corpus/manifests/report_families.tsv`, `tests/test_install.sh`, `tests/test_cmake_install.sh`, INSTALL, maintainer guide, package guards. |
| CI | Reviewed Linux source-of-truth lanes, Linux selected oracle/comparison freshness, macOS selected comparison freshness, macOS static-first install/export proof, and Windows CMake-first subset/install proof. | `.github/workflows/ci.yml`, `.github/workflows/macos-ci.yml`, `.github/workflows/windows-ci.yml`, workflow guard tests, README, maintainer guide. |
| Documentation | Report interpretation anchors and non-claim wording. | README, `docs/maintainer_guide.md`, `benchmarks/README.md`, INSTALL, `tests/corpus/schemas/report_index_fields.md`. |
| Benchmark/sentinel/guardrail/dead-code/coverage | Local generated or advisory report families used for maintainer navigation and bounded local checks. | `Makefile`, benchmark/report scripts, `scripts/normalize_report_index.py`, benchmark docs, maintainer guide. |

## Current Normalizer And Freshness Behavior

`scripts/normalize_report_index.py` currently owns:

- TSV reading/writing and normalized report-index output fields;
- report-family manifest loading from
  `tests/corpus/manifests/report_families.tsv`;
- source commit and branch metadata;
- generated artifact pattern expansion under `build/` and `coverage/`;
- strict freshness policies for `generated_compare_inputs`;
- advisory freshness policies for `generated_local_advisory` and
  `hosted_ci_external`;
- required-generated family checks through `--require-generated`;
- selected oracle expected row counts through embedded
  `SELECTED_ORACLE_ROW_COUNTS`;
- selected oracle fixture keys through embedded `SELECTED_ORACLE_FIXTURE_KEYS`;
- selected comparison row IDs through embedded `SELECTED_COMPARISON_ROW_IDS`;
- selected comparison artifact diagnostics through embedded
  `SELECTED_COMPARISON_ARTIFACTS`;
- remediation wording for `make report-index-oracle-freshness` and
  `make report-index-comparison-freshness`;
- diagnostics for missing generated family output, stale rows, wrong commands,
  non-pass selected rows, and missing selected comparison artifacts.

The normalizer is both a report-index generator and a current owner of selected
target expectations. Sprint 181 should move selected target identity and
expected-count authority into the new manifest while preserving normalizer
behavior and diagnostics.

## Freshness Commands

| Command | Current role | Generated paths |
| --- | --- | --- |
| `make report-index-oracle-freshness` | Regenerates selected local oracle output and runs required oracle freshness. | `build/corpus/oracle/*.tsv`, `build/corpus-reports/manifest.txt`, `build/report-index/normalized-index.tsv` |
| `make report-index-comparison-freshness` | Regenerates four selected local comparison outputs and runs required comparison freshness. | `build/comparison/{qr_minnorm,qr_compatible_ls,partial_svd_diag6_k2,lu_nonsym_square_5}/`, `build/report-index/normalized-index.tsv` |
| `make bench-canonical-report-freshness` | Regenerates and checks selected canonical benchmark report freshness. | `build/bench-reports/canonical/index.tsv`, `build/bench-reports/canonical/manifest.txt` |
| `python3 scripts/normalize_report_index.py --check-freshness` | Produces cross-family freshness diagnostics. | `build/report-index/normalized-index.tsv` when output is requested |
| `python3 scripts/normalize_report_index.py --family package --check-freshness` | Checks source-controlled package proof-owner rows without claiming package-manager support. | No generated package proof artifact required |

## Generated Artifact Path Inventory

| Family | Artifact paths |
| --- | --- |
| Oracle | `build/corpus/oracle/*.tsv`; `build/corpus-reports/manifest.txt` |
| Comparison | `build/comparison/qr_minnorm/*`; `build/comparison/qr_compatible_ls/*`; `build/comparison/partial_svd_diag6_k2/*`; `build/comparison/lu_nonsym_square_5/*` |
| Benchmark | `build/bench-reports/canonical/index.tsv`; `build/bench-reports/canonical/manifest.txt` |
| Sentinel | `build/bench-reports/sentinels/sentinels.tsv`; `build/bench-reports/sentinels/*.tsv` |
| Guardrail | `build/bench-reports/large-matrix-guardrails/index.tsv` |
| Dead-code | `build/deadcode/report.tsv` |
| Coverage | `coverage/coverage-src.info` |
| Normalized report index | `build/report-index/normalized-index.tsv` |

Generated report outputs are ignored local or hosted artifacts unless a later
decision explicitly changes that support tier.

## Workflow Upload Scope Inventory

| Workflow | Selected report scope |
| --- | --- |
| `.github/workflows/ci.yml` | Linux generated-report-freshness runs selected oracle and selected comparison freshness; uploads split oracle artifacts and selected comparison artifacts. |
| `.github/workflows/macos-ci.yml` | macOS selected-comparison-freshness runs selected comparison freshness and uploads selected comparison artifacts. |
| `.github/workflows/windows-ci.yml` | Windows remains CMake-first and does not promote Windows report freshness. |

Day 3 should inspect exact job and upload blocks in these files. Day 2 records
them as current claim surfaces, not as final manifest-owned workflow metadata.

## Documentation Claim Surfaces

| File | Current report/support-tier role |
| --- | --- |
| `README.md` | Lists selected oracle, comparison, and canonical benchmark freshness commands; states Linux selected oracle/comparison and macOS selected comparison hosted lanes; preserves Windows report freshness and broad report-index non-claims. |
| `docs/maintainer_guide.md` | Describes normalized report-index workflow, expected selected oracle/comparison outputs, selected performance freshness, support-tier interpretation, freshness diagnostics, and non-claims. |
| `benchmarks/README.md` | Describes canonical benchmark report freshness, selected performance evidence, report-index handoff, comparison report paths, and performance non-claims. |
| `INSTALL.md` | Describes package proof-owner rows and prevents package-manager or shared-library support inference. |
| `tests/corpus/schemas/report_index_fields.md` | Documents normalized report-index fields, selected oracle/comparison freshness gates, generated artifact locations, support tier, freshness policy, and non-claim boundaries. |

## Duplicate Target-List Owners

Day 2 identifies these duplicate or near-duplicate selected target owners for
Day 3's deeper audit:

- `Makefile` lists selected oracle and comparison freshness commands.
- `scripts/normalize_report_index.py` embeds selected oracle row counts,
  selected oracle fixture keys, selected comparison row IDs, selected
  comparison artifact paths, and selected freshness remediation text.
- `tests/test_normalize_report_index.py` embeds selected oracle/comparison
  expectations and diagnostics.
- `tests/test_selected_comparison_workflow.py` embeds selected comparison
  target keys, artifact directories, expected row counts, and expected files.
- `.github/workflows/ci.yml` embeds selected oracle/comparison commands,
  selected comparison target tuples, summary checks, and upload paths.
- `.github/workflows/macos-ci.yml` embeds selected comparison command, target
  tuples, summary checks, and upload paths.
- README, maintainer guide, benchmark docs, and report-index schema docs repeat
  selected command names, target names, expected rows, artifact paths, support
  tiers, hosted lane meanings, and non-claims.

## Unsupported Or Advisory Separation

| Surface | Day 2 interpretation |
| --- | --- |
| Advisory source-controlled rows | Corpus fixtures, generators, optional data, expected rows, package proof owners, CI lane definitions, documentation guidance, and runtime/backend governance do not create generated pass evidence by themselves. |
| Generated-local selected rows | Oracle and comparison rows can become strict when selected freshness gates require generated artifacts. |
| Generated-local advisory rows | Benchmark, sentinel advisory, dead-code, coverage, and report-index rows are navigation or local inspection aids unless a focused gate makes a selected row strict. |
| Hosted metadata | Hosted CI rows identify reviewed lane definitions and uploaded artifacts, but source-controlled metadata does not substitute for hosted logs. |
| Non-claims | Broad report-index freshness, unselected report families, Windows report freshness, package-manager support, ABI support, release proof, platform parity, broad external-library parity, performance superiority, and state-of-the-art status remain unsupported. |

## Day 3 Handoff

Day 3 should turn this inventory into a precise duplication audit. The highest
value targets are:

1. selected comparison target tuples repeated in workflow YAML and
   `tests/test_selected_comparison_workflow.py`;
2. selected comparison row IDs and artifact paths embedded in
   `scripts/normalize_report_index.py`;
3. selected oracle row counts and fixture keys embedded in
   `scripts/normalize_report_index.py` and normalizer tests;
4. selected command/path/expected-row wording repeated in README, maintainer
   guide, benchmark docs, and report-index schema docs.

## Validation

Day 2 is documentation-only. Validation:

- `git diff --check`

## Completion Criteria Review

| Criterion | Status | Evidence |
| --- | --- | --- |
| Every selected report target surface is accounted for before schema design. | Complete | Selected target inventory, generated artifact path inventory, workflow upload scope inventory, and docs claim-surface inventory above. |
| Current duplicate target-list owners are visible. | Complete | Duplicate target-list owner section above. |
| Unsupported or advisory rows are separated from selected proof rows. | Complete | Unsupported or advisory separation table above. |
