# Sprint 181 Day 5: Manifest Prototype

## Purpose

Day 5 adds the first source-controlled selected report target manifest and
populates baseline rows for the Sprint 181 selected report target surface.
The prototype follows the Day 4 schema without refactoring guards yet.

## Added Manifest

Canonical path:

`tests/corpus/manifests/selected_report_targets.tsv`

The manifest currently has `6` selected target rows:

| Target group | Rows | Target IDs |
| --- | ---: | --- |
| Selected oracle freshness | 1 | `SRT-ORACLE-QR-PSVD-LOCAL` |
| Selected comparison freshness | 4 | `SRT-COMP-QR-MINNORM`, `SRT-COMP-QR-COMPATIBLE-LS`, `SRT-COMP-PSVD-DIAG6-K2`, `SRT-COMP-LU-NONSYM-SQUARE-5` |
| Selected canonical performance freshness | 1 | `SRT-BENCH-REFACTOR-CSC-NOS4` |

## Populated Metadata

The prototype rows include:

- stable `target_id` values for guard and docs references;
- `family` and `subfamily` values aligned with existing
  `report_families.tsv` rows;
- command names currently duplicated across Makefile targets, workflow YAML,
  scripts, tests, and docs;
- generated artifact paths and required file lists;
- expected row counts for selected oracle, comparison, and performance
  evidence;
- selected row IDs or fixture keys where current guards already enforce row
  identity;
- Linux and macOS workflow upload metadata for selected hosted evidence;
- claim scopes and non-claims copied from the current selected target
  boundaries;
- owner and introduction-source notes.

## Selected Oracle Baseline

`SRT-ORACLE-QR-PSVD-LOCAL` captures the selected QR and partial-SVD oracle
freshness contract:

| Field | Value |
| --- | --- |
| Command | `make report-index-oracle-freshness` |
| Artifact pattern | `build/corpus/oracle/*.tsv` |
| Required files | `build/corpus-reports/manifest.txt`; `build/report-index/normalized-index.tsv` |
| Expected rows | `52` |
| Hosted metadata | Linux `generated-report-freshness`, artifact `sprint159-oracle-freshness` |

The expected-row identity field records the selected fixture keys currently
embedded in `scripts/normalize_report_index.py`.

## Selected Comparison Baseline

The four selected comparison rows capture the existing maintained QR,
partial-SVD, and LU target list:

| Target ID | Target key | Expected rows | Artifact pattern |
| --- | --- | ---: | --- |
| `SRT-COMP-QR-MINNORM` | `qr-minnorm` | 6 | `build/comparison/qr_minnorm/study.tsv` |
| `SRT-COMP-QR-COMPATIBLE-LS` | `qr-compatible-ls` | 6 | `build/comparison/qr_compatible_ls/study.tsv` |
| `SRT-COMP-PSVD-DIAG6-K2` | `partial-svd-diag6-k2` | 10 | `build/comparison/partial_svd_diag6_k2/study.tsv` |
| `SRT-COMP-LU-NONSYM-SQUARE-5` | `lu-nonsym-square-5` | 6 | `build/comparison/lu_nonsym_square_5/study.tsv` |

Each comparison row includes the six required generated files currently
validated by `tests/test_selected_comparison_workflow.py`:

- `project_observations.tsv`
- `baseline_observations.tsv`
- `dependency_status.tsv`
- `study.tsv`
- `summary.md`
- `manifest.tsv`

Each comparison row also records Linux and macOS hosted artifact names:

- `sprint175-linux-selected-comparison-freshness`
- `sprint175-macos-selected-comparison-freshness`

## Selected Performance Baseline

`SRT-BENCH-REFACTOR-CSC-NOS4` captures the selected canonical performance
freshness target:

| Field | Value |
| --- | --- |
| Command | `make bench-canonical-report-freshness` |
| Artifact pattern | `build/bench-reports/canonical/bench_refactor_csc.csv` |
| Required files | `bench_refactor_csc.csv`; `index.tsv`; `manifest.txt` |
| Expected rows | `1` |
| Workflow metadata | Linux `hosted-performance-freshness`, artifact `sprint168-selected-performance-freshness` |

The row uses `hosted_selected` for `selection_scope` and `support_tier` because
the selected performance checker already recognizes hosted selected metadata.
This is selected-target metadata only; it does not widen the broad
`benchmark/canonical` report-family row or create a portable performance
claim.

## Explicit Non-Promotions

Day 5 does not add separate selected rows for these report-family surfaces:

| Surface | Reason |
| --- | --- |
| Package | Sprint 180 keeps package-manager support unavailable; static install proof-owner rows are source-controlled metadata, not selected generated report targets. |
| CI | Hosted CI evidence is represented as workflow metadata on selected oracle, comparison, and performance rows, not as standalone generated proof. |
| Documentation | Documentation rows explain interpretation anchors and non-claims; they do not generate proof. |
| Sentinel | Existing sentinel report families remain local/generated or advisory and are not Sprint 181 selected report targets. |
| Guardrail | Large-matrix guardrails remain separate generated-local guardrail evidence. |
| Dead-code | Dead-code report rows remain maintainer navigation, not selected proof. |
| Coverage | Coverage rows remain generated-local advisory metadata. |

This keeps unselected rows from being promoted to selected proof status.

## Day 6 Handoff

Day 6 should add parser/schema checks for:

- exact header fields and non-empty required cells;
- duplicate `target_id`;
- duplicate `family`/`subfamily`/`target_key` tuples;
- allowed `selection_scope`, `support_tier`, and `freshness_policy` values;
- positive integer or `none` `expected_rows`;
- mapping from selected manifest `family`/`subfamily` pairs to
  `report_families.tsv`;
- hosted rows requiring workflow file, job, artifact, and platform metadata;
- generated rows requiring generator command and required files.

Day 6 should also decide whether `hosted_selected` becomes an explicit
selected-target support-tier value in the parser allowed set or remains
limited to benchmark checker metadata.

## Validation

Day 5 changed documentation and TSV metadata only. Validation:

- `awk -F '\t' 'NR==1 {w=NF; print "header_fields=" w; next} NF!=w {print FILENAME ":" NR ": expected " w " fields, got " NF; bad=1} END {if (bad) exit 1; print "rows=" NR-1}' tests/corpus/manifests/selected_report_targets.tsv`
- `git diff --check`

## Completion Criteria Review

| Criterion | Status | Evidence |
| --- | --- | --- |
| Manifest exists with all selected Sprint 181 target categories represented. | Complete | The prototype includes selected oracle, comparison, and performance rows with hosted workflow metadata. |
| Rows carry enough metadata to replace duplicated lists later in the sprint. | Complete | Rows include target keys, commands, artifact paths, required files, counts, row IDs or fixture keys, workflow artifacts, support tiers, claim scopes, and non-claims. |
| Unselected rows are not promoted to selected proof status. | Complete | Package, CI-only, documentation, sentinel, guardrail, dead-code, and coverage surfaces are explicitly excluded from selected proof rows. |
