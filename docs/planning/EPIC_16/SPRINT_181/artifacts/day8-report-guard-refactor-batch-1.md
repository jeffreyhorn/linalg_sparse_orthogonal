# Sprint 181 Day 8: Report Guard Refactor Batch 1

## Purpose

Day 8 implements the first manifest-driven report guard refactor. The batch
moves selected oracle and selected comparison freshness expectations behind
`tests/corpus/manifests/selected_report_targets.tsv` while preserving current
normalizer output, diagnostic names, and freshness command behavior.

## Code Changes

Updated `scripts/normalize_report_index.py` with selected-target manifest
helpers:

| Helper | Role |
| --- | --- |
| `SELECTED_REPORT_TARGET_MANIFEST` | Canonical manifest path under the corpus root. |
| `selected_report_targets` | Loads the selected report target TSV through the existing structured TSV reader. |
| `selected_targets_by_family` | Filters selected target rows by report family. |
| `selected_oracle_contract` | Returns the single Day 8 selected oracle contract. |
| `selected_comparison_contracts` | Returns selected comparison contracts and catches duplicate target keys at runtime. |
| `split_manifest_values` | Parses semicolon-delimited manifest cells, treating `none` as empty. |
| `expected_int` | Parses positive integer expected-row fields with target-aware errors. |
| `selected_oracle_expected_rows` | Reads selected oracle row count from the manifest. |
| `selected_oracle_fixture_keys` | Reads selected oracle fixture keys from the manifest. |
| `selected_comparison_row_ids` | Reads selected comparison row IDs from the manifest. |
| `selected_comparison_expected_rows` | Sums selected comparison expected-row counts from the manifest. |
| `selected_comparison_artifact_diagnostic` | Builds comparison artifact diagnostics from manifest artifact patterns. |

The normalizer now loads selected target metadata only when selected
oracle/comparison freshness policy is in play through `--require-generated`,
`--strict-generated`, or an oracle/comparison family filter. Generic index
generation and unrelated advisory family freshness checks do not require the
selected-target manifest.

## Migrated Expectations

| Former embedded expectation | Manifest-backed source |
| --- | --- |
| Selected oracle expected total `52` | `SRT-ORACLE-QR-PSVD-LOCAL.expected_rows` |
| Selected oracle fixture-key set | `SRT-ORACLE-QR-PSVD-LOCAL.expected_row_ids` |
| Selected comparison row IDs | Union of `expected_row_ids` from `SRT-COMP-*` rows |
| Selected comparison expected row total `28` | Sum of `expected_rows` from `SRT-COMP-*` rows |
| Selected comparison artifact diagnostic paths | `artifact_pattern` from `SRT-COMP-*` rows |

`SELECTED_ORACLE_ROW_COUNTS` remains a local compatibility helper for solver
family buckets. The Day 5 manifest owns selected target identity and total row
count, but it does not yet model per-solver-family bucket counts.

## Diagnostics

Existing diagnostic identifiers remain stable:

- `oracle_selected_row_count`
- `oracle_selected_solver_families`
- `oracle_selected_fixture_keys`
- `comparison_selected_rows`
- `comparison_selected_status`

Day 8 adds manifest context to selected diagnostics where useful:

- oracle count and fixture-key failures include `target_id`;
- comparison row-set and status failures include selected comparison target
  IDs;
- comparison artifact diagnostics are generated from manifest artifact
  patterns.

This keeps existing tests/docs compatible while making selected target
metadata traceable to one reviewed TSV row set.

## Test Updates

Updated `tests/test_normalize_report_index.py` to import the normalizer module
and assert the manifest-derived values match the former embedded test
expectations:

- selected oracle expected rows;
- selected oracle fixture keys;
- selected comparison row IDs;
- selected comparison expected row total;
- selected comparison artifact tuple;
- selected comparison artifact diagnostic string.

Existing generated-row tests continue to cover missing selected generated
rows, stale rows, failed rows, skip/defer rows, duplicate normalized rows, and
advisory/source-controlled family behavior.

Duplicate and unsupported selected manifest rows remain covered by
`tests/test_selected_report_targets_manifest.py`.

## Compatibility

Preserved behavior:

- `make report-index-oracle-freshness` remains runnable and passes.
- `make report-index-comparison-freshness` remains runnable and passes.
- `--require-generated oracle` and `--require-generated comparison` remain the
  user-facing selected freshness gates.
- advisory and unselected families keep their prior freshness behavior.
- selected target manifest rows are not emitted as normalized report-index
  rows in this batch.
- no target-key or support-tier CLI flags were added.

## Day 9 Handoff

Day 9 can use the same selected-target helper surface to decide which
remaining report guards should be manifest-driven and which should remain
advisory or guard-owned:

- benchmark selected performance metadata can move behind
  `SRT-BENCH-REFACTOR-CSC-NOS4`;
- workflow guards can consume workflow metadata fields while keeping exact
  YAML block checks;
- package, CI, documentation, sentinel, guardrail, dead-code, and coverage
  rows should remain non-promoted unless they have explicit selected target
  rows.

## Validation

Validation run:

- `python3 scripts/validate_corpus_schema.py`
- `python3 tests/test_selected_report_targets_manifest.py`
- `python3 tests/test_normalize_report_index.py`
- `python3 -m py_compile scripts/normalize_report_index.py scripts/validate_corpus_schema.py tests/test_normalize_report_index.py tests/test_selected_report_targets_manifest.py`
- `make report-index-oracle-freshness`
- `make report-index-comparison-freshness`
- `git diff --check`

## Completion Criteria Review

| Criterion | Status | Evidence |
| --- | --- | --- |
| Selected oracle and comparison checks read manifest-owned expectations. | Complete | Normalizer selected oracle/comparison diagnostics now derive expected rows, row IDs, fixture keys, and artifact diagnostics from `selected_report_targets.tsv`. |
| Tests cover missing, duplicate, stale, and unsupported selected rows. | Complete | Normalizer tests cover missing generated/stale/invalid selected rows; selected-target manifest tests cover duplicate and unsupported manifest rows. |
| Existing report freshness commands remain runnable. | Complete | `make report-index-oracle-freshness` and `make report-index-comparison-freshness` both pass. |
