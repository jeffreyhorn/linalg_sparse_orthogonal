# Sprint 181 Day 7: Normalizer Refactor Design

## Purpose

Day 7 scopes the report normalizer refactor before implementation. The design
maps current selected oracle and comparison freshness expectations to
`tests/corpus/manifests/selected_report_targets.tsv` while preserving existing
family filtering, advisory behavior, and `--require-generated` semantics.

## Current Normalizer Ownership

`scripts/normalize_report_index.py` currently owns two responsibilities:

- normalized report-index row construction from `report_families.tsv` and
  generated artifacts;
- selected target freshness policy for oracle and comparison rows.

Sprint 181 should keep row construction in the normalizer, but move selected
target expectations behind the selected-target manifest.

## Refactor Scope

Day 8 should refactor only selected oracle and selected comparison checks:

| Area | Current owner | Day 8 manifest owner |
| --- | --- | --- |
| Oracle selected command | `CANONICAL_ORACLE_TARGET` and remediation strings | `SRT-ORACLE-QR-PSVD-LOCAL.generator_command` |
| Oracle expected total rows | `SELECTED_ORACLE_TOTAL_ROWS` | `SRT-ORACLE-QR-PSVD-LOCAL.expected_rows` |
| Oracle selected fixture keys | `SELECTED_ORACLE_FIXTURE_KEYS` | `SRT-ORACLE-QR-PSVD-LOCAL.expected_row_ids` for Day 5 prototype, interpreted as fixture keys for oracle rows |
| Oracle solver-family counts | `SELECTED_ORACLE_ROW_COUNTS` | Keep local helper for Day 8 unless Day 8 adds an optional derived helper; the manifest owns the selected target, not every solver-family bucket yet |
| Comparison selected row IDs | `SELECTED_COMPARISON_ROW_IDS` | Union of `expected_row_ids` from `SRT-COMP-*` rows |
| Comparison expected rows | `len(SELECTED_COMPARISON_ROW_IDS)` | Sum of `expected_rows` from `SRT-COMP-*` rows |
| Comparison artifact diagnostics | `SELECTED_COMPARISON_ARTIFACTS` and joined diagnostic string | `artifact_pattern` from `SRT-COMP-*` rows |
| Comparison remediation | hard-coded `make report-index-comparison-freshness` strings | Shared selected comparison remediation derived from selected target rows or kept as compatibility text if all selected rows share the current Make target |

Day 8 should not change benchmark, sentinel, guardrail, coverage, dead-code,
package, CI, or documentation row behavior. Those rows remain Day 9 scope.

## Parser Integration Decision

The normalizer should load selected targets through a small internal adapter
instead of importing validation internals directly:

- add `SELECTED_REPORT_TARGET_MANIFEST = Path("manifests") / "selected_report_targets.tsv"`;
- reuse the existing normalizer `read_tsv` helper for structured TSV loading;
- create `selected_report_targets(corpus_root)` returning manifest rows;
- create typed helper accessors for the selected target groups needed by
  freshness checks.

Recommended helper shape:

| Helper | Output |
| --- | --- |
| `selected_targets_by_family(rows, family)` | Selected target rows filtered by `family`. |
| `selected_oracle_contract(rows)` | Exactly one selected oracle row for Day 8, with target ID in diagnostics. |
| `selected_comparison_contracts(rows)` | Four selected comparison rows sorted by `target_id` or target key. |
| `split_manifest_values(value)` | Semicolon-delimited cells converted to sets/lists, with `none` as empty. |
| `expected_int(row, field)` | Positive integer parser for `expected_rows`, raising `ReportIndexError` with target ID. |

The validator from Day 6 remains the schema authority. The normalizer adapter
should still fail fast when required selected rows are absent, malformed, or
duplicated at runtime so report freshness failures are actionable.

## Family Filtering And `--require-generated`

Current behavior:

- `--family oracle` limits normalized output to oracle contract rows and
  generated oracle rows.
- `--family comparison` limits normalized output to comparison contract rows
  and generated comparison rows.
- `--require-generated oracle` turns missing selected oracle generated output
  into required freshness failure.
- `--require-generated comparison` turns missing selected comparison generated
  output into required freshness failure.
- `--check-freshness` emits advisory or error diagnostics based on freshness
  policy and required family state.

Day 8 compatibility rules:

1. Do not add new CLI flags.
2. Keep `--require-generated <family>` as the user-facing control.
3. Use manifest rows only when the selected target family is enabled by
   `--require-generated` or strict generated freshness.
4. Continue ignoring selected-target rows for families filtered out by
   `--family`.
5. If `--require-generated oracle` is set and no selected oracle manifest row
   exists, fail with a manifest-specific error before generated-row checks.
6. If `--require-generated comparison` is set and no selected comparison
   manifest rows exist, fail with a manifest-specific error before generated
   row-set checks.

No Day 8 CLI should reference target IDs directly. Target-key or support-tier
CLI filtering can wait until there is a concrete maintainer workflow need.

## Expected-Count Migration

| Current check | Manifest-backed behavior |
| --- | --- |
| `len(oracle_rows) != SELECTED_ORACLE_TOTAL_ROWS` | Compare selected generated oracle row count with `expected_rows` from `SRT-ORACLE-QR-PSVD-LOCAL`. |
| `counts != SELECTED_ORACLE_ROW_COUNTS` | Preserve current helper for Day 8. Record in diagnostics as a compatibility check until a later schema revision owns solver-family counts explicitly. |
| `SELECTED_ORACLE_FIXTURE_KEYS - observed_fixture_keys` | Compare `expected_row_ids` from the oracle manifest row against observed fixture keys. Diagnostic should call them fixture keys for oracle. |
| `SELECTED_COMPARISON_ROW_IDS - observed_ids` | Compare union of `expected_row_ids` from comparison manifest rows against observed comparison row IDs. |
| `observed_ids - SELECTED_COMPARISON_ROW_IDS` | Compare observed comparison row IDs against the same manifest-owned expected set. |
| `len(comparison_rows) != len(SELECTED_COMPARISON_ROW_IDS)` | Compare generated comparison row count with the sum of manifest `expected_rows`. |
| `row["row_id"] in SELECTED_COMPARISON_ROW_IDS and status != pass` | Use manifest-owned expected comparison row ID set. |

## Advisory And Unselected Compatibility

Keep these behaviors unchanged:

- source-controlled `report_families.tsv` rows remain advisory contract rows;
- missing benchmark, sentinel, guardrail, coverage, dead-code, package, CI, and
  documentation outputs keep their existing advisory or source-controlled
  semantics;
- unselected generated oracle/comparison artifacts do not become required
  proof because they are present;
- selected target manifest rows do not become normalized report-index rows in
  Day 8;
- missing selected target manifest data fails only selected freshness policy
  checks, not generic index generation without `--check-freshness`.

This preserves the existing non-claims for broad report-index freshness,
unselected report families, Windows report freshness, package/ABI support,
platform parity, release proof, and performance superiority.

## Diagnostics

New or updated diagnostics should keep the existing `freshness:` prefix.

| Failure | Diagnostic shape |
| --- | --- |
| Missing selected target manifest | `freshness: error: selected_report_targets_manifest: missing_manifest: path=tests/corpus/manifests/selected_report_targets.tsv; run python3 scripts/validate_corpus_schema.py` |
| Missing selected oracle row | `freshness: error: oracle_selected_manifest: missing_target: target_id=SRT-ORACLE-QR-PSVD-LOCAL; family=oracle; run python3 scripts/validate_corpus_schema.py` |
| Duplicate selected comparison manifest row | `freshness: error: comparison_selected_manifest: duplicate_target: family=comparison; target_key=<key>; run python3 scripts/validate_corpus_schema.py` |
| Oracle row-count mismatch | Keep `oracle_selected_row_count`, but include `target_id` and manifest `expected_rows`. |
| Oracle fixture-key mismatch | Keep `oracle_selected_fixture_keys`, but include `target_id` and manifest source path. |
| Comparison row-set mismatch | Keep `comparison_selected_rows`, but include target IDs and artifact patterns from manifest rows. |
| Comparison non-pass selected row | Keep `comparison_selected_status`, but derive expected row IDs from manifest rows. |
| Unsupported manifest value | Let Day 6 schema validation own this; normalizer should report `selected_report_targets_manifest` and the validator command if encountered during freshness. |

Existing diagnostic names should remain stable where tests and docs already
refer to them. Add target IDs as extra context, not as replacements.

## Day 8 Implementation Order

1. Add normalizer manifest loading helpers.
2. Add unit tests that prove current manifest rows produce the same expected
   oracle count, oracle fixture-key set, comparison row-ID set, and comparison
   artifact diagnostic list as the current constants.
3. Switch selected comparison diagnostics to use manifest-derived row IDs and
   artifact patterns.
4. Switch selected oracle diagnostics to use manifest-derived expected count
   and fixture keys.
5. Keep solver-family row counts as a local compatibility helper and document
   that the manifest owns selected target identity, not bucket totals yet.
6. Re-run selected normalizer tests before broader validation.

## Day 8 Validation Targets

Day 8 should run at least:

- `python3 scripts/validate_corpus_schema.py`
- `python3 tests/test_selected_report_targets_manifest.py`
- `python3 tests/test_normalize_report_index.py`
- `python3 -m py_compile scripts/normalize_report_index.py scripts/validate_corpus_schema.py tests/test_normalize_report_index.py tests/test_selected_report_targets_manifest.py`
- `git diff --check`

If Day 8 changes C files through formatting only, run the full quality gate
requested for code changes.

## Validation

Day 7 is documentation-only. Validation:

- `git diff --check`

## Completion Criteria Review

| Criterion | Status | Evidence |
| --- | --- | --- |
| Report normalizer refactor is scoped before implementation. | Complete | Refactor scope limits Day 8 to oracle/comparison selected freshness while deferring broader rows to Day 9. |
| Selected target expectations have a manifest migration path. | Complete | Expected-count migration table maps current constants to manifest fields. |
| Existing advisory behavior remains intentional. | Complete | Compatibility section preserves unselected and advisory family behavior. |
