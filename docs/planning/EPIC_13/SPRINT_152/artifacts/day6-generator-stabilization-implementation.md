# Sprint 152 Day 6 Generator Stabilization Implementation

## Purpose

Day 6 implements the selected oracle generator stabilization designed on Day 5.
The implementation keeps generated oracle evidence local-only while making
required freshness failures actionable and resistant to partial or stale
generated output.

## Implemented Changes

- Added the canonical selected oracle generation command to
  `scripts/normalize_report_index.py`:
  `python3 scripts/run_corpus_oracle.py --include-solver-qr --include-partial-svd`.
- Added selected oracle row-count policy:
  - total oracle rows: `52`;
  - generated-reference `unknown` rows: `3`;
  - QR solver-backed rows: `23`;
  - partial-SVD solver-backed rows: `26`.
- Added selected fixture-key policy for the QR and partial-SVD maintained
  corpus families selected in Days 3-5.
- Added selected oracle diagnostics for:
  - unexpected row counts;
  - missing solver families;
  - missing fixture keys;
  - missing required oracle artifacts;
  - stale oracle source commits;
  - oracle comparison failures.
- Kept selected oracle policy enforcement scoped to callers that require the
  `oracle` generated family or request strict generated freshness.
- Preserved QR-only and partial-SVD-only development variants for local
  debugging; those variants no longer satisfy the combined selected row-count
  policy when oracle freshness is required.
- Preserved `scripts/run_corpus_oracle.py` stale-output cleanup behavior:
  existing oracle TSVs and corpus-report `index.tsv`, `skips.tsv`, and
  `manifest.txt` are removed before writing current output.

## Test Coverage

Updated `tests/test_normalize_report_index.py` with synthetic selected oracle
fixtures so policy tests do not depend on a compiled solver library:

- complete selected oracle output passes required freshness without selected
  row-count or fixture-key diagnostics;
- partial selected oracle output fails required freshness with an actionable
  `oracle_selected_row_count` mismatch;
- existing stale oracle freshness tests continue to verify advisory versus
  strict failure behavior.

## Validation

Commands run:

```sh
python3 -m py_compile scripts/normalize_report_index.py tests/test_normalize_report_index.py
python3 tests/test_normalize_report_index.py
python3 scripts/validate_corpus_schema.py
python3 scripts/run_corpus_oracle.py --include-solver-qr --include-partial-svd
python3 scripts/normalize_report_index.py --family oracle --require-generated oracle --check-freshness
```

All commands passed. The selected combined oracle command produced
`build/corpus/oracle/corpus.oracle.tsv` and the corresponding corpus-report
files, and the required oracle freshness check completed with freshness
warnings only, not selected-family errors.

## Non-Claims

This implementation does not promote local generated oracle rows to hosted CI
proof, release proof, package proof, ABI proof, platform proof, performance
proof, external-library parity, broad QR correctness, broad partial-SVD
correctness, or state-of-the-art sparse linear algebra claims.

Generated `build/` artifacts remain uncommitted.
