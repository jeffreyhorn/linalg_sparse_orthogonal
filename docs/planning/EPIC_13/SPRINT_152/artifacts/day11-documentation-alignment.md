# Sprint 152 Day 11 Documentation Alignment

## Purpose

Day 11 aligns active documentation and report metadata with the selected
generated freshness policy implemented on Days 6-10. The maintained selected
oracle command surface is now `make report-index-oracle-freshness`.

## Updated Documentation

### High-Level Guidance

Updated `README.md` so the QR capability section points at
`make report-index-oracle-freshness` as the local oracle/report freshness gate.
The wording keeps the evidence fixture-local and explicitly excludes hosted CI,
platform, performance, package/ABI, and state-of-the-art claims.

Updated `docs/solver_selection.md` so:

- QR evidence points at the selected local oracle freshness gate;
- partial-SVD evidence points at the selected local oracle freshness gate;
- the partial-SVD-only oracle command remains a focused debugging variant, not
  the selected combined freshness gate.

Updated `docs/algorithm.md` so the QR algorithm evidence section describes the
combined local gate and keeps QR-only oracle runs as focused debug variants.

### Maintainer Guidance

Updated `docs/maintainer_guide.md` with a new
`Selected Oracle Freshness Gate` section.

The section documents:

- command: `make report-index-oracle-freshness`;
- generated paths:
  - `build/corpus/oracle/corpus.oracle.tsv`;
  - `build/corpus-reports/index.tsv`;
  - `build/corpus-reports/skips.tsv`;
  - `build/corpus-reports/manifest.txt`;
- selected row-count policy:
  - `52` total oracle rows;
  - `3` generated-reference rows;
  - `23` QR solver-backed rows;
  - `26` partial-SVD solver-backed rows;
- failure classes:
  - missing artifacts;
  - stale commits;
  - failing comparison rows;
  - row-count mismatches;
  - missing solver families;
  - missing fixture keys;
- local-only artifact and non-claim policy.

The QR and partial-SVD maintenance sections now include
`make report-index-oracle-freshness` and explicitly label QR-only and
partial-SVD-only oracle commands as focused debugging variants that do not
satisfy the Sprint 152 selected combined row-count policy by themselves.

The normalized report-index workflow now lists
`make report-index-oracle-freshness` as the preferred selected oracle check and
documents the selected oracle error classes.

### Report Schema And Metadata

Updated `tests/corpus/schemas/report_index_fields.md` with a selected oracle
freshness gate section that records:

- maintained target command;
- selected row-count policy;
- selected diagnostic identifiers:
  - `oracle_selected_row_count`;
  - `oracle_selected_solver_families`;
  - `oracle_selected_fixture_keys`;
- local-only artifact and non-claim policy.

Updated `tests/corpus/manifests/report_families.tsv` so both selected oracle
contract rows use `make report-index-oracle-freshness` as their
source-controlled generator command.

## Stale Wording Search Result

Searched active documentation for:

- `run_corpus_oracle.py --include-solver-qr`;
- `run_corpus_oracle.py --include-partial-svd`;
- `require-generated oracle`;
- `report-index-oracle-freshness`;
- `105 rows`;
- `QR-only`;
- `partial-SVD-only`.

Remaining QR-only and partial-SVD-only command references are intentional and
documented as focused debugging variants. Active selected freshness wording now
points at `make report-index-oracle-freshness`.

## Validation

Commands run:

```sh
make report-index-oracle-freshness
python3 scripts/validate_corpus_schema.py
python3 -m py_compile scripts/normalize_report_index.py tests/test_normalize_report_index.py
python3 tests/test_normalize_report_index.py
```

All commands passed after the manifest and documentation updates.

## Non-Claims

The documentation alignment does not claim hosted CI oracle proof, release
artifact proof, package-manager availability, shared-library ABI support,
dynamic-loader support, broad platform support, compiler portability, broad QR
correctness, broad partial-SVD correctness, external-library parity, portable
performance, benchmark superiority, complete coverage, zero dead code, or
state-of-the-art sparse linear algebra status.
