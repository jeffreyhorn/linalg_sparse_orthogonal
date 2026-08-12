# Sprint 152 Day 10 CI And Artifact Policy Implementation

## Purpose

Day 10 implements the Day 9 CI/artifact policy follow-through. Sprint 152 keeps
selected oracle freshness local-required and does not add hosted CI artifact
publication for `build/corpus/oracle/`, `build/corpus-reports/`, or
`build/report-index/`.

## Implemented Local Command Surface

Added a maintained Makefile target:

```sh
make report-index-oracle-freshness
```

The target:

- depends on `$(LIB)` so the solver-backed QR oracle path has the static
  library available;
- regenerates selected oracle output with:
  `python3 scripts/run_corpus_oracle.py --include-solver-qr --include-partial-svd`;
- checks required selected oracle freshness with:
  `python3 scripts/normalize_report_index.py --family oracle --require-generated oracle --check-freshness`;
- prints scoped banners so failures have a stable command owner;
- leaves generated files under ignored local `build/` paths.

## CI And Artifact Follow-Through

No hosted workflow files were changed.

No artifact uploads were added for:

- `build/corpus/oracle/`;
- `build/corpus-reports/`;
- `build/report-index/`.

Existing hosted uploads remain unchanged:

- dead-code report artifacts in the Linux dead-code job;
- coverage HTML in the Linux supplemental coverage job.

This implements the Day 9 decision that selected oracle freshness is a local
required gate in Sprint 152, not hosted CI proof or release artifact proof.

## Validation

Commands run:

```sh
make report-index-oracle-freshness
python3 -m py_compile scripts/normalize_report_index.py tests/test_normalize_report_index.py
python3 tests/test_normalize_report_index.py
```

All commands passed. The `make report-index-oracle-freshness` run generated
current local oracle/report files and completed the required freshness check
with `normalize-report-index: freshness ok (54 rows)`.

## Generated Artifact Policy Result

Generated outputs remain uncommitted and ignored:

- `build/corpus/oracle/corpus.oracle.tsv`;
- `build/corpus-reports/index.tsv`;
- `build/corpus-reports/skips.tsv`;
- `build/corpus-reports/manifest.txt`.

Maintainers should regenerate them through the Makefile target rather than
editing or committing generated rows.

## Non-Claims

This implementation does not claim hosted CI oracle proof, release artifact
proof, package-manager availability, shared-library ABI support, dynamic-loader
support, broad platform support, compiler portability, broad QR correctness,
broad partial-SVD correctness, external-library parity, portable performance,
benchmark superiority, complete coverage, zero dead code, or state-of-the-art
sparse linear algebra status.
