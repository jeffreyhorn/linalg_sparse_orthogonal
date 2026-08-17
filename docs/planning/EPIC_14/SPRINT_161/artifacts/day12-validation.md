# Day 12 Local Validation

Day 12 ran the focused validation pass for the selected partial-SVD comparison
publication surface and the QR comparison behavior it extends.

## Comparison Generation

| Command | Result | Evidence |
| --- | --- | --- |
| `python3 scripts/run_external_comparison.py --target partial-svd-diag6-k2` | Passed | Regenerated `build/comparison/partial_svd_diag6_k2/{project_observations.tsv,baseline_observations.tsv,dependency_status.tsv,study.tsv,summary.md,manifest.tsv}` and reported `partial-svd-diag6-k2 project-vs-baseline comparison passed`. |
| `python3 scripts/run_external_comparison.py --target qr-minnorm` | Passed | Regenerated `build/comparison/qr_minnorm/*` and reported `qr-minnorm project-vs-baseline comparison passed`. |
| `python3 scripts/run_external_comparison.py --target qr-compatible-ls` | Passed | Regenerated `build/comparison/qr_compatible_ls/*` and reported `qr-compatible-ls project-vs-baseline comparison passed`. |

## Freshness Gates

| Command | Result | Notes |
| --- | --- | --- |
| `make report-index-comparison-freshness` | Passed | Regenerated the two selected QR comparison families and `partial_svd_diag6_k2`; freshness check reported `normalize-report-index: freshness ok (25 rows)`. |
| `make report-index-oracle-freshness` | Passed | Regenerated selected QR/partial-SVD oracle output; freshness check reported `normalize-report-index: freshness ok (54 rows)`. |
| `python3 scripts/normalize_report_index.py --family corpus --family oracle --family comparison --check` | Passed | Normalized combined corpus/oracle/comparison index reported `153 rows ok`. |

## Schema And Targeted Tests

| Command | Result |
| --- | --- |
| `python3 scripts/validate_corpus_schema.py` | Passed |
| `python3 tests/test_normalize_report_index.py` | Passed |
| `python3 tests/test_run_external_comparison.py` | Passed |
| `python3 -m py_compile scripts/normalize_report_index.py scripts/run_external_comparison.py tests/test_normalize_report_index.py tests/test_run_external_comparison.py scripts/validate_corpus_schema.py` | Passed |
| `git diff --check` | Passed |

## Changed-File Gate Decision

No `.c` or `.h` files are modified on the branch after Day 12 validation.
Per the sprint plan, `make format`, `make lint`, and `make test` are not
required for this validation-only day.

## Remaining Risk

The comparison and oracle outputs remain ignored local artifacts under
`build/`. Passing Day 12 validation is local generated evidence only; it does
not create broad partial-SVD correctness, external-library parity, hosted
proof, release proof, package proof, ABI proof, performance proof, or
state-of-the-art evidence.
