# Sprint 216 Day 10: Focused Contract Validation

## Scope And Environment

This is the current-tree execution of the [Day 9 selected matrix](day9-residual-decision-validation-matrix.md), not a rerun of Sprint 207-215 hosted evidence. Commands ran from the repository root on `sprint-216`, Darwin `24.6.0` x86_64, Apple Clang `17.0.0`, Python `3.14.5`, with Doxygen installed at `/usr/local/bin/doxygen`. The checkout was dirty at `c5f30490fc02d34ed27be363e1586c3472089faf`; generated report metadata correctly records the dirty worktree. No `.c` or `.h` file changed. All rows below exited `0`; no failing check required a code fix.

## Command And Result Log

| Command | Exit | Observed result and boundary |
| --- | ---: | --- |
| `python3 -m unittest tests.test_external_baseline_claim_docs` | 0 | 5 claim and mutation tests passed; text guard, not external solver evidence. |
| `python3 -m py_compile tests/test_external_baseline_claim_docs.py` | 0 | Python syntax passed. |
| `python3 tests/test_selected_performance_docs.py` | 0 | Selected threshold-free wording guard passed. |
| `make support-docs-guard` | 0 | Support quick-reference guard passed. |
| `make api-docs-freshness` | 0 | Doxygen generated 18 reference and 18 source pages for 18 public headers; local-only workflow/output and routing checks passed. Generated HTML is ignored local output. |
| `python3 scripts/validate_corpus_schema.py` | 0 | Source-controlled corpus schema passed; this does not supply external v1 evidence semantics. |
| `python3 tests/test_selected_report_targets_manifest.py` | 0 | Selected report target manifest guard passed. |
| `python3 tests/test_normalize_report_index.py` | 0 | Report-index normalizer tests passed. |
| `python3 tests/test_run_external_comparison.py` | 0 | Source-controlled-helper comparison tests passed. |
| `make report-index-comparison-freshness` | 0 | Generated six local comparison bundles and accepted 46 report-index rows as selected freshness; see inspection below. |
| `make bench-canonical-report-freshness-tests` | 0 | 35 positive/negative selected benchmark freshness cases passed, including threshold, metadata and malformed-row rejection. |
| `make bench-canonical-report-freshness` | 0 | Four local CSVs, `index.tsv` and `manifest.txt` generated; selected `bench_refactor_csc` freshness passed with no timing threshold. |
| `python3 -m unittest tests.test_run_cholmod_pilot` | 0 | 13 synthetic tests passed, including schema-gate fail-closed behavior and SHA-256 of raw completed/timeout stdout and stderr bytes. No solver ran. |
| `make docs-check` | 0 | 18 checked-in public headers, 18 generated reference pages and 18 generated source pages. Local generation only. |

`make api-docs-freshness` also exercised `docs-check` internally; the separate invocation above confirms the Day 10 coverage gate explicitly. The API guard regressions completed with exit `0` after its local-only and routing checks. The generated HTML resides at `docs/api/html/` and was neither staged nor published.

## Generated Artifact Inspection

- `build/comparison/{cholesky_spd_tridiag_5,lu_nonsym_square_5,partial_svd_diag6_k2,qr_compatible_ls,qr_incompatible_ls,qr_minnorm}/` each contains `manifest.tsv`, `dependency_status.tsv`, `study.tsv`, observations and `summary.md`. Across six `study.tsv` files, all 40 rows have `status=pass`, `support_tier=local_only`, baseline type `external-process-source-controlled-helper`, and a `no state-of-the-art claim` non-claim. The manifest pins `source_branch=sprint-216`, the above commit, `worktree_state=dirty`, and `platform=darwin-x86_64`. This is a local source-controlled dense-helper comparison, not CHOLMOD or independent package parity.
- Across the six `dependency_status.tsv` files, Python and the selected source-controlled helper are available; optional NumPy and SciPy rows are `defer` (six each), **not** passes. The generated report index reported 46 fresh/source-controlled rows. The generated comparison manifest does not itself claim a content hash; a Day 10 SHA-256 snapshot of `build/comparison/cholesky_spd_tridiag_5/study.tsv` is `03fdbbd2ceeebe85b7166325dda0a68d20001bd5a2fa8eed57dbebc371bb0751`. Regeneration changes timestamps and therefore may change this snapshot digest.
- `build/bench-reports/canonical/manifest.txt` records `runner_context=local`, `support_tier=local_only`, `status=measurement`, `baseline=n/a`, `threshold=n/a`, `warmup=none_configured`, `variance=not_computed_single_sample`, `selected_matrix_size=n=100`, `backend_context=n/a`, and `methodology_notes=threshold_free_local_measurement;not_portable_performance_claim`. `index.tsv` has four rows; the selected `bench_refactor_csc` command is `tests/data/suitesparse/nos4.mtx --repeat 1`. Its SHA-256 snapshot is `56156b2f5793349d2409c5e1257da731c56c9d06bbbe33269bc51f35591b5aed`. Neither this digest nor an exit-zero freshness check establishes timing acceptance.
- Pilot tests use temporary bundles, not a retained accepted `evidence.json`. The completed-command test verifies exact CRLF/invalid-UTF-8 bytes in both saved streams and hashes against those bytes; schema-preflight tests verify `provisional_status=blocked` and `timing=not_run` before build/solver execution. There is no new external pilot artifact to inspect or hash.

## Stop Conditions And Remaining Gates

The external v1 schema/validator, configured CHOLMOD execution, two-sided accepted correctness, rights/publication approval, hosted Windows freshness, candidate static release, shared ABI loader proof, and hosted timing matrix remain **not run or unavailable**, not passing evidence. These gaps stay in the [residual queue](../RESIDUAL_QUEUE.md). Day 11 owns the serial installed-static/package checks; `tests/test_install.sh` invokes `make clean`, so it was not run during Day 10 report generation. The full `make format && make lint && make test` source chain was not required because there is no C/header change. Rebuild before any later report freshness rerun after install checks. The no-state-of-the-art decision remains unchanged.
