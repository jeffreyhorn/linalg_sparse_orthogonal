# Day 7 Focused Proof-Owner Test Design

## Summary

Day 7 defines the focused test plan for the two-family selected QR comparison
surface introduced by Days 5 and 6.

The test scope stays on generated comparison rows, target dispatch, selected
freshness semantics, and normalization. No C proof-owner expansion is required
for Day 8 unless later work changes QR solver behavior, fixture helpers, or
public C APIs.

## Touched Behavior

| Surface | Current behavior to protect | Test owner |
| --- | --- | --- |
| `scripts/run_external_comparison.py` | Descriptor dispatch preserves `qr-minnorm` and adds `qr-compatible-ls`. | New focused script tests plus existing `--self-check`. |
| `Makefile` | `report-index-comparison-freshness` regenerates both selected targets before normalization. | Make target smoke check. |
| `scripts/normalize_report_index.py` | Selected comparison row set is exactly 12 rows split across `qr_minnorm` and `qr_compatible_ls`. | Existing focused normalizer tests. |
| `tests/test_normalize_report_index.py` | Missing, unexpected, duplicate, stale, fail, and defer rows fail closed. | Existing focused normalizer tests. |
| `tests/corpus/manifests/report_families.tsv` | `comparison/qr_compatible_ls` metadata is source-controlled and local-only. | Corpus schema validation. |
| `tests/test_qr_solve.c` | Existing compatible 5x3 QR solve proof remains the C behavior owner. | Existing C tests only; no new C test unless solver code changes. |

## Existing Coverage

| Coverage area | Existing command or test | Day 7 assessment |
| --- | --- | --- |
| Descriptor internal invariants | `python3 scripts/run_external_comparison.py --self-check` | Keep as a low-cost guard for selected row validation, malformed output handling, and descriptor consistency. |
| Minimum-norm generation | `python3 scripts/run_external_comparison.py --target qr-minnorm` | Required smoke check to prevent regression of the existing selected family. |
| Compatible least-squares generation | `python3 scripts/run_external_comparison.py --target qr-compatible-ls` | Required smoke check for the new selected family. |
| Two-family selected freshness | `make report-index-comparison-freshness` | Required end-to-end check because it regenerates both targets and runs strict normalization. |
| Normalizer row-state failures | `python3 tests/test_normalize_report_index.py` | Already covers complete, missing, unexpected, duplicate, stale, fail, and defer selected comparison rows. |
| Report metadata schema | `python3 scripts/validate_corpus_schema.py` | Required after `report_families.tsv` changes. |
| C QR solve behavior | `tests/test_qr_solve.c` | Existing proof owner already covers `qr_overdetermined_compatible_5x3`; no Day 8 C expansion needed. |

## Day 8 Script Test Additions

Day 8 should add a small focused script test for
`scripts/run_external_comparison.py` rather than expanding broad QR C tests.

Recommended coverage:

1. Unsupported targets fail clearly.
   - Invoke the runner with an invalid target.
   - Assert nonzero exit.
   - Assert diagnostics include `unsupported_target` and the supported target
     names.

2. `qr-compatible-ls` generates the expected selected row family.
   - Run the target with an isolated output directory if possible.
   - Assert `study.tsv`, `summary.md`, `manifest.tsv`,
     `project_observations.tsv`, `baseline_observations.tsv`, and
     `dependency_status.tsv` exist.
   - Assert the six `qr_overdetermined_compatible_5x3` row IDs are present
     exactly once and all selected rows have `status=pass`.

3. `qr-minnorm` remains unchanged.
   - Run the target with an isolated output directory if possible.
   - Assert the six `qr_underdetermined_minnorm_2x4` row IDs are present
     exactly once and all selected rows have `status=pass`.

4. Target-specific metadata remains bounded.
   - Assert the compatible least-squares rows use subfamily
     `qr_compatible_ls`, fixture `qr_overdetermined_compatible_5x3`, and
     operation `least_squares_solve`.
   - Assert the minimum-norm rows keep subfamily `qr_minnorm`, fixture
     `qr_underdetermined_minnorm_2x4`, and operation
     `minnorm_solve`.

5. Optional dependency rows remain non-proof context.
   - Assert dependency status output can represent deferred optional external
     packages without converting them into selected pass evidence.

## C Proof-Owner Decision

No Day 8 C or header test is required.

Rationale:

- Day 5 changed generated comparison harness behavior, not QR numerical
  implementation.
- Day 6 changed report metadata and normalization, not fixture generation or
  C solver semantics.
- `tests/test_qr_solve.c` already owns the selected compatible overdetermined
  QR fixture against the dense helper.
- Adding another C test would duplicate existing solver proof rather than
  covering the newly touched script/report behavior.

C/header gates remain reserved for an actual C or header change. If later work
changes `.c` or `.h` files, the required validation becomes:

```sh
make format && make lint && make test
```

## Row-State Failure Matrix

| Row state | Expected result | Existing or planned owner |
| --- | --- | --- |
| Complete current 12-row selected set with all `pass` statuses | Freshness passes. | Existing normalizer test and `make report-index-comparison-freshness`. |
| Missing selected row | Freshness fails with selected row-set mismatch. | Existing normalizer test. |
| Unexpected selected-family row | Freshness fails with unexpected row-set mismatch. | Existing normalizer test. |
| Duplicate selected row | Freshness fails with duplicate row diagnostics. | Existing normalizer test. |
| Stale `source_commit` | Freshness fails as stale generated evidence. | Existing normalizer test. |
| Selected row with `fail` status | Freshness fails and cannot count as evidence. | Existing normalizer test. |
| Selected row with `defer` or skipped status | Freshness fails and remains non-proof context. | Existing normalizer test. |
| Malformed baseline output | Harness fails before selected rows can be published. | Existing self-check; candidate Day 8 runner test only if cheap to isolate. |
| Project probe build/run failure | Harness fails before selected rows can be published. | Existing harness diagnostics; keep as smoke-command coverage. |
| Tolerance miss | Generated row is `fail`; freshness rejects it. | Existing normalizer failure path plus harness self-check coverage. |

## Validation Command Matrix

| Changed-file surface | Required validation |
| --- | --- |
| Sprint artifacts and notes only | `git diff --check` and trailing-whitespace scan for touched docs. |
| Python script changes | `python3 -m py_compile scripts/run_external_comparison.py scripts/normalize_report_index.py tests/test_normalize_report_index.py`; `python3 scripts/run_external_comparison.py --self-check`; target-specific runner checks; focused Python tests. |
| Report metadata changes | `python3 scripts/validate_corpus_schema.py`. |
| Makefile freshness wiring | `make report-index-comparison-freshness`. |
| Normalizer semantics | `python3 tests/test_normalize_report_index.py`. |
| C or header changes | `make format && make lint && make test`. |

## Day 8 Handoff

Day 8 should implement the missing harness-level script tests, preferably in a
small focused Python test file, while leaving C proof-owner coverage unchanged.
The existing normalizer tests already cover the selected row-state freshness
failures, so Day 8 should avoid duplicating those cases unless it needs a
runner-specific assertion that the generated rows feed the normalizer with the
expected IDs, subfamilies, fixtures, operations, and statuses.
