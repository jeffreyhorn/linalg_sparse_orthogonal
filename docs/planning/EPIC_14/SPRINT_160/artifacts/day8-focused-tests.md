# Day 8 Focused Tests Implementation

## Summary

Day 8 added a focused black-box test for the external comparison runner CLI.
The test covers the descriptor-backed target dispatch and generated TSV
contract introduced for Sprint 160 without expanding QR C proof-owner tests.

## Implemented Test Surface

Added:

```text
tests/test_run_external_comparison.py
```

The test file executes `scripts/run_external_comparison.py` through the CLI
and writes generated artifacts to isolated temporary directories.

## Covered Cases

| Case | Coverage |
| --- | --- |
| Unsupported target | Verifies nonzero exit and `unsupported_target` diagnostics include both supported target names. |
| `qr-minnorm` generation | Verifies required output files, manifest target, fixture key, study path, six selected row IDs, pass statuses, subfamily, operation, support tier, and artifact path. |
| `qr-compatible-ls` generation | Verifies required output files, manifest target, fixture key, study path, six selected row IDs, pass statuses, subfamily, operation, support tier, and artifact path. |
| Row metrics | Verifies each target emits the six expected selected metrics exactly once. |
| Optional dependency rows | Verifies NumPy and SciPy remain deferred non-proof context in `dependency_status.tsv`. |
| Output isolation | Uses `--output-dir` so tests do not depend on stale `build/comparison` artifacts. |

## C Proof-Owner Decision

No C or header tests were added.

The current sprint changes are comparison harness, Make freshness, report
metadata, and normalizer changes. Existing C QR solve behavior remains owned by
`tests/test_qr_solve.c`, including the selected compatible overdetermined 5x3
fixture. Adding a new C test on Day 8 would duplicate solver proof rather than
cover the newly touched script/report behavior.

## Validation Plan

Required commands for Day 8:

```sh
python3 -m py_compile scripts/run_external_comparison.py scripts/normalize_report_index.py tests/test_normalize_report_index.py tests/test_run_external_comparison.py
python3 scripts/run_external_comparison.py --self-check
python3 tests/test_run_external_comparison.py
python3 tests/test_normalize_report_index.py
python3 scripts/validate_corpus_schema.py
make report-index-comparison-freshness
git diff --check
```

No `.c` or `.h` files changed on Day 8, so the full
`make format && make lint && make test` gate is not required for this day.

## Completion Check

- The selected comparison targets now have direct CLI regression coverage.
- Unsupported target dispatch fails with actionable diagnostics.
- Generated row IDs and metadata are checked for both selected QR comparison
  families.
- Optional dependency rows remain explicit non-proof context.
- Existing normalizer row-state coverage remains the freshness failure owner.

## Day 9 Handoff

Day 9 can proceed with report integration design using the now-tested runner
contract:

- two selected targets;
- six selected pass rows per target;
- isolated output directory support;
- target-specific fixture, subfamily, operation, and artifact path metadata;
- optional NumPy/SciPy rows preserved as deferred context only.
