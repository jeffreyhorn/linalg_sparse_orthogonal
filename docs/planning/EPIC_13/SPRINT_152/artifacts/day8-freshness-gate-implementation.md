# Sprint 152 Day 8 Freshness Gate Implementation

## Purpose

Day 8 converts the Day 7 freshness-gate matrix into executable report-index
coverage. The implementation keeps the selected oracle family local-only and
fixture-scoped while making missing, stale, failing, partial, and mismatched
generated oracle output fail deterministically when the oracle family is
required.

## Implemented Coverage

Updated `tests/test_normalize_report_index.py` with expanded synthetic selected
oracle fixtures and focused gate tests.

### Synthetic Oracle Fixtures

The test helper can now generate selected oracle TSV rows with controlled
failure modes:

- complete selected family: `52` rows;
- partial family: one selected row removed;
- missing solver family: all rows for a solver family omitted;
- stale row: one generated row uses an old `source_commit`;
- failing row: one generated row reports `comparison_status=fail`;
- missing fixture key: rows for one selected fixture are remapped while total
  row count and solver-family counts remain unchanged.

This keeps gate tests deterministic and avoids depending on a compiled solver
library for every mismatch case.

### Gate Cases Added

- Missing required oracle artifacts fail with the selected artifact pattern and
  canonical regeneration command.
- Complete selected oracle output passes required freshness without selected
  row-count or fixture-key errors.
- Partial selected oracle output fails with `oracle_selected_row_count`.
- Stale required oracle output fails with recorded commit, current commit, and
  artifact path diagnostics.
- Failing oracle rows fail with fixture key and artifact path diagnostics.
- Missing solver-family output fails with
  `oracle_selected_solver_families`.
- Missing fixture-key output fails with
  `oracle_selected_fixture_keys` even when total row count remains correct.
- Advisory/source-controlled compatibility remains intact for `coverage` and
  `package` families.

## Current Gate Behavior

Required selected oracle gate:

```sh
python3 scripts/run_corpus_oracle.py --include-solver-qr --include-partial-svd
python3 scripts/normalize_report_index.py --family oracle --require-generated oracle --check-freshness
```

This gate now has executable coverage for the Day 7 required behaviors. It
passes complete current selected oracle output and fails missing, stale,
failing, incomplete, or mismatched selected oracle output.

Strict generated mode continues to fail stale, failing, or incomplete selected
oracle rows while preserving advisory non-selected family boundaries.

## Validation

Commands run:

```sh
python3 -m py_compile scripts/normalize_report_index.py tests/test_normalize_report_index.py
python3 tests/test_normalize_report_index.py
python3 scripts/validate_corpus_schema.py
python3 scripts/run_corpus_oracle.py --include-solver-qr --include-partial-svd
python3 scripts/normalize_report_index.py --family oracle --require-generated oracle --check-freshness
```

All commands passed. The generated oracle/report files are local ignored
`build/` outputs and were not added to source control.

## Residuals For Later Days

- Day 9 must decide whether any selected generated freshness gate should run
  in hosted CI or remain local-only.
- Day 10 must implement the selected CI/artifact posture, if any.
- Day 11 must align maintainer and report-index documentation with the
  executable gate surface.
- Strict `generated_present_unchecked` warning semantics remain visible; this
  sprint has not converted those warnings into broad release proof.

## Non-Claims

This implementation does not claim broad QR correctness, broad partial-SVD
correctness, external-library parity, hosted CI proof, release artifact proof,
package-manager availability, shared-library ABI support, broad platform
support, portable performance, benchmark superiority, complete coverage, zero
dead code, or state-of-the-art sparse linear algebra status.
