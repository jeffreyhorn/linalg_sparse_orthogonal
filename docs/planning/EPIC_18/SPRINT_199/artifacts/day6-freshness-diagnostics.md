# Sprint 199 Day 6: Freshness Diagnostics

## Purpose

Ensure selected Windows Cholesky freshness diagnostics cannot silently pass
when selected rows are missing, stale, failed, dependency-only, or generated
for the wrong selected comparison target.

## Changed Files

| File | Change |
| --- | --- |
| `scripts/normalize_report_index.py` | Added a selected-target row-set mismatch diagnostic when generated comparison rows exist but none match the requested selected artifact. |
| `tests/test_normalize_report_index.py` | Added a wrong-target regression test and kept the expanded Windows path/stale-row coverage in the direct runner. |

## Diagnostic Gap Closed

Before Day 6, a command such as:

```sh
python scripts/normalize_report_index.py --family comparison --require-generated comparison --check-freshness --selected-target cholesky-spd-tridiag-5
```

could see generated rows for another comparison artifact and still have no
selected Cholesky generated rows. That made the selected target unavailable,
but the selected comparison policy did not emit a row-set mismatch because the
wrong artifact rows were filtered out before the row-set check.

Day 6 changes this behavior: if generated comparison rows exist and none match
the selected target artifact, the normalizer now emits a target-specific
`comparison_selected_rows` error with:

- selected target ID;
- expected selected row count;
- `observed=0`;
- the missing selected row IDs;
- selected artifact diagnostic;
- selected-target remediation command.

## New Test

| Test | Coverage |
| --- | --- |
| `test_selected_comparison_target_freshness_rejects_wrong_target_rows` | Writes only QR incompatible selected comparison rows, then checks `--selected-target cholesky-spd-tridiag-5` fails with a selected Cholesky row-set mismatch instead of passing silently. |

## Existing Coverage Reconfirmed

| Scenario | Existing test coverage |
| --- | --- |
| No generated comparison family | `test_selected_comparison_target_freshness_accepts_cholesky_subset` first checks the missing-family error before writing rows. |
| Stale selected Cholesky rows | `test_selected_comparison_target_freshness_rejects_cholesky_stale_or_failed` |
| Failed selected Cholesky rows | `test_selected_comparison_target_freshness_rejects_cholesky_stale_or_failed` |
| Windows backslash stale rows | `test_selected_comparison_target_freshness_rejects_windows_path_stale_rows` |
| Dependency-only selected rows | `test_qr_incompatible_selected_freshness_rejects_dependency_only_rows`; equivalent row-set behavior is shared by selected comparison diagnostics. |
| Duplicate selected rows | `test_selected_comparison_required_freshness_rejects_duplicate_rows` |
| Skipped or deferred selected rows | `test_selected_comparison_required_freshness_rejects_stale_and_invalid_rows` |

## Wrong Platform And Build-System Classification

Day 6 did not add a platform/compiler rejection rule because the current
selected target manifest still keeps `SRT-COMP-CHOLESKY-SPD-TRIDIAG-5` at
`workflow_platforms=linux;macos` and does not define an authoritative Windows
compiler tuple. Rejecting or accepting rows by `platform=windows-amd64` and
`compiler=cmake-probe:Visual Studio 17 2022:Release` before manifest
promotion would create policy that the manifest does not yet own.

For Sprint 199, wrong platform/build-system evidence remains a Day 7/Day 8
metadata-policy residual. Any future platform/compiler diagnostic should be
introduced together with the manifest promotion or an explicit manifest field
that states the expected Windows hosted tuple.

## Validation

| Command | Result |
| --- | --- |
| `python3 tests/test_normalize_report_index.py` | Passed |

## Promotion Impact

Day 6 closes the wrong-target silent-pass gap and improves selected
diagnostics for unavailable selected Cholesky rows. Manifest promotion still
remains blocked by:

- generated `support_tier=local_only`;
- generated summary/non-claim wording that still says `no hosted CI proof` and
  `no Windows report freshness`;
- lack of an authoritative selected manifest policy for Windows
  platform/compiler matching;
- public and maintainer documentation still describing the Windows path as
  guarded rather than promoted.

## Day 7 Handoff

Day 7 should decide whether any final normalizer implementation hardening is
needed before workflow/docs alignment. In particular, it should avoid adding
platform/compiler rejection semantics unless the selected manifest is updated
to own those semantics in the same change.
