# Sprint 190 Day 10: Claim Calibration

## Purpose

Calibrate public and maintainer documentation to match the Sprint 190 Windows
selected Cholesky workflow without overstating Windows report freshness.

## Claim Decision

Documentation now uses this split:

- Windows has one bounded selected Cholesky comparison freshness workflow for
  `cholesky-spd-tridiag-5`;
- the workflow uses the Sprint 190 target-specific freshness command and exact
  artifact name;
- broad Windows report freshness remains out of scope;
- Windows selected oracle freshness and selected benchmark freshness remain out
  of scope;
- selected target manifest metadata still omits `windows` until metadata,
  support tier, and claim wording are reviewed together;
- local PowerShell unavailability remains residual evidence, not pass evidence.

## Updated Documentation

Updated `README.md` to describe:

- the one bounded Windows selected Cholesky workflow;
- unsupported Windows Makefile, `pkg-config`, package-manager, shared-library,
  dynamic ABI, runtime-loader, broad report freshness, selected oracle
  freshness, selected benchmark freshness, and broad parity claims;
- the Sprint 190 command and artifact name;
- the Sprint 182 deferral continuing for every other Windows report freshness
  surface.

Updated `INSTALL.md` to include the Windows Cholesky lane in the platform table
without widening the Windows support contract.

Updated `docs/maintainer_guide.md` to document:

- the exact target-specific freshness command;
- artifact `sprint190-windows-selected-comparison-cholesky`;
- the difference between PowerShell validation ownership and the separate
  selected Cholesky report lane;
- the broader Windows report freshness deferral boundary.

Updated `tests/corpus/README.md` and
`tests/corpus/schemas/report_index_fields.md` to keep selected-target manifest
interpretation aligned with the staged metadata decision.

## Claim-Boundary Guard

`scripts/validate_windows_powershell.py` now requires the new bounded-Cholesky
claim markers in:

- `README.md`;
- `INSTALL.md`;
- `docs/maintainer_guide.md`;
- `tests/corpus/README.md`.

The guard still rejects unsupported wording that says Windows report freshness
is broadly supported, promoted, complete, or closed.

## Remaining Metadata Boundary

The source manifest still does not list `windows`. Day 10 intentionally leaves
manifest promotion for a later step so hosted evidence, support tier, and
source metadata can be reviewed together.

## Validation

Commands run:

- `python3 tests/test_validate_windows_powershell.py`
- `python3 tests/test_selected_comparison_workflow.py`
- `python3 tests/test_selected_report_targets_manifest.py`
- `python3 scripts/validate_corpus_schema.py`

All focused Day 10 validation commands passed.

No `.c` or `.h` files were modified, so the full `make format && make lint &&
make test` C gate is not required for Day 10.
