# Day 5: Selected Generator Fixes

## Purpose

Day 5 checked whether Sprint 203 had a justified generator, path, or CMake
repair to apply for `qr-incompatible-ls`. The answer is no for this branch
state: local runner tests, local CMake probe execution, and target-specific
freshness all pass, while the only promotion blocker remains missing hosted
Windows/MSVC evidence.

## Implementation Result

No source, workflow, manifest, or public documentation edits were made on Day
5. The selected generator path already supports the required local behavior:

- explicit CMake probe mode;
- explicit generator, architecture, configuration, and library options;
- forward-slash CMake path literal rendering for Windows-style paths;
- MSVC-safe omission of `m` linkage through `if(NOT MSVC)`;
- selected-target freshness filtering that normalizes backslashes.

## Validation Run

| Command | Result |
| --- | --- |
| `python3 tests/test_run_external_comparison.py` | Passed. |
| `python3 scripts/normalize_report_index.py --family comparison --require-generated comparison --check-freshness --selected-target qr-incompatible-ls` | Passed; six selected rows fresh, `46` comparison rows total. |
| `python3 scripts/run_external_comparison.py --target qr-incompatible-ls --probe-build-system cmake --cmake-config Release --keep-temp` | Passed; regenerated the six-file local QR incompatible artifact bundle. |

## Changed Surface Decision

| Surface | Day 5 state |
| --- | --- |
| `scripts/run_external_comparison.py` | Unchanged; no local or designed defect justified an edit. |
| `scripts/normalize_report_index.py` | Unchanged; QR freshness passed and existing separator normalization remains in place. |
| `.github/workflows/windows-ci.yml` | Unchanged; no QR promotion without hosted evidence. |
| `tests/corpus/manifests/selected_report_targets.tsv` | Unchanged; Windows QR metadata remains unpromoted. |
| Public docs | Unchanged; no claim should change before the Day 7 promotion decision. |
| C source/header files | Unchanged. |

## Promotion Impact

Day 5 strengthens confidence that the local selected generator path remains
healthy, but it does not promote Windows QR incompatible comparison freshness.
The generated artifacts still record local-only semantics and the available
probe is not hosted MSVC evidence. Promotion remains blocked until a hosted
Windows run proves the exact target and artifact set.

## Follow-Up For Day 6

Day 6 should add focused QR-specific artifact path and selected-row filtering
coverage if the current normalizer tests only prove those Windows path forms
for the Cholesky selected target. That work can reduce future Windows
promotion risk without prematurely changing the manifest or workflow.
