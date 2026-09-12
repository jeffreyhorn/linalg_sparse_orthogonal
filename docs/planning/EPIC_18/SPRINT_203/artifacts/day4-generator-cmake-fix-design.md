# Day 4: Generator And CMake Fix Design

## Purpose

Day 4 reviewed the Day 3 probe output and defined the minimal fix boundary for
selected Windows QR incompatible comparison promotion. The day intentionally
did not edit generator code, workflow YAML, manifest metadata, or public docs.

## Evidence Review

| Evidence | Result | Design impact |
| --- | --- | --- |
| Direct `qr-incompatible-ls` generator | Passed locally. | No target fixture or row-generation fix is currently justified. |
| Local CMake probe | Passed locally with `--probe-build-system cmake --cmake-config Release`. | The generated temporary CMake project is locally viable. |
| Required helper dependency | Passed for `tests/qr_external_dense_reference.py`. | Preserve this as required dependency evidence. |
| Optional NumPy/SciPy baselines | Deferred. | Continue treating optional package rows as non-evidence. |
| Target-specific freshness | Passed for six selected QR incompatible rows. | Local freshness works; Windows promotion still needs hosted MSVC evidence. |
| Generated metadata | `support_tier=local_only`, `platform=darwin-x86_64`. | Blocks manifest promotion. |

## Minimal Fix Boundary

The Day 4 design does not call for speculative code changes. Fixes should be
made only when a hosted Windows/MSVC run or focused simulation identifies a
repo-owned issue:

| Trigger | Smallest owner surface |
| --- | --- |
| CMake configure fails from invalid path literals, include paths, or library path syntax. | `scripts/run_external_comparison.py` CMake probe rendering plus runner tests. |
| CMake build succeeds but executable discovery fails. | Probe binary candidate logic plus runner tests. |
| Required helper failure is reported as project failure. | Dependency-status classification plus runner tests. |
| Windows artifact paths are generated but selected freshness drops rows. | `scripts/normalize_report_index.py` artifact matching plus QR-specific path tests. |
| Hosted evidence passes but metadata remains local-only. | Manifest/docs promotion decision, not an isolated manifest edit. |
| Hosted evidence is unavailable. | Re-deferral or blocked evidence record. |

## Preservation Checks

| Surface | Required preservation |
| --- | --- |
| Linux selected comparison workflow | Existing selected comparison artifacts and upload semantics remain intact. |
| macOS selected comparison workflow | Existing selected comparison artifacts and upload semantics remain intact. |
| Windows Cholesky selected workflow | Existing `cholesky-spd-tridiag-5` target, artifact name, required files, and fail-closed upload remain intact. |
| QR incompatible local output | Six expected rows, expected nonzero residual, and fixture-local non-claims remain intact. |
| Selected manifest | Windows metadata remains absent unless exact hosted QR evidence supports promotion. |

## Test Mapping

| Change category | Validation |
| --- | --- |
| Runner/CMake probe fix | `python3 tests/test_run_external_comparison.py` plus focused new fixture. |
| Normalizer path fix | `python3 tests/test_normalize_report_index.py` plus QR Windows artifact-path fixture. |
| Workflow edit | `python3 tests/test_selected_comparison_workflow.py` and `make windows-powershell-guard`. |
| Manifest edit | `python3 tests/test_selected_report_targets_manifest.py`. |
| Documentation edit | `make docs-check` or the specific docs guard changed by the edit. |
| C/header edit | `make format && make lint && make test`. |

## Day 4 Decision

No generator or CMake code should be changed solely from Day 3 local evidence.
Day 5 should either use hosted Windows evidence to drive a targeted fix or
record that promotion remains blocked pending hosted MSVC execution. This
keeps Sprint 203 from broadening selected comparison support before evidence
exists.
