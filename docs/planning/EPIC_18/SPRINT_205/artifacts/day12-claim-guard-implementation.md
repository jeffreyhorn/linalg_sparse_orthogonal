# Sprint 205 Day 12: Claim Guard Implementation

**Date:** 2026-09-19  
**Sprint item:** 205.5 Claim Guard Updates  
**Theme:** Implement focused guard coverage for quick-reference,
support-truth, diagnostics-vocabulary, and retained non-claim wording.

## Purpose

Day 12 implements the Day 11 guard design. It adds executable coverage for the
Sprint 205 cross-document routing and vocabulary changes while reusing existing
specialized guards for API local-only routing and selected-performance claim
boundaries.

## Changed Guard Surfaces

| Surface | Change | Reason |
| --- | --- | --- |
| `tests/test_support_quick_reference_docs.py` | New focused Python regression suite for Sprint 205 support, quick-reference, route-interpretation, diagnostics, benchmark, API, and maintainer wording. | The cross-document quick-reference/support/diagnostics checks do not belong cleanly to package-only, API-only, Windows-only, or benchmark-only guards. |
| `Makefile` | Added standalone `support-docs-guard` target that runs the new Python suite. | Gives maintainers a stable command without changing API docs freshness semantics. |
| `scripts/check_api_docs_routing.py` | Updated README required local-only API text markers to the Sprint 205 source-controlled/local-only wording. | Keeps the existing API routing guard aligned with the simplified README wording. |
| `tests/test_api_docs_routing.py` | Narrowed one allowed-route fixture mutation to replace only the first README API-reference link. | Preserves the required `include/` route in the fixture while testing the intended blockquoted-fence behavior. |

## Guard Coverage Added

The new support docs guard checks durable markers for:

- README routing to `docs/cookbook.md#problem-shape-quick-reference`,
  `INSTALL.md#support-readiness-matrix`, problem-local residual wording,
  run-local convergence wording, and local measurement artifact wording;
- cookbook quick-reference existence, support truth link, solver-selection
  owner link, no-portable-performance boundary, and static-first/no ABI/no
  package-manager boundary;
- examples route interpretation separating local build-tree examples,
  installed consumers, support status, benchmarks, and unsupported claims;
- tutorial and solver-selection diagnostics vocabulary markers;
- benchmark selected/local measurement, skip-scope, and non-passing-evidence
  wording;
- API reference local-only/current-output wording;
- maintainer support-truth and diagnostics-vocabulary routing guidance.

## Regression Fixtures

| Regression | Expected protection |
| --- | --- |
| Remove README quick-reference route. | Fails with missing Sprint 205 marker. |
| Remove cookbook support/readiness route. | Fails with missing Sprint 205 marker. |
| Remove examples local-vs-installed interpretation. | Fails with missing Sprint 205 marker. |
| Replace tutorial run-local diagnostics vocabulary with generic wording. | Fails with missing Sprint 205 marker. |
| Add package-manager support overclaim. | Fails with unsupported Sprint 205 documentation claim. |
| Add portable-performance overclaim. | Fails with unsupported Sprint 205 documentation claim. |
| Add hosted generated API overclaim. | Fails with unsupported Sprint 205 documentation claim. |

## Validation Record

| Command | Result |
| --- | --- |
| `python3 tests/test_support_quick_reference_docs.py` | Passed |
| `make support-docs-guard` | Passed |
| `python3 tests/test_selected_performance_docs.py` | Passed |
| `python3 tests/test_api_docs_routing.py` | Passed |

## Retained Specialized Guard Ownership

Day 12 did not duplicate these existing owners:

- package-manager support and Homebrew boundaries remain owned by
  `scripts/package_manager_deferral_check.sh`;
- static-first, shared-library, dynamic ABI, and Windows package parity
  boundaries remain owned by `scripts/static_package_deferral_check.sh`;
- generated API local-only publication and route parsing remain owned by
  `scripts/check_api_docs_local_only.sh`,
  `scripts/check_api_docs_routing.py`, and their regression suites;
- selected performance boundaries remain owned by
  `tests/test_selected_performance_docs.py`;
- Windows/PowerShell selected workflow claim boundaries remain owned by
  `scripts/validate_windows_powershell.py`;
- selected target exactness remains owned by
  `tests/test_selected_report_targets_manifest.py`.

## Non-Changes

Day 12 did not change production C source, public headers, CMake install
behavior, CI workflows, selected target manifests, report schemas, generated
outputs, package metadata, API behavior, or support/readiness status.

## Completion Criteria Review

- Item 205.5 now has executable coverage.
- Simplified public docs cannot silently drop the support-truth route,
  quick-reference route, route-interpretation boundary, diagnostics scope
  vocabulary, selected/local benchmark boundary, or generated API local-only
  wording.
- Unsupported support, package, ABI, platform, performance, release, hosted
  generated API, and state-of-the-art claims remain guarded by the new focused
  suite plus existing specialized guards.
