# Sprint 205 Day 11: Claim Guard Design

**Date:** 2026-09-19  
**Sprint item:** 205.5 Claim Guard Updates  
**Theme:** Design guard updates that preserve simplified support wording,
quick-reference routing, and diagnostics vocabulary.

## Purpose

Day 11 inventories existing claim/documentation guards and defines the Day 12
implementation plan. This artifact is design-only: it does not change guard
behavior, test execution, source code, headers, workflows, manifests, or
generated outputs.

## Existing Guard Inventory

| Guard or test | Current owner surface | Reuse for Sprint 205 |
| --- | --- | --- |
| `scripts/package_manager_deferral_check.sh` | Package-manager and Homebrew non-claims across README, INSTALL, maintainer guide, Homebrew proof docs, and metadata templates. | Extend/verify support-truth wording still rejects broad package-manager support after README simplification. |
| `scripts/static_package_deferral_check.sh` | Static-first package posture, shared-library and dynamic ABI deferrals, Windows Makefile/`pkg-config` non-claims. | Extend/verify support-truth wording keeps static-first and ABI boundaries discoverable. |
| `tests/test_selected_performance_docs.py` | Selected benchmark/performance claim boundaries and hosted selected lane wording. | Extend with Sprint 205 public wording markers for local measurement artifacts and no portable performance proof. |
| `scripts/check_api_docs_local_only.sh` and `tests/test_api_docs_local_only_guard.py` | Generated API HTML local-only/staging/publication boundary. | Reuse for generated API local-only policy; add only if Day 12 finds Sprint 205 wording no longer covered. |
| `scripts/check_api_docs_routing.py` and `tests/test_api_docs_routing.py` | Source-controlled API route, generated API publication rejection, API docs Makefile routing. | Candidate owner for source-controlled API route and no hosted generated API publication markers. |
| `scripts/validate_windows_powershell.py` | Windows/PowerShell claim-boundary markers and workflow ownership. | Reuse for Windows selected freshness re-deferral; avoid duplicating Windows markers in a new guard. |
| `tests/test_selected_report_targets_manifest.py` | Selected target manifest contracts and non-claim exactness. | Keep as selected-target authority; Sprint 205 should not move quick-reference routing into manifest tests. |
| `Makefile` docs/API targets | Executable wiring for docs and API validation. | Day 12 should wire any new focused docs guard only if it needs a first-class target. |

## Guard-To-Wording Traceability

| Sprint 205 wording or route | Risk guarded | Proposed owner | Day 12 design |
| --- | --- | --- | --- |
| `INSTALL.md#support-readiness-matrix` remains public support truth. | README/cookbook/examples become support ledgers or lose support route. | New focused Python docs guard or extension to package/static guards. | Add a small `tests/test_support_quick_reference_docs.py` style guard if no existing guard cleanly owns cross-doc routing. |
| `docs/cookbook.md#problem-shape-quick-reference` remains the compact user route. | Route disappears, duplicates solver-selection detail, or omits support/benchmark/API boundaries. | New focused docs guard. | Check anchor exists, table has expected high-risk rows, and owner links point to solver selection, examples, benchmarks, API reference, and INSTALL. |
| Examples route installed consumers to INSTALL support matrix. | Examples imply package support or installed proof beyond static-first CMake consumer. | New focused docs guard plus static package guard for ABI/package non-claims. | Check `examples/README.md` keeps route-interpretation wording and support matrix link. |
| Public diagnostics use problem-local/run-local/QR-local/SVD-local vocabulary. | Public docs drift back to broad "status" or "freshness" schema vocabulary. | New focused docs guard. | Check representative markers in README, tutorial, cookbook, solver selection, examples, benchmark docs, API reference, and maintainer guide. |
| Benchmark docs state local measurement artifacts and selected target only. | Selected performance wording becomes portable performance proof or timing gate. | `tests/test_selected_performance_docs.py`. | Add markers for "local measurement artifacts" and "optional data or prerequisites" skip wording if missing. |
| Generated API docs remain local-only ignored output. | API reference simplification implies hosted generated docs or retained artifacts. | Existing API docs routing/local-only guards. | Prefer adding marker expectations to existing API docs tests instead of a new guard. |
| Package-manager/shared-library/dynamic ABI non-claims remain intact. | Simplified README install wording weakens package/ABI boundaries. | Existing package/static shell guards. | Update grep markers only if current simplified wording causes stale-marker failures. |
| Windows selected freshness remains unpromoted. | Quick-reference or report wording implies broad Windows selected freshness. | Existing Windows validator and selected manifest tests. | Do not add duplicate Windows logic unless current marker coverage misses Sprint 205 wording. |

## Failure Message Design

Day 12 guard failures should name the missing boundary, not incidental
formatting. Preferred failure messages:

- `support quick reference missing INSTALL support/readiness route`
- `support quick reference missing problem-shape quick-reference anchor`
- `examples route interpretation no longer separates local examples from installed consumers`
- `diagnostics vocabulary marker missing: problem-local residual`
- `diagnostics vocabulary marker missing: run-local convergence fields`
- `diagnostics vocabulary marker missing: QR-local or SVD-local scope`
- `benchmark docs no longer describe selected rows as local measurement artifacts`
- `benchmark docs no longer define skip as unavailable optional data/prerequisite`
- `generated API docs no longer state local-only generated output`
- `unsupported broad package, ABI, platform, performance, release, or state-of-the-art claim`

## Regression Fixture Design

| Fixture mutation | Expected failure |
| --- | --- |
| Remove `INSTALL.md#support-readiness-matrix` from cookbook quick reference intro. | Missing support/readiness route. |
| Remove `docs/cookbook.md#problem-shape-quick-reference` link from README or tutorial route table. | Missing quick-reference route. |
| Replace examples route interpretation with a package-manager support sentence. | Unsupported package/support claim or missing local-vs-installed separation. |
| Replace "problem-local residual" with generic "residual proves correctness". | Missing diagnostics marker or forbidden broad correctness wording. |
| Remove `QR-local` / `SVD-local` wording from solver selection or tutorial diagnostics. | Missing diagnostics scope marker. |
| Append "selected performance proves portable performance" to README or benchmark docs. | Existing selected-performance forbidden-claim failure. |
| Remove local-only generated API marker from API reference. | Existing API docs routing/local-only failure. |
| Replace `skip` explanation with "skips pass". | Benchmark docs skip-scope failure. |

## Day 12 Implementation Plan

1. Prefer one new focused Python regression suite for Sprint 205 public routing
   and diagnostics vocabulary, because the cross-document quick-reference
   checks do not fit cleanly into package, API, Windows, or benchmark-only
   guards.
2. Reuse existing selected-performance/API/package/static/Windows guards for
   their specialized boundaries. Update existing marker expectations only when
   Sprint 205 wording changed the intended phrase.
3. Add the new guard to a Makefile docs/check target only if it can run without
   generated outputs or external tools. Otherwise record the exact standalone
   invocation for Day 13.
4. Keep tests text-marker based but semantic: check durable phrases and owner
   links, not table ordering or line wrapping.
5. Add standalone runner calls for every new regression test so direct
   `python3 tests/<guard>.py` execution covers the full suite.

## Non-Goals

Day 12 should not:

- parse every Markdown table cell;
- duplicate selected target manifest contract tests;
- invent new support statuses;
- change public headers, C source, Makefile build semantics, CMake install
  behavior, CI workflows, manifests, schemas, or generated outputs unless a
  guard target must be wired;
- claim package-manager, shared-library, dynamic ABI, broad Windows, hosted
  generated API, portable performance, release, external-library parity, or
  state-of-the-art support.

## Completion Criteria Review

- Item 205.5 has an implementation-ready guard design.
- Guard coverage targets simplified wording and owner routes rather than
  incidental formatting.
- Unsupported support, package, ABI, platform, performance, release, and
  state-of-the-art claims remain assigned to existing or planned guards.
