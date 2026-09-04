# Sprint 196 Day 11 Artifact: Integrated Focused Validation

**Date:** 2026-09-03
**Sprint item coverage:** 196.4, focused-validation portion
**Day 11 goal:** Run focused gates across Epic 17 evidence owners and record
results, reruns, skipped/environment-limited surfaces, and residual context.

## Summary

Day 11 ran focused validation across the Epic 17 evidence-owner surfaces:
package/install, Windows PowerShell ownership, selected report metadata,
selected comparison and oracle freshness, selected benchmark freshness,
review-surface guards, symbolic allocation-failure proof, corpus schema, and
docs/API freshness.

All focused gates passed. Three Make gates initially failed because the
generated version header was missing from the current build state while gates
were started in parallel. After normal generated-header creation established
`build/include/sparse_version.h`, the affected gates passed on rerun. No code
or documentation edits were needed for that precondition failure.

## Focused Validation Log

| Gate | Result | Evidence boundary |
| --- | --- | --- |
| `bash scripts/package_manager_deferral_check.sh` | Passed | Package-manager/Homebrew support non-claims and selected proof boundary. |
| `bash scripts/static_package_deferral_check.sh` | Passed | Static-first package and shared-library/dynamic ABI non-claims. |
| `bash tests/test_install.sh` | Passed: 23 checks, 0 failures | Make install and `pkg-config` source-install proof only. |
| `bash tests/test_cmake_install.sh` | Passed: 27 checks, 0 failures, 0 skips | CMake install/export source-install proof only. |
| `make windows-powershell-guard` | Passed | Windows workflow ownership and claim-boundary structure; local `pwsh` absence remains residual. |
| `python3 scripts/validate_corpus_schema.py` | Passed | Corpus schema and manifest consistency. |
| `python3 tests/test_selected_report_targets_manifest.py` | Passed | Selected report target metadata. |
| `python3 tests/test_selected_comparison_workflow.py` | Passed | Selected comparison workflow ownership. |
| `python3 tests/test_normalize_report_index.py` | Passed | Report-index normalizer behavior. |
| `python3 tests/test_run_external_comparison.py` | Passed | External comparison runner behavior. |
| `make report-index-comparison-freshness` | Passed: 46 rows | Local-only generated selected comparison freshness. |
| `make report-index-oracle-freshness` | Passed: 54 rows | Local-only generated selected oracle freshness. |
| `python3 tests/test_selected_performance_docs.py` | Passed | Selected-performance non-claims and docs markers. |
| `python3 tests/test_bench_canonical_freshness.py` | Passed | Selected benchmark freshness checker behavior. |
| `make bench-canonical-report-freshness` | Passed on rerun | One selected threshold-free benchmark evidence lane. |
| `python3 tests/test_qr_external_ref_helper_guard.py` | Passed | QR external-reference helper guard test coverage. |
| `make qr-external-ref-helper-guard` | Passed | QR external-reference helper ownership. |
| `make qr-header-docs-guard` | Passed | QR header/docs coherence. |
| `make ldlt-csc-helper-guard` | Passed | LDLT CSC helper ownership. |
| `python3 tests/test_symbolic_allocation_failure_gate_registration.py` | Passed | Symbolic allocation-failure gate registration. |
| `make symbolic-allocation-failure-gate` | Passed on rerun: 101 tests, 0 failures, 0 skips, 1262 assertions | Selected `sparse_symbolic_cholesky()` allocation-failure owner. |
| `make api-docs-freshness` | Passed | Local-only generated API docs freshness and staging guard. |
| `git diff --check` | Passed | Changed-file whitespace hygiene. |

## Rerun Notes

The first parallel run of `make bench-canonical-report-freshness`,
`make report-index-comparison-freshness`, and
`make symbolic-allocation-failure-gate` failed immediately while compiling
against `include/sparse_types.h` because `build/include/sparse_version.h` was
not yet present. After another build path generated the header, all three
commands passed on rerun.

This is recorded as a local validation sequencing note, not a Sprint
196-caused code defect.

## Environment Residuals

| Surface | Residual |
| --- | --- |
| Local PowerShell | `pwsh` is unavailable locally; `make windows-powershell-guard` preserves that as unavailable evidence rather than pass evidence. |
| Hosted Windows evidence | Hosted Windows selected freshness promotion remains outside this local focused validation pass. |
| Generated outputs | `docs/api/html/`, `build/bench-reports/`, `build/comparison/`, and `build/corpus*` outputs are generated local artifacts and remain ignored evidence surfaces unless future work promotes them. |

## 196.4 Acceptance Evidence

| Completion criterion | Evidence |
| --- | --- |
| Focused Epic 17 evidence owners pass or have exact residual context. | All focused gates passed; local `pwsh` and hosted Windows promotion remain explicitly residual. |
| Claim-boundary and source ownership checks are synchronized. | Package, Windows, selected performance, selected report, API, QR, LDLT, and reliability guards passed. |
| No unclear validation failure is hidden or converted into a pass. | Generated-header precondition failures were recorded and affected commands passed on rerun. |

## Validation

Day 11 is itself the focused validation artifact. No `.c` or `.h` files were
modified, so the full C quality gate is not required for this day. Day 12 owns
the final full-quality decision for Sprint 196 closeout.
