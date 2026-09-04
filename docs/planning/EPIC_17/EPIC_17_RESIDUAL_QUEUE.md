# Epic 17 Residual Queue

## Purpose

This file is the next-epic handoff for Epic 17 residuals. It turns Sprint 196
Day 3 residual triage and Day 9 retrospective drafting into prioritized,
deduplicated work items with exact closure targets, owner surfaces, expected
evidence, validation commands, and claim boundaries.

The queue is intentionally selective. Near-term priorities are the residuals
that can close a concrete gap with bounded evidence. Long-horizon deferrals
remain visible, but they should not be treated as ready implementation work
until a future epic allocates product, platform, methodology, or research
scope.

## Queue Summary

| Priority | Residual ID | Theme | Deferral horizon |
| ---: | --- | --- | --- |
| 1 | E17-RQ-001 | Package-manager/Homebrew support blocker | Near-term product/legal metadata decision |
| 2 | E17-RQ-005 | Selected Cholesky Windows freshness promotion | Hosted evidence and manifest promotion decision |
| 3 | E17-RQ-022 | Additional allocation-failure owner | Near-term selected reliability proof |
| 4 | E17-RQ-016 | Additional QR review-surface cluster | Incremental maintainability |
| 5 | E17-RQ-013 | Windows/macOS selected benchmark freshness | Hosted platform evidence |
| 6 | E17-RQ-006 | Windows QR incompatible freshness | Hosted Windows comparison evidence |

## Priority 1: E17-RQ-001

| Field | Value |
| --- | --- |
| Theme | Package-manager/Homebrew support blocker. |
| Source | Sprint 188; Sprint 196 Day 3 residual triage. |
| Current status | Residualized; support remains unclaimed. |
| Owner surfaces | Root license metadata; `packaging/homebrew/sparse-lu-ortho.rb.in`; `packaging/homebrew/README.md`; `scripts/homebrew_local_formula_proof.sh`; README; INSTALL; package/static guards; install tests. |
| Why it remains | Approved standalone root license metadata and exact Homebrew formula license identifier are missing. The proof path must not invent legal metadata. |
| Closure target | Add approved root license metadata, set the exact Homebrew license identifier, and promote documentation only to the support level earned by proof output. |
| Expected evidence | Local formula proof exits `0`; formula renders; source archive and checksum are created; install succeeds; installed static files are checked; `brew test` passes; uninstall succeeds; proof outputs are cleaned. |
| Validation commands | `bash scripts/homebrew_local_formula_proof.sh`; `bash scripts/package_manager_deferral_check.sh`; `bash scripts/static_package_deferral_check.sh`; `bash tests/test_install.sh`; `bash tests/test_cmake_install.sh`; `make docs-check`; full C gate only if `.c` or `.h` files change. |
| Claim boundary | Until closed, Homebrew remains proof material only. Do not claim Homebrew support, Homebrew/core readiness, bottles, Linuxbrew support, public tap support, package-manager support, or package-manager distribution. |

## Priority 2: E17-RQ-005

| Field | Value |
| --- | --- |
| Theme | Selected Cholesky Windows freshness promotion. |
| Source | Sprint 190; Sprint 196 Day 3 residual triage; Sprint 196 Day 5-6 claim recalibration. |
| Current status | Residual narrowed; guarded workflow path exists. |
| Owner surfaces | `.github/workflows/windows-ci.yml`; selected report target manifest; comparison runner; report normalizer; corpus README; maintainer guide; README; INSTALL. |
| Why it remains | Sprint 190 added one bounded selected Cholesky Windows workflow path, but hosted evidence review and selected manifest metadata promotion were still required before treating the row as promoted Windows selected freshness. |
| Closure target | Observe hosted Windows pass, inspect the exact selected Cholesky bundle, promote only the selected Cholesky manifest metadata if evidence supports it, and recalibrate docs together. |
| Expected evidence | Hosted `windows-2022` selected comparison job passes; artifact contains exactly the selected Cholesky comparison bundle; target `cholesky-spd-tridiag-5` emits the expected rows; manifest support tier and workflow platform metadata match the promoted claim. |
| Validation commands | `python3 tests/test_selected_report_targets_manifest.py`; `python3 tests/test_selected_comparison_workflow.py`; `python3 tests/test_normalize_report_index.py`; `python3 tests/test_run_external_comparison.py`; `python3 tests/test_validate_windows_powershell.py`; `make windows-powershell-guard`; target-specific selected comparison freshness command; hosted Windows workflow evidence review. |
| Claim boundary | Until hosted evidence and manifest promotion are reviewed together, claim only one guarded Windows selected Cholesky workflow path. Do not claim broad Windows report freshness, Windows selected oracle freshness, Windows selected benchmark freshness, broad selected comparison freshness, or broad Windows generated-report parity. |

## Priority 3: E17-RQ-022

| Field | Value |
| --- | --- |
| Theme | Additional allocation-failure owner. |
| Source | Sprint 195; Sprint 196 Day 3 residual triage. |
| Current status | Next-epic candidate. |
| Owner surfaces | Candidate symbolic, analysis, etree, direct-solver, or matrix-construction owner; deterministic allocation harness; focused tests; Make/CTest labels; README; INSTALL; maintainer guide. |
| Why it remains | Sprint 195 proved selected `sparse_symbolic_cholesky()` output allocation only. Other allocation-heavy owners remain outside that proof. |
| Closure target | Select exactly one additional owner and repeat the Sprint 195 pattern: candidate scoring, invariant record, deterministic failure/retry tests, focused gate, docs, and full validation when C changes. |
| Expected evidence | Failed allocation returns the expected status; partial state is cleaned; stale outputs are suppressed; caller-owned inputs are preserved; retry after reset succeeds; focused gate and registration guard prevent drift. |
| Validation commands | New focused owner gate; new registration guard if applicable; relevant CTest label; focused owner binary; `make source-list-check`; `make format && make lint && make test` if `.c` or `.h` files change; `make docs-check`. |
| Claim boundary | Claim only the selected owner and selected allocation path. Do not claim broad allocation-failure coverage, OS OOM behavior, concurrent allocation-hook behavior, generated-tooling reliability, or package/install reliability. |

## Priority 4: E17-RQ-016

| Field | Value |
| --- | --- |
| Theme | Additional QR review-surface cluster. |
| Source | Sprint 193; Sprint 196 Day 3 residual triage. |
| Current status | Next-epic candidate. |
| Owner surfaces | `tests/test_qr.c`; QR helper headers; helper guard scripts/tests; maintainer guide; source-list and CMake test registration. |
| Why it remains | Sprint 193 extracted one selected rank/nullspace/threshold cluster only. Other QR test clusters remain large and review-heavy. |
| Closure target | Select one additional QR cluster, record no-behavior-change invariants, extract helper ownership, add guard coverage, and preserve test registration. |
| Expected evidence | Candidate ranking; selected-cluster rationale; behavior-preservation notes; helper extraction; guard coverage; focused QR tests; source-list or dependency evidence if touched. |
| Validation commands | `make qr-external-ref-helper-guard` or a new cluster-specific guard; `python3 tests/test_qr_external_ref_helper_guard.py` or new guard test; focused QR test binary; `make source-list-check`; `make format && make lint && make test` if `.c` or `.h` files change. |
| Claim boundary | Do not claim broad QR behavior changes, solver correctness expansion, performance improvement, public API changes, or broad review-surface cleanup from one extraction. |

## Priority 5: E17-RQ-013

| Field | Value |
| --- | --- |
| Theme | Windows/macOS selected benchmark freshness. |
| Source | Sprint 192; Sprint 196 Day 3 residual triage. |
| Current status | Next-epic candidate. |
| Owner surfaces | Benchmark workflow YAML; selected target manifest; `scripts/check_bench_canonical_freshness.py`; report normalizer; benchmark docs; maintainer guide. |
| Why it remains | Sprint 192 hardened one selected Linux hosted performance lane. Windows and macOS selected benchmark freshness are not owned. |
| Closure target | Add one hosted platform selected benchmark freshness lane with exact artifact scope and selected row validation, without broadening Linux-selected claims. |
| Expected evidence | Hosted platform pass; exact selected benchmark bundle; selected CSV content matches manifest contract; methodology metadata includes platform/compiler context; docs preserve non-portable and threshold-free wording. |
| Validation commands | `make bench-canonical-report-freshness`; `python3 tests/test_selected_performance_docs.py`; `python3 tests/test_bench_canonical_freshness.py`; `python3 tests/test_selected_report_targets_manifest.py`; report normalizer tests; hosted platform workflow evidence review. |
| Claim boundary | Do not claim portable performance, timing thresholds, backend superiority, platform parity, release benchmark readiness, or state-of-the-art performance. |

## Priority 6: E17-RQ-006

| Field | Value |
| --- | --- |
| Theme | Windows QR incompatible freshness. |
| Source | Sprint 191; Sprint 196 Day 3 residual triage. |
| Current status | Next-epic candidate. |
| Owner surfaces | Windows workflow; QR incompatible comparison target; comparison runner; selected manifest; normalizer; corpus docs; maintainer guide. |
| Why it remains | Sprint 191 added the QR incompatible comparison family as local-only because no MSVC/CMake proof existed. |
| Closure target | Add MSVC/CMake proof for `qr-incompatible-ls`, inspect hosted artifacts, promote exact selected metadata if evidence supports it, and retain broad QR parity non-claims. |
| Expected evidence | Windows CMake probe builds and runs; generated rows match expected QR incompatible target output; artifact paths are normalized; selected manifest metadata matches the promoted platform scope. |
| Validation commands | `python3 tests/test_run_external_comparison.py`; `python3 tests/test_normalize_report_index.py`; `python3 tests/test_selected_report_targets_manifest.py`; `python3 tests/test_selected_comparison_workflow.py`; focused QR solve tests; hosted Windows comparison workflow evidence review. |
| Claim boundary | Do not claim broad QR least-squares parity, broad external-library parity, Windows selected oracle freshness, Windows benchmark freshness, or broad Windows report freshness. |

## Validation and Tooling Follow-Ups

| Residual ID | Theme | Closure target |
| --- | --- | --- |
| E17-RQ-004 | Local PowerShell unavailable | Install `pwsh` locally or keep hosted validation as the owner; do not treat local exit `2` as a pass. |
| E17-RQ-009 | Generated local comparison artifacts | Regenerate ignored local artifacts or inspect uploaded CI artifacts before citing rows as evidence. |
| E17-RQ-010 | Selected comparison review volume | Extract shared constants only when row identity and diagnostics remain explicit. |
| E17-RQ-017 | Header-only focused rebuild caveat | Add dependency tracking for helper headers or preserve forced rebuild guidance. |
| E17-RQ-020 | Markdown link-check target | Add a dedicated Markdown link-check target with fixtures, failure semantics, and exclusions. |
| E17-RQ-024 | Hosted symbolic allocation-failure gate | Add a reviewed hosted lane or keep the gate local-only in support/readiness wording. |

## Documentation-Only Follow-Ups

| Residual ID | Theme | Closure target |
| --- | --- | --- |
| E17-RQ-014 | Unselected canonical CSV publication | Select, document, guard, and publish each promoted row before using it as review evidence. |
| E17-RQ-018 | Large helper size | Split helpers only if review burden drops without source-list or proof-owner ambiguity. |
| E17-RQ-021 | Remaining declaration-adjacent public-header contracts | Move only broad workflow narrative when generated API coverage and docs routing stay valid. |

## Long-Horizon Deferrals

These residuals are intentionally outside the near-term priority queue unless
a future epic explicitly allocates broad design and validation scope:

- E17-RQ-002: shared-library packaging and dynamic ABI support;
- E17-RQ-003: broad Windows parity;
- E17-RQ-007: optional NumPy/SciPy package baselines;
- E17-RQ-008: broader QR least-squares or external-library parity;
- E17-RQ-011: hosted timing thresholds;
- E17-RQ-012: portable performance evidence;
- E17-RQ-015: release benchmark claims;
- E17-RQ-023: OS OOM and concurrent allocation-hook behavior;
- E17-RQ-025: hosted generated API publication;
- E17-RQ-026: unqualified state-of-the-art sparse linear algebra status.

## Out-Of-Scope Historical Note

| Residual ID | Theme | Closure target |
| --- | --- | --- |
| E17-RQ-019 | Existing unrelated warning hygiene | Reproduce under current gates before planning any fix. |

## Source Evidence

- [PROJECT_PLAN.md](./PROJECT_PLAN.md)
- [EPIC_17_RETROSPECTIVE.md](./EPIC_17_RETROSPECTIVE.md)
- [SPRINT_196/artifacts/day2-outcome-ledger.md](./SPRINT_196/artifacts/day2-outcome-ledger.md)
- [SPRINT_196/artifacts/day3-residual-triage.md](./SPRINT_196/artifacts/day3-residual-triage.md)
- [SPRINT_196/artifacts/day7-project-plan-status.md](./SPRINT_196/artifacts/day7-project-plan-status.md)
- [SPRINT_196/artifacts/day9-epic-retrospective-draft.md](./SPRINT_196/artifacts/day9-epic-retrospective-draft.md)
