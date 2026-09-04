# Sprint 196 Day 3 Artifact: Residual Triage

**Date:** 2026-09-03
**Sprint item coverage:** 196.6, with support for 196.1 and 196.2
**Day 3 goal:** Consolidate Epic 17 residuals and rank them by closure value,
evidence requirement, risk, and next-epic fit.

## Summary

Day 3 deduplicated Sprint 187-195 residuals into a retained queue with exact
owner conditions and closure evidence. The queue distinguishes next-epic
candidates from long-horizon deferrals, documentation-only follow-ups,
validation/tooling follow-ups, and out-of-scope historical notes.

The most important planning result is that Sprint 196 should not treat every
residual as a near-term implementation candidate. Several residuals require
product decisions, hosted/platform evidence, or broad methodology work before
they are actionable.

## Classification Counts

| Classification | Count | Queue IDs |
| --- | ---: | --- |
| Next-epic candidate | 6 | E17-RQ-001, E17-RQ-005, E17-RQ-006, E17-RQ-013, E17-RQ-016, E17-RQ-022 |
| Validation/tooling follow-up | 6 | E17-RQ-004, E17-RQ-009, E17-RQ-010, E17-RQ-017, E17-RQ-020, E17-RQ-024 |
| Documentation-only follow-up | 3 | E17-RQ-014, E17-RQ-018, E17-RQ-021 |
| Long-horizon deferral | 10 | E17-RQ-002, E17-RQ-003, E17-RQ-007, E17-RQ-008, E17-RQ-011, E17-RQ-012, E17-RQ-015, E17-RQ-023, E17-RQ-025, E17-RQ-026 |
| Out-of-scope historical note | 1 | E17-RQ-019 |

## Consolidated Queue

| Queue ID | Residual | Classification | Priority | Owner condition | Evidence required to close |
| --- | --- | --- | --- | --- | --- |
| E17-RQ-001 | Package-manager/Homebrew support remains unclaimed because approved standalone root license metadata and exact Homebrew license identifier are missing. | Next-epic candidate | High | Future package-provider/product owner | Add approved root license metadata, set exact Homebrew license identifier, rerun local proof to exit `0`, rerun package/static guards, and update docs only to the proof level earned. |
| E17-RQ-002 | Shared-library packaging and dynamic ABI support remain deferred. | Long-horizon deferral | High | Future ABI/package owner | Add shared-library build/install artifacts, ABI and loader policies, package metadata, static/shared selectors, and consumer tests. |
| E17-RQ-003 | Broad Windows parity remains unclaimed beyond selected hosted/CMake evidence. | Long-horizon deferral | High | Future Windows platform owner | Add reviewed Windows evidence for each desired surface with workflow owners, artifacts, tests, and docs. |
| E17-RQ-004 | Local PowerShell execution is unavailable on this machine. | Validation/tooling follow-up | Medium | Local developer environment or hosted CI owner | Install `pwsh` and rerun `make windows-powershell-validate` for exit `0`, or keep local exit `2` separate from hosted evidence. |
| E17-RQ-005 | Selected Cholesky Windows freshness promotion depends on hosted evidence review and manifest metadata promotion. | Next-epic candidate | High | Selected report target manifest/Windows CI owner | Observe hosted pass, inspect exact Cholesky bundle, promote only the selected Cholesky row to Windows metadata, and rerun coupled guards. |
| E17-RQ-006 | Windows selected QR incompatible freshness remains deferred. | Next-epic candidate | Medium | Hosted Windows comparison owner | Add MSVC/CMake proof, inspect artifacts, update exact selected metadata, and retain broad QR parity non-claims. |
| E17-RQ-007 | Optional NumPy/SciPy package baselines remain deferred advisory rows. | Long-horizon deferral | Low | External comparison baseline owner | Select package-backed baselines, define dependency policy, add availability checks, and avoid package-manager support implications. |
| E17-RQ-008 | Broader QR least-squares or external-library parity remains unproved. | Long-horizon deferral | Medium | Future comparison-family owner | Add bounded fixtures one at a time with exact references, metrics, tolerances, row IDs, freshness diagnostics, and claim calibration. |
| E17-RQ-009 | Generated local comparison artifacts must be regenerated or reviewed from CI uploads before use as evidence. | Validation/tooling follow-up | Medium | Reviewer/CI evidence owner | Regenerate ignored local comparison artifacts or inspect uploaded CI artifacts before citing rows. |
| E17-RQ-010 | Selected comparison review volume may grow as target families accumulate. | Validation/tooling follow-up | Low | Test/report infrastructure owner | Extract shared constants only if row identity and diagnostics remain explicit. |
| E17-RQ-011 | Hosted timing thresholds are not defined. | Long-horizon deferral | Medium | Performance-governance owner | Add baseline, variance model, machine policy, flake budget, and failure wording. |
| E17-RQ-012 | Portable performance evidence remains unclaimed. | Long-horizon deferral | High | Benchmark methodology owner | Add multi-platform, multi-machine, repeated, variance-aware evidence with environment context. |
| E17-RQ-013 | Windows/macOS selected benchmark freshness is not owned. | Next-epic candidate | Medium | Platform CI owner | Add hosted platform lanes and selected artifact validation without broadening Linux selected claims. |
| E17-RQ-014 | Unselected canonical CSV publication remains local-only. | Documentation-only follow-up | Low | Benchmark publication owner | Select, document, guard, and publish each promoted row before using it as review evidence. |
| E17-RQ-015 | Release benchmark claims remain undefined. | Long-horizon deferral | Medium | Release engineering owner | Define release fixtures, reproducible environments, archived artifacts, and acceptance criteria. |
| E17-RQ-016 | Additional QR review-surface clusters remain in `tests/test_qr.c`. | Next-epic candidate | Medium | Future QR review-surface owner | Select one cluster, define helper boundary, preserve registration, add guard coverage, and rerun focused/full validation. |
| E17-RQ-017 | Header-only focused rebuild caveat remains for QR helper edits. | Validation/tooling follow-up | Medium | Test/build owner | Add dependency tracking for helper headers or preserve forced rebuild guidance. |
| E17-RQ-018 | Large helper size may still add review burden. | Documentation-only follow-up | Low | Test-structure owner | Split helpers only if review burden drops without source-list or proof-owner ambiguity. |
| E17-RQ-019 | Existing unrelated warning hygiene remains outside Sprint 193 closure. | Out-of-scope historical note | Low | Future warning-hygiene owner | Reproduce under current gates before planning any fix. |
| E17-RQ-020 | No dedicated Markdown link-check target exists. | Validation/tooling follow-up | Medium | Documentation tooling owner | Add a link-check target, fixtures, failure semantics, and exclusions. |
| E17-RQ-021 | Public headers still contain declaration-adjacent detailed API contracts. | Documentation-only follow-up | Low | API documentation owner | Move only broad workflow narrative when generated API coverage and docs routing stay valid. |
| E17-RQ-022 | Additional allocation-failure owners remain unproved. | Next-epic candidate | High | Future symbolic/analysis/solver/matrix reliability owners | Select one owner, record invariants, add deterministic failure/retry tests, focused gate, docs, and full validation. |
| E17-RQ-023 | OS OOM and concurrent allocation-hook behavior remain unclaimed. | Long-horizon deferral | Medium | Allocator/platform owner | Define allocator policy, concurrency semantics, platform evidence, and stress/sanitizer validation. |
| E17-RQ-024 | No hosted CI lane owns the symbolic allocation-failure gate. | Validation/tooling follow-up | Medium | Future CI owner | Add a reviewed hosted lane or keep the gate local-only in support/readiness wording. |
| E17-RQ-025 | Hosted generated API publication remains unselected. | Long-horizon deferral | Low | Product/docs publication owner | Define hosted publication, freshness, retention, deployment, versioning, and claim ownership. |
| E17-RQ-026 | Unqualified state-of-the-art sparse linear algebra status remains unearned. | Long-horizon deferral | High | Product/research/performance owner | Close broad algorithmic, ecosystem, performance, portability, packaging, reliability, and documentation evidence gaps first. |

## Recommended Next-Epic Candidates

1. E17-RQ-001: close the package-manager/Homebrew proof blocker only after an
   approved license decision exists.
2. E17-RQ-005: finish or formally retain the already narrowed selected
   Cholesky Windows freshness promotion.
3. E17-RQ-022: select one additional allocation-failure reliability owner and
   repeat the Sprint 195 proof pattern.
4. E17-RQ-016: extract one more QR review-surface cluster only if the boundary
   is behavior-preserving and guardable.
5. E17-RQ-013: add one selected benchmark platform freshness lane if hosted CI
   time and artifact semantics are reviewed.
6. E17-RQ-006: promote Windows QR incompatible freshness only after MSVC/CMake
   proof exists.

## Residuals Not Recommended For Near-Term Implementation

Broad Windows parity, shared-library/dynamic ABI, portable performance,
state-of-the-art status, hosted timing thresholds, release benchmark claims,
and OS OOM/concurrency semantics should stay long-horizon until a future epic
can allocate enough design, product, platform, and validation scope.

Optional package baselines, unselected CSV publication, helper-size cleanup,
and hosted generated API publication are lower value than the selected closure
targets above unless adjacent work is already modifying those surfaces.

## Claim Calibration Implications

- Package wording must keep package-manager support unclaimed.
- Windows wording must separate local PowerShell availability, hosted
  PowerShell validation, selected Cholesky freshness, and broad Windows parity.
- Comparison wording must stay selected-target and fixture scoped.
- Benchmark wording must stay methodology-bound and threshold-free.
- Reliability wording must stay selected-owner scoped.
- Epic 17 closeout wording must explicitly state that the project improved
  evidence quality but did not earn unqualified state-of-the-art status.

## Validation

- `sed -n '105,155p' docs/planning/EPIC_17/SPRINT_196/PLAN.md`
- `sed -n '127,240p' docs/planning/EPIC_17/SPRINT_196/WORKING_NOTES.md`
- Reviewed Sprint 187-195 retrospective residual sections.
- Searched Sprint 187-195 working notes and Day 14 closeout artifacts for
  residual, deferred, pending, non-claim, and future-work terms.

Day 3 changed planning documentation only. No `.c` or `.h` files were modified,
so the full C quality gate is not required for this day.
