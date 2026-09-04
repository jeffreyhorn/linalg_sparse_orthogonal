# Sprint 196 Day 4 Artifact: Claim Surface Audit

**Date:** 2026-09-03
**Sprint item coverage:** 196.2, with inputs to 196.3, 196.5, and 196.6
**Day 4 goal:** Audit README, INSTALL, maintainer, benchmark, API, corpus,
packaging, planning, and public-header surfaces for claims that must match
earned Epic 17 evidence.

## Summary

Day 4 found that the active documentation generally preserves the right
boundaries: package-manager support remains unclaimed, Windows evidence is
selected-lane scoped, performance evidence is threshold-free, generated API
HTML is local-only, and state-of-the-art status is not claimed.

The main Sprint 196 calibration need is not a broad rewrite. It is to simplify
and route public wording so users see the current support truth first, while
maintainer/report/package docs retain the detailed proof and guard semantics.

## Audit Table

| Surface | Audience | Status | Finding | Later action |
| --- | --- | --- | --- | --- |
| `README.md` | Public first-read | Needs calibration | Accurate but dense around Windows report freshness, hosted selected performance, package support, generated reports, and install support. | Shorten public summaries and route support details to INSTALL. |
| `INSTALL.md` | Public install/support truth | Keep mostly as-is | Support/readiness matrix is the clearest current owner and matches Day 2/3 evidence boundaries. | Preserve as canonical user support surface; verify Windows selected Cholesky status before final wording. |
| `docs/maintainer_guide.md` | Maintainer evidence owner | Keep detailed | Accurate but intentionally detailed; duplicates public docs where proof interpretation is needed. | Keep gate/proof details here and avoid making README carry the same volume. |
| `benchmarks/README.md` | Benchmark methodology owner | Keep mostly as-is | Threshold-free selected-performance semantics are clear and bounded. | Preserve selected-row, non-portable, non-threshold wording. |
| `docs/api_reference.md` | Public API index | Keep mostly as-is | Local-only generated API HTML and non-claim boundaries are clear. | Preserve local-only/selected-header wording. |
| `docs/solver_selection.md` | Public solver routing | Needs possible consolidation | Selected comparison sections are accurate but dense. | Prefer links to corpus/manifest detail if later edits touch this file. |
| `docs/cookbook.md` | Public task routing | Accurate | Routes support and benchmarks to owning docs and avoids broad claims. | No required Day 5 edit. |
| `docs/tutorial.md` | Public learning path | Accurate | Separates local tutorial usage from install/support and report evidence. | No required Day 5 edit. |
| `examples/README.md` | Public example routing | Accurate | Points installed consumers to INSTALL and benchmark interpretation to benchmark docs. | No required Day 5 edit. |
| `tests/corpus/README.md` | Report/manifest owner | Keep detailed | Correctly owns selected target semantics and non-claims. | Keep as authority for report target details. |
| `tests/corpus/schemas/report_index_fields.md` | Schema owner | Accurate | States selected rows do not widen support or create release proof. | No immediate edit. |
| `packaging/homebrew/README.md` | Package proof owner | Needs review if package docs change | Should remain proof-only and missing-license-blocker focused. | Preserve no-support wording if public package text is tightened. |
| `include/*.h` | Public declaration contracts | Avoid broad edits | Comments appear declaration-adjacent; any header edit expands validation scope. | Edit only if a concrete overclaim is found and run full C/header gates. |
| `docs/planning/EPIC_17/PROJECT_PLAN.md` | Planning status owner | Deferred | Final status not updated yet. | Day 7 status edit after evidence and claim calibration. |
| `docs/planning/EPIC_17/EPIC_17_RETROSPECTIVE.md` | Epic closeout owner | Missing by design | Final retrospective waits for claim/status validation. | Create later in Sprint 196. |

## Overclaim Risks

| Risk | Surface | Why it matters | Calibration target |
| --- | --- | --- | --- |
| Windows selected Cholesky wording can be skimmed as broad Windows report proof. | README, INSTALL, maintainer guide, corpus README | Sprint 190 closed a selected workflow path and narrowed a residual; broad Windows report freshness remains unclaimed. | Keep `cholesky-spd-tridiag-5`, hosted evidence, manifest promotion, and non-broad freshness in the same local context. |
| Hosted selected-performance wording can be read as speed evidence. | README, benchmarks README, maintainer guide | Sprint 192 was threshold-free methodology evidence only. | Keep `selected row only`, `threshold-free`, `not portable speed evidence`, and `not release proof` visible. |
| Package proof material can be read as Homebrew support. | README, INSTALL, packaging Homebrew README, maintainer guide | Sprint 188 explicitly retained a missing-license residual. | Keep proof-only wording and source install routing. |
| Selected reliability proof can be read as broad OOM or allocation-failure safety. | README, INSTALL, maintainer guide | Sprint 195 proved one selected owner only. | Keep selected `sparse_symbolic_cholesky()` scope and retained reliability non-claims. |
| Epic 17 final wording can be read as state-of-the-art status. | Future retrospective, README, project plan | Day 3 E17-RQ-026 remains open. | State evidence improvements without claiming unqualified state-of-the-art sparse linear algebra status. |

## Underclaim Or Stale Risks

| Risk | Surface | Required Day 5/6 check |
| --- | --- | --- |
| Selected Cholesky Windows status may be understated or overstated depending on current manifest/workflow state after PR review fixes. | README, INSTALL, maintainer guide, corpus README | Check selected target manifest and workflow status before final wording. |
| Selected symbolic allocation-failure proof may be hard to discover from public docs. | README, INSTALL | Keep selected-owner scope but ensure the support/readiness path exposes the focused gate. |
| Sprint provenance references may be stale after later PR merges. | README, maintainer guide, corpus README | Keep only provenance needed to interpret active guard behavior. |
| The README CMake install section has awkward line wrapping around the Windows hosted PowerShell/selected Cholesky sentence. | README | Reflow for readability without widening the claim. |

## Duplication Findings

| Duplicated topic | Surfaces | Preferred owner |
| --- | --- | --- |
| Support/readiness status | README, INSTALL, maintainer guide, examples, tutorial, API reference | `INSTALL.md#support-readiness-matrix` |
| Package-manager/Homebrew proof blocker | README, INSTALL, packaging Homebrew README, maintainer guide | INSTALL for public status; packaging/maintainer docs for proof detail |
| Selected report freshness | README, solver selection, corpus README, maintainer guide, report-index schema docs | selected target manifest plus corpus/maintainer docs |
| Selected performance methodology | README, benchmarks README, maintainer guide | `benchmarks/README.md` |
| Generated API local-only status | README, API reference, maintainer guide | `docs/api_reference.md` plus maintainer guide for validation details |
| Reliability proof boundaries | README, INSTALL, maintainer guide | INSTALL for support status; maintainer guide for proof details |

## Documentation Edit Plan

1. Public docs: tighten README first, then touch INSTALL/benchmark/tutorial/
   cookbook/solver-selection/examples only where routing or stale wording
   requires it.
2. Maintainer/report/package docs: keep detailed proof semantics, but sync
   wording with whatever public calibration lands.
3. Project-plan status: wait until Day 7 so status edits can reference the
   Day 2 outcome ledger and Day 3 residual IDs.
4. Epic retrospective and residual queue: wait until after public and
   maintainer claim calibration.
5. Public headers: avoid edits unless a concrete overclaim is found; any header
   edit requires the full C/header quality gate.

## Evidence Requirements

| Claim family | Required evidence before widening |
| --- | --- |
| Package-manager support | Approved standalone root license metadata, exact Homebrew license identifier, proof exit `0`, package guards, install validation, and updated docs. |
| Windows selected report freshness | Hosted workflow pass, exact artifact inspection, selected target manifest promotion, coupled generator/normalizer/PowerShell tests, and docs updates. |
| Broad Windows parity | Per-surface Windows proof, not inference from selected CMake or selected comparison lanes. |
| Performance threshold or speed claim | Baseline, variance model, machine-class policy, flake budget, repeated measurements, and failure wording. |
| Portable performance | Multi-platform, multi-machine, repeated, variance-aware evidence with environment context. |
| Broad external parity | Additional bounded references and fixtures promoted one at a time, with row IDs, tolerances, freshness diagnostics, and non-claims. |
| Broad reliability | One selected owner at a time with invariant record, deterministic failure injection, cleanup/stale-output/retry tests, focused gate, docs, and full validation. |
| State-of-the-art status | Broad algorithmic, ecosystem, performance, portability, packaging, reliability, and documentation evidence well beyond Epic 17 selected closures. |

## Validation

- `rg -n -i "state[- ]of[- ]the[- ]art|release|production|supported|support|validated|package-manager|homebrew|windows|powershell|pkg-config|cmake|shared|dynamic ABI|ABI|performance|benchmark|threshold|portable|comparison|external|suitesparse|eigen|scipy|numpy|parity|reliability|allocation-failure|symbolic|freshness|selected" README.md INSTALL.md docs/maintainer_guide.md benchmarks/README.md docs/api_reference.md docs/solver_selection.md docs/cookbook.md docs/tutorial.md examples/README.md tests/corpus/README.md tests/corpus/schemas/report_index_fields.md packaging/homebrew/README.md`
- `rg -n -i "state[- ]of[- ]the[- ]art|release|package-manager|homebrew|windows|powershell|performance|benchmark|comparison|external|parity|reliability|allocation-failure|freshness|selected|support|readiness|shared|dynamic ABI" include/*.h`
- Inspected high-risk README, INSTALL, maintainer guide, and benchmark
  sections directly.

Day 4 changed planning documentation only. No `.c` or `.h` files were modified,
so the full C quality gate is not required for this day.
