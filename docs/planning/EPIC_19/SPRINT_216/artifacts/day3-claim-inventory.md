# Sprint 216 Day 3: Evidence Freeze And Claim Inventory

## Evidence Freeze For Item 216.1

The [Day 1 source index](../WORKING_NOTES.md#source-index) and
[Day 2 reconciliation](day2-outcome-validation-reconciliation.md) are the
frozen historical evidence map for Sprints 207-215. It consists of each
sprint's decision, retrospective, integrated validation, Day 14 closeout,
merged PR #230-#238, and the current source-controlled manifest and contract
files cited by Day 2. No missing sprint-level decision or closeout record was
found. This is a **source-set freeze**, not a declaration that all product
capabilities or current-tree quality gates passed. New hosted artifacts or
later Sprint 216 validation must be recorded as dated deltas, not silently
substituted for the frozen records.

| Evidence group | Frozen outcome | Limit / disposition |
| --- | --- | --- |
| [207](../../SPRINT_207/artifacts/day5-provider-decision.md), [208](../../SPRINT_208/artifacts/day5-promotion-decision.md), [209](../../SPRINT_209/artifacts/day8-manifest-decision.md) | Provider deferral; selected Windows Cholesky and QR freshness re-deferral. | Local Homebrew proof, one exact hosted Cholesky observation, and QR collection-lane definition do not establish broad package or Windows support. |
| [210](../../SPRINT_210/artifacts/day12-integrated-validation.md), [211](../../SPRINT_211/artifacts/day12-integrated-validation.md) | One no-reorder linked-list LDLT allocation owner and one native-parity test-helper extraction. | Neither proves broad allocation reliability or new production solver behavior. |
| [212](../../SPRINT_212/artifacts/day5-product-policy-decision.md), [213](../../SPRINT_213/artifacts/day5-product-policy-decision.md) | Selected benchmark stays threshold-free; generated API HTML stays local-only. | No portable timing gate or hosted API publication. |
| [214](../../SPRINT_214/artifacts/day2-abi-release-scope-decision.md), [215](../../SPRINT_215/artifacts/day10-selected-pilot-outcome.md) | Static-first ABI/release design and blocked external-pilot fallback. | No shared ABI/release approval; no two-sided external solver result or accepted v1 evidence. |

Unresolved discrepancies from [Day 2](day2-outcome-validation-reconciliation.md#discrepancy-and-gap-register)
remain open, not passed:

| ID | Day 3 disposition | Closure owner / target |
| --- | --- | --- |
| D2-1 | **Planning correction required:** project snapshot still marks 215-216 pending despite merged #238. | Epic owner, Days 6-7. |
| D2-2 | **Current-tree validation pending:** historical Sprint 212-215 checks predate later PR review edits. | Quality owner, Days 9-14. |
| D2-3 | **Historical count retained:** Sprint 215 recorded nine synthetic tests; Day 2 reran 13 current local tests. Neither count is external solver proof. | Quality/docs owner, final validation record. |
| D2-4 | **External-result blocker:** v1 evidence schema/validator absent; pilot remains blocked before either solver. | Pilot/report owner, residual queue and Day 13 assessment. |
| D2-5 | **Hosted currency unknown:** do not infer a current run or artifact from workflow registration or historical observations. | CI/platform owner; exact run review before promotion. |
| D2-6 | **Capability deferred by decision:** API publication, shared ABI/release, and external parity are not delivered capabilities. | Docs/product owner, Days 4-5 claims and Day 8 queue. |

Item 216.1 evidence reconciliation is complete at this bounded source-set level.
Final item and release dispositions remain provisional until the Sprint 216
integrated validation and closeout review.

## Claim Audit Method

Audited the current [README](../../../../../README.md),
[INSTALL](../../../../../INSTALL.md),
[maintainer guide](../../../../maintainer_guide.md),
[benchmark guide](../../../../../benchmarks/README.md),
[API reference](../../../../api_reference.md),
[corpus guide](../../../../../tests/corpus/README.md), and
[Epic 19 project plan](../../PROJECT_PLAN.md) for support, platform,
release, publication, performance, and state-of-the-art wording. The row
location identifies the current statement or the place where a necessary
claim/status is absent. Classification is about *wording against evidence*,
not whether a command happens to pass today. Proposed text is the narrowest
evidence-supported wording; it is scheduled for Days 4-5 or 6-7, not applied
on Day 3.

| ID / class | Claim and surface | Evidence and exact scope | Negative boundary | Reviewer decision / narrow wording |
| --- | --- | --- | --- | --- |
| C01 / **ambiguous** | [README Current Capabilities](../../../../../README.md#direct-solvers) says CSR LU and CSC direct kernels provide "faster large-matrix paths." | [Sprint 212 policy](../../SPRINT_212/artifacts/day5-product-policy-decision.md) and [benchmark guide](../../../../../benchmarks/README.md#report-index-handoff) document selected, context-bound measurement only. | No unqualified comparative speed across large matrices, backends, or platforms. | **Revise Day 4:** describe compressed-storage dispatch and intended workload, without a universal "faster" claim; cite exact selected benchmark if discussing measured speed. |
| C02 / **too broad** | [README CSR LU heading](../../../../../README.md#api-overview) calls it a "high-performance path." | The public CSR LU API exists, but current selected timing evidence is a separate CSC refactor row ([Sprint 212 decision](../../SPRINT_212/artifacts/day5-product-policy-decision.md)). | No measured CSR LU superiority is established by that row. | **Revise Day 4:** "CSR LU (compressed-storage path)." |
| C03 / **supported** | [README selected-performance section](../../../../../README.md#choose-a-workflow) distinguishes one hosted freshness row from local sentinel thresholds. | [Sprint 212 decision](../../SPRINT_212/artifacts/day5-product-policy-decision.md) and selected manifest: `nos4.mtx --repeat 1`, `baseline=n/a`, `threshold=n/a`. | No hosted timing gate, portable speed, or cross-platform parity. | **Retain Day 4:** preserve threshold-free and local-sentinel distinction. |
| C04 / **supported** | [README external-baseline paragraph](../../../../../README.md#choose-a-workflow) names a blocked preflight. | [Sprint 215 Day 10](../../SPRINT_215/artifacts/day10-selected-pilot-outcome.md): v1 schema/validator unavailable, neither solver ran. | Blueprint and provisional attempt are not accepted correctness, timing, or state-of-the-art evidence. | **Retain Day 4:** keep explicit `blocked before either solver` wording. |
| C05 / **supported** | [INSTALL support matrix](../../../../../INSTALL.md#support-readiness-matrix) names static Unix and Windows MSVC CMake install paths. | [Sprint 214 integrated validation](../../SPRINT_214/artifacts/day13-integrated-validation.md) and [release checklist](../../SPRINT_214/RELEASE_READINESS_CHECKLIST.md): local static consumers and specific hosted workflow ownership. | No shared library, dynamic ABI, Windows Make/`pkg-config` parity, or release approval. | **Retain/verify Day 4:** preserve platform/toolchain cells and distinguish local rerun from historical CI registration. |
| C06 / **ambiguous** | [INSTALL broad ecosystem row](../../../../../INSTALL.md#support-readiness-matrix) cites Epic 17 review/todo as its evidence owner, omitting the newer Sprint 215 blocked pilot. | [Sprint 215 claim matrix](../../SPRINT_215/artifacts/day11-evidence-and-claim-matrix.md) is the current external evidence design. | No SuiteSparse parity or state-of-the-art claim. | **Revise Day 4:** cite the Sprint 215 blueprint and blocked Day 10 attempt; retain "not claimed." |
| C07 / **missing** | [INSTALL support matrix](../../../../../INSTALL.md#support-readiness-matrix) has no distinct external-baseline evidence row, so the blocked pilot is not visible from the main readiness table. | [Sprint 215 outcome](../../SPRINT_215/artifacts/day10-selected-pilot-outcome.md). | Do not label a blocked attempt as an observed comparison. | **Add Day 4:** "External baseline: blueprint only; pilot blocked pre-solve; no accepted result." |
| C08 / **supported** | [INSTALL generated API row](../../../../../INSTALL.md#support-readiness-matrix) says local-only. | [Sprint 213 policy](../../SPRINT_213/artifacts/day5-product-policy-decision.md). | No hosted/retained/committed HTML. | **Retain Day 4:** local generated view only. |
| C09 / **ambiguous** | [Maintainer API section](../../../../maintainer_guide.md#api-reference-and-generated-doxygen-html) still calls Sprint 204 the "current policy owner." | [Sprint 213 policy](../../SPRINT_213/artifacts/day5-product-policy-decision.md) is the later local-only publication decision and guard hardening. | No policy promotion to publication. | **Revise Day 4:** name Sprint 213 as current local-only decision; keep 179/186/204 as history. |
| C10 / **supported** | [Maintainer release section](../../../../maintainer_guide.md#packaging-and-abi-contract) treats Sprint 214 checklist as future gates. | [Sprint 214 decision](../../SPRINT_214/artifacts/day2-abi-release-scope-decision.md) and [checklist](../../SPRINT_214/RELEASE_READINESS_CHECKLIST.md). | Checklist is not a tag, artifact, loader test, shared ABI, or approval. | **Retain Day 4:** no release-ready language. |
| C11 / **supported** | [Maintainer external track](../../../../maintainer_guide.md#backend-aware-performance-surface-ownership) separates planning JSON, provisional attempt, and accepted evidence. | [Sprint 215 claim matrix](../../SPRINT_215/artifacts/day11-evidence-and-claim-matrix.md). | No inferred solver pass, speed, or parity. | **Retain Day 4:** exact status vocabulary and v1 blocker. |
| C12 / **supported** | [Benchmark guide external paragraph](../../../../../benchmarks/README.md#report-index-handoff) excludes the blocked pilot from canonical/normalized report rows. | [Sprint 215 outcome](../../SPRINT_215/artifacts/day10-selected-pilot-outcome.md) and [Sprint 212 policy](../../SPRINT_212/artifacts/day5-product-policy-decision.md). | Neither report freshness nor blueprint is an external comparison or timing result. | **Retain Day 5:** preserve row-family separation. |
| C13 / **supported** | [Corpus selected targets](../../../../../tests/corpus/README.md) say Windows QR/Cholesky selected freshness remains unpromoted and benchmark row is threshold-free. | [Sprint 208 decision](../../SPRINT_208/artifacts/day5-promotion-decision.md), [Sprint 209 decision](../../SPRINT_209/artifacts/day8-manifest-decision.md), and [Sprint 212 policy](../../SPRINT_212/artifacts/day5-product-policy-decision.md). | No broad Windows, external-library, or performance parity. | **Retain Day 5:** do not turn collection workflows into selected freshness. |
| C14 / **ambiguous** | [API reference publication history](../../../../api_reference.md) still says Sprint 204 is "the current policy owner." | [Sprint 213 policy](../../SPRINT_213/artifacts/day5-product-policy-decision.md) supersedes it while retaining local-only output. | No hosted publication or completeness beyond configured public headers. | **Revise Day 5:** name Sprint 213 as the latest decision; retain earlier policy history. |
| C15 / **supported** | [API reference local HTML contract](../../../../api_reference.md) routes durable links through checked-in headers and the source-controlled page. | [Sprint 213 validation](../../SPRINT_213/artifacts/day13-integrated-validation.md). | Local generation/staging checks do not prove publication or stable ABI. | **Retain Day 5:** no publication promotion from a passing docs check. |
| C16 / **too broad** | [Epic 19 project snapshot](../../PROJECT_PLAN.md#epic-19-current-status-snapshot) says "Sprints 215-216 remain pending" and combines them in one pending row. | [Sprint 215 closeout](../../SPRINT_215/artifacts/day14-closeout-review.md) and merged [PR #238](https://github.com/jeffreyhorn/linalg_sparse_orthogonal/pull/238). | Sprint 215 closed blueprint/fallback work only; Sprint 216 is in progress. | **Revise Days 6-7:** close 215 with blocker and non-claims, mark 216 in progress; do not mark Epic complete. |
| C17 / **supported** | [Epic 19 non-goals](../../PROJECT_PLAN.md#non-goals) reject broad package, Windows, shared ABI, publication, speed, and state-of-the-art claims. | [Day 2 reconciliation](day2-outcome-validation-reconciliation.md) spans Sprint 207-215 outcomes. | Goals/decisions are not product proof. | **Retain Days 6-7:** update statuses without deleting non-goals. |

## Handoff

Days 4-5 own public wording changes C01-C15 and focused claim guards where
repeatable overclaims can be tested. Days 6-7 own C16 and item-level status
links. Day 8 carries unearned product capabilities and D2-4/D2-5 into the
prioritized residual queue. Days 9-14 rerun current-tree checks, inspect any
promotion-specific hosted evidence, and decide final claims. No stronger
support, release, performance, or state-of-the-art claim is approved today.
