# Sprint 216 Day 7: Project-Plan Status, Sprints 212-215

## Status Rule And Source Set

The [Day 6 ledger](day6-project-plan-status-207-211.md) covers 207.1-211.6.
This record assigns exactly one of `complete`, `promoted`, `deferred`,
`residualized`, or `superseded` to each 212.1-215.6 item. The 24 selected
deliverables are `complete`, including decisions to **defer** a stronger
capability and the planned blocked-pilot fallback. No selected timing gate,
generated HTML publication, shared ABI/release, or accepted external result
was promoted. The [project plan](../../PROJECT_PLAN.md) retains every original
item description and estimate.

All cited sprint records are on merged `master`: Sprint 212 [PR #235](https://github.com/jeffreyhorn/linalg_sparse_orthogonal/pull/235),
Sprint 213 [PR #236](https://github.com/jeffreyhorn/linalg_sparse_orthogonal/pull/236),
Sprint 214 [PR #237](https://github.com/jeffreyhorn/linalg_sparse_orthogonal/pull/237),
and Sprint 215 [PR #238](https://github.com/jeffreyhorn/linalg_sparse_orthogonal/pull/238).
The [Day 2 reconciliation](day2-outcome-validation-reconciliation.md) names
historical validation chronology and missing hosted/external evidence. A
merged PR is not a fresh CI run or product promotion.

| Sprint | Decision and retrospective | Validation and closeout | Remaining gate |
| --- | --- | --- | --- |
| 212 | [Threshold-free decision](../../SPRINT_212/artifacts/day5-product-policy-decision.md); [retrospective](../../SPRINT_212/RETROSPECTIVE.md) | [Day 12 validation](../../SPRINT_212/artifacts/day12-integrated-validation.md); [Day 14 closeout](../../SPRINT_212/artifacts/day14-closeout-review.md) | Stable runner class, repeat/warmup/variance, pinned baseline and threshold, artifact and claim decision before timing promotion. |
| 213 | [Local-only policy](../../SPRINT_213/artifacts/day5-product-policy-decision.md); [retrospective](../../SPRINT_213/RETROSPECTIVE.md) | [Day 13 validation](../../SPRINT_213/artifacts/day13-integrated-validation.md); [Day 14 closeout](../../SPRINT_213/artifacts/day14-closeout-review.md) | Destination, retention, routing, freshness, rollback and claim decision before publishing generated HTML. |
| 214 | [Design/deferral decision](../../SPRINT_214/artifacts/day2-abi-release-scope-decision.md); [retrospective](../../SPRINT_214/RETROSPECTIVE.md) | [Day 13 validation](../../SPRINT_214/artifacts/day13-integrated-validation.md); [Day 14 closeout](../../SPRINT_214/artifacts/day14-closeout-review.md) | Shared artifact/export/loader and installed dynamic consumers; candidate tag/archive/provenance and hosted release gates. |
| 215 | [Feasibility decision](../../SPRINT_215/artifacts/day8-pilot-feasibility-decision.md), [blocked outcome](../../SPRINT_215/artifacts/day10-selected-pilot-outcome.md); [retrospective](../../SPRINT_215/RETROSPECTIVE.md) | [Day 13 validation](../../SPRINT_215/artifacts/day13-integrated-validation.md); [Day 14 closeout](../../SPRINT_215/artifacts/day14-closeout-review.md) | Implement v1 schema/validator, prove runtime/fixture provenance, run both solver sides, validate correctness, then decide publication/timing separately. |

## Item Dispositions

| Item | Status | Evidence and scoped rationale |
| --- | --- | --- |
| 212.1 Benchmark Evidence Inventory | complete | [Runner/variance inventory](../../SPRINT_212/artifacts/day3-runner-and-variance-inventory.md) identifies missing repeat, warmup and variance evidence for the selected row. |
| 212.2 Threshold Decision | complete | [Day 5 decision](../../SPRINT_212/artifacts/day5-product-policy-decision.md) selects threshold-free deferral hardening; no hosted timing threshold was approved. |
| 212.3 Methodology Implementation | complete | [Day 8 implementation](../../SPRINT_212/artifacts/day8-tooling-implementation-batch-two.md) hardens freshness and selected metadata for that threshold-free choice. |
| 212.4 Regression Tests | complete | [Day 9 guards](../../SPRINT_212/artifacts/day9-manifest-and-report-guards.md) cover manifest/report methodology and negative cases; review hardening remains in the merged closeout. |
| 212.5 Documentation Calibration | complete | [Day 10 user docs](../../SPRINT_212/artifacts/day10-documentation-calibration.md) and [Day 11 maintainer docs](../../SPRINT_212/artifacts/day11-maintainer-documentation.md) retain non-portable-performance wording. |
| 212.6 Validation And Closeout | complete | [Day 12 validation](../../SPRINT_212/artifacts/day12-integrated-validation.md) and [closeout](../../SPRINT_212/artifacts/day14-closeout-review.md) record selected freshness/manifest/docs checks, not new hosted timing samples. |
| 213.1 Publication Option Review | complete | [Day 3 inventory](../../SPRINT_213/artifacts/day3-publication-option-inventory.md) compares local-only, Pages, retained artifact and committed HTML options. |
| 213.2 Policy Decision | complete | [Day 5 decision](../../SPRINT_213/artifacts/day5-product-policy-decision.md) chooses stronger local-only closure, not publication. |
| 213.3 Automation Implementation | complete | [Day 8 implementation](../../SPRINT_213/artifacts/day8-automation-implementation-batch-two.md) hardens the selected local-only workflow/staging boundary. |
| 213.4 Routing And Guard Tests | complete | [Day 9 routing](../../SPRINT_213/artifacts/day9-routing-and-link-validation.md) and [Day 10 workflow](../../SPRINT_213/artifacts/day10-workflow-and-staging-validation.md) records cover forbidden generated-output references and routes. |
| 213.5 User And Maintainer Docs | complete | [Day 11 user docs](../../SPRINT_213/artifacts/day11-user-documentation-update.md) and [Day 12 maintainer docs](../../SPRINT_213/artifacts/day12-maintainer-documentation-and-residuals.md) preserve source-controlled API routing. |
| 213.6 Validation And Closeout | complete | [Day 13 validation](../../SPRINT_213/artifacts/day13-integrated-validation.md) and [closeout](../../SPRINT_213/artifacts/day14-closeout-review.md) record local API/docs checks; later PR review changes still require final-tree reruns. |
| 214.1 ABI/Release Scope Decision | complete | [Day 2 decision](../../SPRINT_214/artifacts/day2-abi-release-scope-decision.md) chooses design plus static-first deferral, not a shared proof. |
| 214.2 ABI Policy Design | complete | [Day 4 policy](../../SPRINT_214/artifacts/day4-abi-compatibility-policy.md) specifies provisional ABI compatibility gates; it does not approve a shipped ABI subset. |
| 214.3 Build/Install Proof Or Guard | complete | [Day 6 guard](../../SPRINT_214/artifacts/day6-build-path-or-deferral-guard.md) and [Day 7 contract](../../SPRINT_214/artifacts/day7-install-and-metadata-contract.md) reject unsupported shared modes while preserving static install. |
| 214.4 Downstream Consumer Evidence | complete | [Day 9 positive consumers](../../SPRINT_214/artifacts/day9-installed-consumer-evidence.md) and [Day 10 negatives](../../SPRINT_214/artifacts/day10-consumer-negative-matrix.md) exercise installed static paths, not dynamic loaders. |
| 214.5 Release Checklist | complete | [Release checklist](../../SPRINT_214/RELEASE_READINESS_CHECKLIST.md) names candidate, tag, archive, provenance, platform and approval gates; no release passed them. |
| 214.6 Validation And Closeout | complete | [Day 13 validation](../../SPRINT_214/artifacts/day13-integrated-validation.md) and [closeout](../../SPRINT_214/artifacts/day14-closeout-review.md) record local static installs/consumers and selected checks, not candidate-hosted release proof. |
| 215.1 Baseline Selection | complete | [Day 3 decision](../../SPRINT_215/artifacts/day3-baseline-selection-decision.md) pins a tiny SPD CHOLMOD cell and exclusions; loaded-library and fixture-publication identity remain unproved. |
| 215.2 Methodology Specification | complete | [Day 4 numerical contract](../../SPRINT_215/artifacts/day4-correctness-and-tolerance-contract.md) and [Day 5 timing design](../../SPRINT_215/artifacts/day5-external-comparison-methodology.md) specify gates only; no solver/timing sample was accepted. |
| 215.3 Report Schema Design | complete | [Day 6 schema shape](../../SPRINT_215/artifacts/day6-external-report-schema-design.md) and [Day 7 validation design](../../SPRINT_215/artifacts/day7-schema-validation-design.md) are designs; the external v1 schema and validator are absent. |
| 215.4 Selected Pilot | complete | [Day 10 outcome](../../SPRINT_215/artifacts/day10-selected-pilot-outcome.md) satisfies the planned blocked fallback: preflight stopped before either solver, with `schema_validation_unavailable`, empty sides and `timing=not_run`. |
| 215.5 Claim Documentation | complete | [Day 11 matrix](../../SPRINT_215/artifacts/day11-evidence-and-claim-matrix.md) and [Day 12 docs plan](../../SPRINT_215/artifacts/day12-claim-documentation-and-validation-plan.md) separate blueprint, provisional attempt and accepted evidence. |
| 215.6 Validation And Closeout | complete | [Day 13 validation](../../SPRINT_215/artifacts/day13-integrated-validation.md) and [closeout](../../SPRINT_215/artifacts/day14-closeout-review.md) record local C/report/docs/synthetic checks, not CHOLMOD execution or hosted proof. |

## Cross-Item Consistency And Residuals

| Claim lane | Earned outcome | Unclosed gate and next decision target |
| --- | --- | --- |
| Benchmark | Sprint 212 selected canonical **freshness** row, `--repeat 1`, `threshold=n/a`, no computed variance. | Timing threshold remains **deferred**. Benchmark owner must pin runner/compiler, baseline, repeats/warmup, variance, threshold and retained artifacts, then seek a new policy decision; see [212 closeout](../../SPRINT_212/artifacts/day14-closeout-review.md). |
| Generated API | Sprint 213 ignored local `docs/api/html/` plus source-controlled API route and guard. | Hosted/public generated HTML remains **deferred**. API/docs owner must select destination, retention, freshness, routing, rollback and claim boundary; see [213 closeout](../../SPRINT_213/artifacts/day14-closeout-review.md). |
| ABI/release | Sprint 214 design, static-first deferral guard and local installed static consumers. | Shared ABI and release readiness remain **deferred**. ABI/release owner must approve/export a subset, prove loader and dynamic consumers, then candidate tag/archive/provenance/hosted gates; see [214 residual queue](../../SPRINT_214/RESIDUAL_QUEUE.md). |
| External baseline | Sprint 215 pinned blueprint, optional helper and reproducible **blocked** preflight. | Accepted external comparison remains **residualized**. Report/pilot owner must implement v1 validation, capture actual runtime provenance, approve fixture/license, run two-sided correctness, then decide timing/publication; see [215 residual queue](../../SPRINT_215/RESIDUAL_QUEUE.md). |

The [Day 6 boundaries](day6-project-plan-status-207-211.md#capability-boundaries-and-next-targets)
remain unchanged. In particular, a reviewed selected Windows Cholesky
artifact is not selected Windows freshness, a QR collection lane is not a
hosted QR result, and local package proof is not public distribution.

## Contradiction Log And Sprint 216 Handoff

| ID | Day 7 finding | Disposition |
| --- | --- | --- |
| D2-1 / C16 | Project-plan snapshot combined `215-216` as pending even though Sprint 215 merged in PR #238. | Corrected in the project plan: separate merged Sprint 215 and provisional Sprint 216 rows. |
| D2-2 / D2-3 | Sprint 212-215 integrated check records predate some PR review fixes; Sprint 215's nine synthetic tests were a then-current count, while Day 2 observed 13 locally. | Historical logs unchanged. Days 9-14 must record current-tree focused results separately. |
| D2-4 | Sprint 215 schema/validator absent; provisional `attempt.json` is not accepted `evidence.json`. | Remains a product blocker and Day 8 residual-queue input; not an item 215.4 failure under the planned fallback. |
| D2-5 | No new current hosted artifact was inspected for this status pass. | Keep hosted and platform scope historical; fresh promotion needs exact run/job/artifact inspection. |
| D2-6 | Sprint 213/214/215 decisions are local-only, design-only, and blueprint/blocked respectively. | No publication, ABI/release or external-result promotion. Public claim matrix remains as calibrated on Day 5. |

All 54 Sprint 207-215 items now have one evidence-backed selected-deliverable
status across the Day 6 and Day 7 ledgers. Sprint 216 items 216.1-216.6 are
**unassigned placeholders** until final integrated validation and Day 14
closeout review. Item 216.3's status table is prepared, not a final epic
completion claim. Day 7 changes Markdown only; no C/header or hosted execution
is attributed to this ledger.

## Day 7 Validation

The 54-item uniqueness/status audit, local-link check, `make docs-check`,
`python3 -m unittest tests.test_external_baseline_claim_docs` (5 tests),
`make support-docs-guard`, `python3 tests/test_selected_performance_docs.py`,
trailing-whitespace check, and `git diff --check` passed locally. These checks
validate status/documentation consistency, not external or hosted capability.
