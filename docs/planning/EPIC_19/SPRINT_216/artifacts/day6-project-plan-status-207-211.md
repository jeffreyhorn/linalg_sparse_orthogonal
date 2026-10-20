# Sprint 216 Day 6: Project-Plan Status, Sprints 207-211

## Disposition Rule

This ledger assigns exactly one current status to each project-plan item:
`complete`, `promoted`, `deferred`, `residualized`, or `superseded`.
`Complete` means the **selected item deliverable** was finished, including a
decision to re-defer a capability. It does not mean that the capability under
consideration was promoted. All 30 items below are complete in that narrow
sense; their unearned capabilities remain deferred or residualized in the
separate boundary table. No item is assigned `promoted` solely from a passing
local test or a registered workflow. The original descriptions and estimates
remain unchanged in the [project plan](../../PROJECT_PLAN.md).

The source set is merged Sprint 207-211 evidence in PRs
[#230](https://github.com/jeffreyhorn/linalg_sparse_orthogonal/pull/230),
[#231](https://github.com/jeffreyhorn/linalg_sparse_orthogonal/pull/231),
[#232](https://github.com/jeffreyhorn/linalg_sparse_orthogonal/pull/232),
[#233](https://github.com/jeffreyhorn/linalg_sparse_orthogonal/pull/233), and
[#234](https://github.com/jeffreyhorn/linalg_sparse_orthogonal/pull/234).
The [Day 2 reconciliation](day2-outcome-validation-reconciliation.md)
compares those decisions with their validation and closeout records. These are
historical, bounded results, not a fresh hosted or full-suite rerun.

## Item Dispositions

| Item | Status | Evidence and rationale |
| --- | --- | --- |
| 207.1 Provider Scope Decision | complete | [Day 5 decision](../../SPRINT_207/artifacts/day5-provider-decision.md) selected continued provider deferral; it did not approve a public Homebrew path. |
| 207.2 Formula And Metadata Audit | complete | [Day 3 formula baseline](../../SPRINT_207/artifacts/day3-formula-metadata-baseline.md) and [Day 4 environment baseline](../../SPRINT_207/artifacts/day4-environment-proof-baseline.md) audited the local static source formula, metadata, archive, and proof assumptions. |
| 207.3 Proof Path Implementation | complete | [Day 7 implementation](../../SPRINT_207/artifacts/day7-deferral-guard-implementation.md) strengthened the selected deferral/proof path; the existing developer-mode local proof was retained, not made public. |
| 207.4 Package Guard Alignment | complete | [Day 9 alignment](../../SPRINT_207/artifacts/day9-package-guard-alignment.md) bound provider and static-package guards to the retained non-claims. |
| 207.5 User And Maintainer Docs | complete | [Day 10 user docs](../../SPRINT_207/artifacts/day10-user-package-docs.md) and [Day 11 maintainer docs](../../SPRINT_207/artifacts/day11-maintainer-package-docs.md) narrowed package guidance to the local/static tier. |
| 207.6 Validation And Closeout | complete | [Day 12 validation](../../SPRINT_207/artifacts/day12-integrated-validation.md) and [Day 14 closeout](../../SPRINT_207/artifacts/day14-closeout-review.md) recorded passing local package/install/docs gates and remaining provider residuals. |
| 208.1 Hosted Artifact Intake | complete | [Day 2 inventory](../../SPRINT_208/artifacts/day2-hosted-artifact-inventory.md) inspected exact Windows Cholesky run/job/artifact and six selected files. |
| 208.2 Manifest Promotion Decision | complete | [Day 5 decision](../../SPRINT_208/artifacts/day5-promotion-decision.md) explicitly re-deferred selected Windows freshness despite bounded hosted rows, because support tier and generated non-claims were not aligned. |
| 208.3 Metadata And Guard Implementation | complete | [Day 7 implementation](../../SPRINT_208/artifacts/day7-manifest-guard-implementation.md) kept Windows absent from positive manifest metadata and guarded re-deferral. |
| 208.4 Normalizer Regression Coverage | complete | [Day 10 implementation](../../SPRINT_208/artifacts/day10-normalizer-regression-implementation.md) added selected Windows-path and row-shape failure cases. |
| 208.5 Documentation Calibration | complete | [Day 11 public docs](../../SPRINT_208/artifacts/day11-public-docs-calibration.md) and [Day 12 maintainer/corpus docs](../../SPRINT_208/artifacts/day12-maintainer-corpus-docs.md) aligned wording to bounded workflow evidence and non-promotion. |
| 208.6 Validation And Closeout | complete | [Day 13 validation](../../SPRINT_208/artifacts/day13-integrated-validation.md) and [Day 14 closeout](../../SPRINT_208/artifacts/day14-closeout-review.md) passed recorded guards; missing local `pwsh` was an expected negative fixture, not native PowerShell proof. |
| 209.1 MSVC Probe Design | complete | [Day 2 design](../../SPRINT_209/artifacts/day2-msvc-probe-design.md) specified the bounded QR incompatible command and artifact contract. |
| 209.2 Workflow Implementation | complete | [Day 5 implementation](../../SPRINT_209/artifacts/day5-workflow-implementation.md) installed an evidence-collection lane, not a hosted result. |
| 209.3 Artifact Inspection Tests | complete | [Day 6 tests](../../SPRINT_209/artifacts/day6-artifact-inspection-tests.md) cover selected row/path and artifact inspection failures. |
| 209.4 Manifest Decision | complete | [Day 8 decision](../../SPRINT_209/artifacts/day8-manifest-decision.md) re-deferred selected Windows QR freshness after the branch query returned no hosted run/artifact. |
| 209.5 Docs And Claim Guards | complete | [Day 10 public docs](../../SPRINT_209/artifacts/day10-public-docs-calibration.md) and [Day 11 maintainer/corpus docs](../../SPRINT_209/artifacts/day11-maintainer-corpus-docs.md) kept Windows QR, support tier, and manifest claims bounded. |
| 209.6 Validation And Closeout | complete | [Day 13 validation](../../SPRINT_209/artifacts/day13-integrated-validation.md) and [Day 14 closeout](../../SPRINT_209/artifacts/day14-closeout-review.md) recorded local guard passes without a hosted QR pass. |
| 210.1 Owner Selection | complete | [Day 2 ranking](../../SPRINT_210/artifacts/day2-owner-ranking.md) selected no-reorder linked-list LDLT numeric factorization, not all LDLT paths. |
| 210.2 Lifecycle Invariant Record | complete | [Day 3 baseline](../../SPRINT_210/artifacts/day3-lifecycle-baseline.md) fixed cleanup, stale-output, caller-input, and retry invariants for that owner. |
| 210.3 Harness Extension | complete | [Day 5 implementation](../../SPRINT_210/artifacts/day5-harness-implementation.md) wired deterministic fail-after injection to selected allocations. |
| 210.4 Regression Tests | complete | [Day 6 sweep](../../SPRINT_210/artifacts/day6-failure-sweep.md) and [Day 9 retry proof](../../SPRINT_210/artifacts/day9-retry-proof.md) exercise 25 failure sites and recovery invariants. |
| 210.5 Gate And Documentation | complete | [Day 10 gate](../../SPRINT_210/artifacts/day10-focused-gate.md) and [Day 11 docs](../../SPRINT_210/artifacts/day11-documentation-calibration.md) add focused Make/CTest coverage and selected-only claims. |
| 210.6 Validation And Closeout | complete | [Day 12 validation](../../SPRINT_210/artifacts/day12-integrated-validation.md) and [Day 14 closeout](../../SPRINT_210/artifacts/day14-closeout-review.md) record focused and full C quality passes for the selected owner. |
| 211.1 Candidate Ranking | complete | [Day 2 ranking](../../SPRINT_211/artifacts/day2-candidate-ranking.md) selected one LDLT CSC native-parity test cluster. |
| 211.2 Cluster Boundary | complete | [Day 3 boundary](../../SPRINT_211/artifacts/day3-cluster-boundary.md) kept `tests/test_ldlt_csc.c` as the registration/proof owner. |
| 211.3 Extraction Design | complete | [Day 4 design](../../SPRINT_211/artifacts/day4-extraction-design.md) specified the family-local helper, dependencies, and guards. |
| 211.4 Extraction Implementation | complete | [Day 7 extraction](../../SPRINT_211/artifacts/day7-extraction-batch-two.md) moved selected test bodies while retaining names and registration order. |
| 211.5 Guard And Test Coverage | complete | [Day 9 guard](../../SPRINT_211/artifacts/day9-ownership-guard.md), [Day 10 regression](../../SPRINT_211/artifacts/day10-behavior-regression.md), and [closeout](../../SPRINT_211/artifacts/day14-closeout-review.md) cover ownership and behavior, including later PR hardening. |
| 211.6 Validation And Closeout | complete | [Day 12 validation](../../SPRINT_211/artifacts/day12-integrated-validation.md) and [Day 14 closeout](../../SPRINT_211/artifacts/day14-closeout-review.md) record full C gates and CMake registration parity, not a full CTest run. |

## Capability Boundaries And Next Targets

| Sprint | Delivered boundary | Remaining capability and next acceptance target |
| --- | --- | --- |
| 207 | Developer-mode local static Homebrew source proof and provider-deferral guards. | Public provider distribution remains **residualized**: provider owner must establish stable archive/checksum provenance, non-local install/test/uninstall, cleanup, review, and publication policy before any public tap or broader package claim; see [207 closeout](../../SPRINT_207/artifacts/day14-closeout-review.md). |
| 208 | Reviewed six-row hosted Windows Cholesky workflow artifact plus re-deferral guards. | Selected Windows freshness remains **deferred**: report/Windows owner must align manifest, generated `local_only` support tier, non-claims, workflow metadata, and docs, then re-inspect hosted evidence; see [208 decision](../../SPRINT_208/artifacts/day5-promotion-decision.md). |
| 209 | Bounded Windows/MSVC QR evidence-collection lane and local inspection regressions. | Selected Windows QR freshness remains **deferred**: CI/manifest owner needs an inspected hosted six-file artifact, then generated support/manifest/doc alignment; see [209 decision](../../SPRINT_209/artifacts/day8-manifest-decision.md). |
| 210 | One selected no-reorder linked-list LDLT allocation-failure owner with 25-site proof. | Broad allocation reliability remains **residualized**: solver/test owner must select additional owners and repeat failure, cleanup, stale-output, input-preservation, and retry proof; see [210 closeout](../../SPRINT_210/artifacts/day14-closeout-review.md). |
| 211 | One LDLT CSC native-parity test-owner extraction with unchanged observed selected behavior. | Broader review-surface reduction remains **residualized**: review/test owner must select and validate each further cluster separately; see [211 closeout](../../SPRINT_211/artifacts/day14-closeout-review.md). |

## Open Questions And Day 7 Handoff

- The [project-plan snapshot](../../PROJECT_PLAN.md#epic-19-current-status-snapshot)
  still combines Sprint 215-216 as pending. Day 7 must split that row and
  assign Sprint 212-215 item statuses using merged evidence. Day 6 does not
  silently close discrepancy D2-1 or C16.
- A current hosted Cholesky or QR run was not re-queried for this ledger.
  Promotion requires an exact run/job/artifact and generated-claim alignment,
  not merely the historical workflow definition or six passing old rows.
- Current-tree integrated checks remain scheduled for Days 9-14. The linked
  sprint validations establish their historical scope only; PR review
  chronology and platform limits remain in the
  [Day 2 gap register](day2-outcome-validation-reconciliation.md#discrepancy-and-gap-register).

Day 6 changes documentation only. The 30-item uniqueness/status audit,
local-link check, `make docs-check`, trailing-whitespace check, and
`git diff --check` passed. No C/header or hosted gate was run for this
status-only edit.
