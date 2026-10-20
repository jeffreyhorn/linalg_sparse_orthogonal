# Sprint 216 Day 12: Epic Outcomes Draft Cross-Check

## Draft Scope

The substantial [Epic 19 retrospective draft](../../EPIC_19_RETROSPECTIVE.md)
follows the [Day 11 outline](day11-quality-gate-and-retro-outline.md). It
reconciles merged Sprint 207-215 selected outcomes with the [Day 1 source
inventory](day1-epic-evidence-intake.md), [Day 2 decision/validation
comparison](day2-outcome-validation-reconciliation.md), and the item-level
[Day 6](day6-project-plan-status-207-211.md) and [Day 7](day7-project-plan-status-212-215.md)
ledgers. The draft is **not** an Epic 19 closeout or an assignment of Sprint
216 item statuses.

## Primary-Evidence Cross-Check

| Retrospective lane | Primary source and checked fact | Claim boundary retained |
| --- | --- | --- |
| Package/provider | [Sprint 207 decision](../../SPRINT_207/artifacts/day5-provider-decision.md) selected continued provider deferral; [Day 11 current gates](day11-quality-gate-and-retro-outline.md) passed installed static consumers. | Local developer-mode Homebrew proof and static source install, not a public provider or release. |
| Windows Cholesky | [Sprint 208 artifact intake](../../SPRINT_208/artifacts/day2-hosted-artifact-inventory.md) inspected six selected files; [decision](../../SPRINT_208/artifacts/day5-promotion-decision.md) re-deferred metadata/claim promotion. | Historical bounded hosted observation, no promoted selected Windows freshness or broad Windows parity. |
| Windows QR | [Sprint 209 manifest decision](../../SPRINT_209/artifacts/day8-manifest-decision.md) had no inspected hosted QR artifact. | Workflow and local guard coverage, not a hosted QR pass. |
| Allocation reliability | [Sprint 210 failure sweep](../../SPRINT_210/artifacts/day6-failure-sweep.md) covers 25 deterministic sites for one no-reorder linked-list LDLT owner. | No all-solver, all-LDLT, platform or OS-OOM reliability claim. |
| Review surface | [Sprint 211 extraction](../../SPRINT_211/artifacts/day7-extraction-batch-two.md) and [regression](../../SPRINT_211/artifacts/day10-behavior-regression.md) cover one LDLT CSC test cluster. | 295-line selected test reduction, no new production solver behavior. |
| Benchmark | [Sprint 212 policy](../../SPRINT_212/artifacts/day5-product-policy-decision.md) retains `--repeat 1`, `threshold=n/a`; [Day 10 current output](day10-focused-validation.md) records local freshness. | No timing gate, portable speed or comparative superiority. |
| Generated API | [Sprint 213 policy](../../SPRINT_213/artifacts/day5-product-policy-decision.md) selects ignored local HTML; [Day 10 docs](day10-focused-validation.md) confirm 18 reference/source pages. | Local-only Doxygen output, no hosted/retained/committed HTML publication. |
| ABI and release | [Sprint 214 checklist](../../SPRINT_214/RELEASE_READINESS_CHECKLIST.md) names future gates; [Day 11 current tests](day11-quality-gate-and-retro-outline.md) passed Make 26/26 and CMake 30/30 static install checks. | Local static consumers and design, no shared loader/ABI or approved candidate release. |
| External baseline | [Sprint 215 blocked pilot](../../SPRINT_215/artifacts/day10-selected-pilot-outcome.md) exited `2` before either solver, with `timing=not_run`; [Day 10 current pilot tests](day10-focused-validation.md) passed 13 synthetic cases. | No accepted v1 `evidence.json`, CHOLMOD correctness, timing or external parity. |

The 54-item count is the union of 207.1-211.6 (30) and 212.1-215.6 (24),
each with a single selected-deliverable `complete` disposition in the two
ledgers. It does not count 54 promoted capabilities. Current Sprint 216
items 216.1-216.6 remain `unassigned` in the [project plan](../../PROJECT_PLAN.md).
The [Day 9 queue](../RESIDUAL_QUEUE.md) has 21 acceptance targets; every
unearned capability above maps to at least one RQ ID in the retrospective.

## Negative And Validation Evidence

- The [Day 2 discrepancy register](day2-outcome-validation-reconciliation.md#discrepancy-and-gap-register)
  separates the corrected 215/216 status contradiction (D2-1), later PR
  review changes (D2-2), historical nine versus current 13 pilot tests
  (D2-3), absent v1 schema/validator (D2-4), unrefreshed hosted currency
  (D2-5), and design-only decisions (D2-6). None is called a new hosted pass.
- The Sprint 215 [blocked attempt](../../SPRINT_215/artifacts/day10-selected-pilot-outcome.md)
  is explicit negative evidence: exit `2`, no solver sides, no accepted
  result. Missing schema/validator is not a passing optional-data skip.
- [Day 10](day10-focused-validation.md) passed 14 focused commands locally;
  its six comparison bundles contained 40 `local_only` passing rows from
  source-controlled dense helpers, while NumPy/SciPy rows deferred. The
  benchmark row is `measurement` with `threshold=n/a`, not a speed gate.
- [Day 11](day11-quality-gate-and-retro-outline.md) passed six serial
  static/package commands. The Make install test's `make clean` removed
  ignored Day 10 generated reports; their hashes are snapshots, not current
  retained artifacts. No C/header change triggered the full source chain.
- No new hosted Windows QR/Cholesky, candidate release, shared loader or
  external CHOLMOD/timing run was inspected or claimed by this Day 12 draft.
  The [Day 9 state-of-the-art decision](day9-residual-decision-validation-matrix.md#state-of-the-art-evidence-decision)
  remains negative pending Day 13 assessment.

## Day 13-14 Handoff

Day 13 must challenge the draft against the [Day 3 claim inventory](day3-claim-inventory.md),
[Day 5 claim matrix](day5-benchmark-api-corpus-claims.md), final
[residual queue](../RESIDUAL_QUEUE.md), and exact state-of-the-art entry bar.
Day 14 must rerun checks affected by retrospective/project-plan edits,
verify all local links and tracked/untracked whitespace, audit the final
216.1-216.6 status rows, and record any hosted evidence delta before
changing the no-claim decision. The retrospective must remain a draft if a
required current-tree gate fails or a purported source does not support its
statement.

## Day 12 Documentation Validation

On Darwin `24.6.0` x86_64, the 54-item status audit and 21-ID queue audit
passed, as did relative-link existence checks for the retrospective, this
cross-check, working notes and project plan. The following commands each
exited `0`: `python3 -m unittest tests.test_external_baseline_claim_docs`
(5 tests), `python3 tests/test_selected_performance_docs.py`,
`make support-docs-guard`, `make docs-check` (18 public reference and source
pages for 18 headers), and `git diff --check`. An explicit trailing-whitespace
scan of the new and edited Markdown found no matches. No source/header,
hosted, external-solver or installed-package gate was rerun for this Day 12
documentation draft; Day 10-11 records remain the local execution evidence.
