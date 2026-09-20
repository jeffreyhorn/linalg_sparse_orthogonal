# Day 12: Residual Queue Draft

**Sprint:** 206 - Epic 18 Final Validation, Claim Calibration & Closeout  
**Theme:** Publish a prioritized residual queue with closure targets and
long-horizon deferrals.  
**Time estimate:** 12 hours  
**Branch:** `sprint-206`  
**Base commit:** `4d819093`

## Scope

Day 12 refreshed `docs/planning/EPIC_18/EPIC_18_RESIDUAL_QUEUE.md` from the
current Sprint 197-206 closeout evidence. The update separates selected
closures from broader residual claims so future planning does not treat closed
selected work as unstarted work or treat selected proof as broad support.

## Residual Queue Updates

| Section | Day 12 update |
| --- | --- |
| Purpose | Reframed the queue as the current Sprint 206 Day 12 handoff, with Sprint 197 historical and Sprints 198-205 closed for selected scopes or re-deferrals. |
| Queue summary | Kept ten residual themes and clarified near-term versus long-horizon horizons. |
| E18-RQ-001 | Updated package-manager residual to distinguish closed Sprint 198 local developer-mode Homebrew proof from broader provider/package distribution work. |
| E18-RQ-002 | Updated Windows Cholesky residual to reflect Sprint 199 re-deferral rather than pending implementation. |
| E18-RQ-003 through E18-RQ-008 | Updated each selected closure to preserve current evidence while naming the remaining broader residual scope. |
| Long-horizon deferrals | Kept release/ABI and state-of-the-art as explicit long-horizon work requiring separate product/platform/methodology scope. |
| Final claim decision | Replaced stale Sprint 197-only claim language with current Sprint 198-206 selected evidence and retained broad non-claims. |
| Source evidence | Added Sprint 198-206 retrospective and Sprint 206 artifact links. |

## Prioritization Rules

Residual priority is based on:

- user value;
- claim risk;
- implementation and maintenance cost;
- availability of hosted or local evidence;
- whether a residual can be closed as one selected proof instead of a broad
  platform or product program.

Near-term residuals should still close one bounded claim at a time. Long-horizon
residuals should not be treated as ready implementation work until a future
epic allocates product, platform, methodology, or research scope.

## Claim Boundaries Preserved

The Day 12 residual queue does not promote:

- broad package-manager distribution, Homebrew/core readiness, bottles,
  Linuxbrew, public tap maintenance, or binary package support;
- broad Windows support, selected Windows Cholesky promotion, or selected
  Windows QR incompatible promotion;
- broad allocation-failure guarantees;
- repository-wide review-surface cleanup;
- hosted generated API publication, retained generated-doc artifacts, or
  committed generated HTML;
- shared-library or dynamic ABI support;
- portable performance, release readiness, external-library parity, or
  state-of-the-art status.

## Closure Target Checklist

| Residual | Closure target quality |
| --- | --- |
| Package-manager distribution | Provider-specific proof, support tier, docs, and guards must all align before promotion. |
| Windows selected freshness | Hosted artifact inspection, selected metadata, generated support tier, and non-claim docs must promote together. |
| Allocation failure | One owner at a time with deterministic failure, cleanup, retry, and focused gate proof. |
| Review surface | One selected high-risk cluster at a time with behavior-preservation evidence and ownership guards. |
| Benchmark freshness | One selected hosted lane at a time with exact bundle and threshold-free methodology unless thresholds are designed. |
| Windows QR comparison | Hosted Windows/MSVC proof and artifact inspection required before selected Windows metadata promotion. |
| Generated API publication | Future hosted/artifact/committed policy requires replacing local-only proof with matching publication proof. |
| Adoption/diagnostics | Future UX work must preserve support truth and claim guards. |
| Release/ABI | Product/platform policy must exist before release or ABI readiness claims. |
| State-of-the-art | External baselines, methodology, platform matrix, package provenance, and hosted evidence are prerequisites. |

## Day 12 Validation And Hygiene

| Check | Day 12 result |
| --- | --- |
| `git diff --check` | Passed after Day 12 edits. |
| Stale residual wording search | Passed. Stale pending-future-execution wording, stale Sprint 205 pending wording, old Sprint 197-only final claim language, and Day 12 pending-retrospective wording are absent from the residual queue and retrospective. |
| C/header quality gate | Not required. No `.c` or `.h` files are changed through Day 12. |
| Generated-output status | Existing ignored `docs/api/` output remains generated validation output and is not staged. |

## Completion Criteria Review

| Criterion | Result |
| --- | --- |
| Item 206.6 has a complete prioritized residual queue. | Met. `EPIC_18_RESIDUAL_QUEUE.md` now contains the prioritized Day 12 queue. |
| Residuals are actionable and not vague placeholders. | Met. Each near-term residual has owner surfaces, closure targets, expected evidence, validation commands, and claim boundaries. |
| Deferred claims remain visibly unearned until future evidence closes them. | Met. Broad package, Windows, ABI, hosted API, performance, release, and state-of-the-art claims remain explicit non-claims. |

## Day 12 Disposition

Day 12 is complete after patch hygiene and stale residual wording checks pass.
Day 13 should cross-check project-plan, retrospective, residual queue, working
notes, and public/maintainer claim surfaces for consistency before final
closeout.
