# Sprint 196 Day 10 Artifact: Prioritized Residual Queue

**Date:** 2026-09-03
**Sprint item coverage:** 196.6, with support for 196.1, 196.2, 196.3, and
196.5
**Day 10 goal:** Publish a prioritized next-epic residual queue with exact
closure targets, owner conditions, validation gates, long-horizon deferrals,
and links from the Epic 17 retrospective and project plan.

## Summary

Day 10 published
`docs/planning/EPIC_17/EPIC_17_RESIDUAL_QUEUE.md`. The queue turns the Day 3
triage into a durable next-epic handoff that can seed future planning without
re-auditing all Sprint 187-195 artifacts.

The queue keeps six near-term candidates at the top, separates
validation/tooling and documentation-only follow-ups, lists long-horizon
deferrals explicitly, and preserves the out-of-scope warning-hygiene note.

## Files Updated

| File | Update |
| --- | --- |
| `docs/planning/EPIC_17/EPIC_17_RESIDUAL_QUEUE.md` | Added the publishable Epic 17 residual queue. |
| `docs/planning/EPIC_17/EPIC_17_RETROSPECTIVE.md` | Replaced the residual preview owner note with a link to the published queue and removed the Day 10 open question. |
| `docs/planning/EPIC_17/PROJECT_PLAN.md` | Marked 196.6 Complete and added a closeout residual-queue link. |
| `docs/planning/EPIC_17/SPRINT_196/WORKING_NOTES.md` | Added Day 10 publication notes, priority rationale, boundaries, and validation results. |
| `docs/planning/EPIC_17/SPRINT_196/artifacts/day10-prioritized-residual-queue.md` | Added this Day 10 artifact. |

## Near-Term Queue

| Priority | Residual ID | Theme | Closure target |
| ---: | --- | --- | --- |
| 1 | E17-RQ-001 | Package-manager/Homebrew support blocker | Add approved license metadata, set exact Homebrew license identifier, rerun proof and guards, then update docs only to earned support. |
| 2 | E17-RQ-005 | Selected Cholesky Windows freshness promotion | Observe hosted Windows pass, inspect exact bundle, promote selected manifest metadata if supported, and recalibrate docs. |
| 3 | E17-RQ-022 | Additional allocation-failure owner | Select one owner and repeat the Sprint 195 invariant, harness, regression, gate, docs, and validation pattern. |
| 4 | E17-RQ-016 | Additional QR review-surface cluster | Select one QR cluster, extract helper ownership, preserve behavior, and add guard coverage. |
| 5 | E17-RQ-013 | Windows/macOS selected benchmark freshness | Add one hosted platform selected benchmark freshness lane with exact artifact and selected-row validation. |
| 6 | E17-RQ-006 | Windows QR incompatible freshness | Add MSVC/CMake proof, inspect hosted artifacts, and promote exact selected metadata only if evidence supports it. |

## Separated Follow-Ups

| Class | Queue IDs | Reason for separation |
| --- | --- | --- |
| Validation/tooling | E17-RQ-004, E17-RQ-009, E17-RQ-010, E17-RQ-017, E17-RQ-020, E17-RQ-024 | These improve evidence ownership or tooling, but they are not direct product/support closures unless selected by future work. |
| Documentation-only | E17-RQ-014, E17-RQ-018, E17-RQ-021 | These should ride along with adjacent evidence or API work unless a future sprint selects them explicitly. |
| Long-horizon | E17-RQ-002, E17-RQ-003, E17-RQ-007, E17-RQ-008, E17-RQ-011, E17-RQ-012, E17-RQ-015, E17-RQ-023, E17-RQ-025, E17-RQ-026 | These need broader product, platform, methodology, release, allocator, or research scope. |
| Out-of-scope historical note | E17-RQ-019 | Existing warning hygiene should be reproduced under current gates before planning a fix. |

## Claim Boundaries Preserved

- Package-manager/Homebrew support remains unclaimed until license metadata
  and proof evidence exist.
- Windows selected Cholesky remains guarded workflow evidence until hosted
  evidence and manifest metadata are promoted together.
- Selected comparisons remain fixture/target scoped.
- Selected performance remains methodology-bound, threshold-free, and
  non-portable.
- Selected reliability proof remains limited to selected owners.
- Shared-library support, dynamic ABI compatibility, release readiness, broad
  Windows parity, broad package-manager distribution, hosted generated API
  publication, portable performance, and unqualified state-of-the-art status
  remain non-claims.

## 196.6 Acceptance Evidence

| Completion criterion | Evidence |
| --- | --- |
| Item 196.6 has a publishable residual queue. | `docs/planning/EPIC_17/EPIC_17_RESIDUAL_QUEUE.md` now exists. |
| Residuals can seed future planning without re-auditing Epic 17 from scratch. | Each near-term residual includes owner surfaces, closure target, expected evidence, validation commands, and claim boundary. |
| Deferred work is not presented as completed or implicitly claimed. | Long-horizon, documentation-only, validation/tooling, and out-of-scope residuals are separated from near-term candidates. |

## Validation

- `git diff --check`
- `make docs-check`

Day 10 changed planning documentation only. No `.c` or `.h` files were
modified, so the full C quality gate is not required for this day.
