# Sprint 187 Retrospective

**Sprint:** 187 - Epic 17 Baseline, Gap Ledger & Acceptance Gates
**Duration:** 14 days (Days 1-14 landed on branch `sprint-187`)
**Status:** Complete

## Source Artifact Note

Sprint 187 was executed from the Epic 17 project-plan section for Sprint 187
and lives under `docs/planning/EPIC_17/SPRINT_187/` with its plan, working
notes, daily artifacts, closeout artifact, and retrospective in one package.

## Definition Of Done Checklist

- [x] Created Sprint 187 plan, working notes, artifact directory, daily
      artifacts, closeout artifact, and retrospective.
- [x] Converted Codex review findings into a source-controlled Epic 17 gap
      ledger with owner files, claim risks, candidate sprints, validation
      commands, and non-goals.
- [x] Reconciled Epic 16 residuals with the Epic 17 gap ledger without
      dropping any residual silently.
- [x] Ranked candidate gaps by value, feasibility, dependency pressure,
      validation clarity, and ability to close completely inside Epic 17.
- [x] Selected the exact Sprint 188 through Sprint 195 closure targets and
      retained broad state-of-the-art, ABI, package, platform, comparison,
      performance, generated API, and reliability non-goals.
- [x] Defined acceptance gates for package, Windows, comparison, performance,
      maintainability, adoption, and reliability work.
- [x] Mapped future changed surfaces to required validation commands, hosted
      evidence rules, local skip rules, and the mandatory full C gate for
      `.c` and `.h` edits.
- [x] Created implementation-ready handoffs for Sprints 188 through 195.
- [x] Produced a final closeout summary confirming project-plan item coverage,
      consistency, review readiness, and retrospective inputs.
- [x] Kept Sprint 187 planning-only, with no source, public header, script,
      workflow, package, generated-report, or generated API output changes.

## What Went Well

1. **The ledger made the review actionable.** Day 2 turned the Codex review and
   todo into 16 gap rows with owner surfaces, claim risks, candidate sprints,
   validation expectations, and retained non-goals.

2. **Residual traceability stayed intact.** Day 3 mapped every Epic 16 residual
   to an Epic 17 gap or retained long-horizon decision, so package, Windows,
   comparison, generated API, and review-surface work stayed auditable.

3. **The sprint selected complete closures instead of partial breadth.** Days
   5 and 6 focused Epic 17 on package proof, PowerShell validation, Windows
   report freshness decision, one comparison family, one performance lane, one
   maintainability cluster, adoption/API coherence, and one reliability owner.

4. **Acceptance gates were defined before implementation.** Days 7 through 11
   captured owner files, validation commands, hosted evidence requirements,
   artifact policies, stop conditions, and non-claims for every selected
   closure family.

5. **The quality surface map makes later review cheaper.** Day 12 gives future
   sprint work a surface-by-surface validation policy and preserves the full
   `make format && make lint && make test` requirement for any C/header change.

6. **Future sprint handoffs are ready.** Day 13 gives Sprints 188-195 a clear
   starting package with source artifacts, first implementation steps,
   validation, done state, dependency ordering, and open questions.

## What Didn't Go Well

1. **Several future selections remain intentionally unresolved.** Sprint 187
   prepared the selection criteria, but future sprints still need concrete
   choices for license metadata, Windows report lane, comparison family,
   benchmark lane, review-surface cluster, and reliability owner.

2. **The support story is still distributed.** README, INSTALL, maintainer
   docs, report docs, package docs, benchmark docs, public headers, sprint
   artifacts, and project plans all remain part of the claim boundary until
   Sprint 194 consolidates adoption truth.

3. **Hosted evidence remains a gating dependency.** Windows report freshness,
   hosted PowerShell validation, hosted performance evidence, and any hosted
   comparison claim cannot be promoted from planning evidence alone.

4. **Homebrew support is still blocked by a product decision.** Sprint 187
   defined the acceptance gate, but Sprint 188 must still decide the exact
   approved license identifier and root metadata strategy.

5. **The final plan is deliberately conservative.** Broad state-of-the-art
   positioning, storage replacement, shared-library support, dynamic ABI,
   broad Windows parity, and portable performance claims remain out of scope.

## Final Metrics

### Planning Outputs

| Metric | Sprint 187 close state |
| --- | ---: |
| Sprint plan files added | 1 |
| Working notes files added | 1 |
| Sprint daily artifacts added | 14 |
| Sprint retrospective files added | 1 |
| Epic 17 gap ledger rows | 16 |
| Epic 16 residuals reconciled | 6 |
| Future implementation handoffs | 8 |
| Future sprint closure packages | Sprint 188 through Sprint 195 |

### Closure Selection

| Future sprint | Selected closure target |
| --- | --- |
| Sprint 188 | Homebrew proof completion |
| Sprint 189 | PowerShell validation ownership |
| Sprint 190 | Windows selected report freshness decision |
| Sprint 191 | Bounded external comparison family |
| Sprint 192 | Methodology-bound performance evidence lane |
| Sprint 193 | Selected large review-surface reduction |
| Sprint 194 | Adoption and API coherence simplification |
| Sprint 195 | Selected reliability and failure-path proof |

### Changed Surface

| Metric | Sprint 187 close state |
| --- | ---: |
| Planning documentation files added | 17 |
| Source files changed | 0 |
| Public header files changed | 0 |
| Scripts changed | 0 |
| Workflows changed | 0 |
| Package files changed | 0 |
| Generated report files staged | 0 |
| Generated API files staged | 0 |

### Claim Governance

| Metric | Sprint 187 close state |
| --- | ---: |
| unqualified state-of-the-art claims added | 0 |
| broad external parity claims added | 0 |
| portable performance claims added | 0 |
| package-manager support claims added | 0 |
| shared-library support claims added | 0 |
| dynamic ABI claims added | 0 |
| broad Windows parity claims added | 0 |
| hosted generated API claims added | 0 |
| broad allocation-failure coverage claims added | 0 |

## Closed Claim

Sprint 187 closes this Epic 17 baseline claim:

The post-Epic-16 baseline has been converted into a source-controlled Epic 17
gap ledger, Epic 16 residuals have been reconciled, exact Sprint 188-195
closure targets have been selected, acceptance gates and quality-surface
validation policies have been defined, and implementation handoffs are ready
for the next eight sprints.

This claim is supported by:

- [PLAN.md](./PLAN.md);
- [WORKING_NOTES.md](./WORKING_NOTES.md);
- [day1-baseline-intake.md](./artifacts/day1-baseline-intake.md);
- [day2-review-intake-matrix.md](./artifacts/day2-review-intake-matrix.md);
- [day3-residual-reconciliation.md](./artifacts/day3-residual-reconciliation.md);
- [day4-owner-surface-inventory.md](./artifacts/day4-owner-surface-inventory.md);
- [day5-gap-ranking-and-feasibility.md](./artifacts/day5-gap-ranking-and-feasibility.md);
- [day6-closure-target-selection.md](./artifacts/day6-closure-target-selection.md);
- [day7-package-acceptance-gates.md](./artifacts/day7-package-acceptance-gates.md);
- [day8-windows-acceptance-gates.md](./artifacts/day8-windows-acceptance-gates.md);
- [day9-comparison-performance-gates.md](./artifacts/day9-comparison-performance-gates.md);
- [day10-maintainability-reliability-gates.md](./artifacts/day10-maintainability-reliability-gates.md);
- [day11-adoption-documentation-gates.md](./artifacts/day11-adoption-documentation-gates.md);
- [day12-quality-surface-map.md](./artifacts/day12-quality-surface-map.md);
- [day13-implementation-handoffs.md](./artifacts/day13-implementation-handoffs.md);
- [day14-closeout-summary.md](./artifacts/day14-closeout-summary.md).

No new solver behavior, public API contract, package-manager support claim,
shared-library support, dynamic ABI guarantee, Windows report freshness lane,
broad Windows parity, broad external parity, portable performance claim,
hosted generated API publication path, broad reliability proof, release
readiness claim, or unqualified state-of-the-art claim was added.

## Next-Sprint Readiness

Future sprint planning should start from
[day13-implementation-handoffs.md](./artifacts/day13-implementation-handoffs.md)
and the relevant acceptance-gate artifact.

| Future need | Sprint 187 handoff |
| --- | --- |
| Homebrew proof completion | Resolve exact license metadata, harden formula proof, run package guards, and calibrate package docs. |
| PowerShell validation ownership | Add an owned validation command with local `pwsh` skip semantics and hosted Windows evidence. |
| Windows report freshness | Promote one Windows-safe lane or renew formal deferral with stronger guards and revisit criteria. |
| Bounded external comparison | Select one family, fixture, reference path, metric, tolerance, dependency policy, and freshness proof. |
| Performance evidence | Promote one methodology-bound hosted lane with complete metadata and no portable performance claim. |
| Review-surface reduction | Select exactly one large cluster, preserve behavior, add a focused guard, and run the full C gate. |
| Adoption/API coherence | Consolidate support truth after package, Windows, comparison, and performance outcomes land. |
| Reliability proof | Select one owner, add deterministic failure injection, prove cleanup/retry behavior, and avoid overlap with Sprint 193. |

## Validation Retrospective

Sprint 187 modified planning documentation only. Documentation hygiene checks
were sufficient for the sprint itself, and the final validation policy for
future implementation work is captured in
[day12-quality-surface-map.md](./artifacts/day12-quality-surface-map.md).

Any future `.c` or `.h` change must run:

```sh
make format
make lint
make test
```

## Carry Forward

- Sprint 188 must answer the license metadata question before claiming a
  passing Homebrew formula proof.
- Sprint 189 must keep PowerShell validation separate from Windows report
  freshness promotion.
- Sprint 190 must avoid ambiguous middle states: promote one proven freshness
  lane or renew the deferral.
- Sprint 191 and Sprint 192 must keep comparison credibility and performance
  evidence separate from broad parity or state-of-the-art claims.
- Sprint 193 and Sprint 195 must coordinate owner selection if they touch the
  same source or test cluster.
- Sprint 196 should calibrate final Epic 17 claims from actual earned evidence,
  not from planned handoffs.
