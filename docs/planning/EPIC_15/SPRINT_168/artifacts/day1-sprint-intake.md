# Sprint 168 Day 1: Sprint Intake And Performance Handoff

## Purpose

Day 1 establishes the Sprint 168 baseline from the active Epic 15 project plan
and the Sprint 167 claim-gate handoff. Sprint 168 owns one selected gap:
promote a bounded performance report into hosted, freshness-checked evidence
with methodology-bound claims.

## Source Artifact Note

The prompt referenced `docs/planning/EPIC_12/PROJECT_PLAN.md` and the section
title "Sprint 168: Hosted Performance Publication Date". The active merged
project plan is `docs/planning/EPIC_15/PROJECT_PLAN.md`, section "Sprint 168:
Hosted Performance Publication Lane".

This sprint proceeds from the Epic 15 section because Sprint 167 and PR #186
established Epic 15 as the current planning source.

## Inputs Reviewed

| Input | Source | Day 1 finding |
| --- | --- | --- |
| Sprint 168 project-plan section | `docs/planning/EPIC_15/PROJECT_PLAN.md` | Sprint 168 goal is to promote one selected performance report into hosted freshness-checked CI evidence. |
| Sprint 168 day plan | `docs/planning/EPIC_15/SPRINT_168/PLAN.md` | Day 1 requires intake, working notes, artifact setup, source note, non-claims, and stop conditions. |
| Claim gates | `../SPRINT_167/artifacts/day12-claim-gates.md` | G167-01 requires one family, matrix scope, platform, toolchain, command, runtime budget, report path, metadata, freshness check, and claim-safe docs. |
| Sprint 168 handoff | `../SPRINT_167/artifacts/day13-sprint-reconciliation.md` | Recommended starting candidate is `bench_refactor_csc` through `make bench-canonical-report`. |
| Sprint 167 closeout | `../SPRINT_167/artifacts/day14-sprint-closeout.md` | Sprint 168 should not claim portable superiority, broad backend superiority, platform parity, release proof, or state-of-the-art performance. |

## Selected Gap Carried Forward

| Field | Value |
| --- | --- |
| Gap ID | `G167-01` |
| Gap name | Hosted methodology-bound performance publication |
| Ledger rows | E15-014, E15-017, NC-003 |
| Sprint owner | Sprint 168 for hosted lane creation; Sprint 169 for methodology hardening |
| Closure mode | Promote one selected performance report into hosted freshness-checked evidence with scoped documentation. |

## Initial Candidate Lane

| Candidate field | Initial Day 1 value |
| --- | --- |
| Performance family | Direct repeated-run CSC factorization workflow |
| Benchmark binary | `build/bench_refactor_csc` |
| Primary command owner | `make bench-canonical-report` |
| Script owner | `scripts/bench_canonical_report.sh` |
| Generated output family | `build/bench-reports/canonical/` |
| Initial platform assumption | Linux hosted CI, unless Day 3 selects a better bounded lane |
| Claim boundary | One selected report scope only; no portable superiority or broad backend claim |

## Acceptance Expectations

Sprint 168 must select and document:

- benchmark family;
- matrix or fixture scope;
- hosted platform lane;
- compiler and toolchain;
- build flags and backend/thread settings;
- exact command;
- runtime budget;
- generated report path;
- methodology metadata fields;
- freshness check;
- artifact upload behavior if hosted;
- claim-safe public documentation.

## Retained Non-Claims

Sprint 168 retains these performance non-claims:

| Non-claim | Reason |
| --- | --- |
| Portable performance superiority | One hosted lane cannot prove performance across hardware, compilers, operating systems, backends, threads, or matrix families. |
| Broad backend superiority | A selected workflow may compare paths, but does not prove one backend is generally superior. |
| Broad matrix-family performance | The selected fixture or fixture subset bounds the evidence. |
| External-library performance parity | The selected lane is internal benchmark publication, not an external comparison study. |
| Release benchmark proof | Branch/PR hosted reports are not release-qualified evidence unless a release process owns them. |
| Cross-platform performance parity | A Linux hosted lane does not prove macOS or Windows performance. |
| State-of-the-art performance | Methodology-bound evidence is narrower than state-of-the-art positioning. |

## Day 1 Stop Conditions

| Stop condition | Required handling |
| --- | --- |
| Selected report cannot name platform, compiler, command, fixture, runtime budget, and metadata owners. | Keep the lane local/advisory until Day 3/Day 5 narrows the scope. |
| Runtime appears too long for hosted CI. | Narrow fixture/repeat scope before CI wiring. |
| Generated output includes unstable or unowned fields. | Define deterministic formatting or exclude the field from freshness checks. |
| Documentation implies broad performance superiority. | Rewrite to selected-scope evidence or retain the non-claim. |
| Source or public header files change. | Run `make format && make lint && make test` before completion. |
| Generated report artifacts are staged unintentionally. | Remove from staged changes and keep generated output under ignored paths. |

## Artifact Setup

| Path | Status |
| --- | --- |
| `docs/planning/EPIC_15/SPRINT_168/PLAN.md` | Present |
| `docs/planning/EPIC_15/SPRINT_168/WORKING_NOTES.md` | Created |
| `docs/planning/EPIC_15/SPRINT_168/artifacts/` | Created |
| `docs/planning/EPIC_15/SPRINT_168/artifacts/day1-sprint-intake.md` | Created |

## Day 2 Handoff

Day 2 should inventory the benchmark/report surface before final lane
selection. It should review:

- `Makefile` benchmark and report targets;
- `scripts/bench_canonical_report.sh`;
- related performance report or sentinel scripts;
- `benchmarks/README.md`;
- README performance wording;
- generated canonical report conventions and ignored output paths;
- existing freshness and report-index patterns that can be reused.

## Validation Notes

Day 1 changed only Sprint 168 planning artifacts. No `.c` or `.h` files were
modified, so the full C quality gate is not required for this day.

## Completion Check

| Criterion | Status | Evidence |
| --- | --- | --- |
| Sprint 168 scope is tied to the active Epic 15 project plan. | Complete | Source artifact note records the Epic 15 project-plan section and stale prompt path/title. |
| Sprint 167 acceptance gates are carried forward. | Complete | G167-01, SC-004, acceptance expectations, and the Sprint 168 handoff are recorded. |
| No hosted performance claim is made before a selected lane exists. | Complete | The candidate lane is labeled initial and bounded; hosted proof remains future work. |
