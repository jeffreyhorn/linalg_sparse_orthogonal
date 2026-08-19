# Sprint 169 Day 1: Methodology Intake

## Purpose

Day 1 establishes the Sprint 169 working baseline from the active Epic 15
project plan and the Sprint 168 closeout package. The sprint objective is to
harden the selected performance methodology without broadening claims beyond
the single selected hosted performance lane.

## Source Artifact Note

The prompt referenced `docs/planning/EPIC_12/PROJECT_PLAN.md`. The active
merged planning source for Sprint 169 is
`docs/planning/EPIC_15/PROJECT_PLAN.md`, section
"Sprint 169: Performance Methodology Hardening".

This mismatch is recorded so future artifacts can trace why Sprint 169 files
live under `docs/planning/EPIC_15/SPRINT_169/`.

## Sprint 169 Scope

Sprint 169 implements project-plan items 169.1 through 169.6:

| Item | Scope | Day 1 interpretation |
| --- | --- | --- |
| 169.1 Statistical Policy | Repeat-count, warmup, variance, and threshold policy. | Start from Sprint 168 `configured_repeat_1`, `warmup=not_recorded`, `variance=not_recorded`, and threshold-free publication rows. |
| 169.2 Report Schema Cleanup | Stable and diff-friendly generated performance rows. | Preserve selected/unselected row boundaries while evaluating matrix-size, sample, warmup, and variance fields. |
| 169.3 Regression Sentinel | Bounded sentinel for large regressions without universal speed claims. | Keep sentinel pass/fail behavior separate from selected threshold-free publication evidence. |
| 169.4 Documentation Indexing | Link selected performance report from report index and README evidence table. | Improve discoverability without checking in generated build output by accident. |
| 169.5 Platform Caveats | Exact platform and backend constraints. | Keep Linux hosted runner, compiler, build flags, CPU, build mode, and fixture caveats explicit. |
| 169.6 Quality Gate | Report generation, freshness checks, and relevant tests. | Run focused script/workflow/report checks unless `.c` or `.h` files change, in which case run the full C quality gate. |

## Sprint 168 Handoff Summary

Sprint 168 left a narrow selected performance lane:

| Field | Handoff value |
| --- | --- |
| Benchmark family | Direct repeated-run CSC factorization |
| Selected artifact row | `bench_refactor_csc` |
| Selected command | `tests/data/suitesparse/nos4.mtx --repeat 1` |
| Local freshness target | `make bench-canonical-report-freshness` |
| Hosted checker | `scripts/check_bench_canonical_freshness.py --mode hosted` |
| Hosted CI job | `Linux reviewed hosted selected performance freshness` |
| Hosted artifact | `sprint168-selected-performance-freshness` |
| Selected support tier | `hosted_selected` in hosted mode |
| Selected claim boundary | `hosted_selected_threshold_free` in hosted mode |
| Unselected row boundary | `local_only` / `local_threshold_free` |

Sprint 169 should harden that lane rather than reselecting a benchmark family.

## Methodology-Hardening Questions

| Question | Current baseline | Sprint 169 direction |
| --- | --- | --- |
| Repeat count | `configured_repeat_1` | Decide whether hosted publication should remain one configured repeat or require repeated samples. |
| Warmup | `not_recorded` | Decide whether to preserve literal non-measured metadata or add an explicit warmup policy. |
| Variance | `not_recorded` | Decide whether to preserve literal non-measured metadata or compute variance from repeated samples. |
| Matrix size | `not_recorded` | Decide whether selected-row fixture dimensions should be derived and recorded. |
| Threshold policy | `baseline=n/a`, `threshold=n/a` | Keep publication threshold-free unless a separate bounded sentinel is implemented. |
| Report indexing | Focused checker and docs | Decide whether selected performance evidence needs normalized report-index publication. |
| Hosted artifact review | CI artifact and summary lines | Inspect whether reviewer-facing metadata is sufficient after the first hosted lane. |

## Stop Conditions

Sprint 169 should stop and revise before proceeding if a change:

- applies hosted-selected metadata to unselected canonical rows;
- treats timing values as portable speed evidence;
- adds a regression threshold to the selected publication row instead of a
  separately scoped sentinel;
- leaves warmup, variance, repeat, sample, or matrix-size semantics
  underdefined after schema changes;
- makes generated `build/` output a source-controlled artifact by accident;
- describes Linux hosted evidence as macOS, Windows, package, ABI, or
  external-library parity;
- changes `.c` or `.h` files without running
  `make format && make lint && make test`.

## Retained Non-Claims

Sprint 169 retains the Sprint 168 non-claims:

- no portable performance superiority;
- no broad benchmark-family publication;
- no external-library performance parity;
- no package-manager, shared-library, dynamic ABI, or runtime-loader claim;
- no broad platform parity;
- no release benchmark proof;
- no state-of-the-art sparse linear algebra performance claim;
- no solver correctness claim from benchmark rows.

## Day 1 Deliverables

| Deliverable | Status | Evidence |
| --- | --- | --- |
| Sprint 169 working-notes baseline | Complete | `WORKING_NOTES.md` created. |
| Artifact directory structure | Complete | `artifacts/` created with this Day 1 artifact. |
| Source artifact note | Complete | Prompt path mismatch recorded in plan, working notes, and this artifact. |
| Sprint 168 handoff summary | Complete | Selected lane and methodology questions recorded above. |
| Methodology-hardening stop conditions | Complete | Stop conditions recorded above and in working notes. |
| Day 1 methodology-intake artifact | Complete | This file. |

## Validation

Day 1 changed planning artifacts only. No `.c` or `.h` files were modified, so
the full C quality gate is not required for this day.

Run after writing this artifact:

```sh
git diff --check
```

## Completion Check

| Criterion | Status | Evidence |
| --- | --- | --- |
| Sprint 169 scope is tied to the active Epic 15 project plan. | Complete | Source artifact note and project-plan item mapping are recorded. |
| Sprint 168 selected lane is carried forward without reopening selection. | Complete | Selected `bench_refactor_csc` lane is the Sprint 169 baseline. |
| No new portable performance or broad benchmark claim is introduced. | Complete | Retained non-claims and stop conditions preserve the Sprint 168 boundary. |
