# Sprint 138 Day 1 - Scope & Corpus Contract Setup

## Purpose

Day 1 converts the Sprint 137 handoff artifacts into a bounded Sprint 138
implementation package. It creates the working-notes baseline, artifact
directory, inherited-input map, day-level ownership map, validation
expectations, and sprint-level non-claim register.

This is a documentation-only setup artifact. It does not implement the corpus
taxonomy, storage layout, oracle schema, deterministic fixture lane, optional
data semantics, validation command, or public documentation updates.

## Sprint 138 Scope

Sprint 138 implements seven project-plan items:

| Item | Project-plan name | Day owner |
| --- | --- | --- |
| 1 | Fixture Taxonomy Design | Days 1-3 |
| 2 | Corpus Storage Layout | Days 4-5 |
| 3 | Oracle Row Schema | Days 6-7 |
| 4 | First Corpus Lane Implementation | Days 8-10 |
| 5 | Skip and Optional Data Semantics | Day 11 |
| 6 | Focused Validation | Day 12 |
| 7 | Documentation and Handoff | Days 13-14 |

Day 1 completion means later Sprint 138 days can begin taxonomy and storage
work without redoing Sprint 137 gap selection or redefining claim boundaries.

## Artifact Structure

| Path | Role |
| --- | --- |
| `docs/planning/EPIC_12/SPRINT_138/PLAN.md` | Day-by-day Sprint 138 execution plan. |
| `docs/planning/EPIC_12/SPRINT_138/WORKING_NOTES.md` | Sprint goal, constraints, handoff inventory, day ownership, validation expectations, non-claims, and day notes. |
| `docs/planning/EPIC_12/SPRINT_138/artifacts/day1-scope-corpus-contract-setup.md` | This Day 1 scope and setup artifact. |
| `docs/planning/EPIC_12/SPRINT_138/artifacts/day2-fixture-taxonomy-draft.md` | Planned Day 2 maintained matrix-class taxonomy draft. |
| `docs/planning/EPIC_12/SPRINT_138/artifacts/day3-taxonomy-review-claim-boundaries.md` | Planned Day 3 taxonomy review, promotion gates, and claim boundaries. |
| `docs/planning/EPIC_12/SPRINT_138/artifacts/day4-corpus-storage-layout-design.md` | Planned Day 4 corpus storage, manifest, optional-data, expected-result, and report path design. |
| `docs/planning/EPIC_12/SPRINT_138/artifacts/day5-corpus-storage-layout-implementation.md` | Planned Day 5 corpus directory and manifest skeleton implementation. |
| `docs/planning/EPIC_12/SPRINT_138/artifacts/day6-oracle-row-schema-design.md` | Planned Day 6 oracle row schema and comparison semantics design. |
| `docs/planning/EPIC_12/SPRINT_138/artifacts/day7-oracle-schema-implementation.md` | Planned Day 7 oracle schema implementation and validation helper. |
| `docs/planning/EPIC_12/SPRINT_138/artifacts/day8-deterministic-fixture-lane-design.md` | Planned Day 8 first deterministic fixture lane design. |
| `docs/planning/EPIC_12/SPRINT_138/artifacts/day9-first-corpus-lane-implementation.md` | Planned Day 9 first corpus lane implementation. |
| `docs/planning/EPIC_12/SPRINT_138/artifacts/day10-maintained-oracle-report-command.md` | Planned Day 10 maintained oracle/report command implementation. |
| `docs/planning/EPIC_12/SPRINT_138/artifacts/day11-optional-data-skip-semantics.md` | Planned Day 11 optional-data skip/defer semantics implementation. |
| `docs/planning/EPIC_12/SPRINT_138/artifacts/day12-focused-validation-quality-gates.md` | Planned Day 12 validation and quality-gate evidence. |
| `docs/planning/EPIC_12/SPRINT_138/artifacts/day13-documentation-sprint139-handoff.md` | Planned Day 13 corpus documentation and Sprint 139 QR handoff. |
| `docs/planning/EPIC_12/SPRINT_138/artifacts/day14-closeout.md` | Planned Day 14 Sprint 138 closeout, residuals, validation summary, and Sprint 139 readiness. |

## Sprint 137 Handoff Inventory

| Source | Day 1 reading | Sprint 138 use |
| --- | --- | --- |
| `docs/planning/EPIC_12/SPRINT_137/artifacts/day7-gap-selection-decision.md` | Selects maintained corpus/oracle contract with one deterministic lane as Sprint 138 target. | Fixes scope before taxonomy work starts. |
| `docs/planning/EPIC_12/SPRINT_137/artifacts/day8-corpus-oracle-evidence-templates.md` | Defines fixture, generated-matrix, optional-data, oracle row, and failure templates. | Controls Sprint 138 schema and row semantics. |
| `docs/planning/EPIC_12/SPRINT_137/artifacts/day9-report-index-freshness-templates.md` | Defines row metadata and freshness expectations for later report normalization. | Keeps Sprint 138 oracle/report rows compatible with Sprint 141. |
| `docs/planning/EPIC_12/SPRINT_137/artifacts/day11-quality-surface-map.md` | Defines required validation by touched surface and full C quality gate for `.c`/`.h` edits. | Controls validation selection throughout Sprint 138. |
| `docs/planning/EPIC_12/SPRINT_137/artifacts/day12-public-claim-freeze.md` | Freezes state-of-the-art, parity, package, platform, report, coverage, dead-code, and performance claims. | Blocks unsupported public wording while the corpus lane is implemented. |
| `docs/planning/EPIC_12/SPRINT_137/artifacts/day13-handoff-synthesis-sprint138-readiness.md` | Publishes Sprint 138 checklist, stop conditions, and later-sprint dependencies. | Provides implementation checklist and stop conditions. |
| `docs/planning/EPIC_12/SPRINT_137/artifacts/day14-closeout-and-sprint138-readiness.md` | Publishes final readiness criteria and residual register. | Confirms Sprint 138 can begin without redoing baseline or gap selection. |
| `docs/planning/EPIC_12/SPRINT_137/RETROSPECTIVE.md` | Summarizes selected Epic 12 targets and Sprint 138 handoff. | Provides closeout context for scope and residuals. |

## Initial Validation Expectation Register

| Touched surface | Required validation |
| --- | --- |
| Sprint 138 planning artifacts only | `git diff --check`, trailing-whitespace scan under `docs/planning/EPIC_12/SPRINT_138`, and focused Markdown link/path validation under `docs/planning/EPIC_12`. |
| Corpus manifests, schemas, oracle rows, or generated indexes | Schema/field validation when implemented, `git diff --check`, corpus/report non-claim scan, and focused corpus command if available. |
| Public or maintainer docs | `git diff --check`, focused Markdown link/path validation, and claim-boundary scan against Sprint 137 Day 12. |
| Python scripts | `python3 -m py_compile <script>` plus focused script command where feasible. |
| Shell scripts | `bash -n <script>` plus focused script command where feasible. |
| Makefile, CMake, install, or package metadata | Relevant build/package/install proof plus static/shared support-boundary review. |
| CI workflows | Workflow structural review and hosted-runner support-tier notes; do not report unrun hosted lanes as passed local evidence. |
| `.c` or `.h` files | `make format && make lint && make test` after focused tests needed for the touched behavior. |

## Sprint-Level Non-Claims

| Non-claim | Reason |
| --- | --- |
| Broad corpus completeness | Sprint 138 closes one durable corpus lane before broad fixture volume. |
| Broad SuiteSparse or external-library parity | Optional external data remains skip/defer-gated and fixture-local. |
| QR residual closure | Sprint 138 prepares Sprint 139 inputs; QR behavior closes in Sprint 139. |
| Partial-SVD residual closure | Sprint 138 prepares Sprint 140 inputs; partial-SVD behavior closes in Sprint 140. |
| Report index as release proof | Sprint 138 rows may feed Sprint 141 freshness work, not product release proof. |
| Portable performance or backend parity | Corpus/oracle rows are numerical evidence, not performance evidence. |
| Package, ABI, loader, or package-manager support | Corpus implementation does not affect Sprint 143 package decisions. |
| Platform parity | Corpus implementation does not promote macOS, Windows, or staged Windows lanes. |
| State-of-the-art status | A first corpus lane is prerequisite evidence, not a final competitive claim. |

## Day 1 Completion Criteria

| Criterion | Status | Evidence |
| --- | --- | --- |
| Every Sprint 138 project-plan item has a day-level owner. | Complete | Sprint 138 scope table above and `WORKING_NOTES.md` day-level ownership table. |
| Sprint 137 evidence contracts are visible before implementation begins. | Complete | Sprint 137 handoff inventory lists the selected target, corpus/oracle templates, report templates, quality map, claim freeze, and readiness artifacts. |
| Validation and public-claim boundaries are documented before files change. | Complete | Initial validation expectation register and sprint-level non-claims define required checks and blocked claims. |
