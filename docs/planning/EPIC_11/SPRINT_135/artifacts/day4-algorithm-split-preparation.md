# Sprint 135 Day 4 - Algorithm Split Preparation

## Purpose

Day 4 prepares the algorithm-document split with minimal-risk file layout,
anchors, backlinks, movement batches, and validation requirements. It creates
the appendix target and confirms that Day 5-6 can move content without
guessing ownership boundaries.

## File Layout Preparation

| File | Day 4 status | Role |
| --- | --- | --- |
| `docs/algorithm.md` | Retained and lightly updated with an appendix pointer. | Stable current algorithm reference path. |
| `docs/algorithm_history.md` | Created as an appendix scaffold. | Historical measurement, sprint decision, benchmark/report governance, and planning-link appendix. |
| `benchmarks/README.md` | No Day 4 edit. | Benchmark command and local-measurement interpretation authority. |
| `docs/solver_selection.md` | No Day 4 edit. | Solver-choice and adoption routing authority. |
| `examples/README.md` | No Day 4 edit. | Maintained example index. |
| `docs/maintainer_guide.md` | No Day 4 edit. | Maintainer support-tier and validation ownership authority. |

The appendix scaffold includes these top-level sections:

- Scope and Claim Boundary
- Direct Solver and Factorization History
- Reordering and Fill History
- Benchmark and Report Governance History
- Eigensolver Implementation History
- Planning Artifact Links

## Anchor and Heading Inventory

| Area | Current-reference heading action | Appendix heading action |
| --- | --- | --- |
| Opening orientation | Keep `# Algorithm Description`; add appendix pointer. | Add `# Algorithm History and Measurement Appendix`. |
| Direct solver/factorization history | Keep current direct solver headings until content is moved. | Use `## Direct Solver and Factorization History`. |
| Reorder/fill history | Keep current reorder headings until Day 5 movement. | Use `## Reordering and Fill History`. |
| Benchmark/report history | Keep current reporting headings until Day 6 movement. | Use `## Benchmark and Report Governance History`. |
| Eigensolver history | Keep current eigensolver headings until Day 6 movement. | Use `## Eigensolver Implementation History`. |
| Planning links | Keep historical links where moved content needs them. | Use `## Planning Artifact Links`. |

Compatibility rule: do not rename stable `docs/algorithm.md` headings during
Days 5-6 unless the moved section keeps a nearby pointer or the inbound scan
confirms no maintained links target that heading.

## Inbound-Link Update Queue

| Source | Day 4 finding | Required action |
| --- | --- | --- |
| `README.md` | Documentation index links to `docs/algorithm.md`. | Keep valid; optionally update description after split to say current algorithm reference. |
| `INSTALL.md` | No adoption-critical algorithm heading link found. | No Day 4 action. |
| `docs/tutorial.md` | No direct algorithm heading dependency found in maintained scan. | No Day 4 action; revisit during navigation alignment. |
| `docs/solver_selection.md` | No direct algorithm heading dependency found in maintained scan. | No Day 4 action; revisit during cookbook/navigation alignment. |
| `examples/README.md` | No direct algorithm heading dependency found in maintained scan. | No Day 4 action. |
| `benchmarks/README.md` | Owns benchmark/report interpretation. | No Day 4 action; Day 10 may add report-index adoption links. |
| `docs/maintainer_guide.md` | Owns support-tier and validation policy. | No Day 4 action. |
| Planning artifacts and generated API docs | Many historical references mention `docs/algorithm.md`. | Do not chase planning/generated references during the adoption split; preserve current file path. |

## Full-Split Versus Bounded-Phase Decision

Selected: bounded first phase.

Rationale:

- `docs/algorithm.md` is long and has several independent historical regions;
  a full rewrite would mix movement, rewrite, anchor cleanup, cookbook routing,
  and benchmark/report claim checks in one risky batch.
- The stable public path can remain `docs/algorithm.md`, so the first phase
  can reduce historical density without changing the front-door route.
- The highest-friction content is concentrated enough for Days 5-6:
  Cholesky/CSC performance history, LDLT sprint history, AMD/ND chronology,
  benchmark/report gate history, and eigensolver sprint-history paragraphs.
- Cookbook and navigation integration are already owned by later Sprint 135
  days, so Day 5-6 should not overfit the final adoption flow.

## Day 5 Movement Queue

| Priority | Source region | Target appendix section | Current-reference replacement |
| --- | --- | --- | --- |
| 1 | Cholesky fill comparison and CSC backend performance measurement blocks. | Direct Solver and Factorization History. | Short current-behavior summary plus benchmark/history link. |
| 2 | Supernodal detection measurement and batched-kernel proof trail. | Direct Solver and Factorization History. | Short current-behavior summary. |
| 3 | CSC LDLT scaffolding, supernodal LDLT, and row-adjacency sprint chronology. | Direct Solver and Factorization History. | Short current-behavior summary plus history link. |
| 4 | AMD quotient-graph Sprint 22-24 chronology. | Reordering and Fill History. | Current AMD behavior, characteristics, and caveated history link. |
| 5 | ND Sprint 22-28 chronology and retired Pres_Poisson target. | Reordering and Fill History. | Current ND pipeline summary, current knobs, and caveated history link. |

## Day 6 Movement Queue

| Priority | Source region | Target appendix section | Current-reference replacement |
| --- | --- | --- | --- |
| 1 | Reorder/fill reporting interpretation and performance regression gate history. | Benchmark and Report Governance History. | Compact pointer to `benchmarks/README.md` and appendix. |
| 2 | Performance-sentinel and large-matrix guardrail history. | Benchmark and Report Governance History. | Compact current evidence-boundary note. |
| 3 | Eigensolver Sprint 20/21 chronology, benchmark sweep links, and measured-memory examples. | Eigensolver Implementation History. | Current backend/API behavior summary with history link. |
| 4 | Isolated historical anecdotes in iterative/preconditioner sections. | Direct Solver and Factorization History or Eigensolver Implementation History, as appropriate. | Current behavior summary only. |
| 5 | README documentation-index wording, if needed after movement. | Not applicable. | Optional description update from "Algorithm Description" to current reference wording. |

## Split Risks

| Risk | Mitigation |
| --- | --- |
| Broken anchors from moved headings. | Keep file-level path stable, preserve moved heading text in appendix where practical, and run inbound scans after movement. |
| Historical performance details still crowd the current reference. | Use Day 5-6 historical-heavy term scans to verify reduction or record intentional retention. |
| Benchmark rows become implied performance claims after movement. | Keep benchmark docs authoritative and keep appendix scope/non-claim boundary prominent. |
| Duplicated claims between current reference and appendix. | Current reference should summarize behavior; appendix should preserve evidence and chronology. |
| Planning/generated references create noise in scans. | Preserve `docs/algorithm.md` path and prioritize maintained public/adoption docs over generated or historical planning files. |
| Scope creep into cookbook or navigation work. | Leave cookbook and broader navigation changes to Days 7-11. |

## Validation Commands Prepared

Day 4 ran or prepared these checks:

```bash
test -f docs/algorithm.md && test -f docs/algorithm_history.md
rg -n "docs/algorithm.md|algorithm_history.md|algorithm.md|#.*algorithm" README.md INSTALL.md docs examples benchmarks
rg -n "^## |^### " docs/algorithm_history.md docs/algorithm.md
rg -n "bench_day|Sprint [0-9]|Pres_Poisson|SuiteSparse|wall-check|index.tsv" docs/algorithm.md docs/algorithm_history.md
```

Day 5-6 should also run:

```bash
git diff --check
rg -n "[[:blank:]]$" docs/algorithm.md docs/algorithm_history.md docs/planning/EPIC_11/SPRINT_135
rg -n "portable performance|performance guarantee|shared-library|dynamic ABI|package-manager|reviewed Windows|supplemental" docs/algorithm.md docs/algorithm_history.md
git diff --name-only -- "*.c" "*.h"
```

## Completion Criteria

| Criterion | Status | Evidence |
| --- | --- | --- |
| Target docs can be edited without guessing ownership boundaries. | Complete | File layout, appendix headings, and movement queues define owners. |
| Inbound links are known before movement begins. | Complete | Link scan found the maintained README file-level link and no maintained heading-specific blockers. |
| The selected split scope fits inside the remaining sprint budget. | Complete | Bounded first phase limits Days 5-6 to high-friction historical blocks and defers cookbook/navigation integration to later days. |
