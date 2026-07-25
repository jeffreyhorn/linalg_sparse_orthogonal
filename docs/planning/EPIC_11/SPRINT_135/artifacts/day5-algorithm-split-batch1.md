# Sprint 135 Day 5 - Algorithm Split Batch 1

## Purpose

Day 5 implements the first bounded algorithm-document split batch. It keeps
`docs/algorithm.md` as the current algorithm reference while moving the
highest-friction factorization and reorder history into
`docs/algorithm_history.md`.

## Implemented Current-Reference Changes

| Region | `docs/algorithm.md` result |
| --- | --- |
| Cholesky fill comparison | Replaced the SuiteSparse no-reorder table with a concise fill-behavior summary and appendix link. |
| CSC Cholesky performance | Replaced the full timing table and one-shot/analyze-once narrative with a current performance-boundary note and appendix link. |
| Supernodal Cholesky proof trail | Kept current detection and helper behavior; moved day-specific validation chronology behind appendix link. |
| CSC LDLT scaffold | Renamed current section to `CSC LDL^T Layout`; kept current layout/solve behavior and linked the old scaffold chronology to the appendix. |
| Supernodal LDLT | Removed sprint-day labels from the current heading and kept current LDLT-specific constraints. |
| Row-adjacency LDLT | Removed sprint-day labels and replaced benchmark-impact text with appendix link. |
| AMD quotient graph | Kept current variable-only quotient-graph behavior and moved Sprint 22-24 chronology to appendix. |
| Nested Dissection | Kept current pipeline and tuning knobs; moved Sprint 22-28 chronology, fixture recipes, and retired target narrative to appendix. |
| README docs index | Updated label from Algorithm Description to Algorithm Reference while preserving `docs/algorithm.md`. |

## Historical Appendix Expansion

| Appendix section | Added content |
| --- | --- |
| Direct Solver and Factorization History | Cholesky fill/CSC measurement summary, supernodal proof trail, CSC LDLT scaffold history, supernodal LDLT constraints, row-adjacency index history, and planning evidence links. |
| Reordering and Fill History | AMD quotient-graph chronology, ND Sprint 22-28 chronology, retired Pres_Poisson target, fixture-local caveats, and planning evidence links. |

## Link and Redirect Notes

| Link | Status |
| --- | --- |
| `README.md` -> `docs/algorithm.md` | Preserved; wording now says Algorithm Reference. |
| `docs/algorithm.md` -> `docs/algorithm_history.md` | Preserved via top-of-file pointer and section-specific links. |
| `docs/algorithm_history.md` -> current docs | Preserved via top-of-file links to README, solver selection, examples, benchmarks, current algorithm reference, and maintainer guide. |
| moved planning evidence links | Preserved in appendix summaries. |

## Deferred to Day 6

Day 6 should handle the remaining high-friction history classes:

- benchmark/report gate history under `Reorder/fill reporting interpretation`
  and `Performance regression gates`;
- performance-sentinel and large-matrix guardrail history;
- eigensolver Sprint 20/21 chronology, benchmark sweep links, measured-memory
  examples, and OpenMP reorthogonalization history;
- final duplication cleanup and historical-heavy term scan.

## Claim Boundary Review

- The current reference now summarizes behavior and links to historical
  context; it does not introduce new solver, backend, package, ABI, platform,
  or performance claims.
- The appendix explicitly frames moved measurements as historical,
  branch-local, fixture-specific, and configuration-sensitive evidence.
- Benchmark command and report interpretation authority remains
  `benchmarks/README.md`.
- Package, ABI, and platform support boundaries remain owned by Sprint
  133-134 truth and maintainer/install docs.

## Validation Plan

Day 5 should validate:

```bash
git diff --check
rg -n "[[:blank:]]$" README.md docs/algorithm.md docs/algorithm_history.md docs/planning/EPIC_11/SPRINT_135
test -f docs/algorithm.md && test -f docs/algorithm_history.md
rg -n "algorithm_history.md|docs/algorithm.md|Algorithm Reference" README.md docs/algorithm.md docs/algorithm_history.md
rg -n "portable performance|performance guarantee|shared-library|dynamic ABI|package-manager|reviewed Windows|supplemental" README.md docs/algorithm.md docs/algorithm_history.md docs/planning/EPIC_11/SPRINT_135
git diff --name-only -- "*.c" "*.h"
```

## Completion Criteria

| Criterion | Status | Evidence |
| --- | --- | --- |
| Moved content has exactly one clear primary home. | Complete | Current reference holds summaries; appendix holds historical evidence summaries and links. |
| First-use links point to concise current guidance first. | Complete | README continues to route to `docs/algorithm.md`, now labeled Algorithm Reference. |
| Historical material remains reachable without dominating adoption docs. | Complete | `docs/algorithm.md` links to appendix sections and appendix links back to current adoption/reference docs. |
