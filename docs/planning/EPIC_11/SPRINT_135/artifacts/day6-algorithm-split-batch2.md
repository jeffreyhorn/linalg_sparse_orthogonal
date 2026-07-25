# Sprint 135 Day 6 - Algorithm Split Batch 2

## Purpose

Day 6 completes the bounded algorithm split phase selected on Day 4. It moves
the remaining high-friction benchmark/report and eigensolver history out of
`docs/algorithm.md`, expands `docs/algorithm_history.md`, and records residual
algorithm-doc work for later Sprint 135 days.

## Implemented Current-Reference Changes

| Region | `docs/algorithm.md` result |
| --- | --- |
| Performance regression gates | Replaced long wall-check baseline chronology with concise current gate summary plus links to benchmark docs and appendix. |
| Symmetric eigensolver heading | Removed sprint label from the current-reference heading. |
| Symmetric eigensolver introduction | Reframed as current grow-m Lanczos, thick-restart Lanczos, and LOBPCG public backend behavior. |
| OpenMP reorthogonalization | Replaced sprint-day and measurement detail with current parallelism model and appendix link. |
| Thick-restart memory discussion | Preserved current memory-bound semantics while removing fixture-specific measured-memory detail. |
| Shift-invert route | Preserved current LDLT factor route while removing sprint dispatch chronology. |
| Convergence heuristics | Preserved qualitative behavior; moved fixture-specific SuiteSparse measurement links to appendix. |
| LOBPCG introduction | Reframed as current backend availability and motivation instead of sprint rollout chronology. |

## Historical Appendix Expansion

| Appendix section | Added content |
| --- | --- |
| Benchmark and Report Governance History | Reorder/fill report interpretation, wall-check history, performance-sentinel context, and Sprint 131 report-index boundary. |
| Eigensolver Implementation History | Backend rollout, OpenMP reorthogonalization history, thick-restart/shift-invert history, benchmark sweep links, and LOBPCG history. |

## Cross-Link Status

| Link | Status |
| --- | --- |
| `docs/algorithm.md` -> `benchmarks/README.md` | Current performance-gate section points to benchmark/report authority. |
| `docs/algorithm.md` -> `docs/algorithm_history.md#benchmark-and-report-governance-history` | Added through the performance-gate summary. |
| `docs/algorithm.md` -> `docs/algorithm_history.md#eigensolver-implementation-history` | Added through eigensolver introduction, OpenMP, and convergence sections. |
| `docs/algorithm_history.md` -> planning evidence | Added for wall-check, ND variance, Lanczos, backend sweep, and LOBPCG comparison captures. |
| README -> `docs/algorithm.md` | Preserved from Day 5 as Algorithm Reference. |

## Residual Algorithm Docs Queue

| Residual | Owner day | Reason |
| --- | --- | --- |
| Broad reference reordering | Day 11 or later | Day 5-6 intentionally reduced historical density without reordering every current-reference family. |
| Cookbook integration links | Days 7-11 | Compressed-first cookbook paths should be designed before adding workflow-specific links throughout the reference. |
| Remaining isolated historical anecdotes | Day 12 validation or residual queue | Day 5-6 targeted highest-friction blocks; small anecdotes can be handled during claim/link validation if they still distract. |
| Generated report adoption wording | Day 10 | Report-index adoption language belongs with benchmark/report docs, not inside the algorithm reference. |

## Claim Boundary Review

- `docs/algorithm.md` remains a current technical reference, not first-use
  adoption, install/package support, ABI, platform, benchmark, or performance
  guarantee documentation.
- `docs/algorithm_history.md` explicitly keeps moved measurements historical,
  branch-local, fixture-specific, and configuration-sensitive.
- `benchmarks/README.md` remains the current benchmark/report interpretation
  authority.
- Sprint 131 remains the report-index freshness and generated-versus-curated
  boundary.
- Sprint 133-134 remain the package, ABI, shared-library, and platform support
  boundary.

## Validation Plan

Day 6 should validate:

```bash
git diff --check
rg -n "[[:blank:]]$" README.md docs/algorithm.md docs/algorithm_history.md docs/planning/EPIC_11/SPRINT_135
test -f docs/algorithm.md && test -f docs/algorithm_history.md && test -f benchmarks/README.md
rg -n "algorithm_history.md|benchmarks/README.md|Algorithm Reference" README.md docs/algorithm.md docs/algorithm_history.md
rg -n "portable performance|performance guarantee|shared-library|dynamic ABI|package-manager|reviewed Windows|supplemental" README.md docs/algorithm.md docs/algorithm_history.md docs/planning/EPIC_11/SPRINT_135
rg -n "bench_day|Sprint [0-9]|Pres_Poisson|SuiteSparse|wall-check|index.tsv" docs/algorithm.md docs/algorithm_history.md
git diff --name-only -- "*.c" "*.h"
```

## Completion Criteria

| Criterion | Status | Evidence |
| --- | --- | --- |
| The selected split scope is implemented end to end. | Complete | Day 5 moved factorization/reorder history; Day 6 moved report-gate and eigensolver history. |
| Link targets are coherent from adoption and maintainer entry points. | Complete | README still routes to `docs/algorithm.md`; current reference routes to benchmark docs and appendix. |
| Residual algorithm docs work is explicit rather than hidden in mixed-purpose pages. | Complete | Residual queue names reference reordering, cookbook integration, isolated anecdotes, and report adoption wording. |
