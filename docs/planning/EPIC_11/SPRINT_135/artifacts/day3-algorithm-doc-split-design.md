# Sprint 135 Day 3 - Algorithm Doc Split Design

## Purpose

Day 3 designs the split between concise current algorithm reference material
and historical measurement appendix material before any large documentation
movement. The goal is to keep `docs/algorithm.md` useful as the current
reference while moving sprint-era measurement and decision history behind a
deliberate appendix route.

## Current State

`docs/algorithm.md` already opens with an adoption boundary:

- it is technical background, not the first-use guide;
- it points readers to README, solver selection, examples, and benchmark docs;
- it explicitly avoids acting as install/support, package, ABI, or portable
  performance proof.

The body still mixes several content classes:

| Content class | Current examples | Split pressure |
| --- | --- | --- |
| Current data-structure and algorithm reference | Orthogonal list, slab allocator, permutations, LU, solve, tolerances, Cholesky, iterative solvers, QR, SVD/eigs behavior. | Keep in current reference, possibly tighten headings later. |
| Current compressed-format reference | CSR/CSC compressed formats, transpose relationship, compressed-first implications. | Keep and link to future cookbook. |
| Current solver behavior and limitations | Iterative solvers, preconditioners, QR minimum-norm, eigensolver options, LOBPCG behavior. | Keep, but remove sprint-era rationale where it blocks scanning. |
| Historical bug-fix context | `Snapshot Mechanism (Bug 3.1 Fix)`, `Forward Substitution (Bug 3.3 Fix)`. | Summarize in current reference; move detailed chronology only if needed. |
| Historical measurement narratives | SuiteSparse fill comparisons, Pres_Poisson trajectories, Kuu recipes, sprint day benchmark references. | Move to historical appendix or summarize with benchmark-doc links. |
| Performance/report governance | Reorder/fill reporting interpretation, wall-check history, performance sentinel history. | Prefer benchmark/report docs as authority; move history to appendix. |
| Sprint-labeled implementation chronology | CSC LDLT Sprint 17/19, AMD/ND Sprint 22-28 closures, eigensolver Sprint 20/21 paragraphs. | Move or summarize so current behavior comes first. |

## Target File Responsibilities

| Target | Responsibility | Non-responsibility |
| --- | --- | --- |
| `docs/algorithm.md` | Concise current reference for data structures, solver algorithms, compressed formats, solver behavior, current options, complexity/limitations, and links to cookbook/examples. | Historical benchmark corpus narratives, sprint-by-sprint closure history, report-governance chronology, or portable performance claims. |
| `docs/algorithm_history.md` | Historical measurement appendix for sprint-era algorithm decisions, benchmark measurements, regression-gate rationale, historical advisories, retired targets, and links to planning artifacts. | First-use adoption guidance, current API contract ownership, install/package support truth, or benchmark pass/fail authority. |
| `benchmarks/README.md` | Authoritative benchmark command, local measurement, CSV/report, and generated-index interpretation surface. | Algorithm implementation reference or first-use solver cookbook. |
| `docs/solver_selection.md` | Current solver-choice router and adoption decision reference. | Algorithm history or detailed benchmark chronology. |
| `docs/maintainer_guide.md` | Maintainer support-tier and validation ownership authority. | First-use algorithm learning path. |

## Proposed Current Reference Structure

Keep `docs/algorithm.md` as the stable public path and reorganize toward this
scan order:

1. Orientation and links to first-use docs, cookbook/examples, benchmark docs,
   and historical appendix.
2. Core matrix representation:
   - orthogonal linked-list data structure;
   - slab allocator;
   - CSR/CSC compressed formats and compressed-first ownership notes.
3. Direct solver family:
   - LU, Cholesky, LDLT, QR, solve procedures, tolerances, and limitations;
   - reordering as current behavior, with historical measurement links.
4. Iterative solver family:
   - CG, GMRES, MINRES, BiCGSTAB;
   - stagnation detection, diagnostics, preconditioning.
5. Spectral and SVD family:
   - SVD and low-rank behavior;
   - eigensolver backends, shift-invert, thick-restart, LOBPCG, and result
     fields.
6. Analysis/refactorization and matrix operations:
   - symbolic analysis, numeric refactorization, SpMM, thread/OpenMP notes.
7. Current limitations and evidence-boundary links.

This structure intentionally keeps current behavior in the reference and moves
historical proof trails to the appendix.

## Proposed Historical Appendix Structure

Create `docs/algorithm_history.md` with this target structure:

1. Scope and non-claim boundary:
   - historical measurement context only;
   - not first-use guidance, current API contract, or portable performance
     proof.
2. Direct solver and factorization history:
   - LU bug-fix chronology if retained;
   - Cholesky CSC and supernodal measurement history;
   - CSC LDLT scaffold and supernodal LDLT sprint chronology.
3. Reordering and fill history:
   - AMD quotient-graph history;
   - ND Sprint 22-28 decision chronology;
   - retired Pres_Poisson 0.85 target;
   - advisory env-var recipes and cross-axis cautions.
4. Benchmark/report governance history:
   - reorder/fill reporting interpretation context;
   - wall-check and performance sentinel history;
   - generated report/index boundaries with links to benchmark docs and Sprint
     131 artifacts.
5. Eigensolver implementation history:
   - Sprint 20/21 Lanczos, thick-restart, shift-invert, LOBPCG rationale;
   - benchmark links and local measurement caveats.
6. Planning artifact links and freshness caveats.

## Section Move Map

| `docs/algorithm.md` section | Current-reference action | Historical appendix action |
| --- | --- | --- |
| Opening orientation | Keep and add historical appendix link. | Link back to current reference. |
| Orthogonal linked-list, allocator, permutation arrays | Keep. | None unless future history is needed. |
| LU factorization, solve, refinement, tolerances | Keep concise current mechanics. | Consider moving bug-number chronology only if Day 4 finds it noisy. |
| Cholesky factorization | Keep algorithm, complexity, and current limitations. | Move or summarize SuiteSparse no-reorder fill comparison and historical performance paragraphs. |
| CSC numeric Cholesky backend | Keep current layout and algorithm summary. | Move detailed performance/measurement narrative and sprint-specific benchmark links. |
| Supernodal detection and batched kernel | Keep current behavior summary. | Move sprint-day measurement and cross-check chronology. |
| CSC LDLT scaffolding, supernodal LDLT, row-adjacency index | Summarize current behavior and links. | Move sprint-labeled implementation chronology and benchmark impact. |
| Fill-reducing reordering | Keep current RCM, AMD, ND, COLAMD behavior and current knobs. | Move detailed Sprint 22-28 chronology, retired targets, fixture-specific recipes, and advisory sweep history. |
| Reorder/fill reporting interpretation and performance regression gates | Replace with compact pointer to `benchmarks/README.md`. | Move historical gate rationale and sentinel chronology if retained. |
| SpMM, CSR/CSC compressed formats, thread safety | Keep. | None. |
| Iterative solvers, diagnostics, preconditioners | Keep current behavior. | Move isolated SuiteSparse anecdote only if Day 4 classifies it as historical. |
| QR minimum-norm least squares | Keep current behavior. | None unless historical proof details are discovered. |
| Symmetric eigensolvers | Keep current API/backends and mathematical behavior. | Move Sprint 20/21 chronology, benchmark sweep links, and measured-memory examples where they interrupt current behavior. |

## Redirect and Link Plan

| Link surface | Required update |
| --- | --- |
| `README.md` documentation index | Keep existing `docs/algorithm.md` link valid. Optionally update description to "current algorithm reference" after implementation. |
| `docs/algorithm.md` top section | Add a short link to `docs/algorithm_history.md` once the appendix exists. |
| `docs/algorithm_history.md` top section | Link back to `docs/algorithm.md`, `docs/solver_selection.md`, `examples/README.md`, `benchmarks/README.md`, and `docs/maintainer_guide.md`. |
| `benchmarks/README.md` | No Day 3 change. Day 10 may add a concise report-index adoption link if needed. |
| `docs/solver_selection.md` | No Day 3 change. Day 11 may add cookbook/reference navigation after split implementation. |
| Planning artifact links inside moved text | Preserve as historical links; do not rewrite them into current support claims. |

## Anchor Compatibility Notes

Markdown-generated anchors may change if headings are renamed. Days 4-6 should
avoid renaming stable current-reference headings unless the link scan shows no
inbound users. When headings are moved:

- keep the same heading text in the appendix where practical;
- add a short pointer from the old current-reference neighborhood to the new
  appendix section;
- update any maintained docs that link directly to moved headings;
- record any intentionally broken planning-only historical anchor as a
  residual only if it is not worth preserving.

Current inbound scan found one maintained public file link to
`docs/algorithm.md` from the README documentation index and no maintained
heading-specific inbound links from README, INSTALL, docs, examples, or
benchmark README files.

## Bounded Implementation Plan

| Day | Scope | Exit condition |
| --- | --- | --- |
| Day 4 | Prepare split scope, create appendix shell if selected, confirm anchor and inbound-link plan. | Final list of sections to move in Batch 1 and Batch 2. |
| Day 5 | Move the highest-risk historical measurement blocks: Cholesky/CSC performance, LDLT sprint history, and reorder/ND historical narratives. | `docs/algorithm.md` reads as current reference for direct/reorder sections; appendix has moved history and backlinks. |
| Day 6 | Move or summarize remaining historical-heavy eigensolver/report-gate material, clean duplication, and run validation. | Selected split phase is complete and residual history work is explicit. |

If the Day 5 move proves too large, the bounded first phase should prioritize
the reorder/ND historical block because it is the longest and most likely to
bury current behavior.

## Validation Plan

| Validation | Purpose |
| --- | --- |
| `git diff --check` | Diff hygiene. |
| focused trailing-whitespace scan over touched docs and Sprint 135 artifacts | Markdown whitespace hygiene. |
| `rg -n "docs/algorithm.md|algorithm_history.md|#.*algorithm" README.md INSTALL.md docs examples benchmarks` | Link and inbound-reference scan. |
| `test -f docs/algorithm.md && test -f docs/algorithm_history.md` after appendix creation | Ensure retained and new targets exist. |
| `rg -n "portable performance|performance guarantee|shared-library|dynamic ABI|package-manager|reviewed Windows|supplemental" touched docs` | Claim-boundary drift scan. |
| `rg -n "bench_day|Sprint [0-9]|Pres_Poisson|SuiteSparse|wall-check|index.tsv" docs/algorithm.md` | Confirm historical-heavy terms are reduced or intentionally retained in concise current reference. |

No C/header validation is required for the split unless Days 4-6 unexpectedly
change `.c` or `.h` files. If that happens, run `make format && make lint &&
make test`.

## Claim Boundaries

- Moving measurement history does not make old benchmark rows current
  performance evidence.
- Summarizing current behavior does not create new solver, backend, external
  corpus, package, ABI, platform, or performance claims.
- `benchmarks/README.md` remains the authority for benchmark commands and
  report interpretation.
- Sprint 131 remains the authority for report-index freshness and
  generated-versus-curated semantics.
- Sprint 133-134 remain the authority for package, ABI, shared-library, and
  platform support boundaries.

## Completion Criteria

| Criterion | Status | Evidence |
| --- | --- | --- |
| Current-reference and historical-appendix responsibilities are separated. | Complete | Target file responsibility table and proposed structures define the split. |
| No historical performance detail is left as first-use adoption guidance by default. | Complete | Move map and redirect plan route historical measurement material to appendix/benchmark docs. |
| The implementation batch has a bounded edit plan and validation checklist. | Complete | Day 4-6 implementation plan and validation plan define the next work. |
