# Sprint 49 Day 11 Artifact: Final Residual Review

## Purpose

Revisit every top-level Epic 4 review finding from
`docs/planning/EPIC_4/reviews/review-codex-2026-05-21.md`, classify it against
the final landed state, and determine whether any true blocking gap remains
before the authoritative Day 13 validation sweep.

## Inputs Revisited

Primary review inputs:

- `docs/planning/EPIC_4/reviews/review-codex-2026-05-21.md`
- `docs/planning/EPIC_4/reviews/todo-codex-2026-05-21.md`

Later Epic 4 residual anchors:

- Sprint 42 lifecycle closeout
- Sprint 44 graph closeout
- Sprint 45 iterative residual audit and closeout
- Sprint 46 eigensolver residual audit and closeout
- Sprint 47 auxiliary-surface closeout
- Sprint 48 documentation/quality-contract closeout
- Sprint 49 Day 1-10 artifacts and working notes

## Final Classification Summary

The original review contained seven top-level findings.

Final dispositions:

- fixed:
  - Finding 2 — graph / nested-dissection monolith
  - Finding 3 — fragmented allocation/overflow hardening
  - Finding 4 — one-shot iterative/eigensolver workspaces
  - Finding 5 — over-distributed quality contract
  - Finding 6 — inconsistent benchmark/developer CLI surfaces
  - Finding 7 — overloaded README / missing maintainer-policy home
- accepted tradeoff:
  - Finding 1 — mutable `SparseMatrix` state remains on the compatibility
    surface
- intentionally deferred:
  - none of the original seven findings remains as an unowned Epic 4 deferral

## Finding-By-Finding Disposition

### Finding 1 — Mutable matrix-state model

Disposition:

- accepted tradeoff

Why:

- Epic 4 improved the lifecycle model materially, but it did not replace every
  compatibility-facing in-place factor API with new public factor handles
- the public compatibility surface still includes workflows where callers must
  respect original-matrix / factored-state distinctions

What Epic 4 did land:

- Sprint 42 internal factor-state scaffolding and shared state guards
- the analyze/factor bridge in `include/sparse_analysis.h`
- stronger lifecycle/cancellation documentation and misuse coverage
- Sprint 49 public repeated-run handles for iterative and eigensolver repeated
  work

Why this is an accepted tradeoff instead of unfinished work:

- the remaining mutable-state surface is now deliberate compatibility policy,
  not accidental drift
- replacing it wholesale would be broader public API redesign beyond Epic 4’s
  bounded closeout scope

### Finding 2 — Graph / nested-dissection monolith

Disposition:

- fixed

Evidence:

- Sprint 43/44 split the old graph host into:
  - `src/sparse_graph_core.c`
  - `src/sparse_graph_coarsen.c`
  - `src/sparse_graph_bisect.c`
  - `src/sparse_graph_refine.c`
  - `src/sparse_graph_separator.c`
- the residual `src/sparse_graph.c` is now `801` lines instead of the earlier
  3,500+-line monolith

### Finding 3 — Fragmented allocation/overflow hardening

Disposition:

- fixed

Evidence:

- shared internal helpers now exist in:
  - `src/sparse_alloc_internal.h`
  - `src/sparse_alloc_internal.c`
- the hardened helper/overflow pattern is now used broadly across the core
  library
- later benchmark/example/tooling surfaces were brought closer to the same
  safety standard during the Epic 4 auxiliary sweeps

Interpretation:

- Epic 4 centralized the maintained hardening pattern even though it did not
  attempt to erase every raw allocation call from every support surface

### Finding 4 — One-shot iterative/eigensolver workspaces

Disposition:

- fixed

Evidence:

- Sprint 45 landed reusable iterative workspace/state support
- Sprint 46 landed reusable eigensolver workspace/state support
- Sprint 49 exposed bounded public repeated-run handles and aligned the final
  benchmark/test/docs surfaces with that caller model

### Finding 5 — Over-distributed quality contract

Disposition:

- fixed

Evidence:

- Sprint 48 created `docs/maintainer_guide.md`
- README ownership was reduced and clarified
- policy interpretation now has a stable maintainer-facing home instead of
  repeated README prose

### Finding 6 — Inconsistent benchmark/developer CLI surfaces

Disposition:

- fixed

Evidence:

- Sprint 47 landed shared benchmark CLI parsing helpers
- `bench_main` now has checked parsing, explicit help/error handling, and
  reorder-mode/reporting parity

### Finding 7 — Overloaded README / no maintainer-policy home

Disposition:

- fixed

Evidence:

- Sprint 48 reduced README size and maintainer-policy density
- `docs/maintainer_guide.md` now owns the maintainer-facing policy layer
- tutorial/header/example/benchmark docs use clearer cross-reference boundaries

## Residual Queue After Review

No original Epic 4 review finding remains as a hidden unowned residual.

The real residual queue is now explicit:

- accepted tradeoff:
  - compatibility-facing mutable `SparseMatrix` / in-place factor APIs remain
    supported
- later non-goals rather than unfinished Epic 4 work:
  - broader public factor-handle redesign
  - larger tutorial/example modernization around explicit repeated-run handles
  - benchmark-framework redesign beyond the bounded Sprint 47/49 cleanup

## Blocking-Gap Check

Blocking gap status:

- none

Why:

- the only non-fixed top-level finding is an explicit accepted tradeoff
- the remaining six top-level findings have landed concrete remediation inside
  the validated Epic 4 sprint sequence
- no new repair batch is needed before the final validation sweep

## Sprint 49 Position After Day 11

Sprint 49 now enters the summary/validation closeout phase with:

1. every original Epic 4 review finding given a final disposition
2. no hidden blocker before Day 13
3. an explicit accepted-tradeoff boundary instead of an implied unfinished
   lifecycle migration

## Bottom Line

Day 11 closed the Epic 4 review loop honestly:

- six findings are fixed
- one finding is an accepted compatibility tradeoff
- no original finding remains silently deferred
- no true blocking gap remains before the final validation pass
