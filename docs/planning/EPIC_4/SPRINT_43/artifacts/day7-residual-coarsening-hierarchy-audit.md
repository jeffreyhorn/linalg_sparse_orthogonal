# Sprint 43 Day 7: Residual Coarsening / Hierarchy Audit

## Summary

Day 7 audited the post-Day-6 graph split to determine what still belongs in
the coarsening/hierarchy subsystem and what remains intentionally coupled to
later graph phases.

The main result is that the real coarsening extraction is already
substantially complete:

- `src/sparse_graph_coarsen.c` now owns the hierarchy lifecycle and the
  coarsening implementation
- `src/sparse_graph.c` no longer carries a large hidden coarsening core
- the remaining "coarsening-adjacent" logic is mostly:
  - coarse-bisection code that belongs to the Day 9 seam
  - uncoarsening/orchestration code that still depends on FM refinement and
    separator lifting
  - shared declarations/comment grouping cleanup ahead of the next extraction

That means Day 8 should stay bounded: finish the residual coarsening-facing
cleanup, not reopen FM or orchestration work.

## Files Reviewed

- `src/sparse_graph.c`
- `src/sparse_graph_coarsen.c`
- `src/sparse_graph_internal.h`
- `docs/planning/EPIC_4/SPRINT_43/PLAN.md`
- `docs/planning/EPIC_4/SPRINT_43/artifacts/day6-hierarchy-coarsening-extraction-batch1.md`

## Post-Day-6 Ownership Map

### 1. The coarsening module now owns the implementation seam it should own

`src/sparse_graph_coarsen.c` already contains the main hierarchy/coarsening
behavior:

- coarsening-strategy ownership and override helpers
- `graph_coarsen_with_strategy(...)`
- `graph_coarsen_heavy_edge_matching(...)`
- `graph_coarsen_hcc(...)`
- `sparse_graph_hierarchy_build(...)`
- `sparse_graph_hierarchy_free(...)`

Interpretation:

- the multilevel shrink path and its ownership transitions are already out of
  the monolith
- no second large coarsening implementation block remains stranded in
  `src/sparse_graph.c`

### 2. The remaining monolith now starts with coarse bisection, not hidden coarsening

The first major post-Day-6 region in `src/sparse_graph.c` is now:

- brute-force coarse bisection
- GGGP fallback bisection
- Laplacian builder
- spectral coarsest-bisection path
- coarsest-bisection strategy parsing/dispatch

Interpretation:

- this is the correct next extraction seam for Day 9
- it is not residual coarsening work and should not be forced into the
  coarsening module

### 3. The other coarsening-adjacent region is really uncoarsening/orchestration

The remaining lifecycle that still mentions hierarchy/coarse levels is:

- `graph_uncoarsen(...)`
- top-level retry/orchestration in `partition_once(...)`
- `sparse_graph_partition(...)`

Interpretation:

- this logic is still coupled to FM refinement, separator lifting, and
  top-level partition control flow
- it is correctly deferred from the Day 8 coarsening cleanup batch

## Residual Classification

### Ready for direct Day 8 cleanup

The remaining bounded cleanup work before coarse-bisection extraction is:

- consolidate coarsening-facing declaration grouping in
  `src/sparse_graph_internal.h`
- remove stale monolith-oriented wording in comments that still implies
  hierarchy/coarsening live entirely in `src/sparse_graph.c`
- keep the strategy/helper ownership map explicit so Day 9 can extract
  bisection without re-auditing the coarsening seam

This is small finishing work, not another large code move.

### Still coupled to FM refinement or separator lifting

These must remain out of the Day 8 batch:

- `graph_uncoarsen(...)`
- FM bucket/refinement support
- annealing / ensemble / thread-local FM strategy machinery
- separator lifting and final partition projection

Reason:

- they still compose directly with the coarsest-bisection result and the
  finest-level projection path
- moving them now would turn Sprint 43 Phase 1 into a broader graph cleanup
  sprint instead of a bounded decomposition sprint

### Runtime strategy glue better left for later

These also remain intentionally deferred:

- top-level retry routing around sep=0 cases
- broader environment-variable driven partition strategy glue
- FM-specific parser and strategy interpretation

Reason:

- they cross multiple graph phases
- they do not block the bounded coarse-bisection extraction planned next

## Interface Cleanup Needed Before Day 9

The main pre-bisection interface cleanup is straightforward:

- keep coarsening declarations grouped clearly in
  `src/sparse_graph_internal.h`
- keep coarse-bisection declarations separate enough that the future
  `src/sparse_graph_bisect.c` move is obvious
- avoid moving FM-only or separator-only declarations into shared ownership
  just because they currently sit near the remaining monolith code

The key point is that the next risk is header clarity, not missing
implementation extraction.

## Bounded Day 8 Target Set

Day 8 should now be treated as a completion/consolidation batch for the
Phase-1 coarsening seam:

- finish any residual coarsening-facing declaration cleanup
- tighten remaining comment/ownership wording around the extracted seam
- preserve the current split:
  - `src/sparse_graph_coarsen.c` owns hierarchy/coarsening
  - `src/sparse_graph.c` retains coarse bisection, FM, separator lifting, and
    orchestration until their planned days

Day 8 should **not**:

- move coarse-bisection code early
- move `graph_uncoarsen(...)`
- move FM refinement
- broaden into separator-lifting cleanup

## Day 7 Outcome

Sprint 43's post-Day-6 state is cleaner than the original monolith would
suggest:

- the coarsening/hierarchy extraction is already materially complete
- the remaining work before Day 9 is primarily interface and ownership
  consolidation
- the sprint can stay bounded away from FM/separator churn while still handing
  Day 9 a clearer coarse-bisection extraction seam
