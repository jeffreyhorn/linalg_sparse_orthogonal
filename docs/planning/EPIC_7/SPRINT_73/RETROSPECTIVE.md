# Sprint 73 Retrospective

**Sprint:** 73 — Configuration Surface Modernization Phase 2  
**Duration:** 14 days (Days 1-14 landed on this branch)  
**Status:** Complete

## Definition Of Done Checklist

- [x] Sprint 73 scope, residual-control hotspot map, and validation baseline were fixed before implementation work began
- [x] the strongest live residual configuration lanes were re-ranked from the repo instead of treated as one generic env-var bucket
- [x] the first landing stayed bounded to graph/FM policy integration and did not widen into a broad graph-family redesign
- [x] recognized `SPARSE_FM_*` compatibility controls now parse once at the graph orchestration boundary and lower into one internal typed FM policy/runtime contract
- [x] `src/sparse_graph_refine.c` no longer behaves like a second independent FM parser
- [x] `SPARSE_HCC_DEBUG` and `SPARSE_ND_PROFILE` now each resolve through one explicit internal precedence seam
- [x] focused precedence proof landed in the right owners:
  - `tests/test_graph.c`
  - `tests/test_reorder_nd.c`
- [x] maintained policy wording now states the narrower compatibility vs internal-policy vs developer-only split directly
- [x] proof-owner alignment was closed without redundant regression work
- [x] the full Sprint 73 branch passed the standard code-day gate, the strongest reviewed baseline, and the focused graph/reorder, example, benchmark, and install follow-ons
- [x] Sprint 73 closed with one explicit residual-configuration package and a ranked Sprint 74 carry-forward queue

## What Went Well

1. **Sprint 73 reduced real residual configuration contradictions instead of only documenting them.**
   The branch landed substantive ownership convergence in:
   - `src/sparse_graph.c`
   - `src/sparse_graph_refine.c`
   - `src/sparse_graph_internal.h`
   - `src/sparse_graph_coarsen.c`
   - `src/sparse_reorder_nd.c`
   - `src/sparse_reorder_nd_internal.h`
   and tied those changes to focused proof in:
   - `tests/test_graph.c`
   - `tests/test_reorder_nd.c`

2. **The first implementation lane stayed properly bounded.**
   Sprint 73 did not collapse into:
   - a generic env-var purge
   - a broad graph-family cleanup
   - public capability widening disguised as configuration cleanup
   - platform/install/reviewed-contract drift
   - product-model or backend redesign spill
   That kept the work aligned with the Sprint 70 architecture contract.

3. **The FM-family lane is now materially clearer.**
   The key Day 6 ownership rule is now explicit:
   - recognized `SPARSE_FM_*` compatibility env vars parse once at the graph orchestration boundary
   - they lower into one internal typed FM policy/runtime contract
   - refinement consumes lowered runtime state instead of re-parsing a second public-ish control surface
   - developer-only FM debug flags remain intentionally internal

4. **The strongest developer-only precedence seams were improved without widening the public API.**
   The Day 9 batch gave both residual profile/debug lanes one explicit internal owner:
   - `SPARSE_HCC_DEBUG`
   - `SPARSE_ND_PROFILE`
   and proved:
   - env-set default remains visible when no override is active
   - explicit internal override wins while active
   - clearing the override restores env-driven behavior
   - explicit internal enable/disable also works with the env unset

5. **Docs and proof ownership stayed aligned with the landed code.**
   Sprint 73 correctly followed through in:
   - `docs/maintainer_guide.md`
   without reopening:
   - `README.md`
   - `INSTALL.md`
   - `docs/tutorial.md`
   - `examples/README.md`
   - `benchmarks/README.md`
   - `include/sparse_analysis.h`

6. **The validated close state is strong.**
   Sprint 73 ended with:
   - `make format` passed
   - `make lint` passed
   - `make test` passed
   - `make quality-review-full` passed
   - reviewed CMake parity still exact at `53`
   - Makefile/CMake parity still `53 vs 53`
   - reviewed CMake `ctest` still `53 / 53`
   - focused graph/reorder proof owners revalidated explicitly
   - install/package regressions still clean

## What Didn't Go Well

1. **Sprint 73 only resolves the highest-value residual configuration seams.**
   The branch narrowed the FM-family lane and the strongest debug/profile seams, but it did not yet resolve:
   - the broader ND compatibility/default-policy convergence queue
   - support-only `SPARSE_QG_PROFILE` follow-through
   - later SVD-routing compatibility cleanup

2. **The public typed configuration surface remains intentionally narrow.**
   That is the correct Sprint 73 outcome, but it means Epic 7 still carries real deferred work around:
   - index-width modernization
   - later scalar-surface work
   - later capability expansion

3. **Some residual controls remain consciously support-only rather than structurally improved.**
   `SPARSE_QG_PROFILE` is still explicitly deferred, and Sprint 73 deliberately did not pretend that this lane was solved just because the stronger FM/graph and ND/profile seams moved first.

4. **Runtime asymmetry in the reviewed suite remains visible.**
   The full reviewed path passed, but `test_reorder_nd` still dominated reviewed CMake time. That is not a Sprint 73 blocker, but it remains operational friction for later proof-heavy graph/reorder work.

5. **The branch depended on disciplined non-moves.**
   Sprint 73’s success required not reopening public product docs, headers, platform claims, or broader graph-family work unless the landed code truly forced it. That discipline held, but it means later sprints still need to preserve the same fence.

## Final Metrics

### Validation and reviewed anchors

| Metric | Sprint 73 close state |
|---|---:|
| standard code-day gate | `make format && make lint && make test` passed |
| strongest reviewed baseline | `make quality-review-full` passed |
| reviewed CMake `ctest -N` anchor | `53` |
| Makefile/CMake parity | `53 vs 53` |
| reviewed CMake `ctest` | `53 / 53` |
| reviewed CMake total time | `421.78 sec` |
| reviewed `test_reorder_nd` time | `294.22 sec` |
| install regression | `11 / 11` |
| CMake install regression | `13 / 13` |

### Sprint 73 artifact package

| Metric | Sprint 73 close state |
|---|---:|
| total artifact files under `SPRINT_73/artifacts/` | `15` |
| baseline/audit artifacts | `6` |
| design/landing artifacts | `6` |
| review/closeout artifacts | `3` |

Notes:

- baseline/audit artifacts:
  - `day1-scope-and-configuration-baseline.md`
  - `day1-authoritative-inputs.txt`
  - `day2-validation-baseline-and-truth-surface-recheck.md`
  - `day3-residual-control-inventory-audit.md`
  - `day4-first-modernization-boundary.md`
  - `day7-post-landing-audit-and-rerank.md`
- design/landing artifacts:
  - `day5-typed-internal-policy-design.md`
  - `day6-fm-graph-policy-integration-batch1.md`
  - `day8-debug-profile-rationalization-design.md`
  - `day9-debug-profile-rationalization-batch.md`
  - `day10-docs-maintainer-header-follow-through-design.md`
  - `day11-docs-maintainer-header-follow-through-batch.md`
- review/closeout artifacts:
  - `day12-proof-alignment-and-build-map.md`
  - `day13-full-validation-sweep.md`
  - `day14-closeout-and-handoff.md`

### Landed configuration package

| Metric | Sprint 73 close state |
|---|---:|
| source files touched in landed package | `4` |
| internal headers touched in landed package | `2` |
| focused proof-owner tests touched | `2` |
| maintained policy docs touched | `1` |
| representative example residual anchors retained | `2` |
| maintained reorder/reporting anchors retained | `2` |

Notes:

- source files touched:
  - `src/sparse_graph.c`
  - `src/sparse_graph_refine.c`
  - `src/sparse_graph_coarsen.c`
  - `src/sparse_reorder_nd.c`
- internal headers touched:
  - `src/sparse_graph_internal.h`
  - `src/sparse_reorder_nd_internal.h`
- focused proof-owner tests touched:
  - `tests/test_graph.c`
  - `tests/test_reorder_nd.c`
- maintained policy docs touched:
  - `docs/maintainer_guide.md`
- representative example residual anchors retained:
  - `example_analysis` -> `4.44e-16`
  - `example_basic_solve` -> `0.00e+00`
- maintained reorder/reporting anchors retained:
  - `bench_reorder` -> `Pres_Poisson ND/AMD = 0.923`, `bcsstk14 ND/AMD = 1.124`
  - `bench_amd_qg` -> `Kuu: qg 1252.7 ms vs bitset 3195.9 ms`

## Residual Deferred Debt

Sprint 73 deliberately stopped after the highest-value residual configuration
phase. The main open work it intentionally hands forward is:

- ND compatibility/default-policy convergence beyond the already-landed `SPARSE_ND_PROFILE` precedence seam
- support-only control cleanup where `SPARSE_QG_PROFILE` or later SVD-routing wording truly moves
- capability modernization starting with index width
- later backend/performance maturity only where benchmark-governed ownership seams still justify the proof cost
- later permanent-surface cleanup only where future implementation work actually moves ownership again

Still consciously constrained rather than silently “solved”:

- no new public typed FM option family
- no new public typed debug/profile option family
- no fake “all env-var debt is gone” story
- no widened reviewed/platform/install claim
- no broad graph-family or product-model rewrite hidden inside configuration work

Not carried forward as unresolved Sprint 73 debt:

- the residual env-var inventory and rerank
- the Day 6 FM/graph policy integration batch
- the Day 9 debug/profile precedence rationalization batch
- the bounded maintainer-guide follow-through
- the proof-owner alignment pass
- the full Day 13 validation sweep
- the Day 14 closeout and ranked Sprint 74 handoff queue

## Key Deliverables

- [PLAN.md](./PLAN.md)
- [WORKING_NOTES.md](./WORKING_NOTES.md)
- [day1-scope-and-configuration-baseline.md](./artifacts/day1-scope-and-configuration-baseline.md)
- [day1-authoritative-inputs.txt](./artifacts/day1-authoritative-inputs.txt)
- [day2-validation-baseline-and-truth-surface-recheck.md](./artifacts/day2-validation-baseline-and-truth-surface-recheck.md)
- [day3-residual-control-inventory-audit.md](./artifacts/day3-residual-control-inventory-audit.md)
- [day4-first-modernization-boundary.md](./artifacts/day4-first-modernization-boundary.md)
- [day5-typed-internal-policy-design.md](./artifacts/day5-typed-internal-policy-design.md)
- [day6-fm-graph-policy-integration-batch1.md](./artifacts/day6-fm-graph-policy-integration-batch1.md)
- [day7-post-landing-audit-and-rerank.md](./artifacts/day7-post-landing-audit-and-rerank.md)
- [day8-debug-profile-rationalization-design.md](./artifacts/day8-debug-profile-rationalization-design.md)
- [day9-debug-profile-rationalization-batch.md](./artifacts/day9-debug-profile-rationalization-batch.md)
- [day10-docs-maintainer-header-follow-through-design.md](./artifacts/day10-docs-maintainer-header-follow-through-design.md)
- [day11-docs-maintainer-header-follow-through-batch.md](./artifacts/day11-docs-maintainer-header-follow-through-batch.md)
- [day12-proof-alignment-and-build-map.md](./artifacts/day12-proof-alignment-and-build-map.md)
- [day13-full-validation-sweep.md](./artifacts/day13-full-validation-sweep.md)
- [day14-closeout-and-handoff.md](./artifacts/day14-closeout-and-handoff.md)

## Bottom-Line Closeout

Sprint 73 succeeded because it converted the strongest residual configuration
contradictions into explicit, validated ownership rules without widening into a
generic cleanup campaign.

The branch now closes with:

- a materially smaller and more coherent residual configuration surface
- explicit compatibility vs internal-policy vs developer-only ownership on the
  touched FM and ND/profile lanes
- named proof owners for the landed precedence and compatibility contracts
- a validated handoff package for Sprint 74
