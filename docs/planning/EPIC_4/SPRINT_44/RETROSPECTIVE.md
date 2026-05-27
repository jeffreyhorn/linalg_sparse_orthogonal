# Sprint 44 Retrospective

**Sprint:** 44 — Graph / ND Subsystem Decomposition Phase 2 & Large-Test Maintainability Batch  
**Duration:** 14 days (Days 1-14)  
**Status:** Complete

## Definition Of Done Checklist

- [x] Sprint 44 baseline and Phase-2 graph / test-maintainability scope captured before implementation
- [x] residual `src/sparse_graph.c` seam inventory refreshed against live code
- [x] bounded FM-refinement extraction design completed
- [x] bounded separator/runtime/test design completed
- [x] FM refinement extraction landed in live code
- [x] separator lifting / separator-policy extraction landed in live code
- [x] residual graph runtime/orchestration audit completed
- [x] residual graph orchestration cleanup landed
- [x] focused graph seam-test design completed
- [x] focused graph seam tests landed
- [x] large-test helper audit completed
- [x] first large-test helper consolidation batch landed
- [x] full validation sweep completed
- [x] Sprint 44 closeout and handoff completed from the measured baseline

## What Went Well

1. **Sprint 44 completed the real Phase-2 graph split, not just more residual cleanup.**
   The sprint took the two most important remaining subsystem seams out of the
   residual graph monolith:
   - `src/sparse_graph_refine.c`
   - `src/sparse_graph_separator.c`
   That is a meaningful follow-on to Sprint 43 rather than superficial file
   churn.

2. **The extraction order was again correct.** Sprint 44 did not start by
   reopening orchestration glue. It moved the coherent remaining ownership
   seams first:
   - FM refinement
   - separator-policy / separator-lifting logic
   Only after those landed did the sprint simplify the residual
   runtime/orchestration layer.

3. **The Phase-2 boundary held.** The sprint resisted several obvious ways to
   sprawl:
   - no attempt to fully eliminate `src/sparse_graph.c`
   - no broad retry/fallback redesign
   - no graph public-API churn
   - no sweeping large-test rewrite across multiple binaries
   That kept the sprint structural and maintainability-focused instead of
   turning into a mixed graph/test cleanup grab bag.

4. **The graph seam protections were targeted and useful.** Day 10 added the
   right behavior-level coverage for the newly extracted Phase-2 boundaries:
   - `test_edge_to_vertex_separator_balanced_boundary_prefers_smaller_boundary`
   - `test_partition_fifo_balanced_boundary_smoke`
   Those protect the new separator/orchestration ownership lines without
   forcing private-helper tests.

5. **The first large-test maintainability batch was intentionally small and successful.**
   Sprint 44 did not overclaim a test-architecture rewrite. It landed one
   bounded proof point in `tests/test_qr.c`:
   - `assert_qr_reconstruction_below(...)`
   - `assert_qr_true_residual_below(...)`
   That removed a real repeated assertion seam while preserving the
   one-binary-per-test model.

6. **The sprint closed from a measured maintained baseline.** Day 13 validated
   both the normal code-change floor and the strongest local reviewed path:
   - `make format`
   - `make lint`
   - `make test`
   - `make quality-review-full`
   It also revalidated the direct graph / ND and touched-QR surfaces after the
   extraction and helper batches.

## What Didn't Go Well

1. **A residual orchestration layer still remains after Phase 2.** Sprint 44
   correctly narrowed `src/sparse_graph.c`, but it still owns:
   - `graph_uncoarsen(...)`
   - top-level partition orchestration
   - retry/fallback glue
   - orchestration-scoped runtime parsing
   This is expected, but it means Sprint 44 is not the end of graph cleanup.

2. **The graph subsystem still carries a heavier validation burden than an ordinary local refactor.**
   Even bounded Phase-2 extraction batches required confidence across:
   - `test_graph`
   - `test_graph_fm_buckets`
   - `test_reorder_nd`
   - `test_reorder_amd_qg`
   - `make quality-review-full`
   The sprint handled that correctly, but graph work continues to be one of
   the most expensive-to-validate surfaces in the repository.

3. **The large-test maintainability queue is only partially addressed.**
   Sprint 44 intentionally proved the helper-extraction model in `test_qr`,
   but it left the other real hotspot binaries for later:
   - `tests/test_chol_csc.c`
   - `tests/test_ldlt_csc.c`
   - `tests/test_svd.c`
   That was the right scope call, but the broader maintainability queue is
   still present.

## Final Metrics

### Validated closeout baseline

| Metric | Sprint 44 close state |
|---|---:|
| strongest local reviewed baseline command | `make quality-review-full` |
| reviewed CMake `ctest -N` | `53` |
| Makefile/CMake parity | `53 vs 53` |
| full reviewed CMake `ctest` | `53 / 53` |

### Sprint 44 artifact package

| Metric | Sprint 44 close state |
|---|---:|
| total artifact files under `SPRINT_44/artifacts/` | `15` |
| implementation-focused artifacts (Days 5, 6, 8, 10, 12) | `5` |
| validation / closeout artifacts (Days 13-14) | `2` |

### Phase-2 graph and test-maintainability outputs

| Metric | Sprint 44 close state |
|---|---:|
| new graph implementation modules added | `2` |
| maintained build surfaces updated for Phase-2 graph split | `2` |
| direct graph / ND / touched-QR reruns in Day 13 | `5` |
| focused graph seam regressions added on Day 10 | `2` |
| new local QR maintainability helpers added on Day 12 | `2` |

Notes:

- new graph implementation modules:
  - `src/sparse_graph_refine.c`
  - `src/sparse_graph_separator.c`
- maintained build surfaces:
  - `Makefile`
  - `CMakeLists.txt`
- Day 10 focused graph seam regressions:
  - `test_edge_to_vertex_separator_balanced_boundary_prefers_smaller_boundary`
  - `test_partition_fifo_balanced_boundary_smoke`
- Day 12 QR maintainability helpers:
  - `assert_qr_reconstruction_below(...)`
  - `assert_qr_true_residual_below(...)`

## Residual Deferred Debt

Sprint 44 was designed as graph Phase 2 plus a first large-test maintainability
proof point, not as the end of either queue. The main open work it
intentionally hands forward is:

- deeper residual orchestration simplification in `src/sparse_graph.c`
- any future retry/fallback glue sub-splitting only if later evidence justifies it
- later helper/fixture consolidation candidates:
  - `tests/test_chol_csc.c`
  - `tests/test_ldlt_csc.c`
  - `tests/test_svd.c`
- any future file splitting or broader domain-specific test fixture extraction
  only when a later sprint explicitly chooses that scope

Not carried forward as unresolved Sprint 44 debt:

- missing FM refinement extraction seam
- missing separator-policy extraction seam
- missing residual graph orchestration cleanup
- missing focused Phase-2 graph seam tests
- missing first large-test maintainability landing
- missing measured validation closeout

## Key Deliverables

- [PLAN.md](./PLAN.md)
- [WORKING_NOTES.md](./WORKING_NOTES.md)
- [day3-fm-refinement-module-boundary-design.md](./artifacts/day3-fm-refinement-module-boundary-design.md)
- [day4-separator-runtime-and-large-test-design.md](./artifacts/day4-separator-runtime-and-large-test-design.md)
- [day5-fm-refinement-extraction-batch1.md](./artifacts/day5-fm-refinement-extraction-batch1.md)
- [day6-separator-extraction-batch1.md](./artifacts/day6-separator-extraction-batch1.md)
- [day8-runtime-parsing-and-orchestration-cleanup.md](./artifacts/day8-runtime-parsing-and-orchestration-cleanup.md)
- [day10-focused-graph-seam-tests.md](./artifacts/day10-focused-graph-seam-tests.md)
- [day12-first-test-helper-consolidation-batch.md](./artifacts/day12-first-test-helper-consolidation-batch.md)
- [day13-full-validation-sweep.md](./artifacts/day13-full-validation-sweep.md)
- [day14-closeout-and-handoff.md](./artifacts/day14-closeout-and-handoff.md)

## Bottom Line

Sprint 44 achieved its goal:

- Epic 4 now has the Phase-2 graph decomposition that Sprint 43 intentionally
  left for later
- FM refinement no longer lives only in the residual graph monolith
- separator-policy logic no longer lives only in the residual graph monolith
- the remaining `src/sparse_graph.c` is narrower and more honest about what it
  still owns
- the new Phase-2 boundaries are pinned by focused behavior-level tests
- the large-test maintainability queue now has one real landed proof point in
  `tests/test_qr.c`
- the sprint closed from a measured maintained validation baseline

Later Epic 4 work can now continue from a substantially cleaner graph
subsystem and a proven bounded test-maintainability pattern instead of
reopening whether those seams can be landed at all.
