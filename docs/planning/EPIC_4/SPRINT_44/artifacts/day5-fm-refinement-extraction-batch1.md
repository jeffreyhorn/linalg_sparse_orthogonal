# Sprint 44 Day 5 Artifact: FM Refinement Extraction Batch 1

## Purpose

Land the first real Sprint 44 Phase-2 implementation batch by moving the
bounded FM refinement subsystem out of the residual graph monolith and into its
own implementation unit.

## 1. Extracted FM Module

Day 5 created:

- `src/sparse_graph_refine.c`

The module now owns:

- FM-local thread runtime state
- FM parser/helpers
- thick-restart perturbation helper
- shared cut-weight evaluation
- FM bucket implementation
- `graph_refine_fm(...)`

This matches the Day 3 boundary directly.

## 2. Residual `src/sparse_graph.c` Boundary After the Move

The residual graph file now starts at the intended orchestration seam.

It retains:

- `graph_uncoarsen(...)`
- separator lifting
- top-level partition orchestration
- sep=`0` retry / fallback glue

It no longer owns the FM implementation body.

## 3. Internal Interface Shape

The shared graph contract only grew enough to support the live
uncoarsening/orchestration seam.

Added internal FM support declarations:

- FM schedule / perturbation enums
- `sparse_graph_fm_runtime_t`
- `sparse_graph_parse_fm_anneal_schedule(...)`
- `sparse_graph_parse_fm_thick_restart_perturb(...)`
- `sparse_graph_parse_fm_gain_noise_schedule(...)`
- `sparse_graph_fm_runtime_get(...)`
- `sparse_graph_fm_runtime_set(...)`
- `sparse_graph_compute_cut_weight(...)`
- `sparse_graph_thick_restart_perturb(...)`

Interpretation:

- the extraction kept public headers untouched
- interface growth stayed internal and tied to real cross-file orchestration
  needs

## 4. Build Wiring

The new FM module was added to both maintained build surfaces:

- `Makefile`
- `CMakeLists.txt`

No special-case build logic was required.

## 5. Scope That Intentionally Stayed Deferred

Day 5 did **not** move:

- `graph_uncoarsen(...)`
- separator lifting
- top-level partition orchestration
- `partition_once(...)`
- `sparse_graph_partition(...)`

This preserves the planned Sprint 44 order:

1. Day 5: FM extraction
2. Day 6: separator extraction
3. Day 7: residual runtime/orchestration audit
4. Day 8: runtime/orchestration cleanup

## 6. Validation

Because `*.c` and `*.h` files changed, the full required gate ran:

- `make format`
- `make lint`
- `make test`

All passed.

The authoritative `make test` sweep also re-covered the graph/ND surfaces most
relevant to the extraction:

- `test_graph`
- `test_graph_fm_buckets`
- `test_reorder_nd`
- `test_reorder_amd_qg`

## Bottom Line

Day 5 delivered the first real Sprint 44 Phase-2 graph extraction:

- `src/sparse_graph_refine.c` now owns the FM refinement subsystem
- `src/sparse_graph.c` is materially narrower and more honestly scoped
- the build wiring and internal contracts stayed bounded
- the full validation gate passed cleanly
