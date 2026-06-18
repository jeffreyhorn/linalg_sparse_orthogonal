# Sprint 78 Day 1 - Scope and Large-Source/Giant-Test Baseline

Date: 2026-06-17  
Branch: sprint-78

## Purpose
Establish a precise Sprint 78 starting baseline for large-source maintainability and giant-test architecture using the live tree, the Sprint 70 hotspot inventory, and the current permanent proof-owner surfaces rather than broad Epic 7 review language.

## Main Result
Sprint 78 now starts from a precise long-source and giant-test maintainability queue, not from another planning reset and not from another product, capability, backend, benchmark, or platform batch.

The strongest local reviewed baseline is still:
- `make quality-review-full`

Reviewed CMake parity was re-materialized live and remains explicit:
- `ctest -N --test-dir build/quality-review-cmake` = `53`

The highest-value Sprint 78 pressure is now clearly narrowed to:
- source hotspot rerank
- bounded source decomposition
- giant-test rerank
- bounded giant-test architecture batch
- chronology/comment cleanup on touched permanent files
- docs and proof-ownership alignment
- validation and closeout

## Preserved Fence
Sprint 78 must preserve the current Epic 7 maintainability and truthfulness fence:
- no broad subsystem redesign disguised as maintainability cleanup
- no giant-test breakup wave across every proof owner
- no fake history purge that removes durable algorithm explanation
- no reopening of unrelated product, capability, backend, benchmark, packaging, or platform work
- large-source and giant-test budget should be spent only on the highest-value residual hotspots

## Strongest Likely Sprint 78 Touch Surfaces
Implementation hotspots from the live tree:
- `src/sparse_ldlt_csc.c` = `2130`
- `src/sparse_iterative.c` = `1985`
- `src/sparse_lu_csr.c` = `1665`
- `src/sparse_chol_csc.c` = `1564`
- `src/sparse_qr.c` = `1563`
- `src/sparse_ldlt.c` = `1535`
- `src/sparse_eigs.c` = `1534`
- `src/sparse_svd.c` = `1319`
- `src/sparse_matrix.c` = `1125`

Permanent proof-owner and giant-test hotspots from the live tree:
- `tests/test_chol_csc.c` = `4690`
- `tests/test_ldlt_csc.c` = `3680`
- `tests/test_qr.c` = `3197`
- `tests/test_etree.c` = `2962`
- `tests/test_graph.c` = `2925`
- `tests/test_iterative.c` = `2841`
- `tests/test_ldlt.c` = `2798`
- `tests/test_svd.c` = `2766`
- `tests/test_integration.c` = `2543`
- `tests/test_reorder_nd.c` = `2287`

Likely support and policy surfaces if ownership boundaries move:
- `docs/maintainer_guide.md`
- `README.md`

## High-Signal Live Reading
The live tree already fixes several important Sprint 78 truths:
- Sprint 70 still supplies the giant-test queue and non-goal fence that Sprint 78 should inherit rather than reopen.
- `tests/test_reorder_nd.c` still reads as the densest chronology-heavy proof-owner seam.
- `tests/test_chol_csc.c` still reads as the strongest family-local giant-test seam.
- `docs/maintainer_guide.md` is already the main permanent policy owner for proof ownership and documentation placement.
- `README.md` remains a support-only front-door surface rather than the full maintainability-policy home.
- The largest remaining implementation files are concentrated in direct-solver, iterative, eigensolver, and matrix-core families, so Sprint 78 should rank by ownership pain and extraction value rather than by size alone.

## Interpretation
The strongest Day 1 narrowing is now explicit:
- Sprint 78 is not “make every large file smaller.”
- It is one bounded maintainability phase on the strongest remaining permanent implementation and proof hotspots.
- The highest-value early work is therefore:
  - rerank the current large sources by ownership and extraction value
  - rerank the current giant tests by chronology burden and proof architecture value
  - spend cleanup budget only where the review payoff is clearly highest

## Exit State
- The maintained reviewed baseline is rechecked.
- The reviewed CMake parity anchor is explicit and current.
- The live large-source and giant-test owners, hotspot map, and non-goal fence are fixed in writing.
- Sprint 78 can now move into a ranked source-hotspot audit from a precise Day 1 baseline.
