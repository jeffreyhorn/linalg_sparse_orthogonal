# Sprint 78 Day 7 - Post-Landing Audit and Rerank

Date: 2026-06-17  
Branch: sprint-78

## Purpose
Re-audit the permanent hotspot surface after the Day 6 LDL^T CSC source batch so Sprint 78 targets the strongest remaining bounded seam rather than repeating the first landing.

## Main Result
The Day 6 landing closed the strongest first implementation contradiction:

- `src/sparse_ldlt_csc.c` no longer reads like the strongest remaining Sprint 78 seam
- a second same-family LDL^T CSC source batch is not the highest-value next move

The strongest remaining seam has now shifted to giant-test architecture.

## Why The Rerank Shifted
The Day 6 cleanup removed the densest implementation-only ambiguity:

- writeback ownership now reads more directly
- wrapper/fallback ownership now reads more directly
- the LDL^T CSC internal contract now reads more directly as one bounded owner with local helper clusters

That means the next highest-value contradiction is no longer “which implementation helper cluster should move next?”

It is now “which permanent proof-owner file still carries too many durable proof roles in one giant review surface?”

## Updated Seam Ranking
- required Day 8 audit center:
  - `tests/test_chol_csc.c`
- strongest support-tier giant-test follow-through:
  - `tests/test_ldlt_csc.c`
  - `tests/test_qr.c`
  - `tests/test_integration.c`
  - `tests/test_reorder_nd.c`
- source hotspots still real but now lower than the giant-test lane:
  - `src/sparse_iterative.c`
  - `src/sparse_chol_csc.c`
  - `src/sparse_lu_csr.c`

## Why `tests/test_chol_csc.c` Now Leads
`tests/test_chol_csc.c` is the strongest current permanent proof hotspot because it still combines too many durable roles at once:

- working-format and conversion proof
- elimination and solve proof
- supernodal helper and backend-contract proof
- writeback and dispatch proof
- large corpus and residual regressions

That makes it the best next giant-test audit center:

- highest mixed proof-role density
- strongest bounded architecture payoff
- strongest chronology/comment pressure among the remaining giant tests
- no need to reopen public API or unrelated subsystem work

## Remaining Giant-Test Context
- `tests/test_ldlt_csc.c` remains large, but now reads more like a coherent family-local proof owner than the strongest mixed-role contradiction center.
- `tests/test_qr.c` remains large, but reads more like one algorithm-family proof surface than the strongest architecture problem.
- `tests/test_integration.c` remains the public parity and lifecycle truth owner, but it is more bounded than the Cholesky CSC giant-test seam.
- `tests/test_reorder_nd.c` remains runtime-heavy, but its strongest pressure is runtime cost and breadth rather than the same giant mixed-role architecture burden.

## Day 8 Implication
Day 8 should start from an exact giant-test audit center rather than from the original mixed hotspot backlog:

- exact next audit center:
  - `tests/test_chol_csc.c`
- strongest supporting giant-test context:
  - `tests/test_ldlt_csc.c`
  - `tests/test_qr.c`
  - `tests/test_integration.c`
  - `tests/test_reorder_nd.c`

## Exit State
- Sprint 78 does not need another immediate LDL^T CSC source batch.
- The strongest remaining seam is now explicitly reranked to giant-test architecture.
- Day 8 starts from a current-state giant-test center led by `tests/test_chol_csc.c`.
