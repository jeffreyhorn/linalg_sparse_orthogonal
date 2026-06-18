# Sprint 78 Day 8 - Giant-Test Re-audit

Date: 2026-06-17  
Branch: sprint-78

## Purpose
Re-rank the largest remaining permanent proof-owner files by mixed proof-role density, chronology burden, and bounded architecture payoff so Sprint 78 can choose one exact giant-test batch.

## Main Result
Sprint 78's broad giant-test problem is now reduced to one ranked contradiction map instead of one generic “largest test files first” bucket.

The strongest current giant-test ranking is:
- first:
  - `tests/test_chol_csc.c`
- second:
  - `tests/test_ldlt_csc.c`
- third:
  - `tests/test_qr.c`
- fourth:
  - `tests/test_integration.c`
- later / different-shape pressure:
  - `tests/test_reorder_nd.c`

## Why `tests/test_chol_csc.c` Now Leads
`tests/test_chol_csc.c` is the strongest current contradiction center because it still combines too many durable proof roles in one permanent surface:

- CSC working-format allocation and conversion proof
- symbolic and validation proof
- scalar elimination and solve proof
- supernode detection and postorder proof
- dense helper / backend-contract / panel-solve proof
- writeback, dispatch, and large-corpus residual proof

That makes it the best Day 9 architecture target:

- highest mixed proof-role density
- strongest chronology/comment pressure
- strongest bounded helper/taxonomy payoff
- no need to reopen public API or unrelated implementation work

## Remaining Giant-Test Context
### `tests/test_ldlt_csc.c`
Still very large, but now reads more like a coherent family-local proof owner:
- working format
- supernode and row-adjacency support
- `_with_analysis` path
- native/wrapper cross-checks
- solve and inertia proof

### `tests/test_qr.c`
Still large, but reads more like one algorithm-family proof surface with clearer chronological segmentation than the Cholesky CSC seam.

### `tests/test_integration.c`
Still important because it owns public lifecycle and parity truth, but its architecture problem is weaker because its cross-feature/public-contract role is already clearer.

### `tests/test_reorder_nd.c`
Still a major permanent review cost, but it is a different shape of hotspot:
- runtime-heavy
- policy/history dense
- broad environment and typed-option coverage
- weaker as the first giant-test architecture split for Sprint 78

## Current Contradiction Classes
The strongest remaining giant-test contradiction classes are now explicit:
- mixed lifecycle/property/regression roles inside one file
- helper density without one obvious local support seam
- chronology/comment spill from many sprint-era additions
- proof-owner taxonomy that is still mechanically correct but harder to review than necessary

## Day 9 Implication
Day 9 should start from one exact giant-test design center:
- required design center:
  - `tests/test_chol_csc.c`
- strongest supporting proof context:
  - `tests/test_ldlt_csc.c`
  - `tests/test_qr.c`
  - `tests/test_integration.c`
  - `tests/test_reorder_nd.c`

## Exit State
- The broad giant-test problem is reduced to a concrete seam ranking.
- `tests/test_chol_csc.c` is fixed as the strongest Day 9 architecture target.
- Sprint 78 can now design one bounded proof-architecture batch from a live current-state ranking.
