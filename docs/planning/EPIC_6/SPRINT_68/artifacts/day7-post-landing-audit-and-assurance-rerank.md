# Sprint 68 Day 7: Post-Landing Audit & Assurance Rerank

Date: 2026-06-13
Branch: `sprint-68`

## Purpose

Rerank the remaining Sprint 68 queue after the first landed `test_chol_csc`
helper extraction so the next batch is chosen from the actual branch state
instead of the original Day 3 hotspot map.

## Audit Inputs

- `docs/planning/EPIC_6/SPRINT_68/PLAN.md`
- `docs/planning/EPIC_6/SPRINT_68/WORKING_NOTES.md`
- `docs/planning/EPIC_6/SPRINT_68/artifacts/day6-giant-test-refactor-batch1.md`
- live post-Day-6 measurements and rereads across:
  - `tests/test_chol_csc.c`
  - `tests/test_chol_csc_supernodal_helpers.h`
  - `tests/test_reorder_nd.c`
  - `tests/test_ldlt_csc.c`
  - `tests/test_integration.c`
  - `tests/test_iterative.c`
  - `tests/test_eigs.c`
  - `tests/test_svd.c`

## Post-Landing Measurements

| File | Lines |
|---|---:|
| `tests/test_chol_csc.c` | `4608` |
| `tests/test_chol_csc_supernodal_helpers.h` | `232` |
| `tests/test_reorder_nd.c` | `2262` |
| `tests/test_ldlt_csc.c` | `3680` |
| `tests/test_integration.c` | `2371` |
| `tests/test_iterative.c` | `2802` |
| `tests/test_eigs.c` | `1522` |
| `tests/test_svd.c` | `2766` |

## Audit Conclusions

### 1. Day 6 closed the strongest first-order helper-extraction contradiction

The Day 6 batch reduced the main `test_chol_csc` owner from the pre-landing
`4751` line state to `4608` lines while moving narrow supernodal/writeback
support into the existing family-local helper seam.

That means the file is still large, but it now reads more consistently as:

- one canonical family-local proof owner
- one bounded helper seam for supernodal/writeback scaffolding

So the strongest pure helper-extraction contradiction that justified the first
landing is now materially smaller.

### 2. `tests/test_reorder_nd.c` is now the strongest remaining pure refactor seam

After Day 6, `tests/test_reorder_nd.c` remains the clearest pure
maintainability/refactor target because it still combines:

- chronology-heavy layering
- compatibility/env-policy proof
- supernodal-postorder and dispatch validation

But it is now best treated as the strongest deferred refactor seam rather than
the next immediate move.

### 3. The strongest next move is now the public/oracle owner on the large-`n` CSC-backed Cholesky lane

The highest-value next batch now sits in:

- `tests/test_integration.c`

Why it ranks first now:

- it is the shared owner for public-path oracle/parity proof
- the large-`n` CSC-backed Cholesky path is still one of Epic 6's hardest
  retained numerical lanes
- after Day 6, stronger second-layer assurance now has better payoff than an
  immediate second refactor batch

The likely family-local support context remains:

- `tests/test_chol_csc.c`

### 4. The exact Day 8-10 target set is now fixed

Strongest next batch:

- large-`n` CSC-backed Cholesky public-path oracle/parity expansion

Required likely owner:

- `tests/test_integration.c`

Likely support only if the final oracle shape truly needs it:

- `tests/test_chol_csc.c`

Current non-touch set:

- `tests/test_reorder_nd.c`
- `tests/test_ldlt_csc.c`
- `tests/test_iterative.c`
- `tests/test_eigs.c`
- `tests/test_svd.c`
- implementation `src/` files
- benchmark/docs truth surfaces

## Exit State

Sprint 68 Day 7 closes with one explicit post-landing order:

1. strongest next move:
   - `tests/test_integration.c`
   - one bounded large-`n` CSC-backed Cholesky oracle/parity batch
2. likely support only if needed:
   - `tests/test_chol_csc.c`
3. strongest deferred pure refactor seam:
   - `tests/test_reorder_nd.c`
4. later assurance/follow-through owners:
   - `tests/test_ldlt_csc.c`
   - `tests/test_iterative.c`
   - `tests/test_eigs.c`
   - `tests/test_svd.c`

That gives Day 8 one exact job:

- define the bounded large-`n` CSC-backed Cholesky oracle/parity contract in
  the public integration owner
