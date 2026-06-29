# Sprint 96 Day 6: Direct-Family Source Cleanup Closeout

## Purpose

Day 6 completes the first direct-family source cleanup batch by reconciling
ownership comments, proof-owner expectations, and residual direct-family work
after the Day 5 LDLT dense/backend extraction.

## Completed Cleanup

Updated `src/sparse_ldlt_csc_internal.h` comments to reflect the new ownership
shape:

- the internal header now names `sparse_ldlt_dense.c` as a direct internal
  consumer alongside `sparse_ldlt_csc.c` and
  `sparse_ldlt_csc_supernodal.c`
- the dense LDLT declaration section now states that the implementation lives
  in `src/sparse_ldlt_dense.c`
- the dense factor description no longer ties the active contract to a stale
  sprint-day implementation narrative

No behavior changed. The Day 6 code change is comment-only.

## Current Direct-Family Ownership

| Surface | Current owner |
|---|---|
| dense LDLT block factor | `src/sparse_ldlt_dense.c` |
| LDLT dense backend environment parsing | `src/sparse_ldlt_dense.c` |
| LDLT dense backend runtime probe | `src/sparse_ldlt_dense.c` |
| LDLT CSC allocation/conversion/writeback | `src/sparse_ldlt_csc.c` |
| LDLT CSC native sparse elimination | `src/sparse_ldlt_csc.c` |
| LDLT CSC solve path | `src/sparse_ldlt_csc.c` |
| LDLT CSC supernodal extract/writeback/panel helpers | `src/sparse_ldlt_csc_supernodal.c` |
| shared LDLT CSC internal contract | `src/sparse_ldlt_csc_internal.h` |

## Proof Ownership

The full test run covers the direct-family proof owners affected by the Day 5
and Day 6 cleanup:

- `test_chol_csc`
- `test_ldlt_csc`
- `test_direct_csc_dispatch`
- `test_direct_csc_regression`
- `test_ldlt`
- `test_ldlt_backend_dispatch`

Observed targeted direct proof results in the Day 6 full run:

- `test_chol_csc`: 152 tests passed
- `test_ldlt_csc`: 96 tests passed
- `test_direct_csc_dispatch`: 10 tests passed
- `test_direct_csc_regression`: 8 tests passed
- `test_ldlt_backend_dispatch`: 20 tests passed

## Validation

Required code-day chain:

```sh
make format && make lint && make test
```

Result: passed.

Additional hygiene checks should still be run after the Day 6 planning notes:

- `git diff --check`
- trailing-whitespace scan for touched Sprint 96 and direct-family files

## Residual Direct-Family Queue

The direct-family cleanup is complete for the selected Sprint 96 batch.
Residual work should stay out of the current direct lane unless later sprint
days find a narrow dependency:

- split or simplify native sparse LDLT CSC elimination helpers only if a future
  cleanup can preserve the current proof boundary
- revisit conversion/writeback helper ownership after the giant-test
  architecture work clarifies proof pressure
- leave `src/sparse_ldlt.c`, `src/sparse_lu_csr.c`, and
  `src/sparse_chol_csc.c` as separate residual hotspots
- defer public header changes; no public API issue was found
- defer benchmark or generated documentation changes; no command or generated
  surface changed

## Day 6 Result

The direct-family source cleanup batch is complete. LDLT dense/backend
ownership is separated, the internal contract comments match the new owner
layout, the direct proof owners pass under the full required quality chain, and
remaining direct-family work is explicitly residual rather than part of the
current Sprint 96 direct lane.
