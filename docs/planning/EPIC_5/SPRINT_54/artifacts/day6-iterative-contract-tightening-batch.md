# Sprint 54 Day 6 - iterative contract tightening batch

Date: 2026-06-03
Branch: `sprint-54`

## Purpose

Tighten the already supported public iterative repeated-run handle surface so
the contract is not only exposed, but also directly and symmetrically proved
across the final supported iterative families.

## Landed surface

Day 6 did not broaden Sprint 54 to any new solver families. Instead, it
strengthened the final supported iterative-handle set:

- `CG`
- `GMRES`
- `MINRES`

The patch touched:

- `include/sparse_iterative.h`
- `tests/test_iterative.c`

## What was tightened

### Header contract wording

The top-level iterative handle wording in `include/sparse_iterative.h` now
reads coherently with the landed Day 5 MINRES support instead of still reading
like a CG/GMRES-only repeated-run surface.

Interpretation:

- the public repeated-run iterative-handle owner is now described in a way
  that matches the actual supported family set
- this closes a small but real contract drift introduced by the Day 5 MINRES
  expansion

### Public-proof symmetry

`tests/test_iterative.c` now proves the supported iterative-handle surface more
symmetrically.

The new/strengthened coverage checks:

- `CG`
  - null prepare validation
  - null handle solve validation
  - explicit prepare + repeated reuse
  - zero-init on-demand growth
- `GMRES`
  - null prepare validation
  - null handle solve validation
  - explicit prepare + repeated reuse
  - same-handle growth from a smaller prepared dimension/restart to a later
    larger solve
  - zero-init on-demand growth
- `MINRES`
  - preserved null validation and explicit prepare + reuse
  - same-handle growth from a smaller prepared dimension to a later larger
    solve
  - zero-init on-demand growth

Interpretation:

- the supported iterative-handle contract is now better proved at the
  lifecycle level
- Day 6 closes the proof asymmetry where some supported families had only
  partial handle-behavior coverage

## Boundary preserved

Day 6 stayed inside the Day 4 support fence.

Still explicitly out of scope after this batch:

- `BiCGSTAB` public repeated-run handle exposure
- block iterative public-handle exposure
- backend-specific eigensolver API expansion

## Validation

Required Day 6 gates all passed:

- `make format`
- `make lint`
- `make test`

Focused follow-ons also passed:

- `./build/test_iterative` -> `79 / 79`
- `./build/test_minres` -> `43 / 43`
- `./build/example_ic_minres`
- `./build/bench_iterative_reuse`

Representative direct outputs:

- `example_ic_minres`
  - MINRES on the `42x42` KKT demo converged in `39` iterations
  - Jacobi-preconditioned MINRES converged in `26` iterations
- `bench_iterative_reuse`
  - `cg-tridiag-300`: `1.12x`
  - `gmres-unsym-220`: `1.05x`

## Conclusion

Day 6 closes the remaining high-value iterative lifecycle-tightening seam
without reopening scope:

- the supported iterative repeated-run handle set is now more coherent at the
  contract layer
- the direct public proof is stronger and more symmetric across the supported
  iterative families
- Sprint 54 can now move on to the eigensolver lifecycle/proof/docs tightening
  queue instead of spending more days on iterative handle expansion
