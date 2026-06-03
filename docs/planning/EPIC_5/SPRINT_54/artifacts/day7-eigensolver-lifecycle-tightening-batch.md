# Sprint 54 Day 7 - eigensolver lifecycle tightening batch

Date: 2026-06-03
Branch: `sprint-54`

## Purpose

Tighten the supported public repeated-run eigensolver lifecycle contract so the
final Sprint 54 surface is not only present, but also explicit about which
backends it covers and directly proved on the public LOBPCG repeated-run path.

## Landed surface

Day 7 did not broaden Sprint 54 into a new eigensolver API shape. Instead, it
tightened the existing public repeated-run handle surface:

- `sparse_eigs_handle_t`
- `sparse_eigs_handle_prepare(...)`
- `sparse_eigs_sym_with_handle(...)`
- `sparse_eigs_handle_free(...)`

The patch touched:

- `include/sparse_eigs.h`
- `tests/test_eigs.c`

## What was tightened

### Header contract wording

`include/sparse_eigs.h` now states the real supported repeated-run eigensolver
boundary more explicitly:

- the public repeated-run handle surface covers:
  - grow-m Lanczos
  - thick-restart Lanczos
  - explicit LOBPCG
- Sprint 54 does not introduce backend-specific public handle types
- explicit LOBPCG still runs through the same prepare/run/free public lifecycle
  surface rather than a separate public owner

Interpretation:

- the eigensolver repeated-run contract now reads like the real supported
  backend set instead of a more generic handle abstraction
- this closes the remaining small but real asymmetry between the iterative-side
  Day 6 wording cleanup and the eigensolver-side public contract text

### Public repeated-run LOBPCG proof

`tests/test_eigs.c` now directly proves the supported LOBPCG repeated-run path
through the public handle surface with:

- `test_public_handle_lobpcg_prepare_reuse_and_growth`

That regression checks:

- explicit `SPARSE_EIGS_BACKEND_LOBPCG`
- explicit prepare on a smaller problem/`k`
- repeated reuse on the same prepared shape
- later on-demand growth to a larger problem and larger `k`
- preserved `backend_used == SPARSE_EIGS_BACKEND_LOBPCG`
- correct converged eigenvalues on both the prepared and grown runs

Interpretation:

- the public repeated-run eigensolver proof is no longer only generic-handle
  coverage
- Day 7 closes the highest-value proof gap by exercising the final public
  LOBPCG route directly

## Boundary preserved

Day 7 stayed inside the Day 4 support fence.

Still explicitly out of scope after this batch:

- `BiCGSTAB` public repeated-run handle exposure
- block iterative public-handle exposure
- backend-specific eigensolver public handle types
- broad new eigensolver API families

## Validation

Required Day 7 gates all passed:

- `make format`
- `make lint`
- `make test`

Focused follow-ons also passed:

- `./build/test_eigs` -> `28 / 28`
- `./build/test_eigs_lobpcg` -> `26 / 26`
- `./build/example_eigs`
- `./build/bench_eigs_reuse`

Representative direct outputs:

- `test_eigs`
  - includes the new direct public repeated-run LOBPCG proof:
    - `test_public_handle_lobpcg_prepare_reuse_and_growth`
- `example_eigs`
  - explicit LOBPCG on `bcsstk04` still converged `3 / 3` smallest eigenpairs
    in `62` outer iterations
  - `backend_used = LOBPCG`
  - `reported residual_norm = 8.808e-09`
- `bench_eigs_reuse`
  - `growm-nos4-k5`: `0.96x`
  - `thick-bcsstk14-k5`: `1.04x`
  - both repeated-run cases kept exact eigenvalue parity with the one-shot path

## Conclusion

Day 7 closes the main remaining eigensolver lifecycle/proof drift without
reopening Sprint 54 scope:

- the public repeated-run eigensolver handle surface now describes the actual
  supported backend set more clearly
- the supported LOBPCG repeated-run path is now directly proved through the
  public handle surface
- Sprint 54 can now move on from eigensolver lifecycle tightening to
  benchmark/example/README alignment for the final support boundary
