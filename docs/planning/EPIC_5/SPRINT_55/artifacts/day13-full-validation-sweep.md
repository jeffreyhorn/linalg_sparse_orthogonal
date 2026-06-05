# Sprint 55 Day 13 - full validation sweep

Date: 2026-06-04
Branch: `sprint-55`

## Scope

Run the full Sprint 55 validation checklist from the landed Day 12 state and
confirm the decomposition work preserved the reviewed quality baseline.

## Full gate result

The required Day 13 validation gate passed completely:

- `make format`
- `make lint`
- `make test`
- `make quality-review-full`

No corrective patch was needed during the sweep.

## Reviewed baseline anchors

The maintained reviewed anchors stayed exact:

- `ctest -N --test-dir build/quality-review-cmake` = `53`
- Makefile/CMake parity = `53 vs 53`
- full reviewed CMake `ctest` = `53 / 53`
- `Total Test time (real) = 253.48 sec`

Interpretation:

- Sprint 55 still preserves the strongest local reviewed baseline
- both supported local build surfaces still agree on the full reviewed test
  inventory after the source splits

## Targeted follow-on result

The planned Sprint 55 follow-ons all passed:

- `./build/test_iterative` -> `79 / 79`
- `./build/test_minres` -> `43 / 43`
- `./build/test_eigs` -> `30 / 30`
- `./build/test_eigs_lobpcg` -> `26 / 26`
- `./build/example_iterative`
- `./build/example_eigs`
- `./build/bench_iterative_reuse`
- `./build/bench_eigs_reuse`

Representative retained outputs:

- iterative examples/benchmarks:
  - `example_iterative`: GMRES converged in `25` iterations unpreconditioned
    and `9` with ILU(0)
  - `bench_iterative_reuse`:
    - `cg-tridiag-300` = `0.84x`
    - `gmres-unsym-220` = `1.44x`
    - `minres-kkt-42` = `0.87x`
  - parity remained exact on iterations and reported residuals between one-shot
    and reuse paths
- eigensolver examples/benchmarks:
  - `example_eigs`: explicit `LOBPCG` on `bcsstk04` still converged `3 / 3`
    smallest pairs in `62` outer iterations with residual `8.808e-09`
  - `bench_eigs_reuse`:
    - `growm-nos4-k5` = `1.21x`
    - `thick-bcsstk14-k5` = `1.04x`
    - `lobpcg-diag40-k3` = `1.03x`
  - eigensolver parity remained exact with `|lambda|max diff = 0.000e+00`

## Conclusion

Sprint 55 Day 13 confirms that the large-source decomposition work preserved
the validated public behavior baseline and the full reviewed local quality
contract.

No new reconciliation queue surfaced during validation.
