# Sprint 49 Day 13 Artifact: Full Validation Sweep

## Purpose

Run the authoritative final Epic 4 validation pass from the integrated Sprint
49 end state and confirm that the maintained reviewed-baseline truthfulness
anchors still hold exactly.

## Main Day 13 Conclusion

The full integrated Epic 4 end state validated cleanly.

Primary validation passed:

- `make format`
- `make lint`
- `make test`
- `make quality-review-full`

No bounded repair batch was needed during the sweep.

## Reviewed-Baseline Truthfulness Anchors

The maintained anchor set remained exact:

- `ctest -N --test-dir build/quality-review-cmake` = `53`
- Makefile/CMake parity = `53` vs `53`
- full reviewed CMake `ctest` passed `53 / 53`
- `Total Test time (real) = 414.75 sec`

Interpretation:

- Sprint 49 closes on top of the same reviewed validation contract Epic 4
  inherited and preserved
- the final lifecycle/workspace exposure did not degrade the reviewed parity
  surface

## Targeted Sprint 49 Follow-Ons

The targeted follow-ons from the Day 12 checklist all passed:

- `./build/test_iterative`
- `./build/test_eigs`
- `./build/test_eigs_lobpcg`
- `./build/example_iterative`
- `./build/example_eigs`
- `./build/bench_iterative_reuse`
- `./build/bench_eigs_reuse`

Representative direct results:

- `test_iterative`: all `78` tests passed
- `test_eigs`: all `27` tests passed
- `test_eigs_lobpcg`: all `26` tests passed
- `example_iterative`:
  - GMRES without preconditioning converged in `25` iterations
  - GMRES with ILU(0) converged in `9` iterations
- `example_eigs`:
  - nos4 largest-eigenvalue run converged `5 / 5`
  - KKT nearest-`sigma` run converged `3 / 3`
  - bcsstk04 LOBPCG run converged `3 / 3`

## Public Repeated-Run Evidence Recheck

The final public repeated-run handle path remained behavior-stable in the
benchmarks:

### Iterative repeated-run benchmark

- CG: `90.1570 ms` one-shot vs `88.5640 ms` reuse, `1.02x`
- GMRES: `95.9320 ms` one-shot vs `87.5930 ms` reuse, `1.10x`

### Eigensolver repeated-run benchmark

- grow-m Lanczos: `4.0260 ms` one-shot vs `3.5480 ms` reuse, `1.13x`
- thick-restart: `125.2890 ms` one-shot vs `124.6060 ms` reuse, `1.01x`

Behavior-level parity remained intact:

- iterative reuse matched one-shot iteration counts and residuals
- eigensolver reuse matched one-shot iterations, convergence, `n_converged`,
  residuals, and eigenvalues

Interpretation:

- the final public handle path is now validated at the benchmark surface, not
  only at the unit-test surface

## Failure Status

Day 13 failure status:

- none

New reconciliation queue surfaced by the validation sweep:

- none

That matters because the final closeout can now remain a synthesis/handoff day
rather than turning into another repair day.

## Sprint 49 Position After Day 13

Sprint 49 now enters final closeout from the intended validated state:

1. public lifecycle/workspace exposure is landed
2. migration docs and compatibility sweep are landed
3. final residual classification is complete
4. full baseline and reviewed-baseline validation are green
5. targeted repeated-run/example/test follow-ons are green

## Bottom Line

Day 13 delivered:

- a fully green authoritative validation sweep
- maintained reviewed CMake parity at `53`
- successful targeted Sprint 49 follow-ons
- revalidated public repeated-run benchmark evidence
- no new hidden residual queue

That is the correct validated end state for Sprint 49 before final closeout.
