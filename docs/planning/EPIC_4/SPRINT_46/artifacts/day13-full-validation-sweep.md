# Sprint 46 Day 13: Full Validation Sweep

## Summary

Day 13 ran the authoritative full validation sweep for the Sprint 46
eigensolver workspace reuse and repeated-run benchmark work.

The result is clean:

- the mandatory code-change floor passed
- `make quality-review-full` passed
- reviewed CMake parity remained exact at `53`
- the direct eigensolver, example, and repeated-run benchmark follow-ons all
  passed

No reconciliation queue surfaced.

## Validation Runs

### Mandatory floor

- `make format` → passed
- `make lint` → passed
- `make test` → passed

### Strong reviewed proof

- `make quality-review-full` → passed

Included reviewed-path components:

- reviewed Makefile path
- `deadcode-check`
- reviewed clean CMake rebuild
- `ctest -N`
- full reviewed CMake `ctest`

### Direct eigensolver / example / benchmark reruns

- `./build/test_eigs` → passed, `25` tests, `Time: 0.133 s`
- `./build/test_eigs_thick_restart` → passed, `20` tests, `Time: 0.291 s`
- `./build/test_eigs_lobpcg` → passed, `26` tests, `Time: 0.143 s`
- `./build/example_eigs` → passed
- `./build/bench_eigs_reuse` → passed

## Truthfulness Anchors

The preserved Sprint 40 validation/truthfulness anchors remained exact:

- `ctest -N --test-dir build/quality-review-cmake` = `53`
- Makefile/CMake parity = `53` vs `53`
- full reviewed CMake `ctest` passed `53 / 53`
- `Total Test time (real) = 172.98 sec`

Interpretation:

- Sprint 46 did not disturb the maintained local reviewed baseline
- the eigensolver workspace and repeated-run changes remain aligned with the
  established Makefile/CMake parity contract

## Sprint-46-Specific Outcome

The direct reruns matter because Sprint 46 touched three main eigensolver
family surfaces plus one repeated-run evidence seam:

- grow-m Lanczos
- thick-restart Lanczos
- LOBPCG
- `bench_eigs_reuse`

The explicit Sprint 46 protection points stayed green:

- `test_eigs`
  - all `25` tests passed
- `test_eigs_thick_restart`
  - all `20` tests passed
- `test_eigs_lobpcg`
  - all `26` tests passed
- `example_eigs`
  - nos4 largest-eigenvalue run converged `5 / 5` in `115` Lanczos iterations
    with residual norm `4.326e-14`
  - KKT nearest-sigma run converged `3 / 3` in `6` Lanczos iterations
  - bcsstk04 smallest-eigenvalue LOBPCG run converged `3 / 3` in `62` outer
    iterations with residual norm `8.808e-09`

Interpretation:

- the migrated workspace paths remain stable across the three main eigensolver
  families Sprint 46 targeted
- the example surface still demonstrates the intended solver mix cleanly
- the direct validation surface agrees with the broader reviewed baseline

## Benchmark Note

The Day 13 repeated-run benchmark rerun remained behavior-stable but
timing-sensitive:

- grow-m Lanczos repeated-run case (`nos4`, `k=5`, `repeats=40`)
  - one-shot = `1.4920 ms`
  - reuse = `1.4770 ms`
  - speedup = `1.01x`
- thick-restart repeated-run case (`bcsstk14`, `k=5`, `repeats=8`)
  - one-shot = `51.4250 ms`
  - reuse = `52.1510 ms`
  - speedup = `0.99x`

Both paths still matched exactly on:

- iteration counts
- convergence flags
- relative residuals
- eigenvalue outputs

Interpretation:

- the reusable-workspace seam is real and directly measurable
- the runtime effect remains modest and moves around across local reruns
- Sprint 46 should therefore keep the Day 11/13 claim narrow:
  - repeated-run behavior matches
  - allocation-path reuse is demonstrable
  - no universal speedup claim is justified

## Caveats

No new caveats surfaced beyond the maintained contract:

- dead-code execution remains serialized
- reviewed CMake remains the strongest shared reviewed baseline
- Sprint 46 intentionally does not solve:
  - public explicit eigensolver workspace APIs
  - broad benchmark CLI redesign
  - broad public docs/tutorial refresh for repeated-run guidance

## Outcome

Sprint 46 now enters Day 14 closeout from a measured, validated state:

- reusable internal eigensolver workspace layer validated
- migrated grow-m / thick-restart / LOBPCG workspace paths validated
- compatibility wrapper surface validated
- repeated-run benchmark evidence rechecked
- reviewed baseline and reviewed CMake parity preserved

That is the correct end-state before Sprint 46 closeout and handoff.
