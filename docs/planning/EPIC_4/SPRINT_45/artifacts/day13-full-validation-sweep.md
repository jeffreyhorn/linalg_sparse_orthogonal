# Sprint 45 Day 13: Full Validation Sweep

## Summary

Day 13 ran the authoritative full validation sweep for the Sprint 45 iterative
workspace reuse and repeated-solve benchmark work.

The result is clean:

- the mandatory code-change floor passed
- `make quality-review-full` passed
- reviewed CMake parity remained exact at `53`
- the direct iterative, benchmark, and example follow-ons all passed

No reconciliation queue surfaced.

## Validation Runs

### Mandatory floor

- `make format` → passed
- `make lint` → passed
- `make test` → passed, `real 83.44`

### Strong reviewed proof

- `make quality-review-full` → passed, `real 664.75`

Included reviewed-path components:

- reviewed Makefile path
- `deadcode-check`
- reviewed clean CMake rebuild
- `ctest -N`
- full reviewed CMake `ctest`

### Direct iterative / benchmark / example reruns

- `./build/test_iterative` → passed, `Time: 0.515 s`
- `./build/test_block_solvers` → passed, `Time: 0.001 s`
- `./build/test_minres` → passed, `Time: 0.003 s`
- `./build/test_bicgstab` → passed, `Time: 0.017 s`
- `./build/test_stagnation` → passed, `Time: 0.007 s`
- `./build/bench_iterative_reuse` → passed
- `./build/example_matrix_free` → passed

## Truthfulness Anchors

The preserved Sprint 40 validation/truthfulness anchors remained exact:

- `ctest -N --test-dir build/quality-review-cmake` = `53`
- Makefile/CMake parity = `53` vs `53`
- full reviewed CMake `ctest` passed `53 / 53`
- `Total Test time (real) = 160.79 sec`

Interpretation:

- Sprint 45 did not disturb the maintained local reviewed baseline
- the iterative workspace and repeated-solve changes remain aligned with the
  established Makefile/CMake parity contract

## Sprint-45-Specific Outcome

The direct reruns matter because Sprint 45 touched three different iterative
surface classes:

- direct reusable-workspace migrations:
  - scalar CG / matrix-free CG
  - scalar GMRES / matrix-free GMRES
  - block CG
- compatibility/composition surfaces:
  - block GMRES
  - block MINRES
  - block BiCGSTAB
- repeated-solve measurement surfaces:
  - `bench_iterative_reuse`
  - `example_matrix_free`

The explicit Sprint 45 protection points stayed green:

- `test_iterative`
  - all `76` tests passed
- `test_block_solvers`
  - all `15` tests passed
  - `block_cg iters=17  single_cg iters=17`
- `test_minres`
  - all `43` tests passed
- `test_bicgstab`
  - all `58` tests passed
- `test_stagnation`
  - all `46` tests passed
- `example_matrix_free`
  - both GMRES runs converged in `3` iterations
  - solution error stayed around `2.7e-13`

Interpretation:

- the migrated workspace paths remain stable
- the wrapper/composition surfaces still compose cleanly with the scalar
  solver truth
- the repeated-solve benchmark/example surfaces remain consistent with the
  main iterative validation surface

## Benchmark Note

The Day 13 repeated-solve benchmark rerun remained behavior-stable but
timing-sensitive:

- CG repeated-solve:
  - one-shot = `26.5910 ms`
  - reuse = `25.9270 ms`
  - speedup = `1.03x`
- GMRES repeated-solve:
  - one-shot = `18.0780 ms`
  - reuse = `19.3130 ms`
  - speedup = `0.94x`

Both paths still matched exactly on:

- iteration counts
- convergence flags
- relative residuals

Interpretation:

- the reusable-workspace seam is real and directly measurable
- the runtime effect is modest and moves around across local reruns
- Sprint 45 should therefore keep the Day 11/13 claim narrow:
  - repeated-solve behavior matches
  - allocation-path reuse is demonstrable
  - no universal speedup claim is justified

## Caveats

No new caveats surfaced beyond the maintained contract:

- dead-code execution remains serialized
- reviewed CMake remains the strongest shared reviewed baseline
- Sprint 45 intentionally does not solve:
  - eigensolver workspace reuse
  - public explicit iterative workspace APIs
  - broad benchmark CLI modernization

## Outcome

Sprint 45 now enters Day 14 closeout from a measured, validated state:

- reusable internal iterative workspace layer validated
- migrated scalar and block workspace paths validated
- wrapper compatibility surfaces validated
- repeated-solve benchmark evidence rechecked
- reviewed baseline and reviewed CMake parity preserved

That is the correct end-state before Sprint 45 closeout and handoff.
