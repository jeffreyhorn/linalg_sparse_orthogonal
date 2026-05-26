# Sprint 41 Day 13 Artifact: Full Validation Sweep

## Purpose

Reconfirm the Sprint 41 end-state against the Sprint 40 validation anchor by
running:

- the full required code-quality gate
- the strongest local reviewed baseline
- the targeted follow-on checks justified by Sprint 41's touched surfaces

## Commands Run

### Full required gate

```bash
/usr/bin/time -p make format
/usr/bin/time -p make lint
/usr/bin/time -p make test
```

### Strongest local reviewed baseline

```bash
/usr/bin/time -p make quality-review-full
```

### Targeted follow-on checks for touched auxiliary surfaces

```bash
/usr/bin/time -p make tooling-build
./build/example_iterative
./build/example_matrix_free
./build/example_colamd
```

## Measured Results

### Full required gate

- `make format`
  - passed
  - `/usr/bin/time -p`: `real 5.22`
- `make lint`
  - passed
  - `/usr/bin/time -p`: `real 411.69`
- `make test`
  - passed
  - `/usr/bin/time -p`: `real 116.28`

### Strongest local reviewed baseline

- `make quality-review-full`
  - passed
  - `/usr/bin/time -p`: `real 869.96`

### Targeted follow-on checks

- `make tooling-build`
  - passed
  - `/usr/bin/time -p`: `real 0.43`
- `./build/example_iterative`
  - passed
- `./build/example_matrix_free`
  - passed
- `./build/example_colamd`
  - passed

## Reviewed-Baseline Proof Points

### `quality-review-full`

The aggregate reviewed baseline passed end to end:

- reviewed Makefile path passed
- reviewed CMake parity path passed

### Makefile reviewed path

Inside `quality-review-full`, `quality-review` passed:

- `format-check`
- `lint`
- `test`
- `deadcode-check`

This means Sprint 41 still preserves the maintained local reviewed path,
including the advisory/reporting dead-code sibling check.

### Reviewed CMake parity path

Inside `quality-review-full`, the reviewed CMake path passed fully:

- configure
- clean rebuild
- `ctest -N`
- Makefile/CMake test-count parity
- full `ctest`

Measured parity truth:

- `ctest -N` reported `53` tests
- Makefile/CMake parity remained `53` vs `53`
- full reviewed CMake `ctest` passed `53 / 53`
- `Total Test time (real) = 192.41 sec`

## Targeted Auxiliary-Surface Proof

Because Sprint 41 changed public examples in Day 11, the targeted follow-on
validation stayed focused on those touched surfaces.

### `example_iterative`

Passed with the same expected teaching behavior:

- unpreconditioned GMRES converged in `25` iterations
- ILU(0)-preconditioned GMRES converged in `9` iterations

### `example_matrix_free`

Passed with the expected matrix-free behavior:

- both runs converged in `3` iterations
- computed solution matched `x_exact` to about `1e-13`

### `example_colamd`

Passed end to end:

- LU fill comparison completed
- QR+COLAMD solve residual printed `0.00e+00`

### `tooling-build`

Passed, confirming the sprint's changed example surfaces still build within the
maintained benchmark/example tooling umbrella:

- `14` benchmark binaries built
- `12` example binaries built

## Reconciliation / Caveats

### No source-level reconciliation was needed

The full direct gate, reviewed baseline, and targeted auxiliary checks all
passed without requiring follow-up edits during Day 13.

### Why there was no standalone serial `deadcode-report` rerun

Sprint 41 did not modify:

- dead-code scripts
- dead-code Makefile wiring
- dead-code reporting semantics

So a standalone serial `make deadcode-report` rerun was not required as a
Day 13 follow-on check for the sprint's touched surfaces.

The maintained reviewed path still exercised `deadcode-check` successfully
inside `make quality-review-full`, which is sufficient for this sprint's
validation scope.

## Day 13 Conclusion

Sprint 41 now closes from a measured validated state:

- direct required gate passed
- strongest local reviewed baseline passed
- reviewed CMake parity truth remained exact at `53`
- touched example/tooling surfaces passed targeted follow-on validation
- no reconciliation edits were needed during the sweep
