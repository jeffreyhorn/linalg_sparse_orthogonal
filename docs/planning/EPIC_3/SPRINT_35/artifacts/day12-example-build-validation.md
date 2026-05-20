# Sprint 35 Day 12: Example Build & Snippet Validation

## Scope

Validate the rewritten public-doc surface against the real shipped example and
tooling targets before the final Sprint 35 validation sweep.

Day 12 is the point where the sprint proves that the docs now point at live
code instead of merely looking internally consistent.

## Commands Run

### Build surfaces

- `make examples`
- `make tooling-build`

### Example binaries

- `./build/example_basic_solve`
- `./build/example_least_squares`
- `./build/example_iterative`
- `./build/example_svd_lowrank`
- `./build/example_eigs`

## Main Result

All intended Day 12 validation commands passed.

The example build surface is clean, the compile-only tooling gate is clean, and
the example binaries most directly referenced by the rewritten docs still run
successfully.

## What Was Confirmed

### 1. Example and tooling builds still match the docs

- `make examples` built all `12` example binaries
- `make tooling-build` built all `14` benchmark binaries and all `12` example
  binaries

This confirms that the public-doc rewrite did not leave the documented example
surface or the compile-only benchmark/example gate in stale shape.

### 2. High-traffic public examples still behave as documented

#### `example_basic_solve`

- LU example still uses the copy-before-factor pattern the docs now call out
- produced the expected all-ones solution with zero residual

#### `example_least_squares`

- QR example still demonstrates the overdetermined least-squares path
- reported a valid least-squares solution and residual norm

#### `example_iterative`

- iterative example still demonstrates GMRES with and without ILU(0)
- ILU(0) path converged in fewer iterations than the unpreconditioned run,
  matching the support-doc description

#### `example_svd_lowrank`

- SVD low-rank example still exercises the singular-value spectrum, condition
  number, rank estimation, and low-rank approximation path documented in the
  user-facing material

#### `example_eigs`

- eigensolver example still runs from project root as `examples/README.md`
  describes
- fixture resolution and the mixed SPD / indefinite demonstration path are
  still valid

## Drift Check

No additional doc/code mismatch surfaced on Day 12:

- no stale public type names reappeared
- no snippet-to-binary contradiction surfaced
- no quality-command naming drift surfaced
- no support-doc follow-on rewrite was needed

## Bottom Line

Day 12 closes the last likely place Sprint 35 could have hidden doc/code drift.

Day 13 can now run the full maintained validation set as a pure validation
sweep rather than a mixed validation-and-cleanup pass.
