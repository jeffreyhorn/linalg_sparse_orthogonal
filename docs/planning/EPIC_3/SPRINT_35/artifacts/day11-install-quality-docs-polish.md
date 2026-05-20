# Sprint 35 Day 11: Installation & Quality Docs Polish

## Scope

Update the support-doc layer after the Day 8 through Day 10 public-doc rewrite:

- `INSTALL.md`
- `examples/README.md`
- `benchmarks/README.md`

The Day 11 goal is to make these files reflect the new public wording and the
current reviewed-quality workflow without reopening the main README/tutorial
rewrite.

## Main Result

The support docs now point at the same public-usage and quality-flow story as
the rest of Sprint 35.

No new core API guidance was invented here. Day 11 mainly removed the lag
between the main public-doc rewrite and the supporting docs around it.

## What Changed

### 1. `INSTALL.md` now reflects the maintained local quality path

The install guide now includes:

- `make tooling-build`
- `make quality-review-compile`
- `make quality-review`

in the Quick Start section, and it also points readers at the reviewed local
CMake parity wrappers.

This is intentionally concise. `INSTALL.md` now acknowledges the maintained
workflow names, but still defers to `README.md` for the full operator command
map and rerun guidance.

### 2. `examples/README.md` now matches the public usage story

The examples catalog now says directly:

- LU-style examples copy before in-place factorization
- the least-squares QR example is the overdetermined path
- underdetermined minimum-norm solves belong to `sparse_qr_solve_minnorm()`
- the iterative example builds ILU(0) from a fresh matrix copy

That gives readers the right signposts without turning the examples index into
another tutorial.

### 3. `benchmarks/README.md` now includes the reviewed compile wrapper

The compile-only gate section already had:

- `make tooling-build`
- `make lint`

Day 11 adds:

- `make quality-review-compile`

so the benchmark docs now reflect the current reviewed local compile-quality
entry point as well.

## Validation Scope for the Final Days

Day 12 should validate the rewritten public docs against the real example and
tooling surfaces:

- `make examples`
- targeted example binaries referenced by the rewritten docs
- `make tooling-build`

Day 13 should then rerun the maintained reviewed-quality baseline:

- `make format`
- `make lint`
- `make test`
- `make quality-review-compile`
- `make quality-review`
- `make quality-review-cmake-compile`
- `make quality-review-cmake`

## Bottom Line

Day 11 finishes the support-doc layer cleanly.

The remaining Sprint 35 work is now validation and closeout, not more open
rewrite debt.
