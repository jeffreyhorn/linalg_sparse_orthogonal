# Sprint 31 Day 10 Compile-Only Tooling Gate Design

**Date:** 2026-05-17  
**Branch:** `sprint-31`

## Objective

Define a compile-only quality gate for benchmark and example entry points that:

- catches the Sprint 31 drift class earlier
- fits the existing local validation flow
- avoids forcing maintainers to execute long-running benchmarks during routine quality checks

## Current State

### Existing local validation path

The current default local flow is:

```sh
make format && make lint && make test
```

and:

```sh
make check
```

currently expands to:

- `format-check`
- `lint`
- `test`

### What `lint` currently covers

`lint` currently runs:

- `-fsyntax-only -Werror` over `src/*.c`
- `clang-tidy` over `src/*.c`
- `cppcheck` over `src/` and `tests/`

It does **not** compile benchmark or example entry points.

### Existing tooling-side build targets

The Makefile already has:

- `bench-build`
  - builds all benchmark binaries
  - does not execute them
- `examples`
  - builds all examples
  - but is not framed as a quality gate and is not integrated into `lint` or `check`

### Why this gap matters

Sprint 31 drift lived in benchmark/example entry points rather than core-library source alone:

- stale reorder CLI/help coverage in `bench_main.c`
- feature-test-macro portability drift in benchmark entry points
- brittle positional option-struct initialization in benchmarks/examples
- public example output drift in `example_colamd.c`

Those issues were visible only after explicit benchmark/example rebuilds, not through the default `make lint` path.

## Chosen Design

## Placement

The gate should exist in **both** forms:

1. a new explicit target for focused reruns
2. integration into `make lint`

This keeps the existing default validation habit intact while still giving maintainers a narrow compile-only command for benchmark/example-only work.

## Proposed targets

### `examples-build`

Purpose:

- compile all example binaries
- no execution

Expected shape:

```make
.PHONY: examples-build
examples-build: $(EX_BINS)
	@echo "Built $(words $(EX_BINS)) example binaries (no execution)."
```

### `tooling-build`

Purpose:

- compile all benchmark and example entry points
- no execution

Expected shape:

```make
.PHONY: tooling-build
tooling-build: bench-build examples-build
	@echo "tooling-build: benchmark and example binaries built (no execution)."
```

## Integration point

`lint` should invoke or depend on `tooling-build`.

That means:

- maintainers running `make lint` automatically get benchmark/example compile coverage
- `make check` inherits the gate automatically
- maintainers doing focused reruns can still call `make tooling-build` directly

## Chosen Scope

The gate should compile:

- all `$(BENCH_BINS)`
- all `$(EX_BINS)`

Not just the Sprint 31-touched subset.

Reason:

- the drift pattern is structural, not one-off
- a subset gate would leave untouched benchmark/example binaries free to rot
- compile-only binary builds are still practical because there is no execution phase

## Gate Contract

### What it proves

- benchmark/example entry points still compile against the current headers
- benchmark/example binaries still link against the library
- compile-time drift from:
  - public option-struct growth
  - missing includes
  - feature-test-macro mismatches
  - signature changes
  - missing internal include wiring
  is caught during normal local validation

### What it does not prove

- runtime correctness
- benchmark numerical correctness
- performance regressions
- fixture availability
- authoritative full-tree warning deltas

Those remain the responsibility of:

- targeted benchmark/example reruns
- `bench-fast` / `bench`
- the Sprint 30 warning workflow
- normal test and investigation work

## Why This Design

### Why not only `check`

Rejected because the repository’s normal habit is `make format && make lint && make test`, not `make check`.

### Why not only a new explicit target

Rejected because it relies on maintainers remembering an extra command, which would weaken the “catch it earlier” goal.

### Why not syntax-only lint over benchmarks/examples

Rejected because the stronger compile/link signal is already available and better matches the drift Sprint 31 exposed.

### Why not `bench-fast`

Rejected because it executes benchmark workloads rather than just compiling entry points.

### Why not the Sprint 30 warning workflow

Rejected because that workflow is intentionally heavier and is meant for reproducible warning capture, not routine compile-only validation.

## Day 11 Implementation Guidance

The narrowest practical implementation path is:

1. add `examples-build`
2. add `tooling-build`
3. wire `tooling-build` into `lint`
4. document the new target in benchmark-facing workflow docs

This preserves the existing quality workflow while closing the Sprint 31 tooling gap with minimal conceptual overhead.

## Day 10 Conclusion

Sprint 31 does not need a new build system or a new execution-heavy validation step.

It needs a Makefile-first compile-only gate that:

- builds all benchmark and example binaries
- runs automatically through `lint`
- remains directly callable for focused reruns

That is the smallest design that would have caught the core Sprint 31 drift class earlier.
