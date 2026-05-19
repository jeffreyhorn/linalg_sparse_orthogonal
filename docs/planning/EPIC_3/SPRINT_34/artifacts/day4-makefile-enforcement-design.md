# Sprint 34 Day 4 Makefile Enforcement Design

**Date:** 2026-05-19  
**Branch:** `sprint-34`

## Objective

Define the Makefile target graph that implements Sprint 34 phase-1 reviewed
quality enforcement without breaking the existing meanings of `check`, `lint`,
`test`, or `deadcode-check`.

## Core Decision

Sprint 34 should add new reviewed-quality wrapper targets rather than mutating
the semantics of existing top-level targets.

Chosen targets:

- `quality-review-compile`
- `quality-review`

Existing targets kept intact:

- `check`
- `lint`
- `test`
- `deadcode-check`

## Target Roles

### `quality-review-compile`

Purpose:

- reviewed formatting + compile/static-analysis subset

Recommended step order:

1. `format-check`
2. `lint`

Meaning:

- formatting is clean
- library strict warning gate passes
- benchmark/example compile-only coverage passes through `lint`'s existing
  `tooling-build` dependency

### `quality-review`

Purpose:

- reviewed full local phase-1 quality contract

Recommended step order:

1. `format-check`
2. `lint`
3. `test`
4. `deadcode-check`

Meaning:

- formatting is clean
- compile/static-analysis contract passes
- Makefile-side runtime suite passes
- dead-code report completeness invariants hold

## Implementation Shape

The new targets should be serial wrapper targets implemented with recursive
`$(MAKE)` calls, not broad prerequisite fan-out.

Recommended pattern:

```make
.PHONY: quality-review-compile
quality-review-compile:
	@echo "== quality-review-compile: format-check =="
	@$(MAKE) format-check
	@echo "== quality-review-compile: lint =="
	@$(MAKE) lint

.PHONY: quality-review
quality-review:
	@echo "== quality-review: format-check =="
	@$(MAKE) format-check
	@echo "== quality-review: lint =="
	@$(MAKE) lint
	@echo "== quality-review: test =="
	@$(MAKE) test
	@echo "== quality-review: deadcode-check =="
	@$(MAKE) deadcode-check
```

## Why Serial Wrappers Instead Of Prerequisites

### 1. Better failure attribution

Phase banners make it obvious which stage failed without requiring users to
reverse-engineer mixed tool output.

### 2. Preserves the Sprint 33 dead-code execution constraint

`deadcode-check` still relies on shared paths:

- `build/deadcode-cmake`
- `build/deadcode/`

A serial wrapper is the safest truthful phase-1 integration.

### 3. Avoids ambiguous `make -j` behavior

Large prerequisite lists encourage parallel execution, which is acceptable for
some targets but the wrong default for a wrapper that now includes dead-code.

## Existing Target Interaction Rules

### Keep `check` unchanged

Current meaning:

- `format-check`
- `lint`
- `test`

Reason:

- existing users and CI jobs already understand it
- Sprint 34 should add reviewed quality behavior, not silently repurpose a
  familiar target

### Keep `lint` as the compile-quality ingress point

Do not split `tooling-build` out of `lint`.

Reason:

- Sprint 31 intentionally made `lint` the place where benchmark/example
  compile-only drift is caught
- duplicating that work elsewhere would create noise and redundant builds

### Keep `deadcode-check` unchanged

Meaning retained:

- report completeness check
- not zero findings

Reason:

- this behavior is already documented and validated from Sprint 33

## Deferred From Day 4

Not part of the Makefile design itself:

- CI wiring
- CMake parity expansion
- dead-code compile-db coverage broadening
- shared warning semantics with MSVC
- full Makefile-side strict warning gate for every test translation unit

## Day 4 Conclusion

The right Sprint 34 Makefile design is a thin additive wrapper layer:

- `quality-review-compile` for reviewed compile/static-analysis checks
- `quality-review` for the full reviewed local phase-1 contract

That gives Day 5 a concrete implementation target while preserving the repo's
existing command vocabulary and respecting the current dead-code execution
constraint.
