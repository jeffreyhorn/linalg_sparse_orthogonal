# Sprint 68 Day 10: Property/Fuzz Expansion Batch

Date: 2026-06-13
Branch: `sprint-68`

## Purpose

Land one bounded generative/property assurance lane that materially strengthens
Sprint 68 after the Day 9 public-path oracle batch, while keeping ownership
inside the existing fuzz/property surface instead of reopening giant-test
structure churn.

## Chosen Owner

The landed batch is owned by:

- `tests/test_fuzz.c`

Why this was the right Day 10 owner:

- it already owns bounded property-style solver coverage
- it can add one deterministic lifecycle property without creating a new test
  binary or widening giant-test files again
- it complements the Day 9 exact-fixture oracle with a small seeded generative
  lane

## Chosen Property

The landed property is:

- deterministic large-`n` CSC-backed Cholesky public-lifecycle parity across
  same-pattern SPD stages

The new test is:

- `test_property_large_n_cholesky_public_lifecycle_same_pattern_csc(...)`

The property uses:

- `n = SPARSE_CSC_THRESHOLD + 12`
- fixed seeds:
  - `701`
  - `1103`
  - `1729`
- same-pattern SPD matrices generated via `random_spd(...)`
- one explicit repeated-run lifecycle lane:
  - `sparse_analyze(...)`
  - `sparse_factor_numeric(...)`
  - `sparse_factor_solve(...)`
  - `sparse_refactor_numeric(...)`
- one-shot comparison peers through:
  - `sparse_cholesky_factor_opts(...)`
  - `sparse_cholesky_solve(...)`

## What the Property Proves

For each seeded case, the test now proves:

1. baseline parity:
   - repeated-run solve matches exact solution
   - one-shot solve matches exact solution
   - repeated-run and one-shot agree
2. same-pattern refactor stage 1 parity:
   - repeated-run solve matches exact solution
   - one-shot solve matches exact solution
   - repeated-run and one-shot agree
3. same-pattern refactor stage 2 parity:
   - repeated-run solve matches exact solution
   - one-shot solve matches exact solution
   - repeated-run and one-shot agree
4. CSC-side route publication:
   - `used_csc_path == 1` at every one-shot stage

The landed tolerance contract is:

- vector agreement at `1e-10`

That tolerance is intentionally slightly looser than the Day 9 exact fixture
lane because this is a seeded generative property over larger SPD cases, not an
exact hand-tuned tridiagonal fixture.

## Why This Adds Real Assurance

This lane is additive because it complements, rather than duplicates, the Day 9
integration-owner oracle:

- Day 9 proves one exact large-`n` CSC-backed public-path parity story on a
  hand-controlled same-pattern fixture
- Day 10 proves the same lifecycle idea survives across multiple deterministic
  seeded SPD cases in the property owner

That gives Sprint 68 both:

- exact fixed-fixture assurance
- bounded generative assurance

without widening into noisy randomized volume.

## Non-Widening Fence Preserved

The landing did not widen into:

- `tests/test_integration.c`
- `tests/test_reorder_nd.c`
- `tests/test_ldlt_csc.c`
- `tests/test_iterative.c`
- `tests/test_eigs.c`
- implementation `src/` files
- benchmark/docs truth surfaces

It also did not add:

- a new fuzz harness
- a new test binary
- a shared cross-family property helper layer

## Validation

Because `*.c` changed, the required validation ran:

- `make format`
- `make lint`
- `make test`

Because this is a substantial assurance batch, the stronger reviewed baseline
also ran:

- `make quality-review-full`

All passed. The maintained reviewed anchors stayed exact:

- `ctest -N --test-dir build/quality-review-cmake` = `53`
- Makefile/CMake parity = `53 vs 53`
- full reviewed CMake `ctest` = `53 / 53`
- `Total Test time (real) = 457.07 sec`

Representative retained property output:

- `large-n CSC lifecycle property: 3/3 passed`

## Exit State

Sprint 68 Day 10 closes with one bounded property/fuzz expansion that adds real
assurance:

1. owner stayed small:
   - `tests/test_fuzz.c`
2. new property proves:
   - large-`n` CSC-backed public lifecycle parity
   - same-pattern refactor stability
   - explicit CSC route publication
   - exact-solution agreement on seeded cases
3. the batch stayed bounded:
   - no implementation edits
   - no giant-test-owner widening
   - no benchmark/docs churn

That gives Day 11 a cleaner follow-through question:

- what platform-test confidence wording actually moved now that Sprint 68 has
  changed giant-test and property/fuzz ownership on the validated branch?
