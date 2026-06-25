# Sprint 91 Day 11: Proof Follow-Through Batch

## Purpose

Land the bounded Day 10 proof batch by proving that constructor-built
compressed-input matrices can enter the direct workflows the Sprint 91 public
story now teaches.

## Main Result

Sprint 91 now has a landed focused proof batch inside the retained public
lifecycle owner:

- `tests/test_integration.c` now proves constructor-built CSR input entering a
  one-shot direct solve
- `tests/test_integration.c` now proves constructor-built CSC input entering
  the explicit repeated-run direct lifecycle and agreeing with the one-shot
  Cholesky path across same-pattern refactors

That closes the strongest remaining Day 10 proof gap:

- constructor validity was already covered in `tests/test_csr.c`
- public direct-workflow entry from constructor-built compressed inputs is now
  covered in the integration owner

## Landed Implementation Shape

The Day 11 batch stayed exactly inside the Day 10 fence:

- required center:
  - `tests/test_integration.c`
- directly forced support follow-through:
  - none

No support-only movement was required in:

- `tests/test_csr.c`
- `README.md`
- `docs/maintainer_guide.md`

## Exact Proof Additions

The landed batch adds two focused proofs plus tiny local constructor helpers:

- local helpers:
  - `build_from_csr_constructor(...)`
  - `build_from_csc_constructor(...)`
- one-shot direct proof:
  - `test_create_from_csr_enters_one_shot_lu_workflow`
- repeated-run direct proof:
  - `test_public_lifecycle_constructor_built_csc_refactor_same_pattern_matches_one_shot_cholesky`

The proof reading is now explicit:

1. constructor-built CSR input is not only structurally valid; it actually
   enters a public one-shot direct solve cleanly
2. constructor-built CSC input can feed the explicit public
   analyze/factor/solve/refactor lifecycle
3. that repeated-run lifecycle agrees with the one-shot Cholesky path across
   same-pattern value changes

## Why No Wider Follow-Through Was Needed

The Day 11 batch filled a proof-owner gap, not a contract or implementation
gap:

- the constructors already existed and were already unit-tested
- the Day 9 README had already truthfully taught the compressed-first entry
  story
- the integration owner was simply missing the exact public-workflow proof

So the right landing was:

- one touched integration owner
- no API changes
- no documentation rewrite
- no benchmark or reporting widening

## Validation

Because `tests/test_integration.c` changed, the implementation-day queue was
run:

- `make format`
- `make lint`
- `make test`

All passed cleanly.

Representative retained proof:

- `test_integration` = `58 / 58`
- `test_csr` = `13 / 13`

## Exit State

- The Sprint 91 compressed-first product claims now have focused public-proof
  support.
- Constructor-style compressed entry is now proven both as:
  - a one-shot direct starting path
  - a valid feeder into the explicit repeated-run direct lifecycle
- Sprint 91 can now freeze its final proof-owner map from a landed validated
  batch rather than from a design-only contract.
