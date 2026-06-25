# Sprint 91 Day 10: Proof Follow-Through Design

## Purpose

Define the strongest focused proof follow-through still needed after the Day 6
compressed-first constructor landing and the Day 9 README/public-story
clarification.

## Main Result

Sprint 91 now has one exact Day 11 follow-through center:

- required center:
  - `tests/test_integration.c`

- directly forced support-only follow-through only if the Day 11 batch truly
  needs them:
  - `tests/test_csr.c`
  - `README.md`
  - `docs/maintainer_guide.md`

## Why `tests/test_integration.c` Is The Right Owner

The strongest remaining proof gap is no longer constructor validity by itself:

- `tests/test_csr.c` already proves that:
  - `sparse_create_from_csr(...)`
  - `sparse_create_from_csc(...)`
  produce valid shell objects
- `tests/test_integration.c` already owns the public repeated-run direct
  lifecycle and one-shot-vs-public-lifecycle agreement story
- the README now teaches compressed-first one-shot entry as a real public lane

So the highest-value remaining Sprint 91 proof is:

- show that matrices created from caller-owned compressed inputs can enter the
  direct workflows the public docs now teach
- do it inside the retained public lifecycle owner rather than by widening
  constructor-only unit coverage

That keeps the proof aligned with the actual product-model claim:

- compressed-first entry is a real public starting point
- the linked-list shell remains the mutable compatibility owner
- the repeated-run direct lifecycle remains the long-lived direct owner

## Exact Day 11 Batch Shape

The exact intended Day 11 shape is:

1. add one focused integration proof for constructor-built compressed-input
   matrices on the touched direct workflows
2. keep the proof centered on public behavior, not internal format plumbing
3. touch `tests/test_csr.c` only if a narrow constructor fixture/helper seam
   truly makes the integration owner cleaner
4. touch README or maintainer wording only if the landed proof exposes a real
   wording mismatch

## Bounded Proof Queue

The strongest bounded proof queue is now:

- constructor-built CSR/CSC matrix enters one-shot direct solve correctly
- constructor-built matrix can feed the explicit repeated-run direct lifecycle
  where that lifecycle is the public owner being taught
- agreement and residual expectations stay within the existing direct-workflow
  proof norms

## Explicit Non-Touch List

The following remain explicitly out of scope for Day 11 unless the proof batch
somehow forces them:

- constructor API or implementation changes
- sparse shell/public lifecycle implementation churn
- benchmark harness or reporting-surface widening
- broader Epic 9 external comparison work
- package/install/export follow-through
- iterative or eigensolver proof widening

## Validation Contract

The validation reading is now fixed:

- if Day 11 changes only `*.c` test surfaces:
  - `make format`
  - `make lint`
  - `make test`
- if the proof/public-surface follow-through widens materially beyond that:
  - `make quality-review-full`

## Exit State

- The remaining Sprint 91 proof need is explicit and bounded.
- Day 11 now has one exact owner:
  - `tests/test_integration.c`
- Broader benchmark and comparison work stays deferred to later Epic 9 lanes.
