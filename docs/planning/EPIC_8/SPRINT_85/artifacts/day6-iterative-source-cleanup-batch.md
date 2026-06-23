# Sprint 85 Day 6: Iterative-Source Cleanup Batch

## Purpose

Land one bounded iterative-source cleanup batch inside `src/sparse_iterative.c`
without widening into proof-owner, public-header, or direct-family cleanup.

## Main Result

Sprint 85's first implementation landing stayed inside the Day 5 fence:

- required implementation center:
  - `src/sparse_iterative.c`
- strongest support-only follow-through actually needed:
  - none
- not needed in the batch:
  - `tests/test_iterative.c`
  - `tests/test_iterative_handle_helpers.h`
  - `tests/test_integration.c`
  - `docs/maintainer_guide.md`
  - `README.md`
  - direct-family source surfaces
  - giant-test architecture surfaces

## Landed Surface

The landed cleanup remained source-owned inside `src/sparse_iterative.c`.

The batch extracted one bounded local helper seam:

- `s85_iter_result_reset`
- `s85_iter_result_mark_converged`
- `s85_iter_handle_trivial_system`

That seam now owns the highest-repeat iterative frontend boilerplate:

- result zero/reset setup
- trivial `n == 0` fast-path handling
- zero-right-hand-side converged fast-path handling

The helper seam was applied across the affected iterative frontends and
block-entry points without changing public behavior.

## Proof and Support Follow-Through

No proof-owner migration was required:

- `tests/test_iterative.c` remained the reviewed iterative proof owner
- `tests/test_integration.c` did not require lifecycle follow-through because
  the landed seam did not change repeated-run or handle semantics
- `tests/test_iterative_handle_helpers.h` did not require helper movement

No support-surface wording movement was required:

- `docs/maintainer_guide.md`
- `README.md`

## Strongest Clarification

The useful Day 6 clarification is now explicit:

- the first Sprint 85 landing can be real and valuable while staying entirely
  source-owned inside `src/sparse_iterative.c`
- the strongest early maintainability win was repeated frontend/trivial-case
  boilerplate reduction, not a broad algorithm rewrite
- proof-owner tests did not need to move for the cleanup to be legitimate
- direct-family hotspot cleanup and giant-test architecture cleanup remain
  later Sprint 85 seams, not part of the first batch

## Validation

The landed batch passed:

- `make format`
- `make lint`
- `make test`

## Exit State

- Sprint 85 now has one landed bounded iterative-source cleanup batch.
- The first maintainability move reduced local mixed-responsibility
  concentration without widening into proof-owner or support-surface churn.
- Later Sprint 85 work remains centered on direct-family hotspot cleanup and
  giant-test architecture cleanup.
