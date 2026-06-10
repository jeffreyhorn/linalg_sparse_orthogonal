# Sprint 62 Day 12: Docs And Example Adoption Update

Date: 2026-06-10
Branch: `sprint-62`

## Purpose

Align the highest-value caller-facing and maintainer-facing documentation with
the landed Sprint 62 LU/Cholesky usability work, while keeping the batch
strictly docs-only and preserving the explicit deferred direct-usability
queue.

## Main Result

### 1. The README now tells the shipped direct cancellation story truthfully

The top-level callback summary in `README.md` no longer uses one generic direct
mutation sentence.

It now states the exact shipped split:

- LU no-reorder cancel-at-step-0 preserves the caller matrix
- reordered LU one-shot attempts preserve the caller matrix through a
  temporary reordered working copy
- Cholesky no-reorder linked-list cancellation remains non-bit-identical
- reordered Cholesky one-shot attempts preserve the caller matrix through a
  temporary reordered working copy
- LDL^T / QR leave the input matrix bit-identical because factor state is
  separately owned

That makes the README consistent with the Day 6-11 implementation and proof
surface instead of overgeneralizing the older in-place direct caveat.

### 2. The one-shot versus repeated-run direct workflow is now smaller and clearer

The touched public docs now converge on one practical rule:

- keep the one-shot LU / Cholesky / LDL^T / QR entries for small or occasional
  direct solves
- use a fresh matrix or a fresh `sparse_copy()` when you still need the
  original coefficient view later
- move to `example_analysis` and the explicit repeated-run direct lifecycle
  only when stable-pattern reuse is the point

This landed in:

- `README.md`
- `docs/tutorial.md`
- `examples/README.md`

### 3. The maintainer guide now owns the remaining direct-usability queue

`docs/maintainer_guide.md` now has one explicit post-Sprint-62 direct-family
interpretation block and one explicit deferred queue.

Stable interpretation now recorded there:

- one-shot LU / Cholesky / LDL^T remain first-class/default peer entry points
- repeated direct reuse belongs on the shared analyze / factor / solve /
  refactor lifecycle
- reordered LU / Cholesky one-shot preservation through temporary reordered
  working copies is now part of the stable maintainer story
- no-reorder linked-list Cholesky cancellation remains intentionally
  non-bit-identical

Deferred queue now recorded there:

- no-reorder linked-list Cholesky bit-identical cancellation restoration
- CSC progress-callback parity for Cholesky / LDL^T
- any broader LDL^T / QR wording follow-through only if a new contradiction
  appears
- broader direct-family docs/examples simplification outside the bounded
  Sprint 62 surfaces

## Touched Surface

Touched:

- `README.md`
- `docs/tutorial.md`
- `docs/maintainer_guide.md`
- `examples/README.md`

Not widened into:

- public headers
- example source files
- tests
- benchmarks
- implementation files

## Sanity Checks

Checks run:

- `git diff -- README.md docs/tutorial.md docs/maintainer_guide.md examples/README.md`
- terminology/alignment `rg`
- touched-surface `wc -l`
- branch status recheck

Measured touched-surface result:

- `README.md`: `982 -> 983`
- `docs/tutorial.md`: `454 -> 464`
- `examples/README.md`: `134 -> 142`
- `docs/maintainer_guide.md`: `367 -> 391`

Because this was a docs-only batch, the code/header gate was not required:

- `make format`
- `make lint`
- `make test`
- `make quality-review-full`

## Exit State

Sprint 62 now has a coherent documentation/adoption story on the touched
direct-usability surfaces:

- the README matches the shipped LU/Cholesky cancellation split
- tutorial/example docs now point callers cleanly toward one-shot usage versus
  explicit repeated-run reuse
- maintainers have one clear residual queue for the remaining direct-family
  follow-through
