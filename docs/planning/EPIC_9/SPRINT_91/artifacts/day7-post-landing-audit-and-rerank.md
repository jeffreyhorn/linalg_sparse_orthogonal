# Sprint 91 Day 7: Post-Landing Audit & Rerank

## Purpose

Re-rank the remaining compressed-first work after the first code landing so
Sprint 91's second implementation center is chosen from live post-Day-6
evidence rather than from the original shell-cost audit.

## Main Result

The Day 6 landing closed the strongest first Sprint 91 contradiction:

- compressed CSR/CSC inputs now have first-class public construction entry
  paths
- callers that already own compressed sparse data no longer need to begin
  conceptually from `sparse_create()` plus linked-list insertion just to enter
  the matrix-shell workflow
- the first construction/import seam is no longer the highest-value remaining
  Sprint 91 target

That changes the ranked remaining shell-cost map to:

- strongest first target now:
  - publication and public-surface clarification around the new
    compressed-first entry path
- strongest second target now:
  - one-shot vs repeated-run direct-workflow lifecycle clarification
- strongest third target now:
  - focused proof-owner or integration follow-through only if the publication
    and lifecycle contract truly forces it
- strongest support-only but real target now:
  - README, maintainer, and public-header wording that still over-centers the
    linked-list shell after the Day 6 entry-path landing

## Why The Rerank Changed

Day 6 materially changed the product reading in one important way:

- `include/sparse_csr.h` now exposes:
  - `sparse_create_from_csr(const SparseCsr *csr)`
  - `sparse_create_from_csc(const SparseCsc *csc)`
- the retained `sparse_from_*` imports now read as compatibility wrappers
  rather than as the only public compressed-input entry path
- `tests/test_csr.c` now proves that direct constructor-style compressed entry
  is real

That means the strongest remaining contradiction is no longer "can callers
with compressed data enter directly?" It is now "does the surrounding public
story actually teach and contextualize that path correctly?"

## Strongest Remaining Contradiction

The strongest remaining contradiction is now publication/public-surface
reading:

- `README.md` still teaches CSR/CSC conversion as:
  - `sparse_to_csr(mat, &csr)` / `sparse_from_csr(csr, &mat)`
  - `sparse_to_csc(mat, &csc)` / `sparse_from_csc(csc, &mat)`
- the README still presents the shell-first one-shot path as the more natural
  public center even though compressed-first construction now exists
- `include/sparse_matrix.h` still truthfully describes the shell as the
  mutable compatibility owner, but the public adoption story around that owner
  has not yet been recalibrated against the new Day 6 entry path

This now outranks lifecycle clarification because the lifecycle story cannot
read cleanly until the public entry points themselves are described more
coherently.

## Lifecycle Reading After Day 6

Lifecycle clarification remains real, but it is now second:

- `include/sparse_analysis.h` already gives the repo a real explicit
  repeated-run direct owner
- `README.md` already teaches the repeated-run direct workflow
- the remaining gap is not that the repeated-run owner is missing
- the gap is that the shell-first and compressed-first one-shot entry story
  still needs a cleaner relationship to the repeated-run owner

So the useful order is now:

1. clarify the public publication and entry-path reading
2. tighten one-shot vs repeated-run direct lifecycle wording around that
   clarified public story
3. only then widen proof or support follow-through if the touched contract
   truly forces it

## Exact Day 8 Design Center

The exact Day 8 design center is now fixed to:

- `README.md`

The strongest support-only follow-through, only if the Day 8 contract truly
forces movement, is:

- `include/sparse_matrix.h`
- `include/sparse_analysis.h`
- `docs/maintainer_guide.md`
- `tests/test_sparse_matrix.c`
- `tests/test_integration.c`

## Explicit Non-Needs After Day 6

Sprint 91 no longer needs:

- a second immediate construction/import implementation batch
- broad linked-list-shell deprecation
- a family-wide direct-workflow rewrite
- proof-owner widening detached from the touched public contract

## Exit State

- The strongest remaining Sprint 91 seam is now explicit after the first
  implementation landing.
- The second implementation center is fixed first to publication/public-surface
  clarification, with lifecycle tightening ordered immediately behind it.
- Day 8 can define one exact bounded publication/lifecycle contract from the
  live post-Day-6 tree.
