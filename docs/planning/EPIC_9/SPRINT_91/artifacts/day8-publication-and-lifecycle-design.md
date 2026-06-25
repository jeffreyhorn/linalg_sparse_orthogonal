# Sprint 91 Day 8: Publication & Lifecycle Design

## Purpose

Define the bounded second Sprint 91 implementation contract around
publication/public-surface clarification and one-shot vs repeated-run direct
workflow lifecycle clarity.

## Main Result

Sprint 91 now has one exact second implementation contract:

- required Day 9 center:
  - `README.md`

- directly forced support-only follow-through only if the Day 9 contract truly
  needs them:
  - `include/sparse_matrix.h`
  - `include/sparse_analysis.h`
  - `docs/maintainer_guide.md`
  - `tests/test_sparse_matrix.c`
  - `tests/test_integration.c`

## Required Day 9 Reading

The Day 9 center is deliberately public-surface-first rather than
header-first:

- the Day 6 code already made compressed-first construction real through
  `sparse_create_from_csr(...)` and `sparse_create_from_csc(...)`
- the strongest remaining contradiction is that the README still teaches the
  shell-first conversion story as:
  - `sparse_to_csr(mat, &csr)` / `sparse_from_csr(csr, &mat)`
  - `sparse_to_csc(mat, &csc)` / `sparse_from_csc(csc, &mat)`
- the repeated-run direct owner is already real in `sparse_analysis.h`
- so the highest-value next move is to make the README teach:
  - when compressed-first entry is the right one-shot starting path
  - when the shell-first path is still the right mutable or compatibility path
  - when callers should move to the explicit repeated-run direct lifecycle

That means Day 9 should reduce ambiguity in the public workflow story, not
start by rewriting permanent API-local contracts.

## Fixed Support-Only Follow-Through Map

The strongest support-only follow-through is now fixed to:

- `include/sparse_matrix.h`
  - only if the Day 9 README contract exposes a real mismatch in how the shell
    role is described
- `include/sparse_analysis.h`
  - only if the Day 9 README contract exposes a real mismatch in how the
    repeated-run direct owner is described
- `docs/maintainer_guide.md`
  - only if the Day 9 wording change alters maintainer-facing explanation of
    the public product split
- `tests/test_sparse_matrix.c`
  - only if the touched README contract creates a real new public-behavior
    claim that needs proof
- `tests/test_integration.c`
  - only if the touched README contract creates a real lifecycle claim that is
    not already owned by the existing integration proofs

## Explicit Non-Touch List

The following remain explicitly out of scope for the Day 9 batch unless the
README landing somehow makes them unavoidable:

- a second construction/import code batch
- broad linked-list-shell deprecation
- family-wide direct API redesign
- repeated-run direct implementation changes
- package/install/export contract reopening
- iterative/eigensolver workflow rewriting
- examples, install scripts, benchmark docs, or CI/workflow churn detached from
  the touched direct-workflow story

## Exact Day 9 Batch Shape

The exact intended Day 9 shape is:

1. tighten `README.md` around the direct-workflow adoption split
2. make compressed-first one-shot entry read like a real peer lane
3. keep the linked-list shell framed as:
   - mutable construction owner
   - pedagogy/compatibility owner
   - not the only natural public starting point
4. make the handoff from one-shot direct to repeated-run direct smaller and
   clearer
5. touch support-only headers/docs/tests only if the README contract otherwise
   becomes inconsistent with the live code and proof surface

## Exit State

- Day 9 now has one exact bounded publication/lifecycle contract.
- The second batch is still small enough to validate cleanly.
- Broader lifecycle churn remains fenced off behind the Day 9 public-story
  landing.
