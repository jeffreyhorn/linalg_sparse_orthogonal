# Sprint 91 Day 9: Publication / Lifecycle Batch

## Purpose

Land the bounded Day 8 public-story batch by making compressed-first one-shot
direct entry read like a real peer workflow, while keeping the linked-list
shell and repeated-run direct lifecycle framed truthfully.

## Main Result

Sprint 91 now has a landed publication/public-surface clarification batch:

- `README.md` now teaches compressed-first one-shot direct entry through:
  - `sparse_create_from_csr(...)`
  - `sparse_create_from_csc(...)`
- the linked-list shell remains explicitly framed as:
  - mutable construction owner
  - pedagogy/compatibility owner
  - not the only natural public starting point for compressed-input callers
- the repeated-run direct workflow remains the explicit long-lived direct
  owner, with a smaller and clearer handoff from one-shot entry

That means the Day 6 compressed-first constructor APIs are no longer just
present in code and tests; they are now actually taught in the front-door
direct-workflow story.

## Landed Implementation Shape

The Day 9 landing stayed exactly inside the Day 8 fence:

- required center:
  - `README.md`
- directly forced support follow-through:
  - none

No support-only movement was required in:

- `include/sparse_matrix.h`
- `include/sparse_analysis.h`
- `docs/maintainer_guide.md`
- `tests/test_sparse_matrix.c`
- `tests/test_integration.c`

## Exact Public-Story Changes

The landed README changes now make the compressed-first story explicit in four
places:

1. `Choose a Workflow`
   - adds a compressed-first one-shot direct lane for callers that already
     own CSR or CSC inputs
2. `Quick Start`
   - points compressed-input callers toward constructor-style entry before
     widening into repeated-run direct workflows
3. `Repeated-Run Direct Workflow`
   - keeps repeated-run direct as the long-lived owner, but makes the smaller
     one-shot compressed-first path easier to distinguish from it
4. `API Overview` / `I/O and format conversion`
   - presents:
     - `sparse_create_from_csr(...)`
     - `sparse_create_from_csc(...)`
     as the primary compressed-first construction APIs
   - retains:
     - `sparse_from_csr(...)`
     - `sparse_from_csc(...)`
     as compatibility wrappers when explicit `sparse_err_t` status is wanted

## Why No Proof Or Header Follow-Through Was Needed

The Day 9 batch changed the adoption story, not the underlying API or proof
contract:

- `include/sparse_matrix.h` already remained truthful about the shell's role
- `include/sparse_analysis.h` already remained truthful about the repeated-run
  direct owner
- the existing proof owners already covered the real implementation behavior
- no new lifecycle behavior claim was introduced that needed fresh tests

So the highest-value Day 9 move stayed public-story-only.

## Validation

Because this was a substantial public-surface batch, the reviewed baseline was
rerun:

- `make quality-review-full`

It passed cleanly.

The reviewed anchors stayed exact:

- `ctest -N --test-dir build/quality-review-cmake` = `53`
- Makefile/CMake parity = `53 vs 53`
- reviewed CMake `ctest` = `53 / 53`
- reviewed CMake `Total Test time (real)` = `363.62 sec`

## Exit State

- Sprint 91 now teaches compressed-first one-shot direct entry as a real peer
  public lane.
- The linked-list shell remains real and useful, but it no longer reads like
  the only natural product model for callers who already own compressed data.
- Broader lifecycle churn remains fenced off behind this bounded Day 9
  publication landing.
