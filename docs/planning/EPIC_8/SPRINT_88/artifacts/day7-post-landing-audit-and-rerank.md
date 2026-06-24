# Sprint 88 Day 7: Post-Landing Audit and Re-Rank

## Purpose

Re-audit the touched front-door usability surfaces after the Day 6 landing and
fix the highest-value next implementation center.

## Main Result

The Day 6 landing closed the strongest first usability contradiction:

- `README.md` no longer stands out as the unclear first adoption center
- a second immediate README-only batch is not the highest-value next move
- the strongest remaining seam is now examples / workflow simplification

## Updated Priority Map

The post-Day-6 rerank is now fixed:

- strongest next target:
  - bounded example/workflow adoption simplification centered first on
    `examples/README.md`
- strongest directly adjacent support-only follow-through:
  - `docs/tutorial.md`
  - `examples/cmake_example/CMakeLists.txt`
  - `examples/cmake_example/main.c`
  - `README.md`
- later but real Sprint 88 targets:
  - support-surface consolidation:
    - `INSTALL.md`
    - `benchmarks/README.md`
    - `docs/maintainer_guide.md`
  - public-header / API narrative cleanup:
    - `include/sparse_iterative.h`
    - `include/sparse_eigs.h`
    - `include/sparse_matrix.h`
    - `include/sparse_types.h`

## Strongest Remaining Contradiction

The strongest remaining contradiction is now explicit:

- the README now gives a clearer first-user path from build to workflow choice
  to quick-start follow-on
- but the example and tutorial adoption path still fans out across multiple
  surfaces without one compact “next step” package
- that makes examples/workflow simplification the highest-value next lane,
  not another README-only pass and not a jump to install/support cleanup

## Exact Day 8 Center

The exact Day 8 design center is now fixed to:

- `examples/README.md`

That center keeps the next lane bounded to example discovery, workflow
adoption ordering, and example/support cross-link cleanup without widening
early into package/platform or public-header work.

## Strongest Clarification

The useful Day 7 clarification is now explicit:

- Sprint 88 should not widen next into install/support consolidation just
  because the README now links to those surfaces more clearly
- it should first make the post-README example/workflow path easier to follow
- support-surface consolidation and public-header narrative cleanup remain
  explicitly later unless the example/workflow lane truly forces movement

## Exit State

- Sprint 88 now has a refreshed contradiction map grounded in the post-Day-6
  repo state.
- The next implementation center is explicit.
- Day 8 can design one exact examples/workflow contract without reopening the
  front-door lane.
