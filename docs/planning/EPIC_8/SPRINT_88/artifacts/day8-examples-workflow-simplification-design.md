# Sprint 88 Day 8: Examples / Workflow Simplification Design

## Purpose

Define the bounded example and workflow-adoption contract that Sprint 88 will
actually support on its second usability lane.

## Main Result

Sprint 88 now has one exact second implementation contract:

- required Day 9 center:
  - `examples/README.md`
- directly forced support-only follow-through only if the example/workflow
  batch truly needs it:
  - `docs/tutorial.md`
  - `examples/cmake_example/CMakeLists.txt`
  - `examples/cmake_example/main.c`
  - `README.md`
- lower-value non-touch surfaces unless the Day 9 batch truly forces them:
  - `INSTALL.md`
  - `benchmarks/README.md`
  - `docs/maintainer_guide.md`
  - `tests/test_install.sh`
  - `tests/test_cmake_install.sh`
  - `.github/workflows/ci.yml`
  - `.github/workflows/macos-ci.yml`
  - `.github/workflows/windows-ci.yml`
  - `include/sparse_iterative.h`
  - `include/sparse_eigs.h`
  - `include/sparse_matrix.h`
  - `include/sparse_types.h`

## Ownership Split

The Day 8 ownership split is now fixed:

- example discovery and adoption-order owner:
  - `examples/README.md`
- retained fuller repeated-run workflow owner only if the Day 9 batch truly
  changes how the second-step learning path should read:
  - `docs/tutorial.md`
- retained downstream installed-consumer example owner only if Day 9 truly
  changes the local-vs-installed example boundary:
  - `examples/cmake_example/CMakeLists.txt`
  - `examples/cmake_example/main.c`
- retained front-door handoff owner only if the Day 9 batch truly changes the
  exact README-to-example routing language:
  - `README.md`

## Layering Rules

The strongest layering rules are now explicit:

- `examples/README.md` should own the compact “what to run next” surface after
  the README front door
- it should make example discovery easier before the user needs the fuller
  tutorial walkthrough
- `docs/tutorial.md` should remain the deeper repeated-run and API-learning
  surface, not the first place users go after the front door
- installed-consumer examples should stay distinct from local build-tree
  examples unless the adoption contract truly requires tighter cross-linking
- install/support, benchmark, and maintainer surfaces remain later owners
  rather than being blended into the example discovery lane

## Design Decision

The strongest Day 8 design decision is now explicit:

- Sprint 88 should treat `examples/README.md` as the next-step example map
  after the README front door
- Day 9 should simplify example ordering, workflow grouping, and cross-links
  without widening into tutorial expansion or support-policy cleanup
- the support split should stay explicit:
  - examples = adoption and workflow teaching
  - tutorial = fuller repeated-run and API walkthrough
  - install/support docs = operational setup and advanced reference
  - maintainer guide = policy and ownership interpretation

## Strongest Clarification

The useful Day 8 clarification is explicit now:

- Day 9 should not become tutorial expansion, install/support cleanup, or
  benchmark-policy rewriting
- it should simplify example discovery, ordering, and cross-links inside the
  bounded example/workflow lane
- support-surface consolidation remains explicitly separate unless the
  example/workflow batch truly forces movement

## Exit State

- Sprint 88 now has one bounded second usability contract.
- Ownership between examples, tutorial, installed-consumer example surfaces,
  and retained support-only references is fixed before Day 9 begins.
- Day 9 can land one bounded examples/workflow batch without reopening the
  README-front-door lane.
