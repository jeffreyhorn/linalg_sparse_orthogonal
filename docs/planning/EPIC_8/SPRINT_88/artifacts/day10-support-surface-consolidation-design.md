# Sprint 88 Day 10: Support-Surface Consolidation Design

## Purpose

Define the bounded benchmark/install/support audience cleanup that Sprint 88
will actually support on its third usability lane.

## Main Result

Sprint 88 now has one exact third implementation contract:

- required Day 11 center:
  - `INSTALL.md`
- directly forced support-only follow-through only if the support batch truly
  needs it:
  - `README.md`
  - `examples/README.md`
  - `benchmarks/README.md`
  - `docs/maintainer_guide.md`
- lower-value non-touch surfaces unless the Day 11 batch truly forces them:
  - `tests/test_install.sh`
  - `tests/test_cmake_install.sh`
  - `examples/cmake_example/CMakeLists.txt`
  - `examples/cmake_example/main.c`
  - `.github/workflows/ci.yml`
  - `.github/workflows/macos-ci.yml`
  - `.github/workflows/windows-ci.yml`
  - `include/sparse_iterative.h`
  - `include/sparse_eigs.h`
  - `include/sparse_matrix.h`
  - `include/sparse_types.h`

## Ownership Split

The Day 10 ownership split is now fixed:

- operational setup and installed-consumer handoff owner:
  - `INSTALL.md`
- retained front-door handoff owner only if the Day 11 batch truly changes
  how users are routed from adoption into install/support detail:
  - `README.md`
  - `examples/README.md`
- retained benchmark-local command and proof owner only if the Day 11 batch
  truly changes benchmark-reference wording in adoption/support surfaces:
  - `benchmarks/README.md`
- retained maintainer-only policy owner only if the Day 11 batch truly
  changes repository-wide interpretation boundaries:
  - `docs/maintainer_guide.md`

## Layering Rules

The strongest layering rules are now explicit:

- `INSTALL.md` should own operational setup and installed-consumer detail
- README and example surfaces should link into that support surface without
  trying to repeat its full content
- benchmark-local command syntax and proof interpretation should remain in
  `benchmarks/README.md`, not expand through install/support cleanup
- maintainer-only policy should remain in `docs/maintainer_guide.md`, not
  drift back into user-facing support wording

## Design Decision

The strongest Day 10 design decision is now explicit:

- Sprint 88 should treat `INSTALL.md` as the bounded user-facing owner for
  setup, install, and installed-consumer guidance
- Day 11 should improve one real audience boundary there before widening any
  benchmark or maintainer surface
- the support split should stay explicit:
  - README/examples = adoption and workflow guidance
  - INSTALL = operational setup and installed-consumer detail
  - benchmarks/README = benchmark-local command and proof interpretation
  - maintainer guide = policy and ownership interpretation

## Strongest Clarification

The useful Day 10 clarification is explicit now:

- Day 11 should not become benchmark-governance rewriting or maintainer-guide
  expansion
- it should improve one real install/support audience boundary inside
  `INSTALL.md`
- public-header narrative cleanup remains explicitly later than this
  support-surface lane

## Exit State

- Sprint 88 now has one bounded third usability contract.
- Ownership between install/support detail, benchmark-local references, and
  maintainer-only policy is fixed before Day 11 begins.
- Day 11 can land one bounded support-surface batch without reopening the
  README or example lanes.
