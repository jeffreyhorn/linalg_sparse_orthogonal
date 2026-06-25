# Sprint 88 Day 4: First Usability Boundary

## Purpose

Fix the first bounded Sprint 88 usability implementation fence so the next
design pass can define one real front-door contract instead of another broad
docs or support rewrite.

## Main Result

Sprint 88 now has one explicit first implementation fence:

- required first landing:
  - `README.md`
- directly forced support surfaces only if the first landing truly needs them:
  - `INSTALL.md`
  - `benchmarks/README.md`
  - `docs/maintainer_guide.md`
  - `examples/cmake_example/CMakeLists.txt`
  - `examples/cmake_example/main.c`
- support-only proof and workflow surfaces that stay later unless the first
  landing truly forces movement:
  - `tests/test_install.sh`
  - `tests/test_cmake_install.sh`
  - `.github/workflows/ci.yml`
  - `.github/workflows/macos-ci.yml`
  - `.github/workflows/windows-ci.yml`
- explicitly deferred from the first landing:
  - examples / workflow simplification as a first-batch center
  - support-surface consolidation as a first-batch center
  - public-header / API narrative cleanup as a first-batch center
  - package/platform contract reopening
  - benchmark-policy rewriting detached from adoption guidance
  - correctness-ownership redistribution

## Strongest Clarification

The useful Day 4 clarification is now explicit:

- the best first Sprint 88 move is one bounded front-door simplification pass
  centered on `README.md`
- the first landing should decide how the repo wants the first user path,
  support references, and adoption sequence to read before example or header
  widening moves
- `INSTALL.md`, `benchmarks/README.md`, `docs/maintainer_guide.md`, and the
  example surfaces remain directly allowed support surfaces only if the
  front-door contract truly forces them to move
- install/export proof, workflow surfaces, and public-header cleanup stay
  later unless the front-door landing truly changes their obligations

## Preserved First-Batch Fence

The preserved first-batch non-goal fence is explicit now:

- no package/platform contract reopening
- no correctness-ownership redistribution
- no benchmark-policy rewrite detached from adoption guidance
- no internal architectural rewrite disguised as docs cleanup
- no workflow/platform claim broadening beyond the already-maintained proof
  and support surfaces

## Exit State

- Sprint 88 now has one bounded first front-door landing center.
- Day 5 can design one explicit front-door and support-layering contract
  inside that fence.
- Later examples/workflow simplification, support-surface consolidation, and
  public-narrative cleanup are held back until later lanes.
