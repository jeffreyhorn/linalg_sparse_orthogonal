# Sprint 88 Day 5: Workflow-Simplification Design

## Purpose

Define the bounded adoption-guidance and support-layering contract that Sprint
88 will actually support on its first front-door usability lane.

## Main Result

Sprint 88 now has one explicit first implementation contract:

- required implementation center:
  - `README.md`
- directly forced support surfaces only if the first batch truly needs them:
  - `INSTALL.md`
  - `benchmarks/README.md`
  - `docs/maintainer_guide.md`
  - `examples/cmake_example/CMakeLists.txt`
  - `examples/cmake_example/main.c`
- proof and workflow surfaces remain later owners unless the first batch
  truly changes their obligations:
  - `tests/test_install.sh`
  - `tests/test_cmake_install.sh`
  - `.github/workflows/ci.yml`
  - `.github/workflows/macos-ci.yml`
  - `.github/workflows/windows-ci.yml`
- public-header narrative cleanup remains later than the first front-door
  landing:
  - `include/sparse_iterative.h`
  - `include/sparse_eigs.h`
  - `include/sparse_matrix.h`
  - `include/sparse_types.h`

## Ownership Split

The Day 5 ownership split is now fixed:

- front-door adoption-guidance owner:
  - `README.md`
- retained example/workflow adoption owners only if the front-door landing
  truly changes how downstream examples should be sequenced:
  - `examples/cmake_example/CMakeLists.txt`
  - `examples/cmake_example/main.c`
- retained support-only advanced-reference owners only if the front-door
  landing truly changes where operational detail should live:
  - `INSTALL.md`
  - `benchmarks/README.md`
- retained maintainer-only detail owner only if the front-door landing truly
  changes audience boundaries:
  - `docs/maintainer_guide.md`
- retained proof and workflow evidence owners after the first landing:
  - `tests/test_install.sh`
  - `tests/test_cmake_install.sh`
  - `.github/workflows/ci.yml`
  - `.github/workflows/macos-ci.yml`
  - `.github/workflows/windows-ci.yml`
- retained public narrative owners after the first landing:
  - `include/sparse_iterative.h`
  - `include/sparse_eigs.h`
  - `include/sparse_matrix.h`
  - `include/sparse_types.h`

## Layering Rules

The strongest layering rules are now explicit:

- `README.md` should own the first user path:
  - choosing a workflow
  - finishing one minimal quick-start path
  - understanding when to widen into repeated-run direct usage
  - understanding where iterative/eigensolver, install/support, and advanced
    references begin
- example surfaces should reinforce adoption after the README path is clear,
  not compete with the front door during the first read
- install/support docs should own operational setup, package-consumer detail,
  and advanced reference material rather than expanding the front-door path
- maintainer-facing material should keep internal policy, deeper workflow
  ownership, and maintenance detail out of the first-user read path
- public-header narrative cleanup remains a later lane and should not be
  folded into the first README batch without a real front-door contract need

## Design Decision

The strongest Day 5 design decision is now explicit:

- Sprint 88 should treat `README.md` as an adoption-first surface
- it should not try to teach every benchmark, support, maintainer, and API
  nuance in the same front-door flow
- the preferred first adoption sequence should read like:
  - choose a workflow
  - run a minimal quick start
  - widen to repeated-run direct workflows
  - widen later to iterative/eigensolver, examples, install/support, and
    benchmark references only when the user needs them

## Strongest Clarification

The useful Day 5 clarification is explicit now:

- Day 6 should not try to solve examples, install/support, maintainer, and
  public-header cleanup all at once
- it should simplify the README front door so the first adoption sequence is
  easier to follow and hands off deliberately to later surfaces
- it should preserve the maintained proof, workflow, and package-contract
  owners as later lanes rather than blending them into the first README pass

## Preserved First-Batch Fence

The preserved first-batch fence is explicit:

- no package/platform contract reopening
- no correctness-ownership redistribution
- no benchmark-policy rewrite detached from adoption guidance
- no workflow/platform claim broadening beyond the already-maintained proof
  and support surfaces
- no public-header narrative widening folded into the first README batch
  unless the front-door contract truly forces it

## Exit State

- Sprint 88 now has one bounded front-door usability contract.
- Ownership between README, examples, support references, maintainer detail,
  retained proof/workflow evidence, and later public-header narrative cleanup
  is fixed before Day 6 begins.
- Day 6 can land one bounded README/tutorial batch without reopening design
  questions or widening the sprint scope early.
