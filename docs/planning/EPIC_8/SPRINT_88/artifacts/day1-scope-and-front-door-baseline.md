# Sprint 88 Day 1: Scope and Front-Door Baseline

## Purpose

Turn the Sprint 88 project-plan section and the Sprint 87 validated closeout
into one bounded front-door usability and workflow-simplification execution
package before any README, example, support-surface, or public-narrative
change lands.

## Starting Truth

Sprint 88 begins from a validated Sprint 87 close state, not from another
generic Epic 8 reset:

- strongest local reviewed baseline remains `make quality-review-full`
- reviewed CMake parity was re-materialized live and remains explicit:
  - `ctest -N --test-dir build/quality-review-cmake` = `53`
  - Makefile/CMake parity = `53 vs 53`

Sprint 87 already moved the strongest prior contradiction:

- one bounded packaging/export contract tightening landed
- one bounded local consumer-proof expansion landed
- one bounded supplemental macOS workflow follow-through package landed

That means Sprint 88 can start from the next real Epic 8 contradiction center:

- the current front-door adoption, example discoverability, support-layering,
  and public-narrative ceiling on the highest-value maintained docs, examples,
  and public surfaces

## Sprint 88 Workstreams

The highest-value Sprint 88 package is now fixed explicitly around:

- user-journey audit
- workflow-simplification design
- README / tutorial batch
- examples / workflow batch
- support-surface consolidation
- header / API narrative cleanup
- validation and closeout

## Strongest Front-Door Starting Point

The live maintained front-door contract is already more truthful than earlier
Epic 8 phases:

- the package/export story is sharper and more stable after Sprint 87
- maintained install/export proof remains real and bounded
- the example and support surfaces already point to real maintained proof
- current wording is less likely to overclaim package or ABI support than
  earlier Epic 8 phases

Sprint 88 therefore does not begin from "make docs exist." It begins from one
explicit usability question:

- how to make first adoption easier and audience boundaries clearer without
  weakening the truthfulness, proof ownership, or bounded product contract

## Strongest Likely Touch Surfaces

The live tree currently points most strongly at these Sprint 88 surfaces:

- front-door and support owners:
  - `README.md`
  - `INSTALL.md`
  - `docs/maintainer_guide.md`
  - `benchmarks/README.md`
- maintained local proof and workflow surfaces:
  - `tests/test_install.sh`
  - `tests/test_cmake_install.sh`
  - `.github/workflows/ci.yml`
  - `.github/workflows/macos-ci.yml`
  - `.github/workflows/windows-ci.yml`
- example and public-narrative surfaces:
  - `examples/cmake_example/CMakeLists.txt`
  - `examples/cmake_example/main.c`
  - `include/sparse_iterative.h`
  - `include/sparse_eigs.h`
  - `include/sparse_matrix.h`
  - `include/sparse_types.h`

## Preserved Fence

Sprint 88 is explicitly bounded against:

- reopening the package/platform semantics stabilized in Sprint 87
- redistributing correctness ownership away from the maintained proof owners
- turning benchmark-policy or maintainer detail into front-door content
- hiding internal architectural or policy churn inside generic docs cleanup
- broadening platform, ABI, or performance claims beyond what the repo
  currently maintains

## Day 1 Result

Sprint 88 now starts from one precise front-door usability and
workflow-simplification execution package rather than from a generic "improve
docs" bucket. The strongest likely touch surfaces, preserved non-goals, and
maintained front-door baseline are fixed in writing before the validation /
maintained-surface recheck begins.
