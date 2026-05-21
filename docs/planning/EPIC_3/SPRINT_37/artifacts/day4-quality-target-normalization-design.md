# Sprint 37 Day 4 Quality-Target Normalization Design

**Date:** 2026-05-20  
**Branch:** `sprint-37`

## Objective

Define a clearer maintained quality-target ownership model before editing the
Makefile layout, while folding the Sprint 36 sanitizer/build-tree caveat into
the design as a first-class operational rule.

## Executive Summary

Sprint 37 does not need more quality targets first. It needs a clearer target
topology.

The design choice for the rest of the sprint is:

- keep the existing direct and reviewed command meanings stable
- explicitly separate operator entry points from helper plumbing
- explicitly separate both of those from tree-mutating instrumentation /
  alternate-build modes
- treat `clean` as the canonical reset when returning from those alternate
  modes to the normal direct/reviewed path

This gives the later Makefile batch a concrete, low-risk goal:

- improve target organization, comments, and operator safety without reopening
  the Sprint 34 / Sprint 36 contract definitions

## Current Problem Statement

The current Makefile already contains a rich quality surface:

- atomic direct gates
- reviewed local wrappers
- reviewed CMake wrappers
- dead-code reporting/checking
- support helpers like `tooling-build` and `warning-workflow`
- instrumentation and alternate-build modes
- coverage backends

The problem is not missing functionality. The problem is that these surfaces
still read too much like one flat list.

Operational consequences:

- helper targets can look like peer operator entry points
- instrumentation/coverage modes can look like ordinary neighboring commands
  even though they intentionally mutate the shared `build/` tree
- the Sprint 36 sanitizer caveat remains easy to forget unless the ownership
  model is made explicit

## Chosen Ownership Model

### Category A: Maintained operator entry points

These are the targets maintainers should think of as the stable named commands
they run directly.

#### Direct atomic gates

- `format`
- `format-check`
- `lint`
- `test`
- `check`

#### Reviewed local / CMake wrappers

- `quality-review-compile`
- `quality-review`
- `quality-review-cmake-compile`
- `quality-review-cmake`

#### Maintained specialized validation/report surfaces

- `deadcode-report`
- `deadcode-check`
- `wall-check`

Design rule:

- these should remain the clearly documented top-level entry points
- their semantics should not change in Sprint 37

### Category B: Helper / prerequisite plumbing

These support the operator entry points but are not the main user-facing
commands of the quality contract.

- `bench-build`
- `examples-build`
- `tooling-build`
- `deadcode-compile-db`
- `deadcode`
- `warning-workflow`

Design rule:

- these should be documented and commented as plumbing/support targets
- they should not be presented as equivalent peers of the main quality gates

### Category C: Tree-mutating instrumentation or alternate-build modes

These targets intentionally rebuild the tree in a different mode and therefore
carry different safety expectations.

- `sanitize`
- `asan`
- `sanitize-all`
- `tsan`
- `omp`
- `coverage`
- `coverage-lcov`
- `coverage-gcovr`

Design rule:

- these must be signaled as alternate-build modes, not ordinary validation
  neighbors
- they own a clean-tree reset on entry
- maintainers returning from these modes to the direct/reviewed path should use
  `make clean`

## Sanitizer / Build-Tree Rule

Sprint 36 exposed the immediate symptom:

- a prior `make sanitize` run can leave an instrumented `build/` tree behind
- a later direct `make lint` run may then fail at benchmark link time unless
  the tree is cleaned first

Day 4 design conclusion:

- this is not just a `sanitize` quirk
- it is a property of the broader tree-mutating target family

Evidence already present in the Makefile:

- `sanitize`: `clean test`
- `asan`: `clean test`
- `sanitize-all`: `clean test`
- `omp`: `clean test`
- `tsan`: `clean test`
- `coverage-lcov`: `clean $(TEST_BINS)`
- `coverage-gcovr`: `clean $(TEST_BINS)`

Chosen rule:

1. Tree-mutating instrumentation/build-mode targets own the reset on entry.
2. The normal direct/reviewed quality path remains the stable baseline.
3. When maintainers return from a tree-mutating mode to that baseline, the
   canonical reset is still:
   - `make clean`

## Why `clean` Stays The Canonical Reset

The design explicitly rejects adding a second reset alias in the first pass,
such as:

- `quality-clean-build`
- `reset-build-tree`
- `reviewed-reset`

Reasoning:

- the repo already has a correct reset command: `make clean`
- the problem is ambiguity and visibility, not missing behavior
- adding a second reset name would increase target count while weakening
  ownership clarity

So Sprint 37 should normalize around the existing reset surface instead of
adding another synonym.

## Stable Semantic Commitments

Sprint 37 normalization must keep these meanings stable:

- `lint` = direct compile/static-analysis gate
- `test` = direct runtime gate
- `check` = legacy direct aggregate
- `quality-review-compile` = reviewed local compile-quality wrapper
- `quality-review` = reviewed local full reviewed wrapper
- `quality-review-cmake-compile` = reviewed CMake parity wrapper
- `quality-review-cmake` = full reviewed CMake execution wrapper
- `deadcode-check` = sibling quality category, not part of the warning-clean
  definition itself

This preserves:

- Sprint 34 reviewed-wrapper contract
- Sprint 36 enforced/staged/supplemental platform wording
- truthful separation of dead-code from the warning-clean definition

## Implementation Guidance For The Later Batch

### Highest-value changes

1. Reorganize comments/sections in the Makefile around the three categories.
2. Tighten target banners and docs so tree-mutating modes are visibly different
   from stable reviewed/direct gates.
3. Keep helper targets close to the operator entry points they support.
4. Make the `clean`-after-instrumentation expectation explicit in maintainer
   docs and reviewed-path guidance.

### Avoid in the first batch

1. Renaming established Sprint 34 / Sprint 36 command surfaces.
2. Folding dead-code into the reviewed warning-clean contract.
3. Claiming fake Windows Makefile parity or fake dead-code universality.
4. Adding new reset-target aliases that duplicate `clean`.

## Day 4 Conclusion

The right Sprint 37 target-normalization contract is now explicit:

- stable maintained operator entry points
- clearly demoted helper plumbing
- clearly signaled tree-mutating alternate-build modes
- `clean` as the single canonical reset when returning from those modes

That gives the later Makefile batch a bounded, reviewable goal:

- improve ownership, comments, layout, and operator safety
- without reopening the already-settled quality-contract semantics
