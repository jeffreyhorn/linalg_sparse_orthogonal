# Sprint 34 Day 3 Warning-Gate Design

**Date:** 2026-05-19  
**Branch:** `sprint-34`

## Objective

Define the first Sprint 34 enforcement contract clearly enough that Days 4-6
can wire it into the Makefile and later CI/CMake follow-up without collapsing
compile-quality, runtime-quality, and dead-code quality into one ambiguous
target.

## Core Decision

Sprint 34 phase 1 should preserve the repo's existing command meanings and add
an aggregate enforcement layer above them.

It should **not**:

- redefine `make lint` to mean "all quality checks"
- redefine `make test` to mean "compile-only tooling coverage"
- redefine `make deadcode-check` to mean "zero dead-code findings"
- immediately overload `check` without a clear step boundary

## Phase-1 Contract

### Authoritative local compile/static-analysis path

- `make format-check`
- `make lint`

Meaning:

- formatting is clean
- `src/*.c` remains clean under the strict Makefile warning gate
- benchmark/example entry points still compile via `tooling-build`
- library static-analysis checks still pass

### Authoritative local runtime path

- `make test`

Meaning:

- Makefile-side test binaries build and pass

### Authoritative local dead-code path

- `make deadcode-check`

Meaning:

- the dead-code report regenerates successfully
- every `xunused` finding is categorized
- the coverage-gap section remains present

Meaning it does **not** have:

- "the repo has zero dead-code findings"

### Authoritative parity path

- serialized Apple Clang CMake rebuild
- `ctest -N`
- full `ctest`

Meaning:

- the active CMake suite is still visible, countable, and passing
- the Makefile quality story still lines up with the maintained CMake path

## Benchmark / Example Enforcement Rule

Benchmarks and examples stay compile-only in phase 1.

Included:

- `make tooling-build`
- `make bench-build`
- `make examples-build`

Not folded into the compile-quality gate:

- `make bench`
- `make bench-fast`
- `make wall-check`
- full example execution

Reason:

- compile drift and runtime/performance drift are different failure classes
- Sprint 31 already established the compile-only tooling gate for exactly this
  separation

## Dead-Code Integration Rule

Dead-code should be integrated into the reviewed quality flow as a sibling step,
not as part of the warning-clean definition.

Reason:

- warning regressions and dead-code-report completeness failures are different
  categories
- the dead-code workflow still has explicit inherited limitations:
  - compile-db exclusion list
  - serial/shared-build-tree execution model

## Include / Exclude List

### Include in phase 1

- strict `src/*.c` warning cleanliness
- benchmark/example compile-only reviewed coverage
- Makefile runtime test pass
- CMake suite registry/execution parity
- dead-code report completeness

### Exclude from the phase-1 core contract

- shared warning semantics with MSVC `/W3`
- full strict warning gate for every test translation unit in the Makefile path
- full benchmark/example runtime execution
- any claim that dead-code coverage is complete beyond the documented exclusions

## Recommended Aggregate Direction For Days 4-6

Add a new top-level reviewed-target aggregate target that sequences:

1. `make format-check`
2. `make lint`
3. `make test`
4. `make deadcode-check`

Design properties required:

- stepwise attributable output
- no hidden shell-chain behavior
- preserves direct use of each underlying command
- does not break existing `check` semantics unless a later deliberate choice is
  made

## Day 3 Conclusion

Sprint 34 phase 1 should define one reviewed local quality contract and one
supporting CMake parity contract. That is enough structure to harden the
workflow without overstating what Windows, the full test tree, or the current
dead-code compilation database prove today.
