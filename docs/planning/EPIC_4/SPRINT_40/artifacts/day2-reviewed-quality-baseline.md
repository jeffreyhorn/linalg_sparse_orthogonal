# Sprint 40 Day 2 Artifact: Reviewed-Quality Baseline

## Purpose

Record the current maintained local reviewed-quality baseline as the reference
point for Epic 4.

## Baseline Structure

The current maintained local quality surface is a layered model rather than a
single command:

### Direct commands

- `make format`
- `make format-check`
- `make lint`
- `make test`
- `make check`

### Reviewed Makefile path

- `make quality-review-compile`
  - `make format-check`
  - `make lint`
- `make quality-review`
  - `make format-check`
  - `make lint`
  - `make test`
  - `make deadcode-check`

### Reviewed CMake parity path

- `make quality-review-cmake-compile`
  - configure `build/quality-review-cmake`
  - clean serialized rebuild
  - `ctest -N`
  - Makefile/CMake test-count parity check
- `make quality-review-cmake`
  - `make quality-review-cmake-compile`
  - full `ctest`

### Strongest local reviewed baseline

- `make quality-review-full`
  - `make quality-review`
  - `make quality-review-cmake`

## Authoritative Local Semantics

### Reviewed Makefile path

The reviewed Makefile path is the authoritative local owner of:

- formatting checks
- static-analysis checks
- direct runtime test execution
- local dead-code completeness invocation

It is therefore more than “compile quality”; it is the maintained direct
reviewed local path.

### Reviewed CMake path

The reviewed CMake path is the authoritative local owner of:

- clean configure/rebuild parity
- `ctest -N` suite-truthfulness
- Makefile/CMake test-count parity
- full CTest execution

It does **not** replace Makefile-owned formatting, static-analysis, or
dead-code policy surfaces.

### `quality-review-full`

`quality-review-full` is the strongest local reviewed baseline because it
composes the reviewed Makefile path and the reviewed CMake parity path. It is
not a separate third semantics layer with different quality meaning.

## Known Operational Caveats

### Tree-mutating alternate modes

The following local modes intentionally mutate the shared build tree:

- `make sanitize`
- `make asan`
- `make sanitize-all`
- `make tsan`
- `make omp`
- `make coverage`
- `make coverage-lcov`
- `make coverage-gcovr`

Returning to the normal direct/reviewed path still requires:

- `make clean`

### Dead-code serialization

The `deadcode*` targets still share:

- `build/deadcode-cmake`
- `build/deadcode/`

So authoritative dead-code execution remains serialized.

## Day 2 Conclusion

The current maintained local reviewed baseline is already strong and mostly
coherent. The real Epic 4 opportunity is not to invent new top-level commands;
it is to tighten the ownership boundary between:

- `Makefile` command truth
- script behavior
- CI enforcement
- operator-facing `README` explanations
- later maintainer-policy surfaces
