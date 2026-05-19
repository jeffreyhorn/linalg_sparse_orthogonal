# Sprint 34 Day 2 Warning-Gate Audit

**Date:** 2026-05-19  
**Branch:** `sprint-34`

## Objective

Identify the current reviewed target groups and compiler/toolchain surfaces well
enough to define a truthful Sprint 34 phase-1 warning/enforcement scope before
Makefile, CMake, or CI wiring changes begin.

## Current Compile-Quality Models

Sprint 34 does **not** start from one unified warning contract. It starts from
three overlapping models.

### 1. Makefile `lint`

Current strict library-only compile gate:

- compiler: local default `cc` (`Apple Clang` on this branch state)
- command shape:
  - `-Wstrict-prototypes`
  - `-Wformat=2`
  - `-Werror`
  - `-fsyntax-only`
- base warnings inherited:
  - `-Wall`
  - `-Wextra`
  - `-Wpedantic`
  - `-Wshadow`
  - `-Wconversion`

Scope:

- `src/*.c` only

Supporting analysis in the same target:

- `clang-tidy` on `src/*.c`
- `cppcheck` on `src/` and `tests/`

Interpretation:

- this is already a real warning gate
- but it is still library-centric

### 2. Makefile `tooling-build`

Current compile-only tooling surface:

- `14` benchmark binaries
- `12` example binaries

Current warning flags:

- `-Wall`
- `-Wextra`
- `-Wpedantic`
- `-Wshadow`
- `-Wconversion`

Not currently applied here:

- `-Werror`
- `-Wformat=2`
- `-Wdouble-promotion`

Interpretation:

- this is already valuable compile coverage
- but it is not yet equivalent to the stricter `lint` gate

### 3. CMake / dead-code compile-db path

Current non-MSVC warning options:

- `-Wall`
- `-Wextra`
- `-Wpedantic`
- `-Wshadow`
- `-Wconversion`
- `-Wdouble-promotion`
- `-Wformat=2`
- `-Wno-unused-parameter`

Not currently applied:

- `-Werror`
- `-Wstrict-prototypes`

Current covered translation-unit counts in `build/deadcode-cmake/compile_commands.json`:

- `src`: `25`
- `tests`: `53`
- `benchmarks`: `13`
- `examples`: `6`

Interpretation:

- this path is broader than Makefile `lint` on tests and covered bench/example
  entries
- but it is narrower than Makefile `tooling-build` on total bench/example
  coverage

## Target Groups For Sprint 34 Phase 1

### First-class warning-gate candidates

- `src/*.c`

Reason:

- already has a maintained `-Werror` path in `make lint`

### Reviewed compile-quality gate candidates

- all `bench_*`
- all `example_*`

Reason:

- compile-only coverage already exists and is useful
- these programs were explicitly cleaned up in Sprint 31 and should stay honest

### Key-test parity targets

- active `ctest` suite
- `ctest -N`
- full `ctest`

Reason:

- this is the authoritative executed-suite view
- the tests matter to compile-quality truthfulness
- but there is not yet a dedicated strict Makefile-side test warning gate

### Dead-code support targets

- `deadcode-compile-db`
- `deadcode`
- `deadcode-report`
- `deadcode-check`

Reason:

- these are build-quality support targets
- but they are not direct warning-clean gates

## Compiler / Toolchain Differences

### Real phase-1 boundary

- MSVC warning parity

Why:

- `CMakeLists.txt` uses `/W3` on MSVC instead of the gcc/clang warning family
- several POSIX-only benchmarks are already gated off on Windows

Conclusion:

- Windows remains a portability path
- it should not be treated as the shared phase-1 warning-clean contract

### Real inherited tooling limitation

Current dead-code compile-db exclusion list:

- `bench_svd`
- `example_basic_solve`
- `example_condition`
- `example_iterative`
- `example_least_squares`
- `example_matrix_free`
- `example_svd_lowrank`

Conclusion:

- dead-code enforcement can begin in phase 1
- but this exclusion list must stay explicit until coverage is broadened

### Acceptable phase-1 defer

- full Makefile-side strict warning gate for every test translation unit

Why:

- tests are still compiled and executed routinely
- CMake still provides the broader active-suite warning-bearing compile path

## Current CI Mapping

### Already enforced

- Ubuntu `lint`:
  - `make format-check`
  - `make lint`
- Ubuntu build/test:
  - `make test`
  - `make sanitize`
  - `make asan`
  - `make bench-build`
  - `make bench-fast`
- CMake parity:
  - configure
  - build
  - `ctest`
  - Makefile vs CMake test-count parity
- macOS:
  - build
  - `make test`
  - `make wall-check`
  - Apple Clang `make sanitize`
- Windows:
  - configure
  - build
  - `ctest`

### Not yet enforced

- `make deadcode`
- `make deadcode-report`
- `make deadcode-check`

## Recommended Phase-1 Enforcement Matrix

| Surface | Current path | Recommendation |
|---|---|---|
| `src/*.c` | `make lint` | authoritative warning gate |
| benchmarks | `make tooling-build` / `make bench-build` | compile-quality gate |
| examples | `make tooling-build` / `make examples-build` | compile-quality gate |
| active suite | `make test`, `ctest -N`, full `ctest` | suite-truthfulness and parity target |
| Apple Clang CMake path | serialized rebuild + `ctest` | authoritative parity cross-check |
| dead-code workflow | `make deadcode-report`, `make deadcode-check` | separate quality-flow gate |
| Windows / MSVC | Windows CI | portability cross-check, not shared warning gate |

## Day 2 Conclusion

Sprint 34 phase 1 should begin from the strongest truthful split already present
in the repo:

- strict warning gate for `src/*.c`
- compile-only coverage for all benchmarks/examples
- active-suite parity through CMake and `ctest`
- dead-code completeness as a separate gate

Day 3 should turn that split into an explicit enforcement contract without
pretending the current Windows warning model, full test-tree strict compile
model, or dead-code compile-db coverage are already unified.
