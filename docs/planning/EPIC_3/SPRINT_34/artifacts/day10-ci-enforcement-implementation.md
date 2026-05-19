# Sprint 34 Day 10: CI enforcement implementation

## Goal

Implement the first Sprint 34 CI enforcement pass in the primary Linux workflow without changing the purpose of the existing sanitizer, runtime, portability, or MSVC jobs.

## Shipped workflow changes

### Reviewed Makefile compile-quality job

The Linux `lint` job now uses the maintained reviewed wrapper:

- install tools
- `make quality-review-compile`

This replaces the old split:

- `make format-check`
- `make lint`

The quality surface is unchanged, but the CI output now follows the reviewed wrapper banners directly.

### Reviewed CMake parity job

The Linux `cmake-build-and-test` job now uses:

- `make quality-review-cmake`

This replaces the hand-written workflow-local sequence for:

- configure
- build
- full `ctest`
- separate test-count comparison step

The reviewed wrapper is now the single maintained source of the CMake parity contract.

### Serialized dead-code job

Added a new Linux `deadcode` job that:

1. installs:
   - `cppcheck`
   - `clang-18`
   - `llvm-18-dev`
   - `libclang-18-dev`
   - `ninja-build`
2. clones and builds upstream `xunused`
3. installs `xunused` into `$HOME/.local`
4. runs:
   - `make deadcode-report`
   - `make deadcode-check`
5. uploads artifacts on `if: always()`

Uploaded artifacts:

- `build/deadcode/report.md`
- `build/deadcode/report.tsv`
- `build/deadcode/coverage-notes.txt`
- `build/deadcode/cppcheck.txt`
- `build/deadcode/xunused.txt`

## Preserved execution model

The dead-code flow remains intentionally serialized:

- one Linux job
- no matrix
- no parallel sibling dead-code targets
- one shared build/artifact pair:
  - `build/deadcode-cmake`
  - `build/deadcode/`

That preserves the Sprint 33 shared-path constraint instead of papering over it.

## Local validation

Because this day changed workflow YAML only, the validation was workflow-structure-focused:

- reviewed the resulting `.github/workflows/ci.yml`
- confirmed the intended job anchors:
  - `cmake-build-and-test`
  - `lint`
  - `deadcode`
- confirmed `build-and-test`, `tsan`, and `coverage` stayed unchanged

## Deferred scope

Still intentionally outside Sprint 34 phase 1:

- macOS CI enforcement expansion
- Windows/MSVC warning/dead-code enforcement expansion
- dead-code compile-db gap closure
- sanitizer / runtime / coverage job repurposing

## Result

Sprint 34 now has a first CI enforcement pass that runs the maintained reviewed paths directly:

- `make quality-review-compile`
- `make quality-review-cmake`
- `make deadcode-report`
- `make deadcode-check`

That is enough for phase 1 without conflating reviewed compile-quality enforcement with unrelated CI responsibilities.
