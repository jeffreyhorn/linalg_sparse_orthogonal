# Day 10: macOS Backend and Toolchain Follow-Through

## Purpose

Day 10 reviews macOS backend, Homebrew GCC, coverage, OpenMP, and TSan
assumptions. The goal is to decide whether Sprint 115 should promote any new
macOS evidence lane or preserve the current reviewed/supplemental split.

## Current macOS CI Contract

| Lane | Current status | Evidence |
|---|---|---|
| Apple Clang compile quality | Reviewed | `make quality-review-compile` in `.github/workflows/macos-ci.yml`. |
| Apple Clang CMake parity | Reviewed | `make quality-review-cmake` in `.github/workflows/macos-ci.yml`. |
| Apple Clang wall-check | Reviewed macOS signal | `make wall-check` runs for each matrix leg. |
| Apple Clang sanitizer | Reviewed macOS signal | `make sanitize` runs on the Apple Clang leg. |
| Homebrew GCC | Supplemental | Direct build/test/wall-check only. |
| Make install/`pkg-config` | Supplemental | Separate `tests/test_install.sh` confidence job. |

The workflow comments already state that Homebrew GCC is supplemental
second-compiler coverage and that the Make install/`pkg-config` job does not
claim reviewed macOS install/export parity.

## Coverage Backend Decision

Do not promote macOS coverage into a reviewed Sprint 115 lane.

The Makefile already documents the backend split:

- Apple Clang routes `make coverage` to `coverage-gcovr` because Apple's LLVM
  gcov v4.2-emulation `.gcno` format is incompatible with Homebrew lcov 2.x.
- GCC routes coverage through `coverage-lcov`.
- `INSTALL.md` documents the same Apple Clang `gcovr` route and warns that
  Homebrew GCC on macOS 15+ can hit CommandLineTools SDK mismatches.

Coverage remains a useful local/supplemental signal, but it is tree-mutating,
compiler-sensitive, and not a reviewed macOS CI product claim.

## Homebrew GCC Decision

Do not promote Homebrew GCC beyond supplemental second-compiler evidence.

The current workflow installs Homebrew GCC and uses `gcc-15` for direct
build/test/wall-check coverage. That is useful drift detection, but it depends
on Homebrew's GCC formula and binary naming. It should not own reviewed macOS
package truth, install/export parity, sanitizer truth, or coverage truth.

## OpenMP and `libomp` Decision

Do not promote a macOS OpenMP reviewed lane in Sprint 115.

The Makefile already documents that Apple Clang OpenMP requires Homebrew
`libomp` and explicit flags. This remains a local capability path, not a
reviewed macOS CI contract.

## TSan Decision

Do not promote macOS TSan into reviewed CI.

The existing Makefile and INSTALL wording already capture the important
constraint: Apple Clang TSan on recent macOS can hang during dyld
initialization before test `main` executes. The maintained CI TSan signal
remains Linux-side. The local `sanitize-thread` target can use Homebrew LLVM
for focused eigensolver TSan checks, but that is not a reviewed macOS lane.

## Support Wording Assessment

No wording updates are needed for Day 10:

- `.github/workflows/macos-ci.yml` already distinguishes reviewed Apple Clang
  checks from supplemental Homebrew GCC and install confidence.
- `README.md` describes macOS as reviewed Apple Clang plus supplemental GCC
  and static-first install confidence.
- `INSTALL.md` documents Apple Clang coverage routing, Homebrew GCC caveats,
  and macOS TSan blocking.
- `docs/maintainer_guide.md` records coverage as supplemental and macOS as
  narrower than Linux's reviewed source of truth.

## Non-Claims Preserved

Day 10 does not claim:

- reviewed macOS CMake install/export parity;
- reviewed macOS coverage gating;
- Homebrew GCC reviewed-lane parity;
- macOS TSan reviewed-lane support;
- macOS OpenMP reviewed CI parity;
- macOS package-manager support;
- macOS shared-library, dylib, dynamic ABI, or runtime-loader support.

## Future Promotion Requirements

Future macOS backend/toolchain promotion should provide all of the following:

1. A stable workflow lane with pinned assumptions for compiler, Homebrew
   dependencies, SDK, and runtime.
2. Documentation that names whether the lane is reviewed or supplemental.
3. A clear support claim that does not imply install/export, package-manager,
   dynamic ABI, or Linux/Windows parity.
4. Focused validation evidence showing the lane is reliable on
   `macos-latest`.

## Validation

Day 10 is documentation-only. No workflow, Makefile, CMake, source, header, or
test-registration changes were made.
