# Day 11 macOS Follow-Through

## Purpose

Day 11 confirms macOS package and platform truth after the Sprint 112
static-first support decision. It distinguishes reviewed macOS CI evidence from
local package validation and records backend, SDK, OpenMP, install/export, and
package non-claims.

## Local macOS Environment

| Surface | Observed value |
|---|---|
| Kernel | `Darwin 24.6.0` |
| macOS | `15.7.3` (`24G419`) |
| Default C compiler | Apple Clang `11.0.0` from Command Line Tools |
| CMake | `4.3.2` |
| pkg-config | `2.5.1` |

## Reviewed macOS CI Surface

| Lane | Evidence | Interpretation |
|---|---|---|
| Apple Clang reviewed path | `.github/workflows/macos-ci.yml` runs `make quality-review-compile`, `make quality-review-cmake`, `make wall-check`, and `make sanitize`. | Reviewed macOS platform lane. |
| Homebrew GCC leg | `.github/workflows/macos-ci.yml` runs direct build/test and `wall-check` with `gcc-15`. | Supplemental second-compiler confidence. |
| Make install/pkg-config job | `.github/workflows/macos-ci.yml` runs `bash tests/test_install.sh`. | Supplemental static-first package confidence only. |

## Local Package Validation

| Command | Result | Evidence boundary |
|---|---|---|
| `bash tests/test_install.sh` | Passed: 14 passed, 0 failed. | Local Make install/uninstall and `pkg-config` proof on this macOS host. |
| `bash tests/test_cmake_install.sh` | Passed: 16 passed, 0 failed, 0 skipped. | Local CMake install/export, exact-version package, pkg-config version, and installed CMake consumer proof on this macOS host. |

## Local Make Install Proof Details

The Make install proof validated:

- static library installed;
- no shared-library artifacts installed;
- all 19 public headers installed;
- `sparse.pc` installed;
- `pkg-config --cflags` returned the staged include path;
- `pkg-config --libs` returned the library flag;
- `pkg-config --modversion` returned `2.2.0`;
- a generated downstream pkg-config consumer compiled, linked, and ran;
- `examples/cmake_example/main.c` compiled, linked, and ran through the staged
  pkg-config install;
- uninstall removed the library, headers, and pkg-config file.

## Local CMake Install Proof Details

The CMake install/export proof validated:

- CMake configure, build, and install;
- static library installed;
- no shared-library artifacts installed;
- all 19 public headers installed;
- `SparseConfig.cmake`, `SparseConfigVersion.cmake`,
  `SparseTargets.cmake`, and `sparse.pc` installed;
- `examples/cmake_example/` configured with `find_package(Sparse)`;
- the installed CMake consumer built and ran;
- exact-version `find_package` succeeded;
- mismatched-version `find_package` was rejected;
- `pkg-config` reported version `2.2.0`.

## macOS Claim Decision

No macOS claim needs tightening on Day 11. Current public, maintainer, and CI
wording already matches the evidence:

- macOS has a reviewed Apple Clang lane;
- Homebrew GCC remains supplemental second-compiler confidence;
- Make install/`pkg-config` proof is supplemental package confidence;
- local CMake install/export proof can be recorded as local evidence, but it
  is not promoted to a reviewed macOS install/export parity claim;
- the selected package tier remains static-first.

## macOS Non-Claims

- No full reviewed install/export parity claim for macOS.
- No reviewed macOS dead-code completeness lane.
- No reviewed macOS coverage parity claim.
- No dynamic ABI compatibility claim.
- No shared-library, dylib, install-name, runtime-loader, or symbol-visibility
  support claim.
- No package-manager support claim.
- No guarantee that Homebrew GCC sanitizer behavior matches Apple Clang or
  Linux sanitizer behavior.
- No claim that macOS TSan is supported; the maintained TSan lane remains
  Linux-side.
- No claim that local package proofs replace reviewed CI lanes.

## Backend, SDK, and Runtime Notes

- `INSTALL.md` already documents Apple Clang build/test basics, OpenMP setup
  through Homebrew `libomp`, and coverage backend differences.
- The Homebrew GCC lane remains supplemental because compiler package versions
  and SDK interaction can float over time.
- The macOS install/pkg-config CI job strengthens the static-first package
  story but does not validate the full CMake install/export surface in the
  reviewed macOS workflow.
- Shared-library runtime-loader behavior is outside the Sprint 112 support
  tier.

## Documentation and Workflow Assessment

| Surface | Assessment | Change needed on Day 11? |
|---|---|---:|
| `.github/workflows/macos-ci.yml` header comments | Already distinguish reviewed Apple Clang, supplemental Homebrew GCC, and supplemental install/pkg-config proof. | No |
| `INSTALL.md` macOS section | Already documents Apple Clang, coverage backend differences, OpenMP setup, and platform tier boundaries. | No |
| `INSTALL.md` verification section | Already distinguishes local install scripts from reviewed platform confidence. | No |
| `README.md` compact CI summary | Already says macOS has Apple Clang reviewed path with supplemental GCC and static-first install/pkg-config verification. | No |
| `docs/maintainer_guide.md` package/platform ownership | Already records macOS as narrower with supplemental static-first install confidence. | No |

## Residual macOS Queue

- Promote macOS CMake install/export parity only if CI adds a reviewed
  `tests/test_cmake_install.sh` or equivalent installed-consumer lane.
- Promote macOS coverage parity only after backend/tooling behavior is made
  stable enough to own as reviewed evidence.
- Revisit Homebrew GCC version assumptions when Homebrew's default GCC changes.
- Revisit macOS TSan only if the upstream dyld/runtime limitation is resolved
  and a reviewed lane is added.
- Revisit shared-library or dylib support only if a future sprint changes the
  static-first support decision.

## Completion Criteria

- macOS support wording matches reviewed and local validation evidence.
- Backend, SDK, OpenMP, install/export, and package boundaries are explicit.
- macOS residuals are ready for Sprint 112 closeout.
