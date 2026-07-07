# Day 13 Integrated Package and Platform Validation

## Purpose

Day 13 reruns the package, consumer, platform, and documentation validation set
needed before Sprint 112 closeout. It verifies that the static-first package
support tier, platform-tier wording, and maintainer proof details still agree
with the commands and files touched by the sprint.

## Command Results

| Command | Result | Evidence |
|---|---|---|
| `bash tests/test_install.sh` | Passed | 14 passed, 0 failed. |
| `bash tests/test_cmake_install.sh` | Passed | 16 passed, 0 failed, 0 skipped. |
| `git diff --name-only -- '*.c' '*.h' 'include/*' 'src/*' 'tests/*'` | Passed | No C, header, include, source, or test files changed. |
| `git diff --name-only -- CMakeLists.txt sparse.pc.in cmake/SparseConfig.cmake.in .github/workflows/ci.yml .github/workflows/macos-ci.yml .github/workflows/windows-ci.yml Makefile` | Passed | No build-system, package metadata, workflow, or Makefile files changed. |
| package/platform wording scan | Passed | Static-first, shared-library, dynamic-ABI, reviewed subset, install-validation, and platform-tier wording remains present and bounded across docs/artifacts. |

## Package and Consumer Validation

`bash tests/test_install.sh` validated the Make install and pkg-config path:

- static library installed;
- no shared-library artifacts installed;
- all 19 public headers installed;
- `sparse.pc` installed;
- pkg-config include, library, and version metadata valid;
- generated pkg-config consumer compiled, linked, and ran;
- maintained example source compiled, linked, and ran through pkg-config;
- uninstall removed the library, headers, and pkg-config file.

`bash tests/test_cmake_install.sh` validated the CMake install/export path:

- CMake configure, build, and install passed;
- static library installed;
- no shared-library artifacts installed;
- all 19 public headers installed;
- `SparseConfig.cmake`, `SparseConfigVersion.cmake`,
  `SparseTargets.cmake`, and `sparse.pc` installed;
- `examples/cmake_example/` configured with `find_package(Sparse)`, built, and
  ran;
- exact-version `find_package` succeeded;
- mismatched-version `find_package` was rejected;
- pkg-config version reported `2.2.0`.

## Drift Checks

| Drift area | Status | Interpretation |
|---|---|---|
| Public API / installed headers | No changed `.h` or `include/*` files. | No public API or install-header drift introduced by Sprint 112 Day 13. |
| C source / tests | No changed `.c`, `src/*`, or `tests/*` files. | No C quality chain required for Day 13 docs-only changes. |
| Build/package metadata | No changed `Makefile`, `CMakeLists.txt`, `sparse.pc.in`, or `cmake/SparseConfig.cmake.in`. | Static-first and exact-version package metadata comments remain as previously validated. |
| CI workflows | No changed Linux, macOS, or Windows workflows. | Reviewed/supplemental/staged platform lanes remain as captured by Days 9-11. |
| Windows reviewed CTest surface | No workflow or CMake registration changes. | Expected Windows reviewed count remains the documented `51`; no reviewed Windows scope widened. |

## Documentation Consistency Checklist

| Surface | Day 13 consistency result |
|---|---|
| `README.md` | Keeps the compact package summary, static package surface, and cross-platform CI summary. |
| `INSTALL.md` | Keeps the maintained static-first install contract, local validation commands, and platform-tier caveats. |
| `docs/maintainer_guide.md` | Carries the detailed Sprint 112 package/platform proof snapshot and non-claims. |
| `CMakeLists.txt` | Comments preserve static-first behavior and exact-version CMake package compatibility. |
| `sparse.pc.in` | Emits package metadata without making ABI compatibility claims. |
| `.github/workflows/ci.yml` | Keeps Linux as strongest reviewed source of truth with supplemental lanes. |
| `.github/workflows/macos-ci.yml` | Keeps Apple Clang reviewed lane and supplemental GCC/install confidence split. |
| `.github/workflows/windows-ci.yml` | Keeps Windows reviewed MSVC CMake-first subset, staged exclusions, and no separate install-validation claim. |

## Residual Validation Queue

- Day 14 should rerun final documentation hygiene checks after the closeout
  and residual queue are written.
- Full `make format && make lint && make test` remains unnecessary unless
  later Day 14 work changes `.c` or `.h` files.
- Windows and macOS reviewed CI results still come from their workflows, not
  from this local macOS validation run.

## Completion Criteria

- Required package and consumer validation passed before closeout.
- No public API, install-header, helper-target, package metadata, or reviewed
  CTest drift was introduced.
- Support-tier docs match the validated package/platform behavior.
- Residuals are explicit and non-blocking.
