# Day 3 Linux Install Proof CI No-Promotion Decision

## Purpose

Day 3 applies the Day 2 promotion criteria to the current Linux CI surface and
decides whether the local Unix-side install proof scripts should become a
separate reviewed Linux CI lane.

## Decision

Do not promote `tests/test_install.sh` or `tests/test_cmake_install.sh` to a
separate reviewed Linux CI lane in Sprint 115.

The scripts remain maintained local Unix-side package proof. They should be
run before package/install claim changes, release-style validation, or future
work that changes Make install, CMake install/export, `pkg-config`, or
installed-consumer behavior.

## Evidence Reviewed

| Evidence | Result |
|---|---|
| `tests/test_install.sh` | Proves local Make install/uninstall, static archive installation, public headers, no shared artifacts, `sparse.pc`, `pkg-config` flags/version, downstream compile/link/run, maintained example compile/run, and uninstall cleanup. |
| `tests/test_cmake_install.sh` | Proves local CMake configure/build/install, static archive installation, public headers, CMake package files, `find_package(Sparse)`, exact-version behavior, mismatched-version rejection, `pkg-config` version, and installed CMake consumer run. |
| `.github/workflows/ci.yml` | Explicitly keeps focused install/package regression scripts developer-side rather than a reviewed CI lane. |
| `docs/maintainer_guide.md` | Documents install scripts as local Unix-side proof and distinguishes reviewed Linux source-of-truth lanes from local package proof. |
| `README.md` and `INSTALL.md` | Describe the maintained static package surface while preserving local install-script validation and narrower reviewed platform claims. |

## Criteria Application

| Criterion | Assessment |
|---|---|
| Can both scripts run on Ubuntu? | Likely yes: they rely on ordinary Make, C compiler, CMake, and `pkg-config` assumptions. |
| Does promotion add a new reviewed claim? | Only a narrow reviewed Linux static install/export claim. Existing public docs already present the install surface as maintained but distinguish local scripts from reviewed platform lanes. |
| Does promotion avoid duplicate cost? | No. The scripts rebuild and validate installed consumers after existing reviewed compile-quality and CMake parity gates already cover source/build health. |
| Is runtime/cache impact acceptable? | Probably acceptable in isolation, but it adds another reviewed PR-time lane with new ownership for package metadata failures. |
| Are current docs inaccurate without promotion? | No. Current workflow, maintainer guide, README, and INSTALL wording already say install scripts are local/developer-side proof. |

## No-Promotion Contract

Until a future sprint explicitly promotes these scripts:

- `tests/test_install.sh` remains the local Unix-side proof for Make
  install/uninstall plus `pkg-config`.
- `tests/test_cmake_install.sh` remains the local Unix-side proof for CMake
  install/export plus `find_package(Sparse)`.
- Linux reviewed CI remains the source of truth for Makefile compile quality,
  reviewed CMake parity, and dead-code completeness.
- Maintainers should run both install scripts when changing:
  - `Makefile` install/uninstall or `sparse.pc` behavior;
  - `CMakeLists.txt` install/export behavior;
  - `cmake/SparseConfig.cmake.in`;
  - public installed headers;
  - `examples/cmake_example`;
  - README/INSTALL package claim wording.

## Documentation Impact

No documentation or workflow wording changed on Day 3 because the existing
wording already matches the decision:

- `.github/workflows/ci.yml` says focused install/package regression scripts
  remain developer-side proof.
- `docs/maintainer_guide.md` says the install scripts are local Unix-side
  proof.
- `INSTALL.md` distinguishes local install-surface validation from reviewed
  platform claims.
- `README.md` keeps the maintained static package surface explicit and
  defers shared-library packaging.

## Non-Claims

- No reviewed Linux install CI lane was added.
- No shared-library package claim was added.
- No dynamic ABI compatibility claim was added.
- No package-manager support claim was added.
- No Windows or macOS install parity claim was added.
- No workflow, script, build metadata, public API, or install-header behavior
  changed.

## Day 3 Validation

Day 3 changes documentation only. Required validation:

```text
git diff --check
rg -n '[ \t]+$' docs/planning/EPIC_10/SPRINT_115
```

Full C quality gates are not required for Day 3 because no `.c` or `.h` files
changed.
