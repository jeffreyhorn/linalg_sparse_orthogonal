# Sprint 97 Day 13: Validation And Residual Queue

## Purpose

Day 13 validates the Sprint 97 build/package/platform convergence work and
freezes the residual queue for Day 14 closeout and Sprint 98 handoff.

## Touched Surface Summary

Sprint 97 currently touches these active repository surfaces:

- `Makefile`
- `build-metadata/library_sources.txt`
- `scripts/check_library_sources.py`
- `CMakeLists.txt`
- `README.md`
- `INSTALL.md`
- `docs/maintainer_guide.md`
- `tests/test_install.sh`
- `tests/test_cmake_install.sh`
- `.github/workflows/windows-ci.yml`
- Sprint 97 planning artifacts and working notes

No `.c` or `.h` implementation/header files were modified.

## Validation Run

Day 13 ran the strongest required validation for the touched surfaces:

```sh
python3 scripts/check_library_sources.py
make quality-review-compile
make quality-review-cmake-compile
bash tests/test_install.sh
bash tests/test_cmake_install.sh
git diff --check
rg -n "[ \t]+$" .github/workflows/windows-ci.yml README.md INSTALL.md docs/maintainer_guide.md tests/test_install.sh tests/test_cmake_install.sh docs/planning/EPIC_9/SPRINT_97 build-metadata scripts/check_library_sources.py Makefile CMakeLists.txt
```

Observed results:

- direct source-list checker: `source-list-check: PASS (42 library sources)`
- `make quality-review-compile`: passed
  - `format-check`
  - `source-list-check`
  - `lint`
- `make quality-review-cmake-compile`: passed
  - configure
  - clean rebuild
  - `ctest -N`
  - Makefile/CMake test-count parity
  - CMake tests: 54
  - Makefile tests: 54
- `bash tests/test_install.sh`: 14 passed, 0 failed
- `bash tests/test_cmake_install.sh`: 16 passed, 0 failed, 0 skipped
- `git diff --check`: passed
- trailing-whitespace scan: passed with no matches

Because no `.c` or `.h` files were modified, the full
`make format && make lint && make test` chain was not required. The reviewed
compile-quality path was still run because the sprint changed the Makefile and
added the source-list checker.

## Duplication Delta From Day 2

Day 2 ranked the duplicated build/product surfaces. Day 13 rechecks that map
against the final tree.

| Day 2 candidate | Day 13 status |
| --- | --- |
| Library source list | Reduced silent drift risk. `build-metadata/library_sources.txt` and `scripts/check_library_sources.py` now validate manifest, Makefile `LIB_SRCS`, and CMake `add_library(...)` membership/order. |
| Test registration list | Preserved as independent proof. Local Make/CMake counts still match at 54; Windows remains a smaller reviewed CMake subset with expected CTest count 51. |
| Benchmark registration list | Residual. No change; still explicit in Make and CMake. |
| Example registration list | Residual. No change; Make wildcard and explicit CMake targets remain. |
| Install/export package proof | Strengthened. Static-first decision is explicit; install scripts now assert no shared-library artifacts are installed. |
| Workflow messages and expected counts | Calibrated. README/INSTALL/Makefile command wording agrees, Windows workflow keeps count/exclusions visible, stale sprint-history wording was removed from active workflow output. |

## Sprint 98 Candidates

Ranked candidates for Sprint 98 or later:

1. Test registration convergence or stronger parity guard that preserves
   platform exclusions, local Make/CMake parity, and Windows expected-count
   assertions.
2. Benchmark registration reduction only if a manifest/checker pattern remains
   simple and does not hide benchmark subsets such as `bench-fast`.
3. Example registration reduction only if it improves review cost without
   obscuring CMake target names.
4. Optional checker self-test fixture for `scripts/check_library_sources.py` if
   the parser grows beyond the current narrow Make/CMake structure.
5. Possible Windows CMake install-validation lane only after explicit design
   and proof ownership.

## Package/Product Non-Claims

These remain deliberate non-claims:

- shared-library build output
- shared-library install/export metadata
- dynamic ABI/version compatibility guarantee
- package-manager integration
- full reviewed macOS install/export parity
- separate reviewed Windows install-validation lane
- Windows Makefile parity
- Windows DLL/import-library packaging

## Platform Proof Gaps

Platform gaps that remain explicit:

- Windows Makefile reviewed wrappers
- Windows dead-code flow
- Windows `test_threads`
- Windows `test_sprint4_integration`
- Windows `test_fuzz`
- macOS dead-code lane
- macOS full reviewed install/export parity
- shared-library package lane on every platform

## Preserved Independent Proof

The following repeated surfaces remain intentionally independent:

- Make `LIB_SRCS` and CMake `add_library(...)` stay directly reviewable even
  though the manifest/checker now guards drift.
- `make quality-review-cmake-compile` retains Make/CMake test-count parity.
- Windows workflow keeps `EXPECTED_WINDOWS_CTEST_COUNT: "51"` and staged
  exclusion output in CI logs.
- Make install/`pkg-config` proof remains separate from CMake
  install/`find_package` proof.
- Linux, macOS, and Windows workflow comments keep platform scope visible where
  maintainers encounter CI proof.

## Day 13 Result

Sprint 97 has no hidden build/package/platform contradiction in the touched
surfaces. Validation passed for the source-list guard, reviewed Make
compile-quality path, reviewed CMake compile/parity path, static-first
install/export proof, and documentation/workflow hygiene. The remaining work is
ranked and explicitly preserved as residual, non-claim, or independent proof.
