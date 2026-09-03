# Sprint 194 Day 6 Consumer Tutorial Implementation

## Objective

Implement Sprint 194 Item 194.3 by adding concise, copy-pasteable
installed-consumer tutorial guidance for Unix Make/`pkg-config` and CMake
consumers without broadening the maintained static package support contract.

## Inputs

- `docs/planning/EPIC_17/SPRINT_194/artifacts/day5-consumer-tutorial-audit.md`
- `tests/test_install.sh`
- `tests/test_cmake_install.sh`
- `sparse.pc.in`
- `cmake/SparseConfig.cmake.in`
- `examples/cmake_example/CMakeLists.txt`
- `examples/cmake_example/main.c`

## Implementation Summary

Day 6 added `INSTALL.md#installed-consumer-tutorial` as the canonical public
installed-consumer walkthrough. The tutorial uses a staged local prefix,
installed include paths, the current `pkg-config` package name, and the current
installed CMake target name.

### Unix Make/pkg-config Tutorial

Implemented tutorial details:

- staged local install with `make install PREFIX="$PWD/_install"`;
- `PKG_CONFIG_PATH` setup for `$PWD/_install/lib/pkgconfig`;
- `pkg-config --exists sparse` sanity check;
- installed-header `main.c` using `<sparse/sparse_matrix.h>` and
  `<sparse/sparse_types.h>`;
- compile/link command using `pkg-config --cflags sparse` and
  `pkg-config --libs sparse`;
- expected output markers for version, `nnz: 1`, and `OK`;
- failure wording for missing `PKG_CONFIG_PATH`, missing installed headers, or
  missing installed static archive;
- validation owner: `bash tests/test_install.sh`.

The smoke program mirrors the executable proof names:

- `sparse_create`;
- `sparse_insert`;
- `sparse_nnz`;
- `sparse_free`.

### CMake Tutorial

Implemented tutorial details:

- staged CMake install/export command using `CMAKE_INSTALL_PREFIX`;
- minimal downstream `CMakeLists.txt`;
- `find_package(Sparse REQUIRED)`;
- `Sparse::sparse_lu_ortho`;
- downstream configure/build/run commands using `CMAKE_PREFIX_PATH`;
- exact-version note for `find_package(Sparse <VERSION> EXACT REQUIRED)`;
- Windows/MSVC CMake install command;
- validation owner: `bash tests/test_cmake_install.sh`.

The tutorial states that the exported target is the installed static archive
target and does not imply shared-library, dynamic ABI, runtime-loader, or
package-manager support.

## Changed Files

| File | Change |
| --- | --- |
| `INSTALL.md` | Added the installed-consumer tutorial, updated Start Here and matrix links, expanded CMake consumer guidance, and retained static-only claim boundaries. |
| `README.md` | Linked downstream consumers to `INSTALL.md#installed-consumer-tutorial` while keeping the README summary compact. |
| `docs/tutorial.md` | Updated local-build-tree handoff links to the new installed-consumer anchors. |
| `examples/README.md` | Linked installed CMake consumer references to the new CMake/tutorial anchors. |
| `docs/maintainer_guide.md` | Pointed support ownership text at the public installed-consumer tutorial. |
| `docs/planning/EPIC_17/SPRINT_194/artifacts/day5-consumer-tutorial-audit.md` | Corrected the planning smoke program to use the actual public matrix API names from the install proof. |

## Retained Claim Boundaries

Day 6 did not add support for:

- package-manager distribution, Homebrew/core, bottles, Linuxbrew, vcpkg,
  Conan, pkgsrc, distro/system packages, or broad provider support;
- shared libraries, dynamic ABI compatibility, static/shared selectors,
  runtime-loader behavior, SONAME, install-name/RPATH, DLL/import-library
  behavior, or installed shared consumers;
- Windows Makefile parity;
- Windows `pkg-config` command execution parity;
- broad Windows parity;
- broad report freshness;
- portable performance superiority;
- broad ecosystem or state-of-the-art parity.

## Validation Plan

Because public install and package wording changed, run the install/package
wording guards in addition to markdown whitespace validation:

```sh
git diff --check
bash scripts/static_package_deferral_check.sh
bash scripts/package_manager_deferral_check.sh
make windows-powershell-guard
```

No `.c` or `.h` source files were modified by Day 6, so the full
`make format && make lint && make test` gate is not required by the sprint
review rule.
