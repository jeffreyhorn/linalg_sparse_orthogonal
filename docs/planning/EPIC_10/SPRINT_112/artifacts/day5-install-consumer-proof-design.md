# Day 5 Install and Consumer Proof Design

## Purpose

Day 5 turns the Day 4 static-first support decision into a concrete validation
design for Days 6-8. The design proves the maintained static package surface
without broadening claims to shared-library packaging, dynamic ABI stability,
Windows install parity, or macOS reviewed install/export parity.

## Selected Support Tier

| field | value |
|---|---|
| selected package tier | static-first |
| shared-library support | not claimed |
| dynamic ABI support | not claimed |
| package version behavior | exact-version package metadata |
| Make install proof owner | Day 6 |
| CMake install/export proof owner | Day 7 |
| downstream consumer proof owner | Day 8 |

## Make Install Validation Plan

Primary command:

```sh
bash tests/test_install.sh
```

Expected proof:

| phase | expected behavior |
|---|---|
| clean | `make -C "$ROOT_DIR" clean` succeeds before staged install. |
| install | `make -C "$ROOT_DIR" install PREFIX="$PREFIX"` succeeds. |
| static library | `$PREFIX/lib/libsparse_lu_ortho.a` exists. |
| no shared artifacts | no `.so`, `.so.*`, `.dylib`, or `.dll` artifacts appear under staged lib/bin paths. |
| installed headers | installed header count equals source public headers plus generated `sparse_version.h`. |
| pkg-config metadata | `$PREFIX/lib/pkgconfig/sparse.pc` exists. |
| pkg-config cflags | `pkg-config --cflags sparse` returns an include path. |
| pkg-config libs | `pkg-config --libs sparse` returns `-lsparse_lu_ortho`. |
| pkg-config version | `pkg-config --modversion sparse` matches `VERSION`. |
| generated consumer | generated C consumer compiles, links, and runs through pkg-config flags. |
| maintained example source | `examples/cmake_example/main.c` compiles, links, and runs through pkg-config flags. |
| uninstall | `make uninstall PREFIX="$PREFIX"` removes library, headers, and `sparse.pc`. |

Bounded claim supported:

> The Unix-side Make install path installs and removes the maintained static
> package surface, and installed `pkg-config` metadata supports downstream
> compile/link/run consumers.

Disallowed broader claims:

- shared-library package support;
- dynamic ABI stability;
- CMake install/export behavior;
- Windows install-validation parity;
- platform-wide reviewed parity.

## CMake Install and Export Validation Plan

Primary command:

```sh
bash tests/test_cmake_install.sh
```

Expected proof:

| phase | expected behavior |
|---|---|
| configure | CMake configure succeeds with isolated build directory and staged install prefix. |
| build | CMake build succeeds. |
| install | CMake install succeeds. |
| static library | staged prefix contains `libsparse_lu_ortho.a`. |
| no shared artifacts | no `.so`, `.so.*`, `.dylib`, or `.dll` artifacts appear under staged lib/bin paths. |
| installed headers | staged prefix contains public headers under `include/sparse`. |
| CMake config | `SparseConfig.cmake` is installed. |
| CMake version file | `SparseConfigVersion.cmake` is installed. |
| CMake targets | `SparseTargets.cmake` is installed. |
| pkg-config metadata | `sparse.pc` is installed. |
| installed CMake consumer | `examples/cmake_example/` configures, builds, and runs with `CMAKE_PREFIX_PATH="$PREFIX"`. |
| exact version | `find_package(Sparse ${EXPECTED_VERSION} EXACT REQUIRED)` succeeds. |
| mismatch version | lower same-major requested version is rejected when applicable. |
| pkg-config version | `pkg-config --modversion sparse` matches `VERSION`. |

Bounded claim supported:

> The CMake install/export path installs the maintained static package surface
> and supports an installed downstream `find_package(Sparse)` consumer with
> exact-version package metadata.

Disallowed broader claims:

- dynamic ABI compatibility across versions;
- shared-library runtime-loader behavior;
- Makefile parity on every platform;
- Windows separate install-validation lane.

## pkg-config Validation Plan

The pkg-config proof is owned by the Make install script and supplemented by
the CMake install script's version check.

| check | owner | expected result |
|---|---|---|
| `PKG_CONFIG_PATH` scoping | `tests/test_install.sh` | points only at staged prefix metadata. |
| `pkg-config --cflags sparse` | `tests/test_install.sh` | includes staged include path. |
| `pkg-config --libs sparse` | `tests/test_install.sh` | includes `-L...`, `-lsparse_lu_ortho`, and `-lm`. |
| optional link flags | Make/CMake generation paths | thread/OpenMP flags appear only when configured. |
| `pkg-config --modversion sparse` | both install scripts | matches repo `VERSION`. |
| compile/link/run generated consumer | `tests/test_install.sh` | succeeds with installed headers and library. |
| compile/link/run maintained example source | `tests/test_install.sh` | succeeds with installed headers and library. |

## Downstream Consumer Proof Matrix

| Consumer | Source | Package path | Public surfaces covered | Private-surface risk |
|---|---|---|---|---|
| generated pkg-config smoke consumer | generated inside `tests/test_install.sh` | staged Make install + `pkg-config` | `<sparse/sparse_types.h>`, `<sparse/sparse_matrix.h>`, version macros, matrix create/insert/free | none; generated source includes installed public headers |
| maintained example via pkg-config | `examples/cmake_example/main.c` | staged Make install + `pkg-config` | sparse types, matrix, LU, CSR solve, version macros | none; includes `<sparse/...>` installed headers |
| installed CMake consumer | `examples/cmake_example/` | staged CMake install + `find_package(Sparse)` | installed target `Sparse::sparse_lu_ortho`, installed public headers, solver path | none; configured outside source-tree package metadata |
| exact-version CMake consumer | generated inside `tests/test_cmake_install.sh` | staged CMake install + exact `find_package` | CMake package version metadata and installed target | none |
| mismatch-version CMake probe | generated inside `tests/test_cmake_install.sh` | staged CMake install + lower version request | exact-version rejection behavior | none |

## Repeatability and Cleanup Rules

| Rule | Owner |
|---|---|
| Use `mktemp -d` for staged prefix and build directories. | both install scripts |
| Clean temporary state with `trap 'rm -rf "$TMPDIR"' EXIT`. | both install scripts |
| Scope Make proof to `PREFIX="$TMPDIR/usr"`. | `tests/test_install.sh` |
| Scope CMake proof to `-DCMAKE_INSTALL_PREFIX="$PREFIX"`. | `tests/test_cmake_install.sh` |
| Scope pkg-config lookup with `PKG_CONFIG_PATH="$PREFIX/lib/pkgconfig"`. | both install scripts |
| Scope CMake package lookup with `CMAKE_PREFIX_PATH="$PREFIX"`. | `tests/test_cmake_install.sh` |
| Avoid private headers in all consumers. | Day 6-8 proof checks |
| Treat local script proof as local/Unix-side unless a platform lane explicitly reviews it. | Sprint 112 platform-tier work |

## Day 6-8 Execution Order

1. Day 6 runs and records Make install plus pkg-config proof.
2. Day 7 runs and records CMake install/export plus CMake package proof.
3. Day 8 confirms downstream consumer coverage and identifies whether any
   additional public-header consumer proof is needed before docs alignment.

## Completion Criteria Status

- Validation commands are concrete enough to run directly.
- Proof design matches the Day 4 static-first support decision.
- Downstream checks use installed public headers and avoid private or
  planning-only scaffolding.
