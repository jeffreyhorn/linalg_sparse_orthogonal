# Sprint 194 Day 5 Consumer Tutorial Audit

## Objective

Define the installed-consumer tutorial contract for Sprint 194 Item 194.3
before rewriting public tutorial content. The audit identifies the minimal
copy-pasteable `pkg-config` and CMake consumer flows that match the current
install evidence, separates source-tree developer workflows from
installed-prefix workflows, and preserves unsupported package, shared library,
dynamic ABI, and broad Windows claims as exclusions.

## Evidence Reviewed

| Source | Tutorial relevance |
| --- | --- |
| `INSTALL.md` | Owner for start-here setup, support/readiness status, static install contract, installed file list, `pkg-config`, CMake, Windows, and install verification. |
| `README.md` | Short installation summary and routing link to the support/readiness matrix; should not duplicate full installed-consumer tutorials. |
| `docs/tutorial.md` | Local build-tree learning path; routes installed downstream consumers to `INSTALL.md`. |
| `examples/README.md` | Local example catalog plus installed-consumer handoff to `examples/cmake_example/` and `INSTALL.md`. |
| `examples/cmake_example/` | Maintained installed CMake downstream example using `find_package(Sparse)` and `Sparse::sparse_lu_ortho`. |
| `tests/test_install.sh` | Proof owner for Make install/uninstall, installed files, `sparse.pc`, `pkg-config` cflags/libs, and a compiled installed consumer. |
| `tests/test_cmake_install.sh` | Proof owner for CMake install/export, exact-version `find_package(Sparse)`, generated package files, and downstream example build/run. |
| `sparse.pc.in` | Metadata owner for the current `pkg-config` name, cflags, libs, version, and static archive description. |
| `cmake/SparseConfig.cmake.in` | Metadata owner for the installed CMake package entrypoint. |
| `packaging/homebrew/README.md` | Provider-proof and blocker provenance only; not a user-facing package-manager install path. |

## Current Consumer Surface

### Source-Tree Developer Workflow

The source-tree workflow is for contributors or users building directly from a
checkout:

- build with `make`, `make test`, `make examples`, or focused quality targets;
- run examples from `./build/...`;
- compile local experiments with `-Iinclude` and `-Lbuild`;
- include source-tree headers with quoted names when following local tutorial
  examples;
- run maintainer validation targets from the repository root.

This workflow is not an installed package, package-manager, shared-library,
dynamic ABI, Windows Makefile, Windows `pkg-config`, broad platform parity, or
state-of-the-art claim.

### Installed-Prefix Consumer Workflow

The installed-prefix workflow is for downstream projects consuming an
installed static package:

- install to a selected `PREFIX` with Make or CMake;
- include public headers as `<sparse/...>`;
- link the installed static archive through `pkg-config` on Unix-like systems
  or `find_package(Sparse)` with CMake;
- use `Sparse::sparse_lu_ortho` as the installed CMake target;
- verify the package surface with `tests/test_install.sh` or
  `tests/test_cmake_install.sh` when install rules or consumer docs change.

The installed package contract is static-first. It does not include shared
library artifacts, static/shared selectors, dynamic ABI guarantees, package
manager distribution, runtime-loader behavior, or Windows Makefile/
`pkg-config` execution parity.

## Minimal Make/pkg-config Tutorial Contract

### Audience and Preconditions

Use this flow for Unix-like downstream consumers that have:

- a C11 compiler;
- Make;
- `pkg-config` or `pkgconf`;
- an installed sparse package under a known prefix.

This tutorial path should be documented as a Unix-side installed consumer
surface. It should not be presented as the Windows consumer route.

### Install Commands

Use a local staged prefix in examples so users can copy the flow without
requiring system-wide write access:

```sh
make
make install PREFIX="$PWD/_install"
export PKG_CONFIG_PATH="$PWD/_install/lib/pkgconfig${PKG_CONFIG_PATH:+:$PKG_CONFIG_PATH}"
```

Expected installed files:

- `$PWD/_install/lib/libsparse_lu_ortho.a`;
- `$PWD/_install/include/sparse/*.h`;
- `$PWD/_install/lib/pkgconfig/sparse.pc`.

Expected metadata:

- package name: `sparse`;
- cflags: one include flag for the installed include directory;
- libs: installed library directory, `-lsparse_lu_ortho`, and `-lm`;
- description: static archive package metadata.

### Minimal Consumer File

The tutorial should use installed include paths and a small allocation/link
smoke rather than source-tree quoted headers:

```c
#include <stdio.h>

#include <sparse/sparse_matrix.h>
#include <sparse/sparse_types.h>

int main(void) {
    SparseMatrix *A = sparse_create(3, 3);
    if (!A) {
        fprintf(stderr, "failed to create matrix\n");
        return 1;
    }

    sparse_insert(A, 0, 0, 1.0);
    printf("sparse version: %s\n", SPARSE_VERSION_STRING);
    printf("version int: %d\n", SPARSE_VERSION);
    printf("nnz: %d\n", (int)sparse_nnz(A));
    printf("OK\n");

    sparse_free(A);
    return 0;
}
```

### Build and Run Commands

```sh
cc -std=c11 $(pkg-config --cflags sparse) main.c $(pkg-config --libs sparse) -o sparse_pkgconfig_smoke
./sparse_pkgconfig_smoke
```

Expected output should contain:

- `sparse version:`;
- `version int:`;
- `nnz: 1`;
- `OK`.

### Failure Wording

Use direct diagnostic wording that keeps the support boundary clear:

- if `pkg-config --exists sparse` fails, check `PKG_CONFIG_PATH` points at the
  installed `lib/pkgconfig` directory;
- if `<sparse/...>` headers are missing, check the selected `PREFIX` and
  installed include directory;
- if `-lsparse_lu_ortho` is missing at link time, check the installed static
  archive under the selected `PREFIX`;
- if a workflow needs `.so`, `.dylib`, `.dll`, Homebrew/core, bottles,
  Linuxbrew, vcpkg, Conan, or dynamic ABI behavior, state that those surfaces
  are unsupported by the current package contract.

### Proof Command

```sh
bash tests/test_install.sh
```

This proof validates Make install/uninstall, installed headers, the static
archive, `sparse.pc`, exact package version metadata, `pkg-config` cflags/libs,
compiled installed consumers, runtime output, and absence of unsupported
shared-library or package-manager claims in owned metadata.

## Minimal CMake Tutorial Contract

### Audience and Preconditions

Use this flow for downstream CMake consumers, including the maintained Windows
MSVC route. Preconditions:

- CMake 3.14 or newer;
- a C11 compiler;
- an installed sparse package under a known prefix.

### Install Commands

```sh
cmake -S . -B build/install -DCMAKE_INSTALL_PREFIX="$PWD/_install" -DCMAKE_INSTALL_LIBDIR=lib -DCMAKE_C_STANDARD=11
cmake --build build/install
cmake --install build/install
```

Expected installed files:

- `$PWD/_install/lib/libsparse_lu_ortho.a`;
- `$PWD/_install/include/sparse/*.h`;
- `$PWD/_install/lib/cmake/Sparse/SparseConfig.cmake`;
- `$PWD/_install/lib/cmake/Sparse/SparseConfigVersion.cmake`;
- `$PWD/_install/lib/cmake/Sparse/SparseTargets.cmake`;
- `$PWD/_install/lib/pkgconfig/sparse.pc`.

### Minimal Consumer CMakeLists

```cmake
cmake_minimum_required(VERSION 3.14)
project(sparse_consumer C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

find_package(Sparse REQUIRED)

add_executable(sparse_cmake_smoke main.c)
target_link_libraries(sparse_cmake_smoke PRIVATE Sparse::sparse_lu_ortho)
```

The installed CMake target name must remain exactly
`Sparse::sparse_lu_ortho`.

### Build and Run Commands

```sh
cmake -S . -B build -DCMAKE_PREFIX_PATH="$PWD/_install"
cmake --build build
./build/sparse_cmake_smoke
```

The complete maintained example lives in `examples/cmake_example/` and should
remain the canonical installed CMake example. Its output contract contains:

- `Sparse library version`;
- `Solution:`;
- `OK`.

### Exact-Version Note

When documenting exact version resolution, use the current project `VERSION`
file value and the CMake form:

```cmake
find_package(Sparse <VERSION> EXACT REQUIRED)
```

Do not imply broad semantic-version compatibility guarantees beyond the current
package version proof. `tests/test_cmake_install.sh` verifies an exact match
and verifies that a mismatched exact version is rejected.

### Windows CMake Boundary

Windows documentation should use the Visual Studio CMake route:

```cmd
cmake -S . -B build -G "Visual Studio 17 2022"
cmake --build build --config Release
cmake --install build --config Release --prefix C:\sparse
```

Windows should continue to exclude Makefile parity and `pkg-config` command
execution parity. The Windows route is CMake/MSVC only.

### Proof Command

```sh
bash tests/test_cmake_install.sh
```

This proof validates CMake install/export, installed package files, imported
static target metadata, downstream `find_package(Sparse)` build/run behavior,
exact-version acceptance, mismatched-version rejection, `pkg-config` version
metadata, absence of source/build path leaks, and absence of unsupported
shared-library or dynamic ABI selectors.

## Day 6 Documentation Rewrite Targets

Use this audit to rewrite public docs without duplicating too much content:

| File | Day 6 action |
| --- | --- |
| `INSTALL.md` | Add the full minimal installed-consumer tutorial, split into Make/`pkg-config` and CMake paths, near the existing install usage sections. |
| `README.md` | Keep the short install summary and route to the installed-consumer tutorial. |
| `docs/tutorial.md` | Keep local build-tree tutorial content and route installed-prefix users to `INSTALL.md`. |
| `examples/README.md` | Keep local example catalog; route installed CMake users to `examples/cmake_example/` and the install tutorial. |
| `docs/maintainer_guide.md` | Retain proof-owner interpretation; link to the public tutorial rather than duplicating commands. |

## Acceptance Criteria for Tutorial Rewrite

- Public installed-consumer commands use a staged local prefix by default.
- Installed examples include `<sparse/...>` headers, not source-tree quoted
  headers.
- `pkg-config` command examples use the package name `sparse`.
- CMake examples use `find_package(Sparse REQUIRED)` and
  `Sparse::sparse_lu_ortho`.
- Windows examples use CMake/MSVC only.
- Verification points to `tests/test_install.sh` and
  `tests/test_cmake_install.sh`.
- Docs do not claim package-manager support, shared-library support, dynamic
  ABI support, Windows Makefile parity, Windows `pkg-config` execution parity,
  broad platform parity, broad report freshness, performance superiority, or
  state-of-the-art status.

## Day 5 Conclusion

Sprint 194 Item 194.3 has enough implementation detail to rewrite the
installed-consumer tutorial on Day 6. The smallest useful tutorial surface is a
Unix Make/`pkg-config` smoke program plus the existing CMake downstream example
contract. Both map directly to current proof scripts and installed metadata.
