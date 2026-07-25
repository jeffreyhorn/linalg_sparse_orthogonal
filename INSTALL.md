# Installation Guide

Use this file for operational setup, staged installs, installed-consumer
workflows, and install-surface validation. For first-use solver choice or
build-tree adoption, start with [README.md](README.md) and
[examples/README.md](examples/README.md).

## Start Here

Choose the smallest install/setup path that matches what you actually need:

- **Need one local success before installing anything?**
  - Use the front-door path in [README.md](README.md), then move to
    [examples/README.md](examples/README.md).
- **Need a Unix-side static install plus `pkg-config` downstream use?**
  - Use [Quick Start (Makefile)](#quick-start-makefile).
- **Need an installed CMake consumer or the Windows-supported path?**
  - Use [CMake Build](#cmake-build).
- **Need to prove the installed package surface locally?**
  - Use [Verifying the Installation](#verifying-the-installation).

## Prerequisites

- C11 compiler (GCC >= 7, Clang >= 5, MSVC >= 2019)
- Make or CMake >= 3.14
- Math library (`-lm`, typically provided by the system)

Optional:
- `pkg-config` (for Makefile-based downstream projects)
- `lcov` + `bc` (for `make coverage`)
- `libomp` / GCC libgomp (for `make omp` — OpenMP-parallel SpMV + Lanczos MGS)

## Support Split

Use each support surface for the layer it owns:

- local build-tree adoption:
  - `README.md`
  - `examples/README.md`
- Unix-side installed static package with `pkg-config`:
  - `make install`
  - `tests/test_install.sh`
- installed CMake consumer path:
  - `cmake --install`
  - `find_package(Sparse)`
  - `tests/test_cmake_install.sh`
- reviewed-platform interpretation:
  - `docs/maintainer_guide.md`

This file owns operational setup, installed-consumer detail, and local
install-surface validation. It is not the front-door adoption guide, benchmark
command reference, or maintainer-policy home.

## Quick Start (Makefile)

Use this path when you want the maintained Unix-side static install surface and
`pkg-config` downstream story.

```sh
make
make tooling-build
make quality-review-compile
make test           # run the full test suite
make quality-review
make install PREFIX=/usr/local   # install library, headers, pkg-config
```

`make quality-review-compile` is the maintained local compile-quality wrapper
(`format-check` + `source-list-check` + `lint`). `make quality-review` adds
`test` and `deadcode-check` on top of that reviewed path. For the fuller
command map and failure-rerun guidance, use `README.md` as the canonical
front-door reference.

The default `PREFIX` is `/usr/local`. Set `DESTDIR` for staged installs
(e.g., packaging):

```sh
make install PREFIX=/usr DESTDIR=/tmp/staging
```

## Maintained Install Contract

The maintained install surface is intentionally static-first:

- Unix-like `make install` installs a static archive such as
  `libsparse_lu_ortho.a`
- Windows/MSVC installs use the corresponding static `.lib`
- `cmake --install` exports the same static library through
  `Sparse::sparse_lu_ortho`
- `pkg-config` and `find_package(Sparse)` both describe that installed static
  archive surface
- version metadata comes from the repo `VERSION` file and is propagated through
  `sparse_version.h`, `SparseConfigVersion.cmake`, and `sparse.pc`
- the exported CMake package version file is exact-version only
- CMake rejects `BUILD_SHARED_LIBS=ON` at configure time so a shared-library
  request is not silently treated as supported

This install/export story is real and maintained, but it is not a broad shared
library or dynamic-ABI promise. On Windows, the maintained consumer path
remains the reviewed CMake workflow.

Shared-library packaging is deferred until a future change adds explicit build
rules, package metadata, installed-consumer proof, and platform-specific
runtime-loader coverage.

Use the split below when deciding how much package detail you need:

- installed package shape:
  - static library
  - public headers
  - `pkg-config` metadata
  - exported CMake package files
- downstream consumer story:
  - `pkg-config` and `find_package(Sparse)` both describe that same installed
    static archive surface
- validation story:
  - local Unix-side install scripts validate the Make and CMake install/export
    paths directly
  - Linux CI carries a reviewed static-first package-contract lane for those
    local scripts plus the static deferral guard
  - macOS and Windows reviewed platform claims remain narrower than those local
    scripts

## Supported platforms

| platform | toolchain | CI job | notes |
|---|---|---|---|
| Linux (Ubuntu) | gcc | `.github/workflows/ci.yml` | strongest reviewed source of truth: reviewed Makefile quality, reviewed CMake parity, dead-code, and reviewed static-first package contract; supplemental direct runtime and `bench-fast` also live here |
| Linux (Ubuntu) | clang | `.github/workflows/ci.yml::tsan` | supplemental ThreadSanitizer + OpenMP lane |
| macOS | Apple Clang | `.github/workflows/macos-ci.yml::apple-clang` | reviewed macOS lane: `make quality-review-compile`, `make quality-review-cmake`, `make wall-check`, `make sanitize`; same workflow also carries supplemental static-first Make install/`pkg-config` and CMake install/export confidence |
| macOS | Homebrew GCC (`gcc-15`) | `.github/workflows/macos-ci.yml::homebrew-gcc` | supplemental second-compiler direct build/test/wall-check coverage |
| Windows | MSVC 2022 via CMake | `.github/workflows/windows-ci.yml` | reviewed CMake subset plus supplemental CMake install/downstream confidence; supports the maintained CMake-first consumer story rather than a separate reviewed install-validation lane; pthread/POSIX-backed staged tests remain outside the reviewed Windows subset |

`make tsan` on macOS 15+ is blocked by an upstream dyld initialization
hang that is not specific to this codebase. The maintained TSan job runs on
Linux CI.

### Installed files

| Path | Contents |
|------|----------|
| `$(PREFIX)/lib/libsparse_lu_ortho.a` | Static library |
| `$(PREFIX)/include/sparse/*.h` | Public headers |
| `$(PREFIX)/lib/pkgconfig/sparse.pc` | pkg-config descriptor |
| `$(PREFIX)/lib/cmake/Sparse/SparseConfig*.cmake` | Exported CMake package metadata and version file |
| `$(PREFIX)/lib/cmake/Sparse/SparseTargets.cmake` | Exported CMake target definitions |

### Using via pkg-config

```sh
cc -std=c11 $(pkg-config --cflags sparse) main.c $(pkg-config --libs sparse)
```

### Uninstall

```sh
make uninstall PREFIX=/usr/local
```

## CMake Build

```sh
mkdir build && cd build
cmake .. -DCMAKE_INSTALL_PREFIX=/usr/local
cmake --build .
ctest                # run tests
cmake --install .    # install
```

### CMake options

| Option | Default | Description |
|--------|---------|-------------|
| `SPARSE_OPENMP` | OFF | Enable OpenMP parallel SpMV |
| `SPARSE_MUTEX` | OFF | Enable per-matrix mutex for thread safety |
| `SANITIZE` | (empty) | Sanitizer: `asan`, `ubsan`, `all` |

`BUILD_SHARED_LIBS=ON` is intentionally rejected. The maintained CMake package
contract is the static archive surface; shared-library packaging and dynamic
ABI support are deferred until explicit build rules, package metadata,
installed-consumer proof, and runtime-loader validation exist.

### Using from a CMake project

After installing, downstream projects can use `find_package`:

```cmake
find_package(Sparse REQUIRED)
target_link_libraries(myapp PRIVATE Sparse::sparse_lu_ortho)
```

Headers are included as `#include <sparse/sparse_types.h>`.

See `examples/cmake_example/` for a complete working example.

For the reviewed local CMake parity path used alongside the install flow, run
`make quality-review-cmake-compile` or `make quality-review-cmake` from the
project root before switching into packaging or staged-install work.

## Platform Notes

### Linux (Ubuntu / Debian)

```sh
sudo apt-get install build-essential pkg-config
make && make test
sudo make install
```

For coverage:

```sh
sudo apt-get install gcc lcov bc
make coverage CC=gcc
```

### Linux (Fedora / RHEL)

```sh
sudo dnf install gcc make pkgconf-pkg-config
make && make test
sudo make install
```

### macOS

The default Apple Clang works for building and testing:

```sh
make && make test
make install PREFIX=/usr/local
```

For coverage on macOS, Apple Clang's LLVM gcov v4.2 emulation is
incompatible with Homebrew lcov 2.x.  The Makefile auto-detects the
compiler and routes Apple Clang through `gcovr` (which parses the
format directly):

```sh
brew install gcovr   # one-time
make coverage        # auto-routes to coverage-gcovr on Apple Clang
```

To force the lcov backend (e.g. when using Homebrew GCC):

```sh
brew install gcc lcov
make coverage-lcov CC=gcc-15
```

Note: Homebrew GCC's built-in sysroot may not match the installed
CommandLineTools SDK on macOS 15+. If `make coverage-lcov CC=gcc-15` fails to
build, fall back to `make coverage` (gcovr path).

The Linux CI job uses gcc-native `--coverage` + lcov directly + the
calibrated 80 % threshold.

For OpenMP support:

```sh
brew install libomp
make omp
```

### Windows (MSVC)

Use CMake with the Visual Studio generator:

```cmd
mkdir build && cd build
cmake .. -G "Visual Studio 17 2022"
cmake --build . --config Release
ctest -C Release
cmake --install . --config Release --prefix C:\sparse
```

Note: The Makefile targets (`make install`, etc.) are Unix-only. On
Windows, use the CMake workflow exclusively.

## Verifying the Installation

Use this section when you want explicit local validation of the installed package
surface rather than another build-tree example.

Run the install validation script (Unix):

```sh
bash tests/test_install.sh
```

This installs to a temporary directory, checks all files, compiles a test
program with pkg-config, and cleans up.

For CMake integration verification:

```sh
bash tests/test_cmake_install.sh
```

These focused regression scripts are Unix-oriented local validation surfaces for
the maintained static-first install/export contract:

- `tests/test_install.sh` covers Make install/uninstall plus `pkg-config`
- `tests/test_cmake_install.sh` covers CMake install/export plus
  `find_package(Sparse)`

They complement, rather than replace, the narrower reviewed platform lanes.
Use the split below when reading install confidence:

- local direct validation:
  - the two scripts above exercise the Unix-side Make and CMake install paths
    end to end
- reviewed platform confidence:
  - Linux remains the strongest reviewed source of truth and now includes a
    reviewed static-first package-contract lane
  - macOS carries supplemental Make install/`pkg-config` and CMake
    install/export confidence
  - Windows remains the reviewed CMake subset with supplemental CMake
    install/downstream confidence for the CMake-first consumer path

Do not widen that reading into a broader reviewed install-validation claim:

- macOS supplemental package confidence does not claim full reviewed
  install/export parity
- Windows supplemental CMake install/downstream confidence does not claim a
  separate reviewed install-validation lane beyond the CMake-first consumer
  story
- Linux reviewed package-contract proof does not claim shared-library
  packaging, dynamic ABI compatibility, runtime-loader behavior, or
  package-manager support
