# Installation Guide

## Prerequisites

- C11 compiler (GCC >= 7, Clang >= 5, MSVC >= 2019)
- Make or CMake >= 3.14
- Math library (`-lm`, typically provided by the system)

Optional:
- `pkg-config` (for Makefile-based downstream projects)
- `lcov` + `bc` (for `make coverage`)
- `libomp` / GCC libgomp (for `make omp` — OpenMP-parallel SpMV + Lanczos MGS)

## Supported platforms

| platform | toolchain | CI job | notes |
|---|---|---|---|
| Linux (Ubuntu) | gcc | `.github/workflows/ci.yml` | primary; tsan job runs here (Sprint 29 Day 8 routes from macOS) |
| Linux (Ubuntu) | clang | `.github/workflows/ci.yml::tsan` | TSan + OpenMP build |
| macOS | Apple Clang | `.github/workflows/macos-ci.yml::apple-clang` | Sprint 29 Day 9; runs `make sanitize` |
| macOS | Homebrew GCC (`gcc-15`) | `.github/workflows/macos-ci.yml::homebrew-gcc` | Sprint 29 Day 9 |
| Windows | MSVC 2022 via CMake | `.github/workflows/windows-ci.yml` | Sprint 29 Days 7-8 |

`make tsan` on macOS 15+ is blocked by an upstream dyld initialization
hang (Sprint 28 inheritance; macOS 15.7 platform issue not specific to
this codebase).  Sprint 29 Day 8 routes the TSan job to Linux CI per
`docs/planning/EPIC_2/SPRINT_29/windows_ci_decision.md`.

## Quick Start (Makefile)

```sh
make
make test           # run the full test suite
make install PREFIX=/usr/local   # install library, headers, pkg-config
```

The default `PREFIX` is `/usr/local`. Set `DESTDIR` for staged installs
(e.g., packaging):

```sh
make install PREFIX=/usr DESTDIR=/tmp/staging
```

### Installed files

| Path | Contents |
|------|----------|
| `$(PREFIX)/lib/libsparse_lu_ortho.a` | Static library |
| `$(PREFIX)/include/sparse/*.h` | Public headers (14 files) |
| `$(PREFIX)/lib/pkgconfig/sparse.pc` | pkg-config descriptor |

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

### Using from a CMake project

After installing, downstream projects can use `find_package`:

```cmake
find_package(Sparse REQUIRED)
target_link_libraries(myapp PRIVATE Sparse::sparse_lu_ortho)
```

Headers are included as `#include <sparse/sparse_types.h>`.

See `examples/cmake_example/` for a complete working example.

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

For coverage, Apple Clang's gcov is incompatible with lcov.  Use GCC:

```sh
brew install gcc lcov
make coverage CC=gcc-15        # or whatever version Homebrew currently ships
```

If the GCC sysroot is incompatible with the installed CommandLineTools
SDK (a known macOS 15 issue surfaced Sprint 29 Day 12 — Apple's CLT
clang assembler chokes on `-mmacosx-version-min=15.0` while Homebrew
GCC 15 was configured against MacOSX15.sdk), use `gcovr` as a local
diagnostic instead:

```sh
brew install gcovr
make CFLAGS="-std=c11 -Wall -Wextra -O0 -g --coverage" \
     LDFLAGS="-lm --coverage" test
gcovr --gcov-executable=/usr/bin/gcov --root . \
      --filter 'src/' --exclude 'tests/' --exclude 'benchmarks/' \
      --gcov-ignore-parse-errors=suspicious_hits.warn_once_per_file \
      --print-summary
```

The Linux CI job uses gcc-native `--coverage` + lcov directly + the
calibrated 80 % threshold (Sprint 29 Day 12; see
`docs/planning/EPIC_2/SPRINT_29/coverage_threshold_decision.md`).

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
