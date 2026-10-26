# Installation Guide

Use this file for operational setup, staged installs, installed-consumer
workflows, and install-surface validation. For first-use solver choice or
build-tree adoption, start with [README.md](README.md) and
[examples/README.md](examples/README.md). For CSR, CSC, or Matrix Market
first-use recipes before installation, use [docs/cookbook.md](docs/cookbook.md).

## Start Here

Choose the smallest install/setup path that matches what you actually need.

1. **Prove a local build first:** if you have not run a build-tree example yet,
   start with [README.md#start-here](README.md#start-here) and
   [examples/README.md#start-here](examples/README.md#start-here).
2. **Install the maintained static package:** on Unix-like systems, use
   [Quick Start (Makefile)](#quick-start-makefile). With CMake, use
   [CMake Build](#cmake-build). For 64-bit indices, follow
   [Index Width Contract](#index-width-contract) before installing.
3. **Consume from another project:** use
   [Unix Make/pkg-config consumer](#unix-makepkg-config-consumer) for
   Makefile-style consumers or
   [CMake consumer](#cmake-consumer) for CMake consumers.
4. **Validate the installed surface:** run
   [Verifying the Installation](#verifying-the-installation) when changing
   install rules, package metadata, CI lanes, or downstream-consumer examples.
5. **Read support precisely:** use the
   [Support readiness matrix](#support-readiness-matrix) before interpreting
   install, CI, report, package, or platform evidence as a support claim.

The maintained install contract is static-first. Shared-library packaging,
dynamic ABI compatibility, runtime-loader behavior, package-manager
distribution, static/shared selectors, Windows Makefile parity, and Windows
`pkg-config` execution parity are intentionally out of scope.
These source/install instructions do not claim a tagged release,
reproducible source archive, binary release package, or release readiness.

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

- first local build-tree adoption:
  - `README.md`
  - `examples/README.md`
- Unix-side installed static package with `pkg-config`:
  - `make install`
  - `sparse.pc`
  - `tests/test_install.sh`
- installed CMake consumer path:
  - `cmake --install`
  - `find_package(Sparse)`
  - `tests/test_cmake_install.sh`
- package-manager deferral:
  - broad package-manager support is not currently provided
  - local Homebrew formula proof material exists under `packaging/homebrew/`,
    plus `scripts/homebrew_local_formula_proof.sh`
  - Homebrew local formula proof artifacts exist, root MIT license metadata is
    present, and the selected developer-mode proof uses
    `HOMEBREW_DEVELOPER=1 SPARSE_HOMEBREW_LICENSE=MIT`
  - the latest developer-mode local proof on macOS Intel x86_64 Tier 3
    Homebrew reaches archive creation, checksum calculation, temporary tap
    formula rendering, source install, installed static package surface
    validation, downstream `brew test`, uninstall, cleanup, and proof exit `0`
  - placeholder `SPARSE_HOMEBREW_LICENSE` values remain blocker evidence, not
    formula proof metadata
  - Sprint 207 selected continued package-provider deferral with stronger
    guards, so this proof remains local evidence rather than a public provider
    install method
  - Homebrew/core, bottles, Linuxbrew, vcpkg, Conan, pkgsrc, distro/system
    packages, provider registries, taps, recipes beyond the selected local
    template, binary packages, release packages, and package-manager release
    readiness remain unsupported
  - use source install via Make or CMake for user-facing installs; Sprint 198's
    completed proof is a local static source formula proof, not a user-facing
    Homebrew installation path
- reviewed-platform interpretation:
  - `docs/maintainer_guide.md`

This file owns operational setup, installed-consumer detail, and local
install-surface validation. It is not the front-door adoption guide, benchmark
command reference, or maintainer-policy home.

## Support Readiness Matrix

Use this matrix for current user-facing support status. It separates the
active user path from proof-owner detail and retained non-claims. For
maintainer interpretation of the same evidence, use
[docs/maintainer_guide.md](docs/maintainer_guide.md).

| Surface | Current user status | Primary user path | Evidence owner | Retained non-claims |
|---|---|---|---|---|
| Local source build and first solve | supported | [README.md](README.md), [examples/README.md](examples/README.md), `make`, `make examples` | Makefile, examples, `make examples-build`, `make test` | Not an install, package-manager, performance, or broad platform-parity claim. |
| Unix Make static install | supported | [Quick Start (Makefile)](#quick-start-makefile) | `tests/test_install.sh`, `sparse.pc.in`, Linux/macOS install lanes | No shared-library, dynamic ABI, runtime-loader, or package-manager support. |
| Unix `pkg-config` consumer | validated | [Unix Make/pkg-config consumer](#unix-makepkg-config-consumer) | `tests/test_install.sh`, `sparse.pc.in` | No Windows `pkg-config` execution parity. |
| Installed CMake consumer | supported | [CMake consumer](#cmake-consumer), [`examples/cmake_example/`](examples/cmake_example/) | `tests/test_cmake_install.sh`, `cmake/SparseConfig.cmake.in` | No shared-library or dynamic ABI promise. |
| Windows MSVC CMake install/downstream | validated | [Windows (MSVC)](#windows-msvc), `.github/workflows/windows-ci.yml` | Windows CMake install/downstream lane, PowerShell validator | No Windows Makefile parity, Windows `pkg-config` execution parity, package-manager support, shared-library support, dynamic ABI support, runtime-loader behavior, or broad Windows parity. |
| Windows selected Cholesky comparison workflow | guarded-workflow: Sprint 208 reviewed hosted evidence for the exact path, but support/freshness promotion remains re-deferred on metadata and generated claim wording | Windows selected comparison workflow and selected target manifest | bounded selected Cholesky comparison freshness workflow, `normalize_report_index.py --selected-target cholesky-spd-tridiag-5`, PowerShell guard, Sprint 208 manifest/workflow/normalizer guard artifacts; selected target manifest promotion remains separate | No promoted Windows selected freshness until selected metadata, generated support tier, and generated non-claim wording are promoted together; no broad Windows report freshness, Windows selected oracle freshness, Windows selected benchmark freshness, or unselected Windows comparison families. |
| Windows QR incompatible least-squares comparison | guarded-workflow deferred: Sprint 209 adds an evidence-collection lane, but selected freshness promotion remains absent | Future selected Windows QR evidence | Sprint 203 evidence ledger, Sprint 209 workflow/PowerShell and manifest guards, selected target manifest, normalizer/workflow guards | The QR incompatible least-squares target remains outside Windows selected freshness until hosted MSVC probe evidence, selected artifact review, selected-target manifest metadata, generated support tier, and generated non-claim wording are promoted together; no broad QR parity, broad least-squares parity, broad Windows report freshness, package-manager proof, shared-library ABI proof, performance superiority, or state-of-the-art claim. |
| Linux/macOS selected comparison freshness | hosted-evidence | `make report-index-comparison-freshness` | `tests/corpus/manifests/selected_report_targets.tsv`, Linux/macOS hosted selected comparison lanes, comparison tests | No broad external-library parity, broad report freshness, package/ABI support, performance superiority, or state-of-the-art claim. |
| Linux/macOS selected performance freshness | hosted-evidence | `make bench-canonical-report-freshness` | `tests/corpus/manifests/selected_report_targets.tsv`, `benchmarks/README.md`, `scripts/check_bench_canonical_freshness.py`, Linux/macOS hosted selected lanes | No portable performance, timing threshold, release benchmark, platform parity, package/ABI proof, broad package-manager distribution, or state-of-the-art claim; timing-threshold promotion requires stable runner, compiler, repeat, warmup, variance, baseline, threshold, retained-artifact, and non-claim evidence. |
| Local selected allocation-failure proof | local-only | `make iterative-allocation-failure-gate`, `make matmul-allocation-failure-gate`, `make symbolic-allocation-failure-gate`, `make symbolic-lu-allocation-failure-gate`, `make ldlt-linked-list-allocation-failure-gate` | `test_iterative`, `test_matmul`, `test_etree`, `test_ldlt`, focused registration guards, CTest `allocation_failure` labels | Selected owners only: iterative handle workspace, `sparse_matmul()` workspaces, selected `sparse_symbolic_cholesky()` output construction, selected `sparse_symbolic_lu()` owner, and selected no-reorder linked-list LDLT numeric factorization. No broad allocation-failure coverage, CSC LDLT proof, reordered LDLT proof, OS OOM guarantee, platform parity, package/install proof, generated-tooling proof, performance proof, release proof, or state-of-the-art reliability claim. |
| Local benchmark and sentinel reports | local-only | `make bench-canonical-report`, `make performance-sentinels` | `benchmarks/README.md`, benchmark scripts/tests | Not hosted proof unless named by a hosted lane; no portable speedup claim. |
| Local generated API HTML | local-only | `docs/api_reference.md`, `make api-docs-freshness` | `Doxyfile`, API docs coverage/local-only/routing scripts | No hosted API publication, retained generated-doc artifact, committed generated HTML, or completeness beyond checked-in public headers selected by Doxyfile. |
| Package-manager distribution | not claimed | Use source install via Make or CMake | Sprint 198 Homebrew proof material, Sprint 207 continued-deferral decision, package-manager deferral guard | No user-facing Homebrew install path, Homebrew/core, bottles, Linuxbrew, tap, vcpkg, Conan, pkgsrc, distro/system package, binary package, release package, package-manager release readiness, or broad provider support. |
| Shared-library and dynamic ABI support | deferred | Static install only | Static package deferral guard, Sprint 170 package/ABI decision | No `.so`, `.dylib`, `.dll`, import-library, loader, SONAME, install-name/RPATH, selector, or dynamic ABI support. |
| External baseline comparison | blueprint-only: the Sprint 215 pilot blocked before either solver ran | [Sprint 215 selected pilot outcome](docs/planning/EPIC_19/SPRINT_215/artifacts/day10-selected-pilot-outcome.md) | [Sprint 215 claim matrix](docs/planning/EPIC_19/SPRINT_215/artifacts/day11-evidence-and-claim-matrix.md), absent v1 evidence schema/validator | No accepted CHOLMOD/project correctness result, timing sample, or external-library parity. |
| Broad ecosystem or state-of-the-art parity | not claimed | Use selected evidence docs only for scoped proof | Sprint 215 blueprint and blocked pilot, selected target manifest, benchmark/corpus docs | No broad SuiteSparse/PETSc/Trilinos/Eigen/SciPy parity, portable superiority, or unqualified state-of-the-art status. |

The supported/validated install cells describe the maintained static Make and
CMake consumer contracts. Sprint 214 recorded local installed-consumer checks;
the reviewed platform workflow entries identify their exact CI lanes, not a
fresh hosted result for this checkout or a release approval. The
[release readiness checklist](docs/planning/EPIC_19/SPRINT_214/RELEASE_READINESS_CHECKLIST.md)
lists further candidate, provenance, platform, and approval gates.

Generated API HTML is an on-demand local view, not an install, release,
hosted, or artifact publication surface. Users should link to
`docs/api_reference.md` and checked-in public headers for durable API
documentation, then run `make api-docs-freshness` only when they need to inspect
local Doxygen output for the current checkout.

Maintainers track future release inputs and stop conditions in the
[Sprint 214 release readiness checklist](docs/planning/EPIC_19/SPRINT_214/RELEASE_READINESS_CHECKLIST.md).

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

After installation, a downstream Makefile-style consumer can compile with:

```sh
cc -std=c11 $(pkg-config --cflags sparse) main.c $(pkg-config --libs sparse)
```

For a complete copy-pasteable installed-consumer walkthrough, use
[Installed Consumer Tutorial](#installed-consumer-tutorial).

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

The maintained install surface is intentionally static-first. This section is
the claim boundary for package shape; use the quick-start sections above when
you only need commands.

- Unix-like `make install` installs a static archive such as
  `libsparse_lu_ortho.a`
- Windows/MSVC installs use the corresponding static `.lib`
- `cmake --install` exports the same static library through
  `Sparse::sparse_lu_ortho`
- `pkg-config` and `find_package(Sparse)` both describe that installed static
  archive surface
- the installed `sparse.pc` description identifies static archive package
  metadata and intentionally carries no `Libs.private` or static/shared
  selector under the current self-contained static link contract
- version metadata comes from the repo `VERSION` file and is propagated through
  `sparse_version.h`, `SparseConfigVersion.cmake`, and `sparse.pc`
- index width is selected at library build time; installed headers, CMake
  target definitions, and `sparse.pc` carry the selected 32- or 64-bit width
- the exported CMake package version file is exact-version only
- CMake rejects `BUILD_SHARED_LIBS=ON` at configure time so a shared-library
  request is not silently treated as supported

This install/export story is real and maintained, but it is not a broad shared
library or dynamic-ABI promise. On Windows, the maintained consumer path
remains the reviewed CMake workflow. The installed `sparse.pc` is inspected as
static package metadata there, but Windows CI does not run `pkg-config` and
does not claim Windows `pkg-config` execution parity.

Shared-library packaging is deferred until a future change adds explicit
export/import and symbol visibility policies, a dynamic ABI policy, Linux
SONAME metadata, macOS install-name/RPATH metadata, Windows
DLL/import-library behavior, installed shared consumer proof, and
runtime-loader validation.
The canonical Sprint 170 package and ABI product decision is recorded in
`docs/planning/EPIC_15/SPRINT_170/artifacts/day9-shared-library-abi-product-decision.md`.

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
  - macOS CI carries reviewed static-first install/export proof for those
    local scripts plus the static deferral guard
  - Windows carries reviewed CMake install/downstream validation for the
    maintained static-first package surface, including metadata-only
    `sparse.pc` inspection, but remains narrower than the Unix Makefile and
    `pkg-config` script surface

## Index Width Contract

The default index width is 32 bits. Select 32 or 64 when building the
**library**, not by changing a downstream typedef or adding a consumer-only
`-DSPARSE_IDX_BITS` flag. A width belongs to one clean build tree and one
clean installed prefix. Do not reuse either for the opposite width. The
generated `sparse_version.h` records `SPARSE_BUILT_IDX_BITS`; `idx_t`,
`IDX_MAX`, `SPARSE_PRIDX`, and `SPARSE_SCNIDX` follow that width. The installed
CMake target exports `SPARSE_PACKAGE_IDX_BITS=<width>`; `sparse.pc` includes
the same definition in `Cflags`. A conflicting explicit consumer width or
foreign header fails at compile time. Static linking alone does not detect a
foreign archive, so the consumer must check `sparse_idx_bits()` before its
first index-bearing call.

From the repository root, choose **one** of these fresh Unix-like source
install routes. The prefixes below are absolute because they include `$PWD`:

```sh
# Make, 32-bit (default, stated explicitly)
make BUILDDIR=build/idx-make-32 SPARSE_IDX_BITS=32 \
  PREFIX="$PWD/_install/make-32" install

# Make, 64-bit
make BUILDDIR=build/idx-make-64 SPARSE_IDX_BITS=64 \
  PREFIX="$PWD/_install/make-64" install

# CMake, 32-bit
cmake -S . -B build/idx-cmake-32 -DSPARSE_IDX_BITS=32 \
  -DCMAKE_INSTALL_PREFIX="$PWD/_install/cmake-32" -DCMAKE_INSTALL_LIBDIR=lib
cmake --build build/idx-cmake-32 --target sparse_lu_ortho
cmake --install build/idx-cmake-32

# CMake, 64-bit
cmake -S . -B build/idx-cmake-64 -DSPARSE_IDX_BITS=64 \
  -DCMAKE_INSTALL_PREFIX="$PWD/_install/cmake-64" -DCMAKE_INSTALL_LIBDIR=lib
cmake --build build/idx-cmake-64 --target sparse_lu_ortho
cmake --install build/idx-cmake-64
```

For an isolated Unix `pkg-config` consumer, select the same prefix used by
one install route, then compile against **both** its flags and its archive:

```sh
prefix="$PWD/_install/make-64"  # or make-32, cmake-32, cmake-64
export PKG_CONFIG_LIBDIR="$prefix/lib/pkgconfig"
export PKG_CONFIG_PATH=
pkg-config --variable=prefix sparse  # must print the selected prefix
pkg-config --cflags sparse           # -I<prefix>/include -DSPARSE_PACKAGE_IDX_BITS=64
cc -std=c11 $(pkg-config --cflags sparse) \
  examples/cmake_example/main.c $(pkg-config --libs sparse) -o sparse_width_smoke
./sparse_width_smoke                 # prints Sparse index width 64 before the solve
```

The maintained example calls the executable width check before any matrix
operation. In your own consumer, compare `sizeof(idx_t) * CHAR_BIT`,
`SPARSE_BUILT_IDX_BITS`, and `sparse_idx_bits()` and stop on disagreement.
[`examples/cmake_example/installed_width_check.h`](examples/cmake_example/installed_width_check.h)
is the tested, reusable check. For a CMake install, use a **fresh downstream
build tree** with `find_package(Sparse REQUIRED)` and link
`Sparse::sparse_lu_ortho`; set `CMAKE_PREFIX_PATH` to the selected absolute
prefix, as in [CMake consumer](#cmake-consumer). Do not manually override the
target's width definition. An exact-version request is available, but it does
not promise cross-version ABI compatibility.

The source-tree static validator checks installed header, package metadata,
and the real archive width together. Run it with the route used for the
selected prefix, for example:

```sh
version=$(tr -d '[:space:]' < VERSION)
python3 scripts/check_static_install_contract.py "$prefix" \
  --mode make --platform unix --version "$version"
# Use --mode cmake for a CMake-installed prefix.
```

| Claim | Evidence and boundary |
| --- | --- |
| Separate 32/64 builds agree with their generated headers and archives | [Day 5 parity](docs/planning/EPIC_20/SPRINT_219/artifacts/day5-build-system-parity.md); one width per build tree. |
| Unix static installs and consumers reject mismatched header, metadata, or archive | [Day 9 CMake matrix](docs/planning/EPIC_20/SPRINT_219/artifacts/day9-cmake-consumer-matrix.md) and [Day 10 pkg-config matrix](docs/planning/EPIC_20/SPRINT_219/artifacts/day10-pkgconfig-mixed-prefix-matrix.md); maintained consumers perform the pre-use check. |
| 64-bit algorithms | [Day 11 selected lane](docs/planning/EPIC_20/SPRINT_219/artifacts/day11-selected-64-bit-lane.md) passed locally for selected cases only. [Day 12 hosted jobs](docs/planning/EPIC_20/SPRINT_219/artifacts/day12-hosted-64-bit-registration.md) are registered but not yet green on this branch. No universal 64-bit coverage claim. |
| Windows package | Reviewed CMake install route at the default 32-bit width only; no Windows `pkg-config` execution or selected 64-bit hosted claim. |

This is a static-only package. Mixing widths, cross-version ABI compatibility,
shared-library packaging, dynamic ABI, and runtime-loader support are not
claimed.

## Supported platforms

| platform | toolchain | CI job | notes |
|---|---|---|---|
| Linux (Ubuntu) | gcc | `.github/workflows/ci.yml` | strongest reviewed source of truth: reviewed Makefile quality, reviewed CMake parity, dead-code, and reviewed static-first package contract; supplemental direct runtime and `bench-fast` also live here |
| Linux (Ubuntu) | clang | `.github/workflows/ci.yml::tsan` | supplemental ThreadSanitizer + OpenMP lane |
| macOS | Apple Clang | `.github/workflows/macos-ci.yml::build-and-test` (`compiler=apple-clang`) | reviewed macOS lane: `make quality-review-compile`, `make quality-review-cmake`, `make wall-check`, `make sanitize`; same workflow also carries reviewed static-first Make install/`pkg-config` and CMake install/export proof in the `install-and-pkgconfig` and `cmake-install-export` jobs |
| macOS | Homebrew GCC (`gcc-15`) | `.github/workflows/macos-ci.yml::build-and-test` (`compiler=homebrew-gcc`) | supplemental second-compiler direct build/test/wall-check coverage |
| Windows | MSVC 2022 via CMake | `.github/workflows/windows-ci.yml` | reviewed CMake subset, including the promoted `test_threads`, `test_sprint4_integration`, and `test_fuzz` targets, reviewed CMake install/downstream validation for the maintained static-first package surface, hosted PowerShell validation ownership for selected Windows workflow snippets, and one bounded selected Cholesky comparison freshness workflow as a guarded path for `cholesky-spd-tridiag-5`; Sprint 208 reviewed the latest hosted path and strengthened manifest/workflow/normalizer guards while re-deferring selected freshness promotion until manifest metadata, generated support tier, and generated non-claim wording are promoted together; Sprint 209 adds one bounded Windows QR incompatible evidence-collection lane for `qr-incompatible-ls`, but QR selected freshness remains re-deferred until hosted run evidence, exact artifact inspection, selected manifest metadata, generated support tier, and generated non-claim wording are promoted together; does not imply Windows Makefile parity, `pkg-config` execution parity, broad report freshness, selected oracle freshness, selected benchmark freshness, package-manager support, shared-library support, dynamic ABI support, runtime-loader behavior, or broad Windows parity |

The QR incompatible least-squares target remains outside Windows selected
freshness even with the Sprint 209 evidence-collection lane present; promotion
requires hosted MSVC probe evidence, selected artifact review, selected-target
manifest metadata, generated support tier, and generated non-claim wording to
move together.

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

## Installed Consumer Tutorial

Use this tutorial when a separate downstream project consumes an installed
static package. It is separate from the source-tree examples in
`examples/README.md`, which compile against the checkout and run binaries from
`./build/...`.

The examples below use a staged local prefix so the commands do not require
system-wide write access. Replace `$PWD/_install` with another prefix when
installing into a controlled package or system location.

### Unix Make/pkg-config consumer

Use this path for Unix-like downstream projects that build with Make,
`pkg-config`, or a direct compiler command. It requires a C11 compiler, Make,
and `pkg-config` or `pkgconf`.

Install the static package:

```sh
make
make install PREFIX="$PWD/_install"
export PKG_CONFIG_PATH="$PWD/_install/lib/pkgconfig${PKG_CONFIG_PATH:+:$PKG_CONFIG_PATH}"
pkg-config --exists sparse
```

Create `main.c` in the downstream project:

```c
#include <limits.h>
#include <stdio.h>

#include <sparse/sparse_matrix.h>
#include <sparse/sparse_types.h>

int main(void) {
    size_t header_bits = sizeof(idx_t) * CHAR_BIT;
    size_t archive_bits = sparse_idx_bits();
    if (header_bits != SPARSE_BUILT_IDX_BITS ||
        archive_bits != SPARSE_BUILT_IDX_BITS) {
        fprintf(stderr, "index width mismatch: header=%zu built=%d archive=%zu\n",
                header_bits, SPARSE_BUILT_IDX_BITS, archive_bits);
        return 1;
    }
    printf("index width: %zu\n", archive_bits);
    printf("sparse version: %s\n", SPARSE_VERSION_STRING);
    printf("version int:    %d\n", SPARSE_VERSION);

    SparseMatrix *A = sparse_create(3, 3);
    if (!A) {
        fprintf(stderr, "failed to create sparse matrix\n");
        return 1;
    }

    sparse_insert(A, 0, 0, 1.0);
    printf("nnz: %" SPARSE_PRIDX "\n", sparse_nnz(A));
    sparse_free(A);
    printf("OK\n");
    return 0;
}
```

Compile and run:

```sh
cc -std=c11 $(pkg-config --cflags sparse) main.c $(pkg-config --libs sparse) -o sparse_pkgconfig_smoke
./sparse_pkgconfig_smoke
```

Expected output includes:

- `index width: 32` or `index width: 64`, matching the selected prefix
- `sparse version:`
- `version int:`
- `nnz: 1`
- `OK`

`pkg-config --cflags sparse` should resolve one installed include directory
and `-DSPARSE_PACKAGE_IDX_BITS=<width>`.
`pkg-config --libs sparse` should resolve the installed library directory,
`-lsparse_lu_ortho`, and `-lm`.

If `pkg-config --exists sparse` fails, check that `PKG_CONFIG_PATH` includes
the selected prefix's `lib/pkgconfig` directory. If `<sparse/...>` headers or
`-lsparse_lu_ortho` cannot be found, check that the same `PREFIX` was used for
installing and for the consumer build.

Validate this path after install-rule, package-metadata, or tutorial changes
with:

```sh
bash tests/test_install.sh
```

This `pkg-config` path is a Unix-side installed consumer surface. It is not a
Windows `pkg-config` execution claim.

### pkg-config command summary

```sh
cc -std=c11 $(pkg-config --cflags sparse) main.c $(pkg-config --libs sparse)
```

### Uninstall

```sh
make uninstall PREFIX=/usr/local
```

## CMake Build

Use this path when you want the installed CMake package export, the
Windows-supported build/install route, or a downstream project that calls
`find_package(Sparse)`.

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
| `SPARSE_IDX_BITS` | 32 | Library index width: `32` or `64`; use a separate build tree and prefix per width. |
| `SPARSE_OPENMP` | OFF | Enable OpenMP parallel SpMV |
| `SPARSE_MUTEX` | OFF | Enable per-matrix mutex for thread safety |
| `SANITIZE` | (empty) | Sanitizer: `asan`, `ubsan`, `all` |

`BUILD_SHARED_LIBS=ON` is intentionally rejected. The maintained CMake package
contract is the static archive surface; shared-library packaging and dynamic
ABI support are deferred until explicit export/import and symbol visibility
policies, a dynamic ABI policy, Linux SONAME metadata, macOS
install-name/RPATH metadata, Windows DLL/import-library behavior, installed
shared consumer proof, and runtime-loader validation exist.

### CMake consumer

After installing, downstream projects can use `find_package`. If you have not
already installed to the staged prefix used above, install with CMake first:

```sh
cmake -S . -B build/install -DCMAKE_INSTALL_PREFIX="$PWD/_install" -DCMAKE_INSTALL_LIBDIR=lib -DCMAKE_C_STANDARD=11
cmake --build build/install
cmake --install build/install
```

For a minimal consumer project, create this `CMakeLists.txt`:

```cmake
cmake_minimum_required(VERSION 3.14)
project(sparse_consumer C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

find_package(Sparse REQUIRED)

add_executable(sparse_cmake_smoke main.c)
target_link_libraries(sparse_cmake_smoke PRIVATE Sparse::sparse_lu_ortho)
```

Use the width-checking `main.c` and installed-header form shown above, such as
`#include <sparse/sparse_types.h>` and
`#include <sparse/sparse_matrix.h>`. Configure, build, and run from the
downstream project with the **same absolute prefix** used at install time:

```sh
prefix=/absolute/path/to/_install/cmake-64  # select one installed prefix
cmake -S . -B build/sparse-consumer-64 -DCMAKE_PREFIX_PATH="$prefix"
cmake --build build/sparse-consumer-64
./build/sparse-consumer-64/sparse_cmake_smoke
```

For 32 bits, select the corresponding `cmake-32` prefix and a different
consumer build tree. The imported target propagates the installed width to
the compiler; the program checks the linked archive before matrix use.

For exact version checks, use the current project version from `VERSION`:

```cmake
find_package(Sparse <VERSION> EXACT REQUIRED)
```

The exported CMake target is the installed static archive target
`Sparse::sparse_lu_ortho`. It is not a shared-library, dynamic ABI, runtime
loader, or package-manager support claim.

See `examples/cmake_example/` for a complete working example.

On Windows, this CMake route is the maintained consumer path. The Makefile and
`pkg-config` install flows are Unix-side validation surfaces, not Windows
parity claims.

```cmd
cmake -S . -B build -G "Visual Studio 17 2022"
cmake --build build --config Release
cmake --install build --config Release --prefix C:\sparse
```

Validate this path after CMake install/export, package metadata, downstream
example, or tutorial changes with:

```sh
bash tests/test_cmake_install.sh
```

For the reviewed local CMake parity path used alongside the install flow, run
`make quality-review-cmake-compile` or `make quality-review-cmake` from the
project root before switching into packaging or staged-install work.

## Platform Notes

Use these notes as operational setup hints after reading
[Supported platforms](#supported-platforms). They do not widen the reviewed
platform tiers described above.

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
surface rather than another build-tree example. Run it after changing install
rules, package metadata, exported CMake files, CI install lanes, or downstream
consumer examples.

Run the install validation script (Unix):

```sh
bash tests/test_install.sh
```

This installs to a temporary directory, checks all files, compiles a test
program and the maintained example with `pkg-config`, verifies their runtime
output, and cleans up. The `pkg-config` prefix, libdir, includedir, cflags, and
libs path checks compare installed directories by filesystem identity so staged
temporary-prefix spelling differences do not create false failures.

For CMake integration verification:

```sh
bash tests/test_cmake_install.sh
```

These focused regression scripts are Unix-oriented local validation surfaces for
the maintained static-first install/export contract:

- `tests/test_install.sh` covers Make install/uninstall plus `pkg-config`
- `tests/test_cmake_install.sh` covers CMake install/export plus
  `find_package(Sparse)`, including exact-version configure/build/run proof
- both scripts check static archive metadata and reject unsupported
  shared-library, loader, package-manager, or dynamic ABI wording in the
  installed package files they own

They complement, rather than replace, the reviewed platform lanes.
Use the split below when reading install confidence:

- local direct validation:
  - the two scripts above exercise the Unix-side Make and CMake install paths
    end to end
- reviewed platform confidence:
  - Linux remains the strongest reviewed source of truth and now includes a
    reviewed static-first package-contract lane
  - macOS carries reviewed static-first Make install/`pkg-config` and CMake
    install/export proof for the maintained static archive package contract
  - Windows carries reviewed CMake install/downstream validation for the
    maintained static-first package surface, including installed static `.lib`,
    headers, CMake package metadata, metadata-only `sparse.pc` inspection,
    generated and maintained installed CMake consumers, exact-version behavior,
    mismatch-version rejection, and no DLL/shared imported metadata

Do not widen that reading into a broader platform or package claim:

- macOS reviewed static-first install/export proof does not claim
  shared-library packaging, dynamic ABI compatibility, runtime-loader
  compatibility, package-manager support, static/shared selectors, or broader
  macOS platform parity
- Windows reviewed CMake install/downstream validation does not claim Windows
  Makefile parity, Windows `pkg-config` execution parity, package-manager
  support, shared-library packaging, dynamic ABI compatibility,
  runtime-loader behavior, or broad Windows parity
- Windows `sparse.pc` checks are metadata-only inspection and must not be read
  as `pkg-config` command execution
- Linux reviewed package-contract proof does not claim shared-library
  packaging, dynamic ABI compatibility, runtime-loader behavior, or
  package-manager support

### Normalized Package Rows

Maintainers can inspect the source-controlled package proof-owner rows without
running an install:

```sh
python3 scripts/normalize_report_index.py --family package --check
python3 scripts/normalize_report_index.py --family package --check-freshness
```

These rows identify maintained proof owners and templates:

- `tests/test_install.sh`
- `tests/test_cmake_install.sh`
- `sparse.pc.in`
- `cmake/SparseConfig.cmake.in`
- `scripts/static_package_deferral_check.sh`
- `scripts/package_manager_deferral_check.sh`

They use `freshness_status=source_controlled`. Read them as ownership and
static-first scope metadata, not as proof that an install validation command
was just run. To prove the install surface locally, run the install validation
scripts above. The package-manager guard protects provider non-claims and
checks the selected Homebrew local proof boundary. The current Homebrew proof
script accepts root MIT metadata and
`HOMEBREW_DEVELOPER=1 SPARSE_HOMEBREW_LICENSE=MIT`, creates the local source
archive, renders the temporary formula through a local tap, installs from
source, validates the installed static package surface, runs the downstream
`brew test`, uninstalls, cleans up, and exits `0` in the recorded
developer-mode macOS Intel x86_64 Tier 3 Homebrew proof. That evidence remains
local static source formula proof only, not broad package-manager support.
