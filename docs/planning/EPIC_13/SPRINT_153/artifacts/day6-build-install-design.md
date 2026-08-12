# Sprint 153 Day 6 Build And Install Design

## Purpose

Day 6 designs the implementation for the Day 5 product decision. Sprint 153
will keep the package static-first and strengthen the shared-library deferral
instead of adding a partial shared-library implementation.

## Selected Build Behavior

The selected build behavior is stronger rejection diagnostics, not shared
target support.

`CMakeLists.txt` should continue to:

- reject `BUILD_SHARED_LIBS=ON` before target creation;
- declare `sparse_lu_ortho` as an explicit `STATIC` target;
- avoid `SHARED`, `MODULE`, `SOVERSION`, `install_name`, RPATH, or Windows
  export metadata;
- install only the archive through `ARCHIVE DESTINATION`.

Day 7 should strengthen the `BUILD_SHARED_LIBS=ON` fatal error so it names the
exact blockers from Day 5:

- public export/import macro policy;
- hidden internal-symbol/export-list policy;
- dynamic ABI policy for public structs, callbacks, allocators, error state,
  and version metadata;
- Linux SONAME policy;
- macOS install-name/RPATH policy;
- Windows DLL/import-library and runtime lookup policy;
- installed shared consumer proof;
- runtime-loader validation.

## Install And Export Design

Make and CMake install behavior should remain static archive scoped.

| Surface | Day 7 Design | Reason |
| --- | --- | --- |
| Make install | Keep installing `libsparse_lu_ortho.a`, public headers, generated `sparse_version.h`, and `sparse.pc`. | This is the maintained Unix static package path. |
| CMake install | Keep installing the archive, public headers, generated version header, `SparseTargets.cmake`, `SparseConfig.cmake`, `SparseConfigVersion.cmake`, and `sparse.pc`. | This is the maintained CMake static package path. |
| Unsupported artifacts | Continue rejecting `.so`, `.so.*`, `.dylib`, and `.dll` in install tests. | Prevents silent shared artifact drift. |
| Imported target | Keep `Sparse::sparse_lu_ortho` as `STATIC IMPORTED`. | Downstream CMake consumers should not infer shared-loader support. |
| Package version file | Keep exact-version compatibility. | Exact versioning avoids overstating dynamic ABI compatibility. |

No install destination for `LIBRARY` or `RUNTIME` should be added in Day 7.

## `pkg-config` Design

`sparse.pc.in` should remain static archive scoped:

- keep `Description: Static archive package metadata for sparse linear algebra`;
- keep `Libs: -L${libdir} -lsparse_lu_ortho -lm ...`;
- keep no `Libs.private` stanza under the current self-contained static link
  contract;
- keep no shared-library, dynamic ABI, package-manager, or static/shared
  selector wording.

Day 7 does not need to edit `sparse.pc.in` unless the static deferral guard
needs a stronger assertion tied to the existing text.

## CMake Package Design

`cmake/SparseConfig.cmake.in` should remain minimal:

- import `SparseTargets.cmake`;
- call `check_required_components(Sparse)`;
- avoid package selectors such as `COMPONENTS shared` or `COMPONENTS static`;
- avoid variables or comments that imply shared-library support.

The installed generated target metadata should remain checked by
`tests/test_cmake_install.sh` as a static imported target with no shared
imported metadata.

## Static Deferral Guard Design

`scripts/static_package_deferral_check.sh` is the right owner for the focused
Day 7 proof because it already checks:

- `BUILD_SHARED_LIBS=ON` is rejected;
- the target remains explicit `STATIC`;
- CMake install metadata uses `ARCHIVE DESTINATION`;
- no public export/import macro appears under `include/`;
- no CMake shared ABI metadata appears;
- CMake and `pkg-config` package metadata do not expose static/shared
  selectors;
- README, INSTALL, and maintainer guide preserve support boundaries.

Day 7 should extend this script so the configure-failure text must include the
exact blocker tokens from Day 5. Suggested required tokens:

- `export/import`;
- `symbol visibility`;
- `dynamic ABI policy`;
- `SONAME`;
- `install-name`;
- `DLL/import-library`;
- `installed shared consumer proof`;
- `runtime-loader validation`.

This keeps the Day 5 decision test-backed without adding new shared artifacts
or broadening package claims.

## Install Test Design

`tests/test_install.sh` and `tests/test_cmake_install.sh` already cover the
selected claim:

- static archive is installed;
- no shared-library artifacts are installed;
- installed header count is exact;
- `sparse.pc` describes static archive package metadata;
- `sparse.pc` has no unsupported shared/ABI/package-manager wording;
- CMake package metadata imports the static target;
- CMake package metadata has no shared imported target or shared artifact
  location;
- installed downstream consumers compile, link, and run against the maintained
  static package surface.

Day 7 should not widen these tests unless the implementation changes package
metadata behavior. If only `CMakeLists.txt` rejection wording and
`scripts/static_package_deferral_check.sh` checks change, focused validation
can run the deferral guard plus install scripts.

## Documentation Design

Day 7 may update comments adjacent to the CMake rejection. Broad user-facing
documentation alignment is reserved for Day 11 unless tests require immediate
wording changes.

Any documentation wording must preserve these non-claims:

- no shared-library packaging;
- no dynamic ABI compatibility;
- no runtime loader behavior;
- no package-manager distribution;
- no static/shared selectors;
- no Windows Makefile parity;
- no Windows `pkg-config` execution parity.

## Day 7 Implementation Checklist

1. Update the `BUILD_SHARED_LIBS=ON` fatal error in `CMakeLists.txt` to include
   exact blocker wording.
2. Update `scripts/static_package_deferral_check.sh` to assert those exact
   blocker tokens in the configure-failure output.
3. Keep `add_library(sparse_lu_ortho STATIC ...)` unchanged.
4. Keep CMake install rules archive-only.
5. Keep Make install behavior unchanged.
6. Keep `sparse.pc.in` and `cmake/SparseConfig.cmake.in` static-first unless a
   focused proof need appears during implementation.
7. Run focused validation:
   - `bash scripts/static_package_deferral_check.sh`;
   - `bash tests/test_install.sh`;
   - `bash tests/test_cmake_install.sh`.
8. Because Day 7 will likely edit `CMakeLists.txt` and a shell script, no
   `make format && make lint && make test` C gate is required unless C or
   header files change.
