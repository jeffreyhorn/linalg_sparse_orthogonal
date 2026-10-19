# Sprint 214 Day 9: Installed Consumer Evidence

## Decision And Evidence Boundary

Day 2 selected ABI/release design and continued static-first deferral. This
day exercises the maintained **installed static** consumer routes on local
macOS. There is no shared target, DLL/import library, SONAME, install-name,
runtime-loader contract, ABI epoch, or retained cross-version executable to
test. The future shared fixtures below are specifications, not executions.

## Executed Local Environment

| Input | Observed value |
| --- | --- |
| Host and compiler | `Darwin 24.6.0 x86_64`; Apple Clang `17.0.0` (`x86_64-apple-darwin24.6.0`). |
| Build/package tools | CMake `4.4.3`; `pkg-config` `2.5.1`; `VERSION=2.2.0`. |
| Runtime search variables | `DYLD_LIBRARY_PATH` and `LD_LIBRARY_PATH` were unset. No loader-path override was added by either install test. |
| Prefix isolation | Each script created a fresh temporary `<tmp>/usr` install prefix and separate consumer build/output paths; traps removed them afterward. Make and CMake used different prefixes. |

The commands were run from the repository root on `sprint-214`:

```sh
bash -n tests/test_install.sh tests/test_cmake_install.sh
bash tests/test_install.sh
bash tests/test_cmake_install.sh
nm -g -U build/libsparse_lu_ortho.a
```

The `nm` command inspects the **built static archive**, not the deleted
temporary installed archive. The installed consumer link/run checks below
are the installed symbol-resolution evidence. The `nm` result showed defined
`_sparse_create`, `_sparse_free`, `_sparse_insert`, `_sparse_nnz`, and
`_lu_csr_factor_solve` symbols; these are current archive symbols, not a
dynamic export allowlist.

## Static Consumer Results

| Route | Producer, metadata, and consumer resolution | Observed result |
| --- | --- | --- |
| Make plus `pkg-config` | `tests/test_install.sh` ran `make clean`, `make install PREFIX=<tmp>/usr`, set `PKG_CONFIG_PATH=<prefix>/lib/pkgconfig`, obtained installed `--cflags`/`--libs`, then compiled its standalone basic consumer and `examples/cmake_example/main.c` into temporary executables with `cc`. The installed prefix contained `libsparse_lu_ortho.a`, `sparse.pc`, and 19 headers; the validator passed. | 24 passed, 0 failed. Basic output: `sparse version: 2.2.0`, `version int: 20200`, `nnz: 1`, `OK`. Maintained example output: `Sparse library version 2.2.0 (int 20200)`, `Solution: [1.000000, 1.000000, 1.000000]`, `OK`. Uninstall removed archive, headers, and `sparse.pc`. |
| CMake plus `find_package(Sparse)` | `tests/test_cmake_install.sh` configured a separate build with `CMAKE_INSTALL_PREFIX=<tmp>/usr` and `CMAKE_INSTALL_LIBDIR=lib`, installed the explicit static target, configured `examples/cmake_example` and an exact-version consumer in separate build directories using `CMAKE_PREFIX_PATH=<prefix>`, and linked `Sparse::sparse_lu_ortho`. The imported target was `STATIC IMPORTED` and located the installed `.a`; the installed-prefix validator passed. | 28 passed, 0 failed, 0 skipped. Both example and exact-version executables printed `Sparse library version 2.2.0 (int 20200)`, `Solution: [1.000000, 1.000000, 1.000000]`, `OK`. A mismatched requested package version was rejected at configure. |

Both scripts now print captured successful consumer stdout in their test logs,
so future local and hosted runs expose the actual value checks rather than
only a pass label. The CMake consumer source is maintained in the repository,
but its build directory and package resolution are outside the library build
tree. This establishes installed static linking/execution on this local
macOS toolchain only. Day 8's CI wiring is not a hosted run result.

## Metadata And Failure Diagnostics

The installed-prefix checker requires the static archive name, versioned
`sparse.pc`, and, for CMake, the installed `SparseTargets*.cmake` files with
only `Sparse::sparse_lu_ortho STATIC IMPORTED`. It rejects an absent archive,
missing package file, changed package version, wrong imported location,
extra Windows `.lib`, and `.so*`/`.dylib`/`.dll`/import artifacts. The Day 8
fixture suite exercised these paths; Day 9's real consumers were positive
proof, not deliberately broken installs. A missing or wrongly named archive
must fail validation before consumer link, and a wrong CMake imported location
must fail validation rather than resolving an unrelated build-tree library.
`BUILD_SHARED_LIBS=ON` and unsupported link-mode aliases still fail the
Day 6 configure guard. Day 10 owns the broader executed negative matrix.

## Future Shared-Consumer Fixtures: Not Executed

If a later product decision selects a shared path, create a separate fixture
directory with a C11 `main.c` that includes only installed
`<sparse/sparse_types.h>` and `<sparse/sparse_matrix.h>`. The probe must call
the exact provisional Day 3 eight-function set (`sparse_create`,
`sparse_free`, `sparse_set`, `sparse_get`, `sparse_strerror`, `sparse_errno`,
`sparse_idx_bits`, `sparse_scalar_bits`), round-trip a 1x1 value, verify the
compiled width/scalar tuple against a source-controlled ABI manifest, and
exit nonzero on value, error, or ownership failure. Finalize error semantics
before freezing the expected `sparse_errno`/`sparse_strerror` assertions.
Configure/build this consumer **outside** the source and producer build trees
using only the installed prefix and an explicitly selected shared package
target. Never let a static target or build-tree library satisfy this fixture.

Required fixture inputs for each platform are: the approved ABI identity
manifest (epoch, index/scalar widths, exact export names, architecture),
installed prefix, shared artifact filename, installed package target and
metadata, compiler/toolchain identity, and controlled runtime search path.
The future test harness must perform these steps in order:

1. Install into a fresh prefix; inventory and hash the shared file, package
   metadata, and (on Windows) import library. Assert the package target
   resolves to that installed shared file, not the static archive.
2. Compare actual dynamic exports against the exact approved manifest. Build
   an out-of-tree consumer against installed headers and that shared target;
   inspect its dynamic dependencies/import table before executing it.
3. Run with only the documented installed-prefix loader route enabled; assert
   the 1x1 round trip, width/scalar tuple, and zero exit status. Record actual
   loaded file and its identity. A successful link without execution fails
   the fixture.
4. Remove or rename the installed shared artifact while leaving the package
   metadata in place. The same executable must fail to load, with a diagnostic
   naming the missing expected artifact and no fallback to source/build or
   unrelated system paths. Restore it, then substitute a differently named
   epoch/width or wrong-architecture file; metadata validation or load must
   fail before selected API use. Remove the import library on Windows and
   verify a fresh consumer link fails as well.
5. Preserve the first approved executable and its binary/manifest as a
   baseline. Only a later same-epoch release may run that **unchanged** old
   executable against a newer installed library to claim backward
   compatibility; a rebuilt consumer does not count.

| Future platform lane | Artifact/consumer inspection and loader route | Required missing/wrong-name diagnostic |
| --- | --- | --- |
| Linux GCC/ELF | `readelf -d` on shared object and consumer for `DT_SONAME`, `DT_NEEDED`, and search path; `readelf -Ws` or `nm -D --defined-only` for exact exports. Run using the selected installed RPATH/RUNPATH policy; `LD_LIBRARY_PATH` may isolate a fixture only if explicitly recorded, not silently substitute for product policy. | Loader fails to find the expected SONAME when the installed object is removed; wrong epoch or width cannot resolve as the selected target. |
| macOS Apple Clang/Mach-O | `otool -D` for dylib ID, `otool -L` for consumer dependency, `otool -l` for RPATH, and `nm -gU` for exports. Run under the selected `@rpath`/`@loader_path` policy with no build-tree path. | `dyld` names the missing install-name after removal; relocation and wrong install-name cases fail without a fallback dylib. |
| Windows MSVC/PE | `dumpbin /EXPORTS` on DLL, `/DEPENDENTS` on consumer, and package inspection for installed DLL `IMPORTED_LOCATION` plus matching import `.lib` `IMPORTED_IMPLIB`. Use a controlled DLL directory and record CRT/allocator choice. | Consumer fails DLL load from the controlled path after removal; mismatched DLL name/architecture fails, and absent import `.lib` fails a fresh link. Static `.lib` suffix alone is never DLL-import proof. |

No fixture above has a selected artifact identity or runnable shared package
target today. Linux results would not imply macOS or Windows support. The
current `VERSION=2.2.0` and `find_package(... EXACT)` checks are package
version checks, not an ABI epoch or cross-version compatibility test.

## Handoff

Day 10 should execute negative tests that apply to the selected static path,
keep both installed consumer routes green, and list the unexecutable shared
loader/architecture/export cases as future gates. Windows hosted CMake static
consumer results and Day 8 CI changes remain unverified until a branch run;
local macOS execution cannot replace them. No public support wording changed
on Day 9, and no C or header source file was edited.
