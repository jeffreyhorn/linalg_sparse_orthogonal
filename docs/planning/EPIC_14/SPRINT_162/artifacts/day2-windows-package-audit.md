# Day 2 Windows Package Audit

Day 2 compares the reviewed Windows CMake install/downstream proof against the
Unix Make install and `pkg-config` proof. The goal is to name concrete parity
deltas before Sprint 162 chooses a product decision.

## Windows CMake Proof Map

Current reviewed Windows package proof lives in `.github/workflows/windows-ci.yml`
under `install-and-downstream`.

| Proof Area | Windows Evidence | Notes |
| --- | --- | --- |
| Build system | `cmake -S . -B build-install -G "Visual Studio 17 2022" -A x64` | Reviewed MSVC/CMake route on `windows-2022`. |
| Installed static library | Checks `lib/sparse_lu_ortho.lib`. | Static-first Windows archive proof. |
| Shared artifact rejection | Recursively rejects installed `*.dll`. | Does not check `.so`/`.dylib` because Windows lane is platform-specific. |
| Headers | Requires 19 installed headers and `include/sparse/sparse_version.h`. | Matches current header count. |
| CMake package files | Requires `SparseConfig.cmake`, `SparseConfigVersion.cmake`, `SparseTargets.cmake`, and `SparseTargets-release.cmake`. | CMake install/export proof. |
| `sparse.pc` file | Requires installed `lib/pkgconfig/sparse.pc`. | File metadata proof only, not `pkg-config` execution proof. |
| CMake target metadata | Requires `Sparse::sparse_lu_ortho STATIC IMPORTED`, install-prefix include dirs, and `${_IMPORT_PREFIX}/lib/sparse_lu_ortho.lib`. | Confirms static imported target shape. |
| Path leak checks | Rejects source-root and build-root leaks in installed CMake package metadata. | Protects relocatable package metadata. |
| Unsupported metadata checks | Rejects shared imported metadata, loader metadata, and static/shared selector metadata. | Maintains static-first non-claims. |
| `sparse.pc` metadata | Checks `Name`, static archive description, version, `Cflags`, `Libs`, and no unsupported package/ABI wording. | Metadata shape proof only. |
| Generated downstream consumer | Creates a CMake consumer, uses `find_package(Sparse REQUIRED)`, links `Sparse::sparse_lu_ortho`, builds/runs. | CMake consumer proof. |
| Maintained downstream example | Configures/builds/runs `examples/cmake_example` with `CMAKE_PREFIX_PATH`. | Maintained example proof. |
| Exact version behavior | Configures/builds/runs an exact-version consumer. | Exact-version CMake package proof. |
| Mismatch rejection | Requires lower same-major version mismatch to fail configure. | Confirms exact-version-only contract. |

## Unix Make And `pkg-config` Proof Map

Current Unix package proof lives in `tests/test_install.sh` and is run by the
reviewed Linux package contract lane and reviewed macOS install/pkg-config lane.

| Proof Area | Unix Evidence | Notes |
| --- | --- | --- |
| Build system | `make clean` then `make install PREFIX="$PREFIX"`. | Makefile install proof. |
| Installed static library | Checks `lib/libsparse_lu_ortho.a`. | Unix static archive proof. |
| Shared artifact rejection | Rejects installed `*.so`, `*.so.*`, `*.dylib`, and `*.dll`. | Broader cross-platform shared-artifact guard. |
| Headers | Requires all source headers plus generated `sparse_version.h`. | Current count is 19. |
| `sparse.pc` file | Requires `lib/pkgconfig/sparse.pc`. | Metadata plus execution proof. |
| `pkg-config --exists` | Runs `pkg-config --print-errors --exists sparse`. | Actual `pkg-config` execution proof. |
| Exact version through `pkg-config` | Runs `pkg-config --exists "sparse = $EXPECTED_VERSION"`. | Exact-version pkg-config proof. |
| Variables | Checks `prefix`, `libdir`, and `includedir` values. | Uses path equivalence for robust temp prefixes. |
| Compiler flags | Checks `pkg-config --cflags sparse` resolves installed include path. | Execution and path proof. |
| Link flags | Checks `pkg-config --libs sparse` resolves installed libdir, `-lsparse_lu_ortho`, and `-lm`. | Execution and link metadata proof. |
| Static link flags | Checks `pkg-config --libs --static sparse` matches normal libs. | Current self-contained link-surface proof. |
| Unsupported metadata | Rejects `Libs.private` and unsupported package/ABI wording. | Static-first metadata guard. |
| Modversion | Checks `pkg-config --modversion sparse`. | Metadata execution proof. |
| Basic downstream consumer | Compiles/links/runs generated C consumer using `pkg-config` flags. | Makefile-style consumer proof. |
| Maintained example via `pkg-config` | Compiles/links/runs `examples/cmake_example/main.c` using `pkg-config` flags. | Maintained source consumer proof independent of CMake. |
| Uninstall | Runs `make uninstall` and checks library, headers, and `sparse.pc` removal. | Make uninstall proof. |

## Parity Delta Table

| Area | Windows CMake Proof | Unix Make/`pkg-config` Proof | Delta |
| --- | --- | --- | --- |
| Build entry point | CMake/Visual Studio generator | Makefile | Windows does not prove Makefile install. |
| Install command | `cmake --install` | `make install` | Different install owner and command semantics. |
| Static archive name | `sparse_lu_ortho.lib` | `libsparse_lu_ortho.a` | Expected platform-specific archive extension. |
| CMake package metadata | Strong Windows proof | Strong Unix CMake proof via `tests/test_cmake_install.sh` | Mostly aligned for CMake install/export. |
| `sparse.pc` metadata file | Installed and inspected as text | Installed and executed by `pkg-config` | Windows lacks `pkg-config` execution proof. |
| `pkg-config --exists` | Not run | Run | Open parity gap. |
| `pkg-config --cflags --libs` | Not run | Run and used for downstream compile/link | Open parity gap. |
| `pkg-config --modversion` | Not run | Run | Open parity gap. |
| Downstream `pkg-config` consumer | Not present | Generated consumer and maintained example compile/link/run | Open parity gap. |
| Make uninstall | Not present | Run and validated | Open parity gap for Makefile parity. |
| Shared artifact rejection | Rejects DLLs | Rejects `.so`, `.so.*`, `.dylib`, and DLLs | Platform-specific but aligned in intent. |
| Unsupported package wording | Checks CMake and `sparse.pc` metadata | Checks `sparse.pc` metadata and static deferral guard | Aligned in intent. |
| Exact-version behavior | CMake exact-version proof | `pkg-config` exact version plus CMake exact version in separate script | Different package front ends. |
| Package-manager support | Explicit non-claim | Explicit non-claim | Aligned. |
| Shared-library ABI | Explicit non-claim | Explicit non-claim | Aligned. |

## Windows-Specific Blocker Register

| Blocker | Type | Detail | Decision Impact |
| --- | --- | --- | --- |
| No reviewed Windows `pkg-config` executable/provider | Technical and product | The Windows workflow installs and inspects `sparse.pc`, but does not install or run `pkg-config`. A promotion would need a chosen provider and stable CI setup. | Must be selected explicitly before implementation. |
| POSIX-style `pkg-config` flags may not map cleanly to MSVC `cl.exe` | Technical | `sparse.pc` emits `-I`, `-L`, `-l...`, and `-lm`, which work for Unix-style compilers. MSVC command-line/link conventions differ. | If promoting Windows `pkg-config`, decide whether proof uses MinGW/clang-like shell or translates to MSVC. |
| Makefile commands use POSIX install/rm/shell assumptions | Technical | `Makefile install` uses `install`, shell loops, `sed`, and Unix archive naming. | Windows Makefile parity likely needs a specific shell/toolchain owner or retained non-claim. |
| Existing Windows package proof is CMake-first | Product | Docs and CI intentionally say Windows package proof is CMake install/downstream scoped. | Any promotion requires docs and CI wording changes. |
| `sparse.pc` file existence can be misread as execution parity | Documentation/product | Windows verifies metadata but not `pkg-config` command behavior. | Stronger retained non-claim guard may be enough if not promoting. |
| Uninstall parity is Makefile-owned | Technical | Windows CMake workflow does not run `make uninstall`; CMake uninstall target is not present. | Makefile parity decision is independent from CMake install proof. |

## Decision Register For Day 4

Day 4 must decide among four product options:

1. **Promote Windows `pkg-config` only.**
   Requires selecting a Windows `pkg-config` provider and downstream compiler
   route, then validating `--exists`, version, cflags/libs, modversion, and
   compile/link/run behavior.
2. **Promote Windows Makefile parity only.**
   Requires selecting a Windows shell/toolchain route for Makefile install and
   uninstall behavior.
3. **Promote both.**
   Highest cost; combines the provider/toolchain risks of both surfaces.
4. **Retain both non-claims with stronger guards.**
   Requires making the existing CMake-first Windows proof and installed
   `sparse.pc` metadata boundary harder to misread as execution parity.

## Day 2 Conclusion

The concrete parity gap is not installed package shape: Windows already proves
the static `.lib`, headers, CMake metadata, `sparse.pc` metadata, downstream
CMake consumers, exact-version behavior, mismatch rejection, and static-first
unsupported metadata checks.

The remaining gaps are front-end execution gaps:

- Windows does not run Makefile install/uninstall.
- Windows does not run `pkg-config`.
- Windows does not compile/link/run a downstream consumer using `pkg-config`
  output.

Those gaps are now classified and ready for Day 4 product decision work.
