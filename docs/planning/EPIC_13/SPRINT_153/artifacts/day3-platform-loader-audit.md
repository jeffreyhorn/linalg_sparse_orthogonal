# Sprint 153 Day 3 Platform Loader Audit

## Purpose

Day 3 audits what Linux, macOS, and Windows would require before the project
could claim shared-library support. This is a loader and packaging audit, not
an implementation change. The current maintained package contract remains
static-first.

## Current Loader Claim Status

| Surface | Current Status | Evidence |
| --- | --- | --- |
| CMake library type | Static only | `CMakeLists.txt` declares `add_library(sparse_lu_ortho STATIC ...)`. |
| Shared request behavior | Rejected | `CMakeLists.txt` fails configure when `BUILD_SHARED_LIBS=ON`. |
| CMake install artifact | Static archive only | CMake install uses `ARCHIVE DESTINATION` for `sparse_lu_ortho`. |
| Make install artifact | Static archive only | Make install installs `libsparse_lu_ortho.a`. |
| Shared artifacts | Unsupported and rejected by proof | Install tests fail if `.so`, `.so.*`, `.dylib`, or `.dll` files appear. |
| CMake package metadata | Static imported target | `tests/test_cmake_install.sh` checks `STATIC IMPORTED` metadata and rejects shared imported metadata. |
| `pkg-config` metadata | Static archive package metadata | Install tests require the static-archive description and reject unsupported shared/ABI wording. |
| Platform CI | Static package proof only | Linux, macOS, and Windows workflows explicitly avoid shared-library, dynamic ABI, and runtime-loader claims. |

## Linux `.so` Requirements

Linux shared support would need all of the following before a support claim:

- Build a real ELF shared object, with the target producing
  `libsparse_lu_ortho.so` and, if versioned, appropriate
  `libsparse_lu_ortho.so.<major>` and full-version symlinks.
- Define SONAME policy that ties major version increments to ABI-breaking
  changes.
- Compile shared objects with position-independent code and verify static
  archive behavior remains unchanged when both artifacts coexist.
- Define exported-symbol policy using hidden default visibility plus explicit
  public API annotations, or a linker version script/export list.
- Verify accidental internal symbols from Day 2 are not exported.
- Define dependency metadata for `m`, `dl`, OpenMP, pthread/mutex support, and
  runtime-probed dense backends.
- Decide whether `pkg-config --libs sparse` points at shared or static link
  flags when both artifacts exist, and whether `Libs.private` is introduced
  for static-only dependencies.
- Define CMake imported target semantics for shared artifacts, including
  installed `IMPORTED_LOCATION`, `IMPORTED_SONAME`, interface link libraries,
  and version compatibility.
- Add downstream install, link, run, and loader proof that links against the
  installed `.so` rather than the build tree or the static archive.
- Add inspection proof using tools such as `readelf`, `nm`, or `objdump` to
  verify SONAME, needed libraries, and exported symbols.

### Linux Blockers Today

- No shared target exists.
- No SONAME policy exists.
- No export map or `SPARSE_API` visibility policy exists.
- Public concrete structs are not governed by an ABI compatibility policy.
- Existing tests prove absence of shared artifacts, not loader behavior.

## macOS `.dylib` Requirements

macOS shared support would need all of the following before a support claim:

- Build a real Mach-O dynamic library, with deterministic naming and version
  metadata.
- Define `install_name` policy, including whether installed consumers resolve
  via `@rpath`, an absolute prefix path, or another reviewed convention.
- Define `MACOSX_RPATH` and installed RPATH behavior for downstream CMake
  consumers.
- Compile shared objects with position-independent code and keep static archive
  behavior stable when both artifacts exist.
- Define exported-symbol policy using hidden visibility plus explicit public
  annotations, or an exported-symbols list.
- Verify accidental internal symbols from Day 2 are not exported.
- Define how runtime-probed dense backend candidates interact with dyld lookup,
  Homebrew paths, and optional OpenMP/libomp linkage.
- Decide whether `pkg-config` and CMake installed metadata point at shared or
  static artifacts when both artifacts exist.
- Add downstream install, link, run, and loader proof on the hosted macOS lane.
- Add inspection proof using tools such as `otool -L`, `nm`, and
  `install_name_tool` validation where appropriate.

### macOS Blockers Today

- No dylib target or install-name policy exists.
- No exported-symbols list or visibility annotation policy exists.
- Current macOS CI proves static Make install/`pkg-config` and static CMake
  install/export only.
- Existing docs explicitly say macOS package proof does not imply dynamic
  loader support.

## Windows `.dll` Requirements

Windows shared support is the highest-risk loader path because it needs both
source-level export/import decisions and packaging decisions.

Windows support would need all of the following before a support claim:

- Build a real DLL plus import library, with deterministic artifact names and
  install layout under CMake/MSVC.
- Introduce a public export macro such as `SPARSE_API` and apply it to every
  supported public function and any public data symbol that is intentionally
  exported.
- Decide whether unsupported internal symbols are hidden by default, excluded
  through a `.def` file, or made `static`/private.
- Define `__declspec(dllexport)` for building the DLL and
  `__declspec(dllimport)` for downstream consumers.
- Decide whether public structs remain ABI-stable concrete layouts or are
  migrated toward opaque handles before a DLL claim.
- Audit allocator boundaries carefully so callers do not allocate with one C
  runtime and free with another.
- Define runtime lookup behavior: copied DLL beside executable, installed bin
  directory on `PATH`, CMake runtime dependency copying, or another explicit
  convention.
- Define CMake imported target metadata for `IMPORTED_IMPLIB`,
  `IMPORTED_LOCATION`, runtime DLL location, and configuration-specific paths.
- Define installed consumer proof for MSVC that configures, builds, runs, and
  verifies that the installed DLL is the runtime dependency.
- Define package metadata wording for `sparse.pc`; Windows currently installs
  the file but does not claim reviewed `pkg-config` execution parity.

### Windows Blockers Today

- No `SPARSE_API` or `__declspec` policy exists.
- No DLL/import-library install layout exists.
- No runtime DLL lookup/copy policy exists.
- No CMake imported shared target metadata exists.
- Windows CI explicitly validates CMake-first static package behavior and
  rejects DLL artifacts.
- Windows still does not claim Makefile parity or `pkg-config` execution
  parity, so a Windows DLL claim cannot be inferred from the existing lane.

## Toolchain Constraints

| Toolchain | Loader Constraints | Current Evidence Gap |
| --- | --- | --- |
| GCC on Linux | Needs PIC, ELF shared object, SONAME, exported-symbol control, and dependency metadata for `m`, `dl`, OpenMP, and pthread options. | No shared build, SONAME, or export-list proof. |
| Clang on Linux | Similar to GCC, with visibility and sanitizer interaction checks. | No shared build or sanitizer/shared loader proof. |
| Apple Clang on macOS | Needs Mach-O dylib, install-name/RPATH policy, exported-symbol filtering, and libomp/Homebrew path handling when OpenMP is enabled. | No dylib or install-name proof. |
| Homebrew GCC on macOS | Needs compatibility with macOS dynamic-library naming and OpenMP/runtime dependency behavior. | Current Homebrew GCC lane is static package proof only. |
| MSVC on Windows | Needs `__declspec` export/import policy, import library metadata, DLL runtime lookup, C runtime allocator boundary review, and installed consumer runtime proof. | No export macro, `.def` file, DLL install rule, or runtime proof. |

## Platform Proof Matrix

| Claim | Linux Proof Needed | macOS Proof Needed | Windows Proof Needed | Current Status |
| --- | --- | --- | --- | --- |
| Shared artifact is built | CMake build produces `.so` and static archive behavior remains stable. | CMake build produces `.dylib` and static archive behavior remains stable. | CMake/MSVC build produces `.dll` plus `.lib`. | Not supported. |
| Shared artifact is installed | Install tree contains expected shared artifact and version metadata. | Install tree contains expected dylib and install name. | Install tree contains expected DLL and import library. | Current tests reject shared artifacts. |
| Exported symbols are curated | `nm/readelf` exported-symbol allowlist passes. | `nm` exported-symbol allowlist passes. | `dumpbin` or CMake export/import validation passes. | No export policy. |
| Installed CMake consumer links shared | `find_package(Sparse)` consumer links installed `.so`. | `find_package(Sparse)` consumer links installed `.dylib`. | `find_package(Sparse)` consumer links import library and runs with installed DLL. | Static-only proof exists. |
| Installed `pkg-config` consumer links shared | `pkg-config` flags select shared policy and run passes. | `pkg-config` flags select shared policy and run passes. | Explicitly decide whether Windows `pkg-config` execution is supported. | Static-only Unix proof exists. |
| Loader metadata is correct | SONAME and dependency list inspected. | install name/RPATH and dependency list inspected. | runtime DLL lookup and import library inspected. | No loader metadata proof. |
| ABI compatibility is governed | Public structs, callbacks, enums, versioning, and exported symbols are versioned. | Same as Linux. | Same as Linux plus C runtime allocator rules. | Not governed as dynamic ABI. |

## Product Decision Implications

A Sprint 153 decision to implement shared support must include a small,
reviewed, cross-platform proof scope or a documented platform tier split. A
shared-library claim limited to Linux would still need explicit wording that
macOS and Windows remain unsupported for shared runtime behavior.

A Sprint 153 decision to continue static-first packaging is defensible if it
adds stronger diagnostics and documentation tying the deferral to exact
blockers:

- no visibility/export policy;
- no SONAME/install-name/import-library policy;
- no downstream shared consumer proof;
- no runtime loader proof;
- no dynamic ABI policy for public concrete structs, callbacks, allocators,
  and error state.

## Day 4 Handoff

Day 4 should convert this audit into a decision matrix. Minimum viable shared
support must require platform-specific build, install, exported-symbol,
metadata, downstream consumer, and loader proof. Minimum viable static-first
deferral must make the exact blockers test-backed and keep package metadata
free of unsupported shared-library, dynamic ABI, and loader claims.
