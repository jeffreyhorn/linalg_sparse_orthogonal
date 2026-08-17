# Sprint 162 Day 3 Metadata Boundary Review

## Scope

Day 3 reviews the package metadata and static-first boundaries that constrain
the Sprint 162 Windows package parity decision. The goal is not to promote a
new Windows support claim yet; it is to make the existing metadata surface and
unsupported-surface risks explicit before the Day 4 product decision.

## Static-First Metadata Inventory

| Surface | Source | Current Boundary | Sprint 162 Implication |
| --- | --- | --- | --- |
| Shared-library configure gate | `CMakeLists.txt` | `BUILD_SHARED_LIBS=ON` is rejected with a fatal error. The message names shared-library packaging, dynamic ABI, symbol visibility, SONAME, install-name/RPATH, Windows DLL/import-library behavior, installed shared consumer proof, and runtime-loader validation as deferred. | The package contract remains static archive only. |
| Library target | `CMakeLists.txt` | `sparse_lu_ortho` is declared as a `STATIC` library. | Installed CMake metadata should describe a static archive, not a shared target. |
| CMake install target | `CMakeLists.txt` | Install rules export the target and install only the archive artifact through `ARCHIVE DESTINATION`. | No `RUNTIME` or shared `LIBRARY` install claim is present. |
| Public headers | `CMakeLists.txt` | Public headers and the generated version header are installed under `include/sparse`. | The package exposes headers consistently across install routes. |
| CMake package template | `cmake/SparseConfig.cmake.in` | The package includes `SparseTargets.cmake` and checks required components. It does not expose static/shared selectors. | CMake consumers get the installed target without a product claim for dynamic ABI selection. |
| CMake package version | `CMakeLists.txt` | `SparseConfigVersion.cmake` is generated with exact-version compatibility. | Installed CMake consumers must request the exact maintained package version. |
| `pkg-config` template | `sparse.pc.in` | The description says static archive package metadata. `Cflags` uses `${includedir}` and `Libs` links `-lsparse_lu_ortho -lm` plus configured extras. | Unix-like `pkg-config` metadata is intentionally static and does not advertise private shared dependencies. |
| Make install proof | `tests/test_install.sh` | Linux/macOS validation checks static archive install, headers, `sparse.pc`, exact version, variables, flags, downstream `pkg-config` compile/link/run, maintained example, and uninstall. | This remains the reviewed Make and `pkg-config` execution proof baseline. |
| CMake install proof | `tests/test_cmake_install.sh` | CMake install validation checks static imported target metadata, exact-version behavior, downstream CMake consumers, `sparse.pc` metadata, and no shared-loader metadata. | This proof validates installed CMake consumers and metadata, not Makefile parity. |
| Windows install proof | `.github/workflows/windows-ci.yml` | Windows installs a static `.lib`, headers, CMake package files, and `sparse.pc`; validates CMake consumers, exact version, mismatch rejection, static imported target metadata, no DLLs, no shared imports, and no unsupported wording. | Windows has CMake-first package proof plus `sparse.pc` metadata inspection, not `pkg-config` execution parity. |
| Static deferral guard | `scripts/static_package_deferral_check.sh` | The guard rejects drift toward shared targets, shared install destinations, export/import macros, dynamic ABI wording, runtime-loader wording, package-manager wording, and package selector wording. | This is the right home for retained non-claim checks if Day 4 chooses guard strengthening. |

## Unsupported-Wording Audit

The current metadata and documentation correctly treat these as unsupported or
deferred:

- shared-library builds and packaging;
- dynamic ABI compatibility;
- runtime-loader behavior;
- static/shared package selectors;
- package-manager distribution through Homebrew, apt, dnf, pacman, vcpkg,
  Conan, or similar systems;
- Windows Makefile install/uninstall parity;
- Windows `pkg-config` command execution parity;
- broad Windows parity beyond the reviewed CMake subset.

The main wording risk is not an existing positive claim. It is ambiguity caused
by installing `sparse.pc` on Windows while not running `pkg-config` there. That
metadata is useful for package completeness and cross-platform install shape,
but it can be misread as proof that a reviewed Windows `pkg-config` provider,
compiler flag mapping, and downstream command path exist.

The second risk is equivalence drift between CMake and Make/pkg-config support.
The Windows lane proves installed CMake consumers and package metadata. The
Linux/macOS lane proves Make install, uninstall, `pkg-config` resolution,
flags, and downstream compile/link/run. Sprint 162 should keep these proofs
separate unless Day 4 explicitly promotes a Windows Make or `pkg-config`
execution route.

The third risk is flag-shape drift. The `sparse.pc` template emits Unix-style
`-I`, `-L`, `-l`, and `-lm` flags. That is correct for the maintained Unix-like
proof path, but it is not enough by itself to support an MSVC downstream
consumer on Windows.

## Retained Non-Claim Guard Candidates

If Day 4 retains Windows Makefile and `pkg-config` execution as non-claims, the
strongest next implementation path is guard strengthening rather than new
package proof. Candidate checks:

1. Extend `scripts/static_package_deferral_check.sh` to keep Windows support
   wording tied to CMake install/downstream validation, not Makefile or
   `pkg-config` execution parity.
2. Add a documentation wording guard for `README.md`, `INSTALL.md`, and
   `docs/maintainer_guide.md` that requires explicit Windows Makefile and
   Windows `pkg-config` non-claims whenever Windows package validation is
   described.
3. Keep the Windows workflow metadata check focused on `sparse.pc` file
   presence and static description, with comments or assertions clarifying that
   no `pkg-config` command is executed in the reviewed Windows lane.
4. Preserve sparse.pc unsupported-wording checks for shared libraries, dynamic
   ABI, package-manager availability, runtime-loader behavior, and
   static/shared selectors.
5. Preserve CMake package unsupported-selector checks so `find_package(Sparse)`
   cannot imply a static/shared product choice.
6. Add a planning or report index note that separates CMake install consumer
   evidence from Make/pkg-config execution evidence.

If Day 4 promotes Windows `pkg-config` parity instead, these checks must be
replaced with provider-specific proof: selected `pkg-config` implementation,
compiler/toolchain flag expectations, installed variable checks, downstream
compile/link/run, exact version, and failure diagnostics.

If Day 4 promotes Windows Makefile parity, these checks must be replaced with a
reviewed Windows shell/make/install/uninstall route and evidence that it does
not weaken the CMake-first package surface.

## Package Metadata Risk Register

| Risk | Severity | Evidence | Mitigation Candidate |
| --- | --- | --- | --- |
| `sparse.pc` installed on Windows is mistaken for `pkg-config` execution support. | High | Windows CMake install lane checks the file and contents but does not run `pkg-config`. | Add explicit retained non-claim checks in docs and workflow comments/assertions. |
| Unix-style `pkg-config` flags do not map cleanly to MSVC consumers. | High | `sparse.pc.in` emits `-I`, `-L`, `-l`, and `-lm`. | Do not claim Windows `pkg-config` parity without a selected provider and downstream proof. |
| Makefile install/uninstall assumes POSIX shell utilities. | High | Unix validation uses Make install/uninstall and shell-driven `pkg-config` checks. | Do not claim Windows Makefile parity without a reviewed Windows Make route. |
| Shared-library or dynamic ABI wording drifts back into package metadata. | High | Static-first policy is enforced by CMake rejection and static deferral guard. | Keep static guard active and extend only with source-backed wording. |
| CMake package metadata grows static/shared selectors before the product decision. | Medium | Current package template has no selectors. | Preserve selector scans in the static deferral guard. |
| Public docs conflate Windows CMake package proof with broader Windows package parity. | Medium | README, INSTALL, and maintainer guide mention Windows CMake-first boundaries. | Add retained non-claim wording checks if non-claim is selected. |
| Installed package metadata leaks build-tree or source-tree paths. | Medium | CMake install validation already checks installed metadata for source/build leaks. | Keep existing CMake install tests unchanged. |

## Day 3 Conclusion

The static-first package boundary is explicit in source, package metadata, CI,
and documentation. The remaining Sprint 162 decision is not whether Windows has
a package surface; it does. The decision is whether to promote a Windows
Makefile or `pkg-config` execution proof, or retain those as deliberate
non-claims with stronger guards.

Day 4 should therefore choose one of two clear tracks:

- promote a specific Windows execution path and add direct proof for it; or
- retain the current CMake-first Windows scope and add checks that prevent
  installed metadata from being interpreted as Makefile or `pkg-config`
  parity.
