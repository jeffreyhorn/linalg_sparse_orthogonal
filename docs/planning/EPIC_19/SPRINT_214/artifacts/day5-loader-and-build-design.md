# Sprint 214 Day 5: Loader Policy And Build/Install Design

## Decision Boundary And Local Baseline

Day 2 selected ABI/release design with stronger static-first deferral, not a
shared-library proof. This document finishes the **future** loader policy for
item 214.2 and designs the **selected** item 214.3 guard work for Days 6-8.
It does not enable a shared target, select an ABI epoch, install a dynamic
artifact, or expand platform claims. Day 3's eight-function export list is
provisional; Day 4's ABI identity and retained-binary requirements still
apply before a future dynamic-ABI decision.

Today `CMakeLists.txt` rejects `BUILD_SHARED_LIBS=ON`, creates one explicit
`STATIC` target, and installs only its archive plus headers and static package
metadata. Make installs `libsparse_lu_ortho.a`. The current CMake install test
passed 27/27 checks locally on Day 4, and
`bash scripts/static_package_deferral_check.sh` passed locally on Day 5.
Those tests exercise the static contract, not runtime loading.

## Future Loader Contract By Platform

Every row below is a gate for a *later, explicit* shared-product decision.
Example names illustrate the ABI-identity requirement; the final filename,
epoch, minimum OS/toolchain, and install layout must be selected and recorded
before implementation. An artifact that happens to link does not satisfy a
row.

| Platform proof lane | Artifact and identity requirement | Installed inspection and runtime gate | Missing/wrong-library negative gate |
| --- | --- | --- | --- |
| Linux ELF, named GCC/Ubuntu lane | Real shared object and ABI-distinguishing SONAME, for example `libsparse_lu_ortho.so.<epoch>`; development symlink and installed CMake/`pkg-config` references must resolve to that object. Index/scalar variants need distinct identity or enforced mismatch rejection. | Inspect installed ELF `DT_SONAME`, `DT_NEEDED`, and `DT_RUNPATH`/`DT_RPATH` with `readelf -d` or equivalent. No build-tree path or unexpected dependency. Build and run an out-of-tree consumer using only installed headers/package targets and the intended installed-prefix runtime search policy. Record `ldd`/loader resolution as diagnostic evidence, not as a replacement for execution. | Remove or substitute the installed object, or use wrong epoch/width: consumer must fail predictably before selected API use. No accidental fallback to the build tree or another prefix. |
| macOS Mach-O, named Apple Clang lane | Installed dylib identity must be ABI-distinguishing; a possible form is `@rpath/libsparse_lu_ortho.<epoch>.dylib`. Current/compatibility versions need an explicit mapping to ABI policy, not an assumption that package `VERSION` is ABI. | Inspect `LC_ID_DYLIB`, consumer `LC_LOAD_DYLIB`, and `LC_RPATH` with `otool -D`, `otool -L`, and `otool -l` or equivalent. Consumer must run from an out-of-tree installed prefix using a documented `@rpath`/`@loader_path` or executable-relative policy, without a build-tree dependency. | Relocate the prefix under the selected policy and test it, then remove or mismatch the dylib and verify loader failure. A successful link against a build-tree dylib is not proof. |
| Windows PE/COFF, named MSVC lane | DLL filename must distinguish incompatible ABI variants; its import `.lib` and CMake `IMPORTED_IMPLIB`/`IMPORTED_LOCATION` must bind to the same installed DLL. A static `.lib` is not an import library merely because of its extension. | Inspect DLL exports and dependencies with `dumpbin /EXPORTS` and `/DEPENDENTS` or equivalent, and inspect the consumer's import table. Build/run an out-of-tree installed consumer with the declared DLL search and MSVC CRT/allocator policy. Compare DLL exports to the approved exact manifest. | Remove the DLL from the controlled runtime search location, substitute wrong epoch/architecture, or omit its import library: build/load must fail as expected, with no accidental PATH pickup. |

For each claimed lane, record artifact paths, hashes, target architecture,
compiler/runtime identity, export-list comparison, package files, and hosted
CI job output. Search-path configuration must be explicit and reproducible;
`LD_LIBRARY_PATH`, `DYLD_LIBRARY_PATH`, or `PATH` can isolate test fixtures but
must not silently stand in for the product's selected loader contract.
Runtime loading and old-binary/new-library execution are separate tests. No
Linux result implies macOS or Windows loader support, and none currently has
a reviewed shared lane.

## Artifact, Metadata, And Consumer Resolution Rules

Before enabling any future shared target, compare the **installed prefix**
with a source-controlled manifest that binds Day 4's ABI epoch, index/scalar
tuple, exact approved exports, platform ABI, artifact name, and loader ID.
Inspect both the file and the metadata: a CMake imported target must point to
the installed artifact (and Windows import library), and `sparse.pc` must
advertise only an actually supported link mode. The header configuration and
consumer compile definitions must agree with the selected binary. Reject
missing or contradictory metadata rather than inferring ABI identity from
`VERSION=2.2.0` or CMake `ExactVersion`.

A future same-build consumer must be configured outside the source/build
tree against a clean temporary prefix, linked through the installed package,
and executed under a controlled runtime search path. A retained old
executable, compiled against an older installed baseline, is additionally
required before claiming same-epoch backward compatibility. The Day 4 policy
owns that cross-version test and rollback rule.

## Selected Day 6-8 Static Deferral Design

No shared target is designed for implementation in this sprint. The selected
change is a bounded set of guard and install-test improvements:

| Owner and file | Planned change | Acceptance condition |
| --- | --- | --- |
| CMake contract: `CMakeLists.txt` | Keep actionable `BUILD_SHARED_LIBS=ON` rejection, explicit `STATIC` target, and archive-only install. Do not add `SHARED`, `MODULE`, `SOVERSION`, install-name, import-library, or runtime destination. | Configure with `BUILD_SHARED_LIBS=ON` exits nonzero; default build/install contains only the static target. |
| Guard: `scripts/static_package_deferral_check.sh` | Extend checks to catch alternate shared target/selector spellings and unselected runtime destinations or loader fields, while retaining the Sprint 170 decision and public non-claim checks. Prefer checking the configured target/install result where source text alone is ambiguous. | Baseline passes; each controlled injected selector, shared target, runtime destination, or ABI/export metadata fixture fails with a specific diagnostic. |
| CMake install test: `tests/test_cmake_install.sh` | Keep static imported-target and exact-version consumer tests; expand installed-prefix inventory and generated target/package checks for shared objects, DLL/import metadata, loader names, and selectors. | Clean temporary install passes; injected shared artifact or imported metadata is rejected. Do not misclassify the Windows static `.lib` as an import library solely by suffix. |
| Make install test: `tests/test_install.sh` | Preserve `pkg-config` static consumer and archive install checks; verify no unselected dynamic artifact or shared metadata is retained. | Static consumer still builds/runs; a staged `.so*`, `.dylib`, or `.dll` in the install prefix fails. |
| Package metadata: `sparse.pc.in`, `cmake/SparseConfig.cmake.in` | Remain static-scoped. No component/selector, runtime loader promise, ABI epoch, or library alternative is advertised. | Installed `Sparse::sparse_lu_ortho` remains `STATIC IMPORTED`; `sparse.pc` still describes the static archive and version without implying dynamic ABI. |
| Public claim owners: `README.md`, `INSTALL.md`, `docs/maintainer_guide.md` | Preserve the static-first support matrix and shared/ABI/loader/release non-claims; update only if a guard needs an exact marker. | Relevant docs checks reject a positive shared/loader/ABI claim without a later proof decision. |
| CI: `.github/workflows/ci.yml`, `macos-ci.yml`, `windows-ci.yml` | Keep existing static install/export lanes as owners of the strengthened checks; add wiring only if a new test is not already reached. | Hosted evidence remains platform-scoped: Linux/macOS static install, Windows CMake static downstream and metadata-only `sparse.pc` inspection. |

Source-text inspection is useful for a fast policy guard but is not an
installed-artifact proof. The negative fixtures should create temporary
copies/prefixes, not edit the working tree, and must exercise both guard text
and install-result checks. No test may be made green by accepting a new
shared artifact under the current decision.

## Negative-Fixture Matrix And Rollback

| Fixture for Days 6-8 | Expected result |
| --- | --- |
| `BUILD_SHARED_LIBS=ON`, including an otherwise unused selector | Configure rejection with the existing static-first explanation. |
| Explicit second `SHARED`/`MODULE` target, custom shared output, or new static/shared component | Guard rejects even if the original static target still exists. |
| Runtime/library install destination, generated `.so*`/`.dylib`/`.dll`, or CMake shared imported target/loader fields | Guard or temporary-prefix install test rejects before support wording changes. |
| `sparse.pc` or CMake package advertises shared linkage, ABI epoch, or provider/selector not backed by the selected product decision | Metadata guard rejects; static package version is not treated as ABI proof. |
| Removed shared/ABI deferral line or new positive support claim | Docs guard rejects while current static tests remain green. |
| Unchanged static package on Linux/macOS/Windows maintained lanes | Existing installed static consumers and platform-scoped metadata checks pass. |

If any strengthened guard breaks a maintained static install or cannot
distinguish a legitimate static Windows `.lib` from an import library, revert
only that new guard change and keep the existing static-first rejection in
force. If a future experimental shared build escapes into an install or
package file, remove that experimental output and restore archive-only
metadata before proceeding. Do not weaken `BUILD_SHARED_LIBS=ON` rejection or
declare a shared support tier to accommodate a failing fixture. Failed
required checks block Day 8/14 closeout and must be reported.

## Day 5 Completion And Handoff

Item 214.2 is complete as **policy design**: Day 3 defines provisional
symbols, Day 4 defines ABI identity/compatibility, and this artifact defines
per-platform loader and consumer gates. None is an implemented or supported
dynamic ABI. Days 6-8 own the guard/test changes in the table above; Day 9/10
own static downstream evidence and future dynamic consumer fixture detail.
The selected acceptance gate is static configure/install/metadata/docs
rejection plus unchanged static consumer behavior, not production of a
shared object.

Day 5 changed planning Markdown only. The local static package deferral
guard passed; final whitespace validation is recorded in `WORKING_NOTES.md`.
