# Sprint 214 Day 7: Static Install And Metadata Contract

## Selected Boundary

Day 2 selected ABI/release design with stronger static-first deferral. Day 7
strengthens installed-prefix checks; it does not add a shared artifact,
link-mode selector, ABI epoch, loader contract, or package-manager release
claim. Day 5's Linux/macOS/Windows loader requirements remain future gates.

## Observed Local Install Inventories

| Local path | Installed library/package files | Consumer evidence |
| --- | --- | --- |
| Make install into a temporary prefix | `lib/libsparse_lu_ortho.a`, `lib/pkgconfig/sparse.pc`, and 19 installed headers; no CMake package or runtime binary. | `pkg-config` resolves version `2.2.0` and installed paths/flags; both basic and maintained example consumers compile, link, and run. `bash tests/test_install.sh`: 24 passed, 0 failed. |
| CMake install into a temporary prefix | `lib/libsparse_lu_ortho.a`, `lib/pkgconfig/sparse.pc`, `SparseConfig.cmake`, `SparseConfigVersion.cmake`, `SparseTargets.cmake`, `SparseTargets-noconfig.cmake`, and 19 installed headers; no runtime binary. | `Sparse::sparse_lu_ortho` is `STATIC IMPORTED` and points at the installed archive. Out-of-tree example and exact-version consumers build/run; mismatched version is rejected. `bash tests/test_cmake_install.sh`: 28 passed, 0 failed, 0 skipped. |

Both runs are local macOS evidence. A CMake or `pkg-config` consumer can link
the static archive; neither route proves dynamic loading or cross-version ABI
compatibility. Windows CMake static install/downstream behavior remains
scoped to the existing hosted workflow, not this local run.

## Installed-Prefix Validator

`scripts/check_static_install_contract.py` now validates temporary Make and
CMake prefixes after installation. Both install tests invoke it with their
known version and Unix static archive name. It requires the expected archive,
checks the `lib/` file inventory and empty `bin/` runtime surface, and rejects
`.so*`, `.dylib`, `.dll`, `.dll.a`, `.tbd`, `.exp`, and framework artifacts
anywhere in the prefix. It parses the supported `sparse.pc` fields and
variables, requiring the prefix/lib/include variables to resolve to the
installed paths and the fields to retain static description, version, include
flags, and archive link flags with no extra selector or ABI field.

For CMake installs, it requires only the expected package files, exactly the
`Sparse::sparse_lu_ortho STATIC IMPORTED` declaration, a version matching
`sparse.pc`, and **every** imported configuration location pointing to the
installed static archive. It rejects dynamic imported locations, import-lib
and SONAME fields, loader metadata, static/shared selectors, and extra package
targets/files. Make installs must not contain CMake package metadata.

The validator accepts the expected Windows `lib/sparse_lu_ortho.lib` name
*only* with static imported-target metadata and no extra `.lib` file. A `.lib`
suffix by itself is not treated as proof of an import library. This simulated
Windows fixture does not inspect binary members or replace a hosted MSVC
consumer and controlled DLL-search test.

## Negative-Test Results

`python3 tests/test_static_install_contract.py` passed 10 tests locally.
The temporary-fixture matrix covers:

| Mutation | Expected rejection |
| --- | --- |
| Extra `.so`, versioned `.so`, `.dylib`, `.dll`, `.dll.a`, `.tbd`, or runtime file under `bin/` | Unexpected shared/import or runtime artifact. |
| Extra Windows `.lib` beside the expected static `.lib` | Unexpected installed library file. |
| `SHARED IMPORTED`, `IMPORTED_IMPLIB`, `IMPORTED_SONAME`, ABI marker, or static/shared selector in CMake package files | Unsupported loader/selector metadata. |
| Additional dynamic imported location or second wrong static location | Dynamic path or imported-location mismatch, even when one correct archive path remains. |
| Extra shared target file or CMake metadata in a Make install | Unexpected package file/surface. |
| `Libs.private`, shared-package description, altered version, ABI variable, or redirected prefix/libdir in `sparse.pc` | Package fields/variables differ from the installed static contract. |

Positive fixtures include Unix Make, Unix CMake, and a Windows-style static
`.lib` plus `STATIC IMPORTED` target. The tests validate the checker itself;
the real temporary-prefix runs above validate integration with generated
Make/CMake metadata.

## Scope, Risks, And Handoff

No `Makefile`, `CMakeLists.txt`, `sparse.pc.in`, `cmake/SparseConfig.cmake.in`,
public docs, or C/header file changed on Day 7. Their existing static-only
descriptions remained consistent with installed output. The checker covers
the default installed package used by these tests; optional mutex/OpenMP
link-flag variants need separate fixtures before treating this validator as
their proof. The local Windows-style fixture checks names and metadata, not
COFF archive contents. Day 8 should review CI wiring and hosted Windows
artifact/consumer diagnostics before citing this as a cross-platform gate.

If a check rejects a valid maintained static configuration, correct the
checker or its fixture without weakening `BUILD_SHARED_LIBS=ON` rejection or
accepting a shared artifact. If installed output and metadata disagree, stop
the promotion; do not change the public support matrix to fit the artifact.
Day 7 changed shell/Python/tests/planning files only, so no C/header
format-lint-test gate was required. Whitespace validation is recorded in
`WORKING_NOTES.md`.
