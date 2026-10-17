# Sprint 214 Day 6: Static Build/Configure Deferral Guard

## Selected Scope

Day 2 selected complete ABI/release design with stronger static-first
deferral. Day 6 implements only the first configure/metadata guard pass. It
does not add a shared target, static/shared selector, ABI epoch, import
library, SONAME, install-name, DLL, or public support claim. Day 3's export
list and Days 4-5's ABI/loader policies remain future acceptance designs.

## Implementation

| Owner | Change | Contract |
| --- | --- | --- |
| `CMakeLists.txt` | Reject cache variables whose names begin with `SPARSE_` and contain `SHARED`, plus `SPARSE_LINKAGE`, `SPARSE_LINK_MODE`, `SPARSE_LIBRARY_TYPE`, and `SPARSE_LIBRARY_KIND`. Also reject noncanonical casing of `BUILD_SHARED_LIBS`. | Package-prefixed link-mode selectors are not implemented. Even `OFF` or `STATIC` values fail clearly instead of being silently ignored; canonical `BUILD_SHARED_LIBS=ON` retains its existing actionable rejection. |
| `scripts/static_package_deferral_check.sh` | Configure eleven unsupported selector fixtures in isolated temporary build directories and require the intended static-only diagnostic. | A failed configure for an unrelated reason does not count as a pass. The matrix includes positive shared requests, an `OFF` alias, a `STATIC` link-type alias, mixed-case names, and lower-case `build_shared_libs`. |
| `scripts/static_package_deferral_check.sh` | Configure the default package in a temporary build tree and inspect generated `sparse.pc` and `SparseConfig.cmake`. | Generated `pkg-config` description and link line remain static archive metadata; neither generated file advertises shared/ABI/import behavior. This checks generated output in addition to checked-in templates. |

The existing explicit `STATIC` CMake target, archive-only install rule,
source-level deferral checks, and `BUILD_SHARED_LIBS=ON` rejection remain in
place. No new CMake option is introduced: unsupported names are rejected,
not recognized as configuration switches.

## Focused Acceptance Matrix

| Fixture or command | Expected result | Local Day 6 result |
| --- | --- | --- |
| `bash scripts/static_package_deferral_check.sh` | Existing product-decision/static checks, eleven unsupported selector rejections, and default generated metadata all pass. | Passed locally. |
| `-DBUILD_SHARED_LIBS=ON` | Configure fails with static archive, export/visibility, ABI, and loader blockers. | Passed inside the guard. |
| `-DSPARSE_BUILD_SHARED=ON` and `=OFF`, `-DSPARSE_BUILD_SHARED_LIBS=ON`, `-DSPARSE_ENABLE_SHARED=ON`, `-Dsparse_shared=on` | Configure fails with the unsupported-selector diagnostic, independent of value or case. | Passed inside the guard. |
| `-DSPARSE_LIBRARY_TYPE=SHARED` and `=STATIC`, `-DSPARSE_LIBRARY_KIND=SHARED`, `-DSPARSE_LINKAGE=shared`, `-DSPARSE_LINK_MODE=shared`, `-Dbuild_shared_libs=ON` | Configure fails rather than appearing to accept a package link-mode choice. | Passed inside the guard. |
| Default CMake configure | Generated package files still describe static archive linkage and no shared/ABI import. | Passed inside the guard. |
| `bash tests/test_cmake_install.sh` | Build/install static archive, export `STATIC IMPORTED`, reject shared metadata, and run installed out-of-tree consumers. | Passed: 27 passed, 0 failed, 0 skipped. |
| `bash tests/test_install.sh` | Make installs only static archive/headers/`sparse.pc`; installed consumers build/run and uninstall succeeds. | Passed: 23 passed, 0 failed. |
| `bash -n scripts/static_package_deferral_check.sh` | Shell syntax valid. | Passed locally. |

All results are local macOS results, not new hosted Linux/Windows results.
The CMake exact package-version check remains a static package-resolution
test, not an ABI compatibility test.

## Remaining Risks And Day 7 Handoff

This first pass checks known and package-prefixed selector spellings, not all
possible CMake cache variables. A third-party wrapper can still invent an
unknown option; absent an explicit product decision, it must not be
documented as a supported mode. Source-text and configure-generated checks
do not by themselves prove the final installed prefix is clean. Day 7 owns
the stronger installed-file and package-metadata inventory, including Linux
`.so*`, macOS `.dylib`, Windows `.dll` and import-library metadata, without
misclassifying a legitimate static Windows `.lib` by extension alone. Day 8
owns CI/claim review and further negative fixtures as needed.

If a new check rejects a maintained static build, preserve the existing
`BUILD_SHARED_LIBS=ON` rejection and static target while correcting or
reverting only the new check. If a shared artifact or selector appears in the
installed package, stop; do not broaden support wording to make the test
pass. No C or header files changed on Day 6, so the full C format/lint/test
gate is not required. Markdown whitespace validation is recorded in
`WORKING_NOTES.md`.
