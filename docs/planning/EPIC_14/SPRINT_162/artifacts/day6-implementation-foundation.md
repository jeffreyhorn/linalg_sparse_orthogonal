# Sprint 162 Day 6 Implementation Foundation

## Scope

Day 6 starts the retained non-claim implementation selected on Day 4 and
designed on Day 5. The implementation pass focuses on the executable static
package guard rather than package behavior changes.

The change keeps the Windows package support tier CMake-first and static-first:

- no Windows Makefile install/uninstall parity claim;
- no Windows `pkg-config` command execution parity claim;
- installed `sparse.pc` on Windows remains package metadata inspection only;
- Linux/macOS Make and `pkg-config` proof remains unchanged;
- shared-library, dynamic ABI, runtime-loader, package-manager, and
  static/shared selector non-claims remain unchanged.

## Implementation Patch Set

| File | Change | Reason |
| --- | --- | --- |
| `scripts/static_package_deferral_check.sh` | Added `check_windows_package_nonclaim_wording`. | Make the retained Windows Makefile and `pkg-config` non-claims executable through the existing static package guard. |
| `docs/planning/EPIC_14/SPRINT_162/WORKING_NOTES.md` | Added Day 6 implementation log. | Preserve the sprint evidence trail. |
| `docs/planning/EPIC_14/SPRINT_162/artifacts/day6-implementation-foundation.md` | Added this artifact. | Record the first implementation pass and validation output. |

No package templates, install rules, workflow behavior, C source, or public
headers were changed in this pass.

## New Guard Coverage

The static package deferral guard now checks that:

1. `README.md` states that Windows remains CMake-first.
2. `README.md` keeps Windows Makefile parity and `pkg-config` execution parity
   as non-claims.
3. `INSTALL.md` describes Windows CMake install/downstream validation.
4. `INSTALL.md` keeps Windows Makefile parity and Windows `pkg-config`
   execution parity as non-claims.
5. `docs/maintainer_guide.md` describes Windows CMake install/downstream
   validation.
6. `docs/maintainer_guide.md` keeps Windows Makefile parity and `pkg-config`
   execution parity as non-claims.
7. `.github/workflows/windows-ci.yml` identifies `sparse.pc` as metadata.
8. `.github/workflows/windows-ci.yml` states that Windows does not claim
   Makefile or `pkg-config` execution parity.
9. `.github/workflows/windows-ci.yml` keeps the reviewed Windows install lane
   scoped to CMake install/downstream proof rather than Makefile or
   `pkg-config` execution parity.

## Preserved Surfaces

The Day 6 patch intentionally leaves these surfaces untouched:

- `sparse.pc.in`;
- `CMakeLists.txt`;
- `cmake/SparseConfig.cmake.in`;
- `tests/test_install.sh`;
- `tests/test_cmake_install.sh`;
- `.github/workflows/windows-ci.yml` runtime behavior;
- library `.c` and `.h` files.

## Focused Local Validation

Run during Day 6:

```sh
bash scripts/static_package_deferral_check.sh
rg -n "[ \t]+$" docs/planning/EPIC_14/SPRINT_162 scripts/static_package_deferral_check.sh
git diff --check
git status --short -- '*.c' '*.h'
```

Expected result:

- static package guard passes, including the new Windows package non-claim
  wording check;
- no trailing whitespace;
- no diff hygiene errors;
- no C or header modifications.

Observed static guard result:

```text
static-package-deferral-check: BUILD_SHARED_LIBS rejection ok
static-package-deferral-check: static target declaration ok
static-package-deferral-check: static install metadata ok
static-package-deferral-check: no shared export/ABI metadata found ok
static-package-deferral-check: package metadata has no static/shared selector ok
static-package-deferral-check: support wording remains deferred ok
static-package-deferral-check: Windows package non-claim wording ok
static-package-deferral-check: passed
```

## Day 6 Conclusion

The retained non-claim decision now has an executable implementation
foundation. The next pass should complete diagnostics and any remaining
support-tier wording alignment without changing the package contract or
promoting Windows Makefile or `pkg-config` execution parity.
