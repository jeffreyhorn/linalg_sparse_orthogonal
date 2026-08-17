# Sprint 162 Day 7 Implementation Completion

## Scope

Day 7 completes the retained non-claim guard implementation started on Day 6.
The goal is to make the unsupported Windows package execution surfaces fail
clearly while preserving existing package behavior.

The selected Sprint 162 product decision remains unchanged:

- Windows CMake install/downstream validation is reviewed support.
- Windows `sparse.pc` is installed and inspected as metadata.
- Windows `pkg-config` command execution parity remains a non-claim.
- Windows Makefile install/uninstall parity remains a non-claim.
- Linux/macOS Make install and `pkg-config` execution proof remains reviewed
  support.

## Completed Implementation Changes

| File | Change | Diagnostic Value |
| --- | --- | --- |
| `scripts/static_package_deferral_check.sh` | Added `check_windows_workflow_no_unselected_package_execution`. | Fails if the Windows workflow starts running `pkg-config` or `make install`/`make uninstall` without a new product decision and proof path. |
| `scripts/static_package_deferral_check.sh` | Kept Day 6 wording checks for README, INSTALL, maintainer guide, and Windows workflow. | Fails if docs stop distinguishing Windows CMake-first validation from Windows Makefile or `pkg-config` execution parity. |
| `docs/planning/EPIC_14/SPRINT_162/WORKING_NOTES.md` | Added Day 7 completion log. | Preserves the implementation evidence trail. |

No C source, public header, package template, install rule, or Windows workflow
runtime step was changed.

## Focused Guard Coverage

The completed local guard now covers three retained boundaries:

1. **Static-first package boundary:** shared-library packaging, dynamic ABI,
   runtime-loader behavior, package-manager support, and static/shared
   selectors remain unsupported.
2. **Windows wording boundary:** public docs and the Windows workflow must
   keep Windows CMake install/downstream validation separate from Windows
   Makefile and `pkg-config` execution parity.
3. **Windows execution boundary:** the Windows workflow must not start running
   `pkg-config`, `make install`, or `make uninstall` unless the retained
   non-claim decision is replaced with a reviewed proof path.

## Diagnostic Coverage Notes

| Drift | Current Diagnostic |
| --- | --- |
| README loses CMake-first Windows support wording | `README no longer states that Windows package support remains CMake-first` |
| README loses Windows Makefile non-claim | `README no longer keeps Windows Makefile parity as a non-claim` |
| README loses Windows `pkg-config` non-claim | `README no longer keeps Windows pkg-config execution parity as a non-claim` |
| INSTALL loses CMake install/downstream wording | `INSTALL no longer describes Windows CMake install/downstream validation` |
| Maintainer guide loses Windows package boundary | `maintainer guide no longer describes Windows CMake install/downstream validation` |
| Windows workflow stops identifying `sparse.pc` as metadata | `Windows workflow no longer identifies sparse.pc as metadata` |
| Windows workflow starts running `pkg-config` | `Windows workflow started executing pkg-config without a selected provider and downstream proof` |
| Windows workflow starts running `make install` or `make uninstall` | `Windows workflow started executing make install/uninstall without a reviewed Windows Makefile parity decision` |

## Focused Local Validation

Run during Day 7:

```sh
bash scripts/static_package_deferral_check.sh
rg -n "[ \t]+$" docs/planning/EPIC_14/SPRINT_162 scripts/static_package_deferral_check.sh
git diff --check
git status --short -- '*.c' '*.h'
```

Observed static guard result:

```text
static-package-deferral-check: BUILD_SHARED_LIBS rejection ok
static-package-deferral-check: static target declaration ok
static-package-deferral-check: static install metadata ok
static-package-deferral-check: no shared export/ABI metadata found ok
static-package-deferral-check: package metadata has no static/shared selector ok
static-package-deferral-check: support wording remains deferred ok
static-package-deferral-check: Windows package non-claim wording ok
static-package-deferral-check: Windows workflow has no unselected package execution ok
static-package-deferral-check: passed
```

## Day 7 Conclusion

The retained non-claim guard is complete locally for the first implementation
cycle. The repository now has an executable check that preserves the Windows
CMake-first package support tier and prevents accidental workflow promotion of
Windows Makefile or `pkg-config` execution parity.

Day 8 should review CI wording and expected-count comments for consistency
with this completed guard, without adding unselected Windows package execution
steps.
