# Sprint 162 Day 8 CI Alignment

## Scope

Day 8 aligns Windows CI wording with the Day 4 retained non-claim product
decision and the Day 6-7 static guard implementation.

The supported Windows package surface remains:

- reviewed CMake configure/build/CTest on MSVC;
- reviewed CMake install/downstream validation;
- installed static `.lib`, headers, CMake package metadata, and `sparse.pc`
  metadata;
- CMake downstream consumers, exact-version behavior, and mismatch-version
  rejection;
- no DLL/shared imported metadata, loader metadata, or static/shared selector
  metadata.

Windows still does not claim Makefile parity or `pkg-config` execution parity.

## CI Review

| Workflow Area | Current State | Day 8 Result |
| --- | --- | --- |
| Workflow runner | `windows-2022` | Preserved. |
| CTest count | `EXPECTED_WINDOWS_CTEST_COUNT: "59"` | Preserved; no tests were added or removed. |
| CMake configure/build lane | Visual Studio 17 2022, x64, Release | Preserved. |
| CTest inspection output | Reports promoted portable tests and CMake-first proof scope. | Updated wording to say `sparse.pc` is metadata-only inspection. |
| Install/downstream job name | `Windows reviewed CMake install/downstream validation path` | Preserved. |
| Install/downstream package checks | Static `.lib`, headers, version header, CMake package files, `sparse.pc` metadata, no DLLs, no shared selectors. | Preserved. |
| Downstream consumers | Generated and maintained installed CMake consumers plus exact-version consumer. | Preserved. |
| Unselected package execution | No Windows `pkg-config`, `make install`, or `make uninstall` commands. | Preserved and now guarded by `scripts/static_package_deferral_check.sh`. |

## Wording Updates

Day 8 made a wording-only update to `.github/workflows/windows-ci.yml`:

- changed the top-level package surface description from generic
  `sparse.pc` metadata to `sparse.pc` metadata-only inspection;
- clarified the CTest inspection log line so the hosted output separates
  installed `sparse.pc` metadata from Windows `pkg-config` command proof;
- clarified the install/downstream job comment so reviewers see the lane is
  CMake install/downstream scoped and does not claim Windows Makefile or
  `pkg-config` execution parity.

No workflow command behavior changed.

## Expected-Count And Staged-Exclusion Notes

The Windows CTest count remains 59 because Day 8 did not add or remove tests.
The promoted portable CTest targets remain in scope:

- `test_threads`;
- `test_sprint4_integration`;
- `test_fuzz`.

The staged package exclusions remain:

- no Windows Makefile install/uninstall proof;
- no Windows `pkg-config` provider;
- no Windows `pkg-config --exists`, `--cflags`, `--libs`, or `--modversion`
  proof;
- no downstream Windows consumer compiled from `pkg-config` output.

## Workflow Validation Notes

The local Sprint 162 validation for Day 8 is source-level because hosted
Windows CI cannot be executed locally in this environment.

Run during Day 8:

```sh
bash scripts/static_package_deferral_check.sh
rg -n "[ \t]+$" docs/planning/EPIC_14/SPRINT_162 .github/workflows/windows-ci.yml scripts/static_package_deferral_check.sh
git diff --check
git status --short -- '*.c' '*.h'
```

Expected result:

- static guard passes with Windows package non-claim wording and no unselected
  package execution checks;
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
static-package-deferral-check: Windows workflow has no unselected package execution ok
static-package-deferral-check: passed
```

## Day 8 Conclusion

Windows CI wording now matches the selected Sprint 162 package decision more
precisely. The workflow still proves CMake-first static package install and
downstream behavior on Windows, while explicitly treating `sparse.pc` as
metadata-only inspection and keeping Windows Makefile and `pkg-config`
execution parity as non-claims.
