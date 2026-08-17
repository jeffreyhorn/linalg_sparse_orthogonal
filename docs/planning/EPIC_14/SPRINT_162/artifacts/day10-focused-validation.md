# Sprint 162 Day 10 Focused Validation

## Scope

Day 10 validates the Sprint 162 retained non-claim implementation against the
changed-file surface. The branch has touched the static package guard, Windows
workflow wording, and Sprint 162 planning artifacts. No C source or public
header files were modified.

The validation target is therefore:

- static package guard behavior;
- Unix Make install and `pkg-config` package proof;
- CMake install/export/downstream package proof;
- documentation and diff hygiene;
- changed-file quality-gate decision.

## Validation Commands

| Command | Result | Purpose |
| --- | --- | --- |
| `bash scripts/static_package_deferral_check.sh` | Passed | Validates static-first package contract, retained Windows package non-claim wording, and no unselected Windows package execution in CI. |
| `bash tests/test_install.sh` | Passed | Validates Make install/uninstall, installed static archive, headers, `sparse.pc`, `pkg-config` flags/version, downstream `pkg-config` consumers, and no unsupported package/ABI wording. |
| `bash tests/test_cmake_install.sh` | Passed | Validates CMake install/export, static imported target metadata, downstream CMake example, exact-version behavior, mismatch rejection, installed `sparse.pc` metadata, and no unsupported shared-loader metadata. |
| `rg -n "[ \t]+$" docs/planning/EPIC_14/SPRINT_162 .github/workflows/windows-ci.yml scripts/static_package_deferral_check.sh` | Passed | Catches trailing whitespace in touched docs/scripts/workflow files. |
| `git diff --check` | Passed | Catches diff whitespace errors. |
| `git status --short -- '*.c' '*.h'` | Passed with no output | Confirms whether the full C quality gate is required. |

## Static Guard Output

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

## Make Install And pkg-config Output

```text
=== Install Validation Tests ===
  [PASS] static library installed
  [PASS] no shared-library artifacts installed
  [PASS] all 19 headers installed
  [PASS] pkg-config file installed
  [PASS] pkg-config can resolve sparse
  [PASS] pkg-config exact version constraint works
  [PASS] pkg-config prefix points at install prefix
  [PASS] pkg-config libdir points at installed libdir
  [PASS] pkg-config includedir points at installed includedir
  [PASS] pkg-config --cflags returns installed include path
  [PASS] pkg-config --libs returns installed static archive link flags
  [PASS] pkg-config --static libs match current self-contained link flags
  [PASS] pkg-config file has no private dependency stanza
  [PASS] pkg-config file describes static archive package metadata
  [PASS] pkg-config file has no unsupported packaging or ABI claims
  [PASS] pkg-config --modversion returns 2.2.0
  [PASS] basic pkg-config consumer compiles and links
  [PASS] basic pkg-config consumer runs correctly
  [PASS] maintained example source compiles with pkg-config
  [PASS] maintained example source runs with pkg-config install
  [PASS] library removed after uninstall
  [PASS] headers removed after uninstall
  [PASS] pkg-config file removed after uninstall
Passed: 23
Failed: 0
ALL INSTALL TESTS PASSED
```

## CMake Install/Export Output

```text
=== CMake Install Validation Tests ===
  [PASS] cmake configure
  [PASS] cmake build
  [PASS] cmake install
  [PASS] static library installed
  [PASS] no shared-library artifacts installed
  [PASS] headers installed (19 files)
  [PASS] SparseConfig.cmake installed
  [PASS] SparseConfigVersion.cmake installed
  [PASS] SparseTargets.cmake installed
  [PASS] sparse.pc installed
  [PASS] CMake imported target is static
  [PASS] CMake package has no shared-library imported metadata
  [PASS] CMake package has no unsupported loader or shared-selector metadata
  [PASS] CMake imported target uses install include prefix
  [PASS] CMake imported archive uses install prefix
  [PASS] CMake package has no source-tree paths
  [PASS] CMake package has no build-tree paths
  [PASS] pkg-config metadata describes static archive package
  [PASS] pkg-config metadata has no unsupported package or ABI claims
  [PASS] cmake_example configure (find_package works)
  [PASS] cmake_example build
  [PASS] cmake_example runs correctly
  [PASS] find_package exact installed version works
  [PASS] find_package exact-version consumer builds
  [PASS] find_package exact-version consumer runs correctly
  [PASS] find_package mismatched version is rejected
  [PASS] pkg-config version = 2.2.0
Passed: 27
Failed: 0
Skipped: 0
ALL CMAKE INSTALL TESTS PASSED
```

## Changed-File Quality Gate Decision

The Day 10 changed-file gate is documentation/script/workflow scoped. No
library `.c` or `.h` files were changed, so the full `make format && make lint
&& make test` C quality gate is not required by the Sprint 162 rules.

## Day 10 Conclusion

Focused local validation passed for the selected Sprint 162 package decision.
The retained non-claim guard passes, Unix Make/pkg-config package proof remains
healthy, and CMake install/export/downstream package proof remains healthy.
