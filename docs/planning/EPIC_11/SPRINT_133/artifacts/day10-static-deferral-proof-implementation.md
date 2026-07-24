# Sprint 133 Day 10 - Static Deferral Proof Implementation

## Purpose

Day 10 implements the selected static-first deferral proof from Day 9. The
proof verifies that shared-library packaging and dynamic ABI support remain
explicit deferrals under the maintained static archive package contract.

## Implemented Proof

| File | Change |
| --- | --- |
| `scripts/static_package_deferral_check.sh` | Added a local package-contract guard for static-first deferral proof. |
| `docs/maintainer_guide.md` | Documented the new script as local package-contract proof alongside install validation scripts. |

No C source, public headers, CMake package metadata, `pkg-config` metadata,
install scripts, workflows, or package-manager files changed on Day 10.

## Proof Checks

| Check | Implemented behavior |
| --- | --- |
| `BUILD_SHARED_LIBS=ON` rejection | Runs a temporary CMake configure with `-DBUILD_SHARED_LIBS=ON` and requires configure failure. |
| Deferral wording | Requires token-level wording for `BUILD_SHARED_LIBS=ON`, static archive package surface, shared-library packaging, and deferred dynamic ABI support. |
| Static target declaration | Requires `add_library(sparse_lu_ortho STATIC ...)` in `CMakeLists.txt`. |
| Export/import macro absence | Scans installed public headers for unsupported `SPARSE_API`, `SPARSE_EXPORT`, or `SPARSE_IMPORT`. |
| Shared ABI metadata absence | Scans `CMakeLists.txt` for unsupported shared ABI metadata such as `SOVERSION`, export-all-symbols, visibility presets, soname, or install-name policy. |
| Static/shared selector absence | Scans `cmake/SparseConfig.cmake.in` and `sparse.pc.in` for unsupported shared/static package selectors or `Libs.private`. |
| Support wording | Requires README, INSTALL, and maintainer guide wording to keep shared-library, dynamic ABI, and package-manager support deferred or bounded. |

## Proof Output

Successful run:

```text
static-package-deferral-check: BUILD_SHARED_LIBS rejection ok
static-package-deferral-check: static target declaration ok
static-package-deferral-check: no shared export/ABI metadata found ok
static-package-deferral-check: package metadata has no static/shared selector ok
static-package-deferral-check: support wording remains deferred ok
static-package-deferral-check: passed
```

The first run found a script-pattern issue: INSTALL already contained the
required wording, but the check did not account for Markdown backticks around
`BUILD_SHARED_LIBS=ON`. The pattern was fixed and the proof passed.

## Support Tier and Placement

| Proof | Placement | Support tier |
| --- | --- | --- |
| `scripts/static_package_deferral_check.sh` | Local script and maintainer guide reference. | Local maintainer/package contract guard. |
| `tests/test_cmake_install.sh` | Existing local install/export proof. | Local installed CMake consumer proof. |
| `tests/test_install.sh` | Existing local install proof. | Local Make install and `pkg-config` consumer proof. |

The new script is not a reviewed CI lane and does not claim shared-library,
dynamic ABI, package-manager, or platform install parity support.

## Validation

| Check | Result |
| --- | --- |
| `bash -n scripts/static_package_deferral_check.sh` | Pass |
| `bash scripts/static_package_deferral_check.sh` | Pass |
| Focused failure-path exercise | Pass; initial script wording-pattern failure was detected and fixed. |
| `git diff --check` | Pass |
| Sprint 133/package-doc whitespace scan | Pass |

No `.c` or `.h` files changed, so `make format && make lint && make test` is
not required for Day 10.

## Day 11 Handoff

Day 11 should strengthen the installed CMake consumer proof without changing
the selected static-first product contract:

- exact installed header-count assertion should match the current 19-header
  contract;
- installed CMake target path-origin checks should prove consumers use the
  staged prefix rather than build-tree paths;
- `tests/test_cmake_install.sh` remains the validation owner;
- static/shared support selectors should remain absent unless a future product
  decision changes the contract.

## Completion Criteria

| Criterion | Status | Evidence |
| --- | --- | --- |
| Selected proof fails clearly when the contract drifts. | Complete | Script exits non-zero with `static-package-deferral-check: FAIL: ...` for drift categories. |
| Proof placement matches reviewed or local-only status. | Complete | Maintainer guide and artifact classify the script as local package-contract proof. |
| Package support wording remains evidence-bounded. | Complete | Script checks support wording and preserves shared-library, dynamic ABI, and package-manager non-claims. |
