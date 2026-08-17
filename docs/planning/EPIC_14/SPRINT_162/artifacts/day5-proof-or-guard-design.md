# Sprint 162 Day 5 Retained Guard Design

## Scope

Day 5 converts the Day 4 product decision into exact implementation targets.
The selected path is a retained non-claim guard design:

- preserve Windows CMake install/downstream validation;
- preserve Linux/macOS Make install and `pkg-config` execution validation;
- do not promote Windows Makefile install/uninstall parity;
- do not promote Windows `pkg-config` command execution parity;
- make the boundary explicit in scripts, workflows, docs, and evidence
  artifacts.

No package behavior change is required for Day 5. The implementation target is
guard clarity and failure diagnostics.

## Files And Surfaces To Change

| Surface | File | Change Type | Acceptance Criteria |
| --- | --- | --- | --- |
| Static package guard | `scripts/static_package_deferral_check.sh` | Extend retained non-claim checks. | Guard fails if supported docs or workflow wording claims Windows Makefile parity or Windows `pkg-config` execution parity. |
| Windows CI comments/assertions | `.github/workflows/windows-ci.yml` | Clarify metadata-only `sparse.pc` inspection. | Workflow still installs and checks `sparse.pc`, but wording states it is not a `pkg-config` execution proof. |
| Install docs | `INSTALL.md` | Normalize support-tier wording. | Windows support remains CMake-first; Makefile and `pkg-config` execution parity remain explicit non-claims. |
| README package summary | `README.md` | Normalize support-tier wording. | Reader can distinguish Windows CMake install proof from Linux/macOS Make/pkg-config proof. |
| Maintainer guide | `docs/maintainer_guide.md` | Add or preserve maintainer-facing guard rationale. | Maintainers see which package claims are reviewed, retained, or unsupported. |
| Sprint evidence | `docs/planning/EPIC_14/SPRINT_162/artifacts/*` and `WORKING_NOTES.md` | Keep decision trace. | Sprint artifacts explain why retained non-claim is the selected implementation path. |

## Files Not To Change For This Decision

| Surface | Reason |
| --- | --- |
| `sparse.pc.in` | Current metadata already describes static archive package metadata and is valid for Unix-like `pkg-config` proof. |
| `CMakeLists.txt` target/install shape | Current package surface already rejects shared builds, installs static archive metadata, and emits exact-version CMake package metadata. |
| `cmake/SparseConfig.cmake.in` | Current template exposes the installed target without static/shared selectors. |
| `tests/test_install.sh` | Linux/macOS Make and `pkg-config` proof should remain unchanged. |
| `tests/test_cmake_install.sh` | Existing CMake install/export proof should remain unchanged unless a future check needs stricter non-claim wording. |
| `.c` and `.h` files | Sprint 162 Day 5 does not touch library behavior. |

## Expected Installed Artifacts

The retained guard design preserves existing installed artifact expectations.

| Platform Lane | Expected Artifacts | Non-Claims |
| --- | --- | --- |
| Linux/macOS Make install | `lib/libsparse_lu_ortho.a`, installed headers, generated version header, `lib/pkgconfig/sparse.pc`. | No shared-library ABI, runtime-loader, package-manager, or static/shared selector claim. |
| Linux/macOS CMake install | Static archive, installed headers, generated version header, CMake package files, `sparse.pc`. | No shared imported target or dynamic-loader metadata. |
| Windows CMake install/downstream | `lib/sparse_lu_ortho.lib`, installed headers, generated version header, CMake package files, `lib/pkgconfig/sparse.pc`. | No DLL, shared imported target, runtime-loader behavior, package-manager support, Makefile parity, or `pkg-config` execution parity. |

## Package Metadata Assertions

Implementation should preserve or add checks for these assertions:

1. `sparse.pc` description is exactly static archive package metadata.
2. `sparse.pc` does not contain `Libs.private`.
3. `sparse.pc` does not mention shared libraries, ABI support, package-manager
   distribution, runtime loaders, or static/shared selectors.
4. CMake package metadata does not mention `SHARED IMPORTED`,
   `MODULE IMPORTED`, DLL, `.so`, `.dylib`, runtime loader behavior,
   `SOVERSION`, install-name, RPATH, or static/shared selectors.
5. CMake package metadata does not leak source-tree or build-tree paths.
6. Windows workflow wording treats `sparse.pc` as metadata inspection only.

## Downstream Consumer Behavior

The retained guard design keeps downstream consumer behavior divided by proof
owner:

| Consumer Path | Required Behavior |
| --- | --- |
| Windows CMake consumer | Configure, build, link, and run through `find_package(Sparse REQUIRED)` and `Sparse::sparse_lu_ortho`. |
| Windows exact-version CMake consumer | Configure, build, link, and run with `find_package(Sparse <version> EXACT REQUIRED)`. |
| Windows mismatched-version CMake consumer | Reject a lower same-major version at configure time. |
| Linux/macOS `pkg-config` consumer | Continue using `pkg-config --cflags --libs sparse` to compile, link, and run generated and maintained examples. |
| Windows `pkg-config` consumer | Not a reviewed consumer path in Sprint 162. No implementation should imply this proof exists. |
| Windows Make install/uninstall consumer | Not a reviewed consumer path in Sprint 162. No implementation should imply this proof exists. |

## Exact-Version Behavior

Exact-version behavior remains front-end specific:

- Windows CMake install/downstream proof must keep exact-version CMake
  consumer validation.
- Linux/macOS Make/pkg-config proof must keep `pkg-config --exists
  "sparse = $EXPECTED_VERSION"` and `pkg-config --modversion sparse`.
- Windows does not need a `pkg-config` exact-version proof because Windows
  `pkg-config` execution remains a non-claim.

## Failure Diagnostics

Implementation should fail with diagnostics that identify the violated
boundary:

| Failure | Diagnostic Requirement |
| --- | --- |
| Missing installed package file | Name the expected path and owning proof lane. |
| Unsupported shared artifact | Name the artifact and restate static archive package contract. |
| Stale or broad package metadata | Name the offending file and unsupported token. |
| Windows docs imply Makefile parity | Fail with a message that Windows Makefile install/uninstall parity is a retained non-claim. |
| Windows docs imply `pkg-config` execution parity | Fail with a message that installed `sparse.pc` metadata is not Windows `pkg-config` command proof. |
| Workflow comments lose CMake-first scope | Fail with a message requiring Windows CMake-first package wording. |
| Command absence is misused as support evidence | Fail with a message requiring a selected provider before promoting Windows `pkg-config` or Makefile execution. |

## Support-Tier Wording Map

| Surface | Required Wording |
| --- | --- |
| Windows CMake install/downstream | Reviewed CMake-first static archive package proof. |
| Windows `sparse.pc` | Installed static package metadata inspected by CI; not Windows `pkg-config` execution proof. |
| Windows `pkg-config` | Retained non-claim; no reviewed provider or downstream command proof. |
| Windows Makefile install/uninstall | Retained non-claim; no reviewed Windows Make install/uninstall proof. |
| Linux/macOS Make install | Reviewed install/uninstall package proof. |
| Linux/macOS `pkg-config` | Reviewed command execution and downstream consumer proof. |
| Shared library and dynamic ABI | Deferred and unsupported. |
| Package managers | Unsupported distribution claim. |

## Validation Command Map

| Stage | Command | Purpose |
| --- | --- | --- |
| Documentation whitespace | `rg -n "[ \t]+$" docs/planning/EPIC_14/SPRINT_162 README.md INSTALL.md docs/maintainer_guide.md .github/workflows/windows-ci.yml scripts/static_package_deferral_check.sh` | Catch trailing whitespace in touched docs/scripts/workflows. |
| Diff hygiene | `git diff --check` | Catch whitespace errors before commit. |
| Static guard | `bash scripts/static_package_deferral_check.sh` | Validate static-first and retained non-claim guard behavior. |
| Unix install proof | `bash tests/test_install.sh` | Preserve Linux/macOS Make and `pkg-config` proof where available. |
| CMake install proof | `bash tests/test_cmake_install.sh` | Preserve CMake package install/export behavior where available. |
| Full code gate | `make format && make lint && make test` | Required only if later implementation modifies `*.c` or `*.h`. |
| Hosted Windows proof | Windows CI `cmake-and-ctest` and `install-and-downstream` jobs | Confirm CMake-first Windows package proof remains intact. |

## Implementation Acceptance Criteria

Days 6-7 implementation is complete only when:

1. Windows CMake install/downstream validation still proves static `.lib`,
   headers, generated version header, CMake package files, `sparse.pc`
   metadata, exact-version CMake behavior, mismatch rejection, and downstream
   CMake consumers.
2. Guard checks fail if supported documentation claims Windows Makefile
   install/uninstall parity.
3. Guard checks fail if supported documentation claims Windows `pkg-config`
   command execution parity.
4. Guard checks fail if Windows workflow wording treats `sparse.pc` metadata
   inspection as `pkg-config` execution proof.
5. Linux/macOS Make and `pkg-config` proof remains unchanged.
6. Shared-library, dynamic ABI, runtime-loader, package-manager, and
   static/shared selector non-claims remain guarded.
7. The sprint evidence explains the retained non-claim decision without
   borrowing solver, benchmark, or comparison evidence.

## Day 5 Conclusion

The Day 5 design gives Sprint 162 a concrete implementation target: harden the
retained non-claim boundary around Windows package validation, while preserving
the existing reviewed CMake-first Windows proof and Unix Make/pkg-config proof.

The next implementation pass should start with `scripts/static_package_deferral_check.sh`,
then align `.github/workflows/windows-ci.yml`, `README.md`, `INSTALL.md`, and
`docs/maintainer_guide.md` with the same support-tier wording.
