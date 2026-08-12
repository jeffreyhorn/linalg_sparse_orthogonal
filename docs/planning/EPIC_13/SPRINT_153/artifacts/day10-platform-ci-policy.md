# Sprint 153 Day 10 Platform CI Policy

## Purpose

Day 10 audits hosted CI follow-through for the selected static-first
package/ABI decision. The goal is to decide where the Day 7-9 proof should run
and preserve platform-specific claim boundaries before Day 11 documentation
and workflow alignment.

## Policy Decision

Keep the existing hosted CI lane structure. Add only one Day 11 follow-through
candidate: mirror the new unsupported-loader CMake metadata scan in the
Windows inline CMake install/downstream validation script.

Rationale:

- Linux already runs `tests/test_install.sh`, `tests/test_cmake_install.sh`,
  and `scripts/static_package_deferral_check.sh` in the reviewed static-first
  package-contract lane.
- macOS already runs `tests/test_install.sh`, `tests/test_cmake_install.sh`,
  and `scripts/static_package_deferral_check.sh` across reviewed static-first
  install/export jobs.
- Windows cannot run the Unix shell install scripts as its reviewed proof path;
  it owns an inline PowerShell CMake install/downstream validation. That script
  already checks DLL absence and no shared imported metadata, but should mirror
  Day 9's unsupported loader/static-shared selector scan.

## Local Versus Hosted Proof Matrix

| Proof | Local Command | Linux CI | macOS CI | Windows CI |
| --- | --- | --- | --- | --- |
| Shared request rejection | `bash scripts/static_package_deferral_check.sh` | Reviewed package-contract lane | Reviewed CMake install/export lane | Not currently mirrored; Windows has explicit non-claim wording. |
| Make install and `pkg-config` static consumer proof | `bash tests/test_install.sh` | Reviewed package-contract lane | Reviewed install/pkg-config lane | Explicitly not claimed. |
| CMake install/export static consumer proof | `bash tests/test_cmake_install.sh` | Reviewed package-contract lane | Reviewed CMake install/export lane | Reviewed inline PowerShell install/downstream lane. |
| Unsupported shared artifacts | Install scripts reject `.so`, `.so.*`, `.dylib`, `.dll` | Reviewed package-contract lane | Reviewed install/export lanes | Inline PowerShell rejects `.dll`. |
| Unsupported loader metadata | `tests/test_cmake_install.sh` checks installed package files | Reviewed package-contract lane | Reviewed CMake install/export lane | Day 11 should mirror this check inline. |
| Dynamic loader execution | None | Not claimed | Not claimed | Not claimed. |

## Workflow Scope Review

### Linux

The Linux `package-contract` job is already sufficient for Sprint 153:

- runs Make install and `pkg-config` package proof;
- runs CMake install/export package proof;
- runs static-first package deferral proof;
- comments explicitly limit the lane to static archive install/export and
  downstream proof.

No Linux workflow change is selected for Day 11.

### macOS

The macOS install jobs are already sufficient for Sprint 153:

- `install-and-pkgconfig` runs the Unix Make install/`pkg-config` proof;
- `cmake-install-export` runs the CMake install/export proof and deferral
  guard;
- comments explicitly preserve no shared-library, dynamic ABI,
  runtime-loader, package-manager, static/shared selector, or broad macOS
  parity claims.

No macOS workflow change is selected for Day 11.

### Windows

The Windows inline install/downstream validation remains CMake-first and
static-first. It already checks:

- installed static `.lib`;
- no installed `.dll`;
- `19` installed headers and generated version header;
- installed CMake package files and `sparse.pc`;
- `STATIC IMPORTED` target metadata;
- installed-prefix include and static `.lib` location;
- no source/build path leaks;
- no shared imported metadata;
- generated and maintained installed CMake consumers;
- exact-version behavior and mismatched-version rejection.

Day 11 should add a matching unsupported-loader metadata scan over the
installed CMake package text for:

- `SOVERSION`;
- `IMPORTED_SONAME`;
- `INSTALL_NAME`;
- `MACOSX_RPATH`;
- `IMPORTED_IMPLIB`;
- standalone `RUNTIME`;
- package component selectors mentioning `static` or `shared`;
- `Sparse::*shared` target naming.

This does not add Windows Makefile parity, Windows `pkg-config` execution
parity, DLL support, dynamic ABI support, or runtime-loader support.

## Expected Counts And Workflow Behavior

| Surface | Expected Behavior |
| --- | --- |
| Windows CTest count | Keep `EXPECTED_WINDOWS_CTEST_COUNT: "59"` unless CMake test registration changes. Sprint 153 Day 10 selects no CTest changes. |
| Install proof pass counts | Unix local CMake install proof now reports `27` checks after Day 9. Hosted Linux/macOS should inherit that count through the same script. |
| Windows install proof count | No explicit pass count is emitted; the inline PowerShell job should continue throwing on first failure. |
| Artifact upload | No new artifact upload or retention policy is needed because Sprint 153 adds metadata assertions, not generated reports or runtime loader logs. |
| Unsupported artifacts | Shared artifacts remain hard failures in install proofs. |

## Platform Non-Claim Register

These non-claims remain active after Day 10:

- shared-library packaging;
- dynamic ABI compatibility;
- Linux `.so` support;
- macOS `.dylib` support;
- Windows `.dll` or import-library support;
- SONAME, install-name, RPATH, or runtime-loader behavior;
- static/shared package selectors;
- package-manager distribution;
- Windows Makefile parity;
- Windows `pkg-config` execution parity;
- broad platform parity beyond the reviewed lane wording.

## Day 11 Implementation Checklist

1. Update `.github/workflows/windows-ci.yml` inline install/downstream proof to
   scan installed CMake package metadata for unsupported loader/static-shared
   selector tokens.
2. Keep Linux and macOS workflow commands unchanged.
3. Update workflow comments only if needed to mention unsupported-loader
   metadata proof without claiming runtime-loader support.
4. Search README, INSTALL, maintainer guide, workflows, `sparse.pc.in`,
   `cmake/SparseConfig.cmake.in`, and install tests for stale shared-library,
   ABI, loader, selector, or package-manager wording.
5. Run focused validation:
   - `bash scripts/static_package_deferral_check.sh`;
   - `bash tests/test_install.sh`;
   - `bash tests/test_cmake_install.sh`;
   - `git diff --check`.
6. Run full C quality gates only if C or public header files change.
