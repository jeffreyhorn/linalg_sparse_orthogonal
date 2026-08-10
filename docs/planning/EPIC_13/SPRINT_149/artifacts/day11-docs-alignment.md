# Sprint 149 Day 11: Documentation Alignment

## Purpose

Align public documentation, maintainer guidance, and report-family wording with
the Sprint 149 decision to promote a reviewed Windows CMake
install/downstream validation lane for the maintained static-first package
surface.

## Updated Documents

| File | Change |
| --- | --- |
| `README.md` | Updated the CI summary to name reviewed Windows CMake install/downstream validation while preserving Windows non-claims. |
| `INSTALL.md` | Updated maintained-install, supported-platform, and verification wording for the reviewed Windows CMake install/downstream lane. |
| `docs/maintainer_guide.md` | Updated package/platform ownership guidance to describe the Windows reviewed lane and its metadata/consumer/version checks. |
| `tests/corpus/manifests/report_families.tsv` | Updated the CI reviewed-lanes row to include Windows reviewed CMake install/downstream validation and preserve Windows Makefile/`pkg-config` execution non-claims. |
| `docs/planning/EPIC_13/SPRINT_149/WORKING_NOTES.md` | Recorded the Day 11 documentation alignment. |

## Windows Support Statement After Day 11

The reviewed Windows package lane is:

- hosted MSVC 2022 via `.github/workflows/windows-ci.yml`;
- CMake install/downstream scoped;
- static-first package surface only;
- installed static `.lib`;
- installed headers and `sparse_version.h`;
- installed CMake package metadata and `sparse.pc` metadata;
- generated and maintained installed CMake consumers;
- exact-version package behavior;
- mismatch-version fail-closed behavior;
- no installed DLLs or shared imported metadata.

## Preserved Non-Claims

Day 11 does not claim:

- Windows Makefile install parity;
- Windows Makefile uninstall parity;
- Windows `pkg-config` execution parity;
- Windows `pkg-config` downstream compile/link/run;
- package-manager support;
- shared-library support;
- dynamic ABI compatibility;
- runtime-loader behavior;
- broad Windows parity.

## Report-Family Update

The `ci/reviewed_lanes` row now describes reviewed hosted checks as including:

- Linux source-of-truth lanes;
- macOS reviewed static-first install/export proof;
- Windows reviewed CMake subset lanes;
- Windows reviewed CMake install/downstream validation.

The row's non-claims now explicitly include no Windows Makefile parity and no
Windows `pkg-config` execution parity. The row remains advisory and
source-controlled; hosted CI logs still live outside source control.

## Validation Plan

Day 12 should run:

1. workflow YAML parsing;
2. report-family/schema normalization checks affected by
   `tests/corpus/manifests/report_families.tsv`;
3. stale wording searches for old Windows supplemental install/downstream
   claims;
4. unsupported-claim searches for Windows `pkg-config`, Makefile,
   package-manager, shared-library, dynamic ABI, runtime-loader, and broad
   parity wording;
5. `git diff --check` and trailing-whitespace checks.

No `.c` or `.h` files were changed on Day 11.

## Completion Criteria Status

| Completion Criteria | Status | Evidence |
| --- | --- | --- |
| Public docs describe exactly the selected Windows install-validation tier. | Complete | README and INSTALL now state reviewed Windows CMake install/downstream validation for the static-first package surface. |
| Unsupported package-manager, shared-library, and `pkg-config` claims remain absent. | Complete | Preserved non-claims are explicit in README, INSTALL, maintainer guide, and report-family row. |
| Maintainers can identify which Windows lane is reviewed or supplemental. | Complete | Maintainer guide and workflow wording identify the Windows CMake install/downstream lane as reviewed and scoped. |
