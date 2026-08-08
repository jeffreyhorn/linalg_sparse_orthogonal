# Sprint 143 Day 11 Documentation Alignment

## Purpose

Align README, INSTALL, and maintainer documentation with the selected
static-first package implementation, while preserving shared-library, dynamic
ABI, runtime-loader, package-manager, platform-parity, and performance
non-claims.

## Changes Implemented

| Surface | Change | Reason |
| --- | --- | --- |
| `README.md` | Added a concise installation note that installed `sparse.pc` metadata is static-archive scoped and that install proof covers downstream compile/link/run plus exact package version handling. | Makes the public quick summary reflect the Day 7-10 static-first proof shape. |
| `INSTALL.md` | Added static `.pc` description/no-selector wording to the maintained install contract. | Documents the selected package metadata contract for users. |
| `INSTALL.md` | Clarified that `tests/test_install.sh` compiles and runs both a program and the maintained example via `pkg-config`. | Reflects the stricter Day 9 downstream proof. |
| `INSTALL.md` | Clarified that `tests/test_cmake_install.sh` includes exact-version configure/build/run proof. | Reflects the Day 9 CMake exact-version consumer proof. |
| `docs/maintainer_guide.md` | Updated proof-owner descriptions for static `.pc` metadata, stricter runtime output checks, no shared imported metadata, and exact-version configure/build/run behavior. | Keeps maintainer proof ownership accurate. |
| `docs/maintainer_guide.md` | Updated the Windows reviewed CTest count from 54 to 56. | Aligns support-tier docs with the current Windows workflow. |

## Preserved Non-Claims

- No shared-library packaging support.
- No dynamic ABI compatibility.
- No runtime-loader compatibility.
- No package-manager availability.
- No Windows Makefile or `pkg-config` parity.
- No macOS or Windows reviewed install/export parity from Sprint 143.
- No portable performance or state-of-the-art claim from package proof.

## Focused Validation

Focused checks run for this batch:

```sh
bash scripts/static_package_deferral_check.sh
python3 scripts/normalize_report_index.py --family package --check
python3 scripts/normalize_report_index.py --family package --check-freshness
rg -n "shared-library|dynamic ABI|runtime-loader|package-manager|platform parity|reviewed install/export parity|pkg-config parity" README.md INSTALL.md docs/maintainer_guide.md
git diff --check
rg -n "[ \t]+$" docs/planning/EPIC_12/SPRINT_143 README.md INSTALL.md docs/maintainer_guide.md
```

Results:

| Check | Result |
| --- | --- |
| `scripts/static_package_deferral_check.sh` | Passed |
| Package report index check | Passed: 6 rows |
| Package report freshness check | Passed: 6 source-controlled advisory rows |
| Focused package claim scan | Passed; matches are explicit non-claims, deferrals, or support-tier boundaries |

## Day 12 Input

Day 12 should run focused package validation across the touched package
surfaces and inspect generated-output hygiene. Documentation is now aligned
with the current static-first implementation and should not need more than
claim-boundary review unless validation uncovers a mismatch.

## Completion Criteria

| Criterion | Status | Evidence |
| --- | --- | --- |
| Docs match the selected package implementation. | Complete | README, INSTALL, and maintainer guide describe static `.pc` metadata, downstream proof, and exact-version behavior. |
| Users can distinguish supported package paths from deferrals. | Complete | Static-first support is described separately from shared-library, loader, package-manager, and platform deferrals. |
| No unsupported package claim was added. | Complete | Non-claim list and focused claim scan preserve unsupported boundaries. |
