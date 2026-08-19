# Sprint 170 Day 12: Documentation Alignment

## Purpose

Day 12 aligns the public and maintainer documentation with the accepted
Sprint 170 shared-library ABI product decision. The maintained package claim
remains static-first-only; shared-library packaging, dynamic ABI
compatibility, runtime-loader behavior, package-manager distribution, Windows
Makefile parity, Windows `pkg-config` execution parity, broad platform parity,
and state-of-the-art package/ABI status remain non-claims.

## Documentation Updates

| File | Update |
| --- | --- |
| `README.md` | Added a direct reference to the canonical Sprint 170 package and ABI product decision next to the shared-library deferral wording. |
| `INSTALL.md` | Added the same decision-record reference in the maintained install contract section. |
| `docs/maintainer_guide.md` | Added the decision record to the authoritative packaging contract and updated static deferral guard ownership to include the Day 11 checks. |
| `scripts/static_package_deferral_check.sh` | Added public-documentation citation checks so README, INSTALL, and the maintainer guide must keep the canonical Sprint 170 decision visible. |

## Claim Alignment

Supported package claims remain limited to:

- static archive build and install;
- Unix Make install/uninstall plus `pkg-config` proof;
- Unix CMake install/export plus `find_package(Sparse)` proof;
- Linux and macOS static-first package CI lanes;
- Windows CMake install/downstream validation for the static-first package
  surface;
- generated version metadata and exact CMake package-version compatibility.

Unsupported claims remain explicit:

- shared-library builds or installs;
- dynamic ABI compatibility;
- stable dynamic exported symbols;
- runtime-loader behavior;
- Linux SONAME support;
- macOS install-name/RPATH support;
- Windows DLL/import-library support;
- static/shared package selectors;
- package-manager distribution;
- Windows Makefile install parity;
- Windows `pkg-config` command execution parity;
- broad platform parity;
- state-of-the-art status from package/install/ABI evidence.

## Targeted Claim Scan

The documentation scan stayed targeted to the public and maintainer claim
surfaces:

```sh
rg -n "Shared-library packaging|BUILD_SHARED_LIBS|dynamic ABI|static-first|Sprint 170|pkg-config execution parity|package-manager|state-of-the-art" README.md INSTALL.md docs/maintainer_guide.md
```

The scan showed the expected static-first claims and non-claim wording. It did
not identify a documentation change that expands the selected package or ABI
support surface.

## Validation

Focused validation was run after the documentation and guard updates:

```sh
bash -n scripts/static_package_deferral_check.sh
bash scripts/static_package_deferral_check.sh
git diff --check
```

No `.c` or `.h` files were modified, so the full C quality gate
`make format && make lint && make test` was not required for Day 12.

## Day 12 Deliverables

| Deliverable | Status | Notes |
| --- | --- | --- |
| README package/ABI wording updates | Complete | README now cites the canonical Sprint 170 decision record. |
| Maintainer-guide wording updates | Complete | The authoritative packaging contract and guard ownership were updated. |
| Non-claim table updates | Complete | Existing non-claims remained in place; no new support claim was added. |
| Claim-scan results | Complete | Targeted claim scan confirmed expected static-first and non-claim wording. |
| Day 12 documentation-alignment artifact | Complete | This file. |

## Completion Criteria

| Criterion | Status | Notes |
| --- | --- | --- |
| Docs match the selected decision. | Complete | Public and maintainer docs now cite the canonical Day 9 decision. |
| Unsupported claims remain explicit. | Complete | Shared-library, ABI, loader, package-manager, Windows parity, broad platform, and state-of-the-art non-claims remain documented. |
| Performance evidence is not cited as package/ABI evidence. | Complete | Day 12 did not add performance evidence to package or ABI claims. |
