# Sprint 171 Day 11: User Documentation Update

## Purpose

Day 11 implements the documentation updates designed on Day 10. The selected
Sprint 171 package-manager path remains formal deferral, so the updates route
users to source install via Make or CMake and keep provider package-manager
support as an explicit non-claim.

## Documentation Changes

| Document | Change | Claim Boundary |
| --- | --- | --- |
| `README.md` | Added a short installation-section sentence stating that package-manager support is not currently provided and users should use source install via Make or CMake. | Prevents the short install summary from being read as provider package-manager support. |
| `INSTALL.md` | Added a package-manager deferral entry to Support Split naming unsupported provider families and routing users to source install. | Separates source install, CMake/`pkg-config` package metadata, and provider package-manager support. |
| `INSTALL.md` | Clarified that `scripts/package_manager_deferral_check.sh` protects provider non-claims and does not prove provider install behavior. | Keeps normalized package proof-owner rows from becoming provider support claims. |
| `docs/maintainer_guide.md` | Added a maintainer rule to run the package-manager deferral guard when changing package-manager wording, provider recipes, package metadata templates, or provider claims. | Gives maintainers a direct validation command for package-manager claim changes. |

## Documents Reviewed Without Additional Edits

| Document | Result |
| --- | --- |
| `docs/tutorial.md` | Already delegates static-first package details to `INSTALL.md` and avoids provider package-manager claims. |
| `docs/api_reference.md` | Already states the API reference does not imply package-manager distribution. |
| `docs/cookbook.md` | Remains workflow-focused; no new provider package-manager support wording was needed. |

## Targeted Claim Scans

Day 11 uses targeted scans over current user-facing documentation:

```sh
rg -n "package-manager|package manager|vcpkg|Homebrew|Conan|pkgsrc|apt|dnf|pacman|binary package|registry|tap" \
  README.md INSTALL.md docs/maintainer_guide.md docs/tutorial.md docs/api_reference.md docs/cookbook.md
rg -n "shared-library|dynamic ABI|runtime-loader|BUILD_SHARED_LIBS|static/shared selector|Windows Makefile|Windows.*pkg-config" \
  README.md INSTALL.md docs/maintainer_guide.md docs/tutorial.md docs/api_reference.md docs/cookbook.md
```

Allowed matches must preserve one of these meanings:

- package-manager provider support is unsupported or formally deferred;
- Make/CMake source install remains the maintained path;
- CMake/`pkg-config` metadata describes the static archive package surface;
- shared-library packaging, dynamic ABI compatibility, runtime-loader
  behavior, static/shared selectors, Windows Makefile parity, and Windows
  `pkg-config` execution parity remain non-claims.

## Focused Validation

Day 11 validation commands:

```sh
bash scripts/package_manager_deferral_check.sh
bash scripts/static_package_deferral_check.sh
python3 scripts/normalize_report_index.py --family package --check
python3 scripts/normalize_report_index.py --family package --check-freshness
git diff --check
```

## Day 11 Deliverables

| Deliverable | Status | Notes |
| --- | --- | --- |
| README package-manager wording | Complete | README now states package-manager support is not currently provided. |
| INSTALL package-manager guidance | Complete | Support Split now includes explicit package-manager deferral and unsupported provider families. |
| Maintainer-guide ownership update | Complete | Maintainers now have a direct guard command for package-manager wording and provider-claim changes. |
| Claim-scan results | Complete | Targeted scan commands and allowed interpretations are recorded above. |
| Day 11 documentation-update artifact | Complete | This file. |

## Validation

Day 11 changed Markdown and planning artifacts only. No `.c` or `.h` files
were modified, so the full C quality gate is not required for this day.

Validation command:

```sh
bash scripts/package_manager_deferral_check.sh
bash scripts/static_package_deferral_check.sh
python3 scripts/normalize_report_index.py --family package --check
python3 scripts/normalize_report_index.py --family package --check-freshness
git diff --check
```

## Completion Criteria

| Criterion | Status | Notes |
| --- | --- | --- |
| Docs match the selected provider or deferral decision. | Complete | Docs now explicitly state package-manager support is not currently provided. |
| Unsupported package-manager claims remain explicit. | Complete | INSTALL names unsupported provider families and routes users to source install. |
| Source install, CMake/`pkg-config`, and package-manager support remain separate. | Complete | README and INSTALL distinguish static package metadata from package-manager distribution. |
