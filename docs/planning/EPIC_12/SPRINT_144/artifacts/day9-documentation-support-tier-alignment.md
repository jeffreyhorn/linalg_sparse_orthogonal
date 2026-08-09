# Sprint 144 Day 9 Documentation Support-Tier Alignment

## Purpose

Align README, INSTALL, maintainer guidance, and platform support wording with
the selected Sprint 144 macOS reviewed static-first install/export lane.

## Documentation Updates

| File | Update | Reason |
| --- | --- | --- |
| `README.md` | Updated the CI summary to say macOS now has reviewed static-first Make install/`pkg-config` and reviewed CMake install/export proof. | Public project summary now matches the promoted macOS CI lane. |
| `INSTALL.md` | Updated the validation story and supported-platform table for reviewed macOS static-first install/export proof. | User-facing install guidance now matches workflow support tiers. |
| `INSTALL.md` | Replaced stale macOS supplemental package non-claim wording with a narrower reviewed static-first non-claim boundary. | macOS proof is now reviewed, but still does not imply shared libraries, ABI, loader, package-manager, selectors, or broad platform parity. |
| `docs/maintainer_guide.md` | Updated current package/platform support-tier guidance. | Maintainer interpretation now matches workflow and report semantics. |
| `docs/maintainer_guide.md` | Updated historical package snapshot language to say Sprint 144 promoted macOS static-first package proof. | Historical context remains understandable without preserving stale current-state wording. |

## Current Support-Tier Interpretation

| Platform lane | Current Day 9 status |
| --- | --- |
| Linux | Strongest reviewed source of truth, including reviewed Makefile compile-quality, reviewed CMake parity, dead-code, and reviewed static-first package contract. |
| macOS Apple Clang | Reviewed path for Makefile compile-quality, CMake parity, wall-check, and sanitizer. |
| macOS static-first package proof | Reviewed static-first Make install/`pkg-config` and CMake install/export proof. |
| macOS Homebrew GCC | Supplemental second-compiler coverage. |
| Windows MSVC CMake subset | Reviewed CMake subset with staged exclusions still explicit. |
| Windows CMake install/downstream | Supplemental CMake-first confidence only. |

## Preserved Non-Claims

The docs continue to avoid claiming:

- shared-library packaging;
- dynamic ABI compatibility;
- runtime-loader compatibility;
- package-manager support;
- static/shared selectors;
- Windows Makefile parity;
- Windows `pkg-config` parity;
- Windows reviewed install-validation parity;
- broader macOS platform parity beyond the reviewed static-first
  install/export lane;
- state-of-the-art status from platform or package proof alone.

## Evidence References

| Evidence | Reference |
| --- | --- |
| macOS reviewed static-first workflow jobs | `.github/workflows/macos-ci.yml` |
| CI report-family source-controlled row | `tests/corpus/manifests/report_families.tsv` |
| Make install/`pkg-config` proof owner | `tests/test_install.sh` |
| CMake install/export proof owner | `tests/test_cmake_install.sh` |
| Static package deferral proof owner | `scripts/static_package_deferral_check.sh` |
| Day 7 CI implementation evidence | `docs/planning/EPIC_12/SPRINT_144/artifacts/day7-ci-promotion-implementation.md` |
| Day 8 report integration evidence | `docs/planning/EPIC_12/SPRINT_144/artifacts/day8-package-report-integration.md` |

## Stale Wording Cleanup

Day 9 removed current-state wording that said:

- macOS package jobs were supplemental install/export confidence;
- macOS supplemental package confidence did not claim reviewed install/export
  parity;
- macOS package lanes did not become reviewed install/export parity.

The replacement wording says macOS has reviewed static-first install/export
proof while preserving unsupported-claim boundaries.

## Validation

| Check | Result |
| --- | --- |
| Stale macOS supplemental install/export wording scan | Passed |
| Reviewed macOS/static-first and non-claim boundary scan | Passed |
| `git diff --check` | Passed |

## Day 10 Handoff

Day 10 should run a focused selected-lane validation pass:

1. Re-run package/report checks affected by Days 8-9.
2. Re-run workflow YAML and support-tier scans.
3. Re-run local package proof if time permits or if docs/workflow changes need
   a fresh command record.
4. Confirm no README, INSTALL, or maintainer wording implies package-manager,
   shared-library, dynamic ABI, runtime-loader, Windows parity, or broad macOS
   platform claims.

## Day 9 Completion Criteria

| Criterion | Status | Evidence |
| --- | --- | --- |
| Public docs match the evidence earned this sprint. | Complete | README and INSTALL now name macOS reviewed static-first install/export proof. |
| Remaining platform limitations are concrete and discoverable. | Complete | INSTALL and maintainer guide preserve unsupported package/platform boundaries and Windows staged limits. |
| Docs do not imply package-manager, shared-library, or platform parity claims beyond proof. | Complete | Non-claim wording explicitly excludes unsupported package and broader platform claims. |
