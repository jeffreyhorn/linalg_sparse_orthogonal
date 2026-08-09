# Sprint 143 Day 13 Quality Gate And Claim Closure

## Purpose

Run the required quality gates for touched Sprint 143 surfaces and publish the
earned package/ABI claims, preserved non-claims, and future-owner handoff.

## Changed Surface Classification

| Surface | Changed files | Required gate |
| --- | --- | --- |
| C source/public headers | None | `make format && make lint && make test` not required |
| Build/package metadata | `CMakeLists.txt`, `sparse.pc.in` | Static deferral guard, Make install proof, CMake install/export proof |
| Shell package scripts | `tests/test_install.sh`, `tests/test_cmake_install.sh`, `scripts/static_package_deferral_check.sh` | Shell syntax, focused script execution |
| CI workflows | `.github/workflows/ci.yml`, `.github/workflows/macos-ci.yml`, `.github/workflows/windows-ci.yml` | YAML parse and claim-boundary scan |
| Public/maintainer docs | `README.md`, `INSTALL.md`, `docs/maintainer_guide.md` | Claim-boundary scan and static deferral guard |
| Planning artifacts | `docs/planning/EPIC_12/SPRINT_143/` | Diff and whitespace hygiene |

## Quality Gate Results

| Check | Result |
| --- | --- |
| C/header changed-file check | Passed: no `.c` or `.h` files changed |
| Shell syntax: `tests/test_install.sh`, `tests/test_cmake_install.sh`, `scripts/static_package_deferral_check.sh` | Passed |
| `bash scripts/static_package_deferral_check.sh` | Passed |
| `bash tests/test_install.sh` | Passed: 23 passed, 0 failed |
| `bash tests/test_cmake_install.sh` | Passed: 26 passed, 0 failed, 0 skipped |
| `python3 scripts/normalize_report_index.py --family package --check` | Passed: 6 rows |
| `python3 scripts/normalize_report_index.py --family package --check-freshness` | Passed: 6 source-controlled advisory rows |
| Workflow YAML parse | Passed for Linux, macOS, and Windows workflows |
| Package/docs/workflow claim-boundary scan | Passed; matches are explicit non-claims, support-tier boundaries, or unrelated bounded algorithm notes |
| `git diff --check` | Passed |
| Trailing-whitespace scan | Passed |

## Earned Package Claims

Sprint 143 now earns these static-first package claims:

1. The maintained package surface is static-first.
2. Make install installs a static archive, 19 public headers, and static
   archive `pkg-config` metadata.
3. CMake install/export provides a static imported target,
   `Sparse::sparse_lu_ortho`, with install-prefix include/archive metadata.
4. Installed `sparse.pc` metadata describes static archive package metadata,
   has no `Libs.private` stanza, and contains no unsupported package/ABI
   wording.
5. Unsupported shared-library artifacts are checked as absent from Make and
   CMake installs.
6. `BUILD_SHARED_LIBS=ON` is intentionally rejected.
7. Make/`pkg-config` downstream consumers compile, link, and run against the
   installed static archive.
8. CMake `find_package(Sparse)` downstream consumers compile, link, and run
   against the installed static archive.
9. Exact package version behavior is checked through both `pkg-config` and
   CMake; the CMake exact-version consumer configures, builds, and runs.
10. Linux carries the reviewed static-first package-contract CI lane.

## Preserved Non-Claims

Sprint 143 does not claim:

- shared-library build/install/export support;
- dynamic ABI compatibility;
- runtime-loader compatibility;
- package-manager availability;
- static/shared CMake or `pkg-config` selector support;
- Windows Makefile parity;
- Windows `pkg-config` parity;
- Windows reviewed install-validation parity;
- macOS reviewed install/export parity;
- portable performance from package proof;
- state-of-the-art status from package/ABI work.

## Future-Owner Handoff

| Residual | Future owner | Gate before promotion |
| --- | --- | --- |
| Shared-library build/install/export | Package/ABI owner | Shared target design, export/import policy, symbol allowlist, install/export metadata, downstream shared consumers |
| Dynamic ABI compatibility | Package/ABI owner | ABI epoch/version policy, public layout policy, compatibility tests, docs |
| Runtime-loader behavior | Package/platform owner | Linux RPATH/RUNPATH, macOS install-name, Windows DLL search-path/import-library proof |
| Package-manager distribution | Package/adoption owner | Manager-specific recipes, install roots, upgrade/uninstall proof, downstream tests |
| macOS reviewed install/export parity | Sprint 144 platform owner | Hosted macOS promotion decision and repeated evidence |
| Windows reviewed install-validation parity | Sprint 144 platform owner | Hosted Windows promotion decision, exact static-first scope, failure ownership |

## Day 14 Input

Day 14 should review all Sprint 143 artifacts for consistency with this claim
closure, run final hygiene, and publish the closeout summary plus Sprint 144
platform-promotion handoff.

## Completion Criteria

| Criterion | Status | Evidence |
| --- | --- | --- |
| Required checks for touched surfaces pass. | Complete | Quality gate table records all touched-surface checks as passed. |
| Earned package claims are backed by evidence. | Complete | Earned claims map directly to install, CMake, guard, report, workflow, and docs checks. |
| Unearned package/ABI/platform/distribution claims remain explicit. | Complete | Preserved non-claims and future-owner handoff route deferred work forward. |
