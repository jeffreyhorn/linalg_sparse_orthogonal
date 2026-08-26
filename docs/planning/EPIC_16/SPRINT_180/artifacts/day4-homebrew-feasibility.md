# Sprint 180 Day 4: Homebrew Feasibility Audit

**Sprint:** 180 - Package-Manager Provider Decision
**Epic source:** `docs/planning/EPIC_16/PROJECT_PLAN.md`
**Sprint path:** `docs/planning/EPIC_16/SPRINT_180/`
**Status:** Complete

## Purpose

Evaluate Homebrew as a static-first package-manager provider proof candidate
before the Day 7 decision matrix. Day 4 compares current Homebrew formula and
tap expectations against the project package surface, records prototype
blockers, defines proof requirements, and keeps Homebrew support unclaimed.

## External Homebrew References

Official Homebrew documentation consulted on 2026-08-25:

| Topic | Source |
| --- | --- |
| Formula authoring | <https://docs.brew.sh/Formula-Cookbook> |
| Adding software | <https://docs.brew.sh/Adding-Software-to-Homebrew> |
| Bottles | <https://docs.brew.sh/Bottles> |
| Tap creation | <https://docs.brew.sh/How-to-Create-and-Maintain-a-Tap> |
| FAQ bottle behavior | <https://docs.brew.sh/FAQ> |
| Homebrew support tiers | <https://docs.brew.sh/Support-Tiers> |

Relevant current expectations from those sources:

- formulas need completed metadata, including homepage and license;
- Homebrew/core expects a DFSG-compatible or public-domain license expressed
  with SPDX-style metadata;
- formulae should use canonical homepage and immutable release source, verify
  versioned downloads with SHA-256, declare only required dependencies, and
  include a meaningful test;
- normal formula validation includes build-from-source install, `brew test`,
  strict online audit, and formula style checks;
- bottles are separate binary packages created and published after formula
  build/test review;
- taps are external formula sources and can be local or Git-backed, with
  formulae commonly stored under a `Formula` directory;
- Homebrew support tiers describe host-system support and do not guarantee any
  third-party formula remains runnable everywhere.

## Local Baseline

| Check | Day 4 result |
| --- | --- |
| `brew` on PATH | Present at `/usr/local/bin/brew`. |
| Homebrew version | `Homebrew 6.0.19`. |
| Host | `Darwin 24.6.0 x86_64`. |
| Existing Homebrew artifacts | No active `Formula/` provider tree or formula Ruby file outside planning. |
| Version source | `VERSION` currently reports `2.2.0`. |
| Build system fit | CMake and Make both install the maintained static archive package surface; macOS CI already runs static-first Make install/`pkg-config` and CMake install/export proof. |
| License metadata | No standalone `LICENSE`, `COPYING`, or `NOTICE` file was found. README only states research and educational purpose. |
| Current guard posture | `scripts/package_manager_deferral_check.sh` intentionally fails if `Formula/` provider artifacts appear before the provider decision and guard update. |

## Homebrew Fit Assessment

| Criterion | Assessment |
| --- | --- |
| Static-first fit | Strong for a local formula. The project already has CMake install/export and Make install paths that install a static archive and reject shared-library support. A formula can build with CMake and keep optional OpenMP/mutex behavior off by default. |
| CI feasibility | Medium to high for macOS local/tap proof because `brew` is available locally and macOS CI already exists. CI cost rises if the proof bootstraps a tap, runs `brew audit --strict --new --online`, or attempts bottle behavior. |
| Formula complexity | Medium. A local formula likely needs `desc`, `homepage`, `url`, `sha256`, `license`, `depends_on "cmake" => :build`, an `install` block using CMake, and a `test do` block that compiles or configures a downstream consumer. |
| User value | High for macOS users. Homebrew is a natural macOS adoption path, but its value is narrower than vcpkg for Windows and cross-platform C/C++ consumers. |
| Proof completeness | Medium. Existing macOS install checks prove the underlying static package surface, but no `brew install`, formula test, audit, version query, uninstall cleanup, tap proof, or bottle proof exists yet. |
| Maintenance cost | Medium for local tap proof; high for Homebrew/core or bottle-backed support because release URL, SHA-256, license, audit, bottle, and upstream review expectations must remain current. |
| Claim risk | High unless bounded. A local formula or tap can be misread as Homebrew/core acceptance, bottle availability, Linuxbrew support, broad macOS parity, shared-library support, dynamic ABI support, or provider-managed upgrades. |

## Static-Library, Platform, Bottle, And Version Assessment

| Area | Homebrew implication |
| --- | --- |
| Static library | Feasible through existing CMake or Make install paths, with formula proof checking installed `.a` archive and absence of `.dylib` artifacts. |
| Shared-library requests | Must remain unsupported. Formula proof must not set `BUILD_SHARED_LIBS=ON` or imply dynamic ABI, install-name/RPATH, or runtime-loader behavior. |
| Platform scope | Strongest initial scope is macOS local formula/tap proof. Linuxbrew should remain out of scope unless separately proven. |
| Bottle expectations | Bottle support is not earned by formula source install. Bottle proof would require build-bottle/bottle behavior and published bottle metadata. |
| Test block feasibility | Feasible. A formula `test do` can compile and run a small installed consumer or configure a CMake consumer with `find_package(Sparse)`. |
| Versioning | Formula version can come from `VERSION=2.2.0`, but formula support needs an immutable release URL and SHA-256 policy before reproducible package claims. |
| Dependencies | Baseline formula likely needs CMake as a build dependency and system compiler/math library. Optional OpenMP/mutex behavior should stay off in first proof. |
| License metadata | Blocked for Homebrew/core readiness until provider-compatible license metadata and a standalone license/copyright source exist. |

## Prototype Shape

A local Homebrew prototype would likely include:

| File or command | Role |
| --- | --- |
| `Formula/sparse.rb` or equivalent local tap formula path | Homebrew formula metadata and install/test logic. |
| `url` and `sha256` | Immutable source archive and checksum. For local proof, a temporary local archive may be acceptable only if docs call it local proof, not distribution readiness. |
| `depends_on "cmake" => :build` | Build dependency for CMake install/export path. |
| `install` block | Configure/build/install static package with CMake and default options. |
| `test do` block | Compile/run a downstream installed consumer or configure/build/run a `find_package(Sparse)` consumer. |
| proof script | Runs formula syntax/audit checks where feasible, installs from source into an isolated prefix or tap context, runs `brew test`, checks version and installed files, and uninstalls/cleans generated state. |

Day 4 does not add these files because Sprint 180 has not selected Homebrew as
the product decision and the current deferral guard intentionally rejects
formula artifacts.

## Required Proof Before Homebrew Support

| Evidence | Requirement |
| --- | --- |
| Tool availability | Script handles missing `brew` clearly without implying support. |
| Formula source | Source-controlled formula or tap prototype exists only after product decision and guard update. |
| Source/checksum | Formula has immutable source input and SHA-256 policy, or is explicitly marked local-only proof. |
| License | Formula metadata accurately identifies provider-compatible license status. |
| Static-only install | Installed formula contains static archive and no `.dylib`, shared-library, install-name, RPATH, runtime-loader, or static/shared selector support claim. |
| Test block | `brew test` builds and runs a downstream consumer or validates `find_package(Sparse)` behavior. |
| Version behavior | Formula version, installed package version, and `VERSION` match; version query behavior is recorded. |
| Cleanup | `brew uninstall`, temporary tap/formula files, caches, build outputs, and test artifacts are cleaned or explicitly ignored. |
| Audit | `brew audit` and style checks are run or unavailable-tool failures are recorded claim-safely. |
| Docs and guard | README, INSTALL, maintainer guide, and package-manager guard distinguish local formula/tap proof from Homebrew/core, bottle, and broad provider support. |

## Blockers

| Blocker | Impact |
| --- | --- |
| No selected provider decision yet | Adding formula files today would violate current deferral guard and Sprint 180 sequencing. |
| No formula/tap artifact | No Homebrew install or test proof can exist until a formula is designed and added after decision. |
| No standalone license file | Homebrew/core readiness is blocked by missing provider-compatible license metadata and installable copyright source. |
| No immutable release URL plus SHA-256 policy | A formula cannot make distribution-ready claims without reproducible source input. |
| No formula `test do` proof | Existing package tests prove source install, not Homebrew formula behavior. |
| No bottle proof | There is no `brew bottle`, bottle DSL, published bottle, or bottle CI evidence. |
| No guard update | Current guard correctly rejects formula artifacts until the product decision changes. |

## Claim Risks

- A local formula can be misread as Homebrew/core acceptance.
- A tap formula can be misread as official Homebrew support or bottle
  availability.
- A successful formula source build can be misread as prebuilt binary package
  support.
- macOS formula proof can be misread as Linuxbrew support or broad
  cross-platform package-manager support.
- Homebrew dependency resolution can be misread as provider-managed upgrade
  compatibility for the library.
- Formula install convenience can be misread as shared-library support,
  dynamic ABI compatibility, or runtime-loader behavior.

## Day 4 Decision

Homebrew remains eligible for the Sprint 180 decision matrix as a local
formula or local tap proof candidate. It should not be described as
Homebrew/core-ready, bottle-backed, Linuxbrew-supported, or broadly
package-manager-supported.

If Homebrew is selected on Day 8, the first implementation path should be a
source-controlled local formula/tap prototype plus proof script, not
Homebrew/core submission or bottle publication.

## Day 4 Deliverables

- Homebrew fit assessment
- Homebrew formula complexity notes
- Homebrew local and CI proof requirements
- Homebrew claim-risk notes
- `docs/planning/EPIC_16/SPRINT_180/artifacts/day4-homebrew-feasibility.md`

## Validation

Day 4 changed planning artifacts only. No `.c`, `.h`, package metadata,
workflow, guard, provider formula, or public user-facing docs were modified.

Validation commands:

```sh
bash scripts/package_manager_deferral_check.sh
bash scripts/static_package_deferral_check.sh
git diff --check
```

## Completion Criteria

| Criterion | Status | Evidence |
| --- | --- | --- |
| Homebrew is evaluated against the shared provider criteria. | Complete | Fit assessment and static-library/platform/bottle/version sections above. |
| macOS-specific and cross-platform claim boundaries are explicit. | Complete | Platform, bottle, claim-risk, and Day 4 decision sections above. |
| Homebrew remains eligible or is rejected with evidence. | Complete | Day 4 decision keeps Homebrew eligible for local formula/tap proof only. |
