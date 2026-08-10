# Sprint 147 Day 7 Windows Evidence Gate

## Purpose

Day 7 defines the evidence gate Sprint 148 and Sprint 149 must use before any
Windows staged-test or install-validation claim is promoted. The gate preserves
the current distinction between reviewed Windows CMake test coverage,
supplemental Windows CMake install/downstream confidence, staged pthread/POSIX
tests, and unsupported Windows Makefile or `pkg-config` parity.

## Current Windows Support Tiers

| Tier | Current Surface | Evidence Owner | Current Claim |
| --- | --- | --- | --- |
| Reviewed | `.github/workflows/windows-ci.yml::build-and-test` | CI/platform owner | Windows has a reviewed MSVC CMake configure/build/CTest subset on `windows-2022`. |
| Supplemental | `.github/workflows/windows-ci.yml::install-and-downstream` | Platform/package owner | Windows has supplemental CMake-first static install/downstream confidence. |
| Staged | `test_threads`, `test_sprint4_integration`, `test_fuzz` | Platform/test owner | These tests are intentionally excluded from the reviewed Windows CMake subset. |
| Deferred | Windows Makefile, Windows `pkg-config`, separate reviewed Windows install-validation parity | Platform/package owner | These remain non-claims unless Sprint 149 explicitly promotes or rejects the install-validation lane. |
| Unsupported | Shared-library ABI, dynamic ABI compatibility, runtime-loader support, package-manager distribution | Package/ABI owner | These remain outside current Windows support. |

## Current Reviewed Windows CMake Lane

The reviewed Windows lane is:

- workflow: `.github/workflows/windows-ci.yml`
- job: `Windows enforced reviewed CMake consumer subset (MSVC)`
- runner: `windows-2022`
- generator: `Visual Studio 17 2022`
- architecture: `x64`
- build type: `Release`
- enforced registered CTest count: `EXPECTED_WINDOWS_CTEST_COUNT=56`
- commands:
  - `cmake -S . -B build -G "Visual Studio 17 2022" -A x64`
  - `cmake --build build --config Release`
  - `ctest --test-dir build -C Release -N`
  - `ctest --test-dir build -C Release --output-on-failure`

Current CMake exclusions:

| Test | Current CMake Gate | Blocker |
| --- | --- | --- |
| `test_threads` | `if(Threads_FOUND AND NOT WIN32)` | Source includes pthread APIs directly. |
| `test_sprint4_integration` | `if(Threads_FOUND AND NOT WIN32)` | Source includes pthread APIs directly. |
| `test_fuzz` | `if(NOT WIN32 AND NOT MSVC)` | Source depends on POSIX temp-file behavior. |

## Staged-Test Promotion Gate

Sprint 148 may promote a staged test only when all required evidence lands.

| Test Surface | Promotion Options | Required Evidence | Non-Claim Boundary |
| --- | --- | --- | --- |
| `test_threads` | Port source to a portable thread abstraction, add a Windows-native equivalent, or split POSIX/Windows proof owners. | Source change; CMake registration on Windows; hosted MSVC configure/build/execute proof; updated expected-count policy; docs/report updates. | Promotes only the selected thread lifecycle coverage, not full pthread parity. |
| `test_sprint4_integration` | Port pthread-dependent integration behavior or add an equivalent Windows-specific integration proof. | Source/test change; CMake registration on Windows; hosted MSVC execution; clear relationship to existing POSIX test; docs/report updates. | Promotes only the selected Sprint 4 integration behavior, not broader Windows feature parity. |
| `test_fuzz` | Replace POSIX temp-file assumptions, add a portable temp-file helper, or split a Windows bounded property lane. | Source/test change; CMake registration on Windows; deterministic seed policy; hosted MSVC execution; expected-count update; docs/report updates. | Promotes only the bounded lifecycle/property lane that actually runs on Windows. |

Required checks when `.c` or `.h` changes:

```sh
make format && make lint && make test
```

Additional expected local checks before PR:

```sh
cmake -S . -B build
cmake --build build
ctest --test-dir build --output-on-failure
```

Windows-specific proof must come from hosted GitHub Actions because local Unix
checks cannot prove MSVC or Windows runner behavior.

## Windows Install-Validation Parity Gate

Sprint 149 must make an explicit product decision. It may promote the current
supplemental install/downstream path, keep it supplemental, or reject reviewed
install-validation parity.

Promotion requires:

- workflow job name and comments updated from supplemental to reviewed only if
  evidence supports the promotion;
- hosted Windows run ID, commit SHA, job name, and conclusion recorded;
- static `.lib` installed at the expected path;
- all public headers installed;
- no `.dll` shared-library artifacts installed;
- `SparseConfig.cmake`, `SparseConfigVersion.cmake`, `SparseTargets.cmake`,
  and `sparse.pc` installed;
- installed CMake package metadata contains no shared-library imported
  metadata;
- installed `sparse.pc` preserves static archive wording and no unsupported
  package or ABI wording;
- downstream CMake example configures, builds, runs, and prints the expected
  version, solution, and `OK` markers;
- exact-version downstream consumer configures, builds, and runs;
- mismatched-version consumer fails to configure;
- README, INSTALL, maintainer guide, workflow comments, and report-family rows
  agree with the selected support tier.

Promotion does not imply:

- Windows Makefile parity;
- Windows `pkg-config` compile/link parity;
- shared-library support;
- dynamic ABI compatibility;
- runtime-loader compatibility;
- package-manager distribution;
- broad Windows platform parity.

Rejection or continued supplemental status must record the exact blockers and
preserve existing non-claim wording.

## CTest Count Policy

Current policy:

- `EXPECTED_WINDOWS_CTEST_COUNT` is manually enforced in
  `.github/workflows/windows-ci.yml`.
- The current reviewed count is `56`.
- New Windows-registered tests must intentionally update the count.

Sprint 148 may keep manual enforcement or replace it with a maintained checked
list/generated expectation. Either path must satisfy these rules:

1. A new Windows test registration must be intentional and reviewable.
2. A staged test cannot disappear silently from the reviewed lane.
3. The workflow output must continue to print current staged exclusions and
   blockers until they are resolved.
4. If a staged test is promoted, the output must name the promoted surface and
   any remaining staged tests.
5. CTest enumeration must be checked before execution so registration drift is
   caught even if a later test fails.

## Hosted Log Requirements

A Windows claim cannot be promoted from local evidence alone. Promotion
artifacts must record:

- workflow file and job name;
- GitHub Actions run ID and URL;
- commit SHA;
- branch or PR number;
- runner image;
- generator, architecture, and build configuration;
- CTest registered count before and after promotion;
- promoted tests and remaining staged tests;
- install/downstream proof result if Sprint 149 promotes that lane;
- failed or skipped jobs with final disposition.

## Required Documentation And Report Updates

| Surface | Required Update When Promoted |
| --- | --- |
| `.github/workflows/windows-ci.yml` | Job names, comments, expected count, staged-exclusion output, and install/downstream tier wording. |
| `CMakeLists.txt` | Test registration gates and any new platform-specific proof-owner registration. |
| `README.md` | Cross-platform CI contract and Windows support-tier summary. |
| `INSTALL.md` | Windows CMake install/downstream interpretation and non-claims. |
| `docs/maintainer_guide.md` | Maintainer support-tier interpretation, staged blockers, and package/CI evidence boundaries. |
| `tests/corpus/manifests/report_families.tsv` | CI lane row claim scope/non-claims if reviewed Windows evidence changes. |
| Sprint artifacts | Validation log with hosted run IDs and final promoted/rejected decision. |

## Sprint 148 Prerequisite Checklist

- Confirm branch starts from current `master`.
- Reconfirm `EXPECTED_WINDOWS_CTEST_COUNT=56` before changes.
- Reconfirm CMake gates for `test_threads`, `test_sprint4_integration`, and
  `test_fuzz`.
- Choose per-test path: direct port, Windows-native equivalent, split proof
  owner, or explicit rejection.
- Keep Windows install-validation parity out of Sprint 148 except for
  preserving handoff notes to Sprint 149.
- Run full C quality gate if any `.c` or `.h` file changes.
- Capture hosted Windows run IDs before promoting any Windows claim.

## Stop Conditions

- A Windows claim lacks hosted Windows proof.
- A staged test is counted as promoted without intentional CMake registration.
- The expected CTest count changes without a documented reason.
- A source port weakens Linux/macOS coverage or removes POSIX proof without an
  equivalent.
- Windows CMake install/downstream wording is promoted to reviewed parity
  without Sprint 149 product decision evidence.
- Documentation implies Windows Makefile, Windows `pkg-config`, shared-library,
  dynamic ABI, runtime-loader, package-manager, or broad platform parity.

## Day 8 Handoff

Day 8 should define analogous evidence gates for QR and partial-SVD corpus
family expansion. It should reuse the Day 6 claim boundaries and keep raw
QR-basis identity, raw singular-vector identity, broad solver correctness, and
external parity out of corpus-family claims.

## Completion Criteria

| Criterion | Status | Evidence |
| --- | --- | --- |
| Sprint 148 can implement against an explicit gate. | Complete | Staged-test promotion gate, CTest count policy, hosted log requirements, and prerequisite checklist are defined. |
| Reviewed and supplemental Windows claims stay separate. | Complete | Reviewed CMake lane and supplemental install/downstream lane have separate gates and wording boundaries. |
| Unpromoted Windows parity remains a non-claim. | Complete | Stop conditions and install-validation non-claims preserve Makefile, `pkg-config`, shared-library, ABI, package-manager, and broad platform boundaries. |
