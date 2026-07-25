# Sprint 134 Day 2 - Platform Gap Audit

## Purpose

Day 2 re-audits current Linux, macOS, and Windows CI/install support tiers
against Sprint 133 static-first package truth. The audit separates reviewed
CI, supplemental CI, local proof, staged exclusions, explicit deferrals, and
unsupported platform behavior before Sprint 134 decides whether to promote or
defer additional install lanes.

## Audited Inputs

| Input | Audit role |
| --- | --- |
| `.github/workflows/ci.yml` | Linux reviewed/supplemental CI topology and install-proof non-claim comments. |
| `.github/workflows/macos-ci.yml` | macOS reviewed Apple Clang lane, supplemental Homebrew GCC lane, and supplemental Make install/`pkg-config` job. |
| `.github/workflows/windows-ci.yml` | Windows reviewed MSVC CMake subset, expected CTest count, and staged exclusion output. |
| `CMakeLists.txt` | CMake test registration and Windows staged exclusion mechanics. |
| `tests/test_install.sh` | Local Unix Make install/`pkg-config` proof and macOS supplemental install job command. |
| `tests/test_cmake_install.sh` | Local CMake install/export proof and candidate Linux/macOS/Windows install parity source. |
| `scripts/static_package_deferral_check.sh` | Static-first package non-claim guard. |
| `README.md`, `INSTALL.md`, `docs/maintainer_guide.md` | Public and maintainer support-tier wording. |
| Sprint 133 closeout and retrospective | Static-first package support baseline and Sprint 134 residual handoff. |

## Platform Classification

| Platform | Reviewed evidence | Supplemental/local evidence | Staged/deferred/non-claim evidence |
| --- | --- | --- | --- |
| Linux | `.github/workflows/ci.yml` reviewed Makefile compile-quality path, reviewed CMake parity path, and reviewed dead-code report/check path. | Supplemental direct `make test`, sanitizer, benchmark, TSan, and coverage jobs. Local package proof scripts exist but are not reviewed CI lanes. | No separate reviewed Linux install CI lane yet; package-manager and shared-library support remain deferred. |
| macOS | `.github/workflows/macos-ci.yml` reviewed Apple Clang compile-quality, CMake parity, wall-check, and sanitizer path. | Supplemental Homebrew GCC direct build/test/wall-check and supplemental Make install/`pkg-config` proof via `bash tests/test_install.sh`. | No reviewed macOS CMake install/export parity claim; supplemental Make install proof does not widen macOS to full install/export parity. |
| Windows | `.github/workflows/windows-ci.yml` reviewed MSVC CMake configure, build, `ctest -N`, expected-count check, and full `ctest`. | CTest count inspection and workflow output that names staged exclusions. | No reviewed Windows install-validation lane, no Windows Makefile parity, and no reviewed Windows property/fuzz/thread lane for staged tests. |

## Linux Install CI Gap List

| Gap | Current evidence | Decision needed |
| --- | --- | --- |
| Reviewed Linux install CI lane | Linux workflow comments say focused install/package scripts remain developer-side proof surfaces. | Day 3 should decide whether to promote `tests/test_install.sh`, `tests/test_cmake_install.sh`, `scripts/static_package_deferral_check.sh`, a subset, or none. |
| Runtime and tool cost | Local package proofs passed in Sprint 133 but are not measured as reviewed CI jobs in the Linux workflow. | Day 3 should evaluate runtime budget, tool availability, and flake risk. |
| CI failure ownership | Existing reviewed Linux lanes have clear Makefile/CMake/dead-code ownership; install proof ownership is local. | Day 3 should assign triage ownership if promoted. |
| Package contract scope | Sprint 133 local proofs validate static-first install/export, not package-manager or shared-library support. | Any Linux promotion must preserve static-first non-claims. |

## macOS Install/Export Parity Gap List

| Gap | Current evidence | Decision needed |
| --- | --- | --- |
| Reviewed macOS CMake install/export parity | macOS workflow has reviewed CMake parity via `make quality-review-cmake`, but the install/package job is Make install/`pkg-config` only and explicitly supplemental. | Days 5-6 should decide whether `tests/test_cmake_install.sh` belongs in macOS CI and at what tier. |
| Supplemental Make install scope | `install-and-pkgconfig` runs `bash tests/test_install.sh` on `macos-latest`. | Keep as supplemental unless Day 6 explicitly widens support with proof and docs. |
| CMake package behavior on macOS | Local CMake install proof exists but is not a macOS reviewed lane. | Day 5 should assess `cmake --install`, `find_package(Sparse)`, path leakage checks, and runtime cost on macOS. |
| Toolchain variability | macOS workflow already distinguishes Apple Clang reviewed path and Homebrew GCC supplemental path. | Any install/export addition must specify Apple Clang versus supplemental runner semantics. |

## Windows Install-Validation Gap List

| Gap | Current evidence | Decision needed |
| --- | --- | --- |
| Reviewed Windows install-validation lane | Windows workflow comments explicitly say no separate reviewed install-validation lane. | Days 8-9 should design/implement or defer MSVC install/downstream consumer proof. |
| CMake install consumer proof on Windows | Current reviewed Windows lane builds and runs CTest but does not run `cmake --install` or an installed downstream consumer. | Day 8 should decide whether a Windows-specific installed CMake consumer proof is feasible. |
| Windows Makefile parity | Workflow comments and INSTALL state Windows uses CMake exclusively. | Keep separate from CMake install validation; do not treat Makefile parity as a prerequisite for CMake-first install proof. |
| Package-manager and shared-library support | Sprint 133 defers both. | Windows install work must not introduce package-manager or shared-library claims. |

## Windows Staged-Exclusion Gap List

| Staged area | Current evidence | Gap or decision |
| --- | --- | --- |
| `test_threads` | Excluded from the reviewed Windows CMake subset and named in workflow output. | Day 10 should re-check whether thread support can be registered or remains staged. |
| `test_sprint4_integration` | Excluded from the reviewed Windows CMake subset and named in workflow output. | Day 10 should verify whether the historical integration/thread dependency still blocks promotion. |
| `test_fuzz` | Excluded from the reviewed Windows CMake subset; maintainer guide says property/fuzz lane is not reviewed Windows evidence. | Day 10 should revisit fuzz/property feasibility and proof ownership. |
| CTest expected count | Workflow expects `EXPECTED_WINDOWS_CTEST_COUNT=54`. | Any Windows membership change must update the count and docs together. |

## Windows Makefile Gap Notes

- Current Windows support is CMake-first with MSVC and Visual Studio 2022.
- `INSTALL.md` states Windows users should use CMake exclusively and not
  Makefile install targets.
- `.github/workflows/windows-ci.yml` explicitly says no Windows Makefile
  parity.
- Day 8-11 Windows work should keep Makefile parity as a separate staged or
  deferred item unless a new workflow and validation path is intentionally
  added.

## Support Wording Drift Queue

| Surface | Drift or risk | Day owner |
| --- | --- | --- |
| `docs/maintainer_guide.md` | Sprint 112 snapshot still says Windows has 51 registered CTest tests, while current `.github/workflows/windows-ci.yml` expects 54. | Day 12 support-tier docs alignment, or earlier if Windows CTest work changes counts. |
| `INSTALL.md` | Current platform table and install-validation section align with the reviewed/supplemental/local split. | Re-check after Linux/macOS/Windows decisions. |
| `README.md` | Front-door CI summary aligns with current reviewed/supplemental/staged model. | Re-check after Linux/macOS/Windows decisions. |
| Workflow comments | Linux, macOS, and Windows comments already preserve non-claims for install proof, macOS parity, and Windows staged lanes. | Update only if Days 4, 7, 9, or 11 change tier decisions. |

## Install Proof Owners

| Proof | Current owner | Current tier |
| --- | --- | --- |
| `tests/test_install.sh` | Local Unix Make install/`pkg-config` proof; also macOS supplemental job command. | Local plus macOS supplemental CI. |
| `tests/test_cmake_install.sh` | Local CMake install/export and installed CMake consumer proof. | Local only. |
| `scripts/static_package_deferral_check.sh` | Local static-first package deferral guard. | Local only. |
| Linux workflow install proof | None separate today. | Deferred pending Day 3 decision. |
| macOS CMake install/export proof | None separate today. | Deferred pending Day 6 decision. |
| Windows install/downstream proof | None separate today. | Deferred pending Day 8 decision. |

## Day 3 Handoff

Day 3 should decide the Linux install CI tier:

- whether to promote all, part, or none of the local package proof stack into
  reviewed Linux CI;
- what runtime/tooling and failure-ownership cost promotion would carry;
- whether the Linux workflow should remain as-is with clearer deferral notes;
- what local validation is required for the selected decision.

## Completion Criteria

| Criterion | Status | Evidence |
| --- | --- | --- |
| Each platform has reviewed, supplemental, local, staged, and deferred evidence classified. | Complete | Platform classification and gap lists separate Linux, macOS, and Windows tiers. |
| CTest counts and staged exclusions are recorded before changes. | Complete | Windows baseline records expected count 54 and staged `test_threads`, `test_sprint4_integration`, and `test_fuzz`. |
| Install parity gaps are separated from package-contract gaps. | Complete | Linux/macOS/Windows install proof gaps are separate from Sprint 133 static-first package non-claims. |
