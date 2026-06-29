# Sprint 97 Day 1: Build Topology Baseline

## Purpose

Day 1 opens Sprint 97 by making the live build, package, workflow, and
platform-proof topology explicit. The goal is not to reduce duplication yet.
Day 1 defines the candidate surfaces and validation expectations that Day 2
can audit and rank.

## Sprint 97 Scope

Sprint 97 implements the Epic 9 build, packaging, and cross-platform product
convergence phase centered on:

- build-topology duplication between Make, CMake, CI, and install/export proof
- bounded convergence architecture before build edits
- source-list or workflow reduction where evidence supports it
- a package-surface decision around static-first versus one bounded shared
  library lane
- consumer and workflow follow-through after package decisions
- macOS and Windows truth calibration without fake parity
- validation and closeout

Non-goals for Day 1:

- no Makefile edits
- no CMake edits
- no workflow edits
- no install/export script edits
- no shared-library package claim
- no source-list generation design yet
- no attempt to make Windows, macOS, and Linux proof surfaces equivalent

## Live Topology Inventory

| Surface | Current role | Day 1 reading |
|---|---|---|
| `Makefile` | primary local build, format, lint, test, benchmark, coverage, sanitizer, install, and reviewed-quality target owner | strongest local command topology and source-list owner |
| `CMakeLists.txt` | CMake build, test registration, install/export, package config, and consumer target owner | strongest CMake consumer/package topology owner |
| `.github/workflows/ci.yml` | Linux CI and strongest reviewed source-of-truth workflow | likely authoritative reviewed baseline, not automatically a reduction target |
| `.github/workflows/macos-ci.yml` | Apple Clang reviewed path plus supplemental GCC and static install/pkg-config confidence path | platform-specific proof and package-story support |
| `.github/workflows/windows-ci.yml` | reviewed Windows CMake-first consumer subset | platform-specific proof with explicit staged exclusions |
| `tests/test_install.sh` | Make install and pkg-config downstream proof | Unix-side static install proof owner |
| `tests/test_cmake_install.sh` | CMake install, export, find_package, and version compatibility proof | CMake package/export proof owner |
| `README.md` | public build, CI, package, and platform narrative | public claim surface that must match proof |
| `INSTALL.md` | install/package guidance | package contract front door |

## Measured File-Family Counts

| Family | Count | Day 1 implication |
|---|---:|---|
| `src/*.c` | 42 | library source lists are large enough that Make/CMake duplication is review-costly |
| `src/*.h` | 18 | internal headers matter for format/lint coverage and source ownership |
| `tests/test_*.c` | 54 | test registration parity is a major Make/CMake/CI concern |
| `tests/*.h` | 5 | helper headers are formatting/lint support surfaces |
| `benchmarks/*.c` | 16 | benchmark registration is duplicated but lower correctness risk than tests |
| `examples/*.c` | 12 | Make wildcard and explicit CMake example registration differ |
| `include/*.h` | 18 | public install/export header inventory must stay aligned |

## Initial Duplication Candidates

| Candidate | Evidence | Initial Day 1 classification |
|---|---|---|
| Library source list | `Makefile` `LIB_SRCS` and `CMakeLists.txt` `add_library` both enumerate 42 library sources | high-value Day 2 audit candidate |
| Test registration list | `Makefile` `TEST_SRCS` and `CMakeLists.txt` `add_sparse_test(...)` both enumerate test executables, with platform exclusions layered differently | high-value Day 2 audit candidate |
| Benchmark list | `Makefile` `BENCH_SRCS` and CMake benchmark executable blocks both enumerate benchmark programs | medium-value audit candidate |
| Example list | Make uses `$(wildcard $(EXDIR)/*.c)` while CMake explicitly registers each example | medium-value audit candidate with different ownership semantics |
| Expected Windows CTest count | Windows workflow hard-codes `EXPECTED_WINDOWS_CTEST_COUNT: "51"` | proof assertion; audit for maintenance cost, not automatic removal |
| Package static-first wording | CMake, README, and INSTALL all carry static-first language | claim consistency surface; not harmful duplication if kept synchronized |
| Install/export proof | `tests/test_install.sh`, `tests/test_cmake_install.sh`, CMake install rules, and docs all describe package surfaces | consumer-proof alignment candidate |
| Platform exclusions | README, macOS workflow, and Windows workflow all explain platform limits | truth-calibration surface; preserve explicit limits unless evidence changes |

## Current Package Contract Signals

The live tree currently presents a static-first package story:

- `CMakeLists.txt` builds `sparse_lu_ortho` as a static library.
- `BUILD_SHARED_LIBS=ON` does not change the maintained package output and
  emits a status message explaining that the static archive surface remains
  active.
- `INSTALL.md` says the maintained install surface is static-first.
- README's cross-platform CI contract distinguishes Linux, macOS, and Windows
  proof strength instead of claiming full parity.
- Make install and CMake install both install a static archive and consumer
  metadata.

Day 7 and Day 8 should decide from evidence whether to preserve that contract
or earn one bounded shared-library lane. Day 1 records no decision change.

## Current Platform Proof Signals

### Linux

- README identifies Linux as the strongest reviewed source of truth.
- Local reviewed commands include `make quality-review-compile`,
  `make quality-review-cmake-compile`, and `make quality-review-cmake`.
- Linux owns the broadest Make/CMake/dead-code/benchmark proof story.

### macOS

- `.github/workflows/macos-ci.yml` enforces Apple Clang reviewed compile and
  CMake parity paths.
- A Homebrew GCC matrix leg is supplemental direct build/test coverage.
- The Make install/pkg-config job is supplemental package confidence, not a
  reviewed install/export parity lane.
- Apple Clang sanitizer and wall-check signals are platform-specific proof.

### Windows

- `.github/workflows/windows-ci.yml` enforces a reviewed CMake-first consumer
  subset with MSVC on `windows-2022`.
- The expected Windows CTest count is 51.
- `test_threads`, `test_sprint4_integration`, and `test_fuzz` are explicitly
  outside the reviewed Windows subset.
- The Windows lane does not claim Makefile parity or a separate reviewed
  install-validation lane.

## Validation Expectations

Use this validation split during Sprint 97:

| Change type | Minimum validation expectation |
|---|---|
| Planning/docs-only artifacts | `git diff --check` and whitespace/link sanity as appropriate |
| `.c` or `.h` changes | `make format && make lint && make test` |
| Makefile source/test/bench/example registration changes | full quality chain plus Make/CMake registration parity checks |
| CMake source/test/install/export changes | full quality chain plus `make quality-review-cmake` or targeted CMake configure/build/ctest/install checks |
| Workflow-only changes | docs hygiene plus local equivalent commands where possible; CI remains final proof for platform-specific syntax |
| install/export script changes | targeted install script plus full quality chain if code/build inputs changed |
| package contract docs-only changes | docs hygiene plus targeted claim scan across README, INSTALL, CMake, workflows, and examples |
| shared-library lane changes | full quality chain plus explicit package/install/export proof for the new lane before any public claim |

## Day 1 Result

Sprint 97 starts from a current topology baseline. The strongest Day 2 audit
candidates are the duplicated library source list, duplicated test
registration list, and package/consumer proof alignment surfaces. Platform
workflow messages and expected counts are also maintenance pressure points, but
Day 2 should distinguish proof assertions from duplication that should be
centralized or generated.
