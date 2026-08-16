# Day 3 Test And CI Baseline Inventory

## Scope

Day 3 freezes the current local and hosted validation surface for Sprint 157.
This artifact records what tests and CI lanes exist, how they are interpreted,
and which generated/report surfaces are reviewed, supplemental, advisory, or
local-only. It does not run the full test suite or promote any validation lane.

## Inventory Commands

| Purpose | Command |
| --- | --- |
| Worktree state | `git status --short --branch` |
| Day 3 plan scope | `sed -n '105,137p' docs/planning/EPIC_14/SPRINT_157/PLAN.md` |
| Makefile validation targets | `rg -n "^(TEST_SRCS|TEST_BINS|test:|sanitize|asan|tsan|omp|quality-review|quality-review-cmake|deadcode|coverage|install:|report-index|bench-fast|wall-check|source-list-check)" Makefile` |
| Top-level C tests | `find tests -maxdepth 1 -type f -name 'test_*.c' | sort` |
| Workflow support tiers | `rg -n "name:|runs-on:|EXPECTED_WINDOWS_CTEST_COUNT|make |ctest|test_install|test_cmake_install|static_package|upload-artifact|supplemental|reviewed|staged|pkg-config|CMake" .github/workflows/*.yml` |
| Script/corpus support files | `find tests -maxdepth 3 -type f \( -name '*.py' -o -name '*.sh' -o -name '*.tsv' -o -name '*.md' \) | sort` |
| CMake CTest count | `cmake -S . -B build-s157-baseline && ctest --test-dir build-s157-baseline -N` |

## Test Target Inventory

| Surface | Count / targets | Interpretation |
| --- | --- | --- |
| Top-level C test files | 59 `tests/test_*.c` files | Makefile `TEST_SRCS` and local C proof owners. |
| Makefile test binaries | 59 entries in `TEST_SRCS` | `make test` builds and runs the local test suite through Make. |
| CMake CTest registrations | 59 tests from configure-time `ctest -N` | CMake reviewed path registers the same count on this local baseline. |
| Focused corpus tests | `test_qr_corpus`, `test_svd_partial_corpus` | Fixture-local QR and partial-SVD corpus proof owners. |
| Thread/property tests | `test_threads`, `test_sprint4_integration`, `test_fuzz` | Promoted into the reviewed Windows CMake subset after Epic 13; still not a broad Windows parity claim. |
| Package tests | `tests/test_install.sh`, `tests/test_cmake_install.sh`, `scripts/static_package_deferral_check.sh` | Static-first install/export/downstream and shared-library deferral proof surfaces. |
| Report-index tests | `tests/test_normalize_report_index.py` plus `scripts/normalize_report_index.py` | Normalized report row, freshness, skip/defer, package, and generated-family semantics. |
| External dense-reference helpers | `tests/lu_external_dense_reference.py`, `tests/qr_external_dense_reference.py`, `tests/svd_external_dense_reference.py`, `tests/chol_external_dense_reference.py`, `tests/ldlt_external_dense_reference.py` | Bounded external-oracle helpers, not broad ecosystem parity. |

## Core Local Validation Targets

| Target / command | Surface | Day 3 interpretation |
| --- | --- | --- |
| `make test` | Full Makefile C test suite | Direct local runtime proof; supplemental in Linux CI. |
| `make format` / `make format-check` | C formatting | Formatting gate for C/header changes. |
| `make lint` | Static lint/tooling | Lint gate for C/header changes. |
| `make source-list-check` | Manifest, Makefile, CMake source-list consistency | Library source-list drift guard. |
| `make quality-review-compile` | Format-check, source-list-check, lint | Reviewed Makefile compile-quality path. |
| `make quality-review-cmake` | CMake configure/build/CTest parity | Reviewed CMake parity path. |
| `make quality-review` | Format-check, lint, test, deadcode-check | Stronger local review path with test execution and dead-code completeness. |
| `make quality-review-full` | Reviewed Makefile and CMake paths | Broad local quality wrapper. |
| `make sanitize`, `make asan`, `make tsan`, `make omp` | Sanitizer/OpenMP variants | Supplemental runtime signals; some are platform/toolchain sensitive. |
| `make wall-check` | Benchmark-side bounded wall signal | Enforced macOS signal and local performance-adjacent guard. |
| `make bench-fast` | Fast benchmark subset | Supplemental CI/runtime signal, not a portable performance claim. |
| `make coverage` | Coverage report with `COV_THRESHOLD=80` | Supplemental, tree-mutating coverage signal. |
| `make deadcode-report`, `make deadcode-check` | Dead-code report and completeness check | Reviewed Linux lane; completeness/context gate, not a zero-dead-code guarantee. |

## CI Support-Tier Baseline

| Workflow | Job / lane | Tier | Command surface | Non-claims |
| --- | --- | --- | --- | --- |
| `.github/workflows/ci.yml` | Linux supplemental runtime and bench-fast path | Supplemental runtime and benchmark signal | `make test`, `make sanitize`, `make asan`, `make bench-build`, `make bench-fast` | Not the reviewed source-of-truth package/platform claim by itself; no portable performance claim. |
| `.github/workflows/ci.yml` | Linux enforced reviewed CMake parity path | Reviewed | `make quality-review-cmake` | CMake parity only; not package-manager, shared-library, or performance proof. |
| `.github/workflows/ci.yml` | Linux reviewed static-first package contract | Reviewed | `tests/test_install.sh`, `tests/test_cmake_install.sh`, `scripts/static_package_deferral_check.sh` | Static-first package only; no shared-library, dynamic ABI, runtime-loader, or package-manager support. |
| `.github/workflows/ci.yml` | Linux supplemental ThreadSanitizer coverage | Supplemental | TSan builds/runs for threads and selected eigensolver OpenMP paths | Toolchain/runtime-specific signal; no broad thread or OpenMP portability claim. |
| `.github/workflows/ci.yml` | Linux enforced reviewed Makefile compile-quality path | Reviewed | `make quality-review-compile` | Compile-quality path; not full runtime proof. |
| `.github/workflows/ci.yml` | Linux enforced dead-code report and completeness path | Reviewed | `make deadcode-report`, `make deadcode-check`, artifact upload | Report completeness/context; not zero-dead-code guarantee. |
| `.github/workflows/ci.yml` | Linux supplemental coverage report | Supplemental | `make coverage`, coverage artifact upload | Supplemental coverage; no product-quality or branch-coverage completeness claim. |
| `.github/workflows/macos-ci.yml` | macOS enforced Apple Clang reviewed path + supplemental GCC leg | Mixed reviewed/supplemental | Apple Clang reviewed Make/CMake/wall/sanitize, Homebrew GCC supplemental build/test/wall | No broad macOS parity beyond documented lanes. |
| `.github/workflows/macos-ci.yml` | macOS reviewed static-first install and pkg-config proof | Reviewed | `bash tests/test_install.sh` | Static-first Make install/`pkg-config`; no package-manager/shared-library/dynamic ABI support. |
| `.github/workflows/macos-ci.yml` | macOS reviewed static-first CMake install/export proof | Reviewed | `bash tests/test_cmake_install.sh`, `bash scripts/static_package_deferral_check.sh` | Static CMake package only. |
| `.github/workflows/windows-ci.yml` | Windows enforced reviewed CMake consumer subset (MSVC) | Reviewed | Visual Studio CMake configure/build, `ctest -N`, full `ctest` | CMake-first only; no Windows Makefile parity, Windows `pkg-config` execution parity, package-manager, shared-library, dynamic ABI, runtime-loader, or broad Windows parity. |
| `.github/workflows/windows-ci.yml` | Windows reviewed CMake install/downstream validation path | Reviewed | CMake install, installed static `.lib`, CMake package metadata, `sparse.pc` metadata, generated and maintained CMake consumers, exact-version and mismatch checks | CMake install/downstream only; no Windows Makefile or `pkg-config` execution parity. |

## Windows Reviewed Surface Snapshot

| Field | Current Baseline |
| --- | --- |
| Runner | `windows-2022` |
| Generator | `Visual Studio 17 2022` with `-A x64` |
| Enforced count variable | `EXPECTED_WINDOWS_CTEST_COUNT: "59"` |
| Local configure-only CTest enumeration | `Total Tests: 59` from `ctest --test-dir build-s157-baseline -N` |
| Promoted portable tests | `test_threads`, `test_sprint4_integration`, `test_fuzz` |
| Install validation | reviewed CMake install/downstream validation path |
| Explicit residual/non-claim | Windows Makefile parity and Windows `pkg-config` execution parity remain out of scope. |

The local `ctest -N` command was run after CMake configure only. It printed
the registered tests and expected executable lookup text because the build
directory had not compiled test executables yet. The useful Day 3 signal is the
enumerated `Total Tests: 59`, not test execution.

## Generated, Advisory, And Local-Only Report Surfaces

| Family / target | Artifact path | Day 3 interpretation |
| --- | --- | --- |
| Oracle freshness | `make report-index-oracle-freshness`, `build/corpus/oracle/*.tsv`, `build/corpus-reports/*` | Selected local-only generated freshness; no hosted proof yet. |
| Comparison freshness | `make report-index-comparison-freshness`, `build/comparison/**`, `build/report-index/**` | Selected local-only generated comparison freshness; no broad ecosystem parity. |
| Canonical benchmark report | `make bench-canonical-report`, `build/bench-reports/canonical/index.tsv` | Local measurement context; no portable performance or release benchmark claim. |
| Runtime sentinels | `make performance-sentinels`, `build/bench-reports/sentinels/*.tsv` | Bounded sentinel context; only selected hard-gate rows are gates. |
| Large-matrix guardrails | `make large-matrix-guardrails`, `build/bench-reports/large-matrix-guardrails/index.tsv` | Structural/bounded local checks; no broad scalability claim. |
| Dead-code report | `make deadcode-report`, `build/deadcode/report.tsv` | Reviewed Linux report-completeness surface; no zero-dead-code guarantee. |
| Coverage report | `make coverage`, `coverage/coverage-src.info`, `coverage/html/` | Supplemental coverage; no completeness/product-quality claim. |
| Package report-index rows | `python3 scripts/normalize_report_index.py --family package --check` | Source-controlled proof-owner rows; not proof that install scripts just ran. |

## Draft Validation Command Matrix

| Change type | Required baseline check | Supplemental/focused checks |
| --- | --- | --- |
| Documentation-only planning/docs | `git diff --check` | claim wording scan over touched docs. |
| Public documentation claim changes | `git diff --check` | scan README, INSTALL, API reference, maintainer guide, solver-selection, corpus, benchmark, and example docs for unsupported claims. |
| Python report/comparison/corpus scripts | targeted Python tests such as `python3 tests/test_normalize_report_index.py` | selected freshness commands for touched family. |
| Shell install/package scripts | affected shell script directly | Linux/macOS package proof lanes if install metadata changes. |
| C implementation or test changes | `make format && make lint && make test` | focused test binary, sanitizer, CMake parity, package proof, or generated-report command depending on touched surface. |
| Public header changes | `make format && make lint && make test` | declaration-preservation check and generated API docs policy from Sprint 158. |
| CMake/Makefile source-list changes | `make source-list-check` plus affected build path | `make quality-review-cmake` and package install/export checks when package metadata changes. |
| CI workflow changes | local equivalent command where possible | hosted CI reconciliation after PR run; update support-tier docs and expected-count policies. |
| Generated evidence promotion | selected freshness command | hosted artifact retention and support-tier update. |

## Day 3 Handoff

Day 4 should consume this test/CI baseline and capture the documentation and
claim baseline:

- public and maintainer docs inventory;
- accepted claim and non-claim wording;
- support-tier owners;
- unsupported state-of-the-art, external parity, performance, package,
  Windows, shared-library, dynamic ABI, runtime-loader, and generated-report
  wording scan.

## Completion Check

- Test target inventory is captured.
- CI support-tier baseline is captured.
- Windows reviewed surface count and CMake-first boundary are documented.
- Generated-report, coverage, benchmark, and dead-code advisory/local-only
  boundaries are explicit.
- Validation commands are mapped by touched surface for later Sprint 157 days.
