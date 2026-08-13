# Sprint 156 Day 3 Validation Matrix

## Purpose

Day 3 defines the final validation matrix for Sprint 156 closeout. The matrix
turns the Day 2 evidence inventory into concrete commands, owners, trigger
rules, evidence boundaries, and skip/defer semantics. Day 3 does not run the
full validation package; it defines what Days 4 through 10 must run, review, or
explicitly defer.

## Source Inputs

- `docs/planning/EPIC_13/SPRINT_147/artifacts/day12-quality-surface-map.md`
- `docs/planning/EPIC_13/SPRINT_156/artifacts/day2-evidence-inventory.md`
- `Makefile`
- `tests/test_install.sh`
- `tests/test_cmake_install.sh`
- `scripts/static_package_deferral_check.sh`
- `scripts/validate_corpus_schema.py`
- `scripts/run_corpus_oracle.py`
- `scripts/normalize_report_index.py`
- `scripts/run_external_comparison.py`
- `.github/workflows/ci.yml`
- `.github/workflows/macos-ci.yml`
- `.github/workflows/windows-ci.yml`

## Local Baseline Matrix

| Surface | Trigger | Command Or Review | Owner | Evidence Boundary |
| --- | --- | --- | --- | --- |
| Documentation-only Sprint 156 edits | Any `.md`-only closeout day | `git diff --check`; claim scan against Day 2 inventory; link/path spot checks for edited references | Documentation owner | Proves formatting and local claim hygiene only; does not prove code, platform, package, or numerical behavior. |
| C implementation edits | Any changed `.c` file | `make format && make lint && make test` | Implementation owner | Required before commit; proves current local Makefile build/test baseline only. |
| Public or internal header edits | Any changed `.h` file | `make format && make lint && make test`; declaration-preservation scan for public headers | API owner | Required before commit; declaration preservation is required before claiming no public API drift. |
| Strongest local closeout baseline | Day 4 final local baseline or any broad local code/build confidence claim | `make quality-review-full` | Validation owner | Strongest local Makefile plus CMake reviewed baseline; still not hosted platform proof. |
| Makefile source-list or registration changes | `Makefile`, source lists, test registration, or build target edits | `make source-list-check`; relevant target; full C gate if compiled surface changes | Build owner | Prevents silent source/test registration drift. |
| CMake registration or package export edits | `CMakeLists.txt`, `cmake/*.in`, CMake install/export metadata | `make quality-review-cmake` or direct configure/build/CTest; install proof when package metadata changes | Build/package owner | Proves local CMake path only unless hosted CI confirms platform claims. |

## Package And Install Matrix

| Surface | Trigger | Command Or Review | Owner | Evidence Boundary |
| --- | --- | --- | --- | --- |
| Make install and `pkg-config` proof | Static-first install/package wording, install scripts, `sparse.pc.in`, package report rows | `bash tests/test_install.sh` | Package owner | Proves local Make install, uninstall, installed headers, static archive, and `pkg-config` consumer behavior. |
| CMake install/export proof | CMake package wording, CMake export metadata, downstream CMake examples | `bash tests/test_cmake_install.sh` | Package owner | Proves local CMake install/export and downstream `find_package(Sparse)` behavior. |
| Static package deferral guard | Package/ABI wording or metadata that might imply shared library support | `bash scripts/static_package_deferral_check.sh` | Package owner | Confirms unsupported shared-library, ABI, loader, selector, and package-manager wording remains blocked. |
| Windows install/downstream support | Windows CMake install/downstream claim or support-tier wording | Hosted Windows supplemental/reviewed workflow evidence | Platform/package owner | Local Unix proof cannot establish Windows support; Windows Makefile and Windows `pkg-config` remain non-claims. |

## Corpus And Report Matrix

| Surface | Trigger | Command Or Review | Owner | Evidence Boundary |
| --- | --- | --- | --- | --- |
| Corpus schema | Fixture, expected row, manifest, generator, or report-family metadata edits | `python3 scripts/validate_corpus_schema.py` | Corpus owner | Checks source-controlled corpus metadata structure and required fields. |
| QR and partial-SVD proof-owner tests | Solver corpus code or claim-bearing corpus docs | Focused corpus test binaries, then full C gate if `.c`/`.h` changed | Solver/corpus owner | Proves named fixture families only. |
| Selected oracle freshness | Claim-bearing oracle report rows for QR or partial-SVD | `make report-index-oracle-freshness` | Report owner | Regenerates selected local oracle output and checks selected generated freshness. |
| Report-index structure | Report metadata/doc changes or final closeout audit | `python3 scripts/normalize_report_index.py --family corpus --family oracle --family comparison --check` | Report owner | Checks normalized structure; does not make stale rows fresh. |
| Source-controlled report rows | Checked-in report metadata without generated rerun | Review row status and family semantics | Report owner | Source-controlled rows are inventory/provenance evidence, not fresh pass evidence. |

## Comparison Matrix

| Surface | Trigger | Command Or Review | Owner | Evidence Boundary |
| --- | --- | --- | --- | --- |
| Comparison harness self-check | Comparison script or dependency-policy claim | `python3 scripts/run_external_comparison.py --self-check` | Comparison/tooling owner | Proves harness argument/schema behavior only. |
| Selected `qr-minnorm` comparison freshness | Any claim citing Sprint 154 comparison evidence | `make report-index-comparison-freshness` | Comparison/report owner | Proves one local generated comparison family for `qr_underdetermined_minnorm_2x4`. |
| External dependency availability | Comparison output or study citation | Review `build/comparison/qr_minnorm/dependency_status.tsv` after the freshness command | Comparison owner | Dependency absence is a skip/defer state, not pass evidence. |
| Comparison wording | README, tutorial, cookbook, solver-selection, maintainer guide, API docs, or retrospective text cites comparison | Claim scan for broad parity/performance wording | Documentation owner | Supports only the named dependency/version/fixture/metric/tolerance/platform actually recorded. |

## Platform And CI Matrix

| Surface | Trigger | Command Or Review | Owner | Evidence Boundary |
| --- | --- | --- | --- | --- |
| Linux reviewed lanes | Final platform support or package claim | Review current GitHub Actions Linux job definitions and final PR/branch run outcomes | Platform owner | Hosted evidence is required for reviewed Linux claims. |
| macOS reviewed lanes | macOS static-first install/export or package claim | Review macOS workflow and final PR/branch run outcomes | Platform owner | Hosted evidence is required for reviewed macOS claims. |
| Windows reviewed CMake lane | Windows reviewed CMake test support or CTest count | Review Windows workflow, CTest count, staged exclusions, and final run outcomes | Platform owner | Proves only reviewed MSVC CMake-first test support. |
| Windows install/downstream lane | Windows CMake install/downstream support wording | Review supplemental/reviewed Windows install/downstream job outcome | Platform/package owner | Does not prove Windows Makefile or Windows `pkg-config` execution parity. |
| External service outage | CI setup or action resolution fails before repository commands run | Record outage separately from repository validation | Platform owner | Service outage cannot be counted as pass or fail evidence for code behavior without a rerun. |

## Public Claim Audit Matrix

| Surface | Trigger | Command Or Review | Owner | Evidence Boundary |
| --- | --- | --- | --- | --- |
| Public docs | Day 10 final audit | Search README, INSTALL, tutorial, cookbook, solver-selection, maintainer guide, benchmark docs, API reference, and public headers for claim terms | Documentation owner | Every claim must map to Day 2 evidence or become an explicit non-claim/residual. |
| Package/ABI wording | Static-first, shared-library, ABI, package-manager, loader, selector wording | Static deferral guard plus manual docs scan | Package owner | Static archive package support is the only package product claim unless later evidence exists. |
| Performance wording | Benchmark, sentinel, comparison, or backend-superiority wording | Manual docs scan and benchmark/report boundary review | Benchmark owner | Local benchmark/comparison rows are not portable performance proof. |
| State-of-practice wording | Any "state-of-the-art", "parity", "competitive", or external-library wording | Manual docs scan against comparison and corpus evidence | Documentation/numerical owner | Only narrow named comparison evidence can support narrow state-of-practice language. |

## Skip And Defer Policy

| Case | Required Label | Required Record |
| --- | --- | --- |
| No `.c` or `.h` files changed | `C gate not required` | State that final diff is docs-only and `git diff --check` passed. |
| `.c` or `.h` files changed but full C gate fails | `Blocked` | Stop with failing command and first actionable diagnostic. Do not commit closeout as passed. |
| Hosted CI unavailable or service outage | `Unavailable hosted evidence` | Record workflow/job name, timestamp, failure phase, and why local proof cannot substitute. |
| Optional external dependency absent | `Skipped optional dependency` | Record dependency status output and keep comparison claim out of pass evidence. |
| Generated rows stale or missing | `Stale generated evidence` | Record normalizer/freshness output and preserve non-claim/residual wording. |
| Generated API HTML not refreshed | `Deferred generated reference refresh` | Cite Sprint 155 publication policy and carry residual with refresh gate. |
| Windows staged blocker remains | `Staged platform residual` | Record blocker and promotion gate; do not count it as reviewed Windows support. |
| Source-controlled report rows without regeneration | `Inventory evidence only` | Treat rows as metadata/provenance, not fresh generated proof. |

## Escalation Rules

- Stop if `make format && make lint && make test` fails after `.c` or `.h`
  changes.
- Stop if a package, report, corpus, comparison, or platform check needed for a
  public claim fails and the claim cannot be narrowed cleanly.
- Stop if hosted evidence contradicts local evidence or if CTest count drift is
  unexplained.
- Stop if any public text implies state-of-the-art, ecosystem parity,
  package-manager support, shared-library support, dynamic ABI support,
  runtime-loader behavior, broad Windows parity, or portable performance
  without direct evidence.
- Stop if generated rows are stale but needed as pass evidence.
- Stop if review feedback conflicts with an existing support boundary.

## Day 4 Command Handoff

Day 4 should run the local baseline in this order unless the branch changes
before Day 4:

1. `git status --short --branch`
2. `git diff --check`
3. inspect changed file classes with `git diff --name-only master...HEAD`
4. if `.c` or `.h` changed in the final Sprint 156 delta, run
   `make format && make lint && make test`
5. if broad local build/CMake confidence is selected for final closeout, run
   `make quality-review-full`
6. record environment, command outputs, failures, skips, and deferrals in the
   Day 4 local baseline artifact

## Day 3 Completion Check

- Mandatory docs-only checks are defined.
- Full C quality-gate triggers are defined.
- Package, install, CMake, `pkg-config`, and downstream-consumer checks are
  mapped.
- Corpus and generated-report checks are mapped.
- Comparison-harness checks are mapped.
- Hosted platform evidence remains separate from local proof.
- Skip/defer labels and escalation rules are explicit.
