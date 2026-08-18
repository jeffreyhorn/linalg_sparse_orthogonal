# Sprint 166 Day 4: Validation Baseline Design

## Purpose

Day 4 turns the Day 2 and Day 3 evidence inventories into a validation design
for Sprint 166. The design selects the strongest feasible local command set,
classifies supplemental checks by touched surface and cost, and separates local
proof from hosted-only evidence before Day 5 and Day 6 execute validation.

## Source Inputs

| Source | Day 4 use |
| --- | --- |
| `docs/planning/EPIC_14/SPRINT_166/PLAN.md` | Sprint 166 Day 4 requirements and completion criteria. |
| `docs/planning/EPIC_14/SPRINT_166/artifacts/day2-generated-report-evidence-inventory.md` | Generated API, oracle, comparison, and report-index evidence boundaries. |
| `docs/planning/EPIC_14/SPRINT_166/artifacts/day3-solver-package-performance-api-inventory.md` | Solver, package, performance, public-header/API, and static-first evidence boundaries. |
| `Makefile` | Local command ownership for C, docs, report, package, benchmark, performance, and coverage checks. |
| `tests/test_install.sh` | Unix-like Make install and `pkg-config` package proof. |
| `tests/test_cmake_install.sh` | Unix-like CMake install/export and downstream proof. |
| `scripts/static_package_deferral_check.sh` | Static-first package boundary and unsupported package/ABI wording guard. |
| `.github/workflows/ci.yml` | Linux reviewed static package and hosted oracle/comparison evidence. |
| `.github/workflows/windows-ci.yml` | Windows reviewed CMake build/test/install evidence and non-claim wording. |

## Validation Semantics

| Evidence class | Meaning | Sprint 166 handling |
| --- | --- | --- |
| Hard local gate | A local command failure blocks claim closeout unless fixed or explicitly narrowed. | Run on Day 5 or Day 6 before public claim audit. |
| Touched-surface gate | Required only when the related file family changed. | Apply the Sprint 157 quality-surface rule: `.c`/`.h` changes require `make format && make lint && make test`. |
| Supplemental proof | Local proof for selected docs/report/package/performance surfaces. | Run when the surface is part of the final evidence claim or changed in Sprint 166. |
| Source-controlled advisory evidence | Checked-in metadata describing ownership, support tier, claim scope, or expected rows. | Validate for consistency, but do not treat as generated pass evidence. |
| Local-only generated evidence | Generated in `build/`, `docs/api/`, or temporary install trees on the local machine. | May support local freshness claims only. |
| Hosted-only evidence | Evidence produced by GitHub Actions on Linux or Windows. | Cite only from workflow definitions and reviewed CI results; do not imply local reproduction. |
| Non-claim guard | A scan or validation rule that prevents unsupported claim drift. | Passing preserves boundaries; it does not add support for deferred products. |

## Final Local Baseline Command Matrix

| Command | Evidence surface | Class | Run day | Required when | Notes |
| --- | --- | --- | --- | --- | --- |
| `make format` | C/source/header formatting | Hard local/touched-surface gate | Day 5 | Always for final baseline, and required if `.c`/`.h` files changed. | Mutates files; inspect worktree afterward. |
| `make lint` | strict warnings, clang-tidy, cppcheck | Hard local/touched-surface gate | Day 5 | Always for final baseline, and required if `.c`/`.h` files changed. | Requires local tooling. |
| `make test` | compiled C test suite | Hard local/touched-surface gate | Day 5 | Always for final baseline, and required if `.c`/`.h` files changed. | Local platform proof only. |
| `git diff --check` | whitespace hygiene | Hard local gate | Day 5 and after edits | Any Sprint 166 artifact or code/doc change. | Cheap and repeatable. |
| `python3 scripts/validate_corpus_schema.py` | corpus schema metadata | Hard local gate | Day 5 | Final baseline and any corpus/report metadata changes. | Validates checked-in corpus schema files. |
| `python3 tests/test_normalize_report_index.py` | report-index row-state behavior | Hard local/report gate | Day 5 | Final baseline and report-index changes. | No committed generated output. |
| `python3 tests/test_run_external_comparison.py` | external comparison target generation | Hard local/report gate | Day 5 | Final baseline and comparison-runner changes. | Uses temporary generated output. |
| `python3 -m py_compile scripts/normalize_report_index.py scripts/run_external_comparison.py scripts/run_corpus_oracle.py` | Python syntax for report owners | Supplemental local gate | Day 5 | Script changes or final report baseline. | Cheap syntax proof. |

Day 5 should run the full C gate even if Sprint 166 has only touched planning
artifacts so far, because Sprint 166 is the final Epic 14 closeout sprint and
needs a fresh local baseline before claim recalibration.

## Supplemental Validation Command Matrix

| Command | Evidence surface | Class | Run day | Required when | Claim boundary |
| --- | --- | --- | --- | --- | --- |
| `make docs-check` | generated API HTML and page coverage | Supplemental local generated proof | Day 6 | API docs, public headers, Doxygen, or final API evidence changed. | Local-only generated API proof; no hosted docs publication. |
| `make api-docs-coverage` | generated API page coverage | Supplemental local generated proof | Day 6 | If `make docs-check` output needs separate diagnosis. | Checked-in public-header input set only. |
| `make report-index-oracle-freshness` | selected oracle freshness | Supplemental local generated proof | Day 6 | Final oracle/report claims or report tooling changed. | Fixture-local selected oracle rows only. |
| `make report-index-comparison-freshness` | selected comparison freshness | Supplemental local generated proof | Day 6 | Final comparison claims or comparison tooling changed. | Local selected QR and partial-SVD comparison freshness unless hosted artifacts match. |
| `python3 scripts/normalize_report_index.py --check` | broad report-index consistency | Advisory/source-controlled check | Day 6 | Final closeout diagnostics. | Broad check is navigation/metadata consistency, not hosted proof. |
| `python3 scripts/normalize_report_index.py --family package --check` | package report-index rows | Advisory/source-controlled check | Day 6 | Package evidence changes or final package baseline. | Advisory proof-owner rows only. |
| `python3 scripts/normalize_report_index.py --family package --check-freshness` | package report-index freshness | Advisory/source-controlled check | Day 6 | Package evidence changes or final package baseline. | Source-controlled package rows, not package-manager support. |
| `bash scripts/static_package_deferral_check.sh` | package/ABI non-claim guard | Hard local non-claim guard | Day 6 | Package, docs, workflow, CMake, or public-header claim surfaces changed. | Preserves static-first boundaries; does not add shared-library support. |
| `bash tests/test_install.sh` | Make install and `pkg-config` proof | Supplemental local package proof | Day 6 | Package metadata/install changes or final package baseline. | Unix-like static archive package proof only. |
| `bash tests/test_cmake_install.sh` | CMake install/export proof | Supplemental local package proof | Day 6 | CMake package changes or final package baseline. | Static archive package metadata; exact version is not dynamic ABI. |
| `make bench-canonical-report` | canonical benchmark publication rows | Supplemental local performance proof | Day 6 | Performance docs/report changes or final performance baseline. | Local threshold-free measurements; no portable performance or superiority. |
| `make performance-sentinels` | performance sentinel rows | Supplemental local performance proof | Day 6 | Performance docs/report changes or final performance baseline. | S5 hard local wall-check only; S2/S3 threshold-free context rows. |

## Hosted-Only Evidence Split

| Hosted evidence | Owner | How Sprint 166 may use it | Boundary |
| --- | --- | --- | --- |
| Linux reviewed static-first package contract | `.github/workflows/ci.yml`, job `package-contract` | Cite reviewed CI results for Linux Make install/`pkg-config`, CMake install/export, and static deferral proof. | Linux hosted package proof; no package-manager, shared-library, dynamic ABI, runtime-loader, or broad platform claim. |
| Linux reviewed hosted oracle/comparison freshness | `.github/workflows/ci.yml`, job `generated-report-freshness` | Cite reviewed CI results for selected oracle freshness and whatever comparison artifacts are actually uploaded. | Day 7 must reconcile current multi-family comparison command with QR-minnorm-only artifact naming/content. |
| Windows reviewed CMake consumer subset | `.github/workflows/windows-ci.yml`, job `build-and-test` | Cite hosted MSVC configure/build/CTest count/full CTest evidence. | CMake-first Windows test proof only; no Makefile parity or broad Windows parity. |
| Windows reviewed CMake install/downstream validation | `.github/workflows/windows-ci.yml`, job `install-and-downstream` | Cite hosted static `.lib`, headers, CMake metadata, downstream consumers, version checks, no-DLL, and metadata-only `sparse.pc` inspection. | No Windows `pkg-config` command execution parity, shared-library support, dynamic ABI, runtime-loader behavior, or package-manager support. |

Hosted evidence must be cited with the platform, workflow job, and exact scope.
Local Day 5/6 command results cannot replace Windows-hosted MSVC evidence, and
workflow definitions alone cannot prove that a particular hosted run passed.

## Touched-Surface Rules

| Changed surface | Required checks before claim closeout |
| --- | --- |
| `.c` or `.h` files | `make format && make lint && make test`, plus any focused tests for the touched solver or API surface. |
| Makefile/CMake/package metadata/install scripts | `bash scripts/static_package_deferral_check.sh`, `bash tests/test_install.sh`, `bash tests/test_cmake_install.sh`, plus relevant hosted CI citation after push/PR. |
| report-index scripts/manifests/comparison/oracle docs | `python3 tests/test_normalize_report_index.py`, `python3 tests/test_run_external_comparison.py`, selected `make report-index-*` freshness targets, and corpus schema validation. |
| generated API docs/Doxygen/public API docs | `make docs-check`, `make api-docs-coverage`, and targeted scans for hosted-docs/public-header overclaims. |
| performance benchmark/sentinel scripts/docs | `make bench-canonical-report`, `make performance-sentinels`, report-index normalization, and scans for portable-performance/backend-superiority wording. |
| public docs and planning artifacts only | `git diff --check` plus targeted claim/reference scans. |
| CI workflows | Relevant local command owner where possible, plus Day 7 hosted-only evidence reconciliation after CI runs. |

## Claim Scan Plan

Day 6 should run targeted scans over public docs, workflows, package metadata,
and Sprint 166 artifacts before Day 8/9 claim audit. Suggested patterns:

```sh
rg -n "state.of.the.art|best.in.class|faster than|outperform|portable performance|backend superiority" README.md INSTALL.md docs .github
rg -n "shared library|shared-library|dynamic ABI|ABI compatibility|runtime loader|SONAME|install_name|RPATH|DLL|package-manager|Homebrew|apt|dnf|pacman|vcpkg|Conan" README.md INSTALL.md docs .github CMakeLists.txt sparse.pc.in
rg -n "Windows Makefile|Windows pkg-config|pkg-config execution|broad Windows|platform parity" README.md INSTALL.md docs .github
rg -n "hosted.*comparison|sprint159-comparison|qr_minnorm|qr_compatible_ls|partial_svd_diag6_k2" .github/workflows docs/planning/EPIC_14/SPRINT_166
```

These scans are claim-boundary checks, not proof that the selected solver,
package, API, or performance surfaces are complete.

## Validation Risk Register

| Risk | Impact | Day 4 disposition |
| --- | --- | --- |
| Hosted comparison artifact scope lags current comparison freshness command. | Public hosted comparison claims could overstate QR compatible LS or partial-SVD hosted proof. | Carry to Day 7 CI reconciliation before hosted comparison claims are finalized. |
| Full local C gate may be expensive or tooling-dependent. | Day 5 could block on local environment rather than code behavior. | Treat failures as blockers unless clearly environmental and documented. |
| `make format` mutates source/test/header files. | It can create unrelated formatting diffs. | Inspect `git status` and `git diff` immediately after running. |
| Benchmark/sentinel rows are environment-sensitive. | Local timings can vary and should not become portable claims. | Keep performance evidence methodology-bound and non-superiority. |
| Install tests create temporary install/build trees. | Generated artifacts must not be committed. | Check worktree after package validation. |
| Advisory report-index rows may be mistaken for pass evidence. | Closeout could overclaim source-controlled metadata. | Label advisory/source-controlled rows separately in Day 5/6 records. |
| Windows package evidence cannot be reproduced locally on macOS. | Local closeout could omit hosted MSVC constraints. | Cite workflow/job evidence only after hosted CI result is available. |

## Day 5 Execution Design

Run the fresh final local baseline first:

1. `make format`
2. inspect `git status --short` and any formatting diffs
3. `make lint`
4. `make test`
5. `python3 scripts/validate_corpus_schema.py`
6. `python3 tests/test_normalize_report_index.py`
7. `python3 tests/test_run_external_comparison.py`
8. `python3 -m py_compile scripts/normalize_report_index.py scripts/run_external_comparison.py scripts/run_corpus_oracle.py`
9. `git diff --check`

If any command fails, stop Day 5 follow-on claim work until the failure is
fixed, narrowed with evidence, or reported as a blocker.

## Day 6 Execution Design

After the Day 5 baseline passes, run touched-surface supplemental checks:

1. `make docs-check`
2. `make report-index-oracle-freshness`
3. `make report-index-comparison-freshness`
4. `python3 scripts/normalize_report_index.py --check`
5. `python3 scripts/normalize_report_index.py --family package --check`
6. `python3 scripts/normalize_report_index.py --family package --check-freshness`
7. `bash scripts/static_package_deferral_check.sh`
8. `bash tests/test_install.sh`
9. `bash tests/test_cmake_install.sh`
10. `make bench-canonical-report`
11. `make performance-sentinels`
12. targeted claim scans from the claim scan plan
13. `git diff --check`
14. `git status --short --branch`

If a supplemental command is skipped for local environment reasons, record the
exact missing tool or platform constraint and keep the corresponding claim
local-only, hosted-only, advisory, or residualized as appropriate.

## Completion Check

| Criterion | Status | Evidence |
| --- | --- | --- |
| Final validation scope is explicit before command execution. | Complete | Day 5 and Day 6 command designs enumerate hard, supplemental, advisory, local-only, and hosted-only evidence. |
| Required checks are tied to touched surfaces. | Complete | Touched-surface rules map `.c`/`.h`, package, report, generated API, performance, docs, and CI changes to commands. |
| Hosted-only evidence is not confused with local proof. | Complete | Hosted-only evidence split identifies Linux and Windows workflow jobs and preserves non-claim boundaries. |
