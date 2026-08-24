# Sprint 177 Day 10: Quality Surface Map

**Sprint:** 177 - Epic 16 Baseline, Evidence Matrix & Closure Gates
**Epic source:** `docs/planning/EPIC_16/PROJECT_PLAN.md`
**Requested sprint path:** `docs/planning/EPIC_15/SPRINT_177/`
**Status:** Complete

## Purpose

Map required validation commands by change surface so later Epic 16 sprints
can select quality checks consistently. This artifact complements the Day 8-9
acceptance gates: the gates define what closure means, and this map defines
which commands must run for each kind of change.

## Baseline Rule

When any C source or public/internal header file changes, run:

```bash
make format && make lint && make test
```

This rule applies even when the primary task is documentation, workflow,
package, report, or governance work and the C/header change appears small.

## Validation Command Matrix

| Change surface | Examples | Required validation | Additional focused validation |
| --- | --- | --- | --- |
| Documentation-only | Sprint artifacts, README, INSTALL, maintainer guide, tutorial, cookbook, benchmark docs | `git diff --check` | Targeted grep/readback for protected non-claims when claim wording changes |
| Sprint planning artifacts only | `docs/planning/EPIC_*/SPRINT_*/artifacts/*.md`, `WORKING_NOTES.md`, retrospectives | `git diff --check` | Confirm source-path notes and artifact links if path mismatch exists |
| Public claim wording | README, INSTALL, maintainer guide, workflow comments, benchmark docs | `git diff --check` | Relevant deferral guard if package, ABI, Windows, report, or API wording changes |
| Python/report tooling | `scripts/*.py`, `tests/test_*.py`, report manifests | Python compile check for changed scripts/tests; focused Python tests; `git diff --check` | `python3 tests/test_normalize_report_index.py`, `python3 tests/test_run_external_comparison.py`, or workflow guard tests as relevant |
| Shell tooling | `scripts/*.sh`, `tests/*.sh`, Make shell recipes | `bash -n` for changed shell scripts; focused script tests; `git diff --check` | Install, package, benchmark, or report freshness target that owns the shell path |
| Workflow YAML | `.github/workflows/*.yml` | Focused workflow guard tests; `git diff --check` | YAML block-scope checks, exact artifact upload fail-closed checks, platform count checks |
| Makefile/build registration | `Makefile`, `build-metadata/library_sources.txt` | Relevant target dry/run checks; `make source-list-check`; `git diff --check` | Full C quality gates if C/header or test registration changes |
| CMake registration | `CMakeLists.txt`, CMake package templates | CMake configure/build/CTest checks where feasible; `git diff --check` | `make quality-review-cmake-compile` or `make quality-review-cmake`; Windows count updates if tests change |
| Package/install metadata | `sparse.pc.in`, `cmake/SparseConfig.cmake.in`, install rules, package docs | Install checks; package deferral checks; `git diff --check` | `bash tests/test_install.sh`, `bash tests/test_cmake_install.sh`, `bash scripts/static_package_deferral_check.sh`, `bash scripts/package_manager_deferral_check.sh` |
| Generated API docs | `Doxyfile`, public-header comments, API docs scripts | `make docs-check`; `make api-docs-freshness`; `git diff --check` | Full C quality gates if headers or C files change |
| Public headers | `include/*.h`, generated version template | `make format && make lint && make test`; docs checks; `git diff --check` | Declaration baseline/checksum guard if declarations are reorganized |
| C implementation | `src/*.c`, `src/*.h` | `make format && make lint && make test`; `git diff --check` | Focused solver/test target, CMake quality checks if registration changes |
| Tests in C | `tests/*.c`, test helpers | `make format && make lint && make test`; `git diff --check` | Focused test binary during development; update CMake/Windows count if registration changes |
| Benchmarks | `benchmarks/*.c`, benchmark scripts/docs | `make format && make lint && make test` if C changes; `git diff --check` | `make bench-build`, `make bench-canonical-report-freshness`, or sentinel target as relevant |
| Examples | `examples/*.c`, `examples/cmake_example/*` | Full C gates if C changes; `git diff --check` | `make examples-build` or installed downstream proof if package example changes |

## Epic 16 Target-Specific Quality Map

| Sprint | Target | Minimum validation surface |
| --- | --- | --- |
| 178 | Allocation-failure proof batch 2 | Focused allocation-failure target plus `make format && make lint && make test`; CMake/CTest checks if registration changes. |
| 179 | Generated API HTML status | `make docs-check`, `make api-docs-freshness`, staging/publication guard checks, and workflow checks if hosted/artifact publication changes. |
| 180 | Package-manager provider decision | Provider proof or deferral script, static package deferral check, package-manager deferral check, install checks if metadata changes, and full C gates if code changes. |
| 181 | Selected report target manifest | Python compile checks, report-index tests, selected workflow guard tests, selected freshness targets, and exact duplicate/missing-row checks. |
| 182 | Windows report freshness decision | Workflow guard tests, selected report checks where feasible, PowerShell/YAML review, manifest integration tests, and package/platform wording guards. |
| 183 | Additional comparison family | Comparison freshness, comparison runner tests, report-index tests, Python compile checks, relevant C tests if solver behavior changes. |
| 184 | Public header coherence batch 3 | `make format && make lint && make test`, `make docs-check`, `make api-docs-freshness`, and declaration-preservation checks. |
| 185 | Review-surface reduction | Full C quality gates for C/header changes, source-list checks for new library files, CMake checks for registration changes, and affected focused tests. |
| 186 | Final claim calibration | Quality checks selected from all changed surfaces, plus package checks, report checks, generated API checks, workflow guards, and `git diff --check`. |

## Focused Command Owners

| Command | Use when |
| --- | --- |
| `git diff --check` | Always before closing a documentation or planning day; also after any code/script/workflow edit. |
| `make format` | Any C source, C test, benchmark, example, or header change. |
| `make lint` | Any C/header change and before claiming reviewed local C quality. |
| `make test` | Any C/header/test behavior change. |
| `make quality-review-compile` | Need reviewed compile/source-list/lint wrapper without running all tests separately. |
| `make quality-review` | Need reviewed Makefile path including tests and dead-code check. |
| `make quality-review-cmake-compile` | CMake configure/build/CTest registration needs validation without full CTest run. |
| `make quality-review-cmake` | CMake registration, install/export, or cross-build behavior needs reviewed CMake execution. |
| `make source-list-check` | Library source list or build metadata changes. |
| `make docs-check` | Doxygen/API docs coverage or public header documentation changes. |
| `make api-docs-freshness` | Generated API HTML status, Doxygen inputs, generated API local-only policy, or public API docs navigation changes. |
| `make report-index-oracle-freshness` | Selected QR/partial-SVD oracle report target or freshness changes. |
| `make report-index-comparison-freshness` | Selected comparison target, fixture, expected row, or comparison report changes. |
| `make bench-canonical-report-freshness` | Selected canonical benchmark report/freshness methodology changes. |
| `make performance-sentinels` | Local performance sentinel scripts or sentinel rows change. |
| `make iterative-allocation-failure-gate` | Iterative allocation-failure proof changes or as a pattern check for new allocation-failure work. |
| `bash tests/test_install.sh` | Make install, Unix pkg-config, installed headers, or `.pc` metadata changes. |
| `bash tests/test_cmake_install.sh` | CMake install/export, `find_package`, exact version, or CMake package metadata changes. |
| `bash scripts/static_package_deferral_check.sh` | Static-first, shared-library, ABI, loader, or package metadata wording changes. |
| `bash scripts/package_manager_deferral_check.sh` | Package-manager provider wording, recipes, or metadata templates change. |
| `python3 tests/test_normalize_report_index.py` | Report-family metadata, normalized report behavior, freshness policy, or non-claim rows change. |
| `python3 tests/test_selected_comparison_workflow.py` | Selected comparison workflow guard, upload-artifact block, or workflow target metadata changes. |
| `python3 tests/test_run_external_comparison.py` | External comparison runner, fixtures, reference helpers, or comparison rows change. |

## Review Trap Checks

Use these trap checks when the related surface changes:

| Trap | Required check |
| --- | --- |
| Workflow guard can be bypassed | Ensure guard runs outside the lane it validates or document why independence is impossible. |
| Upload fail-closed check is too broad | Assert `if-no-files-found: error` inside the exact selected upload block. |
| Duplicate manifest/report rows silently overwrite | Add explicit duplicate detection before building keyed maps. |
| Missing row failure is unclear | Assert row presence with a clear message before indexing or `next(...)`. |
| Windows CTest count drifts | Update expected count and explain new/removed tests in Windows workflow comments. |
| Package wording weakens static-first boundary | Run static/package deferral checks and verify README/INSTALL/maintainer wording. |
| Allocation-failure wording drifts | Use "allocation-failure" consistently across target, label, docs, and artifacts. |
| Public error-contract order changes | Validate NULL-handle and invalid-argument ordering in tests and docs. |
| Generated docs are accidentally staged | Run `make api-docs-freshness` and local-only staging guard. |
| Source registration drifts | Run `make source-list-check` and CMake registration checks as applicable. |

## Documentation-Only Policy

Documentation-only changes do not require `make format && make lint &&
make test` unless they also modify C source or header files. They still require
`git diff --check`, and claim-bearing docs changes should run the relevant
focused guard when the docs touch package-manager, static-first, ABI, Windows,
generated API, report freshness, or benchmark support wording.

## Completion Criteria Check

| Criterion | Status | Evidence |
| --- | --- | --- |
| Later sprint days can select validation from the map | Complete | Matrix maps change surfaces, commands, and target-specific checks. |
| C/header quality-gate requirement is explicit | Complete | Baseline rule requires `make format && make lint && make test` for C/header changes. |
| Docs-only and script-only changes have focused validation | Complete | Documentation-only policy and Python/shell rows define scoped validation without over-running unrelated checks. |
