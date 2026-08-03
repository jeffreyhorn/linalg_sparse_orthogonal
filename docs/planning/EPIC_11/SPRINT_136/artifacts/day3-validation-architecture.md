# Sprint 136 Day 3 - Validation Architecture

## Purpose

Day 3 defines the validation architecture for final Epic 11 closeout. It
classifies validation lanes, maps command owners, records local versus hosted
execution boundaries, and defines stop conditions before Day 4 turns this
architecture into exact command sequences.

This artifact does not run validation. It prevents later validation evidence
from being overread: reviewed, supplemental, local, staged, deferred, and
unsupported lanes keep their inherited support-tier meanings.

## Validation Lane Matrix

| Lane | Owner surfaces | Tier | Day 4 command-plan role | Claim boundary |
| --- | --- | --- | --- | --- |
| Documentation hygiene | Sprint 136 artifacts and touched Markdown docs | Reviewed local hygiene | Always required for touched docs. | Whitespace/diff cleanliness is not behavioral evidence. |
| Public-doc link/path checks | README, INSTALL, docs, examples, benchmarks | Reviewed local when public docs change | Required if public docs change; focused to touched surfaces. | Link validity does not prove claim support. |
| Claim-boundary scan | README, INSTALL, docs, examples, benchmarks, maintainer guide | Reviewed local for claim cleanup | Required before and after public/support wording edits. | Finds wording drift; does not create positive claims. |
| C/header quality gate | `src/`, `include/`, `tests/*.c`, helper headers | Reviewed local | Required if any `.c` or `.h` changes: `make format && make lint && make test`. | Passing default tests does not prove broad external parity, performance, or platform parity. |
| Source-list/build-registration checks | Makefile, CMakeLists, scripts, source inventory | Reviewed local when build/source lists change | Required if source, test, example, benchmark, or build registration changes. | Registration parity is structural evidence only. |
| CMake configure/build/test registration | `CMakeLists.txt`, CMake package files, tests | Reviewed local for CMake-touched surfaces | Required if CMake/build support is touched; optional confidence otherwise. | Local CMake proof is platform-local. |
| Make install and `pkg-config` proof | `tests/test_install.sh`, Make install, `sparse.pc.in` | Reviewed local package proof; Linux hosted reviewed CI owner | Required if package/install/pkg-config surfaces change; optional final confidence otherwise. | Static-first package evidence only. |
| CMake install/export proof | `tests/test_cmake_install.sh`, `examples/cmake_example`, CMake config | Reviewed local package proof; Windows/macOS hosted supplemental confidence | Required if CMake package/export surfaces change; optional final confidence otherwise. | Static CMake consumer evidence only. |
| Static package deferral proof | `scripts/static_package_deferral_check.sh`, CMake, docs | Reviewed local package proof | Required if package/ABI/shared-library wording or behavior changes. | Confirms deferral boundaries, not shared support. |
| Canonical benchmark report | `make bench-canonical-report`, benchmark scripts/docs | Local supplemental report evidence | Optional final evidence if performance/benchmark wording is audited or changed. | Threshold-free local snapshot only. |
| Performance sentinels | `make performance-sentinels`, wall-check, sentinel scripts | Local supplemental report evidence with existing wall-check gate | Optional final evidence if performance/sentinel wording is audited or changed. | Local sentinel context; no portable performance guarantee. |
| Large-matrix guardrails | `make large-matrix-guardrails`, guardrail scripts | Mixed reviewed/supplemental report evidence by row | Optional final evidence if guardrail/report wording is audited or changed. | Structural/report guardrail evidence only. |
| Dead-code report/check | `make deadcode-report`, `make deadcode-check` | Reviewed report-completeness gate; serialized local workflow | Optional unless source/public-surface cleanup needs dead-code context. | Not zero-findings or removal-ready proof. |
| Coverage | `make coverage`, coverage CI | Supplemental and tree-mutating | Deferred unless coverage wording or report architecture explicitly requires refresh. | Coverage percentage is not behavioral completeness. |
| Linux CI package lane | `.github/workflows/ci.yml` | Hosted reviewed package-contract lane | Cannot be proven locally; Day 4 may inspect workflow and cite hosted requirement. | Reviewed only when hosted CI passes. |
| macOS package lanes | `.github/workflows/macos-ci.yml` | Hosted supplemental confidence | Cannot be promoted locally; Day 4 may inspect workflow and cite hosted requirement. | Supplemental install/export confidence only. |
| Windows install/downstream lane | `.github/workflows/windows-ci.yml` | Hosted supplemental confidence | Cannot be promoted locally; Day 4 may inspect workflow and cite hosted requirement. | Supplemental CMake-first confidence only. |
| Windows staged pthread/POSIX tests | CMake, Windows CI, staged test sources | Staged | Exclude from reviewed validation unless portability work lands. | No reviewed Windows coverage claim. |
| Shared library / dynamic ABI / package manager | CMake, package metadata, docs | Unsupported/deferred | Validation should confirm non-claims, not execute support proof. | No shared-library, dynamic ABI, runtime-loader, or package-manager claim. |

## Command Ownership Map

| Command or check family | Owner | Runs locally? | Hosted dependency |
| --- | --- | --- | --- |
| `git diff --check` | Repository hygiene | Yes | None. |
| Sprint 136 trailing-whitespace scan | Sprint 136 planning docs | Yes | None. |
| Focused public-doc trailing-whitespace scan | Touched public docs | Yes | None. |
| Focused markdown link/path script or shell scan | Touched public docs | Yes | None. |
| Claim-boundary `rg` scans | Public/support docs and artifacts | Yes | None. |
| `git diff --name-only -- '*.c' '*.h'` | C quality-gate decision | Yes | None. |
| `make format && make lint && make test` | Reviewed C/header quality gate | Yes | None for local proof. |
| `make quality-review` | Reviewed local compile/test/dead-code owner | Yes, if needed and environment supports dependencies | None for local proof. |
| `cmake -S . -B ...` plus `ctest -N`/`ctest` | CMake registration and build proof | Yes | Hosted platform parity still depends on CI. |
| `bash tests/test_install.sh` | Make install/`pkg-config` package proof | Yes on Unix-like local hosts | Hosted Linux reviewed lane for CI claim. |
| `bash tests/test_cmake_install.sh` | CMake install/export package proof | Yes if local CMake/build environment supports it | Hosted Windows/macOS lanes for platform confidence. |
| `bash scripts/static_package_deferral_check.sh` | Static-first deferral guard | Yes | None. |
| `make bench-canonical-report` | Canonical benchmark report generator | Yes, runtime dependent | None; local-only evidence. |
| `make performance-sentinels` | Sentinel report and wall-check owner | Yes, runtime dependent | None; local-only evidence. |
| `make large-matrix-guardrails` | Guardrail report owner | Yes, data/runtime dependent | Optional large/supplemental rows may depend on local data/env. |
| `make deadcode-report` and `make deadcode-check` | Dead-code report owner | Yes, dependency dependent, serialized | CI dead-code lane can provide hosted signal. |
| `make coverage` | Coverage owner | Yes, tree-mutating and dependency dependent | Supplemental CI coverage lane. |
| Hosted Linux/macOS/Windows workflow execution | CI owner workflows | No direct local equivalent | Requires GitHub-hosted runners. |

## Validation Requirements By Touched Surface

| Touched surface | Required validation | Optional confidence |
| --- | --- | --- |
| Sprint 136 planning artifacts only | `git diff --check`; focused trailing-whitespace scan over `docs/planning/EPIC_11/SPRINT_136`; confirm no `.c`/`.h` changes. | Required-section scan for new daily artifact structure. |
| Public docs | Diff/whitespace checks; focused link/path checks; claim-boundary scans for package, platform, performance, report, external parity, and competitive wording. | Search inbound links if files move or headings change. |
| Maintainer guide | Public-doc checks plus support-tier and non-claim scan against Sprint 131-135 handoffs. | Compare against final evidence inventory before closeout. |
| Package/install metadata or docs | Package/platform claim scan; static deferral proof; Make install/`pkg-config` proof when behavior or metadata changes; CMake install proof when CMake package/export changes. | Hosted Linux package CI after push; macOS/Windows supplemental CI when relevant. |
| Benchmark/report docs | Report-index claim scan; generate or inspect relevant report bundle if wording depends on current output. | Canonical, sentinel, or large-matrix reports depending on touched wording. |
| Workflow YAML | YAML parse if available; shell/PowerShell syntax for embedded proof scripts where practical; support-tier scan. | Hosted CI run after push. |
| Scripts | Syntax checks for touched shell/Python scripts; focused execution if low-risk and deterministic. | Full owner command if runtime/dependency budget allows. |
| CMake/build registration | CMake configure/build/registration checks; source-list consistency checks. | Local CTest execution when command budget allows. |
| C sources or headers | `make format && make lint && make test`; focused tests for touched owners before full gate. | CMake parity and package proofs if public API/build behavior changes. |
| Generated report artifacts | Inspect manifest/index/report freshness, row schema, row meaning, support tier, and skipped/missing behavior. | Regenerate reports if stale or absent and command budget permits. |

## Reviewed/Supplemental/Deferred Classification

| Classification | Meaning for Sprint 136 | Examples |
| --- | --- | --- |
| Reviewed local | Required local evidence for touched surfaces; can support closeout wording within local scope. | `git diff --check`, whitespace scan, full C quality gate when C/header files change, static deferral proof. |
| Local reviewed-equivalent package proof | Strong local package proof, but hosted platform claim still depends on CI tier. | `tests/test_install.sh`, `tests/test_cmake_install.sh`. |
| Hosted reviewed | Evidence only fully exists after hosted workflow execution. | Linux reviewed static-first package-contract CI. |
| Supplemental | Useful confidence that must not be promoted to reviewed support by wording. | macOS install/export, Windows install/downstream, coverage, benchmark reports, supplemental guardrail rows. |
| Staged | Known excluded or blocked work with explicit promotion gates. | Windows pthread/POSIX-backed tests. |
| Deferred | Future work preserved with blockers and promotion criteria. | Normalized cross-report schema, shared package manager recipes, deferred QR residual queue. |
| Unsupported/non-claim | Surfaces Sprint 136 should guard against claiming. | Shared-library packaging, dynamic ABI compatibility, runtime-loader behavior, broad external-library parity, portable performance, state-of-the-art superiority. |

## Full-Validation Risks

| Risk | Impact | Mitigation |
| --- | --- | --- |
| Documentation-only work accidentally widens claims. | Public wording may exceed evidence. | Run claim-boundary scans before and after public/support docs edits. |
| Supplemental platform evidence is described as reviewed parity. | Platform support truth drifts. | Preserve Sprint 134 tier labels and require hosted CI for hosted claims. |
| Generated reports are stale or absent. | Report evidence may be misread as current. | Inspect freshness metadata and record absence/staleness before claim use. |
| Coverage or dead-code output is overread. | Coverage/dead-code reports could imply completeness or cleanup readiness. | Preserve Sprint 131 report boundaries. |
| Package proof is treated as shared-library or ABI proof. | Package/ABI claims widen past Sprint 133 decision. | Run static deferral proof and package claim scans. |
| Local CMake proof is treated as cross-platform proof. | Platform parity could be overstated. | Separate local proof from hosted Linux/macOS/Windows workflow evidence. |
| Full C gate becomes required after accidental source/header edit. | Validation cost and failure risk increase. | Check `.c`/`.h` diff before finalizing each day. |
| Benchmark/sentinel rows are treated as portable performance. | Competitive wording could overclaim. | Keep benchmark reports local, threshold/freshness scoped, and non-comparative. |

## Stop Conditions

Stop and ask for user input before proceeding if any of these occur:

- a required reviewed validation command fails and the fix is unclear;
- any `.c` or `.h` file changes and `make format && make lint && make test`
  cannot be run or does not pass;
- public/support wording needs a positive claim that lacks clear evidence;
- a package/platform edit would imply shared-library, dynamic ABI,
  runtime-loader, package-manager, or platform parity support;
- a supplemental macOS/Windows lane is needed as reviewed evidence;
- generated report absence, staleness, or schema ambiguity blocks Day 8-9
  claim classification;
- hosted CI evidence is required but unavailable in the local branch context;
- residual QR work would require implementation rather than publication with
  promotion criteria.

## Completion Criteria

| Criterion | Status | Evidence |
| --- | --- | --- |
| Validation requirements match touched surfaces and inherited support tiers. | Complete | Validation lane matrix and touched-surface requirements preserve Sprint 131-135 boundaries. |
| No supplemental lane is promoted to reviewed evidence by wording alone. | Complete | Classification table separates reviewed, hosted reviewed, supplemental, staged, deferred, and unsupported lanes. |
| Expensive or hosted-runner validation gaps are explicit. | Complete | Command ownership map and risk/stop-condition tables identify hosted CI, tree-mutating coverage, generated reports, and runtime-dependent report commands. |

## Validation Notes

Day 3 changed only Sprint 136 planning artifacts. Required validation remains:

```bash
git diff --check
if rg -n "[[:blank:]]$" docs/planning/EPIC_11/SPRINT_136; then exit 1; fi
git diff --name-only -- '*.c' '*.h'
```
