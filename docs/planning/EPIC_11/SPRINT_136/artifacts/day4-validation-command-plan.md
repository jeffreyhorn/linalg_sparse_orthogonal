# Sprint 136 Day 4 - Validation Command Plan

## Purpose

Day 4 converts the Day 3 validation architecture into an executable command
plan for Days 5-7. The plan names commands, ownership, capture paths,
pass/fail interpretation, and skip/defer decisions before validation execution
starts.

This artifact does not execute the full validation package. It defines what
must run, what may run for supplemental confidence, what requires hosted CI,
and what remains unsupported or deferred.

## Artifact Capture Plan

Day 5-7 validation should write concise summaries under:

| Capture path | Owner |
| --- | --- |
| `docs/planning/EPIC_11/SPRINT_136/validation/day5-reviewed-validation-batch1.md` | Day 5 docs, source-list, and package/static proof summary. |
| `docs/planning/EPIC_11/SPRINT_136/validation/day6-reviewed-validation-batch2.md` | Day 6 CMake, test, install, and local quality summary. |
| `docs/planning/EPIC_11/SPRINT_136/validation/day7-supplemental-report-validation.md` | Day 7 benchmark/report/supplemental summary. |
| `docs/planning/EPIC_11/SPRINT_136/validation/generated-report-metadata.md` | Generated report freshness, manifest, index, and row-meaning notes if report commands run. |
| `docs/planning/EPIC_11/SPRINT_136/validation/skip-defer-register.md` | Explicit skip/defer decisions that remain after Day 7. |

The daily artifact files under `artifacts/` should summarize decisions and link
to these validation summaries if the validation directory is created during
execution.

## Executable Validation Command Matrix

| Batch | Command | Required when | Capture | Pass/fail interpretation |
| --- | --- | --- | --- | --- |
| Day 5 | `git diff --check` | Always before completing each validation batch. | Day 5 summary. | Pass means diff has no whitespace/conflict-marker errors; fail blocks until fixed. |
| Day 5 | `if rg -n "[[:blank:]]$" docs/planning/EPIC_11/SPRINT_136; then exit 1; fi` | Always for Sprint 136 planning artifacts. | Day 5 summary. | Pass means Sprint 136 Markdown has no trailing whitespace; fail blocks docs closeout. |
| Day 5 | `git diff --name-only -- '*.c' '*.h'` | Always before deciding C quality gate. | Day 5 summary. | Empty output means full C gate is not required by sprint rule; non-empty output makes full C gate required. |
| Day 5 | `git diff --name-only` grouped by surface | Always before selecting focused gates. | Day 5 summary. | Determines touched docs, scripts, workflows, CMake, package, benchmark, and source surfaces. |
| Day 5 | Focused public-doc trailing-whitespace scan over touched public docs | If public docs are touched. | Day 5 summary. | Pass means touched public docs are whitespace clean; fail blocks claim cleanup. |
| Day 5 | Focused markdown link/path scan over touched public docs | If public docs are touched or files move. | Day 5 summary. | Pass means referenced repo-local files/headings in touched docs are resolvable by the focused checker; fail blocks until fixed or explicitly deferred. |
| Day 5 | Claim-boundary `rg` scans for package/platform/performance/report/external parity wording | If public or maintainer docs are touched; optional baseline before Day 8-11. | Day 5 summary. | Findings are audit inputs; positive unsupported wording blocks cleanup completion unless fixed or explicitly queued. |
| Day 5 | `bash -n tests/test_install.sh tests/test_cmake_install.sh scripts/static_package_deferral_check.sh` | If package proof scripts change; optional package-proof preflight otherwise. | Day 5 summary. | Syntax pass only; does not prove package behavior. Fail blocks script changes. |
| Day 5 | `bash scripts/static_package_deferral_check.sh` | If package/ABI/shared-library wording or CMake package behavior changes; optional final confidence. | Day 5 summary. | Pass supports static-first deferral boundary; fail blocks package/ABI closeout. |
| Day 5 | `bash tests/test_install.sh` | If Make install/`pkg-config` behavior or metadata changes; optional package confidence. | Day 5 summary. | Pass supports local static-first Make install/`pkg-config` proof; fail blocks package-support claims. |
| Day 6 | `make format && make lint && make test` | Required if Day 5 finds any `.c` or `.h` changes. | Day 6 summary. | Pass required before proceeding after source/header edits; fail stops for fix or user input. |
| Day 6 | Focused owner tests for touched C/header owners | If C/header changes touch a solver/test owner. | Day 6 summary. | Pass supports owner-local behavior; fail blocks full gate or claim use. |
| Day 6 | `cmake -S . -B build-sprint136-cmake` | If CMake/build surfaces change; optional final CMake confidence. | Day 6 summary. | Pass means local configure succeeds; fail blocks CMake/build claims. |
| Day 6 | `cmake --build build-sprint136-cmake` | If CMake configure is required and succeeds. | Day 6 summary. | Pass means local CMake build succeeds; fail blocks CMake/build claims. |
| Day 6 | `ctest --test-dir build-sprint136-cmake -N` | If CMake test registration matters. | Day 6 summary. | Pass lists registered tests; count differences require explanation. |
| Day 6 | `ctest --test-dir build-sprint136-cmake --output-on-failure` | If CMake test execution is required and command budget allows. | Day 6 summary. | Pass supports local CMake test confidence; fail blocks CMake/test claims. |
| Day 6 | `bash tests/test_cmake_install.sh` | If CMake package/export behavior changes; optional install/export confidence. | Day 6 summary. | Pass supports local static CMake installed-consumer proof; fail blocks CMake package claims. |
| Day 6 | Source-list or registration checks such as `python3 scripts/check_library_sources.py` if available and relevant | If source lists or build registration change. | Day 6 summary. | Pass supports structural source-list consistency; fail blocks registration changes. |
| Day 7 | `make bench-canonical-report` | If benchmark/report wording changes or Day 8 needs fresh canonical evidence and runtime budget permits. | Day 7 summary and generated report metadata. | Pass creates local threshold-free snapshot; fail blocks using fresh canonical report evidence. |
| Day 7 | `make performance-sentinels` | If sentinel/performance wording changes or Day 8 needs fresh sentinel evidence and runtime budget permits. | Day 7 summary and generated report metadata. | Pass creates local sentinel bundle and wall-check context; fail blocks using fresh sentinel evidence. |
| Day 7 | `make large-matrix-guardrails` | If guardrail/report wording changes or Day 8 needs fresh guardrail evidence and runtime/data budget permits. | Day 7 summary and generated report metadata. | Pass creates reviewed/supplemental guardrail rows; fail blocks using fresh guardrail evidence. |
| Day 7 | Inspect `build/bench-reports/**/{index.tsv,sentinels.tsv,manifest.txt}` | If any report commands run or stale artifacts exist. | Generated report metadata. | Pass means freshness, row meaning, and artifact paths are recorded; absence/staleness must be recorded before claims. |
| Day 7 | `make deadcode-report` and `make deadcode-check` | If source/public-surface cleanup needs dead-code context or command budget explicitly includes it. | Day 7 summary. | Pass supports report-completeness only; fail blocks dead-code evidence use. |
| Day 7 | `make coverage` | Only if coverage wording changes or coverage evidence is explicitly required. | Day 7 summary. | Supplemental tree-mutating signal only; fail blocks coverage evidence use but not unrelated closeout. |

## Commands Required Before Claim Recalibration

Day 8 competitive recalibration may start only after these are complete or
explicitly deferred with reasons:

1. `git diff --check`.
2. Sprint 136 trailing-whitespace scan.
3. `.c`/`.h` diff check and full C quality-gate decision.
4. Public-doc claim-boundary baseline scan if public/support docs have changed
   before Day 8.
5. Package/static proof status if package/install/package wording has changed.
6. Generated report status for canonical, sentinel, and large-matrix report
   evidence: fresh, stale, absent, skipped, or deferred.
7. Hosted CI evidence status: unavailable locally, pending hosted run, or
   available from branch/PR CI.

If any required command fails and the fix is unclear, stop before Day 8.

## Pass/Fail Interpretation Table

| Result | Interpretation | Next action |
| --- | --- | --- |
| Required reviewed command passes | Evidence may be used within the command's scoped support tier. | Record summary and continue. |
| Required reviewed command fails | Closeout cannot proceed for the affected surface. | Fix if clear; otherwise stop and ask. |
| Optional supplemental command passes | Evidence may be cited only as supplemental/local confidence. | Record support tier and avoid reviewed wording. |
| Optional supplemental command fails | Do not use that evidence for positive claims. | Record failure or defer; continue only if unrelated required lanes pass. |
| Hosted CI lane unavailable locally | Local branch cannot prove hosted support. | Record as hosted-required/pending; avoid claiming current hosted pass. |
| Generated report absent | No fresh generated evidence exists in the working tree. | Generate during Day 7 if required and budget permits; otherwise record absence. |
| Generated report stale or schema ambiguous | Evidence cannot support final claim wording without context. | Record freshness/schema issue and defer or regenerate. |
| C/header diff is non-empty | Full C quality gate is mandatory. | Run `make format && make lint && make test`; stop on failure. |
| Claim-boundary scan finds unsupported positive wording | Wording is an audit finding, not a validation pass. | Fix on Day 11 or queue as explicit residual if out of scope. |

## Explicit Skip And Defer List

| Lane | Day 4 decision | Reason |
| --- | --- | --- |
| Full C quality gate | Skip unless `.c`/`.h` files change. | Sprint 136 Days 1-4 changed planning docs only. |
| CMake build/test execution | Defer until Day 6 and require only if CMake/build surfaces change; optional confidence otherwise. | Avoid running broad build validation before command ownership is finalized. |
| Make install/`pkg-config` proof | Defer execution to Day 5 if package surfaces changed or final confidence is selected. | Day 4 defines commands only. |
| CMake install/export proof | Defer execution to Day 6 if CMake package surfaces changed or final confidence is selected. | Day 4 defines commands only. |
| Benchmark/sentinel/guardrail reports | Defer execution to Day 7. | Runtime-dependent and should be captured with report freshness notes. |
| Dead-code report/check | Defer unless source/public-surface cleanup needs dead-code context. | Serialized and report-completeness only. |
| Coverage | Defer unless coverage wording changes. | Tree-mutating and supplemental. |
| Hosted Linux package CI | Defer to branch/PR CI. | Requires GitHub-hosted runner. |
| Hosted macOS and Windows package confidence | Defer to branch/PR CI. | Hosted supplemental lanes cannot be proven locally. |
| Windows staged pthread/POSIX tests | Defer. | Requires source portability or Windows-native replacement before promotion. |
| Shared-library, dynamic ABI, runtime-loader, package-manager support proof | Do not run. | Unsupported/deferred non-claims under Sprint 133. |
| QR residual implementation | Do not implement in validation. | Sprint 136 publishes residuals with promotion criteria on Day 12. |

## Day 5-7 Execution Plan

| Day | Execution focus | Required outputs |
| --- | --- | --- |
| Day 5 | Reviewed validation batch 1: docs hygiene, touched-surface inventory, claim scans, package/static preflight and proofs when relevant. | Day 5 artifact, validation summary, C/header gate decision, package/static status, skip/defer updates. |
| Day 6 | Reviewed validation batch 2: C quality gate if required, CMake configure/build/registration/test proof if required, CMake install/export proof if relevant. | Day 6 artifact, validation summary, CMake/test/package status, skip/defer updates. |
| Day 7 | Supplemental/report validation: benchmark, sentinel, guardrail, generated report metadata inspection, dead-code/coverage only if required. | Day 7 artifact, validation summary, generated report metadata, final validation execution summary. |

## Completion Criteria

| Criterion | Status | Evidence |
| --- | --- | --- |
| Validation can run without guessing command ownership or pass/fail meaning. | Complete | Executable validation command matrix and pass/fail interpretation table. |
| Generated reports are read as evidence with freshness context. | Complete | Artifact capture plan and report inspection rows require manifest/index freshness notes. |
| Validation scope is bounded before execution starts. | Complete | Required-before-recalibration list, skip/defer register, and Day 5-7 execution plan. |

## Validation Notes

Day 4 changed only Sprint 136 planning artifacts. Required validation remains:

```bash
git diff --check
if rg -n "[[:blank:]]$" docs/planning/EPIC_11/SPRINT_136; then exit 1; fi
git diff --name-only -- '*.c' '*.h'
```
