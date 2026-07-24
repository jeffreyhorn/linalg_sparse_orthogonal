# Sprint 131 Day 9 - Dead-Code and Guardrail Report Architecture

## Purpose

Day 9 defines how dead-code, unused-code, stale-artifact, and large-matrix
guardrail outputs fit into recurring assurance without changing report
semantics.

This is a documentation-only architecture artifact. It does not change
dead-code scripts, guardrail scripts, Makefile targets, tests, benchmarks,
generated outputs, CI, or source code.

## Authoritative Inputs

| Input | Role |
| --- | --- |
| `Makefile` dead-code targets | Own `deadcode-compile-db`, `deadcode`, `deadcode-report`, `deadcode-check`, shared artifact paths, and serialized execution. |
| `scripts/deadcode_workflow.sh` | Owns raw dead-code evidence refresh: compile database validation, `cppcheck`, `xunused`, and compile-db coverage notes. |
| `scripts/deadcode_report.py` | Owns bucket classification, `report.md`, `report.tsv`, and report-completeness validation. |
| `Makefile` large-matrix target | Owns `make large-matrix-guardrails` and the reviewed guardrail command bundle. |
| `scripts/large_matrix_guardrails.sh` | Owns large-matrix guardrail manifest, index rows, reviewed lanes, supplemental lanes, and CSV-shape validation. |
| `docs/maintainer_guide.md` | Owns maintainer interpretation for dead-code reports, coverage, performance sentinels, and large-matrix guardrails. |
| Sprint 131 Day 4-8 artifacts | Own corpus/report taxonomy, report-index requirements, large-matrix first-index design, and coverage-gap boundaries. |

## Output Inventory

| Surface | Command or owner | Primary outputs | Stability | Default interpretation |
| --- | --- | --- | --- | --- |
| Compile database | `make deadcode-compile-db` | `build/deadcode-cmake/compile_commands.json` | Stable contract, environment-dependent content | Dead-code scanner input. Missing translation units are coverage gaps, not proof that omitted files have no unused surface. |
| Raw dead-code workflow | `make deadcode` and `scripts/deadcode_workflow.sh` | `build/deadcode/coverage-notes.txt`, `cppcheck.txt`, `xunused.txt`, `.workflow.stamp` | Tool-version and platform sensitive | Raw evidence only. Must be classified before cleanup or claim decisions. |
| Classified dead-code report | `make deadcode-report` and `scripts/deadcode_report.py` | `build/deadcode/report.md`, `report.tsv`, `.report.stamp` | Stable bucket schema, content depends on tools | Maintainer triage report. Not removal-ready proof. |
| Dead-code completeness check | `make deadcode-check` | Check output over current report files | Stable contract | Report-completeness gate. Failure means malformed or uncategorized report state, not a zero-findings gate. |
| Large-matrix guardrails | `make large-matrix-guardrails` and `scripts/large_matrix_guardrails.sh` | `build/bench-reports/large-matrix-guardrails/index.tsv`, `manifest.txt`, reviewed logs, bounded CSV report | Stable lane IDs and schema | Reviewed structural guardrails plus bounded CSV-shape evidence. Not broad scalability, timing, memory, or coverage proof. |
| Supplemental large-matrix lanes | `SPARSE_LARGE_GUARDRAILS_SUPPLEMENTAL=1 make large-matrix-guardrails` | Full named-matrix reorder CSV and qg-AMD report CSV | Opt-in, platform-local | Threshold-free report context. Default skipped rows are expected. |
| Stale artifact state | Any generated report directory | Missing, old, branch-mismatched, or commit-mismatched artifacts | Depends on freshness metadata | No current evidence. Staleness is a report-freshness problem unless a reviewed lane was expected to run and failed. |

## Actionability and False-Positive Policy

| Bucket or lane | Tool source | Actionability | False-positive risk | Required owner action |
| --- | --- | --- | --- | --- |
| `coverage-gap` | `coverage-notes.txt` | Medium | Medium | Expand or explicitly defer compile-db coverage before treating scanner silence as meaningful. |
| `definitely-unused-internal-candidate` | `xunused` | High after maintainer review | Medium | Confirm no dynamic/internal owner use, batch cleanup separately, and run full code quality gates if code changes. |
| `public-surface-review` | `xunused` | Review-only | High | Do not delete automatically. Installed-header or public API symbols need public-surface audit and compatibility decision. |
| `secondary-candidate-signal` | `cppcheck` | Supporting context | High | Summarize for focused future review only. Do not treat count rows as cleanup instructions. |
| `non-deadcode-static-analysis-noise` | `cppcheck` | Appendix-only | High | Keep visible for tool-health context; do not fail reviewed work on these counts. |
| Reviewed guardrail lane `G1` | `test_reorder_amd_qg` | High | Low | Failure blocks the guardrail target and needs graph/reorder owner triage. |
| Reviewed guardrail lane `G2` | `test_reorder_nd` | High | Low-medium | Failure blocks the guardrail target; explicit skips inside artifacts remain part of the bounded structural context. |
| Reviewed guardrail lane `G3` | `test_graph` | High | Low | Failure blocks the guardrail target and needs graph owner triage. |
| Reviewed guardrail lane `G4` | `bench_reorder --sprint86-slice --skip-factor` | Medium-high | Medium | CSV schema, fixture, or fill-row shape failure blocks the guardrail target; timing is not a claim. |
| Supplemental guardrail lanes `S1` and `S2` | Benchmark reports | Low unless opted in | Medium-high | Default skip rows are expected. Opt-in report failures are supplemental unless a future sprint promotes a lane. |

False positives are not reviewed failures by themselves. A finding becomes a
reviewed failure only when the owning target defines it as such, the output
matches the target's stable contract, and the affected owner can reproduce the
issue with the documented command.

## Suppression, Waiver, and Known-False-Positive Policy

| Case | Policy |
| --- | --- |
| Public API symbols reported unused | Keep in `public-surface-review` unless a public-surface audit records compatibility impact and removal plan. Reviewed public keeps stay visible as closeout context rather than cleanup debt. |
| Internal symbols reported unused by `xunused` | Do not suppress by default. Classify as internal candidates, batch with owner review, and remove only in a code-change sprint with full validation. |
| `cppcheck` secondary unused/static-function signals | Keep summarized as `secondary-candidate-signal`. Promote only after a focused pass confirms symbol-level actionability. |
| `cppcheck` style, portability, or noise checks | Keep in `non-deadcode-static-analysis-noise` appendix. Do not add broad suppressions just to lower counts. |
| Tool partial success | `xunused` nonzero status can be accepted only when a usable scan trace is emitted and report generation classifies it. Missing usable output is a workflow failure. |
| Compile-db omissions | Record as `coverage-gap`; do not present omitted benchmark/example paths as scanned. |
| Large-matrix supplemental skips | Preserve explicit skip rows unless `SPARSE_LARGE_GUARDRAILS_SUPPLEMENTAL=1` is set. Skips are not failures. |
| Stale generated reports | Mark as stale or missing current evidence. Do not edit the stale artifact by hand to make it look fresh. |
| Waivers | Waivers must name the bucket, symbol or lane, owner, reason, validation command, and expiration or revisit trigger. Broad "tool noise" waivers are insufficient. |

## Index Eligibility Decision Table

| Output family | Index eligibility | Reason | Minimum fields |
| --- | --- | --- | --- |
| Large-matrix guardrail `index.tsv` | Eligible now | Stable lane IDs, reviewed/supplemental split, status field, command, artifact, and notes already exist. | `lane_id`, `status`, `category`, `command`, `artifact`, `notes`, `manifest`, `freshness`. |
| Large-matrix guardrail `manifest.txt` | Eligible as companion metadata | Records timestamp, branch, commit, platform, compiler, supplemental mode, and artifact list. | `generated_at_utc`, `git_commit`, `git_branch`, `platform`, `compiler`, `supplemental`, `report_dir`. |
| Dead-code `report.tsv` | Eligible after Day 9 semantics | Stable bucket/tool/symbol/path/line/detail/disposition schema exists; index rows must preserve bucket meanings. | `bucket`, `tool`, `symbol`, `path`, `line`, `detail`, `disposition`, `freshness`, `owner_label`. |
| Dead-code `report.md` | Curated companion | Human-readable report contains context and next-action queue, but is not ideal as a primary machine index. | Link path, generated command, freshness, section list. |
| Dead-code raw `coverage-notes.txt` | Eligible only as source metadata | Compile-db counts and missing benchmark/example lists are useful for coverage-gap rows. | `compile_commands_json`, translation-unit counts, missing benchmark count, missing example count. |
| Dead-code raw `xunused.txt` | Deferred as primary index | Raw tool output is parsed into `report.tsv`; direct indexing risks bypassing classification. | Use through `report.tsv` only unless parser contract changes. |
| Dead-code raw `cppcheck.txt` | Deferred as primary index | Raw output contains broad static-analysis noise and secondary signals. | Use summarized buckets from `report.tsv`. |
| Coverage reports | Deferred to coverage-specific supplemental index | Day 8 requires backend, threshold, tree-mutating, freshness, and claim-boundary fields. | Use Day 8 minimum coverage-index fields. |
| Benchmark reports outside guardrails | Deferred unless canonical report schema applies | Direct benchmark-local CSVs have different schemas and no threshold semantics. | Use Day 6 report-index requirements. |
| Planning artifacts | Curated link index only | Useful for traceability, not generated assurance evidence. | Path, sprint/day, owner, claim boundary. |

## Coverage and Corpus Links

Dead-code, coverage, corpus, and guardrail reports answer different questions:

- dead-code coverage notes identify compile-database coverage gaps;
- line coverage reports identify source lines touched by tests;
- corpus taxonomy identifies fixture provenance, structure, support tier, and
  oracle type;
- large-matrix guardrails identify bounded structural/report checks over named
  large fixtures;
- none of these reports automatically proves broad numerical correctness.

Owner mapping should reuse Day 8 coverage labels when a dead-code or guardrail
row maps to a source family:

| Report row family | Related Day 8 owner label | Corpus taxonomy link |
| --- | --- | --- |
| Compile-db missing benchmark/example | `coverage-workflow` | Report-only coverage gap; not a fixture claim. |
| `xunused` in eigensolver sources | `coverage-eigensolvers` | Public/internal owner depends on declaration surface. |
| `xunused` in iterative sources | `coverage-iterative-preconditioners` | Public/internal owner depends on declaration surface. |
| `cppcheck` secondary source counts | Matching source-family owner label | Supporting context only; no fixture promotion. |
| Guardrail `G1` qg-AMD | `coverage-symbolic-graph` | Generated banded structural guardrail, not broad reorder parity. |
| Guardrail `G2` ND | `coverage-symbolic-graph` | Generated-family and named-matrix structural guardrail. |
| Guardrail `G3` graph | `coverage-symbolic-graph` | Graph partition and separator structural guardrail. |
| Guardrail `G4` reorder CSV shape | `coverage-symbolic-graph` | `bcsstk14` and `Pres_Poisson` checked-in expensive report fixtures. |
| Supplemental guardrail `S1` and `S2` | `coverage-symbolic-graph` | Optional threshold-free report context only. |

## Stale Report Policy

| State | Meaning | Required display |
| --- | --- | --- |
| Missing report directory | No current generated evidence. | `missing`, with command needed to regenerate. |
| Missing expected artifact | Report contract failure if the target was run; otherwise no current evidence. | `missing-artifact`, owner, and command. |
| Manifest commit or branch mismatch | Potentially stale relative to current checkout. | `stale`, recorded artifact commit/branch, current commit/branch if known. |
| Timestamp older than accepted freshness window | Historical evidence only. | `historical`, generated timestamp, refresh command. |
| Reviewed guardrail lane failure | Current guardrail failure. | `fail`, lane owner, artifact path, command. |
| Supplemental skipped lane | Expected default state. | `skip`, opt-in command or environment variable. |
| Dead-code `--check` failure | Report contract or categorization failure. | `invalid-report`, failing bucket or missing section if known. |

## Residual Guardrail Queue

| Residual | Actionability | Blocker | Future owner |
| --- | --- | --- | --- |
| Dead-code workflow shares `build/deadcode-cmake` and `build/deadcode/` | Medium | Need separate artifact roots or locking before parallel execution is safe. | `deadcode-workflow` |
| macOS dead-code enablement remains staged | Medium | Need fresh measurement across tool versions, SDK behavior, and `xunused` availability. | `deadcode-workflow` |
| Compile database may omit benchmark/example surfaces | Medium | Need CMake target coverage expansion or explicit permanent exclusions. | `coverage-workflow` |
| Public-surface unused findings require compatibility review | High | Need API owner decision before removal or waiver. | Affected public header owner plus maintainer guide owner. |
| `cppcheck` secondary signals are count-level only | Low-medium | Need symbol-level confirmation before cleanup batching. | Affected source-family owner. |
| Static-analysis noise remains visible | Low | Need checker-specific policy before adding suppressions. | `deadcode-workflow` |
| Large-matrix supplemental lanes are opt-in | Low | Need runtime, platform, and interpretation policy before promotion. | `large-matrix-guardrails` |
| Guardrail CSV schema changes are not normalized across report families | Medium | Need Day 10 decision on using current guardrail index versus adding common fields. | `report-index-owner` |
| Stale-report detection is not yet generated across all families | Medium | Need common freshness fields and current-checkout comparison policy. | `report-index-owner` |

## Day 10 Handoff

The safest first generated report/index path remains the existing large-matrix
guardrail `index.tsv` because it already has stable lane IDs, reviewed and
supplemental categories, explicit skip rows, and a manifest. Day 10 can either
use that existing index as the first implementation without schema changes or
make a scoped schema addition after preserving all current semantics.

Dead-code `report.tsv` is eligible for future indexing, but broad generation
should wait until the Day 10 decision confirms whether Sprint 131 is changing a
schema or only documenting the first existing generated index.

## Completion Criteria

| Criterion | Status | Evidence |
| --- | --- | --- |
| Guardrail outputs have clear owner and actionability semantics. | Complete | Output inventory, actionability table, owner links, and residual queue assign command owners and future owners. |
| False positives are not treated as reviewed failures. | Complete | False-positive and suppression policy keeps public API, secondary `cppcheck`, noise, and raw tool output out of automatic failure/removal paths. |
| Index eligibility is explicit for each output family. | Complete | Index eligibility table classifies guardrail, manifest, dead-code, raw, coverage, benchmark, and planning outputs. |
