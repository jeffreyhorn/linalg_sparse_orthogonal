# Sprint 215 Day 14: Blueprint And Closeout Review

## Decision

Close Sprint 215 as a version-pinned external-baseline **blueprint with a
reproducible blocked pilot**. The optional pilot was not executed: the Day 10
wrapper stopped before project or CHOLMOD solver commands because the
external v1 schema and validator are absent. Day 13 passed local quality and
compatibility checks; those checks do not create numerical, timing, hosted
or state-of-the-art evidence. No external `evidence.json` exists.

## Epic Item Audit

The six [project-plan](../../PROJECT_PLAN.md) items total 166 estimated
hours. These are scope estimates, not timesheet observations; no actual
hours were recorded. The 11-hour Day 14 allocation is a plan estimate, not
an assertion of elapsed work.

| Item / estimate | Planned deliverable | Observed outcome and evidence | Changed surface / remaining gate |
| --- | --- | --- | --- |
| 215.1 / 26 h | Exact external baseline, solver family and matrix subset | SuiteSparse `v7.12.2` / CHOLMOD 5.3.4, commit and one hash-pinned `symmetric_4.mtx` SPD solve cell selected in [Day 3](day3-baseline-selection-decision.md); alternatives and exclusions recorded in [Day 2](day2-candidate-baselines-and-matrices.md). | Planning artifacts; package runtime and fixture-publication identity still need proof. |
| 215.2 / 34 h | Correctness and performance methodology, platform and provenance | [Day 3 inputs](day3-methodology-inputs.md), [Day 4 fixed numerical gates](day4-correctness-and-tolerance-contract.md) and [Day 5 optional timing contract](day5-external-comparison-methodology.md) complete as specifications. | No solver result or timed sample; future cells require their own gates. |
| 215.3 / 28 h | External identity/status report schema design | [Day 6 shape](day6-external-report-schema-design.md), [Day 7 validation design](day7-schema-validation-design.md), parseable planning JSON and [Day 8 acceptance](day8-pilot-feasibility-decision.md). | The v1 schema/validator and production cell-ID test vector are absent; no semantic v1 acceptance. Selected manifests, normalizer and workflows untouched. |
| 215.4 / 34 h | Selected pilot if feasible, otherwise named blocker and executable fallback | Opt-in `scripts/cholmod_pilot.c`, `scripts/run_cholmod_pilot.py` and synthetic tests added on [Day 9](day9-selected-pilot-harness.md). [Day 10](day10-selected-pilot-outcome.md) captured a blocked preflight with raw hashed logs and no solver commands. | Blocked branch of planned outcome satisfied; external correctness, provenance and publication still open. |
| 215.5 / 20 h | Separate blueprint/pilot from unearned broad claims | [Day 11 claim matrix](day11-evidence-and-claim-matrix.md) and [Day 12 docs plan](day12-claim-documentation-and-validation-plan.md) drove README, benchmark README, solver guide, maintainer guide and `tests/test_external_baseline_claim_docs.py` updates. | Public text says blueprint/blocked attempt, not external parity, portable speed or state of the art. |
| 215.6 / 24 h | Integrated validation and closeout | [Day 13 record](day13-integrated-validation.md) passed full local C gate and affected corpus/report/benchmark/docs/synthetic checks. This Day 14 artifact, [blueprint](../STATE_OF_THE_ART_EVIDENCE_BLUEPRINT.md), [queue](../RESIDUAL_QUEUE.md) and [retrospective](../RETROSPECTIVE.md) close the audit. | External v1 and solver result remain blocked; no hosted job or optional timing run was performed. |

## Planned Versus Observed Pilot

The Day 8 GO authorized *preparation and a gated attempt*, not unconditional
execution. Day 9 produced an opt-in helper and provisional runner. Day 10
recorded the missing-schema/validator gate as
`provisional_status=blocked`, `status_reason=schema_validation_unavailable`,
`schema_validation.status=not_run`, `sides={}` and `timing=not_run`.
The retained ignored `build/external-baseline/.../attempt.json` and 16 raw
preflight logs rehashed on Day 13; there are no project/baseline solver
commands or accepted external result. The planned v1 `evidence_status`
vocabulary must not be confused with the runner's actual provisional fields.

## Validation Reconciliation

[Day 13](day13-integrated-validation.md) records actual exit codes for
`make format && make lint && make test`, helper clang-format and strict
compile/link, corpus schema and selected manifest, report normalizer and
comparison tests/freshness, canonical benchmark tests/freshness, docs guards,
`make docs-check`, planning JSON syntax, and nine synthetic pilot tests.
The full C gate was required because the Sprint added an opt-in `.c` file;
it passed. The helper binary was compiled but not run. The v1 file-existence
probes returned 1, documenting a product blocker rather than a quality-check
pass. Selected comparison freshness covered 46 local source-controlled-helper
rows; canonical benchmark freshness was repeat-one and threshold-free.
Neither is CHOLMOD evidence.

Day 14 edits are documentation-only. All new closeout relative Markdown
links resolved. `git check-ignore -v` confirmed `build/` and `docs/api/`
generated-file exclusions. `git diff --check` and a trailing-whitespace
scan of edited Markdown were clean. `make docs-check` exited 0 with 18
checked-in headers and matching generated reference/source pages;
`python3 tests/test_external_baseline_claim_docs.py` exited 0 (four test
methods), and `python3 tests/test_selected_performance_docs.py` exited 0.
No new C/header change or solver/hosted execution is attributed to this
closeout; the full Day 13 C gate was not rerun for docs-only edits.

## Handoff And Claim

The [blueprint](../STATE_OF_THE_ART_EVIDENCE_BLUEPRINT.md) gives the
conditional rerun order and future acceptance gates. The
[residual queue](../RESIDUAL_QUEUE.md) assigns functional decision owners
for the validator, actual loaded-library provenance, fixture/module
publication review, extra matrices and solver family, hosted runs and
stronger claim review. Sprint 216 may prioritize those residuals; it must
not promote a blocked preflight or a designed schema to pass evidence.

Earned claim: the repository now has an exact, testable one-cell external
comparison blueprint and a fail-closed local attempt. Unearned claims:
two-sided external numerical correctness, matrix/solver/platform parity,
portable speed, hosted thresholds, public external-result approval and
state-of-the-art status.
