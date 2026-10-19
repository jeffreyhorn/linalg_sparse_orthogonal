# Sprint 215 Day 12: Claim Documentation And Validation Plan

## Item 215.5 Documentation Closure

The [README](../../../../../README.md) now routes readers from selected
internal benchmark freshness to the separate Sprint 215 external-baseline
design and [blocked Day 10 preflight](day10-selected-pilot-outcome.md). The
[solver-selection guide](../../../../../docs/solver_selection.md) keeps its
Cholesky recommendation based on SPD input and identifies its existing
`cholesky_spd_tridiag_5` comparison as a source-controlled-helper result,
not the selected CHOLMOD cell. The [benchmark guide](../../../../../benchmarks/README.md)
and [maintainer guide](../../../../../docs/maintainer_guide.md) distinguish
the internal `bench_refactor_csc` / `nos4.mtx --repeat 1` freshness row from
the external `symmetric_4.mtx` correctness-only design. Neither an internal
selected row nor the blocked attempt is external-library parity, hosted
external evidence, portable speed or state-of-the-art proof.

The maintainer text names the exact SuiteSparse `v7.12.2` / CHOLMOD 5.3.4
selection, Day 4 independent numerical gates, runtime build/binary/ABI and
raw-log provenance, the proposed status vocabulary, Day 5 optional timing
rules, and publication attribution/license review. A future one-cell
`correctness_observed` record would still not justify cross-matrix,
cross-solver or cross-platform claims. The [Day 11 claim matrix](day11-evidence-and-claim-matrix.md)
remains the approved/rejected wording register. Item 215.5 is complete as
documentation and focused guard coverage, not as external evidence.

[`tests/test_external_baseline_claim_docs.py`](../../../../../tests/test_external_baseline_claim_docs.py)
checks required blocker/non-claim markers on all four public/maintainer
surfaces. Its mutation cases reject explicit promotions of the Sprint 215
blueprint to state-of-the-art performance, the blocked CHOLMOD pilot to
Linux/macOS timing parity, a single CHOLMOD measurement to portable speed,
and the selected pilot to cross-platform performance. An explicit negative
sentence remains allowed. This is a bounded wording guard, not a general
natural-language claim parser; reviewers must still inspect new claim forms.
The mutations also cover generic "blocked pilot" and "selected pilot"
phrasings without a CHOLMOD qualifier.

## Contract Reconciliation

| Surface | Actual or proposed field/command | Day 12 interpretation |
| --- | --- | --- |
| [Day 7 planning JSON](day7-external-evidence-blueprint.example.json) | `record_kind=blueprint`, `evidence_origin=planning`, `evidence_status=planned`, `cell_id=null`; both correctness sides and timing `not_run` | Parseable schema-design example only. It is not generated evidence and has no actual loaded binary/host/provenance. |
| [Day 6 schema design](day6-external-report-schema-design.md) and [Day 7 validator design](day7-schema-validation-design.md) | Proposed `tests/corpus/schemas/external_baseline_evidence_v1.json` and `scripts/validate_external_baseline_evidence.py`; `evidence_status=planned|not_run|blocked|non_comparable|correctness_failed|correctness_observed|timing_invalid|measured` | Neither proposed file exists. No v1 semantic validation or promotion to `evidence.json` can be claimed. |
| [Opt-in runner](../../../../../scripts/run_cholmod_pilot.py) | `--fixture tests/data/symmetric_4.mtx --output build/external-baseline/<selection_id>/<new_run_id> --correctness-only --execute` | Writes local provisional `attempt.json`; requires a new output directory. It does not write v1 `evidence.json`, a selected report row, or timing evidence. |
| [Day 10 generated preflight](day10-selected-pilot-outcome.md) | `record_kind=unvalidated_attempt`, `provisional_status=blocked`, `status_reason=schema_validation_unavailable`, `schema_validation.status=not_run`, `sides={}`, scalar `timing=not_run`; no `project` or `baseline` command | Exit 2 is the expected named blocker, not a failed numerical solve. Its ignored local bundle has raw discovery logs with hashes, but no solver logs or accepted result. |
| Selected benchmark/comparison reports | `SRT-BENCH-REFACTOR-CSC-NOS4`; selected helper comparison targets; normalized `status` fields | Unchanged. Do not map provisional `blocked`, proposed v1 `correctness_observed` or `measured` into existing `pass` rows without a separately reviewed family/loader/manifest migration. |

The planning JSON's object-shaped `timing.state` and `correctness.project` /
`correctness.baseline` fields are **not** the runner's scalar `timing` and
`sides` fields. The runner may label a later successful process outcome
`two_sided_correctness_candidate`, but that remains unvalidated until the
schema, complete cell ID, loaded dependencies and Day 4 rules are checked.
Do not rename or reshape `attempt.json` to imply v1 acceptance.

## Changed-Surface Validation Matrix

The following is the Day 13 acceptance queue, not a claim that all commands
have already run. A missing proposed validator is a named implementation
blocker, not a skipped passing check. Preserve exact command, exit status,
platform and raw output in the Day 13 record.

| Surface | Exact command | Expected interpretation |
| --- | --- | --- |
| Source-controlled corpus schema | `python3 scripts/validate_corpus_schema.py` | Exit 0; current fixture/manifests remain valid. This does not validate the proposed external v1 schema. |
| Planning JSON syntax | `python3 -m json.tool docs/planning/EPIC_19/SPRINT_215/artifacts/day7-external-evidence-blueprint.example.json` | Exit 0; syntax only, not v1 semantic acceptance. |
| External v1 implementation gate | `test -f tests/corpus/schemas/external_baseline_evidence_v1.json && test -f scripts/validate_external_baseline_evidence.py` | Expected nonzero while both are absent. Do not report an external observed row; implement and test the Day 7 negative fixtures before retrying the solver. |
| Selected target manifest | `python3 tests/test_selected_report_targets_manifest.py` | Exit 0; existing target IDs/counts and helper/benchmark semantics stay unchanged. |
| Normalized report index | `python3 tests/test_normalize_report_index.py` | Exit 0; no external family or provisional attempt is projected as `pass`. |
| Selected helper comparisons | `python3 tests/test_run_external_comparison.py` and `make report-index-comparison-freshness` | Exit 0 when prerequisites are present; inspect generated local rows only as source-controlled-helper comparisons. The generating target may be slower and is not CHOLMOD evidence. |
| Selected canonical benchmark | `make bench-canonical-report-freshness-tests` and `make bench-canonical-report-freshness` | Exit 0 when prerequisites are present; selected `bench_refactor_csc` row remains threshold-free, internal, and separate from the pilot. |
| Claim and docs guards | `python3 tests/test_external_baseline_claim_docs.py`, `python3 tests/test_selected_performance_docs.py`, and `make docs-check` | Exit 0; wording regressions and 18-header generated API coverage pass. This is documentation validation, not solver evidence. |
| Opt-in pilot synthetic checks | `python3 tests/test_run_cholmod_pilot.py` | Exit 0; parser/status/threshold mutations remain synthetic and do not execute CHOLMOD. |
| Existing Day 10 attempt | `jq -e '.provisional_status == "blocked" and .status_reason == "schema_validation_unavailable" and .schema_validation.status == "not_run" and .timing == "not_run" and .sides == {} and (.commands | has("project") | not) and (.commands | has("baseline") | not)' build/external-baseline/s215-cholmod-v7_12_2-symmetric4-spdsolve-v1/day10-preflight-20261019/attempt.json` | Exit 0 only if the retained local bundle exists and still proves no solver ran; otherwise reproduce with a fresh run ID after checking the absent-v1 gate. |
| C quality gate | `make format && make lint && make test` | Mandatory before closeout if any `.c` or `.h` file changed in the sprint. Day 9 added `scripts/cholmod_pilot.c` and recorded a pass; Day 13 must rerun/record the final branch state. |

The Day 12 edits are Markdown and a Python documentation regression test;
they do not modify the v1 schema, report contracts, selected manifests,
normalizer, workflows, or any `.c`/`.h` file. No hosted job or external
solver was run as part of this documentation closure. Missing package,
schema, raw artifact, numerical side or timing sample is never interpreted
as zero, parity or a successful performance result.
