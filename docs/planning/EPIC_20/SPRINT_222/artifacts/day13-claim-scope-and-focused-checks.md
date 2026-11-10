# Sprint 222 Day 13: Claim Scope and Focused Checks

## Applied claim decision

The [Day 12 disposition](day12-independent-rights-and-claim-decision.md) is **no publication while independent review and v1 provenance remain pending**. No root README, install guide, benchmark report, selected-report manifest or generated API page was changed to describe a new external result. The only Sprint 222 claim text remains in planning records and explicitly says the local numerical work is not a validated or approved external comparison. The Day 11 conditional one-cell correctness sentence remains an unapproved reviewer proposal; there is no approved public wording.

The Sprint 215 example in `docs/planning/EPIC_19/SPRINT_215/artifacts/day7-external-evidence-blueprint.example.json` is `record_kind=blueprint`, `evidence_status=planned` and is not a solve. The first Day 9 and repeat Day 10 records are `record_kind=run`, `evidence_status=blocked`, with `cell_id=null`, both v1 correctness side states `blocked`, timing `not_run` and claim scope `no_result`. Both retain `publication_review=pending` and fixture `attribution_review=pending`. The [Day 12 JSON disposition](day12-publication-disposition.json), SHA-256 `be288c15ee87ae5f313fe6c105f9dbeea592fce440acfe96da974e530b549c62`, has no approved external claim or allowed artifact paths. The repeat's dirty-patch digest differs from the first pair. These facts were rechecked from the actual JSON, rather than inferred from the Day 11 prose.

**Current planning wording:**

> Sprint 222 retained local numerical and validation evidence for a single candidate 4×4 project/CHOLMOD comparison. Required v1 execution provenance and publication review are incomplete; the records remain blocked, with no approved external correctness or performance result.

No speedup, winner, rank, hosted-threshold, cross-platform timing parity, state-of-the-art or broad solver-parity statement is supported. The local pair's numerical pass does not override the blocked v1 status or the independent rights review. Item 222.5's planned 18-hour claim-review scope has a documented **pending/no-publication** outcome, not an approval or a claim of 18 elapsed hours.

## Focused test matrix

| Exact command | Outcome | Coverage |
| --- | --- | --- |
| `python3 tests/test_validate_external_baseline_evidence.py` | PASS, 67 tests. | V1 positive/negative status, provenance, identity, raw artifact, numerical, timing and claim mutations. |
| `python3 tests/test_run_cholmod_pilot.py` | PASS, 19 tests on serial rerun after the timeout fixture adjustment. | Runner preflight, parsing, raw streams, process-tree timeout and blocked/failure behavior; no external CHOLMOD solve was rerun. |
| `make build/test_cholmod_pilot_csc` then `build/test_cholmod_pilot_csc` | PASS; `CSC pilot validation: PASS`. | Frozen CSC conversion contract and local C-side pilot assertions. |
| `python3 tests/test_external_baseline_claim_docs.py` | PASS, 5 tests. | Existing Sprint 215 public-document nonclaims plus broad-claim and missing-boundary mutations. |
| `/tmp/s222-api-docs-venv/bin/python tests/test_api_docs_workflow_policy.py` | PASS, 50 tests. | Structured workflow policy positive/negative cases. Temporary environment contains repository-pinned `PyYAML==6.0.3`. |
| `/tmp/s222-api-docs-venv/bin/python tests/test_api_docs_workflow_reader.py` | PASS, 20 tests. | Structured workflow reader and input handling. |
| `/tmp/s222-api-docs-venv/bin/python tests/test_api_docs_workflow_mutations.py` | PASS, 4 tests. | Workflow publication mutations. |
| `API_DOCS_PYTHON=/tmp/s222-api-docs-venv/bin/python bash scripts/check_api_docs_local_only.sh` | PASS; 111 job/step records checked. | Generated API paths ignored and absent from tracked/staged files; workflow publication policy. |
| `/tmp/s222-api-docs-venv/bin/python tests/test_api_docs_local_only_guard.py` | PASS, silent exit `0`. | Local-only guard fixture regressions. |
| `python3 scripts/normalize_report_index.py --no-generated --check` | PASS; 114 rows. | Existing normalized report index is unchanged by Sprint 222 planning work. |
| Direct JSON/manifests assertion and `git ls-files docs/api build/external-baseline` | PASS; blueprint planned, both run records blocked, different patch hashes, no `s222-` or `day10-repeat` in `selected_report_targets.tsv` or `report_families.tsv`; generated API and local evidence have no tracked files. | Selected-report, normalization, raw-bundle and generated-API boundaries. |

The first parallel pilot suite run failed one detached-pipe-holder timeout assertion: its 0.2-second deadline expired before the parent flushed `launched\n`. The isolated case passed. In `tests/test_run_cholmod_pilot.py`, the fixture deadline is now 1.0 second, preserving its under-2.5-second completion assertion against a 3-second detached pipe holder. The serial full suite then passed 19/19. This test-only change improves scheduling margin; it does not change `run_cholmod_pilot.py`, any solver, v1 schema, source fixture or observed evidence. A first API docs check used system Python without pinned PyYAML and exited before policy validation; the temporary pinned environment resolved that setup failure, and the guard and policy then passed. These initial failures are retained here rather than relabeled as successful first attempts.

Day 11's six adversarial v1 mutations also remain in ignored `day11-review-ftih1s_0/mutation-transcripts.json`: missing raw baseline side, changed loaded library bytes, bad baseline ABI, nonfinite metric, stale valid-syntax cell ID and unsupported claim scope were all rejected after a positive synthetic control. The Day 13 validator suite covers those classes again. No Day 13 command promoted the rejected observed candidates or copied raw binary evidence into documentation.

## Remaining Day 14 gates

Recheck all affected pilot, validator, CSC, installation and documentation routes after any changes; retain first/repeat hashes and blocked status. The Day 13 edit touches a Python test and planning docs, with no `.c` or `.h` changes. Independent attribution and rights approval, exact CHOLMOD build/module provenance, Apple BLAS/LAPACK identity, effective BLAS threads and a complete canonical observed cell remain open. Publication stays on hold.
