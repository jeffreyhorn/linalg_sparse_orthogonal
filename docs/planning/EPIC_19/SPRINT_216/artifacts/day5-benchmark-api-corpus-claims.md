# Sprint 216 Day 5: Benchmark, API, Corpus, And Planning Claims

## Scope And Decision

Complete the item 216.2 claim audit begun in the [Day 3 matrix](day3-claim-inventory.md)
and [Day 4 user/maintainer update](day4-user-maintainer-claims.md). The
remaining benchmark, API reference, corpus/schema, and Epic planning text is
now calibrated to its source-controlled policy and Sprint 215 blocker.
This is a documentation decision, not an accepted external comparison, hosted
publication, timing threshold, release approval, or new solver proof.

## Edits And Evidence

| Surface | Change | Evidence and negative boundary |
| --- | --- | --- |
| [Benchmark guide](../../../../../benchmarks/README.md#report-index-handoff) | Renamed an ambiguous "canonical selected publication row" to "canonical selected freshness row." The external paragraph now says provisional `attempt.json` is not an accepted result and names the v1 validator, two-sided correctness, and report decision as gates. | [Sprint 212 threshold-free policy](../../SPRINT_212/artifacts/day5-product-policy-decision.md) keeps the selected `nos4.mtx --repeat 1` row at `baseline=n/a`, `threshold=n/a`; S5/S6 are separate local sentinels. [Sprint 215 Day 10](../../SPRINT_215/artifacts/day10-selected-pilot-outcome.md) blocked pre-solve. No timing or speed result. |
| [API reference](../../../../api_reference.md#generated-html) | Corrected the latest policy owner from Sprint 204 to Sprint 213 while preserving the 179/186/204 history and local-only output. | [Sprint 213 publication decision](../../SPRINT_213/artifacts/day5-product-policy-decision.md): ignored `docs/api/html/`, no hosted Pages, retained artifact, or committed HTML. Passing generation/routing checks does not prove publication or ABI stability. |
| [Corpus guide](../../../../../tests/corpus/README.md#row-interpretation) | Added an explicit distinction between optional-data skip/defer policy, source-controlled selected target metadata, and Sprint 215's blocked external attempt. | The [pilot](../../SPRINT_215/artifacts/day10-selected-pilot-outcome.md) has `status_reason=schema_validation_unavailable`; neither solver ran. `attempt.json` is not `comparison_status=pass`, `evidence.json`, or a selected report row. |
| [Report-index schema text](../../../../../tests/corpus/schemas/report_index_fields.md#external-baseline-boundary) | Documented that current manifests/normalized index do not include the external cell, and that a separately reviewed v1 validator, two-sided result, provenance, and publication decision are prerequisites. | [Sprint 215 claim matrix](../../SPRINT_215/artifacts/day11-evidence-and-claim-matrix.md) is a design, not a schema validator. Existing `external_optional` and source-controlled contract statuses are not pass evidence. |
| [Epic 19 Sprint 215 section](../../PROJECT_PLAN.md#sprint-215-external-baseline-and-state-of-the-art-evidence-blueprint) | Added the actual blocked-evidence boundary to the planned deliverables so the optional pilot is not read as an executed comparison. | [Sprint 215 outcome](../../SPRINT_215/artifacts/day10-selected-pilot-outcome.md); final project status snapshot and item statuses still belong to Days 6-7 under item 216.3. |

## Completed Claim Matrix

| Day 3 claim ID | Day 5 disposition | Remaining work |
| --- | --- | --- |
| C01-C02; D4-1 | README performance labels narrowed on Day 4; no comparative CSR or general sparse-LU speed claim. | None for item 216.2; future measurement would need its own context and decision. |
| C03-C04 | Selected benchmark and blocked-pilot README boundaries retained. | Current-tree checks still due under 216.4. |
| C05-C07 | INSTALL static/platform limits retained; external blueprint/blocked row added on Day 4. | Hosted candidate and v1 external evidence remain residuals, not docs passes. |
| C08-C09 | Local-only API and Sprint 213 latest-owner wording retained/corrected in INSTALL and maintainer guide. | None for item 216.2. |
| C10-C11 | Release checklist-as-gates and planning/provisional/accepted distinction retained. | Actual release and external validation remain residuals. |
| C12 | Benchmark row-family and blocked-pilot distinction retained; ambiguous publication wording removed. | No selected timing threshold or external report row. |
| C13 | Windows selected freshness and threshold-free corpus boundaries retained; external provisional/optional-data distinction added to corpus and schema text. | Windows promotion and external v1 validator remain residuals. |
| C14-C15 | API reference now names Sprint 213 as latest local-only decision; durable-link and no-hosted-publication boundaries retained. | None for item 216.2. |
| C16 | **Open status correction:** Sprint 215/216 snapshot still says pending. The Sprint 215 section now states the blocked evidence boundary. | Item 216.3 Days 6-7 must split/update the snapshot and item-level evidence links. Do not call this a completed status edit. |
| C17 | Epic non-goals retained. | Recheck after Days 6-7 status edits. |

New Day 5 observations: D5-1 was the benchmark "publication row" ambiguity;
D5-2 was the absent external provisional/selected-report distinction in corpus
and schema guidance. Both wording gaps are closed above. No proposed stronger
claim has an evidence bundle sufficient for promotion.

## Guard And Validation

The existing [external-baseline claim guard](../../../../../tests/test_external_baseline_claim_docs.py)
now requires provisional/non-pass markers in benchmark, corpus, schema, and
project-plan text. It rejects broad positive pilot claims on every covered
surface and still accepts explicit negations. This checks wording only; it
does not execute CHOLMOD, validate v1 evidence, or create a report row.

Day 5 changed Markdown and a Python test, not C or headers. The external-
baseline claim unit tests (5), selected-performance docs guard, corpus schema
validator, canonical report freshness tests, API docs freshness check,
support-docs guard, selected-report manifest test, local-link check, trailing-
whitespace check, and `git diff --check` passed. The results are also recorded
in the [working-notes tracker](../WORKING_NOTES.md#changed-surface-tracker).
Item 216.2's **claim-audit matrix is complete** with C16 explicitly handed
to item 216.3; final claim approval remains subject to Sprint 216 integrated
validation and closeout.
