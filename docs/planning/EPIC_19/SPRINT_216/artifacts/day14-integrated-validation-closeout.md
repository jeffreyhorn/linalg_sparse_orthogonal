# Sprint 216 Day 14: Integrated Validation And Closeout

## Decision And Environment

Sprint 216's six selected evidence, claim, status, validation, retrospective
and queue deliverables are **complete on `sprint-216`**, subject to review and
merge. No public package, promoted Windows freshness, shared ABI, release,
external parity, controlled speed or state-of-the-art capability is promoted.
The [Epic retrospective](../../EPIC_19_RETROSPECTIVE.md) and
[residual queue](../RESIDUAL_QUEUE.md) retain their separate acceptance gates.

The current checkout is dirty at `c5f30490fc02d34ed27be363e1586c3472089faf`
on Darwin `24.6.0` x86_64. Runs below are local; generated output remains
ignored and records that dirty-tree provenance. No exact hosted run/job or
artifact was inspected in this closeout. The historical validations and
Day 10-11 current-tree results remain separately dated records.

## Day 14 Current-Tree Commands

Commands ran from the repository root, serially. Exit `0` applies only to
the named local contract, not a broader product claim.

| Command | Exit | Observed boundary |
| --- | ---: | --- |
| `python3 -m unittest tests.test_external_baseline_claim_docs` | 0 | Five external-claim text/mutation tests passed; no solver execution. |
| `python3 tests/test_selected_performance_docs.py` | 0 | Threshold-free selected wording guard passed. |
| `make support-docs-guard` | 0 | Package/platform/support quick-reference guard passed. |
| `make api-docs-freshness` | 0 | 18/18 generated reference and source pages; local-only, workflow/staging and routing checks passed. HTML remains ignored/local. |
| `python3 scripts/validate_corpus_schema.py` | 0 | Existing selected corpus schema passed, not the absent external v1 schema. |
| `python3 tests/test_selected_report_targets_manifest.py` | 0 | Selected target metadata guard passed without Windows promotion. |
| `make report-index-comparison-freshness` | 0 | Six local bundles regenerated; 46 selected report-index rows accepted for local freshness. |
| `make bench-canonical-report-freshness` | 0 | Canonical local report regenerated; selected row remains threshold-free. |
| `python3 -m unittest tests.test_run_cholmod_pilot` | 0 | Thirteen synthetic tests passed, including fail-closed preflight; CHOLMOD was not run. |
| `make docs-check` | 0 | 18 checked-in public headers have 18 reference and 18 source pages. Local generation only. |

The six regenerated `build/comparison/*/study.tsv` files have 40 rows:
all `status=pass`, all `support_tier=local_only`, all
`baseline_type=external-process-source-controlled-helper`, and every row
retains `no state-of-the-art claim`. Each optional NumPy and SciPy dependency
is `defer` in all six bundles. This verifies source-controlled helper
freshness, **not** independent-package parity. The regenerated canonical
manifest records `runner_context=local`, `status=measurement`,
`support_tier=local_only`, `baseline=n/a`, `threshold=n/a`,
`warmup=none_configured`, `variance=not_computed_single_sample`,
`backend_context=n/a`, and an explicit non-portable-performance note.

Day 10 also passed report normalizer/comparison tests, canonical benchmark
negative tests, Python syntax, and the focused matrix. The [Day 11 record](day11-quality-gate-and-retro-outline.md)
reports local static contract tests (21), package/static deferral guards,
the local Homebrew developer-mode boundary, Make install (26/26), and CMake
install (30/30). The install check cleaned the earlier ignored reports;
Day 14 regenerated comparison and benchmark output **after** those checks.
Day 14 did not rerun installed consumers or those Day 10-only tests because
their implementation surfaces have not changed since those passes.

## Status And Claim Reconciliation

The [Day 6](day6-project-plan-status-207-211.md) and
[Day 7](day7-project-plan-status-212-215.md) ledgers contain exactly 54
unique Sprint 207-215 items, each `complete` for its selected deliverable.
The [project plan](../../PROJECT_PLAN.md#final-item-status-on-branch) assigns
216.1-216.6 `complete` on this branch, with this record as their final local
gate. No item is labeled `promoted`. The [Day 13 claim audit](day13-retrospective-review.md)
cross-checks retrospective outcome groups against the Day 3/5 claim matrices
and the Day 10/11 evidence. The queue has 21 distinct RQ-01 through RQ-21
entries; each has an owner, dependency/revisit trigger, acceptance evidence,
stop condition, and claim gate. No acceptance target was removed at closeout.

Discrepancy D2-1 (combined pending Sprint 215/216 snapshot) is corrected.
D2-2/D2-3 are reconciled by current-tree selected reruns and by retaining
historical Sprint 215's nine-test count separately from the current 13.
D2-4 remains RQ-01/RQ-02/RQ-03: both external v1 schema and validator files
are absent, so no accepted `evidence.json` or two-sided solver cell exists.
D2-5 remains RQ-04/RQ-05/RQ-12/RQ-15: current hosted currency was not
rechecked. D2-6 remains the local-only API, static-first ABI/release, and
external non-claim decisions, with RQ-06/RQ-13/RQ-14 and other scoped queue
targets. The [state-of-the-art decision](../RESIDUAL_QUEUE.md#state-of-the-art-decision)
therefore remains **no claim**. A future validated tiny CHOLMOD correctness
cell would still not be a broad ranking.

## Required-Check Boundary

No tracked or newly untracked `.c` or `.h` file changed during Sprint 216.
The conditional `make format && make lint && make test` chain was therefore
**not triggered**. Earlier full C/CTest passes are historical, not Day 14
results. Hosted Windows, candidate-specific package/release, dynamic loader,
actual CHOLMOD solver, external v1 validator and controlled comparative
timing lanes were **not run or unavailable**. Their absence blocks stronger
claims; it does not block closing the selected audit/documentation work.

## Final Document Checks

After the Day 14 status, retrospective and queue edits, the local relative-link
scan passed across the changed public/maintainer docs, project plan,
retrospective and all Sprint 216 Markdown files. The prior-item ledger audit
found 54 unique `complete` selected dispositions; the project-plan table has
exactly six `complete` Sprint 216 dispositions. The queue audit found all
21 unique RQ IDs with owner, dependency/revisit trigger, acceptance, stop and
claim fields. Trailing-whitespace inspection and `git diff --check` passed.

The affected guards were repeated **after** those edits:
`python3 -m unittest tests.test_external_baseline_claim_docs` (5 tests),
`python3 tests/test_selected_performance_docs.py`,
`make support-docs-guard`, and `make docs-check` (18/18 public-header
reference/source pages) all exited `0`. No `.c` or `.h` diff or untracked
source/header was found at final inventory. These results close item 216.4
for its selected changed-surface gate, without claiming the unrun external,
hosted or release lanes.
