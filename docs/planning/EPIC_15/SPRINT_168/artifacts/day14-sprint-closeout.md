# Sprint 168 Day 14: Sprint Validation And Closeout

## Purpose

Day 14 closes Sprint 168 by reconciling the hosted performance publication
lane against the sprint plan, recording final local validation, and preparing
the Sprint 169 methodology-hardening handoff.

The completed selected lane remains intentionally narrow:

- benchmark family: direct repeated-run CSC factorization;
- selected command owner: `make bench-canonical-report`;
- selected row: `artifact=bench_refactor_csc`;
- selected fixture and command: `tests/data/suitesparse/nos4.mtx --repeat 1`;
- selected local check: `make bench-canonical-report-freshness`;
- selected hosted check:
  `scripts/check_bench_canonical_freshness.py --mode hosted`;
- selected hosted CI job:
  `Linux reviewed hosted selected performance freshness`;
- selected hosted artifact:
  `sprint168-selected-performance-freshness`;
- claim boundary: `hosted_selected_threshold_free`.

## Final Validation Record

| Check | Result | Notes |
| --- | --- | --- |
| `.c` / `.h` modification check | Passed | No C source or header files were modified during Sprint 168. |
| `bash -n scripts/bench_canonical_report.sh` | Passed | Canonical report generator shell syntax is valid. |
| `PYTHONDONTWRITEBYTECODE=1 python3 -m py_compile scripts/check_bench_canonical_freshness.py` | Passed | Selected freshness checker parses. |
| `PYTHONDONTWRITEBYTECODE=1 python3 scripts/check_bench_canonical_freshness.py --help` | Passed | Checker CLI exposes `--report-dir` and `--mode`. |
| Ruby YAML parse of `.github/workflows/ci.yml` | Passed | Workflow YAML is parseable locally. |
| `PYTHONDONTWRITEBYTECODE=1 make bench-canonical-report-freshness` | Passed | Local selected-row freshness regenerates and checks the canonical bundle. |
| Hosted-style local generation plus `--mode hosted` | Passed | Hosted metadata requirements can be satisfied by the CI lane. |
| CI summary logic against hosted-style local output | Passed | Summary lines can read selected row and manifest metadata. |
| Targeted claim scan | Passed by inspection | Risky terms remain in explicit non-claim or boundary contexts. |
| `git diff --check` | Passed | No whitespace errors were reported after the Day 14 artifact update. |

The full C quality gate, `make format && make lint && make test`, was not
required because Sprint 168 did not modify any `.c` or `.h` files.

## Project-Plan Item Reconciliation

| Item | Status | Evidence |
| --- | --- | --- |
| 168.1 Performance Lane Selection | Complete | Day 3 selected `bench_refactor_csc` on `nos4.mtx --repeat 1` for the Linux hosted lane after comparing canonical candidates. |
| 168.2 Methodology Metadata | Complete | Day 6 added report/index/manifest metadata for support tier, claim boundary, runner context, build flags, CPU model, build mode, platform, compiler, thread state, repeat semantics, timestamp, branch, commit, baseline, threshold, and methodology notes. |
| 168.3 Freshness Check | Complete | Day 8 added `scripts/check_bench_canonical_freshness.py` and `make bench-canonical-report-freshness`. |
| 168.4 CI Wiring | Complete | Day 10 added the hosted selected-performance freshness job with bounded runtime, hosted metadata, checker execution, summary output, and artifact upload. |
| 168.5 Claim-Safe Docs | Complete | Day 11 updated README, benchmark docs, and maintainer docs with the selected lane and retained non-claims. |
| 168.6 Verification | Complete | Days 12 and 14 ran focused syntax, workflow, local freshness, hosted-mode, summary, claim-scan, and generated-output checks. |

## Hosted Evidence Expectations

The branch should not be treated as having hosted performance evidence until
the PR CI job named below passes:

```text
Linux reviewed hosted selected performance freshness
```

Passing hosted CI should:

1. generate `build/bench-reports/canonical/`;
2. validate the selected row in hosted mode;
3. print `sprint168-performance-summary` lines;
4. upload `sprint168-selected-performance-freshness`.

The uploaded bundle includes all canonical CSV files plus `index.tsv` and
`manifest.txt`, but only the `bench_refactor_csc` row is selected hosted
performance evidence.

## Residual Risks And Boundaries

- Hosted runner CPU model can be variable or unavailable; the checker permits
  `unknown` but still records the field.
- Timing values are published without regression thresholds, speedup
  thresholds, or external-library parity claims.
- Warmup, variance, and matrix-size fields remain explicit methodology values
  rather than portable performance guarantees.
- The hosted job proves one selected Linux GitHub Actions lane only.
- The canonical bundle contains unselected benchmark rows; docs and checker
  language keep those rows advisory.
- Generated report output remains under ignored `build/` paths and should not
  be staged as source.

## Sprint 169 Handoff

Sprint 169 can start from a working selected hosted-performance lane and should
focus on methodology hardening rather than reselecting the lane.

Recommended follow-through:

- Decide whether warmup and variance should remain explicit `not_recorded`
  metadata or become measured policy fields.
- Add matrix-size derivation for selected report rows if the publication
  contract needs dimension-aware interpretation.
- Decide whether selected performance freshness should remain a focused
  checker or also receive normalized report-index publication.
- Review artifact retention and hosted evidence discoverability after the PR
  CI job runs.
- Inspect the first hosted artifact for reviewer readability and adjust summary
  output only if reviewers still need manual context.

## Completion Check

| Criterion | Status | Evidence |
| --- | --- | --- |
| Selected hosted performance lane is implemented or explicitly deferred with evidence. | Complete | The lane is implemented in source-controlled CI and passes local hosted-mode validation; hosted evidence becomes active only after PR CI passes. |
| Sprint 168 artifacts match project-plan items. | Complete | Items 168.1 through 168.6 are reconciled above. |
| Sprint 169 can begin from a clear methodology-hardening baseline. | Complete | Handoff recommendations identify remaining methodology decisions without reopening lane selection. |
