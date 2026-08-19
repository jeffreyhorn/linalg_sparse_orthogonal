# Sprint 169 Day 13: Hosted Evidence Prep

## Purpose

Prepare the reviewer-facing checklist for the Sprint 169 selected performance
methodology changes before PR CI becomes the reviewed hosted evidence source.

## Local Validation Reconciliation

Day 12 local validation proved that the selected performance methodology
surface is internally consistent before hosted CI review:

- shell syntax passed for `scripts/bench_canonical_report.sh` and
  `scripts/performance_sentinels.sh`;
- Python compile checks passed for the freshness checker, report-index
  normalizer, and focused tests;
- `make bench-canonical-report-freshness-tests` passed all positive and
  negative schema cases;
- `make bench-canonical-report-freshness` passed in local mode;
- hosted-style local metadata validation passed in hosted checker mode;
- `make performance-sentinels` passed with S5/S6 hard gates and S2/S3
  threshold-free context rows;
- normalized report-index generation and freshness checks passed;
- targeted documentation claim scans found scoped caveats and non-claims only;
- generated `build/` report output remained ignored.

The hosted CI lane is still required before any README or maintainer wording
can be treated as reviewed hosted evidence. Local hosted-style validation only
checks metadata semantics with local execution.

## Hosted CI Lane To Inspect

The reviewer-facing hosted proof path is the CI job:

- job name: `Linux reviewed hosted selected performance freshness`;
- workflow file: `.github/workflows/ci.yml`;
- target command: `make bench-canonical-report`;
- checker command:
  `python3 scripts/check_bench_canonical_freshness.py --report-dir build/bench-reports/canonical --mode hosted`;
- uploaded artifact bundle: `sprint168-selected-performance-freshness`.

This job promotes only the selected canonical `bench_refactor_csc` row for
`tests/data/suitesparse/nos4.mtx --repeat 1` as hosted freshness and
methodology evidence. It does not promote unselected canonical benchmark rows,
the local S6 smoke ceiling, broad benchmark timing, platform parity, backend
superiority, external-library parity, package/ABI support, or state-of-the-art
performance.

## Expected Hosted Metadata

The selected `index.tsv` row should retain these reviewed values:

| Field | Expected value |
| --- | --- |
| `artifact` | `bench_refactor_csc` |
| `relative_path` | `bench_refactor_csc.csv` |
| `command` | `tests/data/suitesparse/nos4.mtx --repeat 1` |
| `fixture_or_workload` | `nos4.mtx` |
| `matrix_size` | `n=100` |
| `repeat_semantics` | `configured_repeat_1` |
| `warmup` | `none_configured` |
| `variance` | `not_computed_single_sample` |
| `support_tier` | `hosted_selected` |
| `claim_boundary` | `hosted_selected_threshold_free` |
| `baseline` | `n/a` |
| `threshold` | `n/a` |
| `runner_context` | `github-actions-ubuntu-latest` |
| `build_flags` | `default_make_flags` |
| `build_mode` | `serial` |
| `backend_context` | `n/a` |

The selected `manifest.txt` must agree with the selected row for report label,
commit, branch, platform, compiler, runner context, build flags, CPU model,
build mode, OpenMP thread state, support tier, claim boundary, baseline,
threshold, warmup, variance, matrix size, and methodology notes.

## Expected Hosted Summary Output

The CI summary should include three `sprint168-performance-summary:` lines:

- selected identity line with `artifact=bench_refactor_csc`,
  `fixture=nos4.mtx`, `repeat=configured_repeat_1`,
  `support_tier=hosted_selected`, and
  `claim_boundary=hosted_selected_threshold_free`;
- environment line with `report_label=sprint-168-hosted-performance`,
  `runner_context=github-actions-ubuntu-latest`,
  `build_flags=default_make_flags`, `build_mode=serial`, and recorded CPU and
  OpenMP metadata;
- manifest/non-claim line with
  `manifest_claim_boundary=hosted_selected_threshold_free` and
  `non_claims=threshold_free_no_portable_performance_claim`.

## Hosted Artifact Review Checklist

Reviewers should inspect the uploaded
`sprint168-selected-performance-freshness` artifact bundle and confirm:

1. `index.tsv` contains exactly one selected `bench_refactor_csc` row.
2. Unselected canonical rows remain `support_tier=local_only` and
   `claim_boundary=local_threshold_free`.
3. The selected row has `matrix_size=n=100`,
   `warmup=none_configured`, and
   `variance=not_computed_single_sample`.
4. The selected row remains threshold-free with `baseline=n/a` and
   `threshold=n/a`.
5. `manifest.txt` agrees with selected-row metadata for the fields enforced by
   `scripts/check_bench_canonical_freshness.py`.
6. The uploaded `bench_refactor_csc.csv` corresponds to the selected command
   and contains parseable `refactor_csc_ms` data.
7. The artifact bundle remains generated CI evidence, not a checked-in
   release benchmark publication.

## Failure Handling

If hosted CI fails because GitHub Actions infrastructure cannot fetch actions,
provision runners, upload artifacts, or start the job, rerun the job. Do not
change methodology wording or claim scope for infrastructure-only failures.

If `make bench-canonical-report` fails to build or run the selected benchmark,
stop and diagnose the build/runtime failure before updating claims.

If `check_bench_canonical_freshness.py --mode hosted` fails, treat it as a
schema or claim-boundary regression. The fix should preserve selected-row-only
promotion and should not relax unselected-row local-only checks.

If the summary step fails, inspect whether the selected row is missing,
duplicated, or malformed before changing the summary code. The summary is a
review aid; the freshness checker remains the authoritative hosted gate.

If the upload-artifact step fails after report generation and freshness pass,
rerun CI or fix artifact paths. Do not cite hosted evidence without an
available artifact bundle.

If the S6 local selected regression sentinel fails locally, treat it as local
large-regression governance only. It does not invalidate the hosted
threshold-free selected publication row unless the same failure also breaks
canonical report generation or selected freshness.

## Evidence Activation Rule

The Sprint 169 selected performance methodology becomes reviewed hosted
evidence only after the PR CI job `Linux reviewed hosted selected performance
freshness` passes and publishes the `sprint168-selected-performance-freshness`
artifact bundle for the reviewed commit.

Before that point, branch-local validation can be cited only as local
preflight evidence.

## Day 13 Completion

Day 13 completed the hosted-evidence checklist, expected summary-output
contract, artifact review checklist, fallback handling rules, and evidence
activation boundary. No `.c` or `.h` files were modified.
