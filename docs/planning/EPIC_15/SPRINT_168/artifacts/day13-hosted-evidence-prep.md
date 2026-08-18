# Sprint 168 Day 13: Hosted Evidence Reconciliation Prep

## Purpose

Day 13 prepares the PR-hosted evidence review path for the Sprint 168 selected
performance lane. The goal is to make the hosted CI result reviewable without
letting documentation or PR language depend on unobserved hosted success.

The selected hosted lane remains narrow:

- job: `Linux reviewed hosted selected performance freshness`;
- command owner: `make bench-canonical-report`;
- checker: `scripts/check_bench_canonical_freshness.py --mode hosted`;
- selected artifact row: `artifact=bench_refactor_csc`;
- selected fixture and command: `tests/data/suitesparse/nos4.mtx --repeat 1`;
- claim boundary: `hosted_selected_threshold_free`.

## Local Validation Reconciliation

Day 12 local validation proved the local equivalents needed before relying on
hosted CI:

| Check | Result | Hosted relevance |
| --- | --- | --- |
| `bash -n scripts/bench_canonical_report.sh` | Passed | Report generator shell syntax is valid. |
| `python3 -m py_compile scripts/check_bench_canonical_freshness.py` | Passed | Freshness checker parses. |
| `python3 scripts/check_bench_canonical_freshness.py --help` | Passed | Checker CLI exposes `--report-dir` and `--mode`. |
| Ruby YAML parse of `.github/workflows/ci.yml` | Passed | Workflow YAML syntax is parseable locally. |
| `make bench-canonical-report-freshness` | Passed | Local selected-row freshness path regenerates and checks the canonical bundle. |
| Hosted-style local generation plus `--mode hosted` | Passed | Strict hosted metadata requirements can be satisfied before PR CI. |
| CI summary logic run locally | Passed | Summary output can read selected row and manifest metadata. |
| Targeted claim scan | Passed by inspection | Risky wording remains in explicit non-claim or boundary contexts. |

These checks do not replace the hosted CI result. They establish that local
equivalents and source-controlled syntax are ready for PR CI evaluation.

## Expected PR CI Evidence

Reviewers should inspect this job:

```text
Linux reviewed hosted selected performance freshness
```

Expected job behavior:

1. Check out the repository.
2. Set hosted performance metadata:
   - `BENCH_CANONICAL_REPORT_LABEL=sprint-168-hosted-performance`;
   - `SPARSE_CANONICAL_SUPPORT_TIER=hosted_selected`;
   - `SPARSE_CANONICAL_CLAIM_BOUNDARY=hosted_selected_threshold_free`;
   - `SPARSE_CANONICAL_RUNNER_CONTEXT=github-actions-ubuntu-latest`;
   - `SPARSE_CANONICAL_BUILD_FLAGS=default_make_flags`;
   - `SPARSE_CANONICAL_BUILD_MODE=serial`;
   - `SPARSE_CANONICAL_CPU_MODEL=<hosted model or unknown>`.
3. Run `make bench-canonical-report`.
4. Run the hosted freshness checker:

   ```sh
   python3 scripts/check_bench_canonical_freshness.py \
     --report-dir build/bench-reports/canonical \
     --mode hosted
   ```

5. Print `sprint168-performance-summary` lines.
6. Upload the generated canonical report bundle.

## Expected Artifact

Artifact name:

```text
sprint168-selected-performance-freshness
```

Expected uploaded paths:

- `build/bench-reports/canonical/bench_refactor_csc.csv`;
- `build/bench-reports/canonical/bench_chol_csc.csv`;
- `build/bench-reports/canonical/bench_iterative_reuse.csv`;
- `build/bench-reports/canonical/bench_eigs_reuse.csv`;
- `build/bench-reports/canonical/index.tsv`;
- `build/bench-reports/canonical/manifest.txt`.

The full canonical bundle is uploaded because the report generator produces
four CSV files. Only the `bench_refactor_csc` row is promoted to selected
hosted freshness evidence.

## Artifact Review Checklist

In `index.tsv`, reviewers should confirm exactly one row has:

| Field | Expected value |
| --- | --- |
| `artifact` | `bench_refactor_csc` |
| `relative_path` | `bench_refactor_csc.csv` |
| `command` | `tests/data/suitesparse/nos4.mtx --repeat 1` |
| `report_family` | `benchmark` |
| `status` | `measurement` |
| `support_tier` | `hosted_selected` |
| `claim_boundary` | `hosted_selected_threshold_free` |
| `fixture_or_workload` | `nos4.mtx` |
| `repeat_semantics` | `configured_repeat_1` |
| `baseline` | `n/a` |
| `threshold` | `n/a` |

Reviewers should also confirm:

- `methodology_notes` contains `not_portable_performance_claim`;
- `report_label` is `sprint-168-hosted-performance`;
- `runner_context` is not `local`;
- `build_flags` is not `not_recorded`;
- `generated_at_utc` is populated in UTC timestamp shape;
- `manifest.txt` agrees with the selected row for report label, commit,
  branch, platform, compiler, runner context, build flags, CPU model, build
  mode, thread setting, support tier, claim boundary, baseline, threshold, and
  methodology notes.

The checker owns these validations, so artifact review should be a sanity
check rather than a manual substitute for the passing CI job.

## Expected Summary Lines

Passing PR CI should include `sprint168-performance-summary` lines similar to:

```text
sprint168-performance-summary: artifact=bench_refactor_csc command=tests/data/suitesparse/nos4.mtx --repeat 1 fixture=nos4.mtx repeat=configured_repeat_1 support_tier=hosted_selected claim_boundary=hosted_selected_threshold_free
sprint168-performance-summary: report_label=sprint-168-hosted-performance runner_context=github-actions-ubuntu-latest build_flags=default_make_flags cpu_model=<hosted value> build_mode=serial omp_num_threads=unset
sprint168-performance-summary: manifest_report_label=sprint-168-hosted-performance manifest_claim_boundary=hosted_selected_threshold_free non_claims=threshold_free_no_portable_performance_claim
```

CPU model may be `unknown` if the hosted runner does not expose a stable value.

## Fallback And Deferral Wording

Use this wording if hosted performance publication fails due to GitHub Actions
infrastructure:

```text
The selected performance freshness lane is source-controlled and passes local
hosted-mode validation, but hosted publication is deferred because the CI run
did not complete due to infrastructure failure. Do not treat this branch as
having hosted performance evidence until the `Linux reviewed hosted selected
performance freshness` job passes.
```

Use this wording if hosted runtime exceeds the 10-minute budget:

```text
The selected performance lane exceeded the Sprint 168 hosted runtime budget.
Retain local-only `bench-canonical-report-freshness` evidence and defer hosted
publication until the command is narrowed or the runtime budget is redesigned.
Do not claim hosted selected performance freshness.
```

Use this wording if hosted metadata or freshness validation fails:

```text
The selected performance report was generated, but hosted freshness validation
failed. Treat the uploaded artifacts as diagnostic output only. Do not claim
reviewed hosted selected performance freshness until the checker passes with
`hosted_selected_threshold_free` metadata.
```

Use this wording if documentation needs to be merged before hosted evidence is
observed:

```text
Documentation describes the intended selected hosted performance lane and its
non-claims. The hosted evidence claim becomes active only after the PR CI job
`Linux reviewed hosted selected performance freshness` passes and uploads
`sprint168-selected-performance-freshness`.
```

## Non-Claim Checklist

PR review should reject wording that turns this lane into:

- a timing regression threshold;
- portable performance evidence;
- broad benchmark-family publication;
- external-library parity;
- solver correctness evidence;
- package, shared-library, or ABI evidence;
- broad platform support;
- release benchmark proof;
- state-of-the-art sparse linear algebra evidence.

## Current Artifact Currency

Sprint 168 artifacts through Day 13 are current:

- Day 1 sprint intake;
- Day 2 benchmark surface inventory;
- Day 3 candidate lane selection;
- Day 4 runtime suitability;
- Day 5 methodology metadata design;
- Day 6 metadata implementation;
- Day 7 freshness design;
- Day 8 freshness implementation;
- Day 9 CI lane design;
- Day 10 CI implementation;
- Day 11 claim-safe docs;
- Day 12 local validation;
- Day 13 hosted evidence prep.

## Completion Check

| Criterion | Status | Evidence |
| --- | --- | --- |
| PR reviewers know which hosted job and artifacts to inspect. | Complete | Job name, artifact name, report paths, summary lines, and selected row criteria are listed. |
| Fallback handling is documented. | Complete | Infrastructure, runtime, metadata/freshness, and pre-observed-docs fallback wording is provided. |
| No hosted claim depends on unobserved CI success. | Complete | The artifact states that hosted evidence is active only after the named CI job passes and uploads the named artifact. |
