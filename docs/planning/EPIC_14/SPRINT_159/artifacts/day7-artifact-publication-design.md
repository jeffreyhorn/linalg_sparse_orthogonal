# Day 7 Artifact Publication Design

## Scope

Day 7 defines the hosted artifact and summary policy for Sprint 159 selected
oracle/comparison freshness evidence. Day 6 already added one scoped combined
artifact upload. Day 8 should refine that implementation so reviewers can
inspect promoted evidence quickly and so missing artifacts fail clearly.

This design keeps artifact publication limited to selected rows. It does not
promote broad report-index output, advisory report families, generated API
HTML, optional package baselines, or platform/package/performance/ABI claims.

## Publication Decision

Day 8 should split the current combined artifact into two selected uploads:

| Artifact name | Retention | Contents | Reason |
| --- | ---: | --- | --- |
| `sprint159-oracle-freshness` | 7 days | Oracle generated TSV, corpus report index, skips, and manifest. | Keeps QR/partial-SVD oracle evidence separate from comparison evidence and avoids implying all report families are promoted. |
| `sprint159-comparison-qr-minnorm` | 7 days | QR minimum-norm comparison observations, dependency status, study, summary, and manifest. | Keeps the single fixture-local comparison lane inspectable without broad QR parity implications. |

The combined Day 6 artifact name should be retired in Day 8 unless a reviewer
specifically asks for one combined bundle. Split artifacts are clearer and
make failure triage easier when oracle succeeds but comparison fails, or vice
versa.

## Hosted Summary Design

Day 8 should add deterministic console summaries after each selected
freshness command and before artifact upload.

### Oracle Summary

Inputs:

- `build/corpus/oracle/corpus.oracle.tsv`
- `build/corpus-reports/manifest.txt`

Required summary fields:

- total generated oracle rows;
- `solver_family=qr` row count;
- `solver_family=partial_svd` row count;
- `solver_family=unknown` generated-reference row count;
- `comparison_status=pass` count;
- source commit and branch from `manifest.txt`;
- support tier from `manifest.txt`.

Expected passing summary for the current selected family:

```text
sprint159-oracle-summary: total_rows=52 qr_rows=23 partial_svd_rows=26 generated_reference_rows=3 pass_rows=52
sprint159-oracle-summary: source_commit=<commit> source_branch=<branch> support_tier=local_only
```

The `support_tier=local_only` manifest value should not block the hosted run
on Day 8. It records generated-file provenance. Docs and report-family
metadata updates later in the sprint should reconcile wording for selected
hosted rows without rewriting historical generated manifest semantics too
early.

### Comparison Summary

Inputs:

- `build/comparison/qr_minnorm/study.tsv`
- `build/comparison/qr_minnorm/dependency_status.tsv`
- `build/comparison/qr_minnorm/manifest.tsv`

Required summary fields:

- selected comparison generated row count;
- `status=pass` row count;
- fixture key;
- source commit and branch from manifest;
- dependency `pass` count;
- dependency `defer` count;
- optional dependency names that remain deferred.

Expected passing summary for the current selected family:

```text
sprint159-comparison-summary: fixture=qr_underdetermined_minnorm_2x4 selected_rows=6 pass_rows=6
sprint159-comparison-summary: dependency_pass=2 dependency_defer=2 deferred_optional=numpy,scipy
sprint159-comparison-summary: source_commit=<commit> source_branch=<branch>
```

Deferred optional NumPy/SciPy rows are context only. They are not pass
evidence and must not become required dependencies in Day 8.

## Upload Policy

Day 8 should use strict missing-file behavior for each selected artifact
upload:

```yaml
if-no-files-found: error
```

Rationale:

- the oracle upload runs after `make report-index-oracle-freshness`;
- the comparison upload runs after `make report-index-comparison-freshness`;
- if selected outputs are missing after the corresponding command completed,
  the hosted evidence is incomplete and should fail.

Each upload should use `if: always()` so failure artifacts are published when
the generation command produced files before the failure. If a selected
command fails before producing any files, strict missing-file handling should
also fail the artifact step, which is acceptable because the job has already
lost selected evidence.

## Path Structure

### Oracle Artifact Paths

```text
build/corpus/oracle/corpus.oracle.tsv
build/corpus-reports/index.tsv
build/corpus-reports/skips.tsv
build/corpus-reports/manifest.txt
```

These paths prove only selected QR/partial-SVD oracle freshness when the
hosted job passes. They do not promote all corpus metadata or all report-index
families.

### Comparison Artifact Paths

```text
build/comparison/qr_minnorm/project_observations.tsv
build/comparison/qr_minnorm/baseline_observations.tsv
build/comparison/qr_minnorm/dependency_status.tsv
build/comparison/qr_minnorm/study.tsv
build/comparison/qr_minnorm/summary.md
build/comparison/qr_minnorm/manifest.tsv
```

These paths prove only the selected `qr_underdetermined_minnorm_2x4`
minimum-norm comparison when the hosted job passes.

## Row-State Summary Expectations

| State | Hosted command result | Summary/artifact expectation | Interpretation |
| --- | --- | --- | --- |
| Passing selected oracle rows | command exits `0` | Print row counts and upload oracle artifact. | Reviewed hosted evidence for selected oracle rows only. |
| Passing selected comparison rows | command exits `0` | Print row/dependency counts and upload comparison artifact. | Reviewed hosted evidence for selected comparison rows only. |
| Empty selected artifact | command should fail or upload should fail with `if-no-files-found: error` | No pass summary. | Missing selected output is not evidence. |
| Stale selected row | Make target exits nonzero through normalizer | Upload any generated diagnostics if files exist. | Product failure, not retry-only failure. |
| Failing selected row | Make target exits nonzero through generator or normalizer | Upload any generated diagnostics if files exist. | Product failure for selected lane. |
| Partial selected output | Make target or upload fails | Upload generated partial files if present. | Not reviewed evidence. |
| Optional NumPy/SciPy defer | command exits `0` if selected rows pass | Print deferred optional dependencies as context. | Not pass evidence and not failure. |
| Broad advisory family absent | no selected command should require it | No artifact. | Out of Sprint 159 hosted scope. |

## Naming Rules

Artifact and summary names must include:

- `sprint159`;
- selected family type, either `oracle` or `comparison`;
- comparison subfamily where applicable, `qr-minnorm`.

Artifact and summary names must not use:

- `all-report-index`;
- `benchmark`;
- `coverage`;
- `deadcode`;
- `package`;
- `platform`;
- `api-html`;
- `state-of-the-art`;
- broad `qr-parity` or `svd-parity` wording.

## Day 8 Implementation Checklist

1. Replace the combined artifact upload with split oracle and comparison
   uploads.
2. Add a deterministic oracle summary step after
   `make report-index-oracle-freshness`.
3. Add a deterministic comparison summary step after
   `make report-index-comparison-freshness`.
4. Set `retention-days: 7` for both uploads.
5. Set `if-no-files-found: error` for both uploads.
6. Preserve `if: always()` on uploads.
7. Keep selected commands unchanged.
8. Do not upload broad normalized report-index output.
9. Re-run YAML parse, `git diff --check`, and whitespace scans.

## Completion Check

- Reviewers can inspect promoted evidence from hosted runs.
- Artifact retention is explicit and bounded.
- Artifact names distinguish selected reviewed hosted rows from advisory
  local-only families.
- Row-state summary expectations are explicit for pass, empty, skipped,
  stale, partial, and failing output.
