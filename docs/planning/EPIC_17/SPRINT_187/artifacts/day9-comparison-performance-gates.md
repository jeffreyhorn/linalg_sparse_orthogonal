# Sprint 187 Day 9: Comparison and Performance Gates

## Purpose

Define exact evidence requirements for Sprint 191 bounded external comparison
work and Sprint 192 methodology-bound performance work. These gates improve
numerical and performance credibility while protecting against broad ecosystem,
portable performance, and state-of-the-art claims.

## Current Evidence Boundary

Selected report target metadata is owned by
`tests/corpus/manifests/selected_report_targets.tsv`. The manifest already
contains selected comparison rows for QR, partial-SVD, LU, and Cholesky
families plus one selected benchmark row for `bench_refactor_csc`.

The current comparison path is:

- `scripts/run_external_comparison.py`;
- `make report-index-comparison-freshness`;
- `python3 scripts/normalize_report_index.py --family comparison --require-generated comparison --check-freshness`;
- `tests/test_selected_comparison_workflow.py`;
- Linux and macOS selected comparison workflow uploads.

The current performance path is:

- `scripts/bench_canonical_report.sh`;
- `make bench-canonical-report-freshness`;
- `scripts/check_bench_canonical_freshness.py`;
- selected manifest row `SRT-BENCH-REFACTOR-CSC-NOS4`;
- Linux hosted selected performance freshness upload.

These paths are evidence lanes, not broad correctness, parity, portable
performance, or state-of-the-art proof.

## Sprint 191 Gate: Bounded External Comparison Family

Sprint 191 must add exactly one bounded comparison family. It may extend the
existing comparison runner and selected manifest, but it must not turn selected
fixtures into broad external-library parity.

| Requirement | Acceptance criteria | Failure state |
| --- | --- | --- |
| Family selection | Exactly one solver family and subfamily are selected with a stable `target_key`, fixture, operation, baseline type, metrics, and tolerances. | More than one family is partially implemented or the selected family is ambiguous. |
| Fixture ownership | The fixture or generator input is source-controlled, deterministic, and small enough for local and hosted freshness runs. | Fixture generation depends on unowned external state or broad data downloads. |
| Reference path | The baseline/reference implementation is source-controlled or dependency-optional with explicit unavailable behavior. | Optional dependency absence is counted as pass evidence or hidden. |
| Metrics | Each output metric has a stable row ID, row kind, tolerance kind, tolerance value, and claim meaning. | Metrics are ad hoc, uncounted, or lack tolerance semantics. |
| Runner integration | `scripts/run_external_comparison.py` writes project observations, baseline observations, dependency status, `study.tsv`, `summary.md`, and `manifest.tsv` for the selected family. | Required files are missing or stale-output cleanup is incomplete. |
| Manifest integration | `tests/corpus/manifests/selected_report_targets.tsv` includes one selected row with exact expected row count, expected row IDs, required files, artifact pattern, workflow file/job/artifact/platforms, support tier, claim scope, non-claims, owner, and provenance. | Manifest rows duplicate existing keys, omit expected rows, widen platforms accidentally, or misstate support tier. |
| Freshness validation | Selected comparison freshness fails missing artifacts, stale commits, generated comparison failures, skipped/deferred selected rows, duplicate rows, unexpected rows, row-count mismatches, and missing selected families. | Stale or skipped rows can be interpreted as current pass evidence. |
| Workflow artifact scope | Hosted upload includes exact required files for the selected family and rejects broad `build/comparison/**` uploads. | The workflow uploads a broad directory or omits required files. |
| Documentation | Maintainer/report/solver docs state the exact fixture evidence and retained broad-parity non-claims. | Public wording implies external-library parity, broad solver correctness, or state-of-the-art status. |

## Sprint 191 Required Metadata

The new selected comparison row must define:

- `target_id`;
- `family=comparison`;
- `subfamily`;
- `target_key`;
- `row_meaning`;
- `selection_scope`;
- `support_tier`;
- `freshness_policy=generated_compare_inputs`;
- `generator_command`;
- `artifact_pattern`;
- `required_files`;
- `expected_rows`;
- `expected_row_ids`;
- `workflow_file`;
- `workflow_job`;
- `workflow_artifact`;
- `workflow_platforms`;
- `claim_scope`;
- `non_claims`;
- `owner`;
- `introduced_in`.

Generated comparison rows must expose enough metadata for the normalizer to
record:

- fixture key;
- operation;
- metric;
- row kind;
- tolerance kind;
- tolerance value;
- baseline type;
- source commit;
- source branch;
- generated timestamp;
- platform;
- compiler;
- support tier;
- claim scope;
- non-claims;
- caveat or skip/defer reason.

## Sprint 191 Required Validation Commands

Minimum commands:

```sh
python3 scripts/validate_corpus_schema.py
python3 tests/test_selected_report_targets_manifest.py
python3 tests/test_selected_comparison_workflow.py
make report-index-comparison-freshness
python3 scripts/normalize_report_index.py --family comparison --require-generated comparison --check-freshness
```

Additional validation is required when implementation touches code:

```sh
make format
make lint
make test
```

`make format && make lint && make test` is mandatory whenever `.c` or `.h`
files change.

## Sprint 191 Claim Boundary

Allowed claim shape:

> Selected comparison rows are fresh for the named fixture and metric set
> against the selected reference path.

Required paired non-claim shape:

> This does not claim broad solver correctness, broad external-library parity,
> dependency ecosystem parity, Windows report freshness, performance
> superiority, package proof, ABI proof, release proof, or state-of-the-art
> status.

## Sprint 192 Gate: Methodology-Bound Performance Evidence Lane

Sprint 192 must promote exactly one bounded performance lane from local
threshold-free context to methodology-bound hosted evidence. The default
candidate is the existing `bench_refactor_csc` `nos4.mtx --repeat 1` selected
benchmark lane unless Sprint 192 records a stronger selection reason.

| Requirement | Acceptance criteria | Failure state |
| --- | --- | --- |
| Lane selection | Exactly one benchmark artifact, fixture/workload, matrix size, repeat policy, platform, support tier, and claim boundary are selected. | Multiple benchmark lanes are partially promoted or the selected lane is unclear. |
| Runtime budget | Hosted runtime stays within the selected CI budget and has a timeout/failure behavior that does not hide missing artifacts. | Performance freshness is too slow or flaky for hosted proof. |
| Methodology metadata | The report records label, timestamp, commit, branch, platform, compiler, runner context, build flags, CPU model, build mode, thread count, command, fixture, matrix size, repeats, warmup, variance, baseline, threshold, backend context, and methodology notes. | Required methodology fields are empty, local defaults leak into hosted proof, or TSV control characters are accepted. |
| Freshness validation | `scripts/check_bench_canonical_freshness.py --mode hosted` verifies selected row identity, required artifacts, manifest agreement, metadata completeness, support tier, claim boundary, and non-portable-performance methodology token. | Stale, unlabeled, local-context, or missing-artifact rows can pass as hosted evidence. |
| Threshold policy | The lane either remains threshold-free with explicit rationale or defines one conservative regression sentinel with stable baseline and variance policy. | Timing numbers imply superiority without statistical or platform policy. |
| Workflow artifact scope | Hosted workflow uploads only exact selected benchmark artifacts and fails on missing files. | Broad benchmark directories or unrelated timing artifacts are uploaded as evidence. |
| Documentation | Benchmark docs, maintainer guide, README references, and report-index docs state the methodology-bound lane and retained performance non-claims. | Public wording implies portable speedup, architecture independence, or algorithmic superiority. |

## Sprint 192 Required Report Fields

The selected performance report must include these fields:

- `surface`;
- `category`;
- `report_label`;
- `generated_at_utc`;
- `git_commit`;
- `git_branch`;
- `platform`;
- `compiler`;
- `runner_context`;
- `build_flags`;
- `cpu_model`;
- `build_mode`;
- `omp_num_threads`;
- `artifact`;
- `relative_path`;
- `command`;
- `report_family`;
- `status`;
- `support_tier`;
- `claim_boundary`;
- `fixture_or_workload`;
- `matrix_size`;
- `repeat_semantics`;
- `warmup`;
- `variance`;
- `baseline`;
- `threshold`;
- `backend_context`;
- `methodology_notes`.

Hosted promotion requires:

- `support_tier=hosted_selected`;
- `claim_boundary=hosted_selected_threshold_free`, unless a later reviewed
  threshold policy changes this explicitly;
- `runner_context` is not `local`;
- `build_flags` is not `not_recorded`;
- `report_label` is not `unlabeled`;
- `methodology_notes` contains `not_portable_performance_claim`.

## Sprint 192 Required Validation Commands

Minimum commands:

```sh
python3 scripts/validate_corpus_schema.py
make bench-canonical-report-freshness
python3 scripts/check_bench_canonical_freshness.py --mode local
python3 scripts/check_bench_canonical_freshness.py --mode hosted
python3 scripts/normalize_report_index.py --family benchmark --check-freshness
python3 tests/test_selected_comparison_workflow.py
```

Hosted evidence is required before Sprint 192 can claim a hosted performance
lane. Local `--mode hosted` checks can validate artifact shape only when
environment variables emulate hosted metadata; they are not a substitute for
hosted workflow evidence.

Additional validation is required when implementation touches code:

```sh
make format
make lint
make test
```

`make format && make lint && make test` is mandatory whenever `.c` or `.h`
files change.

## Sprint 192 Claim Boundary

Allowed claim shape:

> The selected benchmark lane has fresh methodology-bound hosted evidence for
> the named workload, platform, build metadata, and threshold policy.

Required paired non-claim shape:

> This does not claim portable performance, architecture-independent speedup,
> algorithmic superiority, broad platform parity, package proof, ABI proof,
> external-library parity, release benchmark status, or state-of-the-art
> performance.

## Shared Evidence-Lane Non-Claims

Sprints 191 and 192 must retain these boundaries:

- no broad external-library parity;
- no broad QR, LU, Cholesky, SVD, partial-SVD, or sparse-direct solver
  correctness;
- no raw factor, basis, or singular-vector identity unless the selected row
  states that exact metric;
- no dependency ecosystem coverage;
- no Windows report freshness unless Sprint 190 promoted an exact Windows lane;
- no package-manager proof;
- no shared-library or dynamic ABI proof;
- no portable performance claim;
- no performance superiority claim;
- no release benchmark claim;
- no unqualified state-of-the-art claim.

## Completion Gates

Sprint 191 is complete when one new comparison family has deterministic
fixture/reference ownership, selected manifest metadata, generated report
artifacts, freshness validation, workflow upload scope, focused tests, and
claim-safe documentation.

Sprint 192 is complete when one performance lane has methodology metadata,
hosted freshness evidence, exact artifact upload scope, threshold or
threshold-free policy, report-index integration, and claim-safe documentation.

Either sprint must stop if validation fails, generated artifacts are stale or
missing, support tiers are widened without evidence, or public wording implies
broader parity/performance/state-of-the-art support than the selected evidence
proves.

## Validation

Day 9 changed planning documentation only. No `.c` or `.h` files were
modified, so the full C quality gate is not required.
