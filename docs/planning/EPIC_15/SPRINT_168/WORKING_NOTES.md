# Sprint 168 Working Notes

## Sprint Goal

Promote one selected performance report into a hosted, freshness-checked CI
lane with methodology-bound claims.

## Source Artifact Note

The Sprint 168 request referenced `docs/planning/EPIC_12/PROJECT_PLAN.md` and
the title "Hosted Performance Publication Date", but the active merged Sprint
168 planning source is `docs/planning/EPIC_15/PROJECT_PLAN.md`, section
"Sprint 168: Hosted Performance Publication Lane".

## Branch Baseline

- Branch: `sprint-168`
- Starting point: current `master` after PR #186 merge.
- Sprint 167 status: complete and merged, with an evidence ledger, selected
  gap list, claim gates, and Sprint 168 handoff.
- Sprint 168 plan status: day-by-day plan exists at
  `docs/planning/EPIC_15/SPRINT_168/PLAN.md`.

## Sprint 167 Inputs Carried Forward

| Input | Source | Sprint 168 use |
| --- | --- | --- |
| Selected gap `G167-01` | Sprint 167 Day 11 | Own hosted methodology-bound performance publication. |
| Acceptance gate | Sprint 167 Day 12 | Select one family, matrix scope, platform, toolchain, command, runtime budget, and report path. |
| Stop condition `SC-004` | Sprint 167 Day 12 | Stop or narrow if runtime is too long, variance is unbounded, metadata is missing, or lane lacks a named platform/toolchain. |
| Sprint 168 handoff | Sprint 167 Day 13 | Start from `bench_refactor_csc` through `make bench-canonical-report` unless runtime or methodology review selects a narrower candidate. |
| Final closeout posture | Sprint 167 Day 14 | Keep performance evidence scoped and avoid broad state-of-the-art or portable superiority wording. |

## Candidate Starting Lane

| Field | Initial value |
| --- | --- |
| Preferred benchmark family | Direct repeated-run CSC factorization performance publication |
| Preferred binary | `build/bench_refactor_csc` |
| Preferred command owner | `make bench-canonical-report` |
| Script owner | `scripts/bench_canonical_report.sh` |
| Local output family | `build/bench-reports/canonical/` |
| Public interpretation owner | `benchmarks/README.md` |
| Documentation boundary | Branch-local measurement unless and until hosted freshness proof exists. |

## Retained Performance Non-Claims

Sprint 168 does not add or imply:

- portable performance superiority;
- broad backend superiority;
- broad matrix-family performance;
- performance parity with external libraries;
- release benchmark proof;
- cross-platform performance parity;
- state-of-the-art sparse linear algebra performance;
- general solver correctness from benchmark rows;
- package, ABI, install, or platform support beyond the selected hosted lane.

## Sprint 168 Stop Conditions

Stop and revise before proceeding if a change:

- treats `make bench-fast` smoke coverage as methodology-bound performance
  publication;
- describes local `make bench-canonical-report` output as hosted evidence
  before CI owns the selected report;
- leaves compiler, flags, CPU, platform, thread settings, repeat semantics,
  warmup/variance state, commit, command, fixture, artifact path, or claim
  boundary unspecified for the selected report;
- adds timing thresholds that imply universal speed or superiority without a
  methodology-bound policy;
- broadens docs from one selected lane to all benchmarks, all platforms, all
  matrix families, or external-library parity;
- stages generated build/report/cache output unintentionally;
- changes `.c` or `.h` files without running
  `make format && make lint && make test`.

## Working Assumptions

- Linux is the likely first hosted lane unless Day 3 lane selection finds a
  stronger bounded alternative.
- Sprint 168 may change scripts, Makefile targets, workflows, docs, and
  planning artifacts.
- If only documentation and planning files change on a given day,
  `git diff --check` is sufficient for that day.
- If scripts or workflow files change, run focused syntax/self-checks when
  practical in addition to `git diff --check`.
- If `.c` or `.h` files change, run the full C quality gate.
- Generated performance artifacts remain under ignored build/report paths
  unless a later sprint explicitly chooses a publication route.

## Daily Log

### Day 1: Sprint Intake And Performance Handoff

- Re-read the Sprint 168 section of
  `docs/planning/EPIC_15/PROJECT_PLAN.md`.
- Reviewed Sprint 167 Day 12 claim gates and Day 13 Sprint 168 handoff.
- Created the Sprint 168 artifact directory.
- Recorded the prompt path/title mismatch and active Epic 15 source artifact.
- Carried forward selected gap `G167-01`, acceptance expectations, stop
  condition `SC-004`, and the `bench_refactor_csc` canonical report starting
  candidate.
- Defined retained performance non-claims and Sprint 168 stop conditions.
- Created `artifacts/day1-sprint-intake.md`.

### Day 2: Benchmark Surface Inventory

- Reviewed `Makefile` benchmark targets and report-generation targets.
- Reviewed `scripts/bench_canonical_report.sh` and
  `scripts/performance_sentinels.sh`.
- Reviewed README and `benchmarks/README.md` benchmark/performance wording.
- Inventoried canonical report outputs, manifest/index fields, generated
  output paths, local-only boundaries, and current non-claim language.
- Identified reusable freshness patterns from oracle/comparison report
  freshness checks, normalized report-index behavior, report-family metadata,
  clear Makefile pass/fail messages, and CI artifact upload conventions.
- Confirmed Day 3 should score canonical candidates led by
  `bench_refactor_csc` on `tests/data/suitesparse/nos4.mtx --repeat 1`.
- Created `artifacts/day2-benchmark-surface-inventory.md`.

### Day 3: Candidate Lane Selection

- Compared `bench_refactor_csc`, `bench_chol_csc`,
  `bench_iterative_reuse`, `bench_eigs_reuse`,
  `make performance-sentinels`, and `make bench-fast` for hosted
  publication suitability.
- Scored candidates by runtime suitability, output stability, user value,
  methodology clarity, and claim-risk containment.
- Selected `bench_refactor_csc` on `tests/data/suitesparse/nos4.mtx
  --repeat 1` through `make bench-canonical-report` as the primary hosted
  performance publication lane candidate.
- Defined the selected scope: direct repeated-run CSC factorization, one
  `nos4.mtx` fixture, `configured_repeat_1`, Linux hosted CI as the initial
  platform lane, `build/bench-reports/canonical/bench_refactor_csc.csv`, and
  canonical `index.tsv` / `manifest.txt` metadata.
- Deferred `bench_chol_csc`, `bench_iterative_reuse`, `bench_eigs_reuse`,
  `performance-sentinels`, and `bench-fast` as alternates or adjacent
  evidence.
- Recorded out-of-scope performance claims and claim-safe wording for future
  docs after implementation.
- Created `artifacts/day3-candidate-lane-selection.md`.

### Day 4: Runtime Suitability And Local Dry Run

- Ran `BENCH_CANONICAL_REPORT_LABEL=sprint-168-day4-dry-run make
  bench-canonical-report`; it passed locally in `real 3.21`, `user 1.17`,
  `sys 0.72`.
- Ran the focused selected command
  `build/bench_refactor_csc tests/data/suitesparse/nos4.mtx --repeat 1`; it
  passed locally in `real 0.01`, `user 0.00`, `sys 0.00`.
- Inspected `build/bench-reports/canonical/bench_refactor_csc.csv`,
  `index.tsv`, and `manifest.txt`.
- Recorded selected artifact sizes: `bench_refactor_csc.csv` is 351 bytes,
  `index.tsv` is 2506 bytes, `manifest.txt` is 1882 bytes, and the generated
  canonical report directory is 24K.
- Confirmed generated `build/` output is ignored and should not be staged.
- Identified stable row identity fields and variable timing/context fields.
- Kept `bench_refactor_csc` on `nos4.mtx --repeat 1` as the selected hosted
  performance publication candidate.
- Created `artifacts/day4-runtime-suitability.md`.

### Day 5: Methodology Metadata Design

- Compared current `bench_canonical_report.sh` fields with the Sprint 167
  `G167-01` acceptance criteria and the Day 4 missing-methodology list.
- Defined required metadata for compiler, build flags, CPU, OS/platform,
  runner context, thread settings, build mode, repeat semantics, warmup,
  variance, timestamp, branch, commit, command, fixture, threshold policy,
  support tier, claim boundary, and methodology notes.
- Assigned artifact ownership across `bench_refactor_csc.csv`, `index.tsv`,
  `manifest.txt`, CI workflow metadata, `benchmarks/README.md`, and README.
- Defined deterministic formatting and unknown-value behavior for hosted and
  local runs.
- Recommended extending `scripts/bench_canonical_report.sh` with optional
  metadata overrides rather than creating a separate benchmark runner.
- Preserved threshold-free interpretation and non-claims for portable
  performance, backend superiority, external parity, platform parity, release
  proof, and state-of-the-art performance.
- Created `artifacts/day5-methodology-metadata-design.md`.

### Day 6: Report Metadata Implementation

- Updated `scripts/bench_canonical_report.sh` to keep the existing
  `make bench-canonical-report` interface while adding hosted-lane metadata
  hooks.
- Added optional environment overrides for `support_tier`, `claim_boundary`,
  `runner_context`, `build_flags`, `cpu_model`, and `methodology_notes`.
- Extended canonical `index.tsv` rows and `manifest.txt` with
  `runner_context`, `build_flags`, and `cpu_model`.
- Preserved generated CSV timing schemas and local defaults for current
  local-only canonical report workflows.
- Ran `bash -n scripts/bench_canonical_report.sh`; it passed.
- Ran hosted-style local validation with `BENCH_CANONICAL_REPORT_LABEL`,
  selected support/claim-boundary overrides, `SPARSE_CANONICAL_BUILD_FLAGS`,
  `SPARSE_CANONICAL_CPU_MODEL`, and `SPARSE_CANONICAL_BUILD_MODE=serial`;
  `make bench-canonical-report` passed.
- Confirmed generated `index.tsv` header and first row each have 29 fields.
- Confirmed generated report output remains ignored under `build/`.
- Day 6 changed a shell script and planning artifacts only. No `.c` or `.h`
  files were modified, so the full C quality gate is not required for this day.
- Created `artifacts/day6-metadata-implementation.md`.

### Day 7: Freshness Check Design

- Reviewed existing `report-index-oracle-freshness` and
  `report-index-comparison-freshness` Make targets.
- Reviewed `scripts/normalize_report_index.py` freshness behavior and confirmed
  generic benchmark rows remain advisory under the current report-family
  contract.
- Reviewed `tests/corpus/manifests/report_families.tsv` and preserved the
  broad canonical benchmark family as local/advisory evidence.
- Designed a focused selected-performance freshness checker for the
  `bench_refactor_csc` canonical report row instead of widening all benchmark
  report-index rows.
- Defined strict freshness criteria for required artifacts, `index.tsv`
  schema, exactly one selected row, selected command/fixture/repeat semantics,
  required methodology metadata, threshold-free baseline/threshold policy,
  support tier, claim boundary, and `manifest.txt` agreement.
- Defined local and hosted invocation modes so local runs can keep conservative
  defaults while hosted CI can require `hosted_selected`,
  `hosted_selected_threshold_free`, explicit runner context, explicit build
  flags, and a non-`unlabeled` report label.
- Defined failure-message prefixes and remediation text for missing files, bad
  schema, missing or duplicate selected rows, bad selected values, missing
  metadata, over-broad claim boundaries, and manifest mismatches.
- Explicitly excluded raw timing comparisons, speedup thresholds, external
  parity, solver-correctness expansion, package/ABI inference, and hosted
  promotion of unselected canonical rows.
- Day 7 changed planning artifacts only. No `.c` or `.h` files were modified,
  so the full C quality gate is not required for this day.
- Created `artifacts/day7-freshness-design.md`.

### Day 8: Freshness Check Implementation

- Added `scripts/check_bench_canonical_freshness.py` as the focused selected
  performance freshness checker for the `bench_refactor_csc` canonical row.
- Added `make bench-canonical-report-freshness`, which regenerates the
  canonical benchmark report bundle and then checks the selected row in local
  mode.
- Implemented strict checks for required artifacts, `index.tsv` schema,
  exactly one selected row, selected command/fixture/repeat semantics,
  threshold-free baseline and threshold fields, methodology notes, required
  metadata, local/hosted claim boundaries, and `manifest.txt` agreement.
- Kept the generic benchmark report-index family advisory and did not widen
  selected freshness to `bench_chol_csc`, `bench_iterative_reuse`, or
  `bench_eigs_reuse`.
- Ran `python3 -m py_compile scripts/check_bench_canonical_freshness.py`; it
  passed.
- Ran `python3 scripts/check_bench_canonical_freshness.py --help`; it passed.
- Ran `make bench-canonical-report-freshness`; it passed locally.
- Ran a missing-artifact negative check against an empty temporary report
  directory; it failed as expected with
  `freshness: error: benchmark_selected_artifact_missing`.
- Ran hosted mode against a local-default report; it failed as expected with
  `freshness: error: benchmark_selected_claim_boundary`.
- Ran a hosted-style positive dry run with explicit Sprint 168 metadata
  overrides and `--mode hosted`; it passed.
- Day 8 changed a Python script, `Makefile`, and planning artifacts only. No
  `.c` or `.h` files were modified, so the full C quality gate is not required
  for this day.
- Created `artifacts/day8-freshness-implementation.md`.

### Day 9: CI Lane Design

- Reviewed `.github/workflows/ci.yml`, including the Linux supplemental
  runtime/`bench-fast` job, Linux reviewed CMake parity, package contract,
  generated oracle/comparison freshness, TSan, lint, dead-code, and coverage
  jobs.
- Decided to design a separate `hosted-performance-freshness` job rather than
  mixing selected performance artifacts into oracle/comparison freshness or
  widening the supplemental `bench-fast` lane.
- Defined the hosted job name as
  `Linux reviewed hosted selected performance freshness` with
  `runs-on: ubuntu-latest` and `timeout-minutes: 10`.
- Defined hosted report metadata values for
  `BENCH_CANONICAL_REPORT_LABEL`, `SPARSE_CANONICAL_SUPPORT_TIER`,
  `SPARSE_CANONICAL_CLAIM_BOUNDARY`,
  `SPARSE_CANONICAL_RUNNER_CONTEXT`,
  `SPARSE_CANONICAL_BUILD_FLAGS`, `SPARSE_CANONICAL_CPU_MODEL`, and
  `SPARSE_CANONICAL_BUILD_MODE`.
- Specified hosted steps: checkout, collect CPU model, generate the canonical
  report, run `check_bench_canonical_freshness.py --mode hosted`, summarize the
  selected row and manifest metadata, and upload the canonical report bundle.
- Scoped artifact upload to `build/bench-reports/canonical/*.csv`,
  `index.tsv`, and `manifest.txt` under the artifact name
  `sprint168-selected-performance-freshness`.
- Classified hosted evidence only for the selected `bench_refactor_csc` row on
  `nos4.mtx --repeat 1`; retained non-claims for timing thresholds, portable
  speed, external parity, broad benchmark publication, package/ABI support,
  platform parity, and state-of-the-art performance.
- Day 9 changed planning artifacts only. No `.c` or `.h` files were modified,
  so the full C quality gate is not required for this day.
- Created `artifacts/day9-ci-lane-design.md`.

### Day 10: CI Lane Implementation

- Updated `.github/workflows/ci.yml` with a new
  `hosted-performance-freshness` job named
  `Linux reviewed hosted selected performance freshness`.
- Kept the selected performance lane separate from supplemental `bench-fast`,
  oracle/comparison freshness, package, CMake, TSan, dead-code, and coverage
  jobs.
- Added hosted metadata environment values for report label, support tier,
  claim boundary, runner context, build flags, CPU model, and build mode.
- Wired hosted CI steps to collect CPU model metadata, run
  `make bench-canonical-report`, run
  `scripts/check_bench_canonical_freshness.py --mode hosted`, print selected
  performance summary lines, and upload the canonical report bundle.
- Scoped uploaded artifacts to `build/bench-reports/canonical/*.csv`,
  `index.tsv`, and `manifest.txt` under
  `sprint168-selected-performance-freshness`.
- Preserved the selected evidence boundary for only `bench_refactor_csc` on
  `nos4.mtx --repeat 1`, with threshold-free metadata and retained non-claims
  for portable performance, external parity, broad benchmark publication,
  package/ABI support, platform parity, and state-of-the-art claims.
- Ran hosted-style local report generation and
  `check_bench_canonical_freshness.py --mode hosted`; it passed.
- Parsed `.github/workflows/ci.yml` with Ruby YAML; it passed.
- Ran the CI summary Python logic locally against the hosted-style report; it
  printed the selected artifact, command, fixture, repeat semantics, support
  tier, claim boundary, report label, runner context, build flags, CPU model,
  build mode, thread setting, and manifest non-claim summary.
- Day 10 changed workflow YAML and planning artifacts only. No `.c` or `.h`
  files were modified, so the full C quality gate is not required for this day.
- Created `artifacts/day10-ci-implementation.md`.

### Day 11: Claim-Safe Documentation Update

- Updated `README.md` to add `make bench-canonical-report-freshness` to the
  workflow guidance and command list.
- Added README wording for the reviewed Linux hosted selected-performance lane,
  limited to the selected `bench_refactor_csc` canonical row for
  `nos4.mtx --repeat 1`.
- Updated `benchmarks/README.md` with the selected freshness target, runner
  context/build flags/CPU metadata fields, local versus hosted mode
  requirements, selected row criteria, and unselected canonical-row boundaries.
- Updated `docs/maintainer_guide.md` with canonical report metadata ownership,
  the selected freshness target, hosted selected-performance lane behavior,
  and retained non-claims.
- Ran a targeted claim scan for risky performance, superiority, external
  parity, broad benchmark, timing threshold, regression threshold, and
  state-of-the-art wording across README, benchmark docs, maintainer docs, and
  Sprint 168 artifacts.
- Confirmed the new selected-performance references are present and risky
  terms appear only in explicit non-claim or retained-boundary wording.
- Day 11 changed documentation and planning artifacts only. No `.c` or `.h`
  files were modified, so the full C quality gate is not required for this day.
- Created `artifacts/day11-claim-safe-docs.md`.

### Day 12: Local Validation Sweep

- Ran `bash -n scripts/bench_canonical_report.sh`; it passed.
- Ran `PYTHONDONTWRITEBYTECODE=1 python3 -m py_compile
  scripts/check_bench_canonical_freshness.py`; it passed.
- Ran `PYTHONDONTWRITEBYTECODE=1 python3
  scripts/check_bench_canonical_freshness.py --help`; it passed.
- Parsed `.github/workflows/ci.yml` with Ruby YAML; it passed.
- Ran `PYTHONDONTWRITEBYTECODE=1 make bench-canonical-report-freshness`; it
  passed and regenerated the canonical report bundle under ignored `build/`
  paths.
- Ran hosted-style local generation with explicit Sprint 168 hosted metadata
  and then `check_bench_canonical_freshness.py --mode hosted`; it passed.
- Ran the CI summary Python logic locally against the hosted-style generated
  bundle; it printed the selected artifact, command, fixture, repeat semantics,
  support tier, claim boundary, report label, runner context, build flags, CPU
  model, build mode, thread setting, and manifest non-claim summary.
- Ran a targeted claim scan for risky performance, superiority, external
  parity, broad benchmark, timing threshold, regression threshold, and
  state-of-the-art wording across README, benchmark docs, maintainer docs, and
  Sprint 168 artifacts.
- Confirmed risky terms appear only in explicit non-claim or retained-boundary
  wording.
- Skipped `make format && make lint && make test` because no `.c` or `.h`
  files were modified.
- Created `artifacts/day12-local-validation.md`.

### Day 13: Hosted Evidence Reconciliation Prep

- Reconciled Day 12 local validation with the Day 10 hosted CI lane.
- Identified the expected PR CI job as
  `Linux reviewed hosted selected performance freshness`.
- Identified the expected uploaded artifact as
  `sprint168-selected-performance-freshness`.
- Defined the expected hosted report paths under
  `build/bench-reports/canonical/`.
- Created a reviewer checklist for the selected `bench_refactor_csc` row,
  including command, fixture, repeat semantics, support tier, claim boundary,
  baseline, threshold, report label, runner context, build flags, CPU model,
  UTC timestamp shape, methodology notes, and manifest agreement.
- Recorded expected `sprint168-performance-summary` output lines.
- Added fallback wording for hosted infrastructure failure, runtime-budget
  failure, hosted metadata/freshness failure, and docs landing before hosted
  evidence has been observed.
- Reaffirmed non-claims for timing thresholds, portable performance, broad
  benchmark publication, external-library parity, solver correctness,
  package/ABI support, broad platform support, release proof, and
  state-of-the-art evidence.
- Confirmed Sprint 168 artifacts and working notes are current through Day 13.
- Day 13 changed planning artifacts only. No `.c` or `.h` files were modified,
  so the full C quality gate is not required for this day.
- Created `artifacts/day13-hosted-evidence-prep.md`.

### Day 14: Sprint Validation And Closeout

- Confirmed no `.c` or `.h` files were modified during Sprint 168, so the
  full C quality gate is not required.
- Re-ran `bash -n scripts/bench_canonical_report.sh`; it passed.
- Re-ran `PYTHONDONTWRITEBYTECODE=1 python3 -m py_compile
  scripts/check_bench_canonical_freshness.py`; it passed.
- Re-ran `PYTHONDONTWRITEBYTECODE=1 python3
  scripts/check_bench_canonical_freshness.py --help`; it passed.
- Re-parsed `.github/workflows/ci.yml` with Ruby YAML; it passed.
- Re-ran `PYTHONDONTWRITEBYTECODE=1 make bench-canonical-report-freshness`;
  it passed locally.
- Re-ran hosted-style local generation with Sprint 168 hosted metadata and
  `check_bench_canonical_freshness.py --mode hosted`; it passed.
- Re-ran the CI summary logic locally against the hosted-style generated
  bundle; it printed the selected row, metadata, and manifest summary lines.
- Reconfirmed risky performance wording remains limited to explicit non-claim
  and claim-boundary contexts.
- Reconciled project-plan items 168.1 through 168.6 as complete.
- Recorded hosted evidence expectations, residual risks, generated-output
  policy, and the Sprint 169 methodology-hardening handoff.
- Created `artifacts/day14-sprint-closeout.md`.
