# Sprint 212 Working Notes: Benchmark Methodology And Threshold Policy

## Sprint Goal

Decide and implement one selected benchmark methodology policy: either a
thresholded gate for one selected benchmark row or a stronger threshold-free
deferral proof.

## Scope Boundary

Sprint 212 is a selected benchmark-methodology sprint. It may add one bounded
threshold policy only if the current evidence supports runner, compiler,
repeat, warmup, variance, baseline, threshold, artifact, and claim-boundary
metadata for the selected row. If that evidence is not strong enough, the
sprint should close a threshold-free deferral with stronger tooling, manifest,
and documentation guards.

The sprint must not claim portable performance, broad benchmark superiority,
release benchmark status, package-manager distribution, ABI or shared-library
support, broad platform parity, OpenMP speedup, backend superiority,
state-of-the-art performance, or solver behavior changes.

## Item Checklist

| Epic item | Sprint 212 interpretation | Status |
| --- | --- | --- |
| 212.1 Benchmark Evidence Inventory | Inventory selected hosted benchmark lanes, report fields, runner metadata, freshness checks, selected manifest rows, and non-claims. | Complete for sprint decision input; Day 1 intake, Day 2 baseline, and Day 3 runner/variance inventory complete. |
| 212.2 Threshold Decision | Decide whether one selected threshold gate is supportable or whether threshold-free deferral should be strengthened. | Complete; Day 5 selects threshold-free deferral hardening for hosted selected canonical freshness. |
| 212.3 Methodology Implementation | Implement selected threshold metadata checks or threshold-free guard improvements. | Complete for freshness tooling; Days 7-8 enforce threshold-free methodology notes, selected exact metadata, unselected locality, and report/manifest methodology agreement. |
| 212.4 Regression Tests | Add benchmark freshness, manifest, methodology, missing metadata, and docs guard tests. | Complete for freshness and manifest/report guard layers; Days 7-9 cover missing non-portable methodology notes, all forbidden threshold/performance methodology tokens, manifest methodology drift, exact selected metadata, unselected claim-boundary drift, selected manifest identity, workflow metadata, claim scope, and non-claim exactness. |
| 212.5 Documentation Calibration | Update benchmark README, README, INSTALL, maintainer guide, and selected manifest wording. | Complete for user-facing and maintainer-facing calibration; Days 10-11 update README, INSTALL, benchmark README, maintainer guide, Epic plan status, and docs guards for threshold-free selected benchmark policy and future threshold prerequisites. |
| 212.6 Validation And Closeout | Run benchmark freshness, selected performance docs, manifest tests, docs checks, and relevant quality gates. | Complete; Days 12-14 pass integrated focused validation, review hardening, closeout checks, and confirm no C/header quality gate is required. |

## Day 1: Benchmark Evidence Intake

### Scope Trace

| Epic item | Day 1 intake interpretation | Initial evidence |
| --- | --- | --- |
| 212.1 Benchmark Evidence Inventory | Identify selected benchmark evidence surfaces and current threshold-free metadata owners before running or changing benchmark tooling. | Selected target row, benchmark docs, freshness checker, docs guard, Make targets, and hosted workflow lanes inventoried. |
| 212.2 Threshold Decision | Preserve both possible decision branches for later evaluation. | Day 1 records that the current selected hosted lane is threshold-free and that any thresholded gate needs stronger evidence than freshness alone. |
| 212.3 Methodology Implementation | No implementation on Day 1. | Implementation deferred until inventory, criteria, and decision evidence are complete. |
| 212.4 Regression Tests | Identify current regression-test owners and likely future fixture classes. | Current owners include selected benchmark freshness tests, selected performance docs guard, selected report target manifest tests, and normalized report-index tests. |
| 212.5 Documentation Calibration | Identify user-facing and maintainer-facing wording surfaces that constrain performance claims. | README, INSTALL support/readiness matrix, benchmark README, maintainer guide, corpus README, and report-index schema identified. |
| 212.6 Validation And Closeout | Plan documentation-only validation for Day 1 and later benchmark/doc guard validation. | Day 1 uses `git diff --check`; later days need focused benchmark freshness and docs tests. |

### Evidence Ledger

| Surface | Current owner | Day 1 finding | Sprint 212 relevance |
| --- | --- | --- | --- |
| Sprint source plan | `docs/planning/EPIC_19/PROJECT_PLAN.md` | Sprint 212 is a 166-hour sprint to decide between one selected thresholded benchmark gate and stronger threshold-free deferral proof. | Source of item scope and deliverables. |
| Day plan | `docs/planning/EPIC_19/SPRINT_212/PLAN.md` | Day 1 is evidence intake; Day 5 is the policy decision; Days 7-11 implement and document the selected policy. | Controls sequencing so implementation does not outrun evidence. |
| Selected benchmark manifest | `tests/corpus/manifests/selected_report_targets.tsv` | `SRT-BENCH-REFACTOR-CSC-NOS4` is the only selected benchmark target. It owns `bench_refactor_csc`, `nos4.mtx --repeat 1`, Linux/macOS workflow artifacts, hosted-selected support tier, hosted-selected freshness policy, claim scope, and non-claims. | Primary source of selected benchmark identity and claim boundary. |
| Canonical freshness checker | `scripts/check_bench_canonical_freshness.py` | Validates selected row identity, required columns, manifest agreement, artifact presence, hosted mode metadata, `baseline=n/a`, `threshold=n/a`, `warmup=none_configured`, and `variance=not_computed_single_sample`. It intentionally does not compare timing values. | Main tooling candidate for either threshold metadata enforcement or stronger threshold-free deferral checks. |
| Canonical report generator | `scripts/bench_canonical_report.sh` | Generates the canonical report bundle with benchmark rows, manifest metadata, `support_tier`, `claim_boundary`, `baseline`, `threshold`, `warmup`, `variance`, runner context, compiler, platform, build mode, and methodology notes. Current default is threshold-free. | Source of report fields that any threshold decision must preserve or extend. |
| Local sentinel bundle | `scripts/performance_sentinels.sh` | Owns local sentinel rows. Existing hard timing behavior is limited to the wall-check lane and S6 local selected smoke ceiling; S2/S3 are threshold-free backend-context rows. | Potential contrast: local thresholded behavior exists, but it is not the hosted selected benchmark freshness policy. |
| Make targets | `Makefile` | `bench-canonical-report`, `bench-canonical-report-freshness`, `bench-canonical-report-freshness-tests`, and `performance-sentinels` are the central benchmark methodology targets. | Likely validation and wiring surface. |
| Benchmark docs | `benchmarks/README.md` | Documents benchmark result interpretation, canonical report fields, selected freshness route, Linux/macOS hosted selected threshold-free lanes, sentinel interpretation, and report-index handoff. | Main user-facing benchmark methodology owner. |
| Top-level docs | `README.md` | Routes users to benchmark commands and states that hosted selected-performance lanes check freshness/methodology only, without timing thresholds or portable performance claims. | High-visibility claim boundary. |
| Support/readiness matrix | `INSTALL.md` | Classifies Linux/macOS selected performance freshness as hosted evidence with no portable performance, timing threshold, release benchmark, platform parity, package/ABI proof, package-manager distribution, or state-of-the-art claim. | Support matrix must stay aligned with the final policy. |
| Maintainer docs | `docs/maintainer_guide.md` | Describes selected performance evidence owners, validation commands, hosted lane metadata, threshold-free baseline/threshold fields, and repair guidance. | Maintainer repair and review surface. |
| Corpus docs/schema | `tests/corpus/README.md`, `tests/corpus/schemas/report_index_fields.md` | Document selected performance target and report-index field meanings, including threshold-free semantics. | Schema and corpus interpretation must match tooling. |
| Docs guard | `tests/test_selected_performance_docs.py` | Requires selected performance markers across README, INSTALL, benchmark README, maintainer guide, corpus README, and report-index schema; rejects selected performance overclaims and hosted timing-gate wording. | Existing guard to extend for final policy wording. |
| Freshness tests | `tests/test_bench_canonical_freshness.py` | Exercises local and hosted checker behavior, selected row mutation failures, manifest agreement, artifact checks, and methodology fields. | Existing regression suite for metadata changes. |
| Manifest tests | `tests/test_selected_report_targets_manifest.py` | Guards selected target manifest consistency across report and platform evidence rows. | Likely owner for selected manifest claim-scope and non-claim requirements. |
| Hosted Linux lane | `.github/workflows/ci.yml` | Contains Linux reviewed hosted selected performance freshness lane and artifact `sprint168-selected-performance-freshness`. | Hosted selected evidence source. |
| Hosted macOS lane | `.github/workflows/macos-ci.yml` | Contains macOS reviewed hosted selected performance freshness lane and artifact `sprint202-macos-selected-performance-freshness`. | Hosted selected evidence source added by Sprint 202. |

### Current Selected Benchmark Boundary

| Field | Current value or policy |
| --- | --- |
| Selected target id | `SRT-BENCH-REFACTOR-CSC-NOS4` |
| Benchmark artifact | `bench_refactor_csc` |
| Selected path | `build/bench-reports/canonical/bench_refactor_csc.csv` |
| Command | `tests/data/suitesparse/nos4.mtx --repeat 1` |
| Fixture | `nos4.mtx` |
| Repeat semantics | `configured_repeat_1` |
| Warmup | `none_configured` |
| Variance | `not_computed_single_sample` |
| Baseline | `n/a` for canonical selected freshness |
| Threshold | `n/a` for canonical selected freshness |
| Local claim boundary | `local_threshold_free` |
| Hosted claim boundary | `hosted_selected_threshold_free` |
| Hosted selected platforms | Linux and macOS |
| Explicit non-claims | No portable performance claim, release benchmark claim, algorithmic superiority claim, platform parity, state-of-the-art claim, package or ABI support claim, broad package-manager distribution claim, or Windows selected benchmark freshness. |

### Initial Threshold Eligibility Observations

| Topic | Day 1 observation | Threshold implication |
| --- | --- | --- |
| Runner metadata | Hosted mode requires non-local runner context; Linux and macOS lanes name GitHub-hosted contexts. | Runner context exists, but GitHub-hosted CPU assignment is explicitly variable. |
| Compiler/platform metadata | Canonical reports record platform and compiler strings. | Present as context, not normalized into a stable machine class. |
| Repeat count | Selected canonical row uses `--repeat 1` and records `configured_repeat_1`. | Single sample is weak evidence for a stable threshold. |
| Warmup | Current selected row records `none_configured`. | A hard threshold would need either accepted no-warmup rationale or a new warmup policy. |
| Variance | Current selected row records `not_computed_single_sample`. | This blocks a statistically meaningful threshold unless Sprint 212 deliberately chooses a smoke ceiling policy with clear limits. |
| Baseline/threshold | Canonical selected freshness records `baseline=n/a` and `threshold=n/a`. | Current hosted selected freshness is threshold-free by design. |
| Existing thresholds | `wall-check` and S6 local selected smoke ceiling exist in sentinel tooling. | Existing local thresholds are not the same as hosted selected canonical freshness and cannot be promoted without a decision. |
| Documentation boundary | User and maintainer docs repeatedly state no timing threshold or portable performance claim for hosted selected freshness. | Any thresholded policy must update docs and guards exactly; otherwise threshold-free deferral should be strengthened. |

### Decision Log

| Day | Decision | Rationale |
| --- | --- | --- |
| 1 | No threshold decision yet. | Day 1 is intake only. Current evidence shows a deliberate threshold-free hosted selected benchmark freshness policy, so Days 2-5 must prove any thresholded gate is justified before implementation. |

### Validation Matrix

| Validation | Day 1 status | Notes |
| --- | --- | --- |
| `git diff --check` | Planned for Day 1 closeout. | Documentation-only Day 1 changes. |
| `python3 tests/test_selected_performance_docs.py` | Planned for later days. | Day 1 did not change selected performance docs outside planning artifacts. |
| `python3 tests/test_bench_canonical_freshness.py` | Planned for later days. | Requires/generated benchmark report paths; use when methodology tooling changes. |
| `python3 tests/test_selected_report_targets_manifest.py` | Planned for later days. | Use when manifest or claim-scope fields change. |
| `make bench-canonical-report-freshness` | Planned for baseline/implementation days. | Can compile/run benchmark binaries; not needed for Day 1 planning-only changes. |
| `make performance-sentinels` | Decision-dependent. | Relevant if threshold policy touches local sentinel interpretation. |
| `make format && make lint && make test` | Not required for Day 1. | No `.c` or `.h` files changed. |

### Risk Register

| Risk | Why it matters | Mitigation |
| --- | --- | --- |
| Promoting a threshold without stable methodology | GitHub-hosted runners and single-sample measurements can produce noisy timing. | Require explicit runner/compiler/repeat/warmup/variance/baseline/threshold evidence before thresholded implementation. |
| Confusing freshness with performance proof | Existing selected hosted lanes check artifact and metadata freshness, not timing superiority. | Keep `hosted_selected_threshold_free` wording until a later decision explicitly changes it. |
| Reusing local sentinel thresholds as hosted proof | S5/S6 are local bounded gates with different policy meaning from canonical hosted selected freshness. | Keep sentinel and canonical selected evidence separate in docs, manifest, and tests. |
| Documentation overclaim | README/INSTALL/benchmark docs are high-visibility support surfaces. | Extend docs guards when final wording changes. |
| Manifest/docs drift | Selected target manifest is the source of truth, but docs repeat selected row identity and non-claims. | Tie final policy to manifest tests and selected performance docs tests. |
| Narrow implementation without regression fixtures | Methodology guards can be bypassed by stale metadata or missing fields. | Add positive and negative fixtures for the selected policy on Days 7-9. |

### Changed Surface Tracker

| Path | Day | Change type | Notes |
| --- | --- | --- | --- |
| `docs/planning/EPIC_19/SPRINT_212/WORKING_NOTES.md` | 1 | Added | Sprint ledger, evidence inventory, risk register, validation matrix. |
| `docs/planning/EPIC_19/SPRINT_212/artifacts/day1-benchmark-evidence-intake.md` | 1 | Added | Day 1 intake artifact. |
| `docs/planning/EPIC_19/SPRINT_212/WORKING_NOTES.md` | 2 | Updated | Current benchmark baseline commands, selected row metadata, hosted/local distinction, and threshold blockers. |
| `docs/planning/EPIC_19/SPRINT_212/artifacts/day2-current-benchmark-baseline.md` | 2 | Added | Day 2 baseline artifact. |
| `docs/planning/EPIC_19/SPRINT_212/WORKING_NOTES.md` | 3 | Updated | Runner/variance inventory, retained hosted artifact sample table, threshold metadata minimums, and candidate assessment. |
| `docs/planning/EPIC_19/SPRINT_212/artifacts/day3-runner-and-variance-inventory.md` | 3 | Added | Day 3 runner and variance artifact. |
| `docs/planning/EPIC_19/SPRINT_212/WORKING_NOTES.md` | 4 | Updated | Thresholded and threshold-free decision criteria, stop conditions, and evidence-to-test mapping. |
| `docs/planning/EPIC_19/SPRINT_212/artifacts/day4-decision-criteria.md` | 4 | Added | Day 4 decision criteria artifact. |
| `docs/planning/EPIC_19/SPRINT_212/WORKING_NOTES.md` | 5 | Updated | Product policy decision, rejected alternatives, implementation scope, non-claims, and Day 6 handoff. |
| `docs/planning/EPIC_19/SPRINT_212/artifacts/day5-product-policy-decision.md` | 5 | Added | Day 5 product decision artifact. |
| `docs/planning/EPIC_19/SPRINT_212/WORKING_NOTES.md` | 6 | Updated | Methodology field design, source-of-truth map, allowed values, forbidden wording, fixture plan, and Day 7 handoff. |
| `docs/planning/EPIC_19/SPRINT_212/artifacts/day6-methodology-schema-design.md` | 6 | Added | Day 6 implementation-ready schema design artifact. |
| `scripts/check_bench_canonical_freshness.py` | 7 | Updated | Requires the selected methodology non-claim token and rejects explicit threshold/performance-promotion methodology tokens. |
| `tests/test_bench_canonical_freshness.py` | 7 | Updated | Adds negative fixtures for missing methodology non-claim token, threshold-promoting methodology token, and unselected hosted claim-boundary drift. |
| `docs/planning/EPIC_19/SPRINT_212/WORKING_NOTES.md` | 7 | Updated | Day 7 implementation and validation notes. |
| `docs/planning/EPIC_19/SPRINT_212/artifacts/day7-tooling-implementation-batch-one.md` | 7 | Added | Day 7 tooling implementation artifact. |
| `tests/test_bench_canonical_freshness.py` | 8 | Updated | Adds exact selected threshold-free metadata assertions, broadens forbidden methodology-token coverage, and adds manifest methodology drift regression. |
| `docs/planning/EPIC_19/SPRINT_212/WORKING_NOTES.md` | 8 | Updated | Day 8 tooling completion notes and validation evidence. |
| `docs/planning/EPIC_19/SPRINT_212/artifacts/day8-tooling-implementation-batch-two.md` | 8 | Added | Day 8 tooling implementation completion artifact. |
| `tests/test_selected_report_targets_manifest.py` | 9 | Updated | Adds exact selected benchmark manifest contract checks and drift regressions for identity, workflow metadata, threshold claim scope, and non-claim exactness. |
| `docs/planning/EPIC_19/SPRINT_212/WORKING_NOTES.md` | 9 | Updated | Day 9 manifest/report guard notes and validation evidence. |
| `docs/planning/EPIC_19/SPRINT_212/artifacts/day9-manifest-and-report-guards.md` | 9 | Added | Day 9 manifest and report guard artifact. |
| `README.md` | 10 | Updated | Adds selected benchmark future timing-threshold prerequisite wording and preserves no-threshold/no-portable-performance boundary. |
| `INSTALL.md` | 10 | Updated | Adds support/readiness matrix prerequisite wording for any future selected performance timing-threshold promotion. |
| `benchmarks/README.md` | 10 | Updated | Adds selected methodology fields, required non-portable methodology note, and future timing-threshold prerequisite wording. |
| `tests/test_selected_performance_docs.py` | 10 | Updated | Requires the new user-facing markers and adds a missing-prerequisite regression. |
| `docs/planning/EPIC_19/SPRINT_212/WORKING_NOTES.md` | 10 | Updated | Day 10 documentation calibration notes and validation evidence. |
| `docs/planning/EPIC_19/SPRINT_212/artifacts/day10-documentation-calibration.md` | 10 | Added | Day 10 user-facing documentation calibration artifact. |
| `docs/maintainer_guide.md` | 11 | Updated | Adds selected benchmark methodology-note requirements, forbidden-token interpretation, future threshold prerequisites, and repair workflow. |
| `docs/planning/EPIC_19/PROJECT_PLAN.md` | 11 | Updated | Marks Sprint 212 in progress with Day 1-11 evidence instead of pending future execution. |
| `tests/test_selected_performance_docs.py` | 11 | Updated | Requires maintainer repair and future threshold-prerequisite markers. |
| `docs/planning/EPIC_19/SPRINT_212/WORKING_NOTES.md` | 11 | Updated | Day 11 maintainer documentation notes, changed-file snapshot, and validation evidence. |
| `docs/planning/EPIC_19/SPRINT_212/artifacts/day11-maintainer-documentation.md` | 11 | Added | Day 11 maintainer documentation artifact. |
| `docs/planning/EPIC_19/SPRINT_212/WORKING_NOTES.md` | 12 | Updated | Integrated validation command record and C-gate decision. |
| `docs/planning/EPIC_19/SPRINT_212/artifacts/day12-integrated-validation.md` | 12 | Added | Day 12 integrated validation artifact. |
| `scripts/check_bench_canonical_freshness.py` | 13 | Updated | Normalizes methodology-note tokens before required/forbidden token enforcement. |
| `tests/test_bench_canonical_freshness.py` | 13 | Updated | Adds spaced forbidden methodology-token regression. |
| `tests/test_selected_performance_docs.py` | 13 | Updated | Catches hyphenated selected timing-threshold overclaim wording. |
| `docs/planning/EPIC_19/SPRINT_212/WORKING_NOTES.md` | 13 | Updated | Day 13 review-hardening findings, fixes, and validation evidence. |
| `docs/planning/EPIC_19/SPRINT_212/artifacts/day13-review-hardening.md` | 13 | Added | Day 13 review-hardening artifact. |
| `docs/planning/EPIC_19/PROJECT_PLAN.md` | 14 | Updated | Marks Sprint 212 closed with Day 1-14 evidence. |
| `docs/planning/EPIC_19/SPRINT_212/WORKING_NOTES.md` | 14 | Updated | Final item reconciliation, validation summary, residuals, and closeout status. |
| `docs/planning/EPIC_19/SPRINT_212/artifacts/day14-closeout-review.md` | 14 | Added | Day 14 closeout review artifact. |

### Open Questions For Day 2

1. Do current hosted Linux/macOS artifacts expose enough exact runner/compiler
   metadata to classify a repeatable threshold machine class?
2. Is `configured_repeat_1` acceptable only for freshness, or can it support a
   deliberately broad smoke ceiling with clear non-portable wording?
3. Should Sprint 212 evaluate the existing S6 local selected smoke ceiling as
   the only thresholded candidate, or keep canonical hosted selected freshness
   entirely threshold-free?
4. Which missing-metadata regressions should be added regardless of the final
   threshold decision?
5. Which docs surfaces will need exact text updates after the Day 5 policy
   decision?

### Day 1 Validation

Commands planned for Day 1 closeout:

```sh
git diff --check
git status --short
git diff --name-only -- '*.c' '*.h'
```

Day 1 changes planning documentation only. No `.c` or `.h` files are modified,
so the full C quality gate is not required.

## Day 2: Current Benchmark Baseline

### Baseline Commands

| Command | Result | Evidence captured |
| --- | --- | --- |
| `python3 tests/test_selected_performance_docs.py` | Pass: `test-selected-performance-docs: ok` | Current user-facing, maintainer-facing, corpus, and schema docs contain required selected performance markers and reject overclaims. |
| `python3 tests/test_selected_report_targets_manifest.py` | Pass: `test-selected-report-targets-manifest: ok` | Selected report target manifest rows remain internally consistent. |
| `make bench-canonical-report-freshness` | Pass: selected local canonical benchmark freshness passed for `bench_refactor_csc`. | Generated local canonical bundle under ignored `build/bench-reports/canonical/`; selected row metadata snapshot recorded below. |
| `git diff --check` | Planned for Day 2 closeout. | Documentation-only tracked changes. |

The generated benchmark bundle is ignored build output and was not added to
the branch. It is recorded here as local baseline evidence only.

### Local Generated Selected Row Snapshot

`make bench-canonical-report-freshness` generated four canonical rows and one
selected row for `bench_refactor_csc`.

| Field | Local baseline value |
| --- | --- |
| `surface` | `canonical` |
| `category` | `measurement` |
| `report_label` | `unlabeled` |
| `git_commit` | `dbaa05c4` |
| `git_branch` | `sprint-212` |
| `platform` | Darwin local workstation context |
| `compiler` | Apple clang version 17.0.0 |
| `runner_context` | `local` |
| `build_flags` | `not_recorded` |
| `cpu_model` | `unknown` |
| `build_mode` | `serial` |
| `omp_num_threads` | `unset` |
| `artifact` | `bench_refactor_csc` |
| `relative_path` | `bench_refactor_csc.csv` |
| `command` | `tests/data/suitesparse/nos4.mtx --repeat 1` |
| `report_family` | `benchmark` |
| `status` | `measurement` |
| `support_tier` | `local_only` |
| `claim_boundary` | `local_threshold_free` |
| `fixture_or_workload` | `nos4.mtx` |
| `matrix_size` | `n=100` |
| `repeat_semantics` | `configured_repeat_1` |
| `warmup` | `none_configured` |
| `variance` | `not_computed_single_sample` |
| `baseline` | `n/a` |
| `threshold` | `n/a` |
| `backend_context` | `n/a` |
| `methodology_notes` | `threshold_free_local_measurement;not_portable_performance_claim` |

The generated selected CSV row reported one local timing sample for the
selected `chol_spd` scenario on `nos4.mtx` with residuals around `1e-15`.
Those timing values are local branch evidence only and are not a threshold,
baseline, or portable performance claim.

### Hosted Lane Baseline

| Platform | Workflow job | Artifact | Hosted metadata | Day 2 interpretation |
| --- | --- | --- | --- | --- |
| Linux | `.github/workflows/ci.yml::hosted-performance-freshness` | `sprint168-selected-performance-freshness` | `BENCH_CANONICAL_REPORT_LABEL=sprint-168-hosted-performance`, `SPARSE_CANONICAL_SUPPORT_TIER=hosted_selected`, `SPARSE_CANONICAL_CLAIM_BOUNDARY=hosted_selected_threshold_free`, `SPARSE_CANONICAL_RUNNER_CONTEXT=github-actions-ubuntu-latest`, `SPARSE_CANONICAL_BUILD_FLAGS=default_make_flags`, `SPARSE_CANONICAL_BUILD_MODE=serial`, CPU model captured from `/proc/cpuinfo`. | Reviewed hosted freshness evidence for the exact selected row only; not a timing threshold. |
| macOS | `.github/workflows/macos-ci.yml::selected-performance-freshness` | `sprint202-macos-selected-performance-freshness` | `BENCH_CANONICAL_REPORT_LABEL=sprint202-macos-hosted-performance`, `SPARSE_CANONICAL_SUPPORT_TIER=hosted_selected`, `SPARSE_CANONICAL_CLAIM_BOUNDARY=hosted_selected_threshold_free`, `SPARSE_CANONICAL_RUNNER_CONTEXT=github-actions-macos-latest`, `SPARSE_CANONICAL_BUILD_FLAGS=default_make_flags`, `SPARSE_CANONICAL_BUILD_MODE=serial`, CPU model captured from `sysctl`. | Reviewed hosted freshness evidence for the exact selected row only; not Linux/macOS timing comparability. |

Both hosted workflows explicitly state that the selected performance lane
checks methodology metadata, selected row identity, artifact paths, and
threshold-free claim boundaries. Both say it is not a timing threshold,
portable performance claim, broad benchmark-family publication, package/ABI
claim, or state-of-the-art sparse linear algebra claim.

### Current Documentation Wording

| Document | Current Day 2 wording baseline |
| --- | --- |
| `README.md` | Routes selected performance evidence through `make bench-canonical-report-freshness`; states reviewed Linux/macOS hosted lanes check artifact presence, selected row identity, methodology metadata, manifest agreement, and `hosted_selected_threshold_free` boundaries without timing thresholds or portable performance claims. |
| `INSTALL.md` | Support/readiness matrix lists Linux/macOS selected performance freshness as hosted evidence with no portable performance, timing threshold, release benchmark, platform parity, package/ABI proof, broad package-manager distribution, or state-of-the-art claim. |
| `benchmarks/README.md` | Describes canonical report metadata, selected row identity, hosted mode requirements, `baseline=n/a`, `threshold=n/a`, `warmup=none_configured`, `variance=not_computed_single_sample`, and the distinction between canonical selected freshness and local sentinel thresholds. |
| `docs/maintainer_guide.md` | States selected hosted performance freshness should remain `baseline=n/a`, `threshold=n/a`, and `status=measurement` until a future sprint records hosted-runner baseline, variance model, tolerance, and same-machine comparison policy. |
| `tests/corpus/README.md` | Defines `SRT-BENCH-REFACTOR-CSC-NOS4` as threshold-free methodology/freshness evidence only. |
| `tests/corpus/schemas/report_index_fields.md` | Documents selected benchmark policy as `status=measurement`, `baseline=n/a`, `threshold=n/a`, `warmup=none_configured`, and `variance=not_computed_single_sample`. |

### Threshold Blockers Found On Day 2

| Blocker | Evidence | Policy implication |
| --- | --- | --- |
| Single selected sample | Selected command is `nos4.mtx --repeat 1` and `repeat_semantics=configured_repeat_1`. | Does not support statistical variance or a stable timing threshold by itself. |
| No warmup policy | Current selected row records `warmup=none_configured`. | A thresholded policy would need a deliberate no-warmup smoke-ceiling rationale or a new warmup design. |
| No computed variance | Current selected row records `variance=not_computed_single_sample`. | Blocks a statistically meaningful threshold until repeat/variance evidence exists. |
| No canonical baseline or threshold | Current canonical selected freshness records `baseline=n/a` and `threshold=n/a`; docs guard this wording. | Thresholded implementation would require a policy change across scripts, manifest, docs, and tests. |
| Hosted CPU assignment is context, not stable class | Maintainer guide says GitHub-hosted CPU assignment can vary; workflows capture CPU model as metadata. | Hosted threshold would need a same-machine or machine-class policy before becoming more than a broad smoke ceiling. |
| Local sentinel thresholds are separate | `performance-sentinels` owns S5 wall-check and S6 local selected smoke ceiling separately from canonical hosted selected freshness. | Existing local thresholds cannot be treated as hosted selected benchmark publication proof. |

### Day 2 Validation

Day 2 changed planning documentation only. No `.c` or `.h` files are modified,
so the full C quality gate is not required.

Commands for Day 2 closeout:

```sh
python3 tests/test_selected_performance_docs.py
python3 tests/test_selected_report_targets_manifest.py
make bench-canonical-report-freshness
git diff --check
git status --short
git diff --name-only -- '*.c' '*.h'
```

## Day 3: Runner And Variance Inventory

### Historical Hosted Artifact Inspection

Day 3 inspected retained GitHub Actions artifacts for the selected benchmark
target using `gh run download` and the known artifact names from the selected
target manifest:

- Linux artifact: `sprint168-selected-performance-freshness`
- macOS artifact: `sprint202-macos-selected-performance-freshness`

The latest completed PR runs and three earlier retained PR attempts were
available. Each artifact passed the hosted freshness checker when inspected
with `scripts/check_bench_canonical_freshness.py --mode hosted` for the latest
downloaded Linux and macOS samples.

### Runner And Methodology Matrix

| Lane | Runner context | OS/platform evidence | Compiler evidence | CPU disclosure | Command | Repeat | Warmup | Variance | Retention | Claim boundary |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Local Day 2 baseline | `local` | Darwin local workstation | Apple clang 17.0.0 | `unknown` | `tests/data/suitesparse/nos4.mtx --repeat 1` | `configured_repeat_1` | `none_configured` | `not_computed_single_sample` | ignored local `build/` output | `local_threshold_free` |
| Linux hosted selected | `github-actions-ubuntu-latest` | Ubuntu/Linux hosted runner | GCC 13.3.0 through `cc` | Captured from `/proc/cpuinfo`; observed EPYC 9V74, EPYC 7763, and Intel 8573C across retained runs. | `tests/data/suitesparse/nos4.mtx --repeat 1` | `configured_repeat_1` | `none_configured` | `not_computed_single_sample` | workflow artifact retention is 7 days | `hosted_selected_threshold_free` |
| macOS hosted selected | `github-actions-macos-latest` | macOS hosted runner | Apple clang 21.0.0 in retained hosted artifacts | Captured from `sysctl`; observed `Apple M1 (Virtual)` across retained runs. | `tests/data/suitesparse/nos4.mtx --repeat 1` | `configured_repeat_1` | `none_configured` | `not_computed_single_sample` | workflow artifact retention is 7 days | `hosted_selected_threshold_free` |

### Retained Hosted Timing Samples

These values are measurements from retained hosted artifacts, not threshold
evidence. The commits differ across PR attempts, so the table is useful for
runner and variance risk, not for a controlled same-commit regression model.

| Platform | Run id | Commit | CPU model | `refactor_csc_ms` | `speedup_refactor` |
| --- | --- | --- | --- | ---: | ---: |
| Linux | `36798099138` | `1e39d4c` | AMD EPYC 9V74 80-Core Processor | 0.058 | 1.14 |
| Linux | `36793801250` | `1d3af74` | AMD EPYC 7763 64-Core Processor | 0.047 | 1.78 |
| Linux | `36791438422` | `4105b9c` | AMD EPYC 7763 64-Core Processor | 0.047 | 1.79 |
| Linux | `36790897256` | `0eb0e97` | Intel Xeon Platinum 8573C | 0.061 | 1.35 |
| macOS | `36798099076` | `1e39d4c` | Apple M1 (Virtual) | 0.032 | 1.72 |
| macOS | `36793801179` | `1d3af74` | Apple M1 (Virtual) | 0.030 | 1.80 |
| macOS | `36791438432` | `4105b9c` | Apple M1 (Virtual) | 0.037 | 2.11 |
| macOS | `36790897338` | `0eb0e97` | Apple M1 (Virtual) | 0.091 | 1.48 |

Day 3 variance observations:

- Linux retained samples range from `0.047` to `0.061` ms for
  `refactor_csc_ms`, with CPU model changes across hosted runs.
- macOS retained samples range from `0.030` to `0.091` ms for
  `refactor_csc_ms`, including one outlier roughly 3x the fastest retained
  sample even though the CPU label remains `Apple M1 (Virtual)`.
- All retained selected rows still record `warmup=none_configured`,
  `variance=not_computed_single_sample`, `baseline=n/a`, and `threshold=n/a`.
- All retained selected rows are single-row, single-repeat measurements; none
  provide repeated samples, confidence intervals, or a same-run distribution.

### Minimum Metadata For A Thresholded Gate

| Requirement | Why it is needed | Current evidence |
| --- | --- | --- |
| Exact selected row identity | Avoid thresholding the wrong workload. | Present: selected target and checker require `bench_refactor_csc`, `nos4.mtx`, `--repeat 1`, and one selected row. |
| Stable runner class or same-machine policy | Prevent hosted machine churn from becoming false failures. | Missing for hosted thresholding: runner labels exist, but CPU assignment varies and hosted machines are not normalized into a stable class. |
| Compiler/build/thread policy | Keep timing comparisons meaningful. | Partially present: compiler, build flags, build mode, and `OMP_NUM_THREADS` are recorded. |
| Repeat count and sample count | Support threshold confidence and outlier handling. | Missing: selected row records `configured_repeat_1`. |
| Warmup policy | Avoid first-run effects becoming the measured signal. | Missing: selected row records `none_configured`. |
| Variance rule | Decide when a run is noisy versus regressed. | Missing: selected row records `not_computed_single_sample`. |
| Baseline provenance | Identify what timing baseline is authoritative. | Missing for canonical selected freshness: `baseline=n/a`. |
| Threshold value and tolerance | Define pass/fail behavior and maintenance expectations. | Missing for canonical selected freshness: `threshold=n/a`. |
| Artifact retention and comparison window | Make failures auditable after the run. | Weak for long-term thresholding: hosted artifacts use 7-day retention. |
| Claim-boundary wording | Prevent a local or hosted smoke gate from becoming a portable performance claim. | Present for threshold-free policy; would need updates if thresholded. |

### Threshold Candidate Assessment

| Candidate | Eligibility | Reason |
| --- | --- | --- |
| Hosted Linux selected canonical row | Blocked for Day 3 thresholded gate. | CPU model changed across retained Linux samples; single repeat, no warmup, no variance, no baseline, and no threshold. |
| Hosted macOS selected canonical row | Blocked for Day 3 thresholded gate. | Same virtual CPU label still produced a visible timing outlier; single repeat, no warmup, no variance, no baseline, and no threshold. |
| Cross-platform Linux/macOS selected canonical row | Rejected. | Cross-platform timing comparability is explicitly unclaimed and contradicted by different runner/CPU/compiler contexts. |
| Existing local S6 selected smoke ceiling | Plausible only as a local smoke gate, not hosted selected publication. | It already exists under `performance-sentinels` with local-only semantics and should remain separate unless Sprint 212 deliberately chooses to strengthen that exact local policy. |
| Stronger threshold-free hosted selected policy | Best-supported Day 3 path. | Current evidence is already structured around freshness/methodology validation and has clear blockers for timing thresholds. |

### Day 3 Validation

Day 3 changed planning documentation only. No `.c` or `.h` files are modified,
so the full C quality gate is not required.

Commands for Day 3 closeout:

```sh
gh run list --limit 20 --json databaseId,workflowName,displayTitle,headBranch,headSha,status,conclusion,createdAt,updatedAt,event
gh run view 36798099138 --json jobs,conclusion,workflowName,headSha,createdAt
gh run view 36798099076 --json jobs,conclusion,workflowName,headSha,createdAt
gh run download 36798099138 --name sprint168-selected-performance-freshness --dir <tmpdir>
gh run download 36798099076 --name sprint202-macos-selected-performance-freshness --dir <tmpdir>
python3 scripts/check_bench_canonical_freshness.py --report-dir <linux_tmpdir> --mode hosted
python3 scripts/check_bench_canonical_freshness.py --report-dir <macos_tmpdir> --mode hosted
git diff --check
git status --short
git diff --name-only -- '*.c' '*.h'
```

## Day 4: Threshold Decision Criteria

### Decision Rule

Sprint 212 must select exactly one policy on Day 5:

1. **Thresholded selected gate** only if every required threshold criterion is
   satisfied with current branch evidence and can be enforced by focused tests
   without a portable performance overclaim.
2. **Threshold-free deferral hardening** if any threshold criterion is missing,
   ambiguous, hosted-only, or dependent on unretained/generated evidence.

The default is threshold-free deferral unless the thresholded branch has
complete runner, compiler, repeat, warmup, variance, baseline, threshold,
artifact, retention, and claim-boundary evidence.

### Thresholded-Gate Acceptance Criteria

| Criterion | Required evidence | Enforcement owner | Day 4 status |
| --- | --- | --- | --- |
| Exact selected key | `SRT-BENCH-REFACTOR-CSC-NOS4`, `bench_refactor_csc`, `nos4.mtx`, `--repeat 1`, one selected row. | `tests/corpus/manifests/selected_report_targets.tsv`; `scripts/check_bench_canonical_freshness.py`; `tests/test_bench_canonical_freshness.py`. | Present. |
| Runner policy | Same-machine, stable machine class, or deliberately local-only smoke ceiling with a named runner boundary. | Benchmark methodology docs; manifest fields; freshness checker. | Missing for hosted thresholding; possible only for local smoke ceiling. |
| Compiler/build/thread policy | Exact compiler family/version, build flags, build mode, and `OMP_NUM_THREADS` policy. | Report generator/checker; manifest tests. | Partially present as metadata, not yet a threshold policy. |
| Repeat and sample policy | More than one sample or an explicitly accepted smoke-only threshold that rejects statistical interpretation. | Report generator/checker; benchmark docs. | Missing for hosted thresholding; selected row is `configured_repeat_1`. |
| Warmup policy | Warmup count, no-warmup rationale, or explicit smoke-only policy. | Report generator/checker; docs guard. | Missing for hosted thresholding; current value is `none_configured`. |
| Variance rule | Variance/outlier rule, confidence policy, or explicit smoke-only non-statistical policy. | Freshness tests; methodology docs. | Missing; current value is `not_computed_single_sample`. |
| Baseline provenance | Source-controlled or artifact-retained baseline with owner, date, runner scope, and update process. | Manifest/schema docs; maintainer guide; regression tests. | Missing for canonical selected freshness. |
| Threshold value | Numeric threshold or allowed regression rule with units and repair workflow. | Freshness checker; regression fixtures; docs. | Missing for canonical selected freshness. |
| Artifact retention | Evidence remains inspectable long enough for review and repair, or baseline is source-controlled. | Workflow artifact settings or source-controlled baseline. | Weak: current hosted artifacts retain for 7 days. |
| Claim boundary | Explicit non-portable wording and forbidden-claim guards. | `tests/test_selected_performance_docs.py`; README/INSTALL/benchmark docs. | Present for threshold-free policy; would need threshold-specific wording. |

Acceptance outcome: the hosted selected threshold branch cannot pass these
criteria with Day 1-3 evidence. A local S6 smoke-ceiling policy can proceed to
Day 5 only if it is explicitly scoped as local, non-hosted, non-portable,
non-statistical, and separate from canonical hosted selected freshness.

### Threshold-Free Deferral Acceptance Criteria

| Criterion | Required evidence | Enforcement owner |
| --- | --- | --- |
| Selected row remains fresh | Local and hosted selected freshness continue to require exact selected row identity and required artifacts. | `scripts/check_bench_canonical_freshness.py`; `tests/test_bench_canonical_freshness.py`. |
| Threshold-free fields remain exact | `baseline=n/a`, `threshold=n/a`, `status=measurement`, `warmup=none_configured`, and `variance=not_computed_single_sample` remain required for canonical selected freshness. | Freshness checker and tests. |
| Hosted/local distinction remains explicit | Local rows stay `local_threshold_free`; hosted selected rows stay `hosted_selected_threshold_free`; unselected rows stay local-only. | Freshness checker; manifest tests. |
| Stale threshold wording is rejected | Docs and schema must not imply selected hosted timing gates, portable speed, broad benchmark publication, or cross-platform performance parity. | `tests/test_selected_performance_docs.py`; docs assertions. |
| Existing local sentinel gates stay separate | S5 wall-check and S6 selected smoke ceiling remain local sentinel policy, not hosted selected canonical publication. | `benchmarks/README.md`; maintainer guide; sentinel docs/tests if changed. |
| Missing metadata fails closed | Any missing selected row, missing artifact, local hosted placeholder, stale support tier, or stale claim boundary fails. | Freshness checker and manifest tests. |
| Documentation explains future promotion requirements | Maintainer docs name the evidence required before any future threshold promotion. | `docs/maintainer_guide.md`; benchmark README. |

Acceptance outcome: the threshold-free branch can pass if Sprint 212 strengthens
guards and documentation around the already-observed blockers instead of
creating a timing threshold.

### Stop Conditions

Sprint 212 must stop and ask for direction instead of implementing a policy if
any of these happen:

| Stop condition | Why it stops the sprint |
| --- | --- |
| Hosted artifacts or workflow metadata contradict the selected target manifest. | The source of truth would be ambiguous. |
| A threshold is requested without baseline provenance and update policy. | That would create an unreviewable pass/fail gate. |
| A threshold is requested for Linux/macOS combined timing values. | Cross-platform timing comparability is explicitly unclaimed. |
| A threshold requires changing benchmark behavior or solver behavior. | Sprint 212 is methodology policy, not solver implementation. |
| A threshold requires long-lived hosted history that is not retained or source-controlled. | Failures would not be auditable after retention expires. |
| Documentation cannot preserve non-portable/non-superiority wording. | The policy would overclaim performance support. |
| Required checks fail during implementation. | User instructions require stopping on failed quality checks. |

### Criteria-To-Test Mapping

| Decision criterion | Test/script/docs assertion |
| --- | --- |
| Selected target identity | `tests/test_bench_canonical_freshness.py`; `scripts/check_bench_canonical_freshness.py`; selected manifest tests. |
| Hosted metadata non-locality | Hosted mode checks in `scripts/check_bench_canonical_freshness.py`; hosted regression tests. |
| Threshold-free exact fields | Existing freshness tests for `baseline=n/a`, `threshold=n/a`, status, warmup, and variance; add tests if wording changes. |
| No portable performance overclaim | `tests/test_selected_performance_docs.py`; README/INSTALL/benchmark/maintainer markers. |
| Manifest non-claims | `tests/test_selected_report_targets_manifest.py`. |
| Sentinel/canonical separation | Benchmark README and maintainer guide assertions; add docs-guard markers if Day 5 chooses threshold-free hardening. |
| Future threshold prerequisites | Maintainer guide wording plus selected performance docs markers. |

### Day 4 Validation

Day 4 changed planning documentation only. No `.c` or `.h` files are modified,
so the full C quality gate is not required.

Commands for Day 4 closeout:

```sh
git diff --check
git status --short
git diff --name-only -- '*.c' '*.h'
```

## Day 5: Product Policy Decision

### Decision

Sprint 212 will implement **threshold-free deferral hardening** for the
selected hosted canonical benchmark freshness policy.

It will not add a hosted selected timing threshold for
`SRT-BENCH-REFACTOR-CSC-NOS4` in this sprint. The evidence supports
freshness, artifact, and methodology validation for the selected row, but not
a hosted pass/fail timing gate.

### Selected Scope

| Field | Selected policy |
| --- | --- |
| Selected target | `SRT-BENCH-REFACTOR-CSC-NOS4` |
| Benchmark row | `bench_refactor_csc` |
| Workload | `tests/data/suitesparse/nos4.mtx --repeat 1` |
| Hosted platforms | Linux and macOS reviewed selected lanes only |
| Policy type | Threshold-free freshness and methodology hardening |
| Required canonical fields | `status=measurement`, `baseline=n/a`, `threshold=n/a`, `warmup=none_configured`, `variance=not_computed_single_sample` |
| Required local boundary | `local_threshold_free` |
| Required hosted boundary | `hosted_selected_threshold_free` |
| Explicitly separate local gates | Existing `performance-sentinels` S5/S6 local threshold behavior remains separate local regression governance. |

### Rationale

| Evidence | Decision impact |
| --- | --- |
| Day 2 local baseline generated a fresh selected row with `baseline=n/a`, `threshold=n/a`, `warmup=none_configured`, and `variance=not_computed_single_sample`. | Confirms current policy is threshold-free by construction. |
| Day 3 retained Linux hosted artifacts showed CPU model changes under the same `ubuntu-latest` runner label. | Blocks a hosted Linux threshold without a stable runner class or same-machine policy. |
| Day 3 retained macOS hosted artifacts showed a timing outlier under the same `Apple M1 (Virtual)` CPU label. | Blocks a hosted macOS threshold without repeat/variance/outlier policy. |
| All selected hosted samples use `configured_repeat_1`. | Blocks statistical thresholding. |
| Current docs and guards already reject timing-threshold and portable-performance claims. | Favors strengthening the existing truthful policy instead of creating a new threshold surface. |
| Existing S6 local selected smoke ceiling is already local-only and sentinel-scoped. | Not a substitute for hosted selected canonical freshness. |

### Rejected Alternatives

| Alternative | Decision | Reason |
| --- | --- | --- |
| Hosted Linux selected threshold | Rejected for Sprint 212. | CPU variability, single sample, no warmup, no variance, no baseline, and no selected canonical threshold. |
| Hosted macOS selected threshold | Rejected for Sprint 212. | Retained timing outlier, single sample, no warmup, no variance, no baseline, and no selected canonical threshold. |
| Combined Linux/macOS threshold | Rejected outright. | Cross-platform timing comparability is explicitly unclaimed. |
| Promote S6 local smoke ceiling as hosted selected freshness | Rejected. | S6 is local sentinel governance, not hosted canonical selected publication. |
| Add a broad benchmark methodology threshold | Rejected. | Sprint 212 scope is one selected benchmark methodology policy, not broad performance governance. |

### Required Non-Claims

The selected policy must continue to reject:

- portable performance;
- timing threshold for hosted selected canonical freshness;
- release benchmark status;
- algorithmic superiority;
- platform parity;
- Windows selected benchmark freshness;
- package-manager distribution;
- package, ABI, shared-library, or runtime-loader proof;
- broad benchmark-family publication;
- backend superiority;
- OpenMP speedup;
- state-of-the-art performance.

### Implementation Plan

| Surface | Day 5 implementation direction |
| --- | --- |
| `scripts/check_bench_canonical_freshness.py` | Strengthen threshold-free metadata checks only if Day 6 finds gaps: exact field values, hosted/local boundary, unselected-row locality, and stale threshold rejection. |
| `tests/test_bench_canonical_freshness.py` | Add negative fixtures for stale threshold/baseline promotion, missing methodology notes, hosted/local claim drift, and unsupported threshold carryover as needed. |
| `tests/test_selected_report_targets_manifest.py` | Ensure selected benchmark manifest non-claims and workflow platform metadata preserve threshold-free hosted selected scope. |
| `tests/test_selected_performance_docs.py` | Add or refine docs markers for sentinel/canonical separation and future threshold prerequisites. |
| `benchmarks/README.md` | Keep canonical selected freshness threshold-free and clarify future threshold prerequisites if current wording is insufficient. |
| `README.md` and `INSTALL.md` | Preserve high-level no-threshold/no-portable-performance wording; update only if guard markers need exact scope. |
| `docs/maintainer_guide.md` | Document the Day 5 decision and the evidence required before any future threshold promotion. |
| `tests/corpus/README.md` and report schema docs | Preserve selected target field interpretation and threshold-free semantics. |

### Day 6 Handoff

Day 6 should design a threshold-free methodology schema with exact ownership
for:

- selected target identity;
- support tier and claim boundary;
- baseline and threshold values;
- warmup and variance values;
- methodology notes;
- sentinel/canonical separation;
- future threshold prerequisite wording.

Day 6 should not design a hosted timing threshold unless new evidence appears
that satisfies every Day 4 thresholded criterion.

### Day 5 Validation

Day 5 changed planning documentation only. No `.c` or `.h` files are modified,
so the full C quality gate is not required.

Commands for Day 5 closeout:

```sh
git diff --check
git status --short
git diff --name-only -- '*.c' '*.h'
```

## Day 6: Methodology Schema Design

### Selected Schema Decision

Sprint 212 will treat the selected benchmark methodology schema as a
threshold-free contract made of three aligned layers:

1. **Source-controlled selected target authority** in
   `tests/corpus/manifests/selected_report_targets.tsv`.
2. **Generated report metadata authority** in `index.tsv` and `manifest.txt`
   produced by `scripts/bench_canonical_report.sh`.
3. **Interpretation and claim-boundary authority** in `benchmarks/README.md`,
   `README.md`, `INSTALL.md`, `docs/maintainer_guide.md`,
   `tests/corpus/README.md`, and
   `tests/corpus/schemas/report_index_fields.md`.

The checker and tests should enforce alignment across those layers without
adding a timing threshold.

### Field Design

| Field | Required value or rule | Source of truth | Validation owner | Notes |
| --- | --- | --- | --- | --- |
| `target_id` | `SRT-BENCH-REFACTOR-CSC-NOS4` | Selected target manifest | `tests/test_bench_canonical_freshness.py`; manifest tests | Stable selected benchmark key. |
| `family` / `subfamily` | `benchmark` / `canonical` | Selected target manifest | Freshness tests | Prevents sentinel or guardrail rows from replacing canonical selected freshness. |
| `target_key` / `artifact` | `bench_refactor_csc` | Manifest and generated index | Freshness checker/tests | Selected row identity. |
| `relative_path` | `bench_refactor_csc.csv` | Manifest-derived checker contract | Freshness checker/tests | Rejects path drift and broad artifact publication. |
| `command` | `tests/data/suitesparse/nos4.mtx --repeat 1` | Manifest/checker contract | Freshness checker/tests | Keeps workload exact. |
| `fixture_or_workload` | `nos4.mtx` | Generated index and selected CSV | Freshness checker/tests | Must agree with CSV matrix. |
| `matrix_size` | `n=100` | Generated CSV-derived contract | Freshness checker/tests | Must agree with selected CSV `n`. |
| `status` | `measurement` | Generated index/manifest | Freshness checker/tests | Must not become `pass` or threshold proof. |
| `support_tier` | Local mode allows `local_only` or `hosted_selected`; hosted mode requires selected manifest `hosted_selected`; unselected rows require `local_only`. | Manifest and checker constants | Freshness checker/tests; manifest tests | Hosted selected support applies only to selected row. |
| `claim_boundary` | Local selected rows use allowed threshold-free boundaries; hosted mode requires `hosted_selected_threshold_free`; unselected rows require `local_threshold_free`. | Checker constants and docs | Freshness checker/tests; docs guard | No timing-threshold claim boundary. |
| `repeat_semantics` | `configured_repeat_1` | Generated index/manifest | Freshness checker/tests | Must not be described as statistical sampling. |
| `warmup` | `none_configured` | Generated index/manifest | Freshness checker/tests; docs guard | Must not be described as warmup-controlled. |
| `variance` | `not_computed_single_sample` | Generated index/manifest | Freshness checker/tests; docs guard | Must not be described as variance-controlled. |
| `baseline` | `n/a` | Generated index/manifest | Freshness checker/tests; docs guard | Must not become numeric baseline in canonical selected freshness. |
| `threshold` | `n/a` | Generated index/manifest | Freshness checker/tests; docs guard | Must not become numeric threshold in canonical selected freshness. |
| `backend_context` | `n/a` for selected canonical row | Generated index/manifest | Freshness checker/tests | Prevents backend superiority inference. |
| `methodology_notes` | Must include `not_portable_performance_claim`; should preserve threshold-free measurement wording. | Generated index/manifest and docs | Freshness checker/tests; docs guard | Candidate for stronger exact-token enforcement on Days 7-8. |
| `workflow_platforms` | `linux;macos` | Selected target manifest | Manifest tests | No Windows selected benchmark freshness. |
| `non_claims` | Must include the Day 5 required non-claims. | Selected target manifest and docs | Manifest tests; docs guard | Prevents claim drift. |

### Allowed Values And Forbidden Wording

Allowed selected canonical methodology values:

- `status=measurement`
- `baseline=n/a`
- `threshold=n/a`
- `warmup=none_configured`
- `variance=not_computed_single_sample`
- `repeat_semantics=configured_repeat_1`
- `backend_context=n/a`
- `claim_boundary=local_threshold_free` for local selected rows
- `claim_boundary=hosted_selected_threshold_free` for hosted selected rows
- `support_tier=hosted_selected` only for hosted selected row evidence

Forbidden stale or overclaim wording:

- hosted selected performance is a timing gate;
- selected performance proves or guarantees portable performance;
- selected performance proves speedup, superiority, or state-of-the-art status;
- Linux/macOS selected performance is cross-platform timing parity;
- `bench-canonical-report-freshness` is a regression threshold;
- canonical selected freshness has a numeric baseline or threshold;
- S6 local sentinel smoke ceiling is hosted selected canonical freshness;
- Windows selected benchmark freshness is present.

### Ownership Map

| Owner surface | Owns | Day 7-11 implication |
| --- | --- | --- |
| `scripts/check_bench_canonical_freshness.py` | Generated selected row schema, exact threshold-free values, hosted/local claim boundaries, manifest agreement, selected CSV agreement, unselected row locality. | Strengthen only where gaps remain; avoid threshold implementation. |
| `tests/test_bench_canonical_freshness.py` | Positive/negative fixtures for generated report and checker behavior. | Add fixtures for any strengthened checker rule. |
| `tests/corpus/manifests/selected_report_targets.tsv` | Selected target identity, hosted workflow scope, claim scope, non-claims, owner. | Update only if final wording must be more explicit; avoid adding threshold fields without evidence. |
| `tests/test_selected_report_targets_manifest.py` | Manifest vocabulary and selected row consistency. | Add benchmark-specific non-claim exactness if current tests are too broad. |
| `tests/test_selected_performance_docs.py` | Public/maintainer docs markers and forbidden overclaims. | Add markers for Day 5 decision and sentinel/canonical separation if needed. |
| `benchmarks/README.md` | Benchmark methodology interpretation and report-index handoff. | Clarify threshold-free policy and future threshold prerequisites. |
| `README.md` / `INSTALL.md` | High-level route and support/readiness claim boundary. | Keep concise; update only exact claim-boundary markers as needed. |
| `docs/maintainer_guide.md` | Maintainer repair workflow and future promotion prerequisites. | Add Day 5 decision and evidence required for future threshold promotion if current wording is insufficient. |
| `tests/corpus/README.md` / schema docs | Corpus/report-index interpretation. | Keep selected target field semantics aligned with checker and docs. |

### Guard Fixture Plan

| Fixture | Expected result | Owner |
| --- | --- | --- |
| Selected row `baseline` changed from `n/a` to numeric | Fail with selected value mismatch. | Already covered by freshness tests. |
| Selected row `threshold` changed from `n/a` to numeric | Fail with selected value mismatch. | Already covered by freshness tests. |
| Selected row `status` changed to `pass` | Fail with selected value mismatch. | Already covered by freshness tests. |
| Selected row `warmup` changed to `not_recorded` or warmup-like value | Fail with selected value mismatch. | Covered for `not_recorded`; add warmup-like value only if needed. |
| Selected row `variance` changed to `computed` or `not_recorded` | Fail with selected value mismatch. | Covered for `not_recorded`; add computed-like value only if needed. |
| `methodology_notes` omits `not_portable_performance_claim` | Fail clearly. | Freshness checker/test candidate. |
| `methodology_notes` adds threshold-promoting token | Fail if Day 7 chooses explicit forbidden-token enforcement. | Freshness checker/test candidate. |
| Hosted row uses `runner_context=local`, `build_flags=not_recorded`, or `report_label=unlabeled` | Fail hosted mode. | Already covered by freshness tests. |
| Unselected canonical row becomes `hosted_selected` or hosted claim boundary | Fail unselected row locality. | Partially covered; add claim-boundary companion fixture if missing. |
| Manifest benchmark row drops a required non-claim | Fail manifest test. | Manifest test candidate. |
| Docs claim hosted selected performance is timing gate, speedup proof, or portable performance | Fail docs guard. | Existing docs guard covers several; add cross-platform parity/S6 promotion if needed. |
| Docs omit future threshold prerequisites | Fail docs guard if Day 7-11 adds a required marker. | Docs guard candidate. |

### Day 7 Handoff

Day 7 should inspect the existing tests against this design and implement the
smallest missing enforcement first. Likely Day 7 priorities:

1. add or tighten freshness fixtures for `methodology_notes` and unselected
   claim-boundary drift if not already complete;
2. add manifest-test coverage for the selected benchmark non-claim tuple if
   current coverage is only substring-based;
3. leave documentation wording changes for Days 10-11 unless a test needs an
   exact marker earlier.

### Day 6 Validation

Day 6 changed planning documentation only. No `.c` or `.h` files are modified,
so the full C quality gate is not required.

Commands for Day 6 closeout:

```sh
git diff --check
git status --short
git diff --name-only -- '*.c' '*.h'
```

## Day 7: Tooling Implementation Batch One

### Implementation Summary

Day 7 applies the first threshold-free implementation batch to the selected
canonical benchmark freshness checker and its focused regression suite.

The checker now treats `methodology_notes` as a stronger claim-boundary field:

- it must include `not_portable_performance_claim`;
- it must not include explicit performance or threshold promotion tokens such
  as `portable_performance_claim`, `performance_superiority_claim`,
  `state_of_the_art_claim`, `hosted_timing_gate`,
  `timing_threshold_gate`, `portable_speed_claim`,
  `cross_platform_performance_claim`, `selected_timing_threshold`, or
  `regression_threshold`.

The change preserves the existing threshold-free selected values:
`status=measurement`, `baseline=n/a`, `threshold=n/a`,
`warmup=none_configured`, `variance=not_computed_single_sample`, and
`repeat_semantics=configured_repeat_1`.

### Regression Coverage Added

| Fixture | Expected result |
| --- | --- |
| Selected `methodology_notes` omits `not_portable_performance_claim` | `check_bench_canonical_freshness.py` fails with `field=methodology_notes expected_token=not_portable_performance_claim`. |
| Selected `methodology_notes` includes `selected_timing_threshold` | Checker fails with `field=methodology_notes forbidden_token=selected_timing_threshold`. |
| Unselected `bench_chol_csc` row claims `hosted_selected_threshold_free` | Checker fails with `artifact=bench_chol_csc field=claim_boundary expected=local_threshold_free observed=hosted_selected_threshold_free`. |

### Scope Notes

Day 7 did not introduce a hosted timing threshold, numeric baseline, numeric
threshold, or portable performance claim. Manifest exactness and documentation
calibration remain scheduled for later Sprint 212 implementation days unless
the existing guards prove sufficient.

### Day 7 Validation

| Command | Result | Notes |
| --- | --- | --- |
| `python3 tests/test_bench_canonical_freshness.py` | Passed | Covers the updated checker and the new Day 7 negative fixtures. |
| `git diff --check` | Passed | Whitespace validation for the full Day 7 diff. |
| `git status --short` | Passed | Shows Day 7 script/test edits plus the untracked Sprint 212 planning directory. |
| `git diff --name-only -- '*.c' '*.h'` | Passed | No C/header files changed. |

No `.c` or `.h` files were modified by the Day 7 implementation, so
`make format && make lint && make test` is not required by the sprint quality
rule for this day.

## Day 8: Tooling Implementation Batch Two

### Implementation Summary

Day 8 completes the selected benchmark freshness tooling path for the
threshold-free policy. The implementation remains centered on generated
canonical report metadata and does not add a timing threshold.

The regression suite now includes a positive exact-metadata fixture for the
selected local report row. It verifies that the selected row records:

- `status=measurement`;
- `baseline=n/a`;
- `threshold=n/a`;
- `warmup=none_configured`;
- `variance=not_computed_single_sample`;
- `repeat_semantics=configured_repeat_1`;
- `support_tier=local_only`;
- `claim_boundary=local_threshold_free`;
- `methodology_notes` containing `not_portable_performance_claim`;
- no forbidden methodology-token values;
- matching `methodology_notes` between `index.tsv` and `manifest.txt`.

The Day 7 forbidden-token fixture now iterates over the full checker
`FORBIDDEN_METHODOLOGY_NOTES` tuple instead of only the initial
`selected_timing_threshold` example.

### Regression Coverage Added

| Fixture | Expected result |
| --- | --- |
| `test_positive_local_report_records_exact_threshold_free_methodology` | Positive local canonical report passes with the exact threshold-free selected metadata and matching manifest methodology notes. |
| `test_selected_methodology_notes_reject_threshold_promotion` | Every forbidden methodology token in the checker is rejected with a token-specific diagnostic. |
| `test_manifest_methodology_notes_must_match_selected_row` | Report manifest methodology drift fails with a clear row-versus-manifest diagnostic. |

### Scope Notes

Day 8 leaves selected manifest exact-field hardening and public/maintainer
documentation calibration to Days 9-11. The functional freshness tooling path
for the selected threshold-free policy is complete for the generated report
metadata surfaces.

### Day 8 Validation

| Command | Result | Notes |
| --- | --- | --- |
| `python3 tests/test_bench_canonical_freshness.py` | Passed | Covers Day 7 and Day 8 freshness tooling fixtures. |
| `python3 tests/test_selected_report_targets_manifest.py` | Passed | Confirms selected target manifest validation remains compatible. |
| `git diff --check` | Passed | Whitespace validation for the full Day 8 diff. |
| `git status --short` | Passed | Shows Day 7-8 script/test edits plus the untracked Sprint 212 planning directory. |
| `git diff --name-only -- '*.c' '*.h'` | Passed | No C/header files changed. |

No `.c` or `.h` files were modified by the Day 8 implementation, so
`make format && make lint && make test` is not required by the sprint quality
rule for this day.

## Day 9: Manifest And Report Guards

### Implementation Summary

Day 9 binds the selected benchmark threshold-free policy to the selected
target manifest and report metadata. The new manifest contract helper requires
the exact `SRT-BENCH-REFACTOR-CSC-NOS4` identity, artifact set, expected row,
Linux/macOS hosted workflow metadata, claim scope, and non-claim tuple.

The exact selected benchmark manifest contract now covers:

- `family=benchmark`;
- `subfamily=canonical`;
- `target_key=bench_refactor_csc`;
- `selection_scope=hosted_selected`;
- `support_tier=hosted_selected`;
- `freshness_policy=generated_local_advisory`;
- `generator_command=make bench-canonical-report-freshness`;
- `artifact_pattern=build/bench-reports/canonical/bench_refactor_csc.csv`;
- required files `bench_refactor_csc.csv`, `index.tsv`, and `manifest.txt`;
- expected row id `bench_refactor_csc`;
- workflow files `.github/workflows/ci.yml` and
  `.github/workflows/macos-ci.yml`;
- workflow jobs `hosted-performance-freshness` and
  `selected-performance-freshness`;
- workflow artifacts `sprint168-selected-performance-freshness` and
  `sprint202-macos-selected-performance-freshness`;
- workflow platforms `linux;macos`;
- threshold-free Linux/macOS claim scope;
- the exact selected benchmark non-claim set.

### Regression Coverage Added

| Fixture | Expected result |
| --- | --- |
| Selected benchmark manifest exact contract | Current manifest row must match the exact selected threshold-free contract. |
| Identity drift | Changing family, subfamily, target key, row meaning, selection scope, support tier, freshness policy, generator command, or artifact pattern fails. |
| Workflow metadata drift | Changing selected benchmark workflow file, job, artifact, or platform tuple fails. |
| Threshold claim scope | Replacing threshold-free freshness wording with hosted timing threshold wording fails. |
| Missing non-claim | Removing one required non-claim fails. |
| Missing timing-threshold non-claim | Removing the required `no hosted timing threshold` token fails, preserving the Day 5 non-claim boundary. |

### Schema Confirmation

`tests/corpus/schemas/report_index_fields.md` already describes the selected
benchmark row as threshold-free, names `SRT-BENCH-REFACTOR-CSC-NOS4`, records
`baseline=n/a`, `threshold=n/a`, Linux/macOS-only hosted metadata, and the
absence of portable performance, Windows selected benchmark freshness, broad
package-manager distribution, and timing comparability claims. No Day 9 schema
text change was required.

### Day 9 Validation

| Command | Result | Notes |
| --- | --- | --- |
| `python3 tests/test_selected_report_targets_manifest.py` | Passed | Covers exact selected benchmark manifest contract and new drift fixtures. |
| `python3 tests/test_bench_canonical_freshness.py` | Passed | Confirms generated report freshness remains aligned with the manifest-backed selected row. |
| `python3 scripts/validate_corpus_schema.py` | Passed | Confirms selected manifest/schema validation remains clean. |
| `git diff --check` | Passed | Whitespace validation for the full Day 9 diff. |
| `git status --short` | Passed | Shows Day 7-9 script/test edits plus the untracked Sprint 212 planning directory. |
| `git diff --name-only -- '*.c' '*.h'` | Passed | No C/header files changed. |

No `.c` or `.h` files were modified by the Day 9 implementation, so
`make format && make lint && make test` is not required by the sprint quality
rule for this day.

## Day 10: Documentation Calibration Batch One

### Implementation Summary

Day 10 calibrates user-facing selected benchmark methodology wording across
README, INSTALL, and benchmark documentation. The docs now state that
`make bench-canonical-report-freshness` remains a threshold-free selected row
freshness check and that any future timing-threshold promotion requires
stable-runner, compiler, repeat, warmup, variance, baseline, threshold,
retained-artifact, and non-claim evidence together.

### User-Facing Updates

| Surface | Day 10 update |
| --- | --- |
| `README.md` | Adds future timing-threshold prerequisites near the selected benchmark freshness path and the detailed benchmark surface discussion. |
| `INSTALL.md` | Extends the Linux/macOS selected performance freshness support/readiness row with threshold-promotion prerequisites. |
| `benchmarks/README.md` | Adds `warmup=none_configured`, `variance=not_computed_single_sample`, and `methodology_notes` marker interpretation, plus future threshold-prerequisite wording. |
| `tests/test_selected_performance_docs.py` | Requires the new documentation markers and adds a fixture proving prerequisite wording cannot be removed silently. |

### Claim Boundary

Day 10 keeps these non-claims explicit:

- no portable performance claim;
- no hosted selected timing threshold;
- no release benchmark claim;
- no platform timing parity;
- no package, ABI, shared-library, or package-manager proof;
- no broad benchmark publication;
- no state-of-the-art claim;
- no Windows selected benchmark freshness.

### Day 10 Validation

| Command | Result | Notes |
| --- | --- | --- |
| `python3 tests/test_selected_performance_docs.py` | Passed | Covers new user-facing marker requirements and forbidden selected performance overclaims. |
| `python3 tests/test_selected_report_targets_manifest.py` | Passed | Confirms user-facing wording stays aligned with selected manifest contract. |
| `python3 tests/test_bench_canonical_freshness.py` | Passed | Confirms generated report methodology checks still align with docs wording. |
| `git diff --check` | Passed | Whitespace validation for the full Day 10 diff. |
| `git status --short` | Passed | Shows Day 7-10 docs/script/test edits plus the untracked Sprint 212 planning directory. |
| `git diff --name-only -- '*.c' '*.h'` | Passed | No C/header files changed. |

No `.c` or `.h` files were modified by the Day 10 documentation calibration,
so `make format && make lint && make test` is not required by the sprint
quality rule for this day.

## Day 11: Documentation Calibration Batch Two

### Implementation Summary

Day 11 completes maintainer-facing documentation calibration for the selected
benchmark threshold-free policy. The maintainer guide now gives operational
repair guidance for selected benchmark freshness failures, requires
`not_portable_performance_claim` in selected methodology notes, describes
forbidden methodology-note promotion tokens, and records the evidence required
before any future timing-threshold promotion.

Epic 19 project-plan status now shows Sprint 212 as in progress with Day 1-11
artifacts rather than pending future execution.

### Maintainer Repair Workflow

Maintainers should repair selected benchmark freshness failures by:

- rerunning `make bench-canonical-report-freshness` before editing ignored
  generated artifacts;
- using `python3 tests/test_bench_canonical_freshness.py` to distinguish
  selected identity, threshold-free field, methodology-note, selected CSV,
  hosted metadata, unselected locality, and manifest-agreement failures;
- using `python3 tests/test_selected_report_targets_manifest.py` for exact
  selected manifest tuple drift;
- using `python3 tests/test_selected_performance_docs.py` for user-facing and
  maintainer-facing claim-boundary wording.

### Changed-File Snapshot

| Path | Lines |
| --- | ---: |
| `README.md` | 1132 |
| `INSTALL.md` | 618 |
| `benchmarks/README.md` | 840 |
| `docs/maintainer_guide.md` | 2231 |
| `docs/planning/EPIC_19/PROJECT_PLAN.md` | 440 |
| `scripts/check_bench_canonical_freshness.py` | 557 |
| `tests/test_bench_canonical_freshness.py` | 788 |
| `tests/test_selected_performance_docs.py` | 289 |
| `tests/test_selected_report_targets_manifest.py` | 1296 |

### Day 11 Validation

| Command | Result | Notes |
| --- | --- | --- |
| `python3 tests/test_selected_performance_docs.py` | Passed | Covers maintainer repair markers, user-facing prerequisite markers, and forbidden selected performance overclaims. |
| `git diff --check` | Passed | Whitespace validation for the full Day 11 diff. |
| `git status --short` | Passed | Shows Day 7-11 docs/script/test edits plus the untracked Sprint 212 planning directory. |
| `git diff --name-only -- '*.c' '*.h'` | Passed | No C/header files changed. |

No `.c` or `.h` files were modified by the Day 11 documentation calibration,
so `make format && make lint && make test` is not required by the sprint
quality rule for this day.

## Day 12: Integrated Validation

### Validation Summary

Day 12 runs the focused selected benchmark methodology validation suite and
the relevant documentation/schema checks for the Sprint 212 changed surfaces.

| Command | Result | Purpose |
| --- | --- | --- |
| `make bench-canonical-report-freshness-tests` | Passed | Make-wired benchmark freshness regression suite covering generated report metadata, selected row identity, methodology notes, unselected row locality, and manifest agreement. |
| `make bench-canonical-report-freshness` | Passed | End-user freshness target for the selected threshold-free canonical benchmark row. |
| `python3 tests/test_selected_performance_docs.py` | Passed | User-facing and maintainer-facing selected performance documentation guard. |
| `python3 tests/test_selected_report_targets_manifest.py` | Passed | Selected target manifest exactness and non-claim guard suite. |
| `python3 scripts/validate_corpus_schema.py` | Passed | Corpus and selected target schema validation. |
| `make support-docs-guard` | Passed | Support/readiness documentation guard after INSTALL updates. |
| `git diff --name-only -- '*.c' '*.h'` | Passed | No C/header files changed. |
| `git diff --check` | Passed | Whitespace validation for the full Day 12 diff. |
| `git status --short` | Passed | Shows Day 7-12 docs/script/test edits plus the untracked Sprint 212 planning directory. |

### Quality Gate Decision

No `.c` or `.h` files changed during Sprint 212 Days 1-12. The sprint quality
rule therefore does not require `make format && make lint && make test` for
Day 12. The focused Python, Make, schema, and documentation checks above cover
the actual changed surfaces.

### Known Limitations

Day 12 does not create new hosted Linux/macOS artifacts and does not inspect
live GitHub Actions artifacts. The sprint still treats hosted selected
performance as threshold-free freshness and methodology evidence only. Hosted
timing thresholds, portable performance, platform timing parity, release
benchmark readiness, Windows selected benchmark freshness, package/ABI proof,
and state-of-the-art performance remain unclaimed.

## Day 13: Review Hardening

### Review Findings

| Finding | Severity | Fix |
| --- | --- | --- |
| Methodology-note forbidden-token checks used exact semicolon tokens without trimming whitespace, so a value such as ` selected_timing_threshold ` could bypass the forbidden-token comparison while rendering as the same policy token to a reader. | Moderate | `check_bench_canonical_freshness.py` now trims each `methodology_notes` token before required and forbidden token enforcement. |
| The selected performance docs guard rejected `timing threshold` but not the common hyphenated `timing-threshold` overclaim form for selected canonical benchmark wording. | Low | `tests/test_selected_performance_docs.py` now matches `timing[- ]threshold` and adds a hyphenated overclaim regression. |

### Added Regression Coverage

| Test | Purpose |
| --- | --- |
| `test_selected_methodology_notes_reject_spaced_threshold_promotion` | Proves whitespace around forbidden methodology-note tokens does not bypass the freshness checker. |
| `test_forbidden_selected_timing_threshold_overclaim_fails_clearly` | Proves hyphenated selected timing-threshold wording is rejected by the docs guard. |

### Review Sweep

The Day 13 text sweep searched active docs and Sprint 212 artifacts for
selected performance overclaims, hosted timing-gate wording, selected
canonical timing-threshold wording, performance superiority, and
state-of-the-art performance. Hits were non-claim wording, guard descriptions,
or planning examples of rejected claims rather than active support claims.

### Updated Line Snapshot

| Path | Lines |
| --- | ---: |
| `README.md` | 1132 |
| `INSTALL.md` | 618 |
| `benchmarks/README.md` | 840 |
| `docs/maintainer_guide.md` | 2231 |
| `docs/planning/EPIC_19/PROJECT_PLAN.md` | 440 |
| `scripts/check_bench_canonical_freshness.py` | 557 |
| `tests/test_bench_canonical_freshness.py` | 788 |
| `tests/test_selected_performance_docs.py` | 289 |
| `tests/test_selected_report_targets_manifest.py` | 1296 |

### Day 13 Validation

| Command | Result | Notes |
| --- | --- | --- |
| `python3 tests/test_selected_performance_docs.py` | Passed | Covers hyphenated timing-threshold overclaim hardening. |
| `python3 tests/test_bench_canonical_freshness.py` | Passed | Covers trimmed methodology-note forbidden-token hardening. |
| `python3 tests/test_selected_report_targets_manifest.py` | Passed | Confirms selected manifest guard remains compatible. |
| `python3 scripts/validate_corpus_schema.py` | Passed | Confirms selected target schema remains valid. |
| Selected overclaim text sweep with `rg` | Reviewed | Hits are non-claims, guard descriptions, or planning rejected-claim examples. |
| `git diff --check` | Passed | Whitespace validation for the full Day 13 diff. |
| `git status --short` | Passed | Shows Day 7-13 docs/script/test edits plus the untracked Sprint 212 planning directory. |
| `git diff --name-only -- '*.c' '*.h'` | Passed | No C/header files changed. |

No `.c` or `.h` files were modified by Day 13 review hardening, so
`make format && make lint && make test` is not required by the sprint quality
rule for this day.

## Day 14: Closeout Review

### Final Policy Status

Sprint 212 closes with **threshold-free deferral hardening** for selected
canonical benchmark freshness. It does not add a hosted selected timing
threshold for `SRT-BENCH-REFACTOR-CSC-NOS4`.

The selected policy is:

- selected target: `SRT-BENCH-REFACTOR-CSC-NOS4`;
- selected artifact: `bench_refactor_csc`;
- selected workload: `tests/data/suitesparse/nos4.mtx --repeat 1`;
- local claim boundary: `local_threshold_free`;
- hosted claim boundary: `hosted_selected_threshold_free`;
- status: `measurement`;
- baseline: `n/a`;
- threshold: `n/a`;
- warmup: `none_configured`;
- variance: `not_computed_single_sample`;
- methodology marker: `not_portable_performance_claim`.

### Final Item Reconciliation

| Epic item | Final status | Evidence |
| --- | --- | --- |
| 212.1 Benchmark Evidence Inventory | Complete | Day 1 evidence intake, Day 2 baseline, Day 3 runner/variance inventory. |
| 212.2 Threshold Decision | Complete | Day 4 criteria and Day 5 product decision select threshold-free deferral hardening. |
| 212.3 Methodology Implementation | Complete | Days 7-8 freshness checker/test hardening for methodology notes, exact selected metadata, unselected locality, and manifest agreement. |
| 212.4 Regression Tests | Complete | Days 7-9 and 13 add freshness, manifest, methodology, docs, and review-hardening regressions. |
| 212.5 Documentation Calibration | Complete | Days 10-11 update README, INSTALL, benchmark README, maintainer guide, Epic plan status, and docs guard markers. |
| 212.6 Validation And Closeout | Complete | Days 12-14 integrated validation, review hardening, closeout checks, and no-C/header quality-gate decision. |

### Final Validation Summary

Focused validation passed during closeout:

- `make bench-canonical-report-freshness-tests`;
- `make bench-canonical-report-freshness`;
- `python3 tests/test_selected_performance_docs.py`;
- `python3 tests/test_selected_report_targets_manifest.py`;
- `python3 scripts/validate_corpus_schema.py`;
- `make support-docs-guard`;
- `git diff --check`;
- `git diff --name-only -- '*.c' '*.h'`.

No `.c` or `.h` files changed during Sprint 212, so the full C quality gate
`make format && make lint && make test` is not required by the sprint rule.

### Final Changed-File Snapshot

| Path | Lines |
| --- | ---: |
| `README.md` | 1132 |
| `INSTALL.md` | 618 |
| `benchmarks/README.md` | 840 |
| `docs/maintainer_guide.md` | 2231 |
| `docs/planning/EPIC_19/PROJECT_PLAN.md` | 440 |
| `scripts/check_bench_canonical_freshness.py` | 557 |
| `tests/test_bench_canonical_freshness.py` | 788 |
| `tests/test_selected_performance_docs.py` | 289 |
| `tests/test_selected_report_targets_manifest.py` | 1296 |

### Residual Risks And Non-Claims

Sprint 212 leaves these claims unearned:

- hosted selected timing threshold;
- portable performance;
- Linux/macOS timing parity;
- Windows selected benchmark freshness;
- release benchmark readiness;
- broad benchmark publication;
- package-manager distribution;
- package, ABI, shared-library, or runtime-loader support;
- backend superiority;
- OpenMP speedup;
- state-of-the-art performance.

Future timing-threshold promotion still requires stable runner-class evidence,
compiler evidence, repeat policy, warmup policy, variance rule, baseline
provenance, threshold value, retained-artifact policy, and updated non-claim
evidence.

### Day 14 Outcome

Sprint 212 is ready for retrospective preparation. The implemented branch
closes the selected benchmark methodology decision with stronger
threshold-free tooling, manifest, documentation, and validation guards without
overstating current benchmark support.
