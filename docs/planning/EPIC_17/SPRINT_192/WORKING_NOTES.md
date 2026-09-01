# Sprint 192 Working Notes: Methodology-Bound Performance Evidence Lane

## Sprint Goal

Promote one performance lane from local threshold-free context to a
methodology-bound hosted evidence lane with explicit limits.

## Day 1: Performance Lane Intake

### Scope Trace

| Epic item | Day 1 intake interpretation |
| --- | --- |
| 192.1 Lane Selection | Select exactly one benchmark family, fixture/workload, backend policy, platform, repeat count, runtime budget, support tier, and acceptance meaning. |
| 192.2 Methodology Metadata | Ensure the selected row records compiler, flags, CPU, thread count, warmup, repeats, variance, timestamp, branch, commit, matrix metadata, backend context, baseline, threshold, and methodology notes. |
| 192.3 Hosted Freshness | Promote one hosted CI lane with bounded runtime, selected freshness validation, and exact artifact upload scope. |
| 192.4 Regression Policy | Decide whether to keep the selected lane threshold-free with explicit rationale or add one conservative regression sentinel with stable baseline semantics. |
| 192.5 Docs And Report Index | Align benchmark docs, maintainer guide, README references, selected report manifest metadata, freshness scripts, normalized report index behavior, and claim boundaries. |
| 192.6 Validation | Run benchmark freshness, schema validation, normalizer checks, workflow checks, docs guards, and the full C quality gate if any `.c` or `.h` files change. |

### Baseline Evidence Read

| Source | Day 1 finding |
| --- | --- |
| `docs/planning/EPIC_17/PROJECT_PLAN.md` | Sprint 192 is allocated 168 hours to promote one methodology-bound hosted performance evidence lane with explicit limits. |
| `docs/planning/EPIC_17/SPRINT_187/artifacts/day9-comparison-performance-gates.md` | Sprint 192 must promote exactly one bounded performance lane. The default candidate is `bench_refactor_csc` on `nos4.mtx --repeat 1` unless Sprint 192 records a stronger selection reason. |
| `docs/planning/EPIC_17/SPRINT_191/RETROSPECTIVE.md` | Sprint 191 explicitly did not expand performance claims, so Sprint 192 starts from the existing benchmark/report surface rather than inherited comparison evidence. |
| `tests/corpus/manifests/selected_report_targets.tsv` | `SRT-BENCH-REFACTOR-CSC-NOS4` is the only selected benchmark target row. It names `bench_refactor_csc`, hosted Linux scope, `hosted_selected` support tier, and threshold-free methodology language. |
| `.github/workflows/ci.yml` | The `hosted-performance-freshness` job already runs `make bench-canonical-report`, checks `scripts/check_bench_canonical_freshness.py --mode hosted`, summarizes the selected row, and uploads the canonical report bundle. |
| `Makefile` | `make bench-canonical-report` generates canonical benchmark artifacts, and `make bench-canonical-report-freshness` regenerates them and checks selected local freshness. |
| `scripts/bench_canonical_report.sh` | The report generator emits four canonical CSV files, `index.tsv`, and `manifest.txt` with metadata fields for report label, timestamp, commit, branch, platform, compiler, runner context, build flags, CPU model, build mode, thread count, support tier, claim boundary, warmup, variance, baseline, threshold, and methodology notes. |
| `scripts/check_bench_canonical_freshness.py` | The selected benchmark freshness checker validates required artifacts, index schema, selected row identity, metadata completeness, selected values, hosted/local claim boundaries, unselected row boundaries, manifest agreement, and the `not_portable_performance_claim` methodology token. |
| `tests/test_bench_canonical_freshness.py` | Existing tests cover positive local and hosted report shapes plus matrix size, warmup, variance, manifest mismatch, row-width mismatch, unselected row promotion, and hosted metadata requirements. |
| `benchmarks/README.md` | Benchmark docs explain canonical reports, selected freshness, threshold-free interpretation, performance sentinels, and non-portable timing caveats. |
| `docs/maintainer_guide.md` | Maintainer docs already distinguish canonical maintained performance surfaces, hosted selected performance freshness, local sentinel behavior, and non-claims for portable performance and state-of-the-art status. |
| `README.md` | Public docs mention `make bench-canonical-report`, `make bench-canonical-report-freshness`, the selected hosted Linux lane, threshold-free methodology, and retained non-claims. |

### Current Performance Infrastructure Inventory

| Surface | Current Day 1 state |
| --- | --- |
| Canonical report target | `make bench-canonical-report` writes a threshold-free canonical benchmark bundle under `build/bench-reports/canonical/`. |
| Local freshness target | `make bench-canonical-report-freshness` regenerates the canonical bundle and runs `python3 scripts/check_bench_canonical_freshness.py --report-dir build/bench-reports/canonical --mode local`. |
| Hosted freshness lane | `.github/workflows/ci.yml` has `hosted-performance-freshness` on `ubuntu-latest` with hosted metadata environment variables and selected freshness validation in hosted mode. |
| Selected benchmark row | `SRT-BENCH-REFACTOR-CSC-NOS4` selects `bench_refactor_csc` for `tests/data/suitesparse/nos4.mtx --repeat 1`. |
| Selected artifact | `build/bench-reports/canonical/bench_refactor_csc.csv` is the selected row artifact. |
| Required artifacts | The selected manifest requires `bench_refactor_csc.csv`, `index.tsv`, and `manifest.txt`. The hosted workflow currently uploads those plus the other three canonical CSV files. |
| Canonical generated bundle | `bench_refactor_csc.csv`, `bench_chol_csc.csv`, `bench_iterative_reuse.csv`, `bench_eigs_reuse.csv`, `index.tsv`, and `manifest.txt`. |
| Index row count | `scripts/bench_canonical_report.sh` emits one `index.tsv` row for each canonical artifact. The freshness checker validates only the selected row plus unselected row boundaries. |
| Selected methodology status | Selected row uses `status=measurement`. It is freshness and methodology evidence, not timing pass/fail evidence. |
| Hosted support tier | Hosted mode expects `support_tier=hosted_selected` and `claim_boundary=hosted_selected_threshold_free` for the selected row. |
| Local support tier | Local mode accepts `local_only` or `hosted_selected` for the selected row but requires unselected rows to remain `local_only` and `local_threshold_free`. |
| Threshold policy | Current canonical selected row is threshold-free: `baseline=n/a`, `threshold=n/a`, `warmup=none_configured`, and `variance=not_computed_single_sample`. |
| Methodology token | `methodology_notes` must include `not_portable_performance_claim`. |
| Normalized report index | `scripts/normalize_report_index.py --family benchmark --check-freshness` is listed as required Sprint 192 validation, but Day 1 has not yet verified whether its benchmark path enforces the full Sprint 192 methodology contract. |
| Local sentinel bundle | `make performance-sentinels` owns separate local sentinel governance, including `wall-check` and S6 local smoke ceiling. It is adjacent but not the selected hosted methodology lane by default. |

### Existing Benchmark Families and Lanes

| Lane | Current role | Day 1 interpretation |
| --- | --- | --- |
| `bench_refactor_csc` | Selected canonical benchmark row for `nos4.mtx --repeat 1`. | Default Sprint 192 candidate because it is already selected, hosted, documented, and freshness-checked. |
| `bench_chol_csc` | Canonical maintained Cholesky benchmark artifact. | Useful context, but currently unselected and should not be promoted casually. |
| `bench_iterative_reuse` | Canonical maintained iterative reuse artifact. | Useful context, but broader method and convergence semantics make it a higher-risk Sprint 192 candidate. |
| `bench_eigs_reuse` | Canonical maintained eigensolver reuse artifact. | Useful context, but eigensolver timing/convergence semantics increase claim and variance risk. |
| `bench-fast` | Supplemental fast runtime subset in Linux CI. | Runtime smoke evidence, not methodology-bound hosted performance publication. |
| `performance-sentinels` | Bounded local sentinel bundle with existing threshold behavior for selected local gates. | Candidate source for regression-policy ideas, but separate from hosted selected performance freshness. |
| `wall-check` | Narrow existing local hard timing gate. | Existing threshold gate; not automatically suitable for hosted methodology promotion. |
| exploratory `bench_*` binaries | Focused local measurement and broader benchmark exploration. | Out of scope for Day 1 unless Day 2 finds a stronger selected-lane reason. |

### Candidate Lane List

| Candidate | Evidence value | Initial risk | Day 1 disposition |
| --- | --- | --- | --- |
| Keep and harden `bench_refactor_csc` `nos4.mtx --repeat 1` hosted selected lane | Highest implementation fit; already selected in manifest, Makefile, CI, docs, and freshness checker. | May already be partly promoted, so Sprint 192 must identify remaining methodology/report-index gaps instead of duplicating past work. | Default candidate for Day 2. |
| Promote `bench_chol_csc` canonical row | Adds direct Cholesky timing context. | Could blur comparison evidence, selected Cholesky Windows work, and performance claims; currently unselected. | Candidate but likely lower priority than completing selected lane methodology. |
| Promote `bench_iterative_reuse` canonical row | Adds iterative solver reuse timing context. | More complex convergence, repeat, and interpretation semantics; hosted variance risk. | Defer unless Day 2 finds `bench_refactor_csc` already fully complete. |
| Promote `bench_eigs_reuse` canonical row | Adds eigensolver reuse timing context. | Eigensolver convergence/timing variance and broader algorithmic interpretation risk. | Defer unless the selected row is unsuitable. |
| Promote S6 from `performance-sentinels` into hosted evidence | Adds conservative smoke ceiling tied to selected fixture. | Sentinel semantics are local and threshold-oriented; hosted baseline policy could overclaim quickly. | Candidate for regression-policy decision, not default lane selection. |
| Promote `wall-check` | Existing hard timing gate. | Machine-class baseline and local runtime assumptions are not methodology-bound hosted evidence by default. | Likely reject for Sprint 192 hosted lane. |

### Selection Criteria

A Sprint 192 candidate is acceptable only if it has:

- exactly one selected benchmark artifact and one selected workload;
- deterministic source-controlled fixture or workload input;
- bounded hosted runtime with a reviewed timeout;
- stable generated artifacts, including `index.tsv` and `manifest.txt`;
- complete methodology metadata for platform, compiler, runner context,
  build flags, CPU model, build mode, thread count, timestamp, branch, commit,
  warmup, repeat semantics, variance, baseline, threshold, backend context,
  and methodology notes;
- freshness validation that fails stale, missing, malformed, incomplete, or
  over-promoted evidence;
- exact workflow artifact upload scope;
- conservative threshold-free or sentinel policy with clear diagnostics;
- documentation that states the selected lane and retained non-claims together.

### Rejection Criteria

Reject or defer a candidate if it requires:

- broad benchmark-family publication;
- portable performance, architecture-independent speedup, performance
  superiority, or state-of-the-art wording;
- multiple benchmark lanes in one sprint;
- unowned external fixture downloads or package-manager dependencies;
- runtime that is too slow or flaky for hosted CI;
- missing methodology metadata;
- broad artifact uploads such as entire benchmark report trees without a
  reviewed selected scope;
- ambiguous pass/fail timing claims when the row is threshold-free.

### Owner Surfaces

| Surface | Sprint 192 role |
| --- | --- |
| `scripts/bench_canonical_report.sh` | Primary generator for canonical benchmark CSVs, `index.tsv`, `manifest.txt`, and methodology metadata. |
| `scripts/check_bench_canonical_freshness.py` | Selected benchmark freshness, metadata, manifest agreement, support-tier, claim-boundary, and hosted/local validation owner. |
| `scripts/normalize_report_index.py` | Normalized benchmark report-index freshness owner; Day 2-6 must verify whether it enforces all selected methodology expectations. |
| `Makefile` | `bench-canonical-report`, `bench-canonical-report-freshness`, `bench-canonical-report-freshness-tests`, and adjacent `performance-sentinels` owner. |
| `.github/workflows/ci.yml` | Hosted Linux selected performance freshness job, metadata environment, summary, and artifact upload owner. |
| `tests/corpus/manifests/selected_report_targets.tsv` | Selected benchmark row authority for target ID, artifact, expected rows, workflow file/job/artifact/platforms, support tier, claim scope, and non-claims. |
| `tests/corpus/schemas/report_index_fields.md` | Report metadata and normalized row field documentation owner. |
| `tests/test_bench_canonical_freshness.py` | Benchmark freshness checker regression owner. |
| `tests/test_selected_report_targets_manifest.py` | Selected manifest row validation owner. |
| `tests/test_selected_comparison_workflow.py` | Existing selected workflow guard owner; Day 2 must decide whether benchmark workflow scope needs a dedicated test or extension. |
| `benchmarks/README.md` | Benchmark command groups, CSV schema, report artifacts, and measurement interpretation owner. |
| `docs/maintainer_guide.md` | Authoritative maintainer interpretation for canonical, runtime, sentinel, and hosted selected performance lanes. |
| `README.md` and `INSTALL.md` | Public-facing support and non-claim surfaces if performance wording changes. |
| `benchmarks/bench_refactor_csc.c` | Selected benchmark binary owner if fixture, command, or emitted CSV semantics need changes. Any edits here trigger the full C quality gate. |

### Initial Methodology Gaps

| Gap | Why it matters | Day 2 question |
| --- | --- | --- |
| Sprint 192 asks to promote a methodology-bound hosted lane, but a hosted selected-performance lane already exists. | The sprint should close remaining gaps rather than relabel existing work. | Which acceptance criteria from Sprint 187 are not yet fully enforced by source, tests, workflow, docs, or report index normalization? |
| Hosted workflow uploads all four canonical CSVs, while the selected manifest requires only three files. | Uploaded context may be useful, but broad uploads can blur selected artifact scope. | Should Sprint 192 narrow uploads to only selected required files or explicitly document the contextual canonical bundle? |
| `make bench-canonical-report-freshness` checks local mode, while hosted mode is only run by CI and explicit script invocation. | Local validation must distinguish shape checks from hosted evidence. | Should Day 2 keep the Make target local-only and add an explicit hosted-emulation validation command? |
| Threshold policy is currently `threshold=n/a` and `baseline=n/a`. | Timing numbers can be misread as a pass/fail or superiority claim. | Should Sprint 192 keep rows threshold-free or promote one conservative sentinel from `performance-sentinels`? |
| `warmup=none_configured` and `variance=not_computed_single_sample` are explicit but weak methodology. | The lane may be methodology-bound but statistically thin. | Is the right fix to add repeats/variance or retain the single-sample limitation with stronger non-claims? |
| CPU model on GitHub-hosted runners can vary. | Hosted evidence cannot imply a stable machine class unless the runner context is precise. | What runner metadata is enough to call the lane hosted methodology evidence without portable performance claims? |
| Normalized report index benchmark freshness may be weaker than the dedicated checker. | Sprint 192 includes docs and report-index normalization in scope. | Should `normalize_report_index.py --family benchmark --check-freshness` consume the same selected benchmark contract as the dedicated checker? |

### Initial Claim-Boundary Risks

| Risk | Why it matters | Day 2 question |
| --- | --- | --- |
| "Hosted performance evidence" may be read as performance proof. | Hosted freshness and metadata do not imply speed leadership. | What phrase should be required beside every hosted performance mention? |
| `support_tier=hosted_selected` could look broader than one row. | Only `bench_refactor_csc` should receive hosted selected meaning. | Are unselected canonical rows guarded strongly enough as `local_only`? |
| Uploaded canonical bundle includes unselected CSVs. | Reviewers may cite unselected rows as hosted evidence. | Should upload summaries and docs name unselected rows as context only? |
| Regression sentinel addition could overfit to one CI runner. | A threshold without stable variance policy can create flaky or misleading gates. | Is threshold-free methodology still the better Sprint 192 outcome? |
| Benchmark docs are spread across README, maintainer guide, and `benchmarks/README.md`. | Drift in one doc can imply broader performance support. | What guard or scan should Day 10 add for performance non-claims? |

### Day 1 Validation

Source and planning checks:

```sh
git status --short --branch
sed -n '201,233p' docs/planning/EPIC_17/PROJECT_PLAN.md
sed -n '1,90p' docs/planning/EPIC_17/SPRINT_192/PLAN.md
sed -n '139,272p' docs/planning/EPIC_17/SPRINT_187/artifacts/day9-comparison-performance-gates.md
rg --files benchmarks scripts tests/corpus/manifests tests/corpus/schemas tests | rg 'bench|benchmark|report_index|selected_report|workflow|freshness'
column -t -s $'\t' tests/corpus/manifests/selected_report_targets.tsv
sed -n '342,390p' Makefile
sed -n '1,260p' scripts/bench_canonical_report.sh
sed -n '1,360p' scripts/check_bench_canonical_freshness.py
sed -n '300,400p' .github/workflows/ci.yml
sed -n '1435,1588p' docs/maintainer_guide.md
git diff --check
```

No `.c` or `.h` files were changed on Day 1, so `make format && make lint &&
make test` is not required.

### Day 2 Questions

1. Is `bench_refactor_csc` on `nos4.mtx --repeat 1` still the correct Sprint
   192 lane, or is there a stronger reason to select a different benchmark
   family?
2. Which Sprint 187 required methodology fields are already enforced by
   `scripts/check_bench_canonical_freshness.py`, and which are only documented?
3. Should the hosted workflow upload only manifest-required files
   (`bench_refactor_csc.csv`, `index.tsv`, `manifest.txt`) or retain the full
   canonical bundle as explicitly contextual evidence?
4. Should Sprint 192 remain threshold-free with stronger methodology/non-claim
   wording, or add one conservative regression sentinel?
5. Should benchmark workflow validation live in
   `tests/test_selected_comparison_workflow.py`, a new dedicated test, or the
   existing benchmark freshness tests?

## Day 2: Candidate Benchmark Lane Audit

### Selection Summary

Day 2 selects the existing `bench_refactor_csc` canonical benchmark row on
`tests/data/suitesparse/nos4.mtx --repeat 1` as the Sprint 192
methodology-bound hosted performance evidence lane.

| Field | Decision |
| --- | --- |
| Selected target ID | `SRT-BENCH-REFACTOR-CSC-NOS4` |
| Benchmark artifact | `bench_refactor_csc` |
| Selected workload | `tests/data/suitesparse/nos4.mtx --repeat 1` |
| Fixture | `nos4.mtx` |
| Matrix metadata | `matrix_size=n=100` |
| Report family | `benchmark` |
| Subfamily | `canonical` |
| Platform scope | Hosted Linux selected lane plus local freshness shape checks |
| Hosted workflow | `.github/workflows/ci.yml` job `hosted-performance-freshness` |
| Hosted artifact | `sprint168-selected-performance-freshness` |
| Runtime budget | Existing hosted job timeout is `10` minutes; local Day 2 generation completed in about 14 seconds on this machine. |
| Repeat policy | Keep `configured_repeat_1` pending Day 3/Day 9 methodology review. |
| Warmup policy | Keep `warmup=none_configured` and treat it as a limitation. |
| Variance policy | Keep `variance=not_computed_single_sample` pending Day 9. |
| Threshold policy | Provisional threshold-free policy: `baseline=n/a`, `threshold=n/a`, and `claim_boundary=hosted_selected_threshold_free`. |
| Acceptance meaning | Fresh methodology metadata and selected artifact publication for one hosted benchmark row; not a timing threshold or speed claim. |

This keeps the Sprint 187 default candidate. No stronger replacement lane was
found during Day 2.

### Candidate Ranking

Scores use `1` for weak/high-risk and `5` for strong/low-risk.

| Rank | Candidate | Evidence value | Determinism | Hosted runtime fit | Metadata readiness | Claim safety | Implementation fit | Total | Day 2 disposition |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | `bench_refactor_csc` on `nos4.mtx --repeat 1` | 5 | 5 | 5 | 5 | 4 | 5 | 29 | Selected. |
| 2 | S6 from `performance-sentinels` | 4 | 5 | 4 | 4 | 3 | 3 | 23 | Keep as Day 9 regression-policy input. |
| 3 | `bench_chol_csc` canonical row | 3 | 5 | 5 | 4 | 3 | 3 | 23 | Defer; currently contextual and adjacent to Cholesky comparison work. |
| 4 | `bench_iterative_reuse` canonical row | 4 | 4 | 4 | 3 | 3 | 3 | 21 | Defer; convergence/reuse semantics need separate methodology. |
| 5 | `bench_eigs_reuse` canonical row | 4 | 4 | 4 | 3 | 2 | 3 | 20 | Defer; eigensolver timing/convergence semantics are higher risk. |
| 6 | `wall-check` | 3 | 3 | 3 | 2 | 2 | 2 | 15 | Reject for Sprint 192 hosted evidence. |

### Current Generated Evidence

Day 2 generated the canonical report bundle with:

```sh
make bench-canonical-report
```

The command wrote:

- `build/bench-reports/canonical/bench_refactor_csc.csv`;
- `build/bench-reports/canonical/bench_chol_csc.csv`;
- `build/bench-reports/canonical/bench_iterative_reuse.csv`;
- `build/bench-reports/canonical/bench_eigs_reuse.csv`;
- `build/bench-reports/canonical/index.tsv`;
- `build/bench-reports/canonical/manifest.txt`.

The selected row currently records `status=measurement`,
`support_tier=local_only`, `claim_boundary=local_threshold_free`,
`fixture_or_workload=nos4.mtx`, `matrix_size=n=100`,
`repeat_semantics=configured_repeat_1`, `warmup=none_configured`,
`variance=not_computed_single_sample`, `baseline=n/a`, `threshold=n/a`,
`backend_context=n/a`, and
`methodology_notes=threshold_free_local_measurement;not_portable_performance_claim`.

### Day 2 Decisions

| Question | Decision |
| --- | --- |
| Is `bench_refactor_csc` still the right lane? | Yes. It is already the only selected benchmark row and has the strongest implementation fit. |
| Are required methodology fields enforced? | The dedicated freshness checker enforces required columns, selected values, nonempty metadata, hosted/local claim boundaries, unselected row boundaries, manifest agreement, and `not_portable_performance_claim`. Day 6 must verify or harden normalized benchmark freshness against the same contract. |
| Should workflow uploads narrow immediately? | Defer to Day 7/Day 8. The workflow currently uploads the full canonical bundle; Day 2 flags this as a review-scope risk because the selected manifest requires only `bench_refactor_csc.csv`, `index.tsv`, and `manifest.txt`. |
| Threshold-free or sentinel? | Provisional threshold-free. S6 from `performance-sentinels` remains input for Day 9, but no hosted threshold is selected on Day 2. |
| Where should workflow validation live? | Defer implementation choice. Day 2 favors a benchmark-specific workflow guard or an extension to existing benchmark freshness tests instead of hiding benchmark workflow policy inside comparison-only tests. |

### Selected Lane Boundaries

Sprint 192 remains bounded to:

- one selected target ID: `SRT-BENCH-REFACTOR-CSC-NOS4`;
- one selected artifact: `bench_refactor_csc`;
- one selected workload: `tests/data/suitesparse/nos4.mtx --repeat 1`;
- one hosted workflow job: `hosted-performance-freshness` on Linux;
- no Windows or macOS selected benchmark freshness promotion;
- no promotion of unselected canonical rows to selected hosted rows;
- no portable performance, speedup, architecture-independent,
  external-library, package, ABI, release, or state-of-the-art claim.

### Day 2 Artifact

The candidate benchmark lane audit artifact is
`docs/planning/EPIC_17/SPRINT_192/artifacts/day2-candidate-benchmark-lane-audit.md`.

### Day 2 Validation

Commands run:

```sh
make bench-canonical-report
python3 scripts/check_bench_canonical_freshness.py --report-dir build/bench-reports/canonical --mode local
BENCH_CANONICAL_REPORT_LABEL=sprint-192-day2-hosted-shape SPARSE_CANONICAL_SUPPORT_TIER=hosted_selected SPARSE_CANONICAL_CLAIM_BOUNDARY=hosted_selected_threshold_free SPARSE_CANONICAL_RUNNER_CONTEXT=github-actions-ubuntu-latest SPARSE_CANONICAL_BUILD_FLAGS=default_make_flags SPARSE_CANONICAL_CPU_MODEL=unknown SPARSE_CANONICAL_BUILD_MODE=serial make bench-canonical-report
python3 scripts/check_bench_canonical_freshness.py --report-dir build/bench-reports/canonical --mode hosted
python3 scripts/normalize_report_index.py --family benchmark --check-freshness
python3 tests/test_bench_canonical_freshness.py
git diff --check
git diff --name-only -- '*.c' '*.h'
```

Results:

- canonical report generation passed and wrote the six-file bundle;
- local selected benchmark freshness passed;
- hosted-shape selected benchmark freshness passed with emulated hosted
  metadata;
- normalized benchmark freshness passed with five rows;
- benchmark freshness regression tests passed;
- `git diff --check` passed;
- no `.c` or `.h` files changed on Day 2, so `make format && make lint &&
  make test` is not required.

Generated benchmark artifacts remain ignored under `build/`.

### Day 3 Questions

1. Should Day 3 treat `configured_repeat_1`, `warmup=none_configured`, and
   `variance=not_computed_single_sample` as accepted methodology limitations
   or change the selected lane to record repeated samples?
2. Which metadata fields should be considered required in both `index.tsv` and
   `manifest.txt`, and which should remain advisory context?
3. Should the selected lane keep `baseline=n/a` and `threshold=n/a` through
   Sprint 192, or should Day 9 introduce a sentinel in a separate artifact?
4. What exact claim-boundary phrase should active docs and guards require for
   hosted selected performance evidence?
5. Should unselected canonical CSV uploads remain contextual artifacts or be
   removed from the hosted artifact bundle?

## Day 3: Methodology Contract

### Contract Summary

Day 3 defines the selected Sprint 192 methodology contract for:

```text
SRT-BENCH-REFACTOR-CSC-NOS4
bench_refactor_csc
tests/data/suitesparse/nos4.mtx --repeat 1
```

The lane remains methodology-bound and threshold-free unless the Day 9
regression policy explicitly changes that decision. The selected row is hosted
freshness and methodology evidence for one benchmark row, not a pass/fail
timing gate or portable performance claim.

### Selected Row Identity

| Field | Required value |
| --- | --- |
| `surface` | `canonical` |
| `category` | `measurement` |
| `artifact` | `bench_refactor_csc` |
| `relative_path` | `bench_refactor_csc.csv` |
| `command` | `tests/data/suitesparse/nos4.mtx --repeat 1` |
| `report_family` | `benchmark` |
| `status` | `measurement` |
| `fixture_or_workload` | `nos4.mtx` |
| `matrix_size` | `n=100` |
| `repeat_semantics` | `configured_repeat_1` |

### Required Methodology Metadata

Day 3 classifies these selected-row fields as required methodology metadata:

| Field | Required behavior |
| --- | --- |
| `report_label` | Nonempty; hosted mode must not be `unlabeled`. |
| `generated_at_utc` | Nonempty UTC timestamp matching `YYYY-MM-DDTHH:MM:SSZ`. |
| `git_commit` | Nonempty commit context; `unknown` allowed only when Git metadata is unavailable. |
| `git_branch` | Nonempty branch context; `detached` allowed for detached HEAD. |
| `platform` | Nonempty platform context. |
| `compiler` | Nonempty compiler context. |
| `runner_context` | Nonempty; hosted mode must not be `local`. |
| `build_flags` | Nonempty; hosted mode must not be `not_recorded`. |
| `cpu_model` | Nonempty; `unknown` remains acceptable. |
| `build_mode` | Nonempty build-mode context. |
| `omp_num_threads` | Nonempty thread-setting context. |
| `support_tier` | Hosted selected row must be `hosted_selected`. |
| `claim_boundary` | Hosted selected row must be `hosted_selected_threshold_free`. |
| `warmup` | Required `none_configured`; read as a limitation. |
| `variance` | Required `not_computed_single_sample`; read as a limitation. |
| `baseline` | Required `n/a` until a reviewed threshold policy exists. |
| `threshold` | Required `n/a` until a reviewed threshold policy exists. |
| `backend_context` | Required `n/a` for this selected row. |
| `methodology_notes` | Must contain `not_portable_performance_claim`. |

The dedicated checker already enforces the required column set, selected row
identity, selected values, nonempty metadata, hosted/local claim boundaries,
unselected row boundaries, manifest agreement, and the non-portable
performance token.

### Sample and Threshold Policy

Day 3 keeps the current methodology limits explicit:

- `configured_repeat_1` means the selected command uses one configured repeat;
- `warmup=none_configured` means the row is not warmup-controlled;
- `variance=not_computed_single_sample` means the row is not a statistical
  distribution or variance summary;
- `baseline=n/a` and `threshold=n/a` mean the canonical selected row is not a
  hard timing gate;
- `status=measurement` means the row reports measurement context and
  freshness, not pass/fail performance status.

Day 9 may revisit a separate conservative sentinel, but Day 3 does not promote
one.

### Local and Hosted Mode Semantics

| Mode | Selected support tier | Selected claim boundary | Evidence meaning |
| --- | --- | --- | --- |
| Local checker mode | `local_only` or `hosted_selected` | `local_threshold_free` or `hosted_selected_threshold_free` | Shape, artifact, and metadata validation only. |
| Hosted checker mode | `hosted_selected` | `hosted_selected_threshold_free` | Hosted selected freshness evidence only when produced by the reviewed Linux CI lane. |

Unselected canonical rows must remain `local_only` and
`local_threshold_free`.

### Artifact Scope Decision

The selected manifest requires only:

- `bench_refactor_csc.csv`;
- `index.tsv`;
- `manifest.txt`.

The current hosted workflow uploads those files plus
`bench_chol_csc.csv`, `bench_iterative_reuse.csv`, and
`bench_eigs_reuse.csv`. Day 3 records those additional CSVs as contextual
canonical artifacts, not selected hosted performance rows. Day 7/Day 8 must
either narrow the upload scope or guard the contextual-bundle interpretation.

### Normalized Report Index Note

`scripts/normalize_report_index.py --family benchmark --check-freshness`
currently reports benchmark rows as advisory local measurement freshness. The
dedicated checker remains the stricter selected benchmark authority. Day 6
must decide whether to harden normalized benchmark freshness by reusing this
contract or keep the dedicated checker as the authoritative selected gate.

### Claim Boundary

Allowed hosted claim shape:

> The selected `bench_refactor_csc` canonical benchmark row for
> `tests/data/suitesparse/nos4.mtx --repeat 1` has fresh methodology-bound
> hosted Linux evidence with recorded runner, compiler, build, CPU, thread,
> fixture, repeat, warmup, variance, baseline, threshold, and non-portable
> performance context.

Required paired non-claim shape:

> This is threshold-free selected-row freshness evidence, not a timing
> threshold, portable performance guarantee, architecture-independent speedup,
> algorithmic superiority claim, broad benchmark publication, package proof,
> ABI proof, external-library parity claim, release benchmark result, or
> state-of-the-art performance claim.

### Day 3 Artifact

The methodology contract artifact is
`docs/planning/EPIC_17/SPRINT_192/artifacts/day3-methodology-contract.md`.

### Day 3 Validation

Commands run:

```sh
column -t -s $'\t' build/bench-reports/canonical/index.tsv
sed -n '1,235p' scripts/check_bench_canonical_freshness.py
sed -n '190,235p' scripts/bench_canonical_report.sh
rg -n "benchmark_selected|bench_refactor_csc|hosted_selected_threshold_free|local_threshold_free|not_portable_performance_claim|warmup|variance|baseline|threshold" scripts/normalize_report_index.py tests/test_normalize_report_index.py tests/test_bench_canonical_freshness.py tests/corpus/schemas/report_index_fields.md benchmarks/README.md README.md docs/maintainer_guide.md
git diff --check
git diff --name-only -- '*.c' '*.h'
```

Results:

- selected row identity and methodology fields match the Day 3 contract;
- the dedicated benchmark freshness checker already enforces the selected-row
  field, metadata, mode, manifest, and non-claim contract;
- normalized benchmark freshness remains advisory and is a Day 6 review item;
- `git diff --check` passed;
- no `.c` or `.h` files changed on Day 3, so `make format && make lint &&
  make test` is not required.

### Day 4 Questions

1. Should `matrix_size=n=100` remain a selected-row constant in the generator,
   or should Day 4 parse it from `bench_refactor_csc.csv`?
2. Does `bench_refactor_csc.csv` expose enough row-level workload and backend
   context for the selected methodology contract?
3. Should control-character validation add any fields beyond the current
   environment-provided metadata checks?
4. Should the hosted workflow upload full canonical context or only the three
   manifest-required selected files?
5. Should Day 4 identify a concrete normalizer-hardening path for Day 6?

## Day 4: Generator and Fixture Alignment

### Alignment Summary

Day 4 confirms that the current `bench_refactor_csc` generator and fixture
surface can support the selected methodology contract without a benchmark CSV
schema change.

The selected lane remains:

```text
SRT-BENCH-REFACTOR-CSC-NOS4
bench_refactor_csc
tests/data/suitesparse/nos4.mtx --repeat 1
```

### Fixture Metadata Ownership

| Field | Owner | Day 4 decision |
| --- | --- | --- |
| Fixture path | `scripts/bench_canonical_report.sh` | Keep the selected command as `tests/data/suitesparse/nos4.mtx --repeat 1`. |
| Fixture file | `tests/data/suitesparse/nos4.mtx` | Source-controlled Matrix Market fixture remains the workload input. |
| Benchmark row identity | `benchmarks/bench_refactor_csc.c` | CSV row emits `benchmark=bench_refactor_csc`, `matrix=nos4.mtx`, `scenario=chol_spd`, `n=100`, and `nnz=594`. |
| Selected report row identity | `scripts/bench_canonical_report.sh` and `scripts/check_bench_canonical_freshness.py` | Keep selected `index.tsv` identity as `artifact=bench_refactor_csc`, `fixture_or_workload=nos4.mtx`, `matrix_size=n=100`, and `repeat_semantics=configured_repeat_1`. |
| Selected manifest authority | `tests/corpus/manifests/selected_report_targets.tsv` | Keep `SRT-BENCH-REFACTOR-CSC-NOS4` as the selected benchmark target authority. |

The Matrix Market header for `nos4.mtx` records `100 100 347` stored
symmetric entries. The selected benchmark CSV reports `n=100` and `nnz=594`
after project-side loading/expansion. Day 4 therefore keeps
`matrix_size=n=100` as a dimension label and does not reinterpret it as
nonzero-count evidence.

### Generated CSV Contract

The selected `bench_refactor_csc.csv` row exposes enough workload and backend
context for the selected methodology contract:

| Field group | Observed fields |
| --- | --- |
| Identity | `benchmark=bench_refactor_csc`, `category=proof`, `matrix=nos4.mtx`, `scenario=chol_spd` |
| Matrix context | `n=100`, `nnz=594` |
| Backend context | `ldlt_dense_backend_request=n/a`, `ldlt_dense_backend_selected=n/a`, `ldlt_dense_backend_fallback=n/a` |
| Timing context | `analyze_ms`, `refactor_public_ms`, `refactor_csc_ms`, `solve_public_ms`, `solve_csc_ms`, `speedup_refactor` |
| Residual context | `res_public`, `res_csc` |

`speedup_refactor` remains descriptive timing context only. It is not a
selected speedup, superiority, or regression-threshold claim.

### Metadata Capture Alignment

Current metadata capture aligns with the Day 3 contract:

- report label, support tier, claim boundary, runner context, build flags,
  CPU model, build mode override, and methodology notes are environment-owned;
- timestamp, commit, branch, platform, compiler, build-mode detection, and
  thread setting are generator-owned;
- warmup, variance, baseline, threshold, and backend context are generator
  constants for the selected threshold-free lane;
- TSV control-character rejection covers environment-provided metadata and
  emitted index-row fields.

### Schema Change Decision

No benchmark CSV schema change is required for Day 5.

Day 5 should instead harden methodology metadata with focused tests and small
constant/fixture-contract cleanup if useful:

1. add or centralize selected benchmark constants in the existing checker/test
   path;
2. add fixture coherence tests comparing `bench_refactor_csc.csv` fields to
   selected `index.tsv` fields;
3. add a TSV control-character rejection regression for an
   environment-provided methodology field;
4. avoid editing `benchmarks/bench_refactor_csc.c` unless a real CSV defect is
   found;
5. keep generated `build/bench-reports/canonical/` artifacts ignored.

### Normalizer Alignment

`scripts/normalize_report_index.py --family benchmark --check-freshness`
currently preserves benchmark metadata in normalized rows but treats benchmark
freshness as advisory local measurement freshness. The dedicated checker
remains the stricter selected benchmark authority.

Day 6 should either:

- reuse the dedicated selected benchmark contract from normalization; or
- add tests proving normalized benchmark rows preserve every field needed by
  the dedicated selected benchmark checker.

### Artifact Scope Alignment

The selected manifest requires only:

- `bench_refactor_csc.csv`;
- `index.tsv`;
- `manifest.txt`.

The hosted workflow currently uploads those files plus
`bench_chol_csc.csv`, `bench_iterative_reuse.csv`, and
`bench_eigs_reuse.csv`. Day 4 leaves this unchanged and records it as the
Day 7/Day 8 workflow-scope decision.

### Day 4 Artifact

The generator and fixture alignment artifact is
`docs/planning/EPIC_17/SPRINT_192/artifacts/day4-generator-fixture-alignment.md`.

### Day 4 Validation

Commands run:

```sh
head -n 20 tests/data/suitesparse/nos4.mtx
sed -n '360,430p' benchmarks/bench_refactor_csc.c
sed -n '620,690p' benchmarks/bench_refactor_csc.c
sed -n '70,185p' scripts/bench_canonical_report.sh
sed -n '740,790p' scripts/normalize_report_index.py
python3 - <<'PY'
import csv
from pathlib import Path
p = Path('build/bench-reports/canonical/bench_refactor_csc.csv')
rows = list(csv.DictReader(p.open(newline='')))
print('rows', len(rows))
for row in rows:
    for key in ('benchmark','category','matrix','scenario','n','nnz','ldlt_dense_backend_request','ldlt_dense_backend_selected','ldlt_dense_backend_fallback','speedup_refactor','res_public','res_csc'):
        print(f'{key}={row[key]}')
PY
python3 - <<'PY'
from pathlib import Path
p = Path('tests/data/suitesparse/nos4.mtx')
for line in p.read_text().splitlines():
    if not line.startswith('%'):
        rows, cols, stored = line.split()[:3]
        print(f'matrix_market_rows={rows}')
        print(f'matrix_market_cols={cols}')
        print(f'matrix_market_stored_entries={stored}')
        print('symmetry=real symmetric')
        break
PY
rg -n "reject_tsv_control_chars|SPARSE_CANONICAL|BENCH_CANONICAL_REPORT_LABEL|emit_index_row|selected_matrix_size|artifacts:" scripts/bench_canonical_report.sh
rg -n "matrix_size|warmup|variance|baseline|threshold|runner_context|build_flags|report_label|methodology_notes|selected_matrix_size" tests/test_bench_canonical_freshness.py
git diff --check
git diff --name-only -- '*.c' '*.h'
```

Results:

- `nos4.mtx` is source-controlled and declares a 100-by-100 symmetric Matrix
  Market fixture with 347 stored entries;
- generated `bench_refactor_csc.csv` exposes one selected row with selected
  identity, matrix, backend, timing, speedup-context, and residual-context
  fields;
- generator metadata fields align with the Day 3 contract;
- no benchmark CSV schema change is required for Day 5;
- `git diff --check` passed;
- no `.c` or `.h` files changed on Day 4, so `make format && make lint &&
  make test` is not required.

### Day 5 Questions

1. Should Day 5 centralize selected benchmark constants, or are existing
   explicit constants clearer for review?
2. What exact fixture coherence test should compare `bench_refactor_csc.csv`
   to `index.tsv` without making timing values deterministic?
3. Which environment-provided metadata field should get the first
   control-character rejection regression?
4. Should Day 5 add hosted-shape tests for unselected canonical row boundaries
   if hosted metadata is supplied?
5. Can Day 5 hardening avoid `.c` or `.h` edits and therefore avoid triggering
   the full C quality gate?

## Day 5: Methodology Metadata Hardening

### Implementation Summary

Day 5 hardened the selected benchmark methodology metadata path with focused
tests. No benchmark C source or generated CSV schema change was needed.

Changed source surface:

| Surface | Day 5 change |
| --- | --- |
| `tests/test_bench_canonical_freshness.py` | Added selected CSV-to-index fixture coherence coverage and TSV control-character rejection coverage for environment-provided methodology metadata. |

### Fixture Coherence Coverage

Added `test_selected_benchmark_csv_matches_index_fixture_contract()` to verify
that generated `bench_refactor_csc.csv` and selected `index.tsv` metadata
agree on stable selected-lane fields:

- `bench_refactor_csc`;
- `nos4.mtx`;
- `tests/data/suitesparse/nos4.mtx --repeat 1`;
- `matrix_size=n=100`;
- `repeat_semantics=configured_repeat_1`;
- `scenario=chol_spd`;
- `nnz=594`;
- `n/a` LDLT backend request/selection/fallback fields.

The test intentionally does not assert exact timing values, residual values,
or `speedup_refactor`.

### Metadata Control-Character Coverage

Added
`test_generator_rejects_tsv_control_characters_in_methodology_metadata()` to
verify that `BENCH_CANONICAL_REPORT_LABEL` rejects tabs/newlines before the
value can enter generated TSV metadata.

The generator already applies the same `reject_tsv_control_chars()` path to
support tier, claim boundary, runner context, build flags, CPU model, build
mode override, thread setting, methodology notes, and emitted index-row
fields.

### Day 5 Decisions

| Question | Decision |
| --- | --- |
| Centralize selected benchmark constants? | Defer production changes. Existing checker constants remain the authority, and the new tests make selected row drift visible. |
| Parse `matrix_size` from CSV in production? | Defer. `matrix_size=n=100` remains a selected dimension label, with a test proving it agrees with CSV `n=100`. |
| Add benchmark CSV schema columns? | No. Existing CSV fields provide enough workload, matrix, backend, timing, speedup-context, and residual-context data. |
| Edit `benchmarks/bench_refactor_csc.c`? | No. No C benchmark defect was found. |
| Promote timing assertions? | No. Timing values remain measurement context, not deterministic test inputs or performance pass/fail evidence. |

### Day 5 Artifact

The methodology metadata hardening artifact is
`docs/planning/EPIC_17/SPRINT_192/artifacts/day5-methodology-metadata-hardening.md`.

### Day 5 Validation

Commands run:

```sh
python3 tests/test_bench_canonical_freshness.py
make bench-canonical-report-freshness
python3 scripts/validate_corpus_schema.py
python3 scripts/normalize_report_index.py --family benchmark --check-freshness
python3 -m py_compile tests/test_bench_canonical_freshness.py scripts/check_bench_canonical_freshness.py
git diff --check
git diff --name-only -- '*.c' '*.h'
```

Results:

- benchmark freshness regression tests passed, including the new
  CSV-to-index fixture coherence and TSV control-character rejection tests;
- selected canonical benchmark local freshness passed;
- corpus schema validation passed;
- normalized benchmark freshness passed with advisory local measurement rows;
- Python syntax compilation passed;
- `git diff --check` passed;
- no `.c` or `.h` files changed on Day 5, so `make format && make lint &&
  make test` is not required.

Generated benchmark artifacts remain ignored under `build/`.

### Day 6 Questions

1. Should `normalize_report_index.py --family benchmark --check-freshness`
   call the selected benchmark checker or mirror only the fields needed in
   normalized rows?
2. Which selected benchmark metadata fields should normalized rows preserve in
   `configuration` for downstream report review?
3. Should normalizer tests reject selected benchmark rows missing
   `not_portable_performance_claim`?
4. Should benchmark normalizer freshness remain advisory if the dedicated
   checker is the authoritative selected gate?
5. How can Day 6 avoid duplicating selected benchmark constants across checker
   and normalizer code?

## Day 6: Report Index Normalization

### Implementation Summary

Day 6 added report-index normalization coverage for the selected
methodology-bound benchmark lane. No production normalizer change was needed:
`scripts/normalize_report_index.py` already imports canonical benchmark rows as
advisory local measurements and records stale-rule ownership as deferred to the
benchmark-specific checker.

Changed source surface:

| Surface | Day 6 change |
| --- | --- |
| `tests/test_normalize_report_index.py` | Expanded the benchmark runtime fixture with selected methodology metadata, asserted normalized preservation, and added missing required benchmark artifact coverage. |

### Normalizer Coverage

`test_runtime_report_rows_preserve_boundaries()` now verifies that the
normalized `bench_refactor_csc` benchmark row preserves:

- source commit, branch, platform, and compiler identity;
- advisory benchmark measurement status;
- selected support tier and claim boundary metadata;
- workload, matrix size, repeat, warmup, variance, baseline, threshold, and
  backend context;
- methodology notes including the non-portable performance claim boundary.

`test_required_benchmark_freshness_reports_missing_artifacts()` verifies that
requesting required benchmark generated output with freshness enabled fails
clearly when no benchmark artifacts exist.

### Day 6 Decisions

| Question | Decision |
| --- | --- |
| Should the normalizer call the selected benchmark checker? | No. Keep the dedicated checker as the hard selected benchmark freshness owner. |
| Should normalized benchmark rows preserve selected metadata? | Yes. Preserve methodology fields in `configuration` so report review has the full evidence context. |
| Should the normalizer reject missing non-claim markers? | Not on Day 6. That remains benchmark-checker responsibility to avoid duplicating selected constants. |
| Should benchmark normalizer freshness remain advisory? | Yes. Local benchmark measurements are advisory in the normalized index. |
| Should production normalizer code change? | No. Existing behavior was sufficient once fixture coverage matched the selected lane. |

### Day 6 Artifact

The report-index normalization artifact is
`docs/planning/EPIC_17/SPRINT_192/artifacts/day6-report-index-normalization.md`.

### Day 6 Validation

Commands run:

```sh
python3 tests/test_normalize_report_index.py
python3 tests/test_bench_canonical_freshness.py
python3 scripts/normalize_report_index.py --family benchmark --check-freshness
python3 scripts/validate_corpus_schema.py
python3 -m py_compile tests/test_normalize_report_index.py scripts/normalize_report_index.py tests/test_bench_canonical_freshness.py scripts/check_bench_canonical_freshness.py
git diff --check
git diff --name-only -- '*.c' '*.h'
```

Results:

- report-index normalization regression tests passed;
- benchmark canonical freshness regression tests passed;
- benchmark report-index freshness passed with advisory local measurement rows;
- corpus schema validation passed;
- Python syntax compilation passed;
- `git diff --check` passed;
- no `.c` or `.h` files changed on Day 6, so `make format && make lint &&
  make test` is not required.

### Day 7 Questions

1. Which hosted CI lane should own selected benchmark artifact generation?
2. Should hosted benchmark artifacts be uploaded as workflow artifacts, checked
   into source, or only consumed by freshness diagnostics?
3. What timeout budget is acceptable for the selected `nos4.mtx` benchmark
   lane on hosted Linux?
4. Which runner identity fields should be mandatory for hosted benchmark
   evidence?
5. Should hosted benchmark freshness fail the build immediately or produce an
   explicit gated report first?

## Day 7: Hosted Lane Design

### Implementation Summary

Day 7 tightened the hosted selected performance freshness workflow contract and
added guard tests for the selected-only artifact scope. The hosted lane remains
bounded to Linux and the selected `bench_refactor_csc` row for
`tests/data/suitesparse/nos4.mtx --repeat 1`.

Changed source surface:

| Surface | Day 7 change |
| --- | --- |
| `.github/workflows/ci.yml` | Narrowed the selected performance upload to `bench_refactor_csc.csv`, `index.tsv`, and `manifest.txt`. |
| `tests/test_selected_comparison_workflow.py` | Added selected performance lane contract checks and drift tests for timeout, artifact name, broad uploads, unselected uploads, and missing required files. |

### Hosted Lane Contract

| Field | Day 7 decision |
| --- | --- |
| Workflow/job | `.github/workflows/ci.yml` / `hosted-performance-freshness` |
| Runner | `ubuntu-latest` |
| Timeout | `10` minutes |
| Report label | `sprint-168-hosted-performance` |
| Support tier | `hosted_selected` |
| Claim boundary | `hosted_selected_threshold_free` |
| Generation command | `make bench-canonical-report` |
| Freshness command | `python3 scripts/check_bench_canonical_freshness.py --report-dir build/bench-reports/canonical --mode hosted` |
| Upload artifact | `sprint168-selected-performance-freshness` |

Exact upload paths:

```text
build/bench-reports/canonical/bench_refactor_csc.csv
build/bench-reports/canonical/index.tsv
build/bench-reports/canonical/manifest.txt
```

The workflow no longer uploads unselected canonical benchmark CSV files.

### Day 7 Decisions

| Question | Decision |
| --- | --- |
| Which lane owns hosted selected benchmark generation? | `hosted-performance-freshness` in Linux CI. |
| Should artifacts be uploaded or committed? | Upload only the selected CSV plus index and manifest as short-lived workflow artifacts. |
| What timeout is acceptable? | Keep `timeout-minutes: 10`; the selected workload is bounded and threshold-free. |
| Which runner fields are mandatory? | Report label, support tier, claim boundary, runner context, build flags, CPU model, build mode, and thread setting. |
| Should hosted freshness fail immediately? | Yes. The checker fails the hosted lane when selected metadata or artifacts are missing or malformed. |

### Day 7 Artifact

The hosted lane design artifact is
`docs/planning/EPIC_17/SPRINT_192/artifacts/day7-hosted-lane-design.md`.

### Day 7 Validation

Commands run:

```sh
python3 tests/test_selected_comparison_workflow.py
python3 tests/test_bench_canonical_freshness.py
python3 tests/test_normalize_report_index.py
python3 scripts/validate_corpus_schema.py
python3 -m py_compile tests/test_selected_comparison_workflow.py
git diff --check
git diff --name-only -- '*.c' '*.h'
```

Results:

- selected workflow guard tests passed;
- benchmark canonical freshness regression tests passed;
- report-index normalization regression tests passed;
- selected target schema validation passed;
- Python syntax compilation passed;
- `git diff --check` passed;
- no `.c` or `.h` files changed on Day 7, so `make format && make lint &&
  make test` is not required.

### Day 8 Questions

1. Should Day 8 make any workflow implementation change beyond the selected-only
   upload narrowing already completed on Day 7?
2. Should the hosted performance lane run the full freshness make target instead
   of `make bench-canonical-report` plus direct hosted checker invocation?
3. Should CI summary text include the exact uploaded artifact paths?
4. Should the workflow guard assert retention days for selected performance
   artifacts?
5. Should Day 8 add maintainer-guide wording for the hosted selected
   performance artifact contract?

## Day 8: Hosted Lane Implementation

### Implementation Summary

Day 8 completed the hosted selected performance lane implementation hardening.
The CI job was already present, so Day 8 focused on making the selected-only
artifact contract visible in workflow output, guarded in tests, and documented
for maintainers.

Changed source surface:

| Surface | Day 8 change |
| --- | --- |
| `.github/workflows/ci.yml` | Added exact selected upload path summary output for the hosted selected performance lane. |
| `tests/test_selected_comparison_workflow.py` | Added retention-days guard coverage and exact summary path checks for the selected performance upload. |
| `docs/maintainer_guide.md` | Documented the exact three-file hosted selected performance artifact bundle and unselected CSV non-publication boundary. |

### Implemented Contract

| Field | Implemented value |
| --- | --- |
| Workflow/job | `.github/workflows/ci.yml` / `hosted-performance-freshness` |
| Runner | `ubuntu-latest` |
| Timeout | `10` minutes |
| Generation command | `make bench-canonical-report` |
| Freshness command | `python3 scripts/check_bench_canonical_freshness.py --report-dir build/bench-reports/canonical --mode hosted` |
| Upload artifact | `sprint168-selected-performance-freshness` |
| Retention | `retention-days: 7` |
| Missing files | `if-no-files-found: error` |

Exact upload paths:

```text
build/bench-reports/canonical/bench_refactor_csc.csv
build/bench-reports/canonical/index.tsv
build/bench-reports/canonical/manifest.txt
```

### Day 8 Decisions

| Question | Decision |
| --- | --- |
| Make workflow changes beyond Day 7 upload narrowing? | Yes, add exact upload-path summary output for reviewability. |
| Use the full Make freshness target in hosted CI? | No. Keep `make bench-canonical-report` plus direct hosted checker invocation so hosted mode is explicit. |
| Include exact uploaded paths in CI summary? | Yes. CI logs now print the selected path list. |
| Guard retention days? | Yes. Workflow tests now require `retention-days: 7`. |
| Add maintainer-guide wording? | Yes. The guide now documents selected-only hosted artifact publication. |

### Day 8 Artifact

The hosted lane implementation artifact is
`docs/planning/EPIC_17/SPRINT_192/artifacts/day8-hosted-lane-implementation.md`.

### Day 8 Validation

Commands run:

```sh
python3 tests/test_selected_comparison_workflow.py
python3 tests/test_bench_canonical_freshness.py
python3 tests/test_normalize_report_index.py
python3 scripts/validate_corpus_schema.py
python3 scripts/normalize_report_index.py --family benchmark --check-freshness
python3 -m py_compile tests/test_selected_comparison_workflow.py tests/test_bench_canonical_freshness.py tests/test_normalize_report_index.py scripts/check_bench_canonical_freshness.py scripts/normalize_report_index.py
git diff --check
git diff --name-only -- '*.c' '*.h'
```

Results:

- selected workflow guard tests passed;
- benchmark canonical freshness regression tests passed;
- report-index normalization regression tests passed;
- selected target schema validation passed;
- benchmark report-index freshness passed with advisory local measurement rows;
- Python syntax compilation passed;
- `git diff --check` passed;
- no `.c` or `.h` files changed on Day 8, so `make format && make lint &&
  make test` is not required.

### Day 9 Questions

1. Should the selected hosted performance lane remain threshold-free for the
   rest of Sprint 192?
2. If a sentinel is added, what baseline source and tolerance would be
   defensible for GitHub-hosted Linux?
3. Should local performance sentinels inform the selected hosted policy or stay
   separate?
4. What wording prevents threshold-free freshness from being read as a runtime
   improvement claim?
5. Which test should fail if someone adds a timing threshold without the Day 9
   policy record?

## Day 9: Regression Policy Decision

### Implementation Summary

Day 9 kept the hosted selected performance lane threshold-free and added
explicit regression tests so the selected row cannot silently become a timing
threshold or performance pass claim.

Changed source surface:

| Surface | Day 9 change |
| --- | --- |
| `tests/test_bench_canonical_freshness.py` | Added selected manifest non-claim assertions and policy regressions for `baseline=n/a`, `threshold=n/a`, and `status=measurement`. |
| `docs/maintainer_guide.md` | Documented that hosted selected performance freshness must remain threshold-free until a future sprint records a hosted baseline, variance model, tolerance, and same-machine policy. |

### Policy Decision

| Field | Decision |
| --- | --- |
| Hosted timing threshold | Do not add one in Sprint 192 Day 9. |
| Selected row status | `measurement` |
| Baseline | `n/a` |
| Threshold | `n/a` |
| Warmup | `none_configured` |
| Variance | `not_computed_single_sample` |
| Hard hosted gate | Freshness, selected artifact presence, metadata, and claim boundary. |
| Local sentinels | Keep separate from hosted selected performance freshness. |

### Rationale

GitHub-hosted Linux runners do not provide a stable enough timing baseline for
a single `--repeat 1` selected benchmark row. A threshold would need a hosted
baseline source, runner policy, repeat and warmup strategy, variance model,
tolerance, and same-machine comparison semantics. Without those, threshold-free
freshness is the conservative evidence level.

### Day 9 Artifact

The regression policy decision artifact is
`docs/planning/EPIC_17/SPRINT_192/artifacts/day9-regression-policy-decision.md`.

### Day 9 Validation

Commands run:

```sh
python3 tests/test_bench_canonical_freshness.py
python3 tests/test_selected_comparison_workflow.py
python3 tests/test_normalize_report_index.py
python3 scripts/validate_corpus_schema.py
python3 scripts/normalize_report_index.py --family benchmark --check-freshness
python3 -m py_compile tests/test_bench_canonical_freshness.py tests/test_selected_comparison_workflow.py scripts/check_bench_canonical_freshness.py
git diff --check
git diff --name-only -- '*.c' '*.h'
```

Results:

- selected benchmark freshness tests passed, including the new policy
  regressions;
- workflow guard tests passed;
- report-index normalization regression tests passed;
- selected target schema validation passed;
- benchmark report-index freshness passed with advisory local measurement rows;
- Python syntax compilation passed;
- `git diff --check` passed;
- no `.c` or `.h` files changed on Day 9, so `make format && make lint &&
  make test` is not required.

### Day 10 Questions

1. Which public docs need explicit threshold-free selected performance wording?
2. Should docs guard tests be added for forbidden selected-performance
   overclaims?
3. Should benchmark README link directly to the selected hosted artifact
   contract?
4. Should README/INSTALL mention the selected hosted performance lane, or is the
   maintainer guide sufficient for Day 10?
5. Which exact non-claim markers should become required documentation text?

## Day 10: Claim Calibration

### Implementation Summary

Day 10 added claim-boundary documentation and a dedicated selected-performance
docs guard. The active docs now have explicit anchors for the selected
benchmark row, threshold-free policy, and non-claims.

Changed source surface:

| Surface | Day 10 change |
| --- | --- |
| `tests/corpus/README.md` | Added the exact selected performance target, workload, threshold-free values, and non-claims. |
| `tests/corpus/schemas/report_index_fields.md` | Added selected benchmark target policy fields for normalized report-index interpretation. |
| `tests/test_selected_performance_docs.py` | Added required-marker and forbidden-overclaim docs guard coverage. |

### Claim Calibration

The selected performance lane remains limited to
`SRT-BENCH-REFACTOR-CSC-NOS4`, `bench_refactor_csc`, and
`tests/data/suitesparse/nos4.mtx --repeat 1`.

Required policy fields:

```text
status=measurement
baseline=n/a
threshold=n/a
warmup=none_configured
variance=not_computed_single_sample
support_tier=hosted_selected
claim_boundary=hosted_selected_threshold_free
```

Required non-claims include no portable performance, release benchmark,
algorithmic superiority, platform parity, package/ABI support, runtime-loader
support, external-library parity, OpenMP speedup evidence, backend
superiority, or state-of-the-art status.

### Day 10 Decisions

| Question | Decision |
| --- | --- |
| Which public docs need wording? | Add explicit selected target wording to corpus docs and schema docs; existing README/benchmark/maintainer wording remains guarded. |
| Add docs guard tests? | Yes. `tests/test_selected_performance_docs.py` now enforces required markers and rejects overclaims. |
| Link benchmark README directly to hosted artifact contract? | Existing report-index handoff is enough for Day 10; exact artifact details stay in maintainer guide. |
| Mention selected hosted performance in INSTALL? | No. The lane is not install or package evidence. |
| Which non-claims are required? | Portable performance, release benchmark, algorithmic superiority, platform parity, package/ABI, runtime-loader, external-library parity, OpenMP speedup, backend superiority, and state-of-the-art status. |

### Day 10 Artifact

The claim calibration artifact is
`docs/planning/EPIC_17/SPRINT_192/artifacts/day10-claim-calibration.md`.

### Day 10 Validation

Commands run:

```sh
python3 tests/test_selected_performance_docs.py
python3 tests/test_bench_canonical_freshness.py
python3 tests/test_selected_comparison_workflow.py
python3 scripts/validate_corpus_schema.py
python3 tests/test_normalize_report_index.py
python3 scripts/normalize_report_index.py --family benchmark --check-freshness
python3 -m py_compile tests/test_selected_performance_docs.py tests/test_bench_canonical_freshness.py tests/test_selected_comparison_workflow.py tests/test_normalize_report_index.py scripts/check_bench_canonical_freshness.py scripts/normalize_report_index.py
git diff --check
git diff --name-only -- '*.c' '*.h'
```

Results:

- selected-performance docs guard passed;
- selected benchmark freshness tests passed;
- selected workflow guard tests passed;
- selected target schema validation passed;
- report-index normalization tests passed;
- benchmark report-index freshness passed with advisory local measurement rows;
- Python syntax compilation passed;
- `git diff --check` passed;
- no `.c` or `.h` files changed on Day 10, so `make format && make lint &&
  make test` is not required.

### Day 11 Questions

1. Which generator failure modes are still only implicitly covered?
2. Should malformed benchmark CSV data fail the selected checker before
   normalizer ingestion?
3. Which workflow drift cases remain unguarded after Days 7-8?
4. Should docs guard tests cover missing threshold-free policy fields
   separately from missing artifact markers?
5. Can Day 11 add failure coverage without touching benchmark C sources?

## Day 11: Failure and Drift Coverage

### Implementation Summary

Day 11 added failure coverage for malformed selected benchmark CSV artifacts
and documentation drift. The selected checker now validates CSV content, not
just selected artifact presence.

Changed source surface:

| Surface | Day 11 change |
| --- | --- |
| `scripts/check_bench_canonical_freshness.py` | Added selected CSV artifact validation for required columns, row count, stable selected values, and `matrix_size` agreement. |
| `tests/test_bench_canonical_freshness.py` | Added wrong-fixture, missing-column, and duplicate-row selected CSV regressions. |
| `tests/test_selected_performance_docs.py` | Added missing `threshold=n/a` and hosted timing-gate overclaim regressions. |

### Failure Coverage Added

The selected freshness gate now fails malformed selected CSV artifacts before
normalizer ingestion. It checks:

- required selected CSV columns;
- exactly one selected CSV row;
- `benchmark=bench_refactor_csc`;
- `matrix=nos4.mtx`;
- `n=100`;
- `scenario=chol_spd`;
- LDLT backend fields remain `n/a`;
- index `matrix_size` agrees with CSV `n`.

### Day 11 Decisions

| Question | Decision |
| --- | --- |
| Which generator failures needed coverage? | Selected CSV shape and selected value drift. |
| Should malformed CSV fail before normalizer ingestion? | Yes. The selected checker now owns that failure. |
| Which workflow drift cases remain? | Days 7-8 already cover selected performance timeout, artifact, retention, broad upload, and unselected upload drift. |
| Should docs guard threshold-free fields separately? | Yes. Missing `threshold=n/a` now fails independently. |
| Were benchmark C edits needed? | No. The gap was checker and guard coverage. |

### Day 11 Artifact

The failure and drift coverage artifact is
`docs/planning/EPIC_17/SPRINT_192/artifacts/day11-failure-and-drift-coverage.md`.

### Day 11 Validation

Commands run:

```sh
python3 tests/test_bench_canonical_freshness.py
python3 tests/test_selected_performance_docs.py
python3 tests/test_selected_comparison_workflow.py
python3 tests/test_normalize_report_index.py
python3 scripts/validate_corpus_schema.py
python3 scripts/normalize_report_index.py --family benchmark --check-freshness
python3 -m py_compile scripts/check_bench_canonical_freshness.py tests/test_bench_canonical_freshness.py tests/test_selected_performance_docs.py tests/test_selected_comparison_workflow.py tests/test_normalize_report_index.py
git diff --check
git diff --name-only -- '*.c' '*.h'
```

Results:

- selected benchmark freshness tests passed, including the new selected CSV
  failure coverage;
- selected-performance docs guard passed;
- selected workflow guard tests passed;
- report-index normalization tests passed;
- selected target schema validation passed;
- benchmark report-index freshness passed with advisory local measurement rows;
- Python syntax compilation passed;
- `git diff --check` passed;
- no `.c` or `.h` files changed on Day 11, so `make format && make lint &&
  make test` is not required.

### Day 12 Questions

1. Which generated selected performance files should be inspected manually?
2. Should Day 12 run `make bench-canonical-report-freshness` as the integrated
   anchor before all Python checks?
3. Should generated report files remain ignored after validation regeneration?
4. Which artifact metadata fields should be copied into the integrated
   validation record?
5. What residuals should remain after Day 12 if all focused checks pass?

## Day 12: Integrated Local Validation

### Implementation Summary

Day 12 regenerated the selected canonical benchmark bundle, inspected the
selected artifacts, ran the focused validation set, and confirmed generated
outputs remain ignored under `build/`.

No source code changes were needed for Day 12 beyond the validation artifact
and working-notes update.

### Generated Artifact Inspection

`make bench-canonical-report-freshness` regenerated:

- `build/bench-reports/canonical/bench_refactor_csc.csv`;
- `build/bench-reports/canonical/bench_chol_csc.csv`;
- `build/bench-reports/canonical/bench_iterative_reuse.csv`;
- `build/bench-reports/canonical/bench_eigs_reuse.csv`;
- `build/bench-reports/canonical/index.tsv`;
- `build/bench-reports/canonical/manifest.txt`.

Selected `bench_refactor_csc.csv` inspection:

| Field | Value |
| --- | --- |
| `benchmark` | `bench_refactor_csc` |
| `matrix` | `nos4.mtx` |
| `n` | `100` |
| `nnz` | `594` |
| `scenario` | `chol_spd` |
| `ldlt_dense_backend_request` | `n/a` |
| `ldlt_dense_backend_selected` | `n/a` |
| `ldlt_dense_backend_fallback` | `n/a` |

Selected `index.tsv` / `manifest.txt` policy fields:

| Field | Value |
| --- | --- |
| `git_branch` | `sprint-192` |
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

`git check-ignore -v` confirmed the generated canonical benchmark files are
ignored via `.gitignore:2:build/`, and
`git status --short --ignored build/bench-reports/canonical` reported only
`!! build/`.

### Day 12 Decisions

| Question | Decision |
| --- | --- |
| Which files should be inspected? | Selected CSV, canonical `index.tsv`, and `manifest.txt`; unselected CSV files only for ignored-output status. |
| Should the public Make target anchor validation? | Yes. `make bench-canonical-report-freshness` is the integrated local anchor. |
| Should generated files remain ignored? | Yes. Generated benchmark artifacts stay under ignored `build/`. |
| Which metadata belongs in the record? | Selected identity, branch/commit, platform/compiler context, runner/build/thread fields, and threshold-free policy fields. |
| What residuals remain? | Hosted timing threshold, unselected benchmark publication, non-Linux selected benchmark freshness, and broad performance claims remain out of scope. |

### Day 12 Artifact

The integrated local validation artifact is
`docs/planning/EPIC_17/SPRINT_192/artifacts/day12-integrated-local-validation.md`.

### Day 12 Validation

Commands run:

```sh
make bench-canonical-report-freshness
python3 tests/test_selected_performance_docs.py
python3 tests/test_selected_comparison_workflow.py
python3 scripts/validate_corpus_schema.py
python3 scripts/normalize_report_index.py --family benchmark --check-freshness
python3 tests/test_bench_canonical_freshness.py
python3 tests/test_normalize_report_index.py
python3 -m py_compile scripts/check_bench_canonical_freshness.py scripts/normalize_report_index.py tests/test_bench_canonical_freshness.py tests/test_selected_performance_docs.py tests/test_selected_comparison_workflow.py tests/test_normalize_report_index.py
git diff --check
git diff --name-only -- '*.c' '*.h'
```

Results:

- selected canonical benchmark freshness passed through the public Make target;
- selected-performance docs guard passed;
- selected workflow guard passed;
- selected target schema validation passed;
- benchmark report-index freshness passed with advisory local measurement rows;
- selected benchmark freshness regression tests passed;
- report-index normalization regression tests passed;
- Python syntax compilation passed;
- `git diff --check` passed;
- no `.c` or `.h` files changed on Day 12, so `make format && make lint &&
  make test` is not required.

### Day 13 Questions

1. Does the full diff contain any accidental generated artifact leakage?
2. Are selected-lane constants duplicated in ways that create review risk?
3. Does any active doc imply broad performance, platform, package, ABI, or
   state-of-the-art support?
4. Do workflow upload paths still match selected manifest required files?
5. Which residuals should be carried into Day 14 closeout?

## Day 13: Review Surface and Residual Audit

### Implementation Summary

Day 13 audited the changed branch surface for accidental broadening, generated
artifact leakage, brittle claim wording, and selected-lane identity drift. No
additional production changes were needed.

Changed source surface:

| Surface | Day 13 change |
| --- | --- |
| `docs/planning/EPIC_17/SPRINT_192/artifacts/day13-review-surface-audit.md` | Added final review-surface audit, selected-lane trace, residual queue, and Day 14 closeout checklist. |
| `docs/planning/EPIC_17/SPRINT_192/WORKING_NOTES.md` | Recorded Day 13 audit results and validation. |

### Audit Findings

| Area | Finding |
| --- | --- |
| Generated artifacts | `build/bench-reports/canonical` remains ignored; no generated benchmark outputs are tracked. |
| Selected identity | Manifest, checker, workflow, docs, and tests agree on `SRT-BENCH-REFACTOR-CSC-NOS4`, `bench_refactor_csc`, and `tests/data/suitesparse/nos4.mtx --repeat 1`. |
| Upload scope | Current workflow uploads only `bench_refactor_csc.csv`, `index.tsv`, and `manifest.txt`; guard tests reject broad or unselected uploads. |
| Claim scope | Docs and tests preserve threshold-free methodology freshness without portable performance, release, platform, package, ABI, or state-of-the-art claims. |
| Historical notes | Early Day 2/3 notes still record the previous full-bundle upload risk as historical context; Day 7/8 notes and current workflow supersede it. |

### Residual Queue

| Residual | Status |
| --- | --- |
| Hosted timing threshold | Deferred until a hosted baseline, variance model, repeat/warmup policy, tolerance, and same-machine comparison policy exist. |
| Unselected canonical benchmark publication | Out of scope; generated locally but not uploaded as selected hosted performance evidence. |
| Windows selected benchmark freshness | Out of scope for Sprint 192. |
| macOS selected benchmark freshness | Out of scope for Sprint 192. |
| Portable performance / state-of-the-art claims | Out of scope and guarded as non-claims. |
| Package, ABI, runtime-loader, package-manager evidence | Out of scope and explicitly separate from selected performance freshness. |

### Day 13 Artifact

The review surface audit artifact is
`docs/planning/EPIC_17/SPRINT_192/artifacts/day13-review-surface-audit.md`.

### Day 13 Validation

Commands run:

```sh
python3 tests/test_selected_performance_docs.py
python3 tests/test_selected_comparison_workflow.py
python3 scripts/validate_corpus_schema.py
python3 tests/test_bench_canonical_freshness.py
python3 tests/test_normalize_report_index.py
python3 scripts/normalize_report_index.py --family benchmark --check-freshness
python3 -m py_compile scripts/check_bench_canonical_freshness.py scripts/normalize_report_index.py tests/test_bench_canonical_freshness.py tests/test_selected_performance_docs.py tests/test_selected_comparison_workflow.py tests/test_normalize_report_index.py
git diff --check
git diff --name-only -- '*.c' '*.h'
git status --short --ignored build/bench-reports/canonical
```

Results:

- selected-performance docs guard passed;
- selected workflow guard passed;
- selected target schema validation passed;
- selected benchmark freshness regression tests passed;
- report-index normalization regression tests passed;
- benchmark report-index freshness passed with advisory local measurement rows;
- Python syntax compilation passed;
- `git diff --check` passed;
- no `.c` or `.h` files changed on Day 13, so `make format && make lint &&
  make test` is not required;
- generated canonical benchmark files remain ignored.

### Day 14 Questions

1. Should Day 14 rerun the full Day 12 validation set unchanged?
2. Are any residuals significant enough to block sprint closeout?
3. Should the final handoff call out that early working notes contain
   historical full-bundle upload observations superseded by later workflow
   changes?
4. Do the retrospective inputs need separate sections for hosted lane,
   threshold-free policy, docs guards, and residuals?
5. Is any additional PR-ready summary needed before retrospective creation?

## Day 14: Sprint Closeout and Handoff

### Implementation Summary

Day 14 reran the final focused validation set, regenerated the selected
canonical benchmark report through the public freshness target, inspected the
selected metadata, confirmed generated artifacts remain ignored, and recorded
retrospective inputs.

Changed source surface:

| Surface | Day 14 change |
| --- | --- |
| `docs/planning/EPIC_17/SPRINT_192/artifacts/day14-closeout-and-handoff.md` | Added final closeout, validation evidence, retrospective inputs, residuals, and PR-ready summary. |
| `docs/planning/EPIC_17/SPRINT_192/WORKING_NOTES.md` | Recorded Day 14 closeout results. |

### Final Outcome

Sprint 192 delivers exactly one methodology-bound hosted selected performance
evidence lane:

```text
SRT-BENCH-REFACTOR-CSC-NOS4
bench_refactor_csc
tests/data/suitesparse/nos4.mtx --repeat 1
```

The lane remains threshold-free:

```text
status=measurement
baseline=n/a
threshold=n/a
warmup=none_configured
variance=not_computed_single_sample
```

Hosted publication scope is exactly:

```text
build/bench-reports/canonical/bench_refactor_csc.csv
build/bench-reports/canonical/index.tsv
build/bench-reports/canonical/manifest.txt
```

### Final Artifact Inspection

Final `make bench-canonical-report-freshness` regenerated the canonical bundle
and passed. The selected CSV had one row with `matrix=nos4.mtx`, `n=100`, and
`nnz=594`. The selected index/manifest fields recorded `git_branch=sprint-192`,
`status=measurement`, `support_tier=local_only`,
`claim_boundary=local_threshold_free`, `matrix_size=n=100`,
`repeat_semantics=configured_repeat_1`, `warmup=none_configured`,
`variance=not_computed_single_sample`, `baseline=n/a`, `threshold=n/a`,
`backend_context=n/a`, and
`methodology_notes=threshold_free_local_measurement;not_portable_performance_claim`.

### Retrospective Inputs

What worked:

- the selected checker remained the hard policy owner;
- hosted uploads were narrowed to selected artifacts only;
- docs guards made non-claims executable;
- selected CSV content validation closed the artifact-presence gap;
- normalized report rows preserve methodology metadata without becoming a hard
  performance gate.

Accepted risks and residuals:

- hosted timing thresholds remain deferred;
- local generated timings remain machine/context dependent;
- unselected canonical CSVs remain local generated context only;
- selected benchmark freshness remains Linux hosted only;
- no portable performance, release benchmark, algorithmic superiority,
  platform parity, package/ABI/runtime-loader/package-manager, or
  state-of-the-art claim is made.

### Day 14 Artifact

The closeout and handoff artifact is
`docs/planning/EPIC_17/SPRINT_192/artifacts/day14-closeout-and-handoff.md`.

### Day 14 Validation

Commands run:

```sh
make bench-canonical-report-freshness
python3 tests/test_selected_performance_docs.py
python3 tests/test_selected_comparison_workflow.py
python3 scripts/validate_corpus_schema.py
python3 scripts/normalize_report_index.py --family benchmark --check-freshness
python3 tests/test_bench_canonical_freshness.py
python3 tests/test_normalize_report_index.py
python3 -m py_compile scripts/check_bench_canonical_freshness.py scripts/normalize_report_index.py tests/test_bench_canonical_freshness.py tests/test_selected_performance_docs.py tests/test_selected_comparison_workflow.py tests/test_normalize_report_index.py
git diff --check
git diff --name-only -- '*.c' '*.h'
git status --short --ignored build/bench-reports/canonical
```

Results:

- selected canonical benchmark freshness passed through the public Make target;
- selected-performance docs guard passed;
- selected workflow guard passed;
- selected target schema validation passed;
- benchmark report-index freshness passed with advisory local measurement rows;
- selected benchmark freshness regression tests passed;
- report-index normalization regression tests passed;
- Python syntax compilation passed;
- `git diff --check` passed;
- no `.c` or `.h` files changed on Day 14, so `make format && make lint &&
  make test` is not required;
- generated canonical benchmark files remain ignored under `build/`.
