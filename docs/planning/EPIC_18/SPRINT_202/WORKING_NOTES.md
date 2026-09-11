# Sprint 202 Working Notes: Hosted Selected Benchmark Freshness on One Additional Platform

## Sprint Goal

Add one hosted selected benchmark freshness lane outside the current Linux-only
selected performance proof, without claiming portable performance.

## Day 1: Benchmark Freshness Intake

### Scope Trace

| Epic item | Day 1 intake interpretation | Initial artifact |
| --- | --- | --- |
| 202.1 Platform And Row Selection | Select exactly one additional hosted platform and one benchmark row after Day 2 ranking; do not assume macOS or Windows before candidate scoring. | Platform/row candidate ledger and selection checklist. |
| 202.2 Methodology Metadata | Reuse Sprint 192 threshold-free metadata rules and extend them only as needed for the selected additional platform. | Metadata contract placeholder. |
| 202.3 Workflow Lane | Add one hosted workflow lane with selected artifact generation, upload, and freshness validation. | Workflow lane design placeholder. |
| 202.4 Freshness Tests | Cover missing, stale, duplicate, malformed, deferred, and path-normalized selected benchmark artifacts. | Freshness fixture matrix. |
| 202.5 Docs Calibration | Update benchmark, support, README/INSTALL, and maintainer wording without portable performance claims. | Claim-surface checklist. |
| 202.6 Validation | Run benchmark freshness tests, selected manifest tests, docs checks, hosted evidence review, and full C gate if `.c` or `.h` files change. | Validation matrix. |

### Baseline Evidence Read

| Source | Day 1 finding |
| --- | --- |
| `docs/planning/EPIC_18/PROJECT_PLAN.md` | Sprint 202 is allocated 166 hours to add one hosted selected benchmark freshness lane on one additional platform. |
| `docs/planning/EPIC_18/SPRINT_202/PLAN.md` | Day 1 is intake only; Days 2 and 3 rank and select the platform/row pair before implementation. |
| `docs/planning/EPIC_17/SPRINT_192/WORKING_NOTES.md` | Sprint 192 delivered exactly one methodology-bound hosted selected performance freshness lane for `bench_refactor_csc` on `nos4.mtx --repeat 1`, with threshold-free metadata and Linux-only hosted scope. |
| `docs/planning/EPIC_17/SPRINT_192/artifacts/day12-integrated-local-validation.md` | The public local validation anchor is `make bench-canonical-report-freshness`; generated benchmark artifacts remain ignored under `build/`. |
| `docs/planning/EPIC_17/SPRINT_192/artifacts/day14-closeout-and-handoff.md` | Residuals explicitly include non-Linux selected benchmark freshness, hosted timing thresholds, unselected benchmark publication, and portable performance claims. |
| `tests/corpus/manifests/selected_report_targets.tsv` | `SRT-BENCH-REFACTOR-CSC-NOS4` is the only selected benchmark target row and currently names Linux as the selected hosted platform. |
| `.github/workflows/ci.yml` | `hosted-performance-freshness` runs on `ubuntu-latest`, generates `bench_refactor_csc`, checks hosted freshness, summarizes selected metadata, and uploads the selected artifact set. |
| `Makefile` | `make bench-canonical-report-freshness` regenerates the canonical bundle and runs `scripts/check_bench_canonical_freshness.py --mode local`. |
| `scripts/check_bench_canonical_freshness.py` | The selected benchmark checker validates artifacts, selected row identity, metadata completeness, manifest agreement, support tier, claim boundary, and the `not_portable_performance_claim` methodology token. |
| `tests/test_bench_canonical_freshness.py` | Existing regression tests cover local/hosted positive paths, malformed index rows, manifest mismatch, row-width mismatch, duplicate rows, missing columns, wrong selected CSV fields, and unselected row promotion. |
| `tests/test_selected_performance_docs.py` | Public docs guard selected performance markers and reject portable-performance, timing-gate, speedup, and state-of-the-art overclaims. |
| `benchmarks/README.md` | Benchmark docs say selected freshness is methodology evidence, not portable speed evidence or broad benchmark publication. |
| `README.md`, `INSTALL.md`, `docs/maintainer_guide.md` | Public and maintainer docs describe Linux selected performance freshness as hosted evidence and explicitly exclude Windows selected benchmark freshness, broad platform parity, timing thresholds, package/ABI proof, and state-of-the-art claims. |

### Current Selected Benchmark Freshness Inventory

| Surface | Current Day 1 state |
| --- | --- |
| Selected target id | `SRT-BENCH-REFACTOR-CSC-NOS4`. |
| Selected benchmark command | `bench_refactor_csc` over `tests/data/suitesparse/nos4.mtx --repeat 1`. |
| Selected artifact | `build/bench-reports/canonical/bench_refactor_csc.csv`. |
| Selected support tier | `hosted_selected` in the selected target manifest and hosted checker mode. |
| Current hosted platform | Linux only, through `.github/workflows/ci.yml` job `hosted-performance-freshness`. |
| Local freshness command | `make bench-canonical-report-freshness`. |
| Hosted freshness command | `python3 scripts/check_bench_canonical_freshness.py --report-dir build/bench-reports/canonical --mode hosted`. |
| Generated bundle | `bench_refactor_csc.csv`, unselected canonical CSVs, `index.tsv`, and `manifest.txt` under ignored `build/bench-reports/canonical/`. |
| Uploaded hosted artifact set | Selected `bench_refactor_csc.csv`, `index.tsv`, and `manifest.txt`; unselected CSVs are not the selected hosted publication surface. |
| Threshold policy | `status=measurement`, `baseline=n/a`, `threshold=n/a`, `warmup=none_configured`, `variance=not_computed_single_sample`. |
| Required non-claim token | `methodology_notes` must include `not_portable_performance_claim`. |
| Adjacent local sentinel surface | `make performance-sentinels`; separate from hosted selected benchmark freshness and not a default Sprint 202 target. |
| Normalized report index | `scripts/normalize_report_index.py --family benchmark --check-freshness` preserves benchmark rows as local/advisory measurement context; the dedicated checker remains the hard selected benchmark authority. |

### Initial Platform And Row Candidate Set

Day 2 should rank these first:

1. macOS hosted selected freshness for the existing `bench_refactor_csc`
   `nos4.mtx --repeat 1` row.
2. Windows hosted selected freshness for the existing `bench_refactor_csc`
   `nos4.mtx --repeat 1` row.
3. macOS hosted selected freshness for a different canonical row, only if the
   existing row is unsuitable after runtime and claim-risk review.
4. Windows hosted selected freshness for a different canonical row, only if the
   existing row is unsuitable and Windows benchmark generation can be bounded.
5. Linux second selected benchmark row, only if Day 2 determines that "one
   additional platform" cannot be satisfied safely on macOS or Windows.
6. Local sentinel promotion, likely deferred because Sprint 202 is about hosted
   selected freshness rather than timing thresholds.

### Claim Boundary

- Sprint 202 may claim one additional hosted selected benchmark freshness lane
  only after the platform, row, metadata, workflow, tests, docs, and hosted
  evidence agree.
- The selected lane remains threshold-free unless a future sprint records a
  baseline, variance model, warmup/repeat policy, tolerance, and same-machine
  comparison semantics.
- The lane does not claim portable performance, benchmark superiority, release
  benchmark readiness, platform parity, package-manager support, ABI support,
  runtime-loader behavior, OpenMP speedup, backend superiority, or
  state-of-the-art sparse linear algebra status.
- Unselected canonical benchmark CSVs remain local generated measurement
  context unless separately selected and guarded.
- Windows selected benchmark freshness is currently an explicit non-claim in
  README/INSTALL; any Windows selection must update those claim surfaces with
  exact selected-scope wording.

### Owner Surface Inventory

| Surface | Sprint 202 relevance |
| --- | --- |
| `scripts/bench_canonical_report.sh` | Generator for canonical CSVs, `index.tsv`, `manifest.txt`, and methodology metadata. |
| `scripts/check_bench_canonical_freshness.py` | Hard selected benchmark freshness checker. |
| `tests/test_bench_canonical_freshness.py` | Fixture-based selected benchmark checker regressions. |
| `tests/test_selected_performance_docs.py` | Public documentation claim-boundary guard. |
| `tests/corpus/manifests/selected_report_targets.tsv` | Selected benchmark target authority and hosted platform declaration. |
| `tests/corpus/schemas/report_index_fields.md` | Normalized report-index interpretation fields and selected benchmark policy notes. |
| `.github/workflows/ci.yml` | Current Linux hosted selected performance lane. |
| `.github/workflows/macos-ci.yml` | Candidate additional hosted platform; currently owns selected comparison freshness, not selected benchmark freshness. |
| `.github/workflows/windows-ci.yml` | Candidate additional hosted platform; currently owns Windows build/install/PowerShell/selected comparison workflow evidence, not selected benchmark freshness. |
| `Makefile` | Local benchmark generation/freshness targets and benchmark binary registration. |
| `benchmarks/README.md` | Benchmark interpretation and selected performance documentation. |
| `README.md`, `INSTALL.md`, `docs/maintainer_guide.md` | Public and maintainer claim surfaces. |
| `docs/planning/EPIC_17/SPRINT_192/*` | Prior methodology and hosted Linux evidence source. |
| `docs/planning/EPIC_18/SPRINT_202/*` | Current sprint artifacts and closeout package. |

### Validation Matrix

| Validation | Day 1 status | Notes |
| --- | --- | --- |
| `git diff --check` | Planned for Day 1 closeout. | Day 1 changes planning documentation only. |
| `make bench-canonical-report-freshness` | Deferred. | Implementation has not selected a platform or changed freshness behavior yet. |
| `python3 tests/test_bench_canonical_freshness.py` | Deferred. | Required once checker or fixtures change. |
| `python3 tests/test_selected_performance_docs.py` | Deferred. | Required once public claim surfaces change. |
| `python3 tests/test_selected_report_targets_manifest.py` | Deferred. | Required if selected manifest metadata changes. |
| `python3 scripts/normalize_report_index.py --family benchmark --check-freshness` | Deferred. | Required if report-index benchmark interpretation changes. |
| Workflow guard or local workflow validation | Deferred. | Required once a hosted lane is added or updated. |
| Hosted CI evidence review | Deferred. | Required after the selected lane runs in CI. |
| `make docs-check` | Deferred. | Required after docs updates; optional for planning-only notes. |
| `make format && make lint && make test` | Not required for Day 1. | Required later if `.c` or `.h` files change. |

### Risk Register

| Risk | Why it matters | Mitigation |
| --- | --- | --- |
| Platform selected before evidence ranking | macOS and Windows have different runtime, shell, compiler, and claim risks. | Day 2 must score platform/row pairs before Day 3 selection. |
| Portable performance overclaim | One hosted row on one additional platform does not prove cross-platform performance. | Keep threshold-free metadata and non-claims adjacent to every public mention. |
| Unselected benchmark publication drift | Uploading broad canonical CSVs could look like broad hosted benchmark publication. | Keep artifact upload scope selected and guard required files explicitly. |
| Windows path and shell drift | Backslashes, CMake configurations, and executable paths can break artifact matching or create false freshness negatives. | Day 5 must define path normalization before workflow edits. |
| macOS runtime or dependency drift | Hosted macOS performance measurements can vary and may be slower or less available. | Treat macOS evidence as freshness/methodology only and keep runtime budget bounded. |
| Duplicated selected-row constants | Checker, manifest, docs, and workflow can drift if constants are copied. | Prefer manifest-owned identity and add focused drift tests. |
| Hosted-only false confidence | Local tests cannot prove runner metadata and upload behavior. | Record hosted residuals and inspect CI logs before closeout. |
| Threshold semantics leak | Adding another hosted platform can invite baseline comparisons. | Keep `baseline=n/a`, `threshold=n/a`, and `status=measurement` unless a future sprint explicitly owns thresholds. |

### Open Questions For Day 2

1. Which additional hosted platform has the best evidence value with the lowest
   runtime and claim risk: macOS or Windows?
2. Should Sprint 202 reuse `SRT-BENCH-REFACTOR-CSC-NOS4`, or does the manifest
   need a second selected benchmark target row for the additional platform?
3. Can the existing checker model one additional hosted platform without
   duplicating Linux-only constants?
4. What path normalization is required for selected benchmark artifacts on the
   candidate platform?
5. Which workflow guard pattern should prove the new lane without overfitting
   to one CI YAML layout?

### Day 1 Validation

Commands run:

```sh
git status --short --branch
sed -n '1,90p' docs/planning/EPIC_18/SPRINT_202/PLAN.md
sed -n '235,275p' docs/planning/EPIC_18/PROJECT_PLAN.md
find docs/planning/EPIC_18 -maxdepth 2 -type f | sort
find docs/planning -path '*/SPRINT_192/*' -type f | sort
rg -n 'selected benchmark|benchmark freshness|freshness|bench-fast|benchmark.*artifact|performance evidence|threshold-free|methodology-bound' docs scripts tests .github Makefile README.md INSTALL.md build-metadata -g '!docs/api/**'
find .github scripts tests docs build-metadata benchmarks -type f | rg 'benchmark|bench|freshness|manifest|normalize|report|workflow|performance' | sort
sed -n '1,120p;1300,1448p;1500,1580p' docs/planning/EPIC_17/SPRINT_192/WORKING_NOTES.md
sed -n '294,395p' .github/workflows/ci.yml
sed -n '399,407p' Makefile
sed -n '1,220p' scripts/check_bench_canonical_freshness.py
sed -n '1,220p' tests/test_bench_canonical_freshness.py
sed -n '1,160p' tests/test_selected_performance_docs.py
sed -n '90,110p;210,240p;360,375p' README.md
sed -n '96,106p;580,605p' INSTALL.md
sed -n '95,110p;560,575p' docs/maintainer_guide.md
rg -n 'performance|benchmark' tests/corpus/manifests/selected_report_targets.tsv benchmarks/README.md tests/corpus/schemas/report_index_fields.md
```

Day 1 changed planning documentation only. No `.c` or `.h` files were
modified, so the full C quality gate is not required.

## Day 2: Platform And Row Candidate Ranking

### Candidate Inputs Reviewed

| Surface | Day 2 finding |
| --- | --- |
| `.github/workflows/ci.yml` | Linux already owns the reviewed hosted selected performance freshness lane for `bench_refactor_csc`; a second Linux row would not satisfy the "additional platform" intent. |
| `.github/workflows/macos-ci.yml` | macOS already carries hosted selected comparison freshness, reviewed Make/CMake quality paths, benchmark/example tooling builds, and explicit no-performance-superiority claim wording. |
| `.github/workflows/windows-ci.yml` | Windows carries reviewed CMake/MSVC and one selected Cholesky comparison freshness lane, but selected benchmark freshness is explicitly deferred and has higher path, shell, executable-location, and CMake-config risk. |
| `scripts/bench_canonical_report.sh` | The canonical benchmark generator already emits threshold-free metadata for `bench_refactor_csc` and can be driven by platform-specific environment variables. |
| `scripts/check_bench_canonical_freshness.py` | The hard selected benchmark checker is the likely Sprint 202 owner for any additional hosted platform metadata validation. |
| `tests/corpus/manifests/selected_report_targets.tsv` | The only selected benchmark row is `SRT-BENCH-REFACTOR-CSC-NOS4`, currently hosted on Linux. |
| `README.md`, `INSTALL.md`, `docs/maintainer_guide.md`, `benchmarks/README.md` | Public docs currently describe Linux selected benchmark freshness and explicitly avoid Windows selected benchmark freshness, platform parity, thresholds, and portable performance claims. |

### Platform Candidate Ranking

| Rank | Platform | Evidence value | Runtime/build fit | Freshness diagnosability | Claim-safety risk | Day 2 disposition |
| ---: | --- | --- | --- | --- | --- | --- |
| 1 | macOS hosted runner | High: closes the non-Linux hosted benchmark freshness gap on a maintained Unix-like platform already used for reviewed CI. | Good: Make, shell, benchmark binary layout, Homebrew LLVM tooling, and selected comparison precedent are already present. | Good: existing canonical report path and upload semantics should remain POSIX-style and close to Linux. | Moderate: docs must say hosted macOS selected freshness only, not portable performance or macOS performance parity. | Preferred shortlist candidate for Day 3. |
| 2 | Windows hosted runner | High: would close a larger platform gap and exercise MSVC-generated artifacts. | Risky: benchmark generation would need Windows executable path, build-config, shell, and possibly Python/path normalization work. | Moderate to risky: previous Windows report work showed path and selected-target pitfalls that need careful guards. | High: public docs currently reserve Windows selected benchmark freshness as a non-claim; changing that requires more coordinated wording. | Defer unless Day 3 finds macOS unsuitable. |
| 3 | Linux additional hosted row | Low for this sprint: adds benchmark breadth but not an additional platform. | Good: existing lane already proves Linux selected freshness mechanics. | Good: same checker and artifact pattern. | Moderate: a second row can look like broad benchmark publication. | Defer because it does not meet the additional-platform goal. |
| 4 | Local performance sentinel promotion | Low for this sprint: adjacent timing evidence, not hosted selected freshness. | Good locally, but sentinel timing semantics differ from threshold-free selected freshness. | Moderate: sentinel outputs and canonical report outputs have different contracts. | High: can be mistaken for threshold or regression-baseline support. | Defer to a future methodology sprint. |

### Benchmark Row Candidate Ranking

| Rank | Row | Evidence value | Runtime cost | Methodology fit | Claim risk | Day 2 disposition |
| ---: | --- | --- | --- | --- | --- | --- |
| 1 | `SRT-BENCH-REFACTOR-CSC-NOS4` / `bench_refactor_csc` / `nos4.mtx --repeat 1` | Highest: it is the existing selected benchmark row with manifest, checker, docs, and hosted Linux precedent. | Bounded: one repeat over a 100-row SuiteSparse fixture already used by the canonical report path. | Strong: uses current threshold-free fields and selected-row identity without adding a new benchmark family. | Moderate: adding another hosted platform can imply comparability unless docs stay explicit. | Preferred row for Day 3. |
| 2 | `bench_chol_csc` / `nos4.mtx --repeat 1` | Medium: adjacent direct solver freshness evidence. | Bounded, but currently unselected and emitted as local-only canonical context. | Weaker: would require new selected manifest row and broader docs/test updates. | Higher: selecting a second canonical row may look like broad benchmark publication. | Defer. |
| 3 | `bench_iterative_reuse` | Medium: exercises iterative reuse but not the existing selected direct-refactor target. | Usually bounded, but not tied to the selected `nos4` workload identity. | Weaker: currently local-only canonical row. | Higher: could imply solver-family performance coverage. | Defer. |
| 4 | `bench_eigs_reuse` | Medium: covers eigen reuse surface. | Potentially more variable than the selected direct-refactor row. | Weaker: currently local-only canonical row. | Higher: can be mistaken for broad eigensolver performance proof. | Defer. |
| 5 | `bench_refactor_csc --indefinite-kkt` | Medium: exercises LDLT/indefinite path. | Separate synthetic workload and option surface. | Weaker: not part of the current canonical selected row. | Higher: introduces a second interpretation boundary. | Defer. |

### Ranked Platform/Row Pair Shortlist

| Rank | Platform/row pair | Score rationale | Day 2 result |
| ---: | --- | --- | --- |
| 1 | macOS hosted runner + `SRT-BENCH-REFACTOR-CSC-NOS4` | Best balance of additional-platform evidence, existing Unix-like generator fit, selected comparison freshness precedent, bounded runtime, and claim-safe reuse of current threshold-free metadata. | Preferred candidate for Day 3 selection. |
| 2 | Windows hosted runner + `SRT-BENCH-REFACTOR-CSC-NOS4` | Strong evidence value, but higher implementation risk because CMake/MSVC artifact paths, shell choice, and existing Windows selected benchmark non-claims need coordinated promotion. | Backup candidate only if macOS is rejected. |
| 3 | macOS hosted runner + a new canonical row | Feasible platform, but weaker row choice because it broadens benchmark publication surface before proving the additional-platform lane. | Deferred. |
| 4 | Windows hosted runner + a new canonical row | Combines the highest platform risk with new selected-row semantics. | Deferred. |
| 5 | Linux hosted runner + a new canonical row | Mechanically easiest but does not add a platform. | Deferred. |

### Deferred Candidate Ledger

| Candidate | Deferral reason |
| --- | --- |
| Windows selected benchmark freshness as first Sprint 202 lane | Defer for Day 2 because selected benchmark freshness on Windows is currently a public non-claim and would require more path, shell, build-config, and docs coordination than macOS. |
| Any new selected canonical benchmark row | Defer because Sprint 202 can close the additional-platform gap by reusing the already-selected row, avoiding new row identity and broader publication semantics. |
| Broad canonical benchmark artifact upload | Defer because uploading unselected rows would weaken the selected-artifact boundary. |
| Timing thresholds or regression baselines | Defer because the selected freshness contract is threshold-free and has no variance/warmup model. |
| Portable performance comparison between Linux and macOS | Defer because the additional hosted platform proves artifact freshness only, not comparable runtime behavior. |
| Linux second-row expansion | Defer because the sprint asks for one additional hosted platform. |

### Item 202.1 Progress

Day 2 completes the candidate ranking portion of item 202.1. The preferred
shortlist is intentionally narrow:

1. Primary: macOS hosted selected freshness for
   `SRT-BENCH-REFACTOR-CSC-NOS4`.
2. Backup: Windows hosted selected freshness for
   `SRT-BENCH-REFACTOR-CSC-NOS4`.

Day 3 should make the final selection and freeze the exact platform, row,
artifact path, workflow job, upload name, and out-of-scope platforms.

### Day 2 Validation

Commands run:

```sh
sed -n '1,240p' docs/planning/EPIC_18/SPRINT_202/PLAN.md
sed -n '1,260p' docs/planning/EPIC_18/SPRINT_202/WORKING_NOTES.md
sed -n '1,220p' docs/planning/EPIC_18/SPRINT_202/artifacts/day1-benchmark-freshness-intake.md
git status --short --branch
sed -n '1,240p' .github/workflows/macos-ci.yml
sed -n '1,260p' .github/workflows/windows-ci.yml
sed -n '260,430p' .github/workflows/ci.yml
sed -n '1,180p' scripts/bench_canonical_report.sh
sed -n '240,272p' docs/planning/EPIC_18/PROJECT_PLAN.md
rg -n "bench_refactor_csc|bench-canonical|performance freshness|selected benchmark|hosted_selected|not_portable_performance_claim" .github Makefile scripts tests docs README.md INSTALL.md benchmarks -g '!docs/api/**'
sed -n '1,220p' tests/corpus/manifests/selected_report_targets.tsv
```

Day 2 changed planning documentation only. No `.c` or `.h` files were
modified, so the full C quality gate is not required.

## Day 3: Selected Platform And Row Decision

### Final Selection

| Field | Day 3 decision |
| --- | --- |
| Selected platform | macOS hosted runner. |
| Selected row | `SRT-BENCH-REFACTOR-CSC-NOS4`. |
| Selected benchmark | `bench_refactor_csc`. |
| Selected workload | `tests/data/suitesparse/nos4.mtx --repeat 1`. |
| Selected artifact path | `build/bench-reports/canonical/bench_refactor_csc.csv`. |
| Required companion artifacts | `build/bench-reports/canonical/index.tsv` and `build/bench-reports/canonical/manifest.txt`. |
| Candidate workflow file | `.github/workflows/macos-ci.yml`. |
| Candidate workflow job name | `macos-selected-performance-freshness` or equivalent reviewed macOS selected benchmark freshness job name. |
| Freshness authority | `scripts/check_bench_canonical_freshness.py`. |
| Regression authority | `tests/test_bench_canonical_freshness.py`, with docs-claim coverage in `tests/test_selected_performance_docs.py`. |
| Claim boundary | Hosted selected, threshold-free methodology evidence only. |

### Why This Pair Was Selected

macOS hosted selected freshness for `SRT-BENCH-REFACTOR-CSC-NOS4` closes the
highest-value Sprint 202 gap with the least unnecessary surface area:

- it adds a non-Linux hosted selected benchmark freshness lane;
- it reuses the already-selected benchmark row instead of promoting another
  canonical row;
- it keeps the current benchmark artifact identity and threshold-free metadata
  contract;
- it fits the existing POSIX-style generator and artifact paths;
- it can reuse macOS workflow precedent from selected comparison freshness
  without implying broad benchmark publication;
- it avoids the higher first-lane Windows risks around shell selection,
  `Release` executable paths, CMake generator configuration, and public
  non-claim promotion.

### In-Scope Lane Map

| Surface | In-scope Day 3 decision |
| --- | --- |
| Workflow generation | Add one hosted macOS job that builds the benchmark binaries needed by `make bench-canonical-report` or an equivalent selected canonical report target. |
| Metadata | Record macOS-specific runner, compiler, build flags, build mode, CPU model when available, and hosted selected threshold-free claim boundary. |
| Freshness validation | Run `scripts/check_bench_canonical_freshness.py --report-dir build/bench-reports/canonical --mode hosted` against the generated canonical report bundle. |
| Artifact upload | Upload only `bench_refactor_csc.csv`, `index.tsv`, and `manifest.txt` for the selected row. |
| Tests | Add or update selected benchmark freshness regressions for the second hosted platform and selected artifact filtering. |
| Docs | Calibrate README, INSTALL, maintainer, benchmark, manifest, and report-schema wording to say Linux and macOS selected benchmark freshness only. |

### Out-Of-Scope Freeze

| Deferred surface | Reason |
| --- | --- |
| Windows selected benchmark freshness | Deferred because it carries higher path, shell, CMake config, and public-claim coordination risk than the macOS lane. |
| Any new selected benchmark row | Deferred because the existing selected row is enough to satisfy the additional-platform goal. |
| Broad canonical benchmark uploads | Deferred because Sprint 202 is selected-artifact freshness only. |
| Timing thresholds | Deferred because the lane has no baseline, variance, warmup, or tolerance model. |
| Portable Linux/macOS performance comparison | Deferred because runner hardware and timing context are not comparable evidence. |
| Benchmark superiority or state-of-the-art claims | Deferred because freshness evidence only proves that a selected report was regenerated with the required metadata. |
| Package-manager, ABI, runtime-loader, or OpenMP speedup claims | Deferred because none are part of the selected benchmark freshness contract. |

### Expected Metadata Boundary

The selected macOS lane must preserve these row-level fields:

| Field | Expected selected-row value |
| --- | --- |
| `artifact` | `bench_refactor_csc`. |
| `relative_path` | `bench_refactor_csc.csv`. |
| `command` | `tests/data/suitesparse/nos4.mtx --repeat 1`. |
| `report_family` | `benchmark`. |
| `status` | `measurement`. |
| `support_tier` | `hosted_selected`. |
| `claim_boundary` | `hosted_selected_threshold_free`. |
| `warmup` | `none_configured`. |
| `variance` | `not_computed_single_sample`. |
| `baseline` | `n/a`. |
| `threshold` | `n/a`. |
| `methodology_notes` | Includes `not_portable_performance_claim`. |

The platform-specific fields may differ from Linux, but must not be used as a
cross-platform timing comparison:

- `platform`;
- `compiler`;
- `runner_context`;
- `build_flags`;
- `cpu_model`;
- `build_mode`;
- `omp_num_threads`;
- `generated_at_utc`;
- `git_commit`;
- `git_branch`.

### Focused Validation Checklist

Implementation days should validate the macOS selected lane with:

1. `python3 tests/test_bench_canonical_freshness.py`
2. `python3 tests/test_selected_performance_docs.py`
3. `python3 tests/test_selected_report_targets_manifest.py`
4. `make bench-canonical-report-freshness`
5. A workflow-structure guard or existing selected workflow test update that
   proves the macOS job uploads only selected benchmark artifacts.
6. Hosted CI log review confirming the macOS selected freshness job generated,
   checked, summarized, and uploaded the selected artifact bundle.

### Item 202.1 Status

Item 202.1 is complete for the planning phase: Sprint 202 has exactly one
selected platform/row pair, and the backup/deferred candidates have explicit
runtime and claim-safety reasons. Implementation should not add another
platform or row unless a later review explicitly reopens this decision.

### Day 3 Validation

Commands run:

```sh
sed -n '1,340p' docs/planning/EPIC_18/SPRINT_202/WORKING_NOTES.md
sed -n '90,150p' docs/planning/EPIC_18/SPRINT_202/PLAN.md
sed -n '1,140p' docs/planning/EPIC_18/SPRINT_202/artifacts/day2-platform-row-ranking.md
git status --short --branch
```

Day 3 changed planning documentation only. No `.c` or `.h` files were
modified, so the full C quality gate is not required.

## Day 4: Methodology Metadata Contract

### Contract Owner Surfaces

| Surface | Contract role |
| --- | --- |
| `scripts/bench_canonical_report.sh` | Emits selected and unselected benchmark row metadata, `manifest.txt`, `index.tsv`, and canonical CSV artifacts. |
| `scripts/check_bench_canonical_freshness.py` | Enforces selected row identity, required metadata, hosted claim boundary, selected CSV contents, manifest agreement, and unselected-row demotion. |
| `tests/test_bench_canonical_freshness.py` | Regression owner for malformed, missing, duplicate, mismatched, and hosted/local selected benchmark freshness cases. |
| `tests/corpus/manifests/selected_report_targets.tsv` | Manifest authority for selected target id, artifact pattern, required files, expected row id, workflow file, workflow job, platform list, claim scope, and non-claims. |
| `tests/test_selected_performance_docs.py` | Public and maintainer documentation claim-boundary guard. |
| `.github/workflows/macos-ci.yml` | Selected Day 3 hosted implementation surface for the additional platform lane. |

### Required Selected-Row Metadata

| Field | Required selected-row value | Reason |
| --- | --- | --- |
| `surface` | `canonical` | Keeps the report in the canonical benchmark family. |
| `category` | `measurement` | Prevents interpretation as a pass/fail timing gate. |
| `artifact` | `bench_refactor_csc` | Matches `SRT-BENCH-REFACTOR-CSC-NOS4` target key and expected row id. |
| `relative_path` | `bench_refactor_csc.csv` | Points to the selected uploaded CSV. |
| `command` | `tests/data/suitesparse/nos4.mtx --repeat 1` | Freezes the selected workload and repeat policy. |
| `report_family` | `benchmark` | Matches the manifest family. |
| `status` | `measurement` | Keeps the row threshold-free. |
| `support_tier` | `hosted_selected` | Required in hosted mode for the selected row only. |
| `claim_boundary` | `hosted_selected_threshold_free` | Required in hosted mode; excludes thresholds and portability claims. |
| `fixture_or_workload` | `nos4.mtx` | Prevents accidental workload drift. |
| `matrix_size` | `n=100` | Must agree with the selected CSV row. |
| `repeat_semantics` | `configured_repeat_1` | Records the single configured repeat. |
| `warmup` | `none_configured` | Avoids implying warmup methodology. |
| `variance` | `not_computed_single_sample` | Avoids implying variance or statistical confidence. |
| `baseline` | `n/a` | Avoids baseline regression semantics. |
| `threshold` | `n/a` | Avoids timing threshold semantics. |
| `backend_context` | `n/a` | Keeps backend comparison out of this lane. |
| `methodology_notes` | Contains `not_portable_performance_claim` | Required non-claim marker. |

### Required Platform-Bound Metadata

The macOS hosted lane must populate these fields with non-empty values and must
not reuse local sentinel values where hosted mode forbids them:

| Field | macOS hosted expectation |
| --- | --- |
| `report_label` | A macOS-specific selected freshness label, not `unlabeled`. |
| `generated_at_utc` | UTC timestamp matching `YYYY-MM-DDTHH:MM:SSZ`. |
| `git_commit` | Source commit available to the hosted checkout, or `unknown` only if Git metadata is unavailable. |
| `git_branch` | Hosted source branch or `detached` when the checkout is detached. |
| `platform` | Runner-provided macOS `uname -a` value. |
| `compiler` | First line of `${CC:-cc} --version` from the macOS lane. |
| `runner_context` | A macOS hosted value such as `github-actions-macos-latest`, not `local`. |
| `build_flags` | A macOS hosted build flag descriptor such as `default_make_flags`, not `not_recorded`. |
| `cpu_model` | macOS runner CPU description when available, with `unknown` allowed only when detection fails. |
| `build_mode` | Expected `serial` unless OpenMP runtime detection or explicit environment override records another mode. |
| `omp_num_threads` | `unset` unless the lane configures `OMP_NUM_THREADS`. |

These fields document the runner that generated the selected artifact. They do
not make Linux and macOS timing values comparable.

### Selected Artifact Schema Notes

The selected artifact upload set is exactly:

- `build/bench-reports/canonical/bench_refactor_csc.csv`;
- `build/bench-reports/canonical/index.tsv`;
- `build/bench-reports/canonical/manifest.txt`.

The selected CSV must contain exactly one selected `bench_refactor_csc` row with
these values:

| CSV field | Required value |
| --- | --- |
| `benchmark` | `bench_refactor_csc` |
| `matrix` | `nos4.mtx` |
| `n` | `100` |
| `scenario` | `chol_spd` |
| `ldlt_dense_backend_request` | `n/a` |
| `ldlt_dense_backend_selected` | `n/a` |
| `ldlt_dense_backend_fallback` | `n/a` |

Unselected canonical artifacts may be generated to satisfy the existing
canonical report script, but they must remain `local_only` with
`local_threshold_free` in `index.tsv` and must not be uploaded as selected
hosted artifacts.

### Freshness Diagnostic Vocabulary

| Diagnostic class | Expected meaning |
| --- | --- |
| Missing report directory | The selected report bundle was not generated at the expected report directory. |
| Missing selected artifact | One of `bench_refactor_csc.csv`, `index.tsv`, or `manifest.txt` is absent. |
| Malformed index schema | `index.tsv` is empty, row width is inconsistent, required columns are missing, or header fields are duplicated. |
| Missing selected row | No `bench_refactor_csc` selected row exists in `index.tsv`. |
| Duplicate selected row | More than one `bench_refactor_csc` selected row exists in `index.tsv`. |
| Missing metadata | A required selected row field is empty, or hosted mode uses forbidden local placeholders. |
| Selected value mismatch | Selected row identity, workload, repeat, threshold-free, or methodology fields drift from the contract. |
| Selected CSV schema mismatch | The selected CSV is missing required fields, has the wrong selected row count, or disagrees with the index metadata. |
| Claim-boundary mismatch | Hosted selected rows are not `hosted_selected` / `hosted_selected_threshold_free`, or unselected rows are promoted beyond local-only context. |
| Manifest mismatch | `manifest.txt` and `index.tsv` disagree on required selected metadata fields. |
| Manifest target mismatch | `selected_report_targets.tsv` no longer has exactly one valid `SRT-BENCH-REFACTOR-CSC-NOS4` row. |

### Docs And Validator Vocabulary

Later implementation must keep these surfaces aligned:

| Vocabulary | Required wording direction |
| --- | --- |
| Hosted selected benchmark freshness | Mention Linux and macOS only after the macOS lane is implemented and reviewed. |
| Threshold-free | Always adjacent to `baseline=n/a`, `threshold=n/a`, `status=measurement`, or equivalent wording. |
| Portable performance | Always excluded; do not describe the lane as proof of portable speed, parity, or superiority. |
| Selected artifact | Use selected `bench_refactor_csc` row and selected artifact bundle wording, not broad benchmark publication. |
| Windows selected benchmark freshness | Remains deferred unless a future sprint selects and implements it. |

### Item 202.2 Status

Item 202.2 is complete for the planning phase: the selected macOS lane now has
a metadata, artifact, diagnostic, and documentation vocabulary contract before
workflow or validator edits begin.

### Day 4 Validation

Commands run:

```sh
sed -n '130,210p' docs/planning/EPIC_18/SPRINT_202/PLAN.md
sed -n '250,430p' docs/planning/EPIC_18/SPRINT_202/WORKING_NOTES.md
sed -n '1,180p' docs/planning/EPIC_18/SPRINT_202/artifacts/day3-selected-lane-decision.md
sed -n '180,260p' scripts/bench_canonical_report.sh
sed -n '1,260p' scripts/check_bench_canonical_freshness.py
sed -n '260,560p' scripts/check_bench_canonical_freshness.py
sed -n '1,220p' tests/test_bench_canonical_freshness.py
sed -n '1,120p' tests/test_selected_performance_docs.py
```

Day 4 changed planning documentation only. No `.c` or `.h` files were
modified, so the full C quality gate is not required.

## Day 5: Validator And Manifest Design

### Current Code Path Inventory

| Surface | Current behavior | Day 5 implementation decision |
| --- | --- | --- |
| `Makefile` / `bench-canonical-report` | Builds four canonical benchmark binaries and emits the canonical bundle. | Reuse this target for macOS to avoid adding a second generator path. |
| `scripts/bench_canonical_report.sh` | Emits all canonical CSVs, `index.tsv`, and `manifest.txt`; only `bench_refactor_csc` can receive hosted selected metadata. | No generator redesign planned; drive platform metadata through environment variables. |
| `scripts/check_bench_canonical_freshness.py` | Validates the selected target row, selected CSV, manifest agreement, hosted/local claim boundary, and unselected-row demotion. | Preserve as hard freshness authority; extend only if multi-platform manifest metadata requires stricter validation. |
| `tests/test_bench_canonical_freshness.py` | Owns selected benchmark fixture regressions for local/hosted metadata, malformed rows, duplicates, missing artifacts, CSV mismatch, and unselected promotion. | Add macOS-specific hosted metadata and path-normalization fixtures here if code changes need them. |
| `tests/test_selected_comparison_workflow.py` | Owns selected report workflow structure, selected benchmark upload path safety, and selected manifest/workflow agreement. | Add macOS selected performance lane workflow assertions here. |
| `tests/corpus/manifests/selected_report_targets.tsv` | Lists the selected benchmark row with Linux workflow file, job, artifact, and platform. | Update the existing benchmark row to include macOS workflow metadata instead of adding a new benchmark row. |
| `tests/corpus/schemas/report_index_fields.md` | Documents selected target workflow metadata and benchmark selected freshness policy. | Update wording only if the manifest row gains multi-platform benchmark freshness semantics. |

### Minimal Manifest Design

The selected benchmark target should remain one row:

- `target_id=SRT-BENCH-REFACTOR-CSC-NOS4`;
- `target_key=bench_refactor_csc`;
- `expected_row_ids=bench_refactor_csc`;
- `artifact_pattern=build/bench-reports/canonical/bench_refactor_csc.csv`;
- `required_files=bench_refactor_csc.csv;index.tsv;manifest.txt`.

Planned metadata change:

| Field | Current | Planned |
| --- | --- | --- |
| `workflow_file` | `.github/workflows/ci.yml` | `.github/workflows/ci.yml;.github/workflows/macos-ci.yml` |
| `workflow_job` | `hosted-performance-freshness` | `hosted-performance-freshness;selected-performance-freshness` |
| `workflow_artifact` | `sprint168-selected-performance-freshness` | `sprint168-selected-performance-freshness;sprint202-macos-selected-performance-freshness` |
| `workflow_platforms` | `linux` | `linux;macos` |
| `claim_scope` | Linux selected benchmark freshness wording. | Linux and macOS selected benchmark freshness wording for the same row only. |
| `non_claims` | No portable performance, release benchmark, superiority, platform parity, state-of-the-art, package/ABI. | Preserve all existing non-claims and continue to exclude Windows selected benchmark freshness. |

This keeps selected row identity stable and treats macOS as another hosted
evidence platform for the same row, not a new benchmark publication family.

### Minimal Workflow Design

Add one macOS job in `.github/workflows/macos-ci.yml`:

| Workflow element | Planned value |
| --- | --- |
| Job id | `selected-performance-freshness`. |
| Job name | `macOS reviewed hosted selected performance freshness`. |
| Runner | `macos-latest`. |
| Timeout | 10 or 15 minutes, bounded to the selected freshness lane. |
| Metadata env | `BENCH_CANONICAL_REPORT_LABEL=sprint202-macos-hosted-performance`, `SPARSE_CANONICAL_SUPPORT_TIER=hosted_selected`, `SPARSE_CANONICAL_CLAIM_BOUNDARY=hosted_selected_threshold_free`, `SPARSE_CANONICAL_RUNNER_CONTEXT=github-actions-macos-latest`, `SPARSE_CANONICAL_BUILD_FLAGS=default_make_flags`, `SPARSE_CANONICAL_BUILD_MODE=serial`. |
| CPU metadata | Use a macOS-native command such as `sysctl -n machdep.cpu.brand_string`, falling back to `unknown`. |
| Generation command | `make bench-canonical-report`. |
| Freshness command | `python3 scripts/check_bench_canonical_freshness.py --report-dir build/bench-reports/canonical --mode hosted`. |
| Summary | Print selected row metadata and selected uploaded paths, parallel to the Linux job but with macOS-specific label text. |
| Upload | Upload only `bench_refactor_csc.csv`, `index.tsv`, and `manifest.txt` with `if-no-files-found: error`. |

The macOS job should not upload `bench_chol_csc.csv`,
`bench_iterative_reuse.csv`, or `bench_eigs_reuse.csv`.

### Validator Design

The checker can likely remain mostly unchanged because hosted mode already
requires:

- selected target manifest agreement;
- required files present;
- exactly one selected `bench_refactor_csc` row;
- non-empty hosted metadata;
- `runner_context != local`;
- `build_flags != not_recorded`;
- `report_label != unlabeled`;
- selected row values match the contract;
- selected CSV fields match the contract;
- selected row and manifest agree;
- unselected rows remain `local_only` / `local_threshold_free`.

Potential focused change for Day 6:

- If manifest `workflow_artifact` and `workflow_platforms` become multi-value
  for the benchmark row, use existing `split_manifest_values()` semantics and
  ensure tests cover mismatched multi-platform lengths through the workflow
  guard. The freshness checker itself should continue to validate generated
  artifact contents, not workflow YAML layout.

### Path Normalization Design

The selected benchmark checker currently joins `report_dir / row["relative_path"]`
and the generator emits basename-only relative paths. For the selected macOS
lane, the expected `relative_path` remains `bench_refactor_csc.csv`, so no
Windows-style backslash normalization is needed for the first implementation.

Regression coverage should still protect against path drift:

| Case | Expected handling |
| --- | --- |
| `relative_path=bench_refactor_csc.csv` | Accepted. |
| `relative_path=./bench_refactor_csc.csv` | Rejected as selected value mismatch unless the contract is explicitly widened. |
| `relative_path=build/bench-reports/canonical/bench_refactor_csc.csv` | Rejected as selected value mismatch because upload matching expects basename contract. |
| `relative_path=bench_chol_csc.csv` | Rejected as selected value mismatch and selected CSV mismatch. |
| Backslash path in selected benchmark row | Deferred; not needed for macOS and should remain a Windows-specific design item. |

### Fixture And Regression Matrix

| Planned test | Owner | Expected assertion |
| --- | --- | --- |
| Current docs selected performance markers updated for Linux and macOS | `tests/test_selected_performance_docs.py` | Public docs mention selected Linux and macOS freshness without portable performance claims. |
| Current selected target manifest includes macOS metadata | `tests/test_bench_canonical_freshness.py` or selected workflow test | Benchmark row includes `.github/workflows/macos-ci.yml`, `selected-performance-freshness`, macOS artifact, and `linux;macos`. |
| macOS selected performance workflow structure | `tests/test_selected_comparison_workflow.py` | macOS job has hosted metadata env, generation, checker, summary, and selected-only upload. |
| Missing macOS selected workflow job fails clearly | `tests/test_selected_comparison_workflow.py` | Removing the macOS job or job id fails with a targeted message. |
| Missing macOS selected upload artifact fails clearly | `tests/test_selected_comparison_workflow.py` | Omitting `bench_refactor_csc.csv`, `index.tsv`, or `manifest.txt` fails. |
| Broad macOS benchmark upload fails clearly | `tests/test_selected_comparison_workflow.py` | `build/bench-reports/canonical/**` is rejected. |
| Unselected macOS benchmark upload fails clearly | `tests/test_selected_comparison_workflow.py` | Uploading `bench_chol_csc.csv`, `bench_iterative_reuse.csv`, or `bench_eigs_reuse.csv` is rejected. |
| Hosted macOS metadata fixture passes | `tests/test_bench_canonical_freshness.py` | A hosted fixture with `runner_context=github-actions-macos-latest` passes the checker. |
| Hosted local-placeholder metadata fails | `tests/test_bench_canonical_freshness.py` | `runner_context=local`, `build_flags=not_recorded`, or `report_label=unlabeled` continues to fail. |
| Relative path drift fails | `tests/test_bench_canonical_freshness.py` | Non-basename selected relative paths fail with selected value mismatch. |

### Registration And Source-List Impact

| Surface | Expected impact |
| --- | --- |
| Production `.c` / `.h` files | None planned. |
| Public headers | None planned. |
| CMake source lists | None planned. |
| Makefile benchmark targets | No new target planned; reuse `bench-canonical-report`. |
| GitHub workflows | `.github/workflows/macos-ci.yml` will gain one selected benchmark freshness job. |
| Python tests | `tests/test_selected_comparison_workflow.py`, `tests/test_selected_performance_docs.py`, and possibly `tests/test_bench_canonical_freshness.py` will change. |
| Manifest | Existing `SRT-BENCH-REFACTOR-CSC-NOS4` row will gain macOS workflow metadata. |
| Docs | README, INSTALL, maintainer guide, benchmark docs, corpus docs, and report schema may require claim-safe updates. |
| Generated benchmark artifacts | Remain ignored under `build/`; do not commit generated CSVs. |

### Day 6 Implementation Handoff

Day 6 should implement the minimal manifest/checker/workflow path in this
order:

1. update `selected_report_targets.tsv` for the macOS workflow metadata;
2. add the macOS workflow job with hosted threshold-free env values and
   selected-only upload;
3. add or extend workflow guard tests for the macOS selected performance lane;
4. add checker fixture coverage only where the current tests do not already
   cover the Day 4 contract;
5. run focused Python tests before broader docs calibration.

### Day 5 Validation

Commands run:

```sh
sed -n '170,250p' docs/planning/EPIC_18/SPRINT_202/PLAN.md
sed -n '360,560p' docs/planning/EPIC_18/SPRINT_202/WORKING_NOTES.md
sed -n '1,180p' docs/planning/EPIC_18/SPRINT_202/artifacts/day4-methodology-metadata-contract.md
sed -n '360,430p' Makefile
sed -n '360,430p' tests/test_selected_comparison_workflow.py
sed -n '720,780p' tests/test_selected_comparison_workflow.py
sed -n '1,120p' tests/test_selected_comparison_workflow.py
rg -n "workflow_platforms|hosted-performance-freshness|selected_report_targets|benchmark" tests scripts docs/planning/EPIC_18/SPRINT_202 -g '!docs/api/**'
```

Day 5 changed planning documentation only. No `.c` or `.h` files were
modified, so the full C quality gate is not required.

## Day 6: Freshness Validator Implementation

### Implemented Changes

| Surface | Day 6 change |
| --- | --- |
| `tests/corpus/manifests/selected_report_targets.tsv` | Extended the existing `SRT-BENCH-REFACTOR-CSC-NOS4` row to list both Linux and macOS hosted workflow metadata for the same selected benchmark row. |
| `.github/workflows/macos-ci.yml` | Added one `selected-performance-freshness` job that generates the canonical benchmark report, checks hosted freshness, summarizes selected metadata, and uploads only selected benchmark artifacts. |
| `tests/test_selected_comparison_workflow.py` | Added macOS selected performance workflow assertions using the manifest row, selected-only upload checks, hosted metadata checks, and a macOS unselected-upload regression. |
| `tests/test_bench_canonical_freshness.py` | Updated manifest assertions for Linux/macOS multi-platform metadata and added macOS-style hosted metadata plus selected `relative_path` drift regressions. |

### Manifest Implementation

The selected benchmark row remains a single target:

- `target_id=SRT-BENCH-REFACTOR-CSC-NOS4`;
- `target_key=bench_refactor_csc`;
- `expected_row_ids=bench_refactor_csc`;
- `artifact_pattern=build/bench-reports/canonical/bench_refactor_csc.csv`;
- `required_files=bench_refactor_csc.csv;index.tsv;manifest.txt`.

Day 6 changed only the hosted workflow metadata and claim text:

| Field | Day 6 value |
| --- | --- |
| `workflow_file` | `.github/workflows/ci.yml;.github/workflows/macos-ci.yml` |
| `workflow_job` | `hosted-performance-freshness;selected-performance-freshness` |
| `workflow_artifact` | `sprint168-selected-performance-freshness;sprint202-macos-selected-performance-freshness` |
| `workflow_platforms` | `linux;macos` |
| `claim_scope` | Selected canonical benchmark metadata freshness for the same `bench_refactor_csc` row on reviewed Linux and macOS hosted lanes. |
| `non_claims` | Existing non-claims preserved, with `no Windows selected benchmark freshness` added explicitly. |

### macOS Workflow Lane

The new macOS job:

- runs on `macos-latest`;
- has `timeout-minutes: 10`;
- sets `BENCH_CANONICAL_REPORT_LABEL=sprint202-macos-hosted-performance`;
- sets `SPARSE_CANONICAL_SUPPORT_TIER=hosted_selected`;
- sets `SPARSE_CANONICAL_CLAIM_BOUNDARY=hosted_selected_threshold_free`;
- sets `SPARSE_CANONICAL_RUNNER_CONTEXT=github-actions-macos-latest`;
- sets `SPARSE_CANONICAL_BUILD_FLAGS=default_make_flags`;
- sets `SPARSE_CANONICAL_BUILD_MODE=serial`;
- captures `SPARSE_CANONICAL_CPU_MODEL` with
  `sysctl -n machdep.cpu.brand_string`, falling back to `unknown`;
- runs `make bench-canonical-report`;
- runs `python3 scripts/check_bench_canonical_freshness.py --report-dir build/bench-reports/canonical --mode hosted`;
- prints selected metadata and uploaded selected paths;
- uploads only:
  - `build/bench-reports/canonical/bench_refactor_csc.csv`;
  - `build/bench-reports/canonical/index.tsv`;
  - `build/bench-reports/canonical/manifest.txt`.

### Validator Scope

No production C code or public header changed. The dedicated benchmark
freshness checker remains the hard artifact validator. Day 6 did not move
workflow YAML validation into `scripts/check_bench_canonical_freshness.py`;
workflow structure remains owned by `tests/test_selected_comparison_workflow.py`.

The selected benchmark checker now has explicit regression coverage for a
macOS-style hosted metadata row and selected `relative_path` drift. Existing
hosted checks still enforce non-empty metadata, selected CSV agreement,
manifest agreement, threshold-free values, and unselected-row demotion.

### Claim Boundary

Day 6 implements macOS hosted selected benchmark freshness plumbing, but does
not yet update all public documentation claim surfaces. Until docs calibration
and hosted CI evidence review complete, public wording should remain cautious:

- no portable performance claim;
- no timing threshold or regression baseline;
- no broad benchmark-family publication;
- no Windows selected benchmark freshness;
- no platform parity, benchmark superiority, package/ABI proof, runtime-loader
  proof, OpenMP speedup evidence, backend superiority, or state-of-the-art
  claim.

### Day 6 Validation

Commands run:

```sh
python3 tests/test_selected_comparison_workflow.py
python3 tests/test_bench_canonical_freshness.py
python3 tests/test_selected_report_targets_manifest.py
python3 tests/test_selected_performance_docs.py
git diff --check
find docs/planning/EPIC_18/SPRINT_202 -type f -name '*.md' -print0 | xargs -0 grep -n '[[:blank:]]$'
git diff --name-only -- '*.c' '*.h'
```

Results:

- `python3 tests/test_selected_comparison_workflow.py`: passed.
- `python3 tests/test_bench_canonical_freshness.py`: passed.
- `python3 tests/test_selected_report_targets_manifest.py`: passed.
- `python3 tests/test_selected_performance_docs.py`: passed.
- `git diff --check`: passed.
- Sprint 202 markdown trailing whitespace check: passed.
- `git diff --name-only -- '*.c' '*.h'`: no output.

No `.c` or `.h` files were modified, so the full C quality gate is not
required for Day 6.

## Day 7: Freshness Regression Fixtures

### Fixture Coverage Added

Day 7 expanded `tests/test_bench_canonical_freshness.py` with focused negative
fixtures for the selected benchmark freshness diagnostics required by the
Sprint 202 plan.

| Diagnostic class | Day 7 test coverage |
| --- | --- |
| Passing selected-platform artifact | `test_positive_macos_hosted_report_metadata` confirms hosted macOS-style metadata passes. |
| Missing selected artifact | `test_missing_selected_benchmark_artifact_fails` removes `bench_refactor_csc.csv` and expects `benchmark_selected_artifact_missing`. |
| Missing selected row | `test_missing_selected_index_row_fails` removes the selected `bench_refactor_csc` index row and expects `benchmark_selected_row_missing`. |
| Duplicate selected row | `test_duplicate_selected_index_row_fails` duplicates the selected index row and expects `benchmark_selected_row_duplicate`. |
| Malformed metadata | `test_malformed_selected_timestamp_fails` mutates `generated_at_utc` to a non-ISO value and expects a timestamp-shape failure. |
| Hosted local-placeholder metadata | `test_hosted_runner_context_cannot_be_local_placeholder` and `test_hosted_report_label_cannot_be_unlabeled_placeholder` prove hosted mode rejects local placeholders. |
| Deferred unselected artifacts | `test_unselected_rows_cannot_be_hosted_selected` and `test_positive_hosted_report_keeps_unselected_rows_local` keep unselected canonical rows local-only. |
| Path drift | `test_selected_relative_path_drift_fails` and `test_selected_relative_path_dot_prefix_drift_fails` reject non-basename selected artifact paths. |

### Helper Changes

Added local test helpers:

- `remove_artifact_row(report_dir, artifact)`;
- `duplicate_artifact_row(report_dir, artifact)`.

These helpers mutate copied generated fixtures under a temporary directory and
do not affect committed generated artifacts.

### Existing Coverage Preserved

The expanded suite still covers:

- selected manifest agreement;
- selected CSV field agreement;
- missing required CSV column;
- extra selected CSV row;
- selected matrix size, warmup, variance, baseline, threshold, and status
  drift;
- manifest/index mismatch;
- generator rejection of tab/newline control characters in metadata.

### Day 7 Validation

Commands run:

```sh
python3 tests/test_bench_canonical_freshness.py
python3 tests/test_selected_comparison_workflow.py
python3 tests/test_selected_report_targets_manifest.py
python3 tests/test_selected_performance_docs.py
git diff --name-only -- '*.c' '*.h'
```

Results:

- `python3 tests/test_bench_canonical_freshness.py`: passed.
- `python3 tests/test_selected_comparison_workflow.py`: passed.
- `python3 tests/test_selected_report_targets_manifest.py`: passed.
- `python3 tests/test_selected_performance_docs.py`: passed.
- `git diff --name-only -- '*.c' '*.h'`: no output.

No `.c` or `.h` files were modified, so the full C quality gate is not
required for Day 7.

## Day 8: Hosted Workflow Lane Implementation

### Day 8 Scope Review

Day 8 was the hosted workflow lane implementation checkpoint. The workflow
lane was already implemented during Day 6 so that the manifest, workflow, and
guards could be validated together. Day 8 audited the implemented lane against
the Day 8 completion criteria and ran a local hosted-mode simulation with the
macOS metadata contract.

### Hosted Lane Evidence

| Requirement | Evidence |
| --- | --- |
| Hosted workflow job exists for selected platform/row pair | `.github/workflows/macos-ci.yml` contains job `selected-performance-freshness` with name `macOS reviewed hosted selected performance freshness`. |
| Runtime budget is bounded | The job uses `timeout-minutes: 10`. |
| Selected metadata is wired | The job sets `BENCH_CANONICAL_REPORT_LABEL=sprint202-macos-hosted-performance`, `SPARSE_CANONICAL_SUPPORT_TIER=hosted_selected`, `SPARSE_CANONICAL_CLAIM_BOUNDARY=hosted_selected_threshold_free`, `SPARSE_CANONICAL_RUNNER_CONTEXT=github-actions-macos-latest`, `SPARSE_CANONICAL_BUILD_FLAGS=default_make_flags`, and `SPARSE_CANONICAL_BUILD_MODE=serial`. |
| macOS CPU metadata is captured | The job uses `sysctl -n machdep.cpu.brand_string`, falling back to `unknown`. |
| Benchmark command executes | The job runs `make bench-canonical-report`. |
| Hosted freshness check executes | The job runs `python3 scripts/check_bench_canonical_freshness.py --report-dir build/bench-reports/canonical --mode hosted`. |
| Selected artifact upload is bounded | The job uploads only `bench_refactor_csc.csv`, `index.tsv`, and `manifest.txt`. |
| CI wording avoids broad claims | The job comment excludes timing thresholds, portable performance, external-library comparison, broad benchmark-family publication, package/ABI claims, Windows selected benchmark freshness, and state-of-the-art claims. |

### Local Hosted-Mode Simulation

Command run:

```sh
BENCH_CANONICAL_REPORT_LABEL=sprint202-macos-hosted-performance \
SPARSE_CANONICAL_SUPPORT_TIER=hosted_selected \
SPARSE_CANONICAL_CLAIM_BOUNDARY=hosted_selected_threshold_free \
SPARSE_CANONICAL_RUNNER_CONTEXT=github-actions-macos-latest \
SPARSE_CANONICAL_BUILD_FLAGS=default_make_flags \
SPARSE_CANONICAL_BUILD_MODE=serial \
SPARSE_CANONICAL_CPU_MODEL=local-macos-simulation \
make bench-canonical-report &&
python3 scripts/check_bench_canonical_freshness.py \
  --report-dir build/bench-reports/canonical \
  --mode hosted
```

Result:

```text
bench-canonical-report: wrote build/bench-reports/canonical
bench-canonical-freshness: passed (mode=hosted; artifact=bench_refactor_csc; report_dir=build/bench-reports/canonical)
```

Selected row fields observed from `build/bench-reports/canonical/index.tsv`:

| Field | Observed value |
| --- | --- |
| `report_label` | `sprint202-macos-hosted-performance` |
| `runner_context` | `github-actions-macos-latest` |
| `build_flags` | `default_make_flags` |
| `cpu_model` | `local-macos-simulation` |
| `build_mode` | `serial` |
| `support_tier` | `hosted_selected` |
| `claim_boundary` | `hosted_selected_threshold_free` |
| `baseline` | `n/a` |
| `threshold` | `n/a` |
| `warmup` | `none_configured` |
| `variance` | `not_computed_single_sample` |
| `methodology_notes` | `threshold_free_local_measurement;not_portable_performance_claim` |

### Selected Upload Scope

The macOS job uploads:

- `build/bench-reports/canonical/bench_refactor_csc.csv`;
- `build/bench-reports/canonical/index.tsv`;
- `build/bench-reports/canonical/manifest.txt`.

The job does not upload:

- `build/bench-reports/canonical/**`;
- `build/bench-reports/**`;
- `build/bench-reports/canonical/bench_chol_csc.csv`;
- `build/bench-reports/canonical/bench_iterative_reuse.csv`;
- `build/bench-reports/canonical/bench_eigs_reuse.csv`.

### Item 202.3 Status

Item 202.3 has workflow evidence for the selected macOS hosted lane. Hosted CI
execution still needs to be reviewed after the branch is pushed and GitHub
Actions runs, but the source-controlled workflow definition, selected manifest
metadata, local hosted-mode checker simulation, and workflow guard coverage are
present.

### Day 8 Validation

Commands run:

```sh
sed -n '250,330p' docs/planning/EPIC_18/SPRINT_202/PLAN.md
sed -n '760,980p' docs/planning/EPIC_18/SPRINT_202/WORKING_NOTES.md
sed -n '130,380p' .github/workflows/macos-ci.yml
sed -n '380,470p' tests/test_selected_comparison_workflow.py
sed -n '850,940p' tests/test_selected_comparison_workflow.py
BENCH_CANONICAL_REPORT_LABEL=sprint202-macos-hosted-performance \
SPARSE_CANONICAL_SUPPORT_TIER=hosted_selected \
SPARSE_CANONICAL_CLAIM_BOUNDARY=hosted_selected_threshold_free \
SPARSE_CANONICAL_RUNNER_CONTEXT=github-actions-macos-latest \
SPARSE_CANONICAL_BUILD_FLAGS=default_make_flags \
SPARSE_CANONICAL_BUILD_MODE=serial \
SPARSE_CANONICAL_CPU_MODEL=local-macos-simulation \
make bench-canonical-report &&
python3 scripts/check_bench_canonical_freshness.py \
  --report-dir build/bench-reports/canonical \
  --mode hosted
python3 - <<'PY'
import csv
from pathlib import Path
rows = list(csv.DictReader(Path('build/bench-reports/canonical/index.tsv').open(newline=''), delimiter='\t'))
selected = [row for row in rows if row['artifact'] == 'bench_refactor_csc']
if len(selected) != 1:
    raise SystemExit(f'expected one selected row, got {len(selected)}')
row = selected[0]
for key in ['report_label','runner_context','build_flags','cpu_model','build_mode','support_tier','claim_boundary','baseline','threshold','warmup','variance','methodology_notes']:
    print(f'{key}={row[key]}')
PY
git diff -- .github/workflows/macos-ci.yml tests/corpus/manifests/selected_report_targets.tsv tests/test_selected_comparison_workflow.py tests/test_bench_canonical_freshness.py | sed -n '1,260p'
git status --short --branch
```

No `.c` or `.h` files were modified, so the full C quality gate is not
required for Day 8.

## Day 9: Workflow Guard And Local Simulation

### Day 9 Scope Review

Day 9 focused on local/static protection for the macOS selected benchmark
freshness workflow lane. The Day 8 workflow implementation already provided
the hosted job, selected upload, and local hosted-mode simulation; Day 9 added
macOS-specific guard drift fixtures so common workflow and manifest mistakes
fail with targeted diagnostics.

### Guard Additions

Added negative fixtures to `tests/test_selected_comparison_workflow.py`:

| Fixture | Drift guarded |
| --- | --- |
| `test_macos_performance_workflow_missing_job_fails_clearly` | Missing `selected-performance-freshness` job. |
| `test_macos_performance_workflow_wrong_runner_fails_clearly` | Runner drift away from `macos-latest`. |
| `test_macos_performance_workflow_missing_generation_step_fails_clearly` | Missing `run: make bench-canonical-report`. |
| `test_macos_performance_workflow_missing_checker_mode_fails_clearly` | Hosted checker mode drift away from `--mode hosted`. |
| `test_macos_performance_workflow_wrong_upload_path_fails_clearly` | Upload path drift from selected `bench_refactor_csc.csv` to an unselected benchmark CSV. |
| `test_macos_performance_manifest_missing_platform_fails_clearly` | Manifest platform mapping that omits `macos` while preserving list shape. |

These fixtures supplement the existing positive lane test and existing
unselected-upload rejection fixture.

### Guarded Contract

The selected macOS lane is still bounded to:

- `.github/workflows/macos-ci.yml`;
- job `selected-performance-freshness`;
- selected row `SRT-BENCH-REFACTOR-CSC-NOS4`;
- artifact `build/bench-reports/canonical/bench_refactor_csc.csv`;
- upload artifact `sprint202-macos-selected-performance-freshness`;
- `scripts/check_bench_canonical_freshness.py --mode hosted`.

### Local Hosted-Mode Simulation

Command retained for Day 9 validation:

```sh
BENCH_CANONICAL_REPORT_LABEL=sprint202-macos-hosted-performance \
SPARSE_CANONICAL_SUPPORT_TIER=hosted_selected \
SPARSE_CANONICAL_CLAIM_BOUNDARY=hosted_selected_threshold_free \
SPARSE_CANONICAL_RUNNER_CONTEXT=github-actions-macos-latest \
SPARSE_CANONICAL_BUILD_FLAGS=default_make_flags \
SPARSE_CANONICAL_BUILD_MODE=serial \
SPARSE_CANONICAL_CPU_MODEL=local-macos-simulation \
make bench-canonical-report &&
python3 scripts/check_bench_canonical_freshness.py \
  --report-dir build/bench-reports/canonical \
  --mode hosted
```

### Hosted Residuals

Hosted GitHub Actions still needs to confirm:

- the job starts on the actual `macos-latest` hosted runner;
- `sysctl -n machdep.cpu.brand_string` records usable CPU metadata;
- uploaded artifact contents remain exactly the selected CSV plus `index.tsv`
  and `manifest.txt`;
- the workflow summary reports one selected row without promoting portable
  performance or timing-threshold claims.

### Day 9 Validation

Commands run:

```sh
python3 tests/test_selected_comparison_workflow.py
python3 tests/test_bench_canonical_freshness.py
python3 tests/test_selected_report_targets_manifest.py
python3 tests/test_selected_performance_docs.py
BENCH_CANONICAL_REPORT_LABEL=sprint202-macos-hosted-performance \
SPARSE_CANONICAL_SUPPORT_TIER=hosted_selected \
SPARSE_CANONICAL_CLAIM_BOUNDARY=hosted_selected_threshold_free \
SPARSE_CANONICAL_RUNNER_CONTEXT=github-actions-macos-latest \
SPARSE_CANONICAL_BUILD_FLAGS=default_make_flags \
SPARSE_CANONICAL_BUILD_MODE=serial \
SPARSE_CANONICAL_CPU_MODEL=local-macos-simulation \
make bench-canonical-report &&
python3 scripts/check_bench_canonical_freshness.py \
  --report-dir build/bench-reports/canonical \
  --mode hosted
git diff --check
git diff --name-only -- '*.c' '*.h'
```

No `.c` or `.h` files were modified, so the full C quality gate is not
required for Day 9.

## Day 10: Documentation Calibration

### Day 10 Scope Review

Day 10 calibrated public and maintainer documentation after the Sprint 202
macOS hosted selected benchmark freshness lane and workflow guards were in
place. The goal was to describe the additional hosted freshness evidence
without implying portable performance, timing thresholds, broad platform
support, package-manager readiness, or state-of-the-art status.

### Documentation Updates

Updated claim surfaces:

| Surface | Change |
| --- | --- |
| `README.md` | Selected performance freshness now says reviewed Linux and macOS hosted performance lanes run the same selected-row freshness check with platform-specific hosted metadata. |
| `INSTALL.md` | Support matrix row now reads `Linux/macOS selected performance freshness` and explicitly keeps performance, threshold, package, platform-parity, and state-of-the-art non-claims. |
| `benchmarks/README.md` | Benchmark docs now name Linux/macOS hosted selected-performance lanes, include the macOS runner context, and reject Linux/macOS performance parity. |
| `docs/maintainer_guide.md` | Maintainer guide now lists both selected hosted artifacts, both runner contexts, and both hosted CPU metadata sources. |
| `tests/corpus/README.md` | Corpus selected-target interpretation now records Linux/macOS hosted metadata scope for the same selected row and rejects comparable timing claims. |
| `tests/corpus/schemas/report_index_fields.md` | Report-index schema guidance now states that hosted selected workflow metadata covers only Linux/macOS for the exact selected benchmark row. |
| `tests/test_selected_performance_docs.py` | Documentation guard now enforces the calibrated Linux/macOS selected-performance markers and INSTALL support row. |

### Item 202.5 Status

Item 202.5 documentation is complete for the branch path:

- the selected hosted freshness lane is documented as Linux/macOS hosted
  evidence for `SRT-BENCH-REFACTOR-CSC-NOS4`;
- the metadata contract remains `hosted_selected` /
  `hosted_selected_threshold_free`;
- the selected row stays `status=measurement`, `baseline=n/a`, and
  `threshold=n/a`;
- generated benchmark artifacts remain under ignored `build/` paths;
- public wording does not promote portable performance or timing pass/fail
  proof.

### Retained Non-Claims

The documentation now consistently rejects:

- portable performance;
- timing thresholds;
- Linux/macOS performance parity;
- Windows selected benchmark freshness;
- broad benchmark-family publication;
- Homebrew/core, bottles, Linuxbrew, public tap maintenance, or broad
  package-manager distribution;
- package/ABI proof;
- release readiness;
- external-library parity or backend superiority;
- state-of-the-art sparse linear algebra status.

### Day 10 Validation

Commands run:

```sh
python3 tests/test_selected_performance_docs.py
rg -n "reviewed Linux hosted selected-performance|Linux hosted selected-performance|Linux hosted performance lane|mirrored by reviewed Linux hosted CI|reviewed Linux hosted selected performance|Windows/macOS performance parity|macOS CI/install evidence remains separate platform/package evidence and does not inherit selected Linux hosted performance meaning" README.md INSTALL.md benchmarks/README.md docs/maintainer_guide.md tests/corpus/README.md tests/corpus/schemas/report_index_fields.md tests/test_selected_performance_docs.py
```

The stale selected-performance wording scan found only the unrelated
Linux-only oracle freshness command comment in `README.md`.

No `.c` or `.h` files were modified, so the full C quality gate is not
required for Day 10.

## Day 11: Focused Freshness Validation

### Day 11 Scope Review

Day 11 ran the focused validation set after the Sprint 202 macOS hosted
selected benchmark workflow, guard, manifest, and documentation updates. The
goal was to verify selected freshness behavior, manifest/report-index
interpretation, workflow guards, docs guards, and stale-claim scans before the
Day 12 integrated/hosted-evidence review.

### Freshness Validation

| Command | Result | Evidence |
| --- | --- | --- |
| `make bench-canonical-report-freshness` | Passed | Regenerated `build/bench-reports/canonical/` and passed the selected local freshness checker for `bench_refactor_csc`. |
| `python3 tests/test_bench_canonical_freshness.py` | Passed | Covered selected row identity, selected CSV contract, hosted metadata, missing/duplicate rows, path drift, malformed timestamp, threshold-free policy, and TSV control-character rejection. |
| `python3 scripts/normalize_report_index.py --family benchmark --check-freshness` | Passed | Reported advisory stale diagnostics for local benchmark rows and ended with `normalize-report-index: freshness ok (5 rows)`. |
| Hosted-mode local simulation | Passed | Generated the selected bundle with Sprint 202 macOS hosted metadata env values and passed `scripts/check_bench_canonical_freshness.py --mode hosted`. |

Hosted-mode simulation command:

```sh
BENCH_CANONICAL_REPORT_LABEL=sprint202-macos-hosted-performance \
SPARSE_CANONICAL_SUPPORT_TIER=hosted_selected \
SPARSE_CANONICAL_CLAIM_BOUNDARY=hosted_selected_threshold_free \
SPARSE_CANONICAL_RUNNER_CONTEXT=github-actions-macos-latest \
SPARSE_CANONICAL_BUILD_FLAGS=default_make_flags \
SPARSE_CANONICAL_BUILD_MODE=serial \
SPARSE_CANONICAL_CPU_MODEL=local-macos-simulation \
make bench-canonical-report &&
python3 scripts/check_bench_canonical_freshness.py \
  --report-dir build/bench-reports/canonical \
  --mode hosted
```

### Workflow, Manifest, And Docs Validation

| Command | Result | Evidence |
| --- | --- | --- |
| `python3 tests/test_selected_comparison_workflow.py` | Passed | Validated selected workflow guards, including macOS selected-performance job, runner, generation, checker mode, summary, and selected-only upload path. |
| `python3 tests/test_selected_report_targets_manifest.py` | Passed | Validated selected target manifest structure and Linux/macOS selected benchmark workflow metadata. |
| `python3 tests/test_selected_performance_docs.py` | Passed | Validated README, INSTALL, benchmark README, maintainer guide, corpus README, and report-index schema selected-performance markers. |
| `python3 -m py_compile ...` | Passed | Syntax-checked changed Python guards and selected freshness scripts. |

### Stale-Claim Scan

Command:

```sh
rg -n "selected performance (proves|guarantees) portable performance|selected performance (proves|is) state-of-the-art|hosted selected performance (is|acts as) a timing gate|bench-canonical-report-freshness (proves|guarantees) speedup|Linux/macOS performance parity" README.md INSTALL.md benchmarks/README.md docs/maintainer_guide.md tests/corpus/README.md tests/corpus/schemas/report_index_fields.md
```

Result:

- Found only the intentional non-claim in `benchmarks/README.md`:
  `Neither hosted row creates Linux/macOS performance parity...`.
- Found no unsupported selected-performance portable-performance,
  state-of-the-art, timing-gate, or speedup claims.

### Hosted Residual

At Day 11, the remaining evidence was hosted-only:

- GitHub Actions starts `selected-performance-freshness` on `macos-latest`;
- CPU metadata is captured through `sysctl -n machdep.cpu.brand_string`;
- uploaded artifact `sprint202-macos-selected-performance-freshness` contains
  only `bench_refactor_csc.csv`, `index.tsv`, and `manifest.txt`;
- hosted checker passes with `--mode hosted`;
- workflow summary reports one selected row and no broad performance claim.

### Day 11 Quality-Gate Decision

No `.c` or `.h` files were modified, so the full C quality gate is not
required for Day 11.

## Day 12: Integrated Validation And Hosted Evidence Review

### Day 12 Scope Review

Day 12 ran the integrated local validation set for changed workflow, manifest,
documentation, report-index, and selected benchmark freshness surfaces. The
same pass checked whether hosted CI evidence was available for the macOS
selected-performance lane.

### Changed Surface

Tracked changed files at validation time:

- `.github/workflows/macos-ci.yml`;
- `README.md`;
- `INSTALL.md`;
- `benchmarks/README.md`;
- `docs/maintainer_guide.md`;
- `tests/corpus/README.md`;
- `tests/corpus/manifests/selected_report_targets.tsv`;
- `tests/corpus/schemas/report_index_fields.md`;
- `tests/test_bench_canonical_freshness.py`;
- `tests/test_selected_comparison_workflow.py`;
- `tests/test_selected_report_targets_manifest.py`;
- `tests/test_selected_performance_docs.py`.

No `.c` or `.h` files were modified.

### Integrated Local Validation

| Command | Result | Evidence |
| --- | --- | --- |
| `make bench-canonical-report-freshness` | Passed | Regenerated the canonical report bundle and passed local selected freshness for `bench_refactor_csc`. |
| `python3 tests/test_bench_canonical_freshness.py` | Passed | Covered selected row identity, hosted metadata, manifest agreement, path drift, malformed metadata, threshold-free policy, and unselected-row demotion. |
| `python3 tests/test_selected_comparison_workflow.py` | Passed | Covered selected workflow guards, including the Sprint 202 macOS selected-performance lane. |
| `python3 tests/test_selected_performance_docs.py` | Passed | Covered selected-performance documentation markers and overclaim rejection. |
| `python3 tests/test_selected_report_targets_manifest.py` | Passed | Covered selected target manifest structure and Linux/macOS selected benchmark workflow metadata. |
| `python3 tests/test_normalize_report_index.py` | Passed | Covered normalized report-index construction and freshness behavior. |
| `python3 scripts/normalize_report_index.py --check` | Passed | Reported `normalize-report-index: 151 rows ok`. |
| `python3 scripts/normalize_report_index.py --family benchmark --check-freshness` | Passed | Reported advisory local benchmark rows and `normalize-report-index: freshness ok (5 rows)`. |
| `python3 -m py_compile ...` | Passed | Syntax-checked changed Python guards and selected freshness/normalizer scripts. |
| Hosted-mode local simulation | Passed | Generated canonical reports with Sprint 202 macOS hosted metadata and passed `check_bench_canonical_freshness.py --mode hosted`. |

Hosted-mode local simulation command:

```sh
BENCH_CANONICAL_REPORT_LABEL=sprint202-macos-hosted-performance \
SPARSE_CANONICAL_SUPPORT_TIER=hosted_selected \
SPARSE_CANONICAL_CLAIM_BOUNDARY=hosted_selected_threshold_free \
SPARSE_CANONICAL_RUNNER_CONTEXT=github-actions-macos-latest \
SPARSE_CANONICAL_BUILD_FLAGS=default_make_flags \
SPARSE_CANONICAL_BUILD_MODE=serial \
SPARSE_CANONICAL_CPU_MODEL=local-macos-simulation \
make bench-canonical-report &&
python3 scripts/check_bench_canonical_freshness.py \
  --report-dir build/bench-reports/canonical \
  --mode hosted
```

### Claim Scan

Command:

```sh
rg -n "selected performance (proves|guarantees) portable performance|selected performance (proves|is) state-of-the-art|hosted selected performance (is|acts as) a timing gate|bench-canonical-report-freshness (proves|guarantees) speedup|Linux/macOS performance parity" README.md INSTALL.md benchmarks/README.md docs/maintainer_guide.md tests/corpus/README.md tests/corpus/schemas/report_index_fields.md
```

Result:

- Found only the intended non-claim in `benchmarks/README.md`:
  `Neither hosted row creates Linux/macOS performance parity...`.
- Found no unsupported selected-performance portable-performance,
  state-of-the-art, timing-gate, or speedup claims.

### Hosted Evidence Review

Commands run:

```sh
gh run list --branch sprint-202 --limit 10
git rev-parse --abbrev-ref --symbolic-full-name @{u}
```

Results:

- `gh run list --branch sprint-202 --limit 10`: returned no runs.
- `git rev-parse --abbrev-ref --symbolic-full-name @{u}` failed with
  `fatal: no upstream configured for branch 'sprint-202'`.

This was the pre-push state. After PR #224 ran, hosted evidence was reviewed:

- run `34517520951`, job `103006563210`, completed successfully;
- check name: `macOS reviewed hosted selected performance freshness`;
- commit: `52890043be0d2325f2e4534a480d5f1cfc7ce23e`;
- CPU metadata: `Apple M1 (Virtual)`;
- hosted freshness checker passed with `--mode hosted`;
- artifact `sprint202-macos-selected-performance-freshness`, id
  `10168283869`, digest
  `sha256:308ee8f780a28673fc02ff40b10aeb17ae6219fcdf55095657526b76f7016af4`;
- upload log reported exactly three files: `bench_refactor_csc.csv`,
  `index.tsv`, and `manifest.txt`;
- workflow summary retained `support_tier=hosted_selected`,
  `claim_boundary=hosted_selected_threshold_free`, and
  `threshold_free_no_portable_performance_claim`.

### Hosted Evidence Checklist

After branch push and PR creation, the macOS workflow run was reviewed for:

- job `selected-performance-freshness` started on `macos-latest`: passed in
  job `103006563210`;
- CPU metadata captured through `sysctl -n machdep.cpu.brand_string`: reported
  `Apple M1 (Virtual)`;
- `make bench-canonical-report` completed: passed in hosted job;
- `scripts/check_bench_canonical_freshness.py --mode hosted` passed: passed in
  hosted job;
- artifact `sprint202-macos-selected-performance-freshness` uploaded exactly
  `bench_refactor_csc.csv`, `index.tsv`, and `manifest.txt`: upload log
  reported exactly three files and artifact id `10168283869`;
- workflow summary reported one selected row and no timing-threshold,
  portable-performance, broad-platform, package/ABI, release, or
  state-of-the-art claim: summary retained bounded selected metadata and
  non-claim wording.

### Day 12 Quality-Gate Decision

No `.c` or `.h` files changed, so `make format && make lint && make test` is
not required for Day 12.

## Day 13: Review Hardening

### Day 13 Scope Review

Day 13 audited the accumulated Sprint 202 diff for unnecessary breadth, stale
claims, stale paths, diagnostic traceability, and residual-queue consistency.
The branch still changes exactly one additional hosted selected benchmark
freshness lane: macOS hosted selected freshness for
`SRT-BENCH-REFACTOR-CSC-NOS4`.

### Diff-Scope Audit

The reviewed selected lane is:

- workflow: `.github/workflows/macos-ci.yml`;
- job: `selected-performance-freshness`;
- selected row: `SRT-BENCH-REFACTOR-CSC-NOS4`;
- benchmark artifact: `build/bench-reports/canonical/bench_refactor_csc.csv`;
- upload artifact: `sprint202-macos-selected-performance-freshness`;
- checker: `scripts/check_bench_canonical_freshness.py --mode hosted`.

No production C source, public C header, Makefile target, CMake registration,
benchmark binary, source-list, package recipe, install script, or Windows
workflow was changed.

### Review-Hardening Changes

| Surface | Change |
| --- | --- |
| `docs/maintainer_guide.md` | Updated the high-level selected performance evidence row to name Sprint 202 artifacts, benchmark workflows, and Linux/macOS hosted selected lanes. |
| `tests/test_selected_performance_docs.py` | Added markers that guard the Sprint 202 maintainer-summary wording. |
| `docs/planning/EPIC_18/EPIC_18_RESIDUAL_QUEUE.md` | Updated `E18-RQ-005` from pending-future wording to the current Sprint 202 state; after PR creation it records hosted CI evidence for run `34517520951`. |

### Diagnostic Traceability

| Diagnostic area | Evidence |
| --- | --- |
| Missing selected artifact/report directory | `test_missing_selected_benchmark_artifact_fails`; Day 12 local freshness pass. |
| Missing/duplicate selected index row | `test_missing_selected_index_row_fails`; `test_duplicate_selected_index_row_fails`. |
| CSV schema/value drift | `test_selected_benchmark_csv_missing_required_column_fails`; `test_selected_benchmark_csv_wrong_fixture_fails`; `test_selected_benchmark_csv_extra_row_fails`. |
| Methodology metadata drift | `test_selected_matrix_size_is_required`; `test_selected_warmup_is_required`; `test_selected_variance_is_required`; `test_malformed_selected_timestamp_fails`. |
| Threshold-free policy drift | `test_selected_baseline_stays_threshold_free`; `test_selected_threshold_stays_threshold_free`; `test_selected_status_cannot_become_performance_pass_claim`. |
| Hosted metadata drift | `test_positive_macos_hosted_report_metadata`; `test_hosted_runner_context_cannot_be_local_placeholder`; `test_hosted_report_label_cannot_be_unlabeled_placeholder`. |
| Selected path drift | `test_selected_relative_path_drift_fails`; `test_selected_relative_path_dot_prefix_drift_fails`; workflow wrong-upload-path fixture. |
| Unselected row promotion | `test_unselected_rows_cannot_be_hosted_selected`; `test_positive_hosted_report_keeps_unselected_rows_local`; workflow unselected-upload fixtures. |
| Manifest/platform mapping drift | `test_selected_benchmark_manifest_matches_checker_contract`; `test_macos_performance_manifest_missing_platform_fails_clearly`. |

### Residual Queue Audit

`E18-RQ-005` now records:

- local/static proof is complete for the Sprint 202 branch path;
- hosted GitHub Actions evidence was reviewed after PR creation through run
  `34517520951`, job `103006563210`;
- closure is scoped to macOS hosted selected benchmark freshness for
  `SRT-BENCH-REFACTOR-CSC-NOS4`;
- retained non-claims include portable performance, timing thresholds,
  Linux/macOS performance parity, Windows selected benchmark freshness, broad
  benchmark-family publication, package-manager distribution, package/ABI
  support, backend superiority, release benchmark readiness, and
  state-of-the-art performance.

### Day 13 Validation

Commands run:

```sh
python3 tests/test_selected_performance_docs.py
rg -n "FreshnessError|raise ValidationError|benchmark_selected|hosted|runner_context|report_label|relative_path|duplicate|missing|threshold|baseline|not_portable" scripts/check_bench_canonical_freshness.py tests/test_bench_canonical_freshness.py
rg -n "sprint202-macos-selected-performance-freshness|github-actions-macos-latest|selected-performance-freshness|bench_refactor_csc.csv|bench_chol_csc.csv|bench_iterative_reuse.csv|bench_eigs_reuse.csv" .github/workflows/macos-ci.yml tests/test_selected_comparison_workflow.py tests/corpus/manifests/selected_report_targets.tsv README.md INSTALL.md benchmarks/README.md docs/maintainer_guide.md tests/corpus/README.md tests/corpus/schemas/report_index_fields.md docs/planning/EPIC_18/EPIC_18_RESIDUAL_QUEUE.md
rg -n "portable performance|timing threshold|timing thresholds|performance parity|state-of-the-art|package-manager|bottles|Linuxbrew|Windows selected benchmark|broad benchmark" README.md INSTALL.md benchmarks/README.md docs/maintainer_guide.md tests/corpus/README.md tests/corpus/schemas/report_index_fields.md docs/planning/EPIC_18/EPIC_18_RESIDUAL_QUEUE.md
```

The broad claim/path scan found only intended non-claims and scoped selected
performance references.

No `.c` or `.h` files changed, so the full C quality gate is not required for
Day 13.

## Day 14: Closeout And Retrospective Prep

### Day 14 Scope Review

Day 14 finalized the Sprint 202 evidence set and prepared retrospective inputs.
The selected lane remains bounded to one additional hosted selected benchmark
freshness path:

- platform: macOS hosted runner;
- workflow: `.github/workflows/macos-ci.yml`;
- job: `selected-performance-freshness`;
- selected row: `SRT-BENCH-REFACTOR-CSC-NOS4`;
- benchmark artifact: `build/bench-reports/canonical/bench_refactor_csc.csv`;
- workflow artifact: `sprint202-macos-selected-performance-freshness`;
- checker: `scripts/check_bench_canonical_freshness.py --mode hosted`.

### Final Item Status

| Item | Status | Evidence |
| --- | --- | --- |
| 202.1 Platform And Row Selection | Complete for selected macOS lane | Days 1-3 selected macOS hosted freshness for `SRT-BENCH-REFACTOR-CSC-NOS4` and deferred broader row/platform expansion. |
| 202.2 Methodology Metadata | Complete for selected macOS lane | Days 4-6 defined hosted metadata, selected row identity, and threshold-free interpretation. |
| 202.3 Workflow Lane | Complete for selected macOS lane | Day 8 added the macOS selected-performance job and PR run `34517520951` confirmed hosted execution. |
| 202.4 Freshness Tests | Complete for selected macOS lane | Days 6-7 and Day 13 cover selected freshness diagnostics, hosted metadata, path drift, and unselected-row promotion. |
| 202.5 Docs Calibration | Complete for selected macOS lane | Days 10 and 13 calibrated public, maintainer, corpus, schema, and residual-queue wording. |
| 202.6 Validation | Complete for selected macOS lane | Days 11-14 record focused checks; PR run `34517520951` records hosted execution and selected artifact upload evidence. |

### Closeout Artifact

Day 14 added:

- `docs/planning/EPIC_18/SPRINT_202/artifacts/day14-closeout-retrospective-inputs.md`.

The artifact records final item status, changed surface, validation scope,
hosted residuals, retrospective inputs, and generated-file/environment
boundaries.

### Hosted Evidence Status

`E18-RQ-005` is closed for the selected macOS hosted lane after PR #224 hosted
evidence review. Run `34517520951`, job `103006563210`, completed successfully,
passed hosted-mode freshness, and uploaded artifact
`sprint202-macos-selected-performance-freshness` with exactly
`bench_refactor_csc.csv`, `index.tsv`, and `manifest.txt`. Broader benchmark,
platform, package, release, timing-threshold, and state-of-the-art claims remain
deferred.

### Retrospective Inputs

Prepared retrospective inputs:

- completed work: macOS hosted selected performance workflow, selected-only
  artifact upload, manifest metadata, freshness fixtures, workflow guards, and
  claim-calibrated docs;
- validation: focused Python guards/regressions, selected manifest checks,
  benchmark freshness checks, py_compile, whitespace checks, and the Day 12
  hosted-mode local simulation;
- residuals: no selected macOS hosted evidence residual remains after PR run
  `34517520951`; deferred non-selected claims remain separate;
- deferred items: Windows selected benchmark freshness, broad benchmark matrix
  publication, timing-threshold promotion, package-manager distribution,
  package/ABI support, and state-of-the-art performance claims;
- recommendation: use the hosted run and artifact evidence as the Sprint 202
  selected macOS evidence anchor without broadening the public claim surface.

### Environment And Generated-File Check

The Day 14 closeout keeps local absolute paths, temporary directories,
generated benchmark CSV contents, and hosted secrets out of the planned change
set. Paths recorded in the closeout are repository-relative or workflow artifact
paths.

### Day 14 Quality-Gate Decision

No `.c` or `.h` files changed, so `make format && make lint && make test` is
not required for Day 14.
