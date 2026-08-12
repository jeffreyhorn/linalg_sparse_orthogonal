# Sprint 154 Working Notes

## Goal

Sprint 154 builds the first direct external comparison harness and publishes
one narrow evidence-backed comparison study without overclaiming ecosystem
parity.

## Starting Evidence

- Sprint 150 expanded the maintained QR corpus to six fixture-local rows:
  `qr_rank_deficient_6x4_nullspace_v1`,
  `qr_rankdef_duplicate_5x4_v1`,
  `qr_rankdef_dependent_row_4x3_v1`,
  `qr_underdetermined_minnorm_2x4`,
  `qr_minnorm_3x6_exact_values`, and
  `qr_minnorm_5x10_exact_values`.
- Sprint 151 expanded the maintained partial-SVD corpus to four fixture-local
  rows:
  `partial_svd_clustered_repeated_diag8x6_k3_v1`,
  `partial_svd_rankdef_diag6x4_k2_range_projector_v1`,
  `partial_svd_lowrank_rect5x7_k3_sparse_output_v1`, and
  `partial_svd_fail_closed_diag6_k2_v1`.
- Sprint 152 selected the local generated oracle freshness gate
  `make report-index-oracle-freshness`, which regenerates combined QR plus
  partial-SVD oracle output and checks selected oracle freshness.
- Sprint 153 selected stronger static-first package deferral and explicitly
  rejected shared-library support until export/import, symbol visibility,
  dynamic ABI, platform loader metadata, installed shared consumer proof, and
  runtime-loader validation exist.
- Current corpus manifests live under `tests/corpus/manifests/`; expected
  result rows live under `tests/corpus/expected/`; report-index schema docs
  live under `tests/corpus/schemas/`.
- Current report families include source-controlled corpus/package/CI/docs
  policy rows and generated-local oracle/benchmark/sentinel/guardrail/deadcode
  rows.

## Item-To-Day Owner Map

| Sprint 154 Item | Primary Days | Closeout Owner |
| --- | --- | --- |
| Item 1: Comparison Target Selection | Days 1-3 | Day 1 records boundaries, Day 2 audits candidates, Day 3 selects one target. |
| Item 2: Dependency Pinning | Day 4 | Day 4 defines external dependency, version, provenance, and skip/defer policy. |
| Item 3: Output Schema | Day 5 | Day 5 designs comparison row fields, status semantics, and report meaning. |
| Item 4: Harness Implementation | Days 6-9 | Day 6 designs the harness, Days 7-9 implement runners and comparison logic. |
| Item 5: Report Integration | Days 10-11 | Day 10 selects report behavior, Day 11 implements normalization or artifact-only policy. |
| Item 6: Documentation Alignment | Day 12 | Day 12 aligns maintainer, report, solver, README, and non-claim wording. |
| Item 7: Validation And Study Publication | Days 13-14 | Day 13 publishes the first study, Day 14 closes with Sprint 155 handoff. |

## Stop Conditions

- A narrow fixture-local comparison is described as broad external-library,
  LAPACK, NumPy, SciPy, SuiteSparse, Eigen, CHOLMOD, PETSc, Trilinos, or
  ecosystem parity.
- Missing optional external dependencies produce pass evidence.
- Skipped or deferred comparison rows are counted as proof.
- Local generated comparison rows are described as hosted CI, release,
  package, ABI, platform, performance, or state-of-the-art proof.
- Timing or wall-clock data is used as portable performance superiority
  evidence without a separate benchmark methodology and platform policy.
- Static archive package proof is used to infer shared-library support,
  dynamic ABI compatibility, runtime-loader behavior, package-manager
  distribution, Windows Makefile parity, or Windows `pkg-config` execution
  parity.
- Raw QR basis identity, QR sign/orientation/order parity, raw singular-vector
  identity, singular-vector sign/phase/order parity, broad convergence-rate
  guarantees, or portable iteration-count guarantees are claimed from one
  narrow comparison.
- Report-index rows are added without provenance fields that identify command,
  version, platform, compiler, fixture, metric, status, caveat, and artifact
  path or without documenting why those fields are intentionally deferred.
- `.c` or public `.h` changes land without the required full
  `make format && make lint && make test` quality gate.

## Daily Log

### Day 1: Sprint Intake And Comparison Boundary

- Re-read the Sprint 154 section of
  `docs/planning/EPIC_13/PROJECT_PLAN.md`.
- Reviewed the Sprint 150 QR retrospective, Sprint 151 partial-SVD
  retrospective, Sprint 152 generated report freshness baseline, and Sprint
  153 closeout handoff.
- Created the Sprint 154 artifact directory.
- Created
  `docs/planning/EPIC_13/SPRINT_154/artifacts/day1-comparison-boundary.md`.
- Inventoried current comparison starting points:
  - `tests/corpus/manifests/fixtures.tsv` defines six maintained QR fixtures
    and four maintained partial-SVD fixtures;
  - `tests/corpus/manifests/generators.tsv` defines deterministic generator
    metadata and hashes for the maintained QR and partial-SVD fixture rows;
  - `tests/corpus/expected/*.tsv` currently holds `49` source-controlled
    expected rows across the selected QR and partial-SVD fixtures;
  - `tests/corpus/manifests/report_families.tsv` defines source-controlled
    corpus/package/CI/documentation policy rows and generated-local oracle
    report rows;
  - `Makefile` owns the selected local oracle freshness target
    `make report-index-oracle-freshness`;
  - `scripts/run_corpus_oracle.py` emits generated-local QR and partial-SVD
    oracle rows under ignored `build/` paths;
  - `scripts/normalize_report_index.py` validates normalized report-index
    structure and freshness meaning.
- Recorded Day 1 comparison stop conditions for external-library parity,
  performance, package-manager, shared-library, hosted CI, platform, ABI, and
  state-of-the-art claims.
- Day 2 handoff: audit QR and partial-SVD candidate families against external
  dependency availability, metric clarity, skip/defer behavior, report
  integration cost, and overclaim risk.

### Day 2: Target Candidate Audit

- Created
  `docs/planning/EPIC_13/SPRINT_154/artifacts/day2-target-candidate-audit.md`.
- Audited the Sprint 150 QR corpus candidate:
  - six maintained fixtures;
  - `23` source-controlled expected rows;
  - focused proof owner `tests/test_qr_corpus.c`;
  - generated-local oracle path via
    `scripts/run_corpus_oracle.py --include-solver-qr`;
  - existing prior-art dense-reference helper
    `tests/qr_external_dense_reference.py`.
- Audited the Sprint 151 partial-SVD corpus candidate:
  - four maintained fixtures;
  - `26` source-controlled expected rows;
  - focused proof owner `tests/test_svd_partial_corpus.c`;
  - generated-local oracle path via
    `scripts/run_corpus_oracle.py --include-partial-svd`;
  - existing prior-art dense singular-value helper
    `tests/svd_external_dense_reference.py`.
- Identified external baseline candidates:
  - existing source-controlled Python stdlib dense-reference helpers;
  - optional NumPy;
  - optional SciPy;
  - LAPACK/system tooling;
  - SuiteSparse/CHOLMOD;
  - Eigen helper binary.
- Recorded that the existing Python helpers are external-process dense
  references, not external package or ecosystem parity proof.
- Scored QR higher for the first narrow study because the minimum-norm fixture
  has simpler metrics, lower parsing cost, and lower wording risk than the
  richer partial-SVD subspace/sparse-output/fail-closed fixtures.
- Recommended Day 3 bias: select `qr_underdetermined_minnorm_2x4` as the
  first narrow comparison target unless dependency policy rules it out.
- Day 3 handoff: select exactly one target and freeze fixture keys, baseline
  type, accepted metrics, tolerance policy, skip/defer semantics, non-claims,
  and report integration expectation.

### Day 3: Comparison Target Selection

- Created
  `docs/planning/EPIC_13/SPRINT_154/artifacts/day3-comparison-target-selection.md`.
- Selected the first narrow comparison target:
  `qr_underdetermined_minnorm_2x4`.
- Frozen fixture facts:
  - QR underdetermined minimum-norm family;
  - shape `2 x 4`;
  - `4` nonzeros;
  - full row rank `2`;
  - nullity `2`;
  - explicit RHS;
  - expected solution `[0.5, 0.5, 0.5, 0.5]`;
  - expected solution norm `1.0`;
  - expected residual `<= 1e-10`.
- Selected baseline posture:
  external-process dense reference, source-controlled helper first, with
  optional NumPy/SciPy package comparison deferred unless Day 4 can define
  clean dependency discovery, version capture, skip/defer behavior, and
  non-package-manager wording.
- Frozen accepted metrics:
  - project status;
  - baseline status;
  - residual norm;
  - solution norm;
  - solution values;
  - project-vs-baseline max absolute solution delta.
- Frozen tolerance policy from
  `tests/corpus/expected/qr_underdetermined_minnorm_2x4.tsv`:
  status exact, residual absolute `1e-10`, solution norm absolute `1e-10`,
  and solution values absolute `1e-10` per component.
- Defined statuses `pass`, `fail`, `skip`, `defer`, and `error`; only `pass`
  counts as fixture-local comparison proof.
- Deferred broader QR, QR subspace, QR reorder/COLAMD, partial-SVD,
  NumPy/SciPy package-baseline, and timing/performance comparison targets.
- Day 4 handoff: define canonical baseline command, baseline dependency
  posture, version capture, interpreter/executable discovery, skip/defer
  behavior, provenance fields, and external package safety boundaries.

### Day 4: Dependency Pinning Policy

- Created
  `docs/planning/EPIC_13/SPRINT_154/artifacts/day4-dependency-pinning-policy.md`.
- Selected a dependency-light baseline posture for the first study:
  source-controlled external-process dense reference, not optional external
  package baseline.
- Defined the canonical baseline command:
  `python3 tests/qr_external_dense_reference.py qr_underdetermined_minnorm_2x4`.
- Defined required dependencies:
  - current Git worktree and source provenance;
  - `python3` interpreter;
  - `tests/qr_external_dense_reference.py`;
  - project-side static build or runner selected by later harness design.
- Deferred NumPy and SciPy package baselines; missing optional packages should
  be `defer`, not `pass` or `fail`, until explicitly selected.
- Defined provenance fields for source commit/branch/worktree state, baseline
  name/type/version/command/Python executable/Python version, project command,
  project version, platform, compiler, configuration, fixture key, metric,
  tolerance, status, caveat, and artifact path.
- Defined status semantics for dependency outcomes:
  required selected baseline missing is `error`; optional package missing is
  `defer`; only selected metric matches can be `pass`.
- Defined security and reproducibility boundaries: no network access, package
  installation, arbitrary binary discovery, or package-manager support
  inference.
- Day 5 handoff: design comparison row or artifact schema around the selected
  source-controlled baseline and decide whether normalized comparison rows are
  ready or the first study should remain artifact-only.

### Day 5: Comparison Output Schema Design

- Created
  `docs/planning/EPIC_13/SPRINT_154/artifacts/day5-comparison-output-schema-design.md`.
- Inventoried current report-index schema and report-family metadata:
  - source-controlled contract rows are advisory or policy rows, not pass
    evidence;
  - generated-local oracle rows use
    `freshness_policy=generated_compare_inputs`;
  - local generated outputs are not hosted CI, package, ABI, platform,
    performance, release, external-library parity, or state-of-the-art proof.
- Selected artifact-first comparison output for the first study rather than
  adding a normalized `comparison` report family immediately.
- Reserved the generated local output root:
  `build/comparison/qr_minnorm/`.
- Proposed generated artifact names:
  - `study.tsv` for metric-level rows;
  - `summary.md` for the human-readable narrow study;
  - `manifest.tsv` for run-level provenance and row-count summary.
- Defined required `study.tsv` fields for row identity, baseline provenance,
  project provenance, source/worktree provenance, platform/compiler/config,
  fixture/metric/tolerance/status/caveat/artifact fields, support tier,
  claim scope, and non-claims.
- Defined selected metrics:
  - `project_status`;
  - `baseline_status`;
  - `residual_norm`;
  - `solution_norm`;
  - `solution_values`;
  - `project_vs_baseline_max_abs_delta`.
- Defined status semantics for `pass`, `fail`, `skip`, `defer`, and `error`;
  only `pass` counts as fixture-local proof.
- Defined stale-output policy around source commit, branch, dirty state,
  generated timestamp, exact commands, selected row presence, duplicate rows,
  malformed output, and selected row status.
- Day 6 handoff: design a harness that writes `study.tsv`, `summary.md`, and
  `manifest.tsv`, captures provenance before comparison, parses the baseline
  `OK <value_count>` protocol, emits stable row ids, and fails closed on
  missing or malformed selected rows.

### Day 6: Harness Architecture Design

- Created
  `docs/planning/EPIC_13/SPRINT_154/artifacts/day6-harness-architecture-design.md`.
- Inspected existing implementation inputs:
  - `scripts/run_corpus_oracle.py` temporary static-library probe patterns;
  - `build_sprint150_minnorm_qr_rows()` project-side QR minimum-norm
    observation flow;
  - `tests/qr_external_dense_reference.py` baseline protocol;
  - `tests/corpus/expected/qr_underdetermined_minnorm_2x4.tsv` expected rows.
- Designed a new harness entry point:
  `python3 scripts/run_external_comparison.py --target qr-minnorm`.
- Designed generated output paths under
  `build/comparison/qr_minnorm/`:
  - `study.tsv`;
  - `manifest.tsv`;
  - `summary.md`.
- Designed command flow:
  resolve root/output, collect provenance, reset output directory, run and
  parse baseline, build/run project-side probe, compare metrics, emit
  artifacts, and fail closed on required dependency or selected metric errors.
- Designed project-side runner output:
  `status`, `residual_norm`, `solution_norm`, and `solution_values` for the
  selected fixture.
- Designed baseline parser:
  require `OK 6`, parse four solution values, residual norm, and solution
  norm, and treat malformed/non-zero baseline output as `error`.
- Defined stable comparison row ids and stable failure classes for unsupported
  target, missing baseline helper, malformed output, project build/probe
  failure, tolerance misses, missing/duplicate selected rows, and unsupported
  claim boundary wording.
- Day 7 handoff: implement `scripts/run_external_comparison.py` with CLI,
  provenance capture, selected fixture constants, output paths, and
  project-side probe scaffold before adding full baseline comparison logic on
  Day 8.

### Day 7: Harness Implementation Batch 1

- Created
  `docs/planning/EPIC_13/SPRINT_154/artifacts/day7-harness-project-runner-scaffold.md`.
- Added `scripts/run_external_comparison.py`.
- Implemented CLI parsing for:
  - `--target qr-minnorm`;
  - `--root`;
  - `--output-dir`;
  - `--library`;
  - `--keep-temp`.
- Implemented repository, source commit, branch, dirty/clean worktree state,
  project version, platform, compiler, Python, and project command provenance
  capture.
- Implemented selected fixture constants for
  `qr_underdetermined_minnorm_2x4`.
- Implemented output directory scaffolding under
  `build/comparison/qr_minnorm/`.
- Implemented project-side temporary C probe generation and execution for
  `sparse_qr_solve_minnorm(A, b, x, NULL)`.
- Implemented project-side observation parsing for `status`,
  `residual_norm`, `solution_norm`, and `solution_values`.
- Implemented provisional local outputs:
  - `project_observations.tsv`;
  - `manifest.tsv`.
- Recorded `baseline_status=not_yet_integrated` in the Day 7 manifest so the
  scaffold is not interpreted as complete external comparison proof.
- Ran `python3 scripts/run_external_comparison.py --target qr-minnorm`; the
  scaffold generated `build/comparison/qr_minnorm/manifest.tsv` and
  `build/comparison/qr_minnorm/project_observations.tsv`.
- Project-side scaffold validation passed:
  - `project_status=SPARSE_SUCCESS`;
  - `residual_norm=1.5700924586837752e-16`;
  - `solution_norm=0.99999999999999989`;
  - `solution_values=0.49999999999999989,0.49999999999999989,0.5,0.5`.
- Day 8 handoff: add baseline command execution and `OK 6` parser, record
  baseline command/Python provenance, and keep NumPy/SciPy rows deferred.

### Day 8: Harness Implementation Batch 2

- Created
  `docs/planning/EPIC_13/SPRINT_154/artifacts/day8-baseline-runner-implementation.md`.
- Extended `scripts/run_external_comparison.py` with the selected
  source-controlled external-process dense baseline path for
  `qr_underdetermined_minnorm_2x4`.
- Implemented baseline helper discovery for
  `tests/qr_external_dense_reference.py`.
- Implemented baseline execution through the current Python interpreter.
- Implemented strict `OK 6` baseline parsing:
  - four solution values;
  - residual norm;
  - solution norm.
- Added fail-closed baseline failure classes for:
  - missing selected baseline helper;
  - non-zero baseline command;
  - malformed baseline output.
- Added generated baseline observation output:
  `build/comparison/qr_minnorm/baseline_observations.tsv`.
- Added generated dependency diagnostics:
  `build/comparison/qr_minnorm/dependency_status.tsv`.
- Recorded baseline provenance in `manifest.tsv`:
  - baseline name/type/version;
  - command;
  - helper path;
  - Python executable;
  - Python version;
  - baseline and dependency artifact paths.
- Preserved deferred optional-package behavior:
  - `numpy` is `defer`, not proof;
  - `scipy` is `defer`, not proof.
- Ran `python3 scripts/run_external_comparison.py --target qr-minnorm`; the
  runner generated project, baseline, dependency, and manifest artifacts.
- Baseline scaffold validation passed:
  - `baseline_status=success`;
  - `baseline_residual_norm=0`;
  - `baseline_solution_norm=1`;
  - `baseline_solution_values=0.5,0.5,0.5,0.5`.
- Day 9 handoff: add project-vs-baseline comparison rows and tolerance checks
  without claiming broad QR, NumPy, SciPy, package, performance, ABI, hosted
  CI, or state-of-the-art evidence.

### Day 9: Comparison Logic Implementation

- Created
  `docs/planning/EPIC_13/SPRINT_154/artifacts/day9-comparison-logic-implementation.md`.
- Extended `scripts/run_external_comparison.py` with schema-complete
  `study.tsv` emission for the selected QR minimum-norm comparison.
- Added stable selected comparison row ids:
  - `comparison_qr_underdetermined_minnorm_2x4_project_status_v1`;
  - `comparison_qr_underdetermined_minnorm_2x4_baseline_status_v1`;
  - `comparison_qr_underdetermined_minnorm_2x4_residual_norm_v1`;
  - `comparison_qr_underdetermined_minnorm_2x4_solution_norm_v1`;
  - `comparison_qr_underdetermined_minnorm_2x4_solution_values_v1`;
  - `comparison_qr_underdetermined_minnorm_2x4_project_vs_baseline_max_abs_delta_v1`.
- Implemented project-vs-baseline tolerance evaluation for:
  - residual norm;
  - solution norm;
  - solution values;
  - maximum absolute solution delta.
- Implemented selected-row validation for:
  - missing selected rows;
  - duplicate selected rows;
  - non-pass selected rows.
- Added `build/comparison/qr_minnorm/summary.md` as a human-readable narrow
  study summary scaffold.
- Added manifest references for:
  - `study_path`;
  - `summary_path`.
- Added `python3 scripts/run_external_comparison.py --self-check` smoke
  coverage for:
  - successful selected-row validation;
  - `missing_selected_row`;
  - `duplicate_selected_row`;
  - `metric_tolerance_miss`;
  - `metric_comparison_malformed`;
  - NumPy/SciPy `defer` status semantics.
- Ran `python3 scripts/run_external_comparison.py --self-check`; self-check
  passed.
- Ran `python3 scripts/run_external_comparison.py --target qr-minnorm`; the
  harness generated `project_observations.tsv`, `baseline_observations.tsv`,
  `dependency_status.tsv`, `study.tsv`, `summary.md`, and `manifest.tsv`.
- Day 9 selected comparison validation passed:
  - `project_status=pass`;
  - `baseline_status=pass`;
  - residual delta `1.5700924586837752e-16 <= 1e-10`;
  - solution norm delta `1.1102230246251565e-16 <= 1e-10`;
  - solution value delta `1.1102230246251565e-16 <= 1e-10`;
  - max absolute solution delta `1.1102230246251565e-16 <= 1e-10`.
- Day 10 handoff: decide whether comparison output remains artifact-only or
  receives report-index integration, and keep local-only freshness semantics
  explicit.

### Day 10: Report Integration Design

- Created
  `docs/planning/EPIC_13/SPRINT_154/artifacts/day10-report-integration-design.md`.
- Reviewed the Day 9 generated comparison outputs:
  - `build/comparison/qr_minnorm/project_observations.tsv`;
  - `build/comparison/qr_minnorm/baseline_observations.tsv`;
  - `build/comparison/qr_minnorm/dependency_status.tsv`;
  - `build/comparison/qr_minnorm/study.tsv`;
  - `build/comparison/qr_minnorm/summary.md`;
  - `build/comparison/qr_minnorm/manifest.tsv`.
- Selected the Day 10 report-integration product decision:
  keep the first comparison study artifact-only for now.
- Decided not to add a source-controlled `comparison` report family on Day 10
  because report-index promotion needs complete policy and implementation for
  selected row count, selected row ids, missing rows, duplicate rows, non-pass
  selected rows, stale source commit, dirty worktree caveats, optional
  dependency `defer`, local-only support tier, and `--require-generated`
  behavior.
- Defined the future comparison report-family contract if Day 11 promotes
  rows:
  - family `comparison`;
  - subfamily `qr_minnorm`;
  - row meaning `external_process_dense_reference_comparison`;
  - generated-local origin;
  - local-only support tier;
  - strict generated-input freshness only when comparison freshness is
    required.
- Defined the six selected comparison row ids that any future report-index
  integration must require exactly once.
- Defined status mapping:
  - `pass` may support fixture-local proof only when every selected row is
    present and pass;
  - `fail` is not proof;
  - `skip` is not proof;
  - `defer` is not proof;
  - `error` is not proof.
- Reaffirmed non-claims for broad QR, NumPy, SciPy, LAPACK, SuiteSparse,
  Eigen, external-library ecosystem, hosted CI, release, platform, package,
  shared-library ABI, performance, and state-of-the-art proof.
- Prepared the Day 11 implementation checklist for either complete
  report-index integration or explicit artifact-only deferral.

### Day 11: Report Integration Implementation

- Created
  `docs/planning/EPIC_13/SPRINT_154/artifacts/day11-report-integration-implementation.md`.
- Promoted the selected comparison output into the normalized report-index
  model as a narrow generated-local family:
  - `report_family=comparison`;
  - `subfamily=qr_minnorm`;
  - `row_meaning=external_process_dense_reference_comparison`;
  - `support_tier=local_only`;
  - `freshness_policy=generated_compare_inputs`;
  - `artifact_pattern=build/comparison/qr_minnorm/study.tsv`.
- Added the comparison report-family contract row to
  `tests/corpus/manifests/report_families.tsv`.
- Updated `scripts/validate_corpus_schema.py` so the maintained report-family
  manifest accepts `external_process_dense_reference_comparison`.
- Updated `scripts/normalize_report_index.py` with a comparison generated-row
  loader for `build/comparison/qr_minnorm/study.tsv`.
- Preserved comparison-row provenance in normalized rows:
  - source commit;
  - branch;
  - generated timestamp;
  - platform;
  - compiler;
  - configuration;
  - artifact path;
  - support tier;
  - claim scope;
  - non-claims.
- Added selected comparison freshness diagnostics for:
  - exact selected-row count;
  - missing selected row ids;
  - duplicate selected row ids;
  - unexpected comparison row ids;
  - non-pass selected rows;
  - `skip` and `defer` rows as visible non-proof states.
- Added `make report-index-comparison-freshness`:
  - regenerates selected local comparison output;
  - checks required comparison freshness;
  - reports local-only generated comparison freshness.
- Validation passed:
  - `python3 scripts/run_external_comparison.py --target qr-minnorm`;
  - `python3 scripts/validate_corpus_schema.py`;
  - `python3 scripts/normalize_report_index.py --family comparison --check`;
  - `python3 scripts/normalize_report_index.py --family comparison --require-generated comparison --check-freshness`;
  - `make report-index-comparison-freshness`;
  - `python3 scripts/run_external_comparison.py --self-check`;
  - `python3 scripts/normalize_report_index.py --family corpus --family oracle --family comparison --check`;
  - `git diff --check`.
- Negative required-generated smoke check passed: using an empty temporary
  build root with `--require-generated comparison --check-freshness` returned
  non-zero and reported `required generated family missing: comparison`.
- Day 12 handoff: align maintainer and public documentation with the new
  comparison freshness command, local-only support tier, six selected-row
  requirement, and non-claim boundaries.

### Day 12: Documentation Alignment

- Created
  `docs/planning/EPIC_13/SPRINT_154/artifacts/day12-documentation-alignment.md`.
- Updated `README.md` to:
  - list `make report-index-comparison-freshness` in the maintained Make
    command block;
  - state that normalized comparison rows do not become release proof;
  - describe the narrow local comparison gate for
    `qr_underdetermined_minnorm_2x4`.
- Updated `docs/maintainer_guide.md` to:
  - add a `Selected Comparison Freshness Gate` section;
  - document generated comparison artifacts under
    `build/comparison/qr_minnorm/`;
  - list the six required selected comparison rows;
  - add `make report-index-comparison-freshness` to common focused checks;
  - update the QR trust-boundary row with the new comparison gate.
- Updated `benchmarks/README.md` to:
  - include comparison output in the report-index handoff table;
  - add `comparison` to the normalized report-index example;
  - state that comparison rows remain fixture-local correctness evidence, not
    broad external-library parity.
- Updated `docs/solver_selection.md` to mention the new QR minimum-norm
  comparison gate while preserving the no-broad-parity boundary.
- Ran a focused stale-wording search for state-of-the-art, broad parity,
  ecosystem parity, optional package parity, package-manager,
  shared-library, performance-superiority, and hosted-CI-proof wording.
- Search result: active public and maintainer docs only contained non-claims
  or scoped boundaries; no active wording claimed broad ecosystem or
  state-of-the-art parity.
- Day 13 handoff: run the selected comparison freshness gate, inspect
  generated `study.tsv` and `summary.md`, and publish the first narrow study
  artifact or study-summary documentation with the same local-only boundaries.

### Day 13: Integrated Validation And Study Publication

- Created
  `docs/planning/EPIC_13/SPRINT_154/artifacts/first-narrow-qr-minnorm-comparison-study.md`.
- Created
  `docs/planning/EPIC_13/SPRINT_154/artifacts/day13-integrated-validation-and-study-publication.md`.
- Ran `make report-index-comparison-freshness`; the selected comparison
  output regenerated and required comparison freshness passed.
- Published the first narrow source-controlled study snapshot for
  `qr_underdetermined_minnorm_2x4`.
- Recorded the selected project-vs-baseline rows:
  - `project_status=pass`;
  - `baseline_status=pass`;
  - residual delta `1.5700924586837752e-16 <= 1e-10`;
  - solution norm delta `1.1102230246251565e-16 <= 1e-10`;
  - solution value delta `1.1102230246251565e-16 <= 1e-10`;
  - max absolute solution delta `1.1102230246251565e-16 <= 1e-10`.
- Recorded dependency status:
  - `python3=pass`;
  - `tests/qr_external_dense_reference.py=pass`;
  - `numpy=defer`;
  - `scipy=defer`.
- Confirmed `numpy` and `scipy` defers remain visible non-proof states.
- Recorded residual comparative gaps:
  - QR comparison beyond `qr_underdetermined_minnorm_2x4`;
  - optional NumPy and SciPy package baselines;
  - LAPACK, SuiteSparse, Eigen, PETSc, Trilinos, and other ecosystem baselines;
  - QR raw Q/R basis, sign/orientation/order, pivot-order, and rank-threshold
    comparison;
  - broad rank-deficient, nullspace, economy-mode, sparse-mode, and reorder
    comparison;
  - partial-SVD publication under the normalized `comparison` family;
  - portable runtime or performance comparison;
  - hosted CI comparison publication;
  - package-manager, shared-library, loader, and ABI comparison lanes.
- Quality-gate decision: Day 13 changed documentation only, so no `.c` or
  public `.h` changes required the full `make format && make lint && make test`
  gate. Focused comparison/report/schema/whitespace checks remain the selected
  gate.
- Day 14 handoff: rerun focused validation, confirm non-claims, record final
  residuals, and close the sprint with Sprint 155 handoff inputs.

### Day 14: Closeout And Sprint 155 Handoff

- Created
  `docs/planning/EPIC_13/SPRINT_154/artifacts/day14-closeout-sprint155-handoff.md`.
- Finalized the Sprint 154 artifact index:
  - Day 1 comparison boundary;
  - Day 2 target candidate audit;
  - Day 3 target selection;
  - Day 4 dependency pinning policy;
  - Day 5 output schema design;
  - Day 6 harness architecture;
  - Day 7 project runner scaffold;
  - Day 8 baseline runner implementation;
  - Day 9 comparison logic implementation;
  - Day 10 report integration design;
  - Day 11 report integration implementation;
  - Day 12 documentation alignment;
  - Day 13 first narrow study publication;
  - Day 14 closeout and Sprint 155 handoff.
- Confirmed the maintained comparison command:
  `make report-index-comparison-freshness`.
- Confirmed the normalized report family:
  `comparison/qr_minnorm`.
- Confirmed required comparison freshness expects one contract row and six
  generated selected rows for `qr_underdetermined_minnorm_2x4`.
- Confirmed all selected generated rows must be present exactly once and pass
  before the fixture-local comparison statement is supported.
- Reconfirmed optional NumPy and SciPy package baselines remain `defer`, not
  proof.
- Reconfirmed Sprint 154 non-claims:
  broad QR, NumPy, SciPy, LAPACK, SuiteSparse, Eigen, ecosystem parity, hosted
  CI, release, platform, package-manager, shared-library ABI, performance, and
  state-of-the-art proof remain out of scope.
- Recorded Sprint 155 handoff for tutorial/API-reference work:
  - audit `docs/tutorial.md` for report-index and comparison wording;
  - keep `make report-index-comparison-freshness` in maintainer or
    advanced-report contexts;
  - preserve fixture-local scope around `sparse_qr_solve_minnorm`;
  - do not imply external-library, platform, package, ABI, or performance
    proof from comparison rows;
  - run declaration-preservation checks and full quality gates if public
    headers change.
- Final Day 14 validation passed:
  - `make report-index-comparison-freshness`;
  - `python3 scripts/run_external_comparison.py --self-check`;
  - `python3 scripts/validate_corpus_schema.py`;
  - `python3 scripts/normalize_report_index.py --family corpus --family oracle --family comparison --check`;
  - `python3 scripts/normalize_report_index.py --family comparison --require-generated comparison --check-freshness`;
  - focused stale-wording scan;
  - `git diff --check`.
- Quality-gate decision: no `.c` or public `.h` files changed on Day 14, so
  the full `make format && make lint && make test` gate was not required for
  the Day 14 closeout.
