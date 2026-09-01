# Sprint 191 Working Notes: Bounded External Comparison Family

## Sprint Goal

Add one bounded external comparison family that improves numerical credibility
without claiming broad ecosystem parity.

## Day 1: Comparison Family Intake

### Scope Trace

| Epic item | Day 1 intake interpretation |
| --- | --- |
| 191.1 Family Selection | Build a candidate ledger for exactly one additional comparison family, including fixture, operation, reference path, metrics, tolerances, dependency policy, and claim boundary. |
| 191.2 Fixture And Reference | Identify current fixture/reference patterns and decide what a new deterministic fixture must own before implementation starts. |
| 191.3 Runner Integration | Inventory `scripts/run_external_comparison.py`, selected manifest fields, generated files, study rows, summary output, and manifest output that a new family must extend. |
| 191.4 Focused Tests | Identify existing parser, dependency, tolerance, pass/fail, stale-output, manifest, workflow, and normalizer tests that should be copied or extended. |
| 191.5 Docs And Non-Claims | Identify public and maintainer documentation that must describe only the selected fixture evidence and retain broad-parity non-claims. |
| 191.6 Validation | Define the minimum Sprint 191 validation set and the mandatory full C gate if any `.c` or `.h` files change. |

### Baseline Evidence Read

| Source | Day 1 finding |
| --- | --- |
| `docs/planning/EPIC_17/PROJECT_PLAN.md` | Sprint 191 is allocated 168 hours to add one bounded external comparison family with source-controlled fixtures, reports, manifest rows, tests, and claim-safe documentation. |
| `docs/planning/EPIC_17/SPRINT_187/artifacts/day9-comparison-performance-gates.md` | Sprint 191 acceptance requires exactly one selected family, deterministic fixture ownership, explicit reference behavior, stable metrics/tolerances, runner outputs, selected manifest metadata, freshness validation, exact workflow artifact scope, and calibrated docs. |
| `docs/planning/EPIC_17/SPRINT_190/WORKING_NOTES.md` | Sprint 190 hardened selected comparison workflow/freshness mechanics and promoted one Windows-safe Cholesky selected-comparison lane, but did not add a new comparison family. |
| `tests/corpus/manifests/selected_report_targets.tsv` | Existing selected comparison rows cover QR minimum-norm, QR compatible least-squares, partial-SVD diagonal top-k, LU nonsymmetric square solve, and Cholesky SPD tridiagonal solve. |
| `scripts/run_external_comparison.py` | Current comparison target table and output writer provide reusable patterns for project observations, baseline observations, dependency status, study rows, summary, manifest, row IDs, tolerances, and non-claims. |
| `scripts/normalize_report_index.py` | Selected comparison freshness already validates expected row IDs, row counts, stale source commits, non-pass rows, skip/defer rows, missing artifacts, selected-target filtering, and cross-platform artifact path matching. |
| `tests/test_run_external_comparison.py` | Runner unit coverage already checks selected target generation, malformed inputs, dependency behavior, CMake probe path formatting, and expected row IDs for existing families. |
| `tests/test_normalize_report_index.py` | Normalizer tests already cover complete selected comparison row sets, target-specific freshness, row-set mismatch, duplicate rows, stale rows, failed rows, skipped/deferred rows, support tiers, and Windows-style artifact paths. |
| `tests/test_selected_comparison_workflow.py` | Linux and macOS selected comparison workflow upload scopes are guarded, and the Windows lane is allowlisted only for the Sprint 190 Cholesky selected target. |

### Current Comparison Infrastructure Inventory

| Surface | Current Day 1 state |
| --- | --- |
| Runner command | `make report-index-comparison-freshness` regenerates all selected local comparison outputs, then runs `python3 scripts/normalize_report_index.py --family comparison --require-generated comparison --check-freshness`. |
| Individual generator | `python3 scripts/run_external_comparison.py --target <target_key>` generates one selected comparison target. |
| Existing target keys | `qr-minnorm`, `qr-compatible-ls`, `partial-svd-diag6-k2`, `lu-nonsym-square-5`, and `cholesky-spd-tridiag-5`. |
| Generated files | Each selected target writes `project_observations.tsv`, `baseline_observations.tsv`, `dependency_status.tsv`, `study.tsv`, `summary.md`, and `manifest.tsv`. |
| Common rows | QR, LU, and Cholesky solve-style targets generate 6 study rows: project status, baseline status, residual norm, solution norm, solution values, and project-vs-baseline max absolute delta. |
| Partial-SVD rows | `partial-svd-diag6-k2` generates 10 rows covering project/baseline status, two singular values, singular-value delta, residual norm, U/V orthogonality, and U/V projector diagonals. |
| Manifest authority | `tests/corpus/manifests/selected_report_targets.tsv` defines target ID, family, subfamily, target key, generator command, required files, expected rows, expected row IDs, workflow metadata, claim scope, non-claims, owner, and provenance. |
| Freshness authority | `scripts/normalize_report_index.py` enforces selected comparison row identity, artifact matching, stale source-commit diagnostics, selected-target filtering, and remediation commands. |
| Linux hosted path | `.github/workflows/ci.yml` runs the full selected comparison freshness gate and uploads `sprint175-linux-selected-comparison-freshness`. |
| macOS hosted path | `.github/workflows/macos-ci.yml` runs the full selected comparison freshness gate and uploads `sprint175-macos-selected-comparison-freshness`. |
| Windows hosted path | `.github/workflows/windows-ci.yml` runs only the Sprint 190 Cholesky selected comparison target with CMake/MSVC probe mode and uploads `sprint190-windows-selected-comparison-cholesky`. |
| Documentation surfaces | `README.md`, `INSTALL.md`, `docs/maintainer_guide.md`, and `tests/corpus/README.md` describe selected comparison evidence and retained non-claims. |

### Existing Selected Comparison Families

| Target key | Solver family | Fixture | Metrics | Current claim boundary |
| --- | --- | --- | --- | --- |
| `qr-minnorm` | QR | `qr_underdetermined_minnorm_2x4` | solve status, residual, solution norm, solution values, max absolute delta | Fixture-local QR minimum-norm comparison against a source-controlled dense reference helper. |
| `qr-compatible-ls` | QR | `qr_overdetermined_compatible_5x3` | solve status, residual, solution norm, solution values, max absolute delta | Fixture-local QR compatible least-squares comparison against a source-controlled dense reference helper. |
| `partial-svd-diag6-k2` | Partial SVD | `partial_svd_diag6_k2` | status, singular values, singular-value delta, residual, orthogonality, projector diagonals | Fixture-local diagonal top-k partial-SVD comparison against a source-controlled dense singular-value reference helper. |
| `lu-nonsym-square-5` | Linked-list LU | `lu_nonsym_square_5` | solve status, residual, solution norm, solution values, max absolute delta | Fixture-local nonsymmetric square-solve comparison against a source-controlled dense reference helper. |
| `cholesky-spd-tridiag-5` | Cholesky CSC SPD | `cholesky_spd_tridiag_5` | solve status, residual, solution norm, solution values, max absolute delta | Fixture-local Cholesky SPD tridiagonal solve comparison against a source-controlled dense Cholesky reference helper. |

### Candidate Family Ledger

| Candidate | Candidate value | Likely owner surfaces | Initial risk | Day 1 disposition |
| --- | --- | --- | --- | --- |
| QR rank-threshold solve comparison | Adds evidence for numerical-rank handling beyond exact minimum-norm and compatible least-squares fixtures. | `scripts/run_external_comparison.py`, `tests/qr_external_dense_reference.py`, selected manifest, normalizer tests, solver docs. | Rank-threshold wording can overclaim broad rank-deficient behavior; fixture and tolerance policy must be exact. | Keep as a strong candidate for Day 2 scoring. |
| QR incompatible least-squares comparison | Adds residual-only least-squares evidence for inconsistent rectangular systems. | Runner target table, QR dense reference helper, selected manifest, docs. | Easy to confuse with broad least-squares parity; must separate residual/minimizer claim from exact-solution claim. | Keep as a strong candidate because fixture pattern likely reuses existing QR path. |
| Partial-SVD nonsymmetric rectangular top-k comparison | Adds evidence outside diagonal-only SVD fixtures. | Runner target table, SVD dense reference helper, partial-SVD metric rows, normalizer expected row IDs. | Larger metric surface, vector orientation/sign caveats, and potentially more review overhead. | Keep as a higher-value but higher-risk candidate. |
| Cholesky non-tridiagonal SPD comparison | Adds SPD coverage beyond the current tridiagonal fixture. | Runner target table, Cholesky dense reference helper, selected manifest, docs. | Incremental value may be smaller than adding a new family; broad SPD/reordering wording risk remains. | Keep as fallback if Day 2 rejects QR/SVD candidates. |
| LU singular expected-failure comparison | Adds failure-mode evidence for singular systems. | Runner target table, LU dense reference helper, dependency/status rows, docs. | Failure rows may not fit selected comparison pass/freshness assumptions unless carefully modeled. | Keep as exploratory candidate; likely needs special status semantics. |
| Sparse matrix-vector product comparison | Adds a simple operation family with low dependency and fixture risk. | New runner path, possible new dense helper, selected manifest, docs. | May not be viewed as solver-family evidence; public API surface and target naming need review. | Keep as a low-risk candidate if solver-family candidates prove too broad. |

### Selection Criteria

A Sprint 191 candidate is acceptable only if it has:

- exactly one stable `target_key`, one fixture, and one operation;
- deterministic source-controlled fixture material with no external downloads;
- a source-controlled reference helper or explicit optional-dependency
  unavailable behavior;
- stable metric rows with row IDs, row kinds, tolerance kinds, tolerance
  values, and claim meanings;
- generated `project_observations.tsv`, `baseline_observations.tsv`,
  `dependency_status.tsv`, `study.tsv`, `summary.md`, and `manifest.tsv`;
- selected manifest metadata with exact required files, expected row count,
  expected row IDs, workflow file/job/artifact/platforms, claim scope,
  non-claims, owner, and provenance;
- freshness diagnostics that fail stale, missing, duplicated, unexpected,
  failed, skipped, or deferred selected rows;
- workflow upload scope that names only the selected target artifacts;
- documentation that states the selected fixture evidence and retained
  non-claims together.

### Rejection Criteria

Reject or defer a candidate if it requires:

- broad SuiteSparse, Eigen, SciPy, LAPACK, PETSc, Trilinos, or ecosystem
  parity wording;
- external downloads or unowned dependency state for the fixture or reference;
- multiple partially implemented families in one sprint;
- performance superiority or state-of-the-art claims;
- platform support expansion beyond the evidence already owned by the selected
  workflows;
- ambiguous pass evidence when an optional dependency is missing;
- generated rows that cannot be represented by current freshness and manifest
  contracts.

### Owner Surfaces

| Surface | Sprint 191 role |
| --- | --- |
| `scripts/run_external_comparison.py` | Primary runner, fixture, metric, reference, generated file, and target-table owner. |
| `tests/corpus/manifests/selected_report_targets.tsv` | Selected comparison metadata authority for the new target row. |
| `scripts/normalize_report_index.py` | Freshness validation, row identity, stale-output, target-filtering, and remediation owner. |
| `Makefile` | Full selected comparison freshness command owner. |
| `.github/workflows/ci.yml` | Linux hosted selected comparison freshness and artifact upload owner. |
| `.github/workflows/macos-ci.yml` | macOS hosted selected comparison freshness and artifact upload owner. |
| `.github/workflows/windows-ci.yml` | Windows selected comparison non-expansion guard; only Sprint 190 Cholesky is currently promoted. |
| `tests/test_run_external_comparison.py` | Runner target, parser, dependency, tolerance, and output-shape regression owner. |
| `tests/test_normalize_report_index.py` | Freshness, expected-row, stale-row, duplicate-row, selected-target, and support-tier regression owner. |
| `tests/test_selected_report_targets_manifest.py` | Manifest schema and selected target metadata regression owner. |
| `tests/test_selected_comparison_workflow.py` | Hosted workflow command and exact artifact upload-scope regression owner. |
| `tests/corpus/README.md` | Report/corpus evidence interpretation and selected comparison command guidance. |
| `docs/maintainer_guide.md` | Maintainer-facing solver evidence and claim-boundary guidance. |
| `README.md` and `INSTALL.md` | Public support/evidence claim surfaces if Sprint 191 changes user-visible comparison wording. |

### Initial Risks

| Risk | Why it matters | Day 2 question |
| --- | --- | --- |
| Candidate value is incremental rather than family-expanding | Sprint 191 should improve credibility, not merely add a near-duplicate fixture. | Which candidate most improves evidence while still fitting one bounded family? |
| QR rank and least-squares wording is easy to overbroaden | Existing docs already carry many QR caveats, and new wording could imply broad rank policy. | Can the candidate claim be written as one fixture/metric set without broad QR parity? |
| Partial-SVD vector behavior has sign/orientation ambiguity | More complex metrics can create fragile tests or misleading row meanings. | Can Day 2 define metrics that avoid raw vector identity claims? |
| Missing optional dependency could look like pass evidence | Sprint 187 requires unavailable dependency behavior to be explicit. | Is the reference path source-controlled, or does it need dependency-status rows for absence? |
| Workflow upload scope may broaden accidentally | Existing Linux/macOS jobs upload selected comparison artifacts and reject broad comparison uploads. | What exact artifact paths must be added for only one new target? |
| Windows lane should remain bounded | Sprint 190 promoted only Cholesky on Windows, not all selected comparisons. | Should Sprint 191 leave Windows metadata untouched unless the selected family has hosted proof? |
| C/header edits may trigger full quality gate | Some candidates may need solver or test-code changes rather than runner-only work. | Can the selected comparison family be implemented without changing public C surfaces? |

### Day 1 Validation

Source and planning checks:

```sh
git status --short --branch --ahead-behind
sed -n '1,95p' docs/planning/EPIC_17/SPRINT_191/PLAN.md
sed -n '1,170p' docs/planning/EPIC_17/SPRINT_187/artifacts/day9-comparison-performance-gates.md
column -t -s $'\t' tests/corpus/manifests/selected_report_targets.tsv
rg -n "report-index-comparison-freshness|run_external_comparison|selected-comparison-freshness|sprint175|sprint190" Makefile .github/workflows/ci.yml .github/workflows/macos-ci.yml .github/workflows/windows-ci.yml
git diff --check
```

No `.c` or `.h` files were changed on Day 1, so `make format && make lint &&
make test` is not required.

### Day 2 Questions

1. Which candidate has the highest evidence value while still fitting one
   fixture, one operation, one target key, and one selected manifest row?
2. Can the reference path remain source-controlled, or does the selected family
   require optional dependency semantics and dependency-status rows?
3. What exact metrics and tolerances make the selected family meaningful
   without implying broad external-library parity?
4. Which workflow artifact files must be added to Linux/macOS without widening
   upload scope to `build/comparison/**`?
5. Should Windows selected comparison metadata remain unchanged except for the
   existing Sprint 190 Cholesky lane?

## Day 2: Candidate Family Audit

### Selection Summary

Day 2 selects `qr-incompatible-ls` as the Sprint 191 bounded external
comparison family.

| Field | Decision |
| --- | --- |
| Target key | `qr-incompatible-ls` |
| Fixture key | `qr_overdetermined_incompatible_4x2` |
| Solver family | QR |
| Subfamily | `qr_incompatible_ls` |
| Operation | `least_squares_solve` |
| Reference path | Source-controlled dense QR reference helper in `tests/qr_external_dense_reference.py`. |
| Dependency policy | Required source-controlled Python helper; no external package dependency. |
| Metric shape | Six solve-style selected comparison rows: project status, baseline status, residual norm, solution norm, solution values, and project-vs-baseline max absolute delta. |
| Claim scope | Fixture-local QR incompatible least-squares comparison against the selected source-controlled dense reference helper. |
| Day 3 focus | Confirm exact fixture entries, expected residual semantics, row IDs, tolerances, and manifest wording. |

The selected candidate adds inconsistent least-squares evidence that is not
covered by the existing QR minimum-norm and compatible least-squares selected
comparison rows. It also has the lowest implementation risk because the
fixture and source-controlled dense reference helper already exist and are
already used by `tests/test_qr_solve.c`.

### Scored Candidate Ranking

Scores use `1` for weak/unacceptable and `5` for strong/low-risk.

| Rank | Candidate | Evidence value | Reference availability | Determinism | CI/runtime cost | Claim safety | Implementation fit | Total | Day 2 disposition |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | QR incompatible least-squares comparison | 5 | 5 | 5 | 5 | 4 | 5 | 29 | Selected. |
| 2 | QR rank-threshold solve comparison | 4 | 5 | 5 | 5 | 3 | 4 | 26 | Defer; useful but easier to overread as a global rank policy. |
| 3 | Cholesky non-tridiagonal SPD comparison | 3 | 4 | 5 | 5 | 4 | 4 | 25 | Reject for Sprint 191; incremental after Cholesky Sprint 183/190 work. |
| 4 | Sparse matrix-vector product comparison | 2 | 5 | 5 | 5 | 5 | 3 | 25 | Reject for Sprint 191; too low-value as solver-family evidence. |
| 5 | LU singular expected-failure comparison | 4 | 4 | 5 | 5 | 3 | 3 | 24 | Defer; expected-failure row semantics need separate design. |
| 6 | Partial-SVD nonsymmetric rectangular top-k comparison | 5 | 4 | 4 | 4 | 3 | 3 | 23 | Defer; high value but larger metric and review surface. |

### Candidate Audit Findings

| Candidate | Audit finding | Rationale |
| --- | --- | --- |
| QR incompatible least-squares comparison | Selected. | Existing helper `qr_overdetermined_incompatible_4x2` defines a deterministic inconsistent 4-by-2 system with exact solution `[2.0, -1.0]` and known nonzero residual. Existing C tests already compare project output to the dense helper. |
| QR rank-threshold solve comparison | Deferred. | Existing rank-threshold helper coverage is strong, but selected comparison wording could imply a global rank-threshold policy. It is a better follow-up after this sprint keeps QR comparison claims narrow. |
| Partial-SVD nonsymmetric rectangular top-k comparison | Deferred. | Stronger numerical evidence than diagonal-only SVD, but it brings sign/orientation, orthogonality, projector, and nonsymmetric wording complexity. |
| Cholesky non-tridiagonal SPD comparison | Rejected for Sprint 191. | Technically feasible, but too incremental after selected Cholesky comparison and Windows Cholesky freshness work. |
| LU singular expected-failure comparison | Deferred. | Failure-mode evidence is valuable, but selected comparison pass/freshness semantics need careful expected-failure design. |
| Sparse matrix-vector product comparison | Rejected for Sprint 191. | Low implementation risk, but it is not the strongest solver-family credibility gain for this sprint. |

### Selected Family Boundaries

The Sprint 191 implementation should stay within these boundaries:

- one target key: `qr-incompatible-ls`;
- one fixture key: `qr_overdetermined_incompatible_4x2`;
- one operation: `least_squares_solve`;
- one reference helper: `tests/qr_external_dense_reference.py`;
- no external package dependency;
- no new broad QR, NumPy, SciPy, LAPACK, SuiteSparse, Eigen, platform,
  package, ABI, performance, or state-of-the-art claim;
- Linux and macOS selected comparison workflow metadata may expand only by
  exact target artifact paths if the generated artifacts are added to those
  hosted lanes;
- Windows selected comparison metadata remains unchanged unless a later day
  records hosted evidence for this exact target.

### Day 2 Artifact

The candidate audit artifact is
`docs/planning/EPIC_17/SPRINT_191/artifacts/day2-candidate-family-audit.md`.

### Day 2 Validation

Read-only/source checks:

```sh
git status --short --branch --ahead-behind
sed -n '55,100p' docs/planning/EPIC_17/SPRINT_191/PLAN.md
sed -n '1,260p' docs/planning/EPIC_17/SPRINT_191/WORKING_NOTES.md
sed -n '1,170p' docs/planning/EPIC_17/SPRINT_187/artifacts/day9-comparison-performance-gates.md
sed -n '1,220p' tests/qr_external_dense_reference.py
sed -n '230,380p' tests/test_qr_solve.c
rg -n "qr_overdetermined|rank_threshold|incompatible|partial_svd_nonsym|partial_svd_tall|lu_singular|cholesky" tests scripts include src docs/maintainer_guide.md
git diff --check
```

No `.c` or `.h` files were changed on Day 2, so `make format && make lint &&
make test` is not required.

### Day 3 Questions

1. Should `qr-incompatible-ls` reuse the six solve-style comparison rows
   exactly, or should residual value and residual delta be split into separate
   row meanings?
2. What exact tolerance should apply to the nonzero residual agreement with
   the dense helper?
3. Should `expected_solution_norm` be recorded as `sqrt(5)` and compared with
   the existing solution-norm tolerance path?
4. Which target, subfamily, artifact directory, and row IDs should become the
   canonical names before implementation begins?
5. Should the selected target start as Linux/macOS hosted only, with Windows
   left as the Sprint 190 Cholesky-only lane?

## Day 3: Fixture and Metric Contract

### Fixture Decision

Day 3 confirms that Sprint 191 will implement `qr-incompatible-ls` with the
existing deterministic fixture `qr_overdetermined_incompatible_4x2`.

| Field | Contract |
| --- | --- |
| Target key | `qr-incompatible-ls` |
| Fixture key | `qr_overdetermined_incompatible_4x2` |
| Subfamily | `qr_incompatible_ls` |
| Solver family | QR |
| Operation | `least_squares_solve` |
| Matrix shape | 4 rows by 2 columns |
| Matrix entries | `(0,0)=1`, `(1,1)=1`, `(2,0)=1`, `(2,1)=1`, `(3,0)=2`, `(3,1)=-1` |
| Right-hand side | `[1.0, -2.0, 2.0, 5.0]` |
| Expected solution | `[2.0, -1.0]` |
| Expected solution norm | `2.2360679774997898` (`sqrt(5)`) |
| Expected residual norm | `1.7320508075688772` (`sqrt(3)`) |
| Fixture ownership | Existing handwritten source-controlled fixture in `tests/qr_external_dense_reference.py` and matching C test fixture in `tests/test_qr_solve.c`. |

The fixture is intentionally inconsistent. The nonzero residual is part of
the evidence, so Sprint 191 must compare residual agreement with the dense
helper instead of treating the residual as a near-zero solve residual.

### Reference Output

Day 3 verified the selected dense helper output:

```sh
python3 tests/qr_external_dense_reference.py qr_overdetermined_incompatible_4x2
```

```text
OK 3
1.9999999999999998
-1
1.7320508075688772
```

The helper returns solution values followed by residual norm. The runner can
derive solution norm from those solution values.

### Expected Study Rows

The new target should emit exactly six study rows:

```text
comparison_qr_overdetermined_incompatible_4x2_project_status_v1
comparison_qr_overdetermined_incompatible_4x2_baseline_status_v1
comparison_qr_overdetermined_incompatible_4x2_residual_norm_v1
comparison_qr_overdetermined_incompatible_4x2_solution_norm_v1
comparison_qr_overdetermined_incompatible_4x2_solution_values_v1
comparison_qr_overdetermined_incompatible_4x2_project_vs_baseline_max_abs_delta_v1
```

### Metric and Tolerance Contract

| Metric | Expected value | Comparison basis | Tolerance kind | Tolerance value | Pass meaning |
| --- | --- | --- | --- | ---: | --- |
| `project_status` | `SPARSE_SUCCESS` | Project probe status | `status_only` | n/a | Project QR solve completes for the fixture. |
| `baseline_status` | `success` | Dense helper status | `status_only` | n/a | Source-controlled dense reference completes. |
| `residual_norm` | `1.7320508075688772` | `abs(project_residual - baseline_residual)` | `absolute` | `1e-10` | Project and baseline agree on the expected nonzero residual. |
| `solution_norm` | `2.2360679774997898` | `abs(project_solution_norm - baseline_solution_norm)` | `absolute` | `1e-10` | Project and baseline agree on solution norm. |
| `solution_values` | `2,-1` | Max absolute per-component solution delta | `absolute_per_component` | `1e-10` | Project and baseline agree on solution values. |
| `project_vs_baseline_max_abs_delta` | `<=1e-10` | Max absolute per-component solution delta | `absolute` | `1e-10` | Project and baseline solution vectors agree within tolerance. |

### Runner Contract Implication

`comparison_study_rows()` already compares project residual against baseline
residual, which is the right study-row behavior for the incompatible fixture.
The implementation still needs a small target-level extension for
`project_observation_rows()` and `baseline_observation_rows()`: their current
solve-style residual observation rows pass only when
`residual_norm <= residual_tolerance`, which would incorrectly fail the
expected `sqrt(3)` residual. A target field such as
`expected_residual_norm` should make observation rows pass when the observed
residual matches the expected nonzero residual within `1e-10`.

### Generated Artifact Contract

The new target should generate:

```text
build/comparison/qr_incompatible_ls/project_observations.tsv
build/comparison/qr_incompatible_ls/baseline_observations.tsv
build/comparison/qr_incompatible_ls/dependency_status.tsv
build/comparison/qr_incompatible_ls/study.tsv
build/comparison/qr_incompatible_ls/summary.md
build/comparison/qr_incompatible_ls/manifest.tsv
```

The selected target manifest should use
`artifact_pattern=build/comparison/qr_incompatible_ls/study.tsv`,
`expected_rows=6`, and `freshness_policy=generated_compare_inputs`.

### Claim Boundary

Allowed wording:

> Selected QR incompatible least-squares comparison rows are fresh for
> `qr_overdetermined_incompatible_4x2` against the selected source-controlled
> dense QR reference helper.

Required non-claims:

- no broad QR parity;
- no broad least-squares parity;
- no global rank-threshold policy;
- no broad rank-deficient solve behavior;
- no NumPy, SciPy, LAPACK, SuiteSparse, Eigen, or external-library ecosystem
  parity;
- no Windows report freshness expansion;
- no package-manager proof;
- no shared-library ABI proof;
- no performance superiority;
- no release proof;
- no state-of-the-art claim.

### Day 3 Artifact

The fixture and metric contract artifact is
`docs/planning/EPIC_17/SPRINT_191/artifacts/day3-fixture-metric-contract.md`.

### Day 3 Validation

Read-only/source checks:

```sh
git status --short --branch --ahead-behind
sed -n '95,135p' docs/planning/EPIC_17/SPRINT_191/PLAN.md
python3 tests/qr_external_dense_reference.py qr_overdetermined_incompatible_4x2
sed -n '1120,1475p' scripts/run_external_comparison.py
sed -n '1620,1775p' scripts/run_external_comparison.py
sed -n '1935,1975p' scripts/run_external_comparison.py
git diff --check
```

No `.c` or `.h` files were changed on Day 3, so `make format && make lint &&
make test` is not required.

### Day 4 Questions

1. Should the source-controlled dense QR helper be documented as the only
   required reference dependency, with NumPy/SciPy remaining deferred optional
   rows as in existing QR targets?
2. Should missing-helper and malformed-helper behavior reuse current
   `missing_baseline_helper`, `baseline_command_failed`, and
   `baseline_malformed_output` failure classes?
3. Should `expected_residual_norm` be required only for intentionally
   incompatible least-squares targets, leaving compatible solve targets on
   near-zero residual checks?
4. Should Day 4 explicitly decide not to add Windows workflow metadata for
   `qr-incompatible-ls` unless hosted Windows proof is added later?

## Day 4: Reference Dependency Policy

### Dependency Decision

Day 4 confirms that `qr-incompatible-ls` will reuse the existing QR selected
comparison dependency policy.

| Dependency | Required | Status policy | Pass evidence? | Decision |
| --- | --- | --- | --- | --- |
| Current Python executable | Yes | `pass` with `selected_interpreter_available` when the runner executes. | Limited execution-environment evidence only. | Reuse existing `python3` dependency row semantics. |
| `tests/qr_external_dense_reference.py` | Yes | `pass` with `baseline_helper_available`, or `error` with `baseline_helper_missing`. | Yes, as source-controlled dense-reference evidence. | Required selected baseline helper. |
| NumPy | No | `defer` with `optional_package_baseline_not_selected`. | No. | Keep as optional non-proof row. |
| SciPy | No | `defer` with `optional_package_baseline_not_selected`. | No. | Keep as optional non-proof row. |

No external package dependency is selected for Sprint 191. NumPy and SciPy
must remain deferred optional rows and must not be recommended as remediation
for `qr-incompatible-ls`.

### Reference Process Model

The selected baseline command remains:

```sh
python3 tests/qr_external_dense_reference.py qr_overdetermined_incompatible_4x2
```

The runner should invoke the helper with `sys.executable`, matching existing
QR target behavior. The helper contract is `OK 3` followed by two solution
components and the residual norm.

### Failure Semantics

| Condition | Required handling |
| --- | --- |
| Missing dense helper | `missing_baseline_helper`; fail generation and record required helper as unavailable. |
| Helper exits nonzero | `baseline_command_failed`; include captured command output. |
| Helper emits no output, wrong header, wrong count, wrong value count, or non-numeric values | `baseline_malformed_output`; fail before pass evidence is written. |
| Project probe failure | `project_probe_failed`; fail generation and avoid hidden pass evidence. |
| Residual or solution disagreement | Failing study row and `metric_tolerance_miss` during selected row validation. |

### Unsupported Environment Policy

Windows selected comparison metadata remains unchanged for Day 4. The only
current Windows selected comparison lane is the Sprint 190
`cholesky-spd-tridiag-5` lane. `qr-incompatible-ls` may be added to Linux and
macOS selected comparison workflow artifact scopes later in the sprint, but
only with exact file paths and no broad `build/comparison/**` upload.

### Remediation Wording

Preferred target-specific generator command:

```sh
python3 scripts/run_external_comparison.py --target qr-incompatible-ls
```

Preferred target-specific freshness command:

```sh
python3 scripts/normalize_report_index.py --family comparison --require-generated comparison --check-freshness --selected-target qr-incompatible-ls
```

Preferred full selected comparison gate after Makefile integration:

```sh
make report-index-comparison-freshness
```

### Day 4 Artifact

The dependency policy artifact is
`docs/planning/EPIC_17/SPRINT_191/artifacts/day4-reference-dependency-policy.md`.

### Day 4 Validation

Read-only/source checks:

```sh
git status --short --branch --ahead-behind
sed -n '130,170p' docs/planning/EPIC_17/SPRINT_191/PLAN.md
sed -n '1,260p' docs/planning/EPIC_17/SPRINT_191/artifacts/day3-fixture-metric-contract.md
sed -n '1500,1565p' scripts/run_external_comparison.py
sed -n '180,225p' tests/test_run_external_comparison.py
rg -n "dependency_status|numpy|scipy|missing_baseline_helper|baseline_command_failed|baseline_malformed_output|dependency_status_rows|optional" scripts/run_external_comparison.py tests/test_run_external_comparison.py tests/corpus/README.md docs/maintainer_guide.md
git diff --check
```

No `.c` or `.h` files were changed on Day 4, so `make format && make lint &&
make test` is not required.

### Day 5 Questions

1. Should Day 5 add only the runner target descriptor because the fixture
   already exists in the source-controlled dense helper?
2. Should fixture coherence tests assert matrix shape, RHS, expected solution,
   expected solution norm, and expected residual norm before runner generation
   is enabled?
3. Should the target descriptor include an explicit `expected_residual_norm`
   field to support intentionally nonzero residual checks?

## Day 5: Fixture Material Implementation

### Implementation Summary

Day 5 implemented the selected `qr-incompatible-ls` fixture material in the
comparison runner and added focused fixture coherence coverage.

Changed source surfaces:

| Surface | Day 5 change |
| --- | --- |
| `scripts/run_external_comparison.py` | Added `QR_INCOMPATIBLE_LS_ENTRIES`, the `qr-incompatible-ls` target descriptor, and target-level `expected_residual_norm` handling for solve-style observation and study rows. |
| `tests/test_run_external_comparison.py` | Added target expectations for `qr-incompatible-ls` and a fixture contract test for entries, dimensions, RHS, expected solution, expected norms, and nonzero-residual observation status. |

### Implemented Target Descriptor

| Field | Value |
| --- | --- |
| Target key | `qr-incompatible-ls` |
| Fixture key | `qr_overdetermined_incompatible_4x2` |
| Subfamily | `qr_incompatible_ls` |
| Operation | `least_squares_solve` |
| Output directory | `build/comparison/qr_incompatible_ls` |
| RHS | `[1.0, -2.0, 2.0, 5.0]` |
| Expected solution | `[2.0, -1.0]` |
| Expected solution norm | `2.2360679774997898` |
| Expected residual norm | `1.7320508075688772` |
| Baseline value count | `3` |
| Claim scope | Fixture-local QR incompatible least-squares comparison only. |

### Nonzero Residual Handling

The incompatible least-squares fixture has a valid nonzero residual. Day 5
therefore added `expected_residual_norm` as an optional target field:

- existing solve-style targets default to expected residual `0.0`;
- `qr-incompatible-ls` expects residual `1.7320508075688772`;
- project and baseline observation rows pass when observed residuals match the
  expected residual within `residual_tolerance`;
- study rows continue comparing project residual against baseline residual and
  now report the nonzero expected residual value for this target.

### Fixture Coherence Coverage

`test_qr_incompatible_ls_fixture_contract()` now checks:

- descriptor entries, row count, and column count;
- RHS and expected solution;
- expected solution norm and expected residual norm;
- baseline value count;
- project observation pass state for the expected nonzero residual;
- baseline observation pass state for the expected nonzero residual.

The existing selected target generation test now includes `qr-incompatible-ls`.
Report-family metadata checking for this target is intentionally deferred
until the manifest/report-index integration days add source-controlled
metadata rows.

### Generated Output Policy

The Day 5 smoke generation wrote ignored scratch files under
`build/comparison/qr_incompatible_ls/`. No generated output was added to
source control.

### Day 5 Artifact

The fixture material implementation artifact is
`docs/planning/EPIC_17/SPRINT_191/artifacts/day5-fixture-material-implementation.md`.

### Day 5 Validation

Commands run:

```sh
python3 scripts/run_external_comparison.py --target qr-incompatible-ls
python3 tests/test_run_external_comparison.py
git diff --check
git diff --name-only -- '*.c' '*.h'
```

Results:

- `qr-incompatible-ls` generated successfully;
- `tests/test_run_external_comparison.py` passed;
- `git diff --check` passed;
- no `.c` or `.h` files changed, so `make format && make lint && make test`
  is not required for Day 5.

### Day 6 Questions

1. Are existing missing-helper and malformed-output tests enough for the new
   QR target, or should Day 6 add target-specific parser failure coverage?
2. Should dependency rows be asserted directly for `qr-incompatible-ls` now
   that the target can generate successfully?
3. Should Day 6 add a regression test that corrupts the helper output count
   for a solve-style QR target with `baseline_value_count=3`?

## Day 6: Reference Execution

### Implementation Summary

Day 6 hardened the `qr-incompatible-ls` reference execution path with direct
tests for normalized helper observations, dependency rows, and baseline
failure modes. No new external package dependency was added.

Changed source surface:

| Surface | Day 6 change |
| --- | --- |
| `tests/test_run_external_comparison.py` | Added target-specific reference observation, dependency-row, malformed-output, command-failure, and missing-helper tests for `qr-incompatible-ls`. |

The runner implementation from Day 5 already routes QR targets through
`tests/qr_external_dense_reference.py`, so no additional reference adapter was
needed on Day 6.

### Verified Reference Observations

For `qr-incompatible-ls`, `run_baseline_reference()` returns:

| Observation | Verified behavior |
| --- | --- |
| `status` | `success` |
| `solution_values` | `1.9999999999999998,-1` |
| `residual_norm` | `1.7320508075688772` |
| `solution_norm` | within `1e-15` of `2.2360679774997898` |
| `baseline_helper_path` | `tests/qr_external_dense_reference.py` |
| `baseline_command` | includes `qr_overdetermined_incompatible_4x2` |

The solution-norm assertion is numeric rather than exact text because the
helper's first solution component is `1.9999999999999998`; the difference is
well inside the selected `1e-10` tolerance.

### Dependency Rows

Day 6 verifies this dependency policy for `qr-incompatible-ls`:

| Dependency | Required | Expected state |
| --- | --- | --- |
| `python3` | yes | `pass` with `selected_interpreter_available`. |
| `tests/qr_external_dense_reference.py` | yes | `pass` with `baseline_helper_available`, or `error` with `baseline_helper_missing` when absent. |
| `numpy` | no | `defer` with `optional_package_baseline_not_selected`. |
| `scipy` | no | `defer` with `optional_package_baseline_not_selected`. |

### Failure Coverage

Added tests cover:

- malformed helper count causing `baseline_malformed_output`;
- synthetic helper command failure causing `baseline_command_failed`;
- missing helper dependency row causing `baseline_helper_missing`;
- missing helper execution causing `missing_baseline_helper`.

### Day 6 Artifact

The reference execution artifact is
`docs/planning/EPIC_17/SPRINT_191/artifacts/day6-reference-execution.md`.

### Day 6 Validation

Commands run:

```sh
python3 scripts/run_external_comparison.py --target qr-incompatible-ls
python3 tests/test_run_external_comparison.py
git diff --check
git diff --name-only -- '*.c' '*.h'
```

Results:

- `qr-incompatible-ls` generated successfully;
- `tests/test_run_external_comparison.py` passed;
- `git diff --check` passed;
- no `.c` or `.h` files changed, so `make format && make lint && make test`
  is not required for Day 6.

### Day 7 Questions

1. Should project observations be asserted directly from generated
   `project_observations.tsv`, or should the test call
   `project_observation_rows()` with synthetic project output?
2. Should Day 7 add a regression for project residual mismatch against
   `expected_residual_norm`?
3. Are any project-side changes needed, or is the generated C probe already
   sufficient for the selected fixture?

## Day 7: Project Observation

### Implementation Summary

Day 7 confirmed that the generated project probe path already supports the
selected `qr-incompatible-ls` fixture through the generic QR
`least_squares_solve` probe logic. No solver or public-header changes were
needed.

Changed source surface:

| Surface | Day 7 change |
| --- | --- |
| `tests/test_run_external_comparison.py` | Added direct project probe observation coverage plus synthetic residual and solution mismatch tests for `qr-incompatible-ls`. |

### Generated Project Observation Evidence

`python3 scripts/run_external_comparison.py --target qr-incompatible-ls`
generated `build/comparison/qr_incompatible_ls/project_observations.tsv` with:

| Metric | Value | Status | Status reason |
| --- | --- | --- | --- |
| `project_status` | `SPARSE_SUCCESS` | `pass` | `project_status_match` |
| `residual_norm` | `1.7320508075688772` | `pass` | `project_residual_matches_expected` |
| `solution_norm` | `2.2360679774997894` | `pass` | `project_solution_norm_within_tolerance` |
| `solution_values` | `1.9999999999999996,-1.0000000000000002` | `pass` | `project_solution_values_within_tolerance` |

The representation differences in solution values are within the selected
`1e-10` tolerance.

### Project-Side Failure Coverage

Added tests verify:

- the actual generated project probe reports `SPARSE_SUCCESS`;
- project residual matches `expected_residual_norm`;
- project solution norm and solution values match the Day 3 contract;
- a synthetic zero residual fails with
  `project_residual_expected_mismatch`;
- a synthetic bad solution vector fails with
  `project_solution_values_tolerance_miss`.

### Day 7 Artifact

The project observation artifact is
`docs/planning/EPIC_17/SPRINT_191/artifacts/day7-project-observation.md`.

### Day 7 Validation

Commands run:

```sh
python3 scripts/run_external_comparison.py --target qr-incompatible-ls
python3 tests/test_run_external_comparison.py
column -t -s $'\t' build/comparison/qr_incompatible_ls/project_observations.tsv
git diff --check
git diff --name-only -- '*.c' '*.h'
```

Results:

- `qr-incompatible-ls` generated successfully;
- generated project observations passed for status, residual, solution norm,
  and solution values;
- `tests/test_run_external_comparison.py` passed;
- `git diff --check` passed;
- no `.c` or `.h` files changed, so `make format && make lint && make test`
  is not required for Day 7.

### Day 8 Questions

1. Should `qr_incompatible_ls` report-family metadata use a new Sprint 191
   stage token instead of the generic QR comparison stage token?
2. Should the selected manifest row be added before or after Makefile
   integration so normalizer tests can exercise target-specific freshness?
3. Should Linux/macOS hosted upload scopes include exact
   `build/comparison/qr_incompatible_ls/*` files on Day 8, or should workflow
   changes wait for Day 9 freshness integration?

## Day 8: Runner Study Integration

### Integration Summary

Day 8 promoted `qr-incompatible-ls` from a directly runnable target to a full
selected comparison study family.

| Surface | Day 8 update |
| --- | --- |
| `scripts/run_external_comparison.py` | Added a Sprint 191 stage tag for `qr_incompatible_ls` generated-row configuration while retaining the Day 5 target contract and Day 7 project-observation behavior. |
| `Makefile` | Added `python3 scripts/run_external_comparison.py --target qr-incompatible-ls` to `report-index-comparison-freshness`. |
| `tests/corpus/manifests/report_families.tsv` | Added `comparison / qr_incompatible_ls` report-family metadata with bounded local-only claim scope and explicit broad-parity non-claims. |
| `tests/corpus/manifests/selected_report_targets.tsv` | Added `SRT-COMP-QR-INCOMPATIBLE-LS` with six required generated rows and six required generated files. |
| `scripts/normalize_report_index.py` | Added the six selected QR incompatible row IDs and `build/comparison/qr_incompatible_ls/study.tsv` to selected comparison freshness diagnostics. |
| `.github/workflows/ci.yml` | Added the target to the Linux selected comparison summary and exact artifact upload paths. |
| `.github/workflows/macos-ci.yml` | Added the target to the macOS selected comparison summary and exact artifact upload paths. |
| `tests/test_run_external_comparison.py` | Added target metadata, fixture, reference, project-observation, tolerance, parser, command-failure, and dependency regression coverage. |
| `tests/test_normalize_report_index.py` | Added selected-row, artifact, synthetic writer, support-tier, and freshness regression coverage for the new subfamily. |
| `tests/test_selected_comparison_workflow.py` | Existing manifest-driven workflow tests now include the new Linux/macOS selected target automatically. |
| `tests/corpus/README.md` | Updated selected comparison family count and table entry from five to six families. |

### Generated Output Contract

The selected target command is:

```sh
python3 scripts/run_external_comparison.py --target qr-incompatible-ls
```

It writes exactly six generated artifacts under
`build/comparison/qr_incompatible_ls/`:

| Artifact | Purpose |
| --- | --- |
| `project_observations.tsv` | Project probe status, residual, solution norm, and solution values for the selected fixture. |
| `baseline_observations.tsv` | Source-controlled dense QR helper status, residual, solution norm, and solution values. |
| `dependency_status.tsv` | Required helper discovery and executable status. |
| `study.tsv` | Six normalized comparison rows. |
| `summary.md` | Human-readable local study summary. |
| `manifest.tsv` | Generator command, commit, platform, compiler, configuration, output path, and claim metadata. |

The study contributes exactly six selected comparison row IDs:

| Row ID | Meaning |
| --- | --- |
| `comparison_qr_overdetermined_incompatible_4x2_project_status_v1` | Project probe completed with `SPARSE_SUCCESS`. |
| `comparison_qr_overdetermined_incompatible_4x2_baseline_status_v1` | Source-controlled dense QR helper completed. |
| `comparison_qr_overdetermined_incompatible_4x2_residual_norm_v1` | Project residual matches the expected nonzero residual `1.7320508075688772`. |
| `comparison_qr_overdetermined_incompatible_4x2_solution_norm_v1` | Project solution norm matches the expected norm `2.2360679774997898`. |
| `comparison_qr_overdetermined_incompatible_4x2_solution_values_v1` | Project solution values match `[2.0, -1.0]` within tolerance. |
| `comparison_qr_overdetermined_incompatible_4x2_project_vs_baseline_max_abs_delta_v1` | Project and baseline solution vectors agree within tolerance. |

### Freshness Scope

`make report-index-comparison-freshness` now regenerates six selected local
comparison families:

1. `qr-minnorm`
2. `qr-compatible-ls`
3. `qr-incompatible-ls`
4. `partial-svd-diag6-k2`
5. `lu-nonsym-square-5`
6. `cholesky-spd-tridiag-5`

The resulting normalized comparison freshness check passed with 46 rows:
six report-family contract rows, 40 generated study rows, and no stale,
missing, duplicate, skipped, deferred, or failed selected generated rows.

### Claim Boundary

The new family proves only one fixture-local QR incompatible least-squares
comparison against the selected source-controlled dense QR helper. It does not
claim broad QR parity, broad least-squares parity, raw QR basis identity, Q sign
or orientation identity, global rank-threshold behavior, broad rank-deficient
solve behavior, NumPy/SciPy/LAPACK/SuiteSparse/Eigen parity, Windows report
freshness, package-manager proof, shared-library ABI proof, performance
superiority, or state-of-the-art status.

### Day 8 Artifact

The runner study integration artifact is
`docs/planning/EPIC_17/SPRINT_191/artifacts/day8-study-integration.md`.

### Day 8 Validation

Commands run:

```sh
python3 scripts/run_external_comparison.py --self-check
python3 scripts/validate_corpus_schema.py
python3 tests/test_selected_report_targets_manifest.py
python3 tests/test_run_external_comparison.py
python3 tests/test_selected_comparison_workflow.py
python3 tests/test_normalize_report_index.py
python3 scripts/run_external_comparison.py --target qr-incompatible-ls
make report-index-comparison-freshness
```

Results:

- external comparison self-check passed;
- corpus schema validation passed;
- selected target manifest test passed;
- external comparison runner tests passed;
- selected comparison workflow tests passed;
- normalizer tests passed;
- direct `qr-incompatible-ls` generation passed and wrote six artifacts;
- full selected comparison freshness passed with 46 normalized rows.

No `.c` or `.h` files changed on Day 8, so `make format && make lint &&
make test` is not required for this day.

### Day 9 Questions

1. Should Day 9 add a target-specific freshness smoke command for
   `--selected-target qr-incompatible-ls`, or is the full selected comparison
   gate sufficient?
2. Should Windows selected comparison metadata remain unchanged until a future
   sprint explicitly promotes QR incompatible least-squares on MSVC?
3. Should stale, skipped, deferred, and failed-row regression tests be split by
   selected family, or is the manifest-driven all-family row-set test enough?

## Day 9: Report Index and Freshness Integration

### Freshness Integration Summary

Day 9 added target-specific freshness coverage for `qr-incompatible-ls` on top
of the Day 8 full selected-comparison integration.

| Freshness surface | Day 9 behavior |
| --- | --- |
| Missing target artifacts | `--selected-target qr-incompatible-ls` reports the exact `build/comparison/qr_incompatible_ls/study.tsv` artifact diagnostic and includes target-specific remediation. |
| Accepted target subset | A synthetic row set containing only the six `qr_incompatible_ls` selected rows passes target-specific freshness. |
| Cross-platform artifact matching | Windows-style `build\comparison\qr_incompatible_ls\study.tsv` paths still match the selected target and stale rows fail. |
| Dependency-only rows | A `baseline_status` dependency row alone does not satisfy selected freshness; missing metric rows produce a row-set mismatch. |
| Remediation text | Target-specific failures include `--selected-target qr-incompatible-ls` in the remediation command. |
| Full gate | The complete `make report-index-comparison-freshness` gate still passes for all six selected comparison families. |

### Normalizer Test Coverage

Added focused regression coverage in `tests/test_normalize_report_index.py`:

| Test | Covered behavior |
| --- | --- |
| `test_selected_comparison_target_freshness_accepts_qr_incompatible_subset` | Missing-artifact diagnostic, accepted six-row target subset, normalized row identity, artifact path, support tier, and non-claim preservation. |
| `test_qr_incompatible_selected_freshness_rejects_windows_path_stale_rows` | Backslash artifact paths match selected artifacts and stale rows fail with target-specific remediation. |
| `test_qr_incompatible_selected_freshness_rejects_dependency_only_rows` | Dependency-status evidence alone is not accepted as complete selected comparison proof. |

The synthetic selected-comparison writer now has a dedicated
`qr_incompatible_ls` bucket so normalizer tests write the new subfamily under
`build/comparison/qr_incompatible_ls/study.tsv` instead of accidentally folding
it into QR minimum-norm output.

### Target-Specific Freshness Smoke

The direct freshness command now succeeds against the generated Day 8/Day 9
artifacts:

```sh
python3 scripts/normalize_report_index.py --family comparison --require-generated comparison --check-freshness --selected-target qr-incompatible-ls
```

Result:

- one source-controlled report-family contract row is advisory;
- all six generated `qr_incompatible_ls` rows are fresh against current HEAD;
- the command reports `normalize-report-index: freshness ok (46 rows)`.

### Windows Scope Decision

Windows selected comparison metadata remains unchanged on Day 9. Sprint 190
promoted only the Cholesky selected comparison lane on Windows. The new QR
incompatible least-squares family is selected for Linux/macOS full selected
comparison freshness only until a future sprint explicitly proves the MSVC
project probe and Windows artifact semantics for this target.

### Day 9 Artifact

The freshness integration artifact is
`docs/planning/EPIC_17/SPRINT_191/artifacts/day9-freshness-integration.md`.

### Day 9 Validation

Commands run:

```sh
python3 tests/test_normalize_report_index.py
python3 scripts/validate_corpus_schema.py
python3 tests/test_selected_report_targets_manifest.py
python3 tests/test_selected_comparison_workflow.py
python3 scripts/normalize_report_index.py --family comparison --require-generated comparison --check-freshness --selected-target qr-incompatible-ls
python3 tests/test_run_external_comparison.py
make report-index-comparison-freshness
```

Results:

- normalizer regression tests passed;
- corpus schema validation passed;
- selected target manifest test passed;
- selected comparison workflow test passed;
- target-specific `qr-incompatible-ls` freshness passed;
- external comparison runner tests passed;
- full selected comparison freshness passed with 46 normalized rows.

No `.c` or `.h` files changed on Day 9, so `make format && make lint &&
make test` is not required for this day.

### Day 10 Questions

1. Should failure coverage add runner-level generated-output corruption cases,
   or is normalizer-level stale/missing/dependency-only coverage enough?
2. Which QR incompatible error classes should remain hard failures versus
   structured diagnostics for future review comments?
3. Should Day 10 add a direct skipped/deferred target-specific regression for
   `qr-incompatible-ls` even though the all-family selected test already
   covers skip/defer behavior?

## Day 10: Focused Failure Coverage

### Failure Coverage Summary

Day 10 added runner-level failure coverage for the selected
`qr-incompatible-ls` family and retained the Day 9 normalizer-level
stale/missing/dependency-only coverage.

| Failure area | Day 10 coverage |
| --- | --- |
| Tolerance pass boundary | Project and baseline observation rows still pass when residual, solution norm, and solution values differ from expected values by less than the selected tolerance. |
| Tolerance hard failure | Project and baseline observation rows fail when residual, solution norm, or solution values exceed the selected tolerance. |
| Study-row delta failure | Project-vs-baseline study rows fail for residual, solution norm, solution values, and max-absolute-delta mismatches. |
| Selected-row validation | `validate_selected_study_rows()` raises `metric_tolerance_miss` when any required QR incompatible study row reports a failure. |
| Project parser failure | Missing `solution_norm` or `solution_values` fields in project probe output raise structured `project_probe_failed` diagnostics. |
| Baseline parser failure | Existing malformed baseline output coverage remains in place for source-controlled helper output. |
| Dependency failure | Existing missing-helper coverage remains in place for required source-controlled dense QR helper discovery and execution. |
| Freshness failure | Day 9 target-specific freshness tests continue to reject stale rows, missing artifacts, and dependency-only selected proof. |

### Runner Tests Added

Added focused tests in `tests/test_run_external_comparison.py`:

| Test | Purpose |
| --- | --- |
| `test_qr_incompatible_ls_tolerance_boundaries_pass_and_fail` | Confirms near-tolerance project/baseline observations pass and beyond-tolerance observations fail. |
| `test_qr_incompatible_ls_study_rows_reject_tolerance_miss` | Confirms study rows surface project-vs-baseline residual, norm, value, and max-delta tolerance misses. |
| `test_qr_incompatible_ls_project_parser_rejects_missing_fields` | Confirms malformed project probe output fails before rows can be interpreted as evidence. |

Added synthetic observation and manifest helpers to keep these tests local,
deterministic, and independent of optional external packages.

### Failure Classification Decision

| Failure class | Day 10 decision |
| --- | --- |
| `metric_tolerance_miss` | Hard failure for selected study rows because a generated selected comparison row exists but does not prove the required metric. |
| `project_probe_failed` | Structured hard failure for malformed project output or wrong project vector shape. |
| `baseline_malformed_output` | Structured hard failure for malformed source-controlled dense reference output. |
| `missing_baseline_helper` | Structured hard failure for the required source-controlled helper. |
| skipped/deferred selected rows | Covered at normalizer level as incomplete proof rather than accepted evidence. |
| optional NumPy/SciPy rows | Remain deferred advisory dependency rows and are not selected proof. |

### Day 10 Artifact

The failure coverage artifact is
`docs/planning/EPIC_17/SPRINT_191/artifacts/day10-failure-coverage.md`.

### Day 10 Validation

Commands run:

```sh
python3 -m py_compile tests/test_run_external_comparison.py tests/test_normalize_report_index.py scripts/run_external_comparison.py scripts/normalize_report_index.py
python3 tests/test_run_external_comparison.py
python3 tests/test_normalize_report_index.py
python3 scripts/validate_corpus_schema.py
python3 tests/test_selected_report_targets_manifest.py
python3 tests/test_selected_comparison_workflow.py
make report-index-comparison-freshness
```

Results:

- Python syntax compilation passed;
- external comparison runner tests passed;
- normalizer tests passed;
- corpus schema validation passed;
- selected target manifest test passed;
- selected comparison workflow test passed;
- full selected comparison freshness passed with 46 normalized rows.

No `.c` or `.h` files changed on Day 10, so `make format && make lint &&
make test` is not required for this day.

### Day 11 Questions

1. Which public and maintainer docs need updates from five to six selected
   comparison families?
2. Should solver-selection wording mention QR incompatible least-squares
   evidence directly, or keep it in maintainer/corpus evidence docs only?
3. Which non-claims must be repeated near any new public QR incompatible
   wording to avoid broad least-squares or ecosystem parity claims?

## Day 11: Documentation and Claim Calibration

### Documentation Summary

Day 11 updated current public, maintainer, corpus, schema, and guard wording for
the new selected QR incompatible least-squares comparison family.

| Surface | Day 11 update |
| --- | --- |
| `docs/solver_selection.md` | Added `qr_overdetermined_incompatible_4x2` to selected QR comparison wording and added a dedicated fixture-local `qr_incompatible_ls` evidence paragraph. |
| `README.md` | Updated the QR evidence section to mention selected QR minimum-norm, compatible least-squares, and incompatible least-squares fixtures. |
| `docs/cookbook.md` | Updated the QR selection note to include selected incompatible least-squares rows. |
| `docs/maintainer_guide.md` | Added QR incompatible least-squares to selected comparison interpretation and clarified that Windows metadata remains unpromoted outside the reviewed Cholesky workflow path. |
| `tests/corpus/README.md` | Added broad least-squares parity to the selected comparison non-claims near the six-family comparison table. |
| `tests/corpus/schemas/report_index_fields.md` | Added broad least-squares parity to report-index selected comparison non-claims. |
| `scripts/check_qr_header_docs_guard.sh` | Updated fixed-string docs guard checks to require the new incompatible least-squares wording. |

### Claim Boundary

The updated docs describe the new family as selected fixture-local evidence
only:

- target key: `qr-incompatible-ls`;
- report subfamily: `qr_incompatible_ls`;
- fixture: `qr_overdetermined_incompatible_4x2`;
- command: `make report-index-comparison-freshness`;
- reference: selected source-controlled dense QR helper;
- metrics: status, expected nonzero residual norm, solution norm, solution
  values, and project-vs-baseline max absolute delta;
- tolerance: `1e-10`;
- hosted scope: reviewed Linux/macOS selected comparison freshness only.

The wording keeps these non-claims visible near QR comparison credibility
statements:

- no broad QR correctness;
- no broad least-squares parity;
- no raw QR basis identity;
- no Q sign or orientation identity;
- no global rank-threshold policy;
- no broad rank-deficient solve behavior;
- no external-library parity;
- no broad Windows report freshness;
- no package/ABI proof;
- no performance proof;
- no release proof;
- no state-of-the-art status.

### Windows Scope Decision

Day 11 kept Windows selected comparison metadata unchanged. The Sprint 190
Cholesky workflow path remains the only reviewed Windows selected comparison
path. `qr-incompatible-ls` remains selected for Linux/macOS comparison
freshness until a future change proves and promotes its MSVC project probe.

### Day 11 Artifact

The documentation and claim-calibration artifact is
`docs/planning/EPIC_17/SPRINT_191/artifacts/day11-claim-calibration.md`.

### Day 11 Validation

Commands run:

```sh
bash scripts/check_qr_header_docs_guard.sh
python3 scripts/validate_corpus_schema.py
python3 tests/test_selected_report_targets_manifest.py
rg -n "five fixture|five selected|five generated|minimum-norm and compatible|QR minimum-norm and compatible|compatible least-squares rows from" README.md INSTALL.md docs/maintainer_guide.md docs/solver_selection.md docs/cookbook.md tests/corpus/README.md tests/corpus/schemas/report_index_fields.md scripts/check_qr_header_docs_guard.sh
```

Results:

- QR header/docs guard passed;
- corpus schema validation passed;
- selected target manifest test passed;
- stale active-doc selected-comparison wording was removed, with only the
  intended new cookbook phrase and guard assertion matching the final scan.

No `.c` or `.h` files changed on Day 11, so `make format && make lint &&
make test` is not required for this day.

### Day 12 Questions

1. Should integrated local validation rerun the full Python/report validation
   set from Days 8-10, or also include broader project checks despite no C
   changes?
2. Should Day 12 inspect generated `summary.md` and `manifest.tsv` contents
   manually for the new family after `make report-index-comparison-freshness`?
3. Should Day 12 include a target-specific freshness command in addition to
   the full selected comparison freshness gate?

## Day 12: Integrated Local Validation

### Validation Summary

Day 12 ran the selected `qr-incompatible-ls` family end to end and validated
the surrounding runner, report-index, manifest, workflow, docs, and QR solve
surfaces.

| Validation area | Result |
| --- | --- |
| Direct selected target generation | Passed with six generated artifacts under `build/comparison/qr_incompatible_ls/`. |
| Target-specific freshness | Passed with all six `qr_incompatible_ls` generated rows fresh against current HEAD. |
| Full selected comparison freshness | Passed for all six selected comparison families with 46 normalized rows. |
| Runner tests | Passed, including Day 10 tolerance and parser failure coverage. |
| Normalizer tests | Passed, including Day 9 target-specific freshness coverage. |
| QR solve C owner | `make build/test_qr_solve` and `./build/test_qr_solve` passed. |
| Documentation guard | QR header/docs guard passed. |
| Manifest/schema/workflow checks | Corpus schema, selected-target manifest, and selected-comparison workflow checks passed. |

### Generated Artifact Inspection

After regeneration, `build/comparison/qr_incompatible_ls/study.tsv` contained
six passing rows:

| Metric | Expected | Project | Baseline | Delta | Status |
| --- | --- | --- | --- | --- | --- |
| `project_status` | `SPARSE_SUCCESS` | `SPARSE_SUCCESS` | | | `pass` |
| `baseline_status` | `success` | | `success` | | `pass` |
| `residual_norm` | `1.7320508075688772` | `1.7320508075688772` | `1.7320508075688772` | `0` | `pass` |
| `solution_norm` | `2.2360679774997898` | `2.2360679774997894` | `2.2360679774997894` | `0` | `pass` |
| `solution_values` | `2,-1` | `1.9999999999999996,-1.0000000000000002` | `1.9999999999999998,-1` | `2.2204460492503131e-16` | `pass` |
| `project_vs_baseline_max_abs_delta` | `<=1e-10` | `1.9999999999999996,-1.0000000000000002` | `1.9999999999999998,-1` | `2.2204460492503131e-16` | `pass` |

The generated `manifest.tsv` recorded:

- `target=qr-incompatible-ls`;
- `fixture_key=qr_overdetermined_incompatible_4x2`;
- `baseline_helper_path=tests/qr_external_dense_reference.py`;
- `baseline_type=external-process-source-controlled-helper`;
- `configuration=stage=sprint191_day8_comparison_logic;baseline_status=integrated_and_compared;support_tier=local_only`;
- `source_branch=sprint-191`;
- `worktree_state=dirty`;
- `study_path=build/comparison/qr_incompatible_ls/study.tsv`.

The generated `dependency_status.tsv` recorded required `python3` and
`tests/qr_external_dense_reference.py` rows as `pass`, with optional `numpy`
and `scipy` rows as `defer`.

### C Quality Gate Decision

No `.c` or `.h` files changed, so the full `make format && make lint &&
make test` gate is not required by the sprint instruction. Day 12 still ran
the affected QR solve C test owner because the selected comparison fixture is
also covered there:

```sh
make build/test_qr_solve
./build/test_qr_solve
```

The QR solve binary reported 19 tests run, 0 failed, 0 skipped, and 1104
assertions.

### Day 12 Artifact

The integrated local validation artifact is
`docs/planning/EPIC_17/SPRINT_191/artifacts/day12-integrated-validation.md`.

### Day 12 Validation Commands

```sh
python3 scripts/run_external_comparison.py --target qr-incompatible-ls
make build/test_qr_solve
./build/test_qr_solve
python3 tests/test_run_external_comparison.py
python3 scripts/normalize_report_index.py --family comparison --require-generated comparison --check-freshness --selected-target qr-incompatible-ls
python3 tests/test_normalize_report_index.py
bash scripts/check_qr_header_docs_guard.sh
python3 scripts/validate_corpus_schema.py
python3 tests/test_selected_report_targets_manifest.py
python3 tests/test_selected_comparison_workflow.py
make report-index-comparison-freshness
python3 -m py_compile tests/test_run_external_comparison.py tests/test_normalize_report_index.py scripts/run_external_comparison.py scripts/normalize_report_index.py
```

All commands passed.

### Day 13 Questions

1. Can review surface be reduced by consolidating repeated QR incompatible
   constants or keeping the explicit tests clearer for reviewers?
2. Are any Day 8-12 docs too broad after the integrated validation pass?
3. Should generated artifact inspection remain a planning artifact only, or
   should any additional automated assertion cover manifest/summary fields?

## Day 13: Review Surface Reduction

### Review Surface Summary

Day 13 reviewed the Sprint 191 implementation, tests, manifests, workflows,
and documentation as one bounded QR incompatible least-squares comparison
change.

| Review area | Finding |
| --- | --- |
| Target identity | `qr-incompatible-ls`, `qr_incompatible_ls`, and `qr_overdetermined_incompatible_4x2` are consistent across runner, manifests, normalizer, workflows, docs, and tests. |
| Row identity | The six selected row IDs in `selected_report_targets.tsv` match `runner.expected_study_row_ids()` for the target. |
| Artifact identity | `build/comparison/qr_incompatible_ls/study.tsv` is consistent across runner output, manifests, normalizer diagnostics, tests, and docs. |
| Workflow scope | Linux and macOS upload only the six exact `build/comparison/qr_incompatible_ls/*` artifacts; Windows workflow contains no QR incompatible selected-target references. |
| Claim scope | Active docs retain fixture-local QR incompatible least-squares wording and repeat broad QR, broad least-squares, platform, package, ABI, performance, release, and state-of-the-art non-claims. |
| Generated output | No generated comparison artifacts are source-controlled; generated `build/` outputs remain ignored local validation evidence. |
| Test clarity | Explicit QR incompatible constants and tests were kept because they make the selected row set and target-specific freshness behavior easy to review. |
| Cleanup | Updated one live maintainer trust-boundary table that still listed selected QR generated comparisons as only QR minimum-norm and compatible least-squares. |

### Consistency Checks

Day 13 confirmed:

- manifest `target_key=qr-incompatible-ls`;
- manifest `subfamily=qr_incompatible_ls` matches the runner target;
- manifest artifact pattern is `build/comparison/qr_incompatible_ls/study.tsv`;
- manifest `expected_rows=6` matches `runner.expected_study_row_ids()`;
- manifest row IDs match the runner row IDs in order;
- manifest workflow platforms remain `linux;macos`;
- Linux and macOS workflows include all six exact QR incompatible artifact
  upload paths;
- Windows workflow does not mention `qr-incompatible-ls`,
  `qr_incompatible_ls`, or `qr_overdetermined_incompatible_4x2`.

### Review Decisions

| Question | Decision |
| --- | --- |
| Consolidate repeated QR incompatible constants? | Defer. The explicit constants in the tests are intentional review anchors for selected target identity and row identity. |
| Add more production abstraction around expected residuals? | No. The optional `expected_residual_norm` field keeps existing zero-residual targets stable while supporting the one nonzero residual target. |
| Promote generated artifact inspection into more automated tests? | Not on Day 13. Existing runner tests already assert required files, manifest fields, expected rows, metrics, support tier, and dependency rows. |
| Broaden Windows metadata? | No. Windows remains limited to the Sprint 190 Cholesky workflow path until QR incompatible LS has its own MSVC proof. |
| Update historical planning docs? | No. Historical sprint artifacts remain records of prior state; only current docs and Sprint 191 artifacts were updated. |

### Day 13 Artifact

The review surface audit artifact is
`docs/planning/EPIC_17/SPRINT_191/artifacts/day13-review-surface-audit.md`.

### Day 13 Validation

Commands run:

```sh
python3 tests/test_run_external_comparison.py
python3 tests/test_normalize_report_index.py
bash scripts/check_qr_header_docs_guard.sh
python3 tests/test_selected_comparison_workflow.py
python3 tests/test_selected_report_targets_manifest.py
python3 scripts/validate_corpus_schema.py
make report-index-comparison-freshness
python3 -m py_compile tests/test_run_external_comparison.py tests/test_normalize_report_index.py scripts/run_external_comparison.py scripts/normalize_report_index.py
rg -n 'five fixture|five selected|five generated|minimum-norm and compatible|QR minimum-norm and compatible|compatible least-squares rows from|selected generated comparisons for `qr_underdetermined_minnorm_2x4` and `qr_overdetermined_compatible_5x3`' README.md INSTALL.md docs/maintainer_guide.md docs/solver_selection.md docs/cookbook.md tests/corpus/README.md tests/corpus/schemas/report_index_fields.md scripts/check_qr_header_docs_guard.sh
```

Results:

- runner tests passed;
- normalizer tests passed;
- QR docs guard passed;
- selected comparison workflow guard passed;
- selected report target manifest test passed;
- corpus schema validation passed;
- full selected comparison freshness passed with 46 normalized rows;
- Python syntax compilation passed;
- stale active-doc wording scan returned only the intended cookbook phrase and
  guard assertion.

No `.c` or `.h` files changed on Day 13, so `make format && make lint &&
make test` is not required for this day.

### Day 14 Closeout Checklist

1. Rerun final focused validation, including direct generator, target-specific
   freshness, full selected freshness, runner tests, normalizer tests, docs
   guard, manifest/schema/workflow checks, and QR solve owner test.
2. Reinspect `build/comparison/qr_incompatible_ls/summary.md` and
   `manifest.tsv` after the final freshness run.
3. Confirm no generated artifacts, Python caches, or unrelated files are left
   in the worktree.
4. Verify no `.c` or `.h` files changed; otherwise run `make format &&
   make lint && make test`.
5. Write Day 14 closeout and handoff artifact with residuals and PR-ready
   evidence.

## Day 14: Sprint Closeout and Handoff

### Closeout Summary

Day 14 closed Sprint 191 with exactly one added bounded external comparison
family: `qr-incompatible-ls`.

The final implementation keeps the comparison local-only and fixture-scoped.
It does not expand QR platform, package-manager, ABI, release, performance, or
broad least-squares support claims.

### Final Evidence

The final direct generator run wrote all expected QR incompatible artifacts:

- `build/comparison/qr_incompatible_ls/project_observations.tsv`;
- `build/comparison/qr_incompatible_ls/baseline_observations.tsv`;
- `build/comparison/qr_incompatible_ls/dependency_status.tsv`;
- `build/comparison/qr_incompatible_ls/study.tsv`;
- `build/comparison/qr_incompatible_ls/summary.md`;
- `build/comparison/qr_incompatible_ls/manifest.tsv`.

The final `study.tsv` contained six rows, all `pass`:

| Metric | Closeout result |
| --- | --- |
| `project_status` | Project returned `SPARSE_SUCCESS`. |
| `baseline_status` | Source-controlled dense helper returned `success`. |
| `residual_norm` | Project and baseline both reported `1.7320508075688772`. |
| `solution_norm` | Project and baseline agreed within tolerance. |
| `solution_values` | Project and baseline agreed within tolerance for `2,-1`. |
| `project_vs_baseline_max_abs_delta` | Solution max-delta was within `<=1e-10`. |

The final manifest recorded:

- `target=qr-incompatible-ls`;
- `fixture_key=qr_overdetermined_incompatible_4x2`;
- `baseline_helper_path=tests/qr_external_dense_reference.py`;
- `baseline_type=external-process-source-controlled-helper`;
- `configuration=stage=sprint191_day8_comparison_logic;baseline_status=integrated_and_compared;support_tier=local_only`;
- `source_branch=sprint-191`;
- `worktree_state=dirty`.

Dependency status retained the intended trust boundary:

- `python3`: `pass`, required;
- `tests/qr_external_dense_reference.py`: `pass`, required;
- `numpy`: `defer`, optional package baseline not selected;
- `scipy`: `defer`, optional package baseline not selected.

### Day 14 Validation

Commands run:

```sh
python3 scripts/run_external_comparison.py --target qr-incompatible-ls
make build/test_qr_solve
./build/test_qr_solve
python3 tests/test_run_external_comparison.py
python3 scripts/normalize_report_index.py --family comparison --require-generated comparison --check-freshness --selected-target qr-incompatible-ls
python3 tests/test_normalize_report_index.py
bash scripts/check_qr_header_docs_guard.sh
python3 scripts/validate_corpus_schema.py
python3 tests/test_selected_report_targets_manifest.py
python3 tests/test_selected_comparison_workflow.py
make report-index-comparison-freshness
python3 -m py_compile tests/test_run_external_comparison.py tests/test_normalize_report_index.py scripts/run_external_comparison.py scripts/normalize_report_index.py
rg -n 'five fixture|five selected|five generated|minimum-norm and compatible|QR minimum-norm and compatible|compatible least-squares rows from|selected generated comparisons for `qr_underdetermined_minnorm_2x4` and `qr_overdetermined_compatible_5x3`' README.md INSTALL.md docs/maintainer_guide.md docs/solver_selection.md docs/cookbook.md tests/corpus/README.md tests/corpus/schemas/report_index_fields.md scripts/check_qr_header_docs_guard.sh
git diff --check
git diff --name-only -- '*.c' '*.h'
```

Results:

- direct QR incompatible generator passed;
- QR solve owner test passed with 19 tests, 0 failures, and 1104 assertions;
- external comparison runner tests passed;
- target-specific freshness passed for the QR incompatible generated rows;
- normalizer tests passed;
- QR docs guard passed;
- corpus schema validation passed;
- selected report target manifest test passed;
- selected comparison workflow guard passed;
- aggregate comparison freshness passed with 46 normalized rows;
- Python syntax compilation passed;
- active-doc stale wording scan returned only the intended cookbook phrase and
  guard assertion;
- whitespace validation passed;
- no `.c` or `.h` files changed, so `make format && make lint && make test`
  is not required for Sprint 191 Day 14.

An initial parallel target-specific freshness invocation raced the generator
before `study.tsv` existed. The same command was rerun after generation and
passed, so the final evidence is ordered and valid.

### Residual Queue

1. Windows QR incompatible selected freshness remains deferred until an MSVC
   proof and manifest/workflow ownership are added.
2. Optional NumPy/SciPy package baselines remain deferred and advisory, not
   pass evidence.
3. Broader QR least-squares external parity remains out of scope.
4. Generated comparison artifacts remain ignored local build outputs and must
   be regenerated for evidence.
5. Future comparison families should reuse the bounded fixture, exact manifest,
   target-specific freshness, and non-claim calibration pattern.

### Day 14 Artifact

The closeout and handoff artifact is
`docs/planning/EPIC_17/SPRINT_191/artifacts/day14-closeout-and-handoff.md`.

### Retrospective Inputs

- Sprint 191 delivered one complete additional bounded comparison family.
- Nonzero least-squares residual semantics are explicit in the generator and
  tests.
- Selected manifests, normalizer diagnostics, workflow artifact paths, docs,
  and guard checks agree on the QR incompatible family boundary.
- Local validation passed across runner, normalizer, schema, workflow, docs,
  QR owner, and aggregate freshness checks.
- Remaining work is documented as residual scope instead of implied support.
