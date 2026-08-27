# Sprint 183 Working Notes

**Sprint:** 183 - Additional Bounded External Comparison Family
**Epic source:** `docs/planning/EPIC_16/PROJECT_PLAN.md`
**Sprint path:** `docs/planning/EPIC_16/SPRINT_183/`
**Status:** In progress

## Source Artifact Note

The Sprint 183 source section lives in
`docs/planning/EPIC_16/PROJECT_PLAN.md` under "Sprint 183: Additional Bounded
External Comparison Family". Sprint 183 artifacts in this directory follow the
Epic 16 scope.

## Sprint Goal

Add one fully maintained external comparison family with fixtures, metrics,
report freshness, selected-target metadata, and scoped claims.

## Baseline Inputs

- `docs/planning/EPIC_16/PROJECT_PLAN.md`
- `docs/planning/EPIC_16/SPRINT_183/PLAN.md`
- `docs/planning/EPIC_16/SPRINT_182/RETROSPECTIVE.md`
- `docs/planning/EPIC_16/SPRINT_182/artifacts/day14-closeout-and-handoff.md`
- `tests/corpus/manifests/selected_report_targets.tsv`
- `tests/corpus/manifests/report_families.tsv`
- `scripts/run_external_comparison.py`
- `tests/test_run_external_comparison.py`
- `tests/test_selected_comparison_workflow.py`
- `tests/test_selected_report_targets_manifest.py`
- `README.md`
- `docs/solver_selection.md`
- `docs/maintainer_guide.md`
- `tests/corpus/README.md`
- `tests/corpus/schemas/report_index_fields.md`

## Starting Branch Snapshot

- Branch: `sprint-183`
- Starting commit: `d50762a2bdc6`
- Recent base context:
  - `d50762a2` Merge pull request #202 from `sprint-182`
  - `b036c8d1` Address PR #202 review comments
  - `e89cccbc` Complete Sprint 182 Windows report freshness decision

## Sprint 183 Project-Plan Items

| Item | Name | Status | Notes |
| --- | --- | --- | --- |
| 183.1 | Family Selection | Complete | Day 1 establishes scope, inherited selected-comparison authority, current selected families, and candidate evaluation criteria. Day 2 audits the existing runner, manifest rows, generated artifact conventions, freshness target, and workflow guard invariants. Day 3 inventories candidates and shortlists Cholesky SPD solve and LDLT KKT solve. Day 4 selects Cholesky SPD tridiagonal solve as the single Sprint 183 family. |
| 183.2 | Fixture and Metric Contract | Complete | Day 5 defines the Cholesky SPD tridiagonal fixture, RHS, expected solution, six selected rows, tolerances, helper behavior, dependency defer rules, and draft manifest metadata. Day 6 implements key-based Cholesky helper support and focused helper contract tests without changing production C behavior. |
| 183.3 | Harness Extension | Complete | Day 7 designs the Cholesky runner target, probe mode, baseline dispatch, dependency rows, self-check coverage, focused tests, output shape, and failure handling. Day 8 implements the Cholesky runner target, project probe mode, baseline dispatch, focused runner tests, self-check coverage, and local generated output inspection. |
| 183.4 | Report Integration | Complete | Day 9 registers selected target metadata, report-family semantics, Makefile freshness generation, and report-index tests. Day 10 adds Linux/macOS hosted selected comparison workflow paths, guard-test coverage, and Windows non-promotion checks. |
| 183.5 | Documentation Alignment | Complete | Day 11 aligns README, solver-selection docs, maintainer guide, corpus/report docs, and claim boundaries for the selected Cholesky comparison family. |
| 183.6 | Validation | Complete | Day 12 runs integrated focused, workflow, freshness, deferral, formatting, lint, full test, generated-output, and whitespace checks. Day 13 completes claim review and hardening. Day 14 completes closeout and handoff checks. |

## Inherited Selected Comparison Authority

Sprint 183 starts from the Sprint 181 selected target manifest and Sprint 182
Windows deferral boundary. The selected target manifest is positive
selected-target authority and currently selects four external comparison
families:

| Target ID | Target key | Family/subfamily | Selected rows | Hosted platforms | Claim boundary |
| --- | --- | --- | ---: | --- | --- |
| `SRT-COMP-QR-MINNORM` | `qr-minnorm` | `comparison/qr_minnorm` | 6 | Linux, macOS | Fixture-local QR minimum-norm comparison against the selected dense QR helper; no broad QR or external-library parity. |
| `SRT-COMP-QR-COMPATIBLE-LS` | `qr-compatible-ls` | `comparison/qr_compatible_ls` | 6 | Linux, macOS | Fixture-local QR compatible least-squares comparison; no raw QR basis, global rank-threshold, or broad rank-deficient solve claim. |
| `SRT-COMP-PSVD-DIAG6-K2` | `partial-svd-diag6-k2` | `comparison/partial_svd_diag6_k2` | 10 | Linux, macOS | Fixture-local partial-SVD diagonal top-k comparison; no broad SVD, vector identity, convergence, package, or performance claim. |
| `SRT-COMP-LU-NONSYM-SQUARE-5` | `lu-nonsym-square-5` | `comparison/lu_nonsym_square_5` | 6 | Linux, macOS | Fixture-local linked-list LU nonsymmetric square solve comparison; no broad LU, nonsymmetric ecosystem, sparse-direct, or pivoting superiority claim. |

Windows report freshness remains formally deferred by Sprint 182. Sprint 183
should not add `windows` to selected comparison `workflow_platforms` unless it
deliberately implements the Windows-safe generator, workflow, manifest, guard,
and documentation promotion path.

## Candidate Evaluation Criteria

| Criterion | Day 1 interpretation |
| --- | --- |
| User value | Does the family answer a real solver-selection or correctness question users already face? |
| Fixture stability | Can a small deterministic fixture produce stable project and baseline values across local and hosted runs? |
| Comparator availability | Is there a source-controlled dense-reference helper or a feasible helper that avoids optional package requirements as pass evidence? |
| Implementation cost | Can runner/probe changes fit into one sprint without broad refactoring or unrelated solver work? |
| Validation cost | Are focused Python tests, selected freshness checks, and any relevant C tests small enough to run reliably? |
| Claim risk | Can the claim stay fixture-local without implying broad external-library parity, platform parity, package support, performance, release readiness, or state-of-the-art status? |
| Manifest fit | Can the family be represented by exact selected target metadata, row IDs, required files, and artifact patterns? |
| Workflow fit | Can Linux/macOS selected workflow guards include the new family without broad upload paths or accidental Windows promotion? |

## Daily Log

### Day 1: Comparison Family Intake

- Re-read the Sprint 183 project-plan section and Day 1 plan.
- Reviewed Sprint 182 retrospective and closeout handoff notes.
- Confirmed Sprint 181 selected target manifest authority is still the source
  of truth for selected comparison metadata.
- Confirmed current selected comparison families are QR minimum-norm, QR
  compatible least-squares, partial-SVD diagonal top-k, and LU nonsymmetric
  square solve.
- Confirmed Sprint 182 Windows report freshness remains formally deferred and
  must not be accidentally widened during Sprint 183.
- Defined candidate evaluation criteria for user value, fixture stability,
  comparator availability, implementation cost, validation cost, claim risk,
  manifest fit, and workflow fit.
- Added Day 1 comparison-family-intake artifact.

### Day 2: Existing Comparison Surface Audit

- Inspected `scripts/run_external_comparison.py` target registration,
  constants, project probe paths, baseline helper dispatch, output writers,
  and selected-row validation behavior.
- Inspected `tests/test_run_external_comparison.py` expectations for required
  files, selected metrics, row IDs, dependency rows, and report-family
  metadata.
- Inspected `tests/test_selected_comparison_workflow.py` guard behavior for
  selected targets, fail-closed uploads, broad upload path rejection, missing
  required files, and retained Windows deferral.
- Confirmed comparison manifest rows share Linux/macOS workflow artifacts,
  local-only support tier, generated-compare freshness policy, exact required
  files, and explicit non-claims.
- Confirmed existing local generated comparison artifacts are present under
  ignored `build/comparison/` directories and are not staged.
- Added Day 2 existing-comparison-surface-audit artifact.

### Day 3: Candidate Family Inventory

- Inventoried comparison candidates not already selected by the QR,
  partial-SVD, and LU selected comparison rows.
- Confirmed source-controlled dense helpers already exist for Cholesky SPD
  solves and LDLT KKT-style symmetric-indefinite solves.
- Confirmed Cholesky has focused C coverage for small SPD solves, SuiteSparse
  SPD fixtures, AMD/RCM reordering, nearly singular SPD input, and fill-in
  comparisons.
- Confirmed LDLT has focused C coverage for SPD and indefinite solves,
  inertia, mixed 1x1/2x2 pivots, KKT-style fixtures, and backend dispatch.
- Rejected broad candidates that would require optional external packages,
  broad parity claims, unstable eigenspace identity, or performance claims.
- Shortlisted Cholesky SPD 5x5 tridiagonal solve and LDLT scaled KKT 10x10
  solve for Day 4 selection.
- Added Day 3 candidate-family-inventory artifact.

### Day 4: Family Selection

- Compared the Cholesky SPD and LDLT KKT shortlisted candidates against Sprint
  183 family-selection criteria.
- Selected exactly one Sprint 183 family: Cholesky SPD tridiagonal solve.
- Defined the closed claim as fixture-local project-vs-baseline agreement for
  one deterministic SPD tridiagonal Cholesky solve.
- Deferred LDLT KKT coverage because it carries higher pivot-pattern, inertia,
  backend, and sparse-direct ecosystem claim risk.
- Identified the implementation surface: target registry, generated fixture or
  inline entries, Cholesky project probe mode, Cholesky baseline dispatch,
  runner tests, report-family row, selected target manifest row, freshness
  target, workflow guard expectations, and documentation updates.
- Added Day 4 family-selection artifact.

### Day 5: Fixture And Metric Contract

- Defined the selected fixture contract for `cholesky_spd_tridiag_5`: a 5x5
  symmetric positive-definite tridiagonal matrix with diagonal 4 and
  off-diagonal -1.
- Defined `x_expected = [1, 2, 3, 4, 5]`, `rhs = [2, 4, 6, 8, 16]`, and
  `expected_solution_norm = sqrt(55) = 7.416198487095663`.
- Selected the standard six solve comparison rows and exact row IDs.
- Set residual, solution, and project-vs-baseline delta tolerances to
  `1e-10`.
- Defined required helper behavior for `tests/chol_external_dense_reference.py`
  and optional NumPy/SciPy defer rows.
- Drafted report-family and selected-target manifest metadata for the
  Cholesky selected comparison row.
- Added Day 5 fixture-and-metric-contract artifact.

### Day 6: Helper And Fixture Implementation

- Extended `tests/chol_external_dense_reference.py` with key-based
  `cholesky_spd_tridiag_5` fixture support.
- Preserved existing Matrix Market path behavior for Cholesky external dense
  reference checks, including missing-file skip behavior for `.mtx` paths.
- Added `tests/test_chol_external_dense_reference.py` focused tests for fixture
  matrix/RHS contract, dense Cholesky solution values, CLI output, unknown
  fixture diagnostics, and missing Matrix Market skip behavior.
- Confirmed the helper output matches the Day 5 tolerance contract; dense
  roundoff appears only at the `1e-15` scale for values 3 through 5.
- Confirmed no production C code changed and no new C fixture test is required
  on Day 6 because `tests/test_cholesky.c` already owns the 5x5 tridiagonal
  solve proof.
- Added Day 6 helper-and-fixture-implementation artifact.

### Day 7: Runner Extension Design

- Mapped `cholesky-spd-tridiag-5` into the existing
  `scripts/run_external_comparison.py` solve-shaped target contract.
- Designed the `TARGETS` entry with fixture entries, RHS, expected solution,
  tolerances, output directory, summary text, success message, and non-claims.
- Designed a new `cholesky_spd_solve` project probe mode that includes
  `sparse_cholesky.h`, factors with `sparse_cholesky_factor`, solves with
  `sparse_cholesky_solve`, and emits the existing solve observation fields.
- Designed Cholesky-specific baseline dispatch to
  `tests/chol_external_dense_reference.py cholesky_spd_tridiag_5`.
- Confirmed project, baseline, study, summary, manifest, dependency, and
  self-check output shapes can reuse the existing solve-row machinery.
- Defined focused Day 8 runner tests and failure behavior for unsupported
  target, missing helper, malformed baseline output, vector length mismatch,
  duplicate selected row, missing selected row, and metric tolerance miss.
- Added Day 7 runner-extension-design artifact.

### Day 8: Runner Implementation

- Added `cholesky-spd-tridiag-5` to `scripts/run_external_comparison.py`
  target registration with the Day 5 fixture entries, RHS, expected solution,
  tolerances, output directory, summary text, success message, and non-claims.
- Added `cholesky_spd_solve` project probe support using
  `sparse_cholesky_factor` and `sparse_cholesky_solve`.
- Added Cholesky baseline name, version, comparison configuration, helper
  dependency row, and baseline dispatch to `tests/chol_external_dense_reference.py`.
- Generalized the LU solve-baseline parser into a shared solve-baseline helper
  used by LU and Cholesky.
- Extended `tests/test_run_external_comparison.py` with Cholesky target
  expectations and unsupported-target diagnostics.
- Generated local Cholesky comparison output under
  `build/comparison/cholesky_spd_tridiag_5/` and inspected study,
  dependency, and manifest rows.
- Confirmed generated `build/comparison/` output remains unstaged.
- Added Day 8 runner-implementation artifact.

### Day 9: Report Integration

- Added `comparison/cholesky_spd_tridiag_5` to
  `tests/corpus/manifests/report_families.tsv` with generated-local origin,
  generated-compare freshness, local-only support, exact runner command,
  exact study artifact, owner, sprint provenance, and narrow non-claims.
- Added `SRT-COMP-CHOLESKY-SPD-TRIDIAG-5` to
  `tests/corpus/manifests/selected_report_targets.tsv` with six selected row
  IDs, the six required generated files, Linux/macOS selected comparison
  workflow metadata, and no Windows freshness promotion.
- Added `cholesky-spd-tridiag-5` to `make report-index-comparison-freshness`
  so local selected comparison freshness generation includes the new family
  before report-index normalization.
- Removed the temporary Day 8 report-family metadata bypass from
  `tests/test_run_external_comparison.py`.
- Updated `tests/test_normalize_report_index.py` selected comparison fixtures,
  artifact diagnostics, expected row IDs, subfamily assertions, and generated
  row construction for `cholesky_spd_tridiag_5`.
- Ran focused manifest, runner, normalizer, corpus-schema, and selected
  comparison freshness checks. The selected freshness pass produced 39
  comparison rows and included all six Cholesky selected rows.
- Confirmed generated `build/comparison/` and `build/report-index` outputs
  remain ignored and unstaged.
- Added Day 9 report-integration artifact.

### Day 10: Freshness Gate And Workflow Guard Update

- Added `cholesky-spd-tridiag-5` to the Linux hosted selected comparison
  freshness summary target list in `.github/workflows/ci.yml`.
- Added the six `build/comparison/cholesky_spd_tridiag_5/` upload paths to
  the Linux selected comparison freshness artifact allowlist.
- Added `cholesky-spd-tridiag-5` to the macOS hosted selected comparison
  freshness summary target list in `.github/workflows/macos-ci.yml`.
- Added the six `build/comparison/cholesky_spd_tridiag_5/` upload paths to
  the macOS selected comparison freshness artifact allowlist.
- Updated the macOS lane comment so the promoted hosted selected comparison
  target set names Cholesky SPD tridiagonal solve alongside QR, partial-SVD,
  and LU.
- Extended `tests/test_selected_comparison_workflow.py` with an explicit
  selected Cholesky target assertion for Linux and macOS lanes.
- Added a Cholesky-specific upload drift test that fails if the selected
  `study.tsv` path is missing from the hosted Linux upload allowlist.
- Re-ran selected workflow, manifest, normalizer, schema, and freshness
  checks. Windows report freshness remains formally deferred: no selected
  manifest row lists `windows`, and the Windows workflow guard still rejects
  selected freshness commands and selected comparison artifact names.
- Added Day 10 freshness-gate-and-workflow-guard artifact.

### Day 11: Documentation Alignment

- Updated `README.md` selected comparison non-claims so the maintained gate
  explicitly excludes broad Cholesky correctness, broad SPD coverage, and
  CSC-vs-linked-list parity.
- Updated `docs/solver_selection.md` direct-solver guidance for Cholesky with
  the fixture-local `cholesky_spd_tridiag_5` selected comparison evidence and
  its non-claims.
- Updated `docs/solver_selection.md` selected comparison narrative so QR
  references point to the selected partial-SVD, LU, and Cholesky families.
- Updated `docs/maintainer_guide.md` trust-boundary and selected comparison
  freshness sections with Cholesky ownership, selected row shapes,
  regeneration guidance, local-only support tier, optional dependency defers,
  hosted Linux/macOS boundary, and non-claims.
- Updated `tests/corpus/README.md` selected comparison docs from four to five
  fixture-local families, added the Cholesky target row, and documented its six
  generated solve rows.
- Updated `tests/corpus/schemas/report_index_fields.md` so report-index schema
  guidance names the manifest-selected Cholesky comparison family and keeps the
  selected hosted proof narrow.
- Scanned current public docs for stale selected-comparison family wording and
  for broad parity, package, platform, performance, release, or
  state-of-the-art overclaims.
- Added Day 11 documentation-alignment artifact.

### Day 12: Integrated Validation

- Ran focused Cholesky helper and external comparison runner tests for the new
  selected family.
- Ran selected report target, report-index normalizer, corpus schema, and
  selected workflow guard checks.
- Ran `make report-index-comparison-freshness`; it regenerated the selected
  comparison outputs and reported 39 fresh comparison rows, including all six
  Cholesky selected rows.
- Ran static package and package-manager deferral guards because Day 11 touched
  package/ABI and package-manager non-claim wording.
- Attempted `make test_cholesky`; the repository has no such Makefile target.
  Replaced it with the actual built Cholesky test binary,
  `build/test_cholesky`, which passed all 21 Cholesky tests.
- Ran `make format`, `make lint`, and full `make test`; all passed.
- Confirmed `make format` did not introduce tracked C/header diffs.
- Confirmed generated `build/comparison/` and `build/report-index` outputs
  remain ignored and unstaged.
- Removed Python validation cache files and directories.
- Added Day 12 integrated-validation artifact.

### Day 13: Claim Review And Hardening

- Reconciled Sprint 183 project-plan items against produced artifacts and
  confirmed items 183.1 through 183.5 are complete.
- Confirmed item 183.6 has completed the integrated Day 12 validation pass and
  only Day 14 closeout/handoff remains.
- Confirmed the selected target manifest row
  `SRT-COMP-CHOLESKY-SPD-TRIDIAG-5` has target key
  `cholesky-spd-tridiag-5`, subfamily `cholesky_spd_tridiag_5`, six expected
  rows, six required generated files, Linux/macOS workflow platforms, and
  explicit Cholesky and Windows non-claims.
- Confirmed `scripts/run_external_comparison.py` has exactly one
  `cholesky-spd-tridiag-5` target, one `cholesky_spd_solve` project probe path,
  the selected dense Cholesky helper dispatch, and matching non-claims.
- Confirmed Linux and macOS workflows each list the Cholesky target tuple once
  and upload each of the six selected generated Cholesky files exactly once.
- Confirmed docs, corpus guidance, schema guidance, tests, manifests, and
  workflow guards describe the same bounded fixture-local Cholesky claim.
- Reviewed diagnostics and found no Day 13 hardening code change needed:
  unsupported-target diagnostics include `cholesky-spd-tridiag-5`, workflow
  drift tests fail on missing Cholesky upload paths, selected manifest tests
  validate row metadata, and normalizer tests assert Cholesky subfamily,
  artifacts, row IDs, and non-claims.
- Added Day 13 claim-review-and-hardening artifact with retrospective inputs
  and Sprint 184 risk notes.

### Day 14: Closeout And Handoff

- Re-read Sprint 183 artifacts, working notes, plan items, and validation log
  for closeout consistency.
- Finalized the closed claim: one selected fixture-local Cholesky SPD
  tridiagonal solve comparison for `cholesky_spd_tridiag_5` against the
  selected source-controlled dense Cholesky helper.
- Confirmed the branch now selects five comparison families in the active
  selected target manifest: QR minimum-norm, QR compatible least-squares,
  partial-SVD diagonal top-k, linked-list LU nonsymmetric square solve, and
  Cholesky SPD tridiagonal solve.
- Confirmed generated `build/comparison/` and `build/report-index` outputs
  remain ignored and unstaged.
- Confirmed no tracked C/header diffs remain after Day 12 `make format`; full
  `make format`, `make lint`, and `make test` passed locally on Day 12.
- Recorded residual risks for hosted Linux/macOS CI execution and the retained
  Windows report freshness deferral.
- Added Day 14 closeout-and-handoff artifact.

## Day 1 Candidate Selection Boundaries

- Select exactly one additional bounded external comparison family.
- Prefer a family with an existing or straightforward source-controlled dense
  helper.
- Prefer fixture-local evidence with deterministic rows and narrow
  non-claims.
- Avoid candidates that require package-manager support, optional dependency
  pass evidence, broad external-library parity, broad platform parity,
  portable performance claims, or generated Windows report freshness.
- Keep generated comparison outputs under ignored `build/comparison/` paths
  until explicitly inspected and validated.

## Day 2 Runner Surface Inventory

| Surface | Current state | New-family invariant |
| --- | --- | --- |
| Target registry | `TARGETS` contains `qr-minnorm`, `qr-compatible-ls`, `partial-svd-diag6-k2`, and `lu-nonsym-square-5`. | Add exactly one target key with stable fixture, subfamily, operation, output directory, claim scope, summary, and success message. |
| Project probes | QR/LU use generated C probes linked against `build/libsparse_lu_ortho.a`; partial-SVD uses a dedicated partial-SVD probe. | Keep probe source deterministic and fixture-local; identify any C test validation needed if a new solver path is touched. |
| Baseline helpers | QR uses `tests/qr_external_dense_reference.py`, partial-SVD uses `tests/svd_external_dense_reference.py`, and LU uses `tests/lu_external_dense_reference.py`. | Use a source-controlled helper; optional NumPy/SciPy rows remain defer context, not pass evidence. |
| Required files | Every selected target emits `project_observations.tsv`, `baseline_observations.tsv`, `dependency_status.tsv`, `study.tsv`, `summary.md`, and `manifest.tsv`. | New family must emit the same file set unless a manifest-backed exception is justified. |
| Study rows | QR/LU selected solve families emit six rows; partial-SVD emits ten rows. | Row IDs and expected row count must be exact and manifest-owned. |
| Freshness target | `make report-index-comparison-freshness` runs all selected comparison targets then `normalize_report_index.py --family comparison --require-generated comparison --check-freshness`. | New family must be generated by the freshness target before being required. |

## Day 2 Manifest And Workflow Invariants

| Invariant | Current state |
| --- | --- |
| Support tier | Selected comparison rows are `local_only` generated evidence promoted by reviewed Linux/macOS hosted lanes only for uploaded selected artifacts. |
| Workflow platforms | Selected comparison rows list `linux;macos`; Windows remains absent under the Sprint 182 deferral. |
| Workflow artifact names | Comparison rows share `sprint175-linux-selected-comparison-freshness;sprint175-macos-selected-comparison-freshness`. |
| Required upload files | Workflow guards require each manifest-owned comparison directory to upload the six selected files exactly. |
| Broad upload paths | Workflow guards reject broad `build/comparison/**` uploads. |
| Summary checks | Workflow guards require row counts, pass counts, required manifest fields, uploaded file existence, and source commit/branch/platform columns. |
| Non-claims | Current comparison rows reject broad solver correctness, external-library parity, package-manager proof, shared-library ABI proof, Windows report freshness, performance superiority, and state-of-the-art claims. |

## Day 2 Generated Artifact Observation

Existing generated comparison outputs are present locally under:

- `build/comparison/qr_minnorm/`
- `build/comparison/qr_compatible_ls/`
- `build/comparison/partial_svd_diag6_k2/`
- `build/comparison/lu_nonsym_square_5/`

Each directory contains the six standard generated files. `git status --short
-- build/comparison` reports no staged or unstaged tracked changes, so Day 2
does not add generated report artifacts to the branch.

## Day 3 Candidate Shortlist

| Candidate | Status | Rationale |
| --- | --- | --- |
| Cholesky SPD 5x5 tridiagonal solve | Shortlisted | Strong user value for SPD direct solves, existing source-controlled dense helper, small deterministic fixture, familiar six-row solve metrics, and low claim risk when scoped to one fixture. |
| LDLT scaled KKT 10x10 solve | Shortlisted | Covers symmetric-indefinite KKT-style solve evidence not represented by current selected rows, has an existing source-controlled dense helper, and can stay fixture-local with explicit no-broad-LDLT and no-sparse-direct-parity non-claims. |
| Eigensolver diagonal top-k | Rejected for Sprint 183 | Useful, but eigenvector sign/order/subspace semantics and convergence-budget behavior would require a different metric contract than the solve-shaped selected families. |
| CG SPD solve | Rejected for Sprint 183 | Existing iterative evidence is valuable, but iteration counts and convergence budgets increase maintenance and claim risk compared with direct solve candidates. |
| GMRES/BiCGSTAB nonsymmetric solve | Rejected for Sprint 183 | Overlaps the existing LU nonsymmetric solve user question while adding convergence, restart, and preconditioner claim risk. |
| Cholesky/LDLT backend or performance comparison | Rejected for Sprint 183 | Backend and performance rows would imply speed, dispatch, or implementation-layout claims outside the selected external-comparison contract. |

## Day 3 Comparator Availability Notes

| Candidate | Helper availability | C coverage signal | Expected selected metrics |
| --- | --- | --- | --- |
| Cholesky SPD 5x5 tridiagonal solve | `tests/chol_external_dense_reference.py` already loads a Matrix Market SPD matrix, builds a deterministic RHS, and solves through a source-controlled dense Cholesky routine. | `tests/test_cholesky.c` covers 5x5 tridiagonal solve, reordering variants, SuiteSparse SPD fixtures, and nearly singular SPD behavior. | `project_status`, `baseline_status`, `residual_norm`, `solution_norm`, `solution_values`, `project_vs_baseline_max_abs_delta`. |
| LDLT scaled KKT 10x10 solve | `tests/ldlt_external_dense_reference.py` already builds `ldlt_kkt_scaled_10`, deterministic RHS, and a source-controlled dense Gaussian solve. | `tests/test_ldlt.c`, `tests/test_ldlt_backend_dispatch.c`, and `tests/test_ldlt_csc.c` cover indefinite solves, KKT-style fixtures, mixed pivots, inertia, and backend dispatch. | Six solve metrics, with optional later extension for inertia only if Day 4 accepts added row count. |

Day 4 should select exactly one family. The Cholesky candidate is the lower-risk
implementation path because it matches the existing solve-shaped runner rows
and has a simpler SPD-only claim. The LDLT candidate has higher user value for
indefinite systems but needs tighter non-claims around pivot patterns,
backend identity, inertia, and sparse-direct ecosystem parity.

## Day 4 Selected Family Decision

| Field | Decision |
| --- | --- |
| Selected family | Cholesky SPD tridiagonal solve |
| Proposed target key | `cholesky-spd-tridiag-5` |
| Proposed subfamily | `cholesky_spd_tridiag_5` |
| Proposed fixture key | `cholesky_spd_tridiag_5` |
| Fixture shape | 5x5 symmetric positive-definite tridiagonal matrix, diagonal 4 and off-diagonal -1 |
| Solver mode | one-shot `sparse_cholesky_factor` plus `sparse_cholesky_solve` |
| Baseline helper | `tests/chol_external_dense_reference.py`, extended or invoked through a source-controlled generated Matrix Market fixture path |
| Expected row count | 6 selected solve rows |
| Support tier | `local_only` generated comparison evidence, promoted only by selected Linux/macOS workflow uploads |
| Windows status | No Windows report freshness promotion in Sprint 183 |

Closed claim: for the selected 5x5 SPD tridiagonal fixture, the project
one-shot Cholesky solve and source-controlled dense Cholesky helper both
succeed and agree within the selected residual, solution-norm, solution-value,
and max-delta tolerances.

Non-claims: no broad Cholesky correctness, no broad SPD coverage, no broad
reordering coverage, no CSC-vs-linked-list parity, no fill superiority, no
external-library ecosystem parity, no package-manager proof, no
shared-library ABI proof, no Windows report freshness, no portable performance
claim, no release readiness claim, and no state-of-the-art claim.

## Day 4 Required Implementation Surface

| Surface | Required change |
| --- | --- |
| Fixture contract | Define exact entries, RHS, expected solution, expected norm, row IDs, and tolerances for `cholesky_spd_tridiag_5`. |
| C proof | Reuse `tests/test_cholesky.c` 5x5 tridiagonal solve coverage; add focused coverage only if implementation touches a new Cholesky probe path not already tested. |
| Runner target | Add one `TARGETS` entry for `cholesky-spd-tridiag-5` with `comparison_kind` `cholesky`, six solve metrics, standard output files, and narrow non-claims. |
| Project probe | Extend `project_probe_source` for a Cholesky solve mode and include `sparse_cholesky.h`. |
| Baseline dispatch | Add Cholesky-specific baseline name/version/configuration/dependency rows and call `tests/chol_external_dense_reference.py`. |
| Tests | Extend `tests/test_run_external_comparison.py` target expectations, unsupported-target diagnostics, row IDs, helper path, and report-family metadata checks. |
| Report metadata | Add `comparison/cholesky_spd_tridiag_5` metadata with generated-local origin, generated-compare freshness, exact command, exact artifact pattern, owner, and non-claims. |
| Selected manifest | Add one `SRT-COMP-CHOLESKY-SPD-TRIDIAG-5` row using Linux/macOS selected comparison artifacts and `local_only` support. |
| Freshness target | Add `python3 scripts/run_external_comparison.py --target cholesky-spd-tridiag-5` before `normalize_report_index.py --family comparison --require-generated comparison --check-freshness`. |
| Workflow guards | Require the same six uploaded files under `build/comparison/cholesky_spd_tridiag_5/` and keep broad upload paths rejected. |
| Documentation | Update README, solver-selection, maintainer guide, corpus README, and report-index schema wording with the bounded Cholesky claim. |

## Day 5 Fixture And Metric Contract

| Field | Contract |
| --- | --- |
| Target key | `cholesky-spd-tridiag-5` |
| Fixture key | `cholesky_spd_tridiag_5` |
| Subfamily | `cholesky_spd_tridiag_5` |
| Operation | `cholesky_spd_solve` |
| Matrix | 5x5 SPD tridiagonal, diagonal 4, off-diagonal -1 |
| Entries | Full symmetric coordinate entries, 13 stored values |
| RHS | `[2, 4, 6, 8, 16]` |
| Expected solution | `[1, 2, 3, 4, 5]` |
| Expected solution norm | `7.416198487095663` |
| Residual tolerance | `1e-10` absolute |
| Solution tolerance | `1e-10` absolute max element delta |
| Baseline helper | `tests/chol_external_dense_reference.py` |
| Output directory | `build/comparison/cholesky_spd_tridiag_5/` |
| Required files | `project_observations.tsv`; `baseline_observations.tsv`; `dependency_status.tsv`; `study.tsv`; `summary.md`; `manifest.tsv` |

Expected selected rows:

| Row ID | Metric | Meaning |
| --- | --- | --- |
| `comparison_cholesky_spd_tridiag_5_project_status_v1` | `project_status` | Project Cholesky factor/solve status is success. |
| `comparison_cholesky_spd_tridiag_5_baseline_status_v1` | `baseline_status` | Source-controlled dense Cholesky helper status is success. |
| `comparison_cholesky_spd_tridiag_5_residual_norm_v1` | `residual_norm` | Project residual norm for `A*x - rhs` is within `1e-10`. |
| `comparison_cholesky_spd_tridiag_5_solution_norm_v1` | `solution_norm` | Project solution 2-norm matches `sqrt(55)` within `1e-10`. |
| `comparison_cholesky_spd_tridiag_5_solution_values_v1` | `solution_values` | Project solution vector matches `[1, 2, 3, 4, 5]` and the baseline vector within `1e-10`. |
| `comparison_cholesky_spd_tridiag_5_project_vs_baseline_max_abs_delta_v1` | `project_vs_baseline_max_abs_delta` | Maximum project-vs-baseline solution delta is within `1e-10`. |

Draft report-family metadata:

| Field | Value |
| --- | --- |
| `report_family` | `comparison` |
| `subfamily` | `cholesky_spd_tridiag_5` |
| `row_meaning` | `external_process_dense_reference_comparison` |
| `row_origin` | `generated_local` |
| `status` | `unknown` |
| `support_tier` | `local_only` |
| `freshness_policy` | `generated_compare_inputs` |
| `generator_command` | `python3 scripts/run_external_comparison.py --target cholesky-spd-tridiag-5` |
| `artifact_pattern` | `build/comparison/cholesky_spd_tridiag_5/study.tsv` |
| `owner` | `Report maintainer` |

Draft selected-target metadata:

| Field | Value |
| --- | --- |
| `target_id` | `SRT-COMP-CHOLESKY-SPD-TRIDIAG-5` |
| `family` | `comparison` |
| `subfamily` | `cholesky_spd_tridiag_5` |
| `target_key` | `cholesky-spd-tridiag-5` |
| `row_meaning` | `selected Cholesky SPD tridiagonal solve comparison freshness` |
| `selection_scope` | `reviewed_cross_platform_selected` |
| `support_tier` | `local_only` |
| `freshness_policy` | `generated_compare_inputs` |
| `workflow_file` | `.github/workflows/ci.yml;.github/workflows/macos-ci.yml` |
| `workflow_job` | `generated-report-freshness;selected-comparison-freshness` |
| `workflow_artifact` | `sprint175-linux-selected-comparison-freshness;sprint175-macos-selected-comparison-freshness` |
| `workflow_platforms` | `linux;macos` |

Dependency behavior: `tests/chol_external_dense_reference.py` is a required
source-controlled helper and must be a pass row. Optional `numpy` and `scipy`
rows remain `defer` with `optional_package_baseline_not_selected`; they are
not pass evidence.

Day 6 should implement helper support either by adding the named fixture to
`tests/chol_external_dense_reference.py` or by writing a deterministic
Matrix Market fixture before invoking the helper. The runner target can use
inline entries because `descriptor_entries` already supports target-local
solve fixtures.

## Day 6 Helper Implementation Notes

| Surface | Result |
| --- | --- |
| Helper fixture support | `tests/chol_external_dense_reference.py` now accepts `cholesky_spd_tridiag_5` as a fixture key. |
| Matrix Market compatibility | Existing path arguments still load `.mtx` files; missing `.mtx` paths still emit `SKIP matrix file not found` with exit code 0. |
| Unknown fixture diagnostics | Non-path unknown fixture keys now emit `ERROR unknown fixture <name>` and exit nonzero. |
| Focused helper tests | `tests/test_chol_external_dense_reference.py` covers fixture values, RHS, dense solve, CLI success, unknown fixture failure, and missing `.mtx` skip behavior. |
| C fixture coverage | No C change was made; existing `tests/test_cholesky.c` 5x5 tridiagonal solve coverage remains the project-side fixture proof. |

Day 7 can design the runner extension against a stable helper contract. The
runner should invoke `tests/chol_external_dense_reference.py
cholesky_spd_tridiag_5` and parse the standard `OK 5` plus five solution-value
lines.

## Day 7 Runner Extension Design Summary

| Surface | Day 8 design |
| --- | --- |
| Target registration | Add `cholesky-spd-tridiag-5` to `TARGETS` with `comparison_kind=cholesky`, `solve_mode=cholesky_spd_solve`, `fixture_key=cholesky_spd_tridiag_5`, and output directory `build/comparison/cholesky_spd_tridiag_5/`. |
| Project probe | Extend `project_probe_source` for `cholesky_spd_solve`; include `sparse_cholesky.h`, call `sparse_cholesky_factor(A)`, then `sparse_cholesky_solve(A, rhs, x)`. |
| Baseline dispatch | Add Cholesky branches in `baseline_name`, `baseline_version`, `comparison_configuration`, `dependency_status_rows`, and `run_baseline_reference`. |
| Baseline parser | Reuse the LU solve parser shape: expect `OK 5`, parse five float solution values, compute residual and solution norm from target entries and RHS. |
| Study rows | Reuse existing solve rows from `comparison_study_rows`; expected row IDs already match Day 5. |
| Output files | Emit the standard six files under `build/comparison/cholesky_spd_tridiag_5/`. |
| Self-check | Existing `run_self_check` loops over `TARGETS`; adding the target should automatically cover missing-row, duplicate-row, and tolerance-miss validation. |
| Focused tests | Extend `tests/test_run_external_comparison.py` target expectations and unsupported-target diagnostics for `cholesky-spd-tridiag-5`. |
| Generated artifacts | Day 8 may generate local output for inspection, but must leave `build/comparison/` unstaged. |

Failure behavior is unchanged from existing selected solve targets:

| Failure | Expected class |
| --- | --- |
| Unknown `--target` | `unsupported_target` |
| Missing `build/libsparse_lu_ortho.a` | `missing_project_library` |
| Project probe compile failure | `project_build_failed` |
| Unsupported solve mode | `unsupported_target` |
| Missing Cholesky helper | `missing_baseline_helper` |
| Helper nonzero exit | `baseline_command_failed` |
| Malformed helper output or wrong value count | `baseline_malformed_output` |
| Project or baseline solution vector length mismatch | `project_probe_failed` or `baseline_malformed_output` |
| Missing selected row | `missing_selected_row` |
| Duplicate selected row | `duplicate_selected_row` |
| Non-pass selected row | `metric_tolerance_miss` |

## Day 8 Runner Implementation Notes

| Surface | Implementation result |
| --- | --- |
| Target key | `cholesky-spd-tridiag-5` |
| Fixture key | `cholesky_spd_tridiag_5` |
| Project probe | Generates a Cholesky probe, includes `sparse_cholesky.h`, factors with `sparse_cholesky_factor`, solves with `sparse_cholesky_solve`, and emits solve fields. |
| Baseline helper | Invokes `tests/chol_external_dense_reference.py cholesky_spd_tridiag_5`. |
| Baseline metadata | Uses `source-controlled-dense-cholesky-reference`, `chol_external_dense_reference.py`, and `stage=sprint183_day8_comparison_logic`. |
| Dependency rows | `python3` and `tests/chol_external_dense_reference.py` are required pass rows; `numpy` and `scipy` remain deferred optional rows. |
| Focused tests | `tests/test_run_external_comparison.py` now covers Cholesky output generation, row IDs, dependency rows, helper path, summary/manifest behavior, and unsupported-target diagnostics. |
| Report-family metadata | Temporarily bypassed for the Cholesky target until Day 9 adds `comparison/cholesky_spd_tridiag_5` to `report_families.tsv`; Day 9 must remove that bypass. |
| Generated local output | `build/comparison/cholesky_spd_tridiag_5/` contains the six standard generated files, all ignored and unstaged. |

Local generated row inspection:

| Observation | Result |
| --- | --- |
| Selected row count | 6 rows |
| Row statuses | all `pass` |
| Project residual | `5.7560540319981793e-15` |
| Baseline residual | `5.7560540319981793e-15` |
| Solution norm | `7.4161984870956648` |
| Max project-vs-baseline delta | `0` |
| Worktree provenance | `dirty`, expected during in-progress sprint work |

Day 9 should add the report-family and selected-target manifest rows, remove
the temporary `require_report_family_metadata=False` Cholesky test bypass, wire
the comparison freshness target/workflow surfaces, and rerun selected freshness.

## Validation Log

| Day | Validation | Status |
| --- | --- | --- |
| 1 | `git diff --check` | Pass |
| 2 | `git status --short -- build/comparison` | Pass |
| 2 | `git diff --check` | Pass |
| 3 | `git diff --check` | Pass |
| 4 | `git diff --check` | Pass |
| 5 | `git diff --check` | Pass |
| 6 | `python3 tests/test_chol_external_dense_reference.py` | Pass |
| 6 | `python3 tests/chol_external_dense_reference.py cholesky_spd_tridiag_5` | Pass |
| 6 | `python3 tests/chol_external_dense_reference.py not_a_fixture` | Pass |
| 6 | `python3 tests/chol_external_dense_reference.py tests/data/symmetric_4.mtx` | Pass |
| 6 | `python3 tests/chol_external_dense_reference.py missing_fixture.mtx` | Pass |
| 6 | `git diff --check` | Pass |
| 7 | `git diff --check` | Pass |
| 8 | `python3 scripts/run_external_comparison.py --self-check` | Pass |
| 8 | `python3 tests/test_chol_external_dense_reference.py` | Pass |
| 8 | `python3 tests/test_run_external_comparison.py` | Pass |
| 8 | `python3 scripts/run_external_comparison.py --target cholesky-spd-tridiag-5` | Pass |
| 8 | `git status --short -- build/comparison` | Pass |
| 8 | `git diff --check` | Pass |
| 9 | `python3 scripts/validate_corpus_schema.py` | Pass |
| 9 | `python3 tests/test_selected_report_targets_manifest.py` | Pass |
| 9 | `python3 tests/test_run_external_comparison.py` | Pass |
| 9 | `python3 tests/test_normalize_report_index.py` | Pass |
| 9 | `make report-index-comparison-freshness` | Pass |
| 9 | `git status --short -- build/comparison build/report-index` | Pass |
| 9 | `git diff --check` | Pass |
| 10 | `python3 tests/test_selected_comparison_workflow.py` | Pass |
| 10 | `python3 tests/test_selected_report_targets_manifest.py` | Pass |
| 10 | `python3 tests/test_normalize_report_index.py` | Pass |
| 10 | `python3 scripts/validate_corpus_schema.py` | Pass |
| 10 | `make report-index-comparison-freshness` | Pass |
| 10 | `git status --short -- build/comparison build/report-index` | Pass |
| 10 | `git diff --check` | Pass |
| 11 | `rg -n "selected QR plus partial-SVD plus LU|QR, Partial-SVD, And LU|selected QR, partial-SVD, and LU|four fixture-local comparison|manifest-selected QR, partial-SVD, and LU|selected QR, partial-SVD, and LU" README.md docs/solver_selection.md docs/maintainer_guide.md tests/corpus/README.md tests/corpus/schemas/report_index_fields.md` | Pass; no matches |
| 11 | `python3 scripts/validate_corpus_schema.py` | Pass |
| 11 | `python3 tests/test_selected_report_targets_manifest.py` | Pass |
| 11 | `python3 tests/test_normalize_report_index.py` | Pass |
| 11 | `git status --short -- build/comparison build/report-index` | Pass |
| 11 | `git diff --check` | Pass |
| 12 | `python3 tests/test_chol_external_dense_reference.py` | Pass |
| 12 | `python3 tests/test_run_external_comparison.py` | Pass |
| 12 | `python3 tests/test_selected_comparison_workflow.py` | Pass |
| 12 | `python3 tests/test_selected_report_targets_manifest.py` | Pass |
| 12 | `python3 tests/test_normalize_report_index.py` | Pass |
| 12 | `python3 scripts/validate_corpus_schema.py` | Pass |
| 12 | `make test_cholesky` | Superseded; no such Makefile target |
| 12 | `build/test_cholesky` | Pass |
| 12 | `make report-index-comparison-freshness` | Pass |
| 12 | `bash scripts/static_package_deferral_check.sh` | Pass |
| 12 | `bash scripts/package_manager_deferral_check.sh` | Pass |
| 12 | `make format` | Pass |
| 12 | `make lint` | Pass |
| 12 | `make test` | Pass |
| 12 | `git diff --name-only | rg '\.(c|h)$' || true` | Pass; no tracked C/header diffs |
| 12 | `git status --short -- build/comparison build/report-index` | Pass |
| 12 | `git diff --check` | Pass |
| 13 | `python3 - <<'PY' ... selected Cholesky manifest consistency probe ... PY` | Pass |
| 13 | `python3 - <<'PY' ... Cholesky runner consistency probe ... PY` | Pass |
| 13 | `python3 - <<'PY' ... Linux/macOS workflow Cholesky upload probe ... PY` | Pass |
| 13 | `python3 tests/test_selected_comparison_workflow.py` | Pass |
| 13 | `python3 tests/test_run_external_comparison.py` | Pass |
| 13 | `python3 tests/test_selected_report_targets_manifest.py` | Pass |
| 13 | `python3 tests/test_normalize_report_index.py` | Pass |
| 13 | `git status --short -- build/comparison build/report-index` | Pass |
| 13 | `git diff --check` | Pass |
| 14 | `ls docs/planning/EPIC_16/SPRINT_183/artifacts` | Pass |
| 14 | `git diff --name-only | rg '\.(c|h)$' || true` | Pass; no tracked C/header diffs |
| 14 | `git status --short -- build/comparison build/report-index` | Pass |
| 14 | `find scripts tests -name __pycache__ -type d -print` | Pass after final cleanup; no cache dirs |
| 14 | `python3 tests/test_selected_comparison_workflow.py` | Pass |
| 14 | `python3 tests/test_run_external_comparison.py` | Pass |
| 14 | `git diff --check` | Pass |
