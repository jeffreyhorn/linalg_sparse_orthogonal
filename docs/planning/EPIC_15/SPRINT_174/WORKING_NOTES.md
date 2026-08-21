# Sprint 174 Working Notes

## Sprint Goal

Add one more complete bounded external comparison family with generated
report, freshness checks, and claim-safe documentation.

## Source Artifact Note

The Sprint 174 request referenced `docs/planning/EPIC_12/PROJECT_PLAN.md`,
but the active merged Sprint 174 planning source is
`docs/planning/EPIC_15/PROJECT_PLAN.md`, section
"Sprint 174: Additional Bounded External Comparison Family".

## Branch Baseline

- Branch: `sprint-174`
- Starting point: current `master` after PR #192 merge.
- Sprint 173 status: complete and merged, with guarded local-only generated
  API HTML freshness under `make api-docs-freshness`.
- Sprint 174 plan status: day-by-day plan exists at
  `docs/planning/EPIC_15/SPRINT_174/PLAN.md`.

## Prior Evidence Carried Forward

| Input | Source | Sprint 174 use |
| --- | --- | --- |
| Evidence ledger and claim gates | `docs/planning/EPIC_15/SPRINT_167/` | Treat comparison evidence as fixture-local unless explicitly promoted by source-controlled rows, generated artifacts, and freshness gates. |
| Selected comparison freshness gate | `make report-index-comparison-freshness` | Existing freshness command for selected QR and partial-SVD generated comparison families. |
| Comparison runner | `scripts/run_external_comparison.py` | Existing generator for selected local comparison artifacts. |
| Report normalization and freshness | `scripts/normalize_report_index.py` | Existing source-controlled row and generated-output freshness checker. |
| Report-family manifest | `tests/corpus/manifests/report_families.tsv` | Source-controlled proof-owner rows for selected comparison subfamilies and non-claims. |
| Comparison docs | `tests/corpus/README.md`, `docs/maintainer_guide.md`, `README.md` | User and maintainer interpretation of selected comparison rows. |
| Generated API handoff | `docs/planning/EPIC_15/SPRINT_173/RETROSPECTIVE.md` | Keep generated API HTML local-only and separate from generated comparison evidence. |
| Static package/shared ABI deferral | `scripts/static_package_deferral_check.sh` | Guard support wording if Sprint 174 touches package, ABI, or runtime-loader surfaces. |
| Package-manager deferral | `scripts/package_manager_deferral_check.sh` | Guard provider/package-manager non-claims if Sprint 174 touches adoption wording. |

## Current Comparison Surface

The maintained selected comparison freshness command is:

```sh
make report-index-comparison-freshness
```

It currently runs:

- `python3 scripts/run_external_comparison.py --target qr-minnorm`;
- `python3 scripts/run_external_comparison.py --target qr-compatible-ls`;
- `python3 scripts/run_external_comparison.py --target partial-svd-diag6-k2`;
- `python3 scripts/normalize_report_index.py --family comparison
  --require-generated comparison --check-freshness`.

The existing selected comparison proof-owner rows in
`tests/corpus/manifests/report_families.tsv` are:

| Subfamily | Fixture | Artifact | Meaning |
| --- | --- | --- | --- |
| `qr_minnorm` | `qr_underdetermined_minnorm_2x4` | `build/comparison/qr_minnorm/study.tsv` | QR minimum-norm solve against the source-controlled dense QR reference helper. |
| `qr_compatible_ls` | `qr_overdetermined_compatible_5x3` | `build/comparison/qr_compatible_ls/study.tsv` | QR compatible least-squares solve against the source-controlled dense QR reference helper. |
| `partial_svd_diag6_k2` | `partial_svd_diag6_k2` | `build/comparison/partial_svd_diag6_k2/study.tsv` | Partial-SVD diagonal top-k comparison against the source-controlled dense SVD reference helper. |

The existing selected generated comparison row inventory is:

- 6 rows for `qr_underdetermined_minnorm_2x4`;
- 6 rows for `qr_overdetermined_compatible_5x3`;
- 10 rows for `partial_svd_diag6_k2`;
- 22 generated selected comparison rows total;
- 3 source-controlled comparison contract rows.

## Existing External Dense-Reference Helpers

The repository already has external dense-reference helper scripts for:

- Cholesky CSC: `tests/chol_external_dense_reference.py`;
- LDLT CSC: `tests/ldlt_external_dense_reference.py`;
- linked-list LU: `tests/lu_external_dense_reference.py`;
- QR: `tests/qr_external_dense_reference.py`;
- SVD and partial SVD: `tests/svd_external_dense_reference.py`.

Some direct-solver helpers already back tests and maintainer-guide claims, but
only selected QR and partial-SVD families currently participate in the
generated comparison report/freshness family.

## Retained Claim Non-Claims

Sprint 174 starts with no support claim for:

- broad QR, SVD, partial-SVD, LU, LDLT, Cholesky, iterative, or eigensolver
  ecosystem parity;
- LAPACK, NumPy, SciPy, SuiteSparse, Eigen, PETSc, Trilinos, ARPACK, or broad
  external-library parity;
- raw QR basis identity;
- raw singular-vector identity;
- vector sign or orientation identity;
- global rank-threshold policy;
- broad rank-deficient solve behavior;
- hosted report-freshness coverage beyond already selected reviewed lanes;
- release evidence;
- broad platform support;
- package-manager behavior;
- shared-library ABI support;
- runtime-loader behavior;
- performance superiority;
- state-of-the-art sparse linear algebra coverage.

Generated comparison rows may support only named fixtures, selected solver
family behavior, selected external comparator, selected metrics, selected
tolerances, current source commit, and the explicitly documented support tier.

## Sprint 174 Stop Conditions

Stop and revise before proceeding if a change:

- implements a comparison family before a selection artifact exists;
- adds broad parity wording before fixtures, comparator, metrics, tolerances,
  and non-claims are recorded;
- adds generated comparison rows without report-family proof-owner rows;
- adds proof-owner rows without generated-output freshness checks;
- treats optional NumPy/SciPy/helper absence as pass evidence;
- stages generated files under `build/` or `docs/api/`;
- mixes Sprint 173 generated API HTML freshness with comparison evidence;
- weakens package-manager, static package, shared-library ABI, platform,
  performance, or state-of-the-art non-claims;
- changes `.c` or `.h` files without running `make format && make lint &&
  make test`.

## Working Assumptions

- Day 1 is planning and intake only.
- If only planning files change on a given day, `git diff --check` is
  sufficient for that day.
- If comparison runner scripts, Make targets, report manifests, docs, tests, or
  generated-output rules change later, run focused comparison freshness,
  report-index, claim-scan, and deferral-guard checks.
- If `.c` or `.h` files change, run the full C quality gate.
- Generated comparison outputs remain ignored local artifacts unless a future
  decision explicitly promotes them.

## Daily Log

### Day 1: Sprint Intake And Comparison Boundary

- Re-read the active Sprint 174 section of
  `docs/planning/EPIC_15/PROJECT_PLAN.md`.
- Confirmed the prompt path points at an older Epic 12 planning file, while
  the active Sprint 174 section lives in Epic 15.
- Reviewed Sprint 173 closeout and retrospective to preserve generated API
  local-only boundaries and keep generated API HTML separate from comparison
  evidence.
- Reviewed existing selected comparison freshness behavior in Makefile,
  `scripts/run_external_comparison.py`,
  `scripts/normalize_report_index.py`,
  `tests/corpus/manifests/report_families.tsv`,
  `tests/corpus/README.md`, README, and `docs/maintainer_guide.md`.
- Identified current selected generated comparison families:
  `qr_minnorm`, `qr_compatible_ls`, and `partial_svd_diag6_k2`.
- Identified existing external dense-reference helpers for Cholesky CSC, LDLT
  CSC, linked-list LU, QR, SVD, and partial SVD.
- Recorded retained comparison non-claims and stop conditions before family
  selection.
- Created Sprint 174 artifact directory structure.
- Day 1 changed planning artifacts only. No `.c` or `.h` files were modified,
  so the full C quality gate is not required for this day.
- Created `artifacts/day1-comparison-intake.md`.

### Day 2: Candidate Family Inventory

- Inventoried candidate generated comparison families using the Day 1 intake,
  Sprint 167 evidence-ledger notes, current maintainer-guide support tables,
  existing dense-reference helpers, and current comparison report/freshness
  infrastructure.
- Confirmed current selected generated comparison families remain
  `qr_minnorm`, `qr_compatible_ls`, and `partial_svd_diag6_k2`.
- Reviewed existing external dense-reference helper-backed candidates:
  Cholesky CSC SPD, LDLT CSC KKT, linked-list LU, QR, and SVD/partial-SVD.
- Ranked linked-list LU on `lu_nonsym_square_5` as the strongest Day 3
  candidate because it has an existing source-controlled helper,
  deterministic fixture, high user value, clean solve-quality metrics, and
  contained non-claims.
- Ranked LDLT CSC KKT and Cholesky CSC SPD as strong alternatives, with more
  wording/schema risk around factorization interpretation and Matrix Market
  fixture/reorder choices.
- Deferred additional QR and partial-SVD/SVD comparison families as lower
  closure value because they already have selected generated comparison rows.
- Deferred LU CSR, iterative solver, and eigensolver comparison families
  because they need more API, external-helper, convergence, or eigenpair
  architecture before implementation.
- Day 2 changed planning artifacts only. No `.c` or `.h` files were modified,
  so the full C quality gate is not required for this day.
- Created `artifacts/day2-candidate-family-inventory.md`.

### Day 3: Family Selection Decision

- Selected exactly one Sprint 174 generated comparison family: linked-list LU
  on `lu_nonsym_square_5`.
- Selected `tests/lu_external_dense_reference.py` as the source-controlled
  dense external comparator.
- Confirmed the selected fixture already exists in `tests/test_sparse_lu.c`
  and has helper-backed expected solution behavior.
- Ran `python3 tests/lu_external_dense_reference.py lu_nonsym_square_5`; it
  returned `OK 5` and the expected solution values `[1, 2, 3, 4, 5]` within
  floating precision.
- Defined the initial report shape as six generated rows:
  `project_status`, `baseline_status`, `residual_norm`, `solution_norm`,
  `solution_values`, and `project_vs_baseline_max_abs_delta`.
- Defined the initial tolerance policy as `1e-10` for residual, solution
  values, solution norm delta, and project-vs-baseline max absolute delta,
  matching the existing C external LU test threshold.
- Explicitly excluded LU CSR, direct CSR/CSC public solve API, broad
  nonsymmetric parity, singular-report-family coverage, pivoting superiority,
  package, ABI, platform, performance, external-library parity, and
  state-of-the-art claims.
- Deferred LDLT CSC KKT, Cholesky CSC SPD, additional QR, additional
  partial-SVD/SVD, LU CSR, iterative, and eigensolver comparison families.
- Day 3 changed planning artifacts only. No `.c` or `.h` files were modified,
  so the full C quality gate is not required for this day.
- Created `artifacts/day3-family-selection.md`.

### Day 4: Fixture Design

- Converted the Day 3 linked-list LU selection into a precise fixture design
  for `lu_nonsym_square_5`.
- Reused the existing matrix definitions from
  `tests/lu_external_dense_reference.py` and `tests/test_sparse_lu.c` rather
  than inventing a new fixture.
- Recorded the selected 5x5 nonsymmetric matrix, expected solution
  `[1, 2, 3, 4, 5]`, and right-hand side
  `[12.5, 10.5, 18.0, 24.0, 48.0]`.
- Computed dense-reference diagnostics from the helper: infinity norm
  `4.999999999999999`, 2-norm `7.416198487095663`, and residual infinity norm
  `7.105427357601002e-15`.
- Defined the project solve path as
  `sparse_lu_factor(..., SPARSE_PIVOT_COMPLETE, 1e-12)` followed by
  `sparse_lu_solve(...)`.
- Preserved the six-row generated comparison shape from Day 3 and assigned
  exact fixture-local tolerance and diagnostic policy.
- Deferred singular-report, LU CSR, pivot-strategy, multi-pivot, larger corpus,
  performance/fill, and hosted-promotion work.
- Day 4 changed planning artifacts only. No `.c` or `.h` files were modified,
  so the full C quality gate is not required for this day.
- Created `artifacts/day4-fixture-design.md`.

### Day 5: Comparator Output Design

- Reviewed the existing `scripts/run_external_comparison.py` target registry,
  shared `study.tsv` schema, observation rows, selected-row validation, output
  reset behavior, manifest writer, and report-index freshness integration.
- Confirmed the selected LU target can reuse the existing non-partial-SVD
  generated comparison row shape without adding new schema fields.
- Designed the runner target as `lu-nonsym-square-5`, subfamily
  `lu_nonsym_square_5`, output directory
  `build/comparison/lu_nonsym_square_5/`, and study artifact
  `build/comparison/lu_nonsym_square_5/study.tsv`.
- Designed the comparator command as
  `python3 tests/lu_external_dense_reference.py lu_nonsym_square_5`.
- Defined six selected generated row IDs for project status, baseline status,
  residual norm, solution norm, solution values, and project-vs-baseline max
  absolute delta.
- Defined fail-closed behavior for missing helper, failed baseline command,
  malformed baseline output, project probe failure, missing selected rows,
  duplicate selected rows, non-pass rows, stale source commits, missing
  artifacts, and row-count mismatches.
- Planned report-index integration order: runner target, selected row IDs,
  selected artifact list, report-family manifest row, Make freshness target,
  and docs.
- Day 5 changed planning artifacts only. No `.c` or `.h` files were modified,
  so the full C quality gate is not required for this day.
- Created `artifacts/day5-comparator-output-design.md`.

### Day 6: Fixture Implementation

- Added `tests/test_lu_external_dense_reference.py` as a focused
  source-controlled guard for the selected `lu_nonsym_square_5` dense
  reference helper contract.
- Guarded the exact 5x5 matrix, fixture-key lookup, generated right-hand side,
  dense solution, CLI output format, and unknown-fixture failure behavior.
- Kept the generated comparison target, report-family manifest, selected row
  list, Make freshness target, and generated outputs unchanged for Day 6; those
  remain Day 7 through Day 10 work.
- Ran `python3 tests/test_lu_external_dense_reference.py`; it passed.
- Ran `python3 tests/lu_external_dense_reference.py lu_nonsym_square_5`; it
  returned the expected `OK 5` helper output.
- Day 6 changed a Python test and planning artifacts only. No `.c` or `.h`
  files were modified, so the full C quality gate is not required for this
  day.
- Created `artifacts/day6-fixture-implementation.md`.

### Day 7: Harness Extension Design

- Reviewed the existing generated-comparison runner entry points in
  `scripts/run_external_comparison.py`, including `TARGETS`,
  `run_project_probe()`, `run_baseline_reference()`,
  `comparison_study_rows()`, exact selected-row validation, output reset, and
  manifest/report writing.
- Reviewed report-index freshness ownership in
  `scripts/normalize_report_index.py`, `make report-index-comparison-freshness`,
  and `tests/corpus/manifests/report_families.tsv`.
- Designed the narrow runner target as `lu-nonsym-square-5`, with fixture key
  `lu_nonsym_square_5`, subfamily `lu_nonsym_square_5`, output directory
  `build/comparison/lu_nonsym_square_5/`, and support tier `local_only`.
- Preserved the Day 6 dense-reference helper contract as solution-only `OK 5`
  output and selected a narrow LU baseline adapter in the runner to compute
  baseline residual and solution norm from target metadata.
- Defined the LU project probe as a temporary generated C program that builds
  the Day 4 matrix, runs `sparse_lu_factor(..., SPARSE_PIVOT_COMPLETE,
  1e-12)` plus `sparse_lu_solve(...)`, and emits the existing observation keys.
- Reused the existing six non-partial-SVD comparison row IDs and study schema;
  no new `study.tsv` fields are needed.
- Defined generated output ownership as ignored `build/comparison/` artifacts:
  `project_observations.tsv`, `baseline_observations.tsv`,
  `dependency_status.tsv`, `study.tsv`, `summary.md`, and `manifest.tsv`.
- Planned implementation order for the target registry, LU adapter, tests,
  selected comparison rows, selected artifact list, report-family manifest row,
  and Make freshness target.
- Kept the claim boundary fixture-local and explicitly excluded broad LU
  correctness, sparse-direct solver parity, external-library parity, hosted CI,
  package, ABI, performance, and state-of-the-art claims.
- Day 7 changed planning artifacts only. No `.c` or `.h` files were modified,
  so the full C quality gate is not required for this day.
- Created `artifacts/day7-harness-extension-design.md`.

### Day 8: Harness Extension Implementation

- Added the `lu-nonsym-square-5` target to
  `scripts/run_external_comparison.py` with fixture key
  `lu_nonsym_square_5`, subfamily `lu_nonsym_square_5`, operation
  `square_solve`, output directory `build/comparison/lu_nonsym_square_5/`,
  and `local_only` support tier.
- Embedded the selected Day 4 LU matrix entries, right-hand side, expected
  solution, solution norm, and `1e-10` comparison tolerances in the runner
  target metadata.
- Extended the temporary project probe generator with
  `solve_mode == "lu_square_solve"` so the generated C probe runs
  `sparse_lu_factor(A, SPARSE_PIVOT_COMPLETE, 1e-12)` followed by
  `sparse_lu_solve(A, rhs, x)`.
- Added a narrow LU baseline adapter that preserves the Day 6 helper CLI
  contract (`OK 5` plus five solution values) and computes baseline residual
  and solution norm in the runner from target metadata.
- Added LU-specific baseline manifest naming and configuration labeling:
  `source-controlled-dense-lu-reference`,
  `lu_external_dense_reference.py`, and
  `stage=sprint174_day8_comparison_logic`.
- Extended `tests/test_run_external_comparison.py` so the focused runner suite
  generates and validates the LU target's output files, manifest target,
  fixture key, row IDs, metrics, pass status, dependency rows, support tier,
  and artifact path. Report-family metadata checks remain deferred to Day 9.
- Generated local LU comparison artifacts under
  `build/comparison/lu_nonsym_square_5/`, including `study.tsv` with exactly
  six passing rows.
- Observed LU comparison diagnostics: project residual
  `5.3290705182007514e-15`, baseline residual
  `8.1402896778041619e-15`, residual delta
  `2.8112191596034105e-15`, solution norm delta `0`, and solution max absolute
  delta `8.8817841970012523e-16`.
- Ran `python3 tests/test_lu_external_dense_reference.py`; it passed.
- Ran `python3 scripts/run_external_comparison.py --self-check`; it passed.
- Ran `python3 tests/test_run_external_comparison.py`; it passed.
- Ran `python3 scripts/run_external_comparison.py --target
  lu-nonsym-square-5`; it generated the selected LU artifacts and passed.
- Day 8 changed Python runner/test code and planning artifacts only. No `.c`
  or `.h` source files were modified, so the full C quality gate is not
  required for this day.
- Created `artifacts/day8-harness-implementation.md`.

### Day 9: Report Index Integration

- Added the six selected LU comparison row IDs to
  `SELECTED_COMPARISON_ROW_IDS` in `scripts/normalize_report_index.py`.
- Added `build/comparison/lu_nonsym_square_5/study.tsv` to
  `SELECTED_COMPARISON_ARTIFACTS`.
- Added `python3 scripts/run_external_comparison.py --target
  lu-nonsym-square-5` to `make report-index-comparison-freshness` so the
  owning freshness command regenerates the full selected comparison set.
- Added the `comparison	lu_nonsym_square_5` proof-owner row to
  `tests/corpus/manifests/report_families.tsv` with the source-controlled
  generator command, artifact pattern, `local_only` support tier,
  `generated_compare_inputs` freshness policy, and bounded claim wording.
- Enabled report-family metadata assertions for `lu-nonsym-square-5` in
  `tests/test_run_external_comparison.py`.
- Confirmed the existing report-family schema represents the LU comparison
  family without adding new fields.
- Observed that running the normalizer directly before regenerating every
  selected comparison artifact fails with missing selected-family artifacts;
  this confirms `make report-index-comparison-freshness` is the correct
  proof-owner command.
- Ran `python3 tests/test_run_external_comparison.py`; it passed.
- Ran `python3 scripts/run_external_comparison.py --self-check`; it passed.
- Ran `make report-index-comparison-freshness`; it regenerated QR, partial-SVD,
  and LU comparison artifacts and passed with
  `normalize-report-index: freshness ok (32 rows)`.
- Day 9 changed Make/Python/report manifest/test/planning artifacts only. No
  `.c` or `.h` source files were modified, so the full C quality gate is not
  required for this day.
- Created `artifacts/day9-report-integration.md`.

### Day 10: Freshness Gate Implementation

- Extended `tests/test_normalize_report_index.py` so the selected comparison
  freshness test fixtures include the LU family, its six selected row IDs, and
  `build/comparison/lu_nonsym_square_5/study.tsv` in the artifact diagnostic.
- Added `SELECTED_LU_COMPARISON_ROW_IDS` and positive assertions that LU rows
  normalize with pass status, `local_only` support tier, the expected artifact
  path, and bounded `no broad LU correctness` non-claim wording.
- Updated the selected row-set mismatch negative proof to drop
  `comparison_lu_nonsym_square_5_project_vs_baseline_max_abs_delta_v1`, proving
  missing selected LU rows fail closed.
- Confirmed no normalizer schema changes were required; the existing selected
  comparison freshness logic already rejects missing, stale, duplicate,
  non-pass, skip/defer, and mismatched selected rows.
- Ran `python3 tests/test_normalize_report_index.py`; it passed.
- Ran `python3 tests/test_run_external_comparison.py`; it passed.
- Ran `make report-index-comparison-freshness`; it regenerated all selected
  comparison artifacts, including `build/comparison/lu_nonsym_square_5/study.tsv`,
  and passed with `normalize-report-index: freshness ok (32 rows)`.
- Ran a controlled empty-build-root negative proof; it failed closed with
  `required generated family missing: comparison`, named
  `build/comparison/lu_nonsym_square_5/study.tsv` in the artifact diagnostic,
  and pointed to `run make report-index-comparison-freshness`.
- Day 10 changed Python test/planning artifacts only. No `.c` or `.h` source
  files were modified, so the full C quality gate is not required for this day.
- Created `artifacts/day10-freshness-gate.md`.

### Day 11: Claim Documentation Update

- Updated `README.md` so quick validation and report-index guidance describe
  selected QR plus partial-SVD plus LU comparison freshness.
- Updated `docs/maintainer_guide.md` so linked-list LU names
  `make report-index-comparison-freshness` as an evidence owner, the selected
  comparison workflow runs `lu-nonsym-square-5`, expected outputs include
  `build/comparison/lu_nonsym_square_5/`, and the required selected comparison
  count is four contract rows plus 28 generated rows.
- Updated `docs/solver_selection.md` with fixture-local linked-list LU
  comparison evidence for `lu_nonsym_square_5`, including comparator,
  diagnostics, `1e-10` tolerance, local/hosted interpretation, and non-claims.
- Updated `tests/corpus/README.md` and
  `tests/corpus/schemas/report_index_fields.md` so selected comparison
  freshness includes LU, names the fourth artifact, and records the six LU
  generated rows.
- Updated `benchmarks/README.md` report-index handoff language so comparison
  output paths and interpretation cover all selected comparison families, not
  only `qr_minnorm`.
- Preserved non-claims for broad LU correctness, broad nonsymmetric solve
  correctness, LU CSR parity, external-library parity, platform, package,
  ABI, performance, release, and state-of-the-art support.
- Ran a targeted stale comparison wording scan; remaining matches are
  historical planning artifacts only, not maintained public or maintainer
  docs.
- Ran package/ABI non-claim scans and then
  `bash scripts/package_manager_deferral_check.sh` plus
  `bash scripts/static_package_deferral_check.sh`; both passed.
- Ran `python3 tests/test_normalize_report_index.py`; it passed.
- Ran `python3 tests/test_run_external_comparison.py`; it passed.
- Ran `python3 scripts/normalize_report_index.py --family comparison
  --require-generated comparison --check-freshness`; it passed with
  `normalize-report-index: freshness ok (32 rows)`.
- Day 11 changed documentation/planning artifacts only. No `.c` or `.h` files
  were modified, so the full C quality gate is not required for this day.
- Created `artifacts/day11-claim-documentation.md`.

### Day 12: Integrated Comparison Validation

- Ran `make report-index-comparison-freshness`; it regenerated all selected
  comparison artifacts, including `build/comparison/lu_nonsym_square_5/study.tsv`,
  and passed with `normalize-report-index: freshness ok (32 rows)`.
- Ran `python3 tests/test_lu_external_dense_reference.py`; it passed.
- Ran `python3 tests/test_run_external_comparison.py`; it passed.
- Ran `python3 tests/test_normalize_report_index.py`; it passed.
- Ran `python3 scripts/run_external_comparison.py --self-check`; it passed.
- Ran `python3 scripts/normalize_report_index.py --family comparison
  --require-generated comparison --check-freshness`; it passed with
  `normalize-report-index: freshness ok (32 rows)`.
- Ran a stale selected-comparison wording scan against maintained public and
  maintainer docs; no stale maintained-doc matches remained.
- Ran a broad-claim/non-claim scan for state-of-the-art, broad LU,
  nonsymmetric solve, LU CSR, package-manager, and shared-library ABI wording;
  matches were bounded non-claims or fixture-local evidence statements.
- Ran `bash scripts/package_manager_deferral_check.sh`; it passed.
- Ran `bash scripts/static_package_deferral_check.sh`; it passed.
- Confirmed the selected LU comparison validates end to end through helper,
  runner, generated artifact, report-index freshness, documentation, and
  claim-boundary checks.
- Day 12 changed planning artifacts only. No `.c` or `.h` files were modified,
  so the full C quality gate is not required for this day.
- Created `artifacts/day12-integrated-validation.md`.

### Day 13: Integrated Claim Review

- Reviewed Sprint 174 artifacts and working notes from Day 1 through Day 12.
- Reconciled the selected LU comparison claim across fixture, helper, runner,
  temporary project probe, generated rows, report-family manifest, selected
  row IDs, Make freshness target, tests, and maintained documentation.
- Confirmed all source-controlled surfaces agree on target
  `lu-nonsym-square-5`, fixture/subfamily `lu_nonsym_square_5`, operation
  `square_solve`, helper `tests/lu_external_dense_reference.py`, artifact
  `build/comparison/lu_nonsym_square_5/study.tsv`, support tier `local_only`,
  and six selected generated row IDs.
- Confirmed generated comparison outputs remain ignored local artifacts under
  `build/comparison/*/`; source-controlled proof surfaces are the runner,
  tests, selected row/artifact lists, report-family manifest row, Make target,
  and docs.
- Ran a stale selected-comparison wording scan against maintained public and
  maintainer docs; it returned no matches.
- Ran a broad-claim scan for state-of-the-art, broad LU, nonsymmetric solve,
  LU CSR, package-manager, shared-library ABI, performance, and platform
  wording; matches were fixture-local evidence statements or explicit
  non-claims.
- Ran `bash scripts/package_manager_deferral_check.sh`; it passed.
- Ran `bash scripts/static_package_deferral_check.sh`; it passed.
- Ran `python3 scripts/normalize_report_index.py --family comparison
  --require-generated comparison --check-freshness`; it passed with
  `normalize-report-index: freshness ok (32 rows)`.
- Identified Sprint 175 handoff boundaries: do not infer broad LU correctness,
  LU CSR parity, hosted expansion, broad report-index freshness, package,
  ABI, performance, platform, release, or state-of-the-art claims from this
  fixture-local comparison.
- Day 13 changed planning artifacts only. No `.c` or `.h` files were modified,
  so the full C quality gate is not required for this day.
- Created `artifacts/day13-integrated-claim-review.md`.

### Day 14: Sprint Closeout

- Reconciled Sprint 174 outcomes against project-plan items 174.1 through
  174.6.
- Confirmed the selected family is fully bounded to target
  `lu-nonsym-square-5`, fixture/subfamily `lu_nonsym_square_5`, operation
  `square_solve`, helper `tests/lu_external_dense_reference.py`, and artifact
  `build/comparison/lu_nonsym_square_5/study.tsv`.
- Confirmed generated comparison outputs remain ignored local build artifacts
  under `build/comparison/*/`; source-controlled proof lives in the runner,
  selected-row enforcement, manifest row, focused tests, Make freshness target,
  and maintained docs.
- Confirmed the selected comparison gate is now QR minnorm, QR compatible
  least-squares, partial-SVD diagonal top-k, and linked-list LU nonsymmetric
  square-solve.
- Preserved Sprint 175 handoff boundaries: no broad LU correctness, no broad
  nonsymmetric solve claim, no LU CSR parity, no sparse-direct parity, no
  external-library parity, no hosted-publication claim, no package-manager
  support claim, no shared-library ABI claim, no platform-portability claim,
  no performance-superiority claim, and no state-of-the-art claim.
- Planned final Day 14 validation with the comparison freshness target, focused
  Python tests, report-index freshness check, stale wording scans,
  package/ABI deferral checks, and `git diff --check`.
- Ran `make report-index-comparison-freshness`; it regenerated QR minnorm,
  QR compatible least-squares, partial-SVD diagonal top-k, and
  `lu_nonsym_square_5` comparison reports, then passed with
  `normalize-report-index: freshness ok (32 rows)`.
- Ran `python3 tests/test_lu_external_dense_reference.py`; it passed.
- Ran `python3 tests/test_run_external_comparison.py`; it passed.
- Ran `python3 tests/test_normalize_report_index.py`; it passed.
- Ran `python3 scripts/run_external_comparison.py --self-check`; it passed.
- Ran `python3 scripts/normalize_report_index.py --family comparison
  --require-generated comparison --check-freshness`; it passed with
  `normalize-report-index: freshness ok (32 rows)`.
- Ran the maintained-doc stale selected-comparison wording scan; it returned
  no matches.
- Ran the broad-claim/non-claim wording scan; matches were fixture-local
  evidence statements or explicit non-claims.
- Ran `bash scripts/package_manager_deferral_check.sh`; it passed.
- Ran `bash scripts/static_package_deferral_check.sh`; it passed.
- Ran `git diff --check`; it passed.
- Day 14 changed planning artifacts only. No `.c` or `.h` files were modified,
  so the full C quality gate is not required for this day.
- Created `artifacts/day14-sprint-closeout.md`.
