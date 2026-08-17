# Sprint 160 Working Notes

## Goal

Sprint 160 adds one bounded QR comparison family beyond the current
minimum-norm seed and publishes normalized freshness evidence without
broadening QR, external-library parity, platform, package, performance, ABI, or
state-of-the-art claims.

## Starting Evidence

- Sprint 159 promoted selected oracle and QR minimum-norm comparison freshness
  into a reviewed Linux hosted lane only.
- Sprint 159 comparison artifact publication uses the split artifact group
  `sprint159-comparison-qr-minnorm` with 7-day retention.
- Current selected comparison freshness is:
  - `make report-index-comparison-freshness`
  - `python3 scripts/run_external_comparison.py --target qr-minnorm`
  - `python3 scripts/normalize_report_index.py --family comparison --require-generated comparison --check-freshness`
- Current comparison output lives under `build/comparison/qr_minnorm/`.
- Current selected comparison row IDs are fixed in
  `scripts/normalize_report_index.py` and
  `tests/test_normalize_report_index.py`.
- `tests/qr_external_dense_reference.py` already owns dense helper builders for
  `qr_overdetermined_incompatible_4x2`,
  `qr_overdetermined_compatible_5x3`,
  `qr_rankdef_duplicate_5x4_residual_only`,
  `qr_rankdef_dependent_row_4x3_residual_only`, and
  `qr_underdetermined_minnorm_2x4`.

## Branch Baseline

| Field | Value |
| --- | --- |
| Branch | `sprint-160` |
| Starting commit | `cd92502465ca21c96fc81ac5b268bba715a56a88` |
| Starting commit summary | `cd925024 Merge pull request #177 from jeffreyhorn/sprint-159` |
| Upstream state | created from current `master` after PR #177 merge |
| Initial scope note | The prompt cited an older Epic 12 location, but the current Sprint 160 section lives in `docs/planning/EPIC_14/PROJECT_PLAN.md` as `Sprint 160: QR Comparison Family Closure`. |

## Current QR Comparison Surface

| Surface | Current owner | Current state | Day 1 interpretation |
| --- | --- | --- | --- |
| Comparison Make gate | `Makefile` target `report-index-comparison-freshness` | Regenerates `qr-minnorm` then checks selected comparison freshness. | Required validation owner for current selected comparison rows. |
| Comparison generator | `scripts/run_external_comparison.py` | Supports only `--target qr-minnorm`; emits study, manifest, summary, project observations, baseline observations, and dependency status. | Main extension point for any new selected QR comparison family. |
| Dense reference helper | `tests/qr_external_dense_reference.py` | Source-controlled baseline helper with multiple QR fixture builders. | Candidate reference owner for a new overdetermined least-squares family. |
| Normalizer | `scripts/normalize_report_index.py` | Enforces selected comparison row set for current `qr_minnorm` rows. | Must be extended only after exact new row IDs and support tier are defined. |
| Normalizer tests | `tests/test_normalize_report_index.py` | Covers complete, missing, stale, duplicate, unexpected, fail, and defer selected comparison rows. | Test pattern for any added selected family row semantics. |
| Report-family metadata | `tests/corpus/manifests/report_families.tsv` | `comparison/qr_minnorm` remains `local_only` metadata while hosted CI runs selected freshness. | New family needs explicit row meaning, support tier, artifact pattern, claim scope, and non-claims. |
| QR proof owners | `tests/test_qr.c`, `tests/test_qr_solve.c`, `tests/test_qr_corpus.c`, `tests/test_qr_helpers.h` | Own QR behavior tests for rank, residual, nullspace, minimum-norm, and corpus fixtures. | C tests should be touched only if implementation or fixture-helper behavior changes. |
| Public QR wording | `docs/solver_selection.md`, `docs/maintainer_guide.md`, `README.md`, `tests/corpus/README.md` | Describes one selected QR minimum-norm comparison and explicit non-claims. | Docs must not be broadened before new generated evidence and freshness semantics exist. |
| Hosted lane | `.github/workflows/ci.yml` | Runs selected oracle/comparison freshness on Linux reviewed CI and uploads selected artifacts. | CI changes are deferred until target, metrics, row IDs, runtime, and artifact policy are explicit. |

## Candidate Target Families For Day 2

| Candidate | Existing helper support | Likely metric shape | Initial Day 1 posture |
| --- | --- | --- | --- |
| `qr_overdetermined_compatible_5x3` | Dense helper builder available. | Project status, baseline status, residual norm, solution norm, solution values, project-vs-baseline max delta. | Preferred first candidate from Sprint 159 handoff because it is compatible least-squares with exact solution semantics. |
| `qr_overdetermined_incompatible_4x2` | Dense helper builder available. | Least-squares residual and solution comparison with nonzero residual. | Candidate if Day 2 chooses residual-focused least-squares behavior, but residual semantics need more care. |
| `qr_rankdef_duplicate_5x4_residual_only` | Dense helper builder available. | Residual-only or solution diagnostic comparison. | Deferred unless Day 2 can avoid broad rank-deficient solve and basis claims. |
| `qr_rankdef_dependent_row_4x3_residual_only` | Dense helper builder available. | Residual-only or solution diagnostic comparison. | Deferred unless Day 2 can define stable rank-deficient semantics. |
| broader NumPy/SciPy/LAPACK comparison | Not required for source-controlled helper path. | External package status and version fields. | Non-goal for Sprint 160 Day 1. |

## Explicit Non-Goals

- Do not claim broad QR parity with LAPACK, NumPy, SciPy, SuiteSparse, Eigen,
  or any external-library ecosystem.
- Do not claim raw QR basis identity, Q sign/orientation identity, or basis
  ordering parity.
- Do not claim a global rank-threshold policy or broad rank-deficient solve
  correctness.
- Do not promote optional NumPy/SciPy package baselines into pass evidence.
- Do not add macOS or Windows report-index parity in this sprint.
- Do not use QR comparison freshness as package, ABI, shared-library,
  package-manager, dynamic-linking, performance, or release proof.
- Do not edit hosted CI before target selection, metric contract, row IDs,
  runtime budget, artifact paths, and row-state semantics are explicit.
- Do not touch C/header files unless the selected family requires actual QR
  implementation or proof-owner changes; if touched, run
  `make format && make lint && make test`.

## Assumptions

- The first additional family should reuse the source-controlled dense helper
  rather than adding a new external runtime dependency.
- The selected family should fit the existing generated report model:
  project observations, baseline observations, dependency status, study rows,
  summary, and manifest.
- The new comparison should be fixture-local and row-level, not a general QR
  solver-quality claim.
- Any reviewed hosted promotion must preserve the Sprint 159 split between
  selected evidence and advisory/local-only generated rows.

## Stop Conditions

- A proposed row lacks exact fixture key, row ID, metric, tolerance,
  support-tier, artifact path, claim scope, and non-claim wording.
- Skip, defer, missing, stale, duplicate, unexpected, or failing rows can be
  interpreted as pass evidence.
- The selected family requires unstable basis identity or ordering semantics.
- Runtime or artifact size is unknown before workflow changes.
- Public docs imply broad QR or external-library parity.
- C/header files change without the full required quality gate.

## Daily Log

### Day 1: Sprint Intake

- Re-read the Sprint 160 section of
  `docs/planning/EPIC_14/PROJECT_PLAN.md`.
- Confirmed Sprint 160 implements `QR Comparison Family Closure`, even though
  the prompt cited an older Epic 12 line range.
- Created Sprint 160 working notes and artifact directory under
  `docs/planning/EPIC_14/SPRINT_160/`.
- Reviewed Sprint 159 readiness notes and captured the settled boundaries:
  Linux reviewed hosted evidence exists only for selected oracle and selected
  QR minimum-norm comparison gates; optional NumPy/SciPy defers remain context;
  broad report-index output remains advisory/local.
- Inventoried current QR comparison owners: Makefile gate, comparison
  generator, dense reference helper, normalizer, normalizer tests,
  report-family metadata, QR proof-owner tests, public docs, and hosted CI.
- Recorded candidate Day 2 target families and marked
  `qr_overdetermined_compatible_5x3` as the preferred first candidate to
  evaluate because it matches the Sprint 159 handoff and has exact solution
  semantics.
- Recorded explicit non-goals and stop conditions for broad QR parity, broad
  external-library claims, basis identity, platform/package/performance/ABI
  claims, and premature hosted CI edits.
- Day 2 handoff: select one bounded QR fixture family, document rejected
  candidates, and tie the chosen target to fixture owner, row IDs, metrics,
  output paths, support tier, and non-claims.

### Day 2: Target Selection

- Reviewed maintained QR fixture and comparison surfaces:
  `tests/qr_external_dense_reference.py`, `tests/test_qr_solve.c`,
  `scripts/run_external_comparison.py`, `scripts/normalize_report_index.py`,
  `tests/test_normalize_report_index.py`, `tests/corpus/manifests/report_families.tsv`,
  `docs/maintainer_guide.md`, `docs/solver_selection.md`, and Sprint 159
  comparison handoff artifacts.
- Selected `qr_overdetermined_compatible_5x3` as the Sprint 160 QR comparison
  family.
- Selection basis:
  - it is already owned by the source-controlled dense helper;
  - it is already bounded QR evidence in `tests/test_qr_solve.c`;
  - it has exact compatible least-squares semantics with reference output
    `[1, -2, 0.5]` and residual `0`;
  - it adds a different QR comparison family than the current
    `qr_underdetermined_minnorm_2x4` minimum-norm row set;
  - it avoids raw Q-basis, Q sign/orientation, nullspace, rank-deficient,
    rank-threshold, optional NumPy/SciPy, and broad external-parity semantics.
- Ran source-controlled helper probes for selection evidence:
  - `python3 tests/qr_external_dense_reference.py qr_overdetermined_compatible_5x3`
    emitted `OK 4`, solution values `1`, `-2`, `0.5`, and residual `0`;
  - `python3 tests/qr_external_dense_reference.py qr_overdetermined_incompatible_4x2`
    emitted `OK 3`, solution values approximately `2`, `-1`, and residual
    `1.7320508075688772`, confirming its nonzero-residual semantics need a
    separate metric contract.
- Deferred `qr_overdetermined_incompatible_4x2`,
  `qr_rankdef_duplicate_5x4_residual_only`, and
  `qr_rankdef_dependent_row_4x3_residual_only` with blockers recorded in the
  Day 2 artifact.
- Drafted initial selected family mapping:
  - target name: `qr-compatible-ls`;
  - fixture key: `qr_overdetermined_compatible_5x3`;
  - output root: `build/comparison/qr_compatible_ls/`;
  - row ID prefix: `comparison_qr_overdetermined_compatible_5x3_`;
  - likely selected rows: project status, baseline status, residual norm,
    solution norm, solution values, and project-vs-baseline max absolute delta;
  - support tier remains local generated until later days explicitly promote
    selected freshness and hosted artifact behavior.
- Day 3 handoff: turn the selected family into a precise metric contract with
  residual, solution, tolerance, skip/defer, stale/missing, claim-bearing, and
  diagnostic fields.

### Day 3: Metric Contract

- Reviewed the Day 2 selected target, existing Sprint 154 comparison schema,
  and current generated-row policy in `scripts/run_external_comparison.py` and
  `scripts/normalize_report_index.py`.
- Confirmed the existing C proof owner
  `tests/test_qr_solve.c::test_qr_external_dense_reference_overdetermined_compatible_5x3`
  already compares the selected fixture against the dense helper with local
  `1e-8` solution and residual thresholds.
- Set the generated comparison contract to use the stricter report-row
  tolerance policy already used by the selected QR minimum-norm family:
  `1e-10` absolute tolerance for residual delta, solution-norm delta, solution
  component delta, and project-vs-baseline max absolute delta.
- Defined six selected rows for `qr_overdetermined_compatible_5x3`:
  - `comparison_qr_overdetermined_compatible_5x3_project_status_v1`;
  - `comparison_qr_overdetermined_compatible_5x3_baseline_status_v1`;
  - `comparison_qr_overdetermined_compatible_5x3_residual_norm_v1`;
  - `comparison_qr_overdetermined_compatible_5x3_solution_norm_v1`;
  - `comparison_qr_overdetermined_compatible_5x3_solution_values_v1`;
  - `comparison_qr_overdetermined_compatible_5x3_project_vs_baseline_max_abs_delta_v1`.
- Defined expected values:
  - project status: `SPARSE_SUCCESS`;
  - baseline status: `success`;
  - residual norm: `<=1e-10`, with baseline residual `0`;
  - solution norm: `2.2912878474779199`;
  - solution values: `1,-2,0.5`;
  - max absolute delta: `<=1e-10`.
- Explicitly excluded rank, nullspace/projector, and minimum-norm fields from
  the selected family because this compatible overdetermined fixture does not
  support those claims.
- Defined row-state semantics: only complete current selected rows with
  `status=pass` count as fixture-local evidence; skip, defer, missing, stale,
  duplicate, unexpected, malformed, error, or fail states do not count as pass
  evidence.
- Day 4 handoff: design the harness extension around target
  `qr-compatible-ls`, output root `build/comparison/qr_compatible_ls/`, the
  six selected rows, fail-closed row validation, and unchanged non-claims.

### Day 4: Harness Design

- Reviewed current `scripts/run_external_comparison.py` implementation shape:
  single `TARGET = "qr-minnorm"`, single `FIXTURE_KEY`, single
  `DEFAULT_OUTPUT_DIR`, one hardcoded project probe source, one hardcoded
  `expected_study_row_ids()` set, and summary text specific to QR
  minimum-norm.
- Reviewed `Makefile` target `report-index-comparison-freshness`, which
  currently runs only `python3 scripts/run_external_comparison.py --target
  qr-minnorm` before strict comparison freshness normalization.
- Reviewed normalizer selected-row policy in
  `scripts/normalize_report_index.py`; it currently treats all generated
  comparison rows as one selected row set and points diagnostics at
  `build/comparison/qr_minnorm/study.tsv`.
- Designed the Day 5 implementation around a target descriptor model instead
  of adding a second set of disconnected globals:
  - keep `qr-minnorm` unchanged;
  - add descriptor `qr-compatible-ls`;
  - descriptor fields include fixture key, generator key or explicit entries,
    operation, RHS, expected solution, expected solution norm, residual
    tolerance, solution tolerance, output subdirectory, row IDs, claim scope,
    non-claims, summary title, baseline parse count, and project probe solve
    mode.
- Selected output root for the new family:
  `build/comparison/qr_compatible_ls/`.
- Selected artifact files for the new family:
  `project_observations.tsv`, `baseline_observations.tsv`,
  `dependency_status.tsv`, `study.tsv`, `summary.md`, and `manifest.tsv`.
- Defined implementation failure diagnostics for unsupported target, missing
  fixture metadata, baseline command failure, baseline malformed output,
  project build/probe failure, tolerance miss, missing selected row, duplicate
  selected row, unexpected selected row, stale generated rows, and unsupported
  claim boundary.
- Identified touched surfaces for implementation:
  `scripts/run_external_comparison.py`, `Makefile`,
  `scripts/normalize_report_index.py`, `tests/test_normalize_report_index.py`,
  `tests/corpus/manifests/report_families.tsv`, and later documentation.
- Day 5 handoff: implement descriptor-backed `qr-compatible-ls` generation
  while preserving `qr-minnorm`, write the new artifacts, and keep selected
  freshness enforcement for Day 10 unless the implementation can safely add it
  earlier with tests.

### Day 5: Harness Implementation

- Refactored `scripts/run_external_comparison.py` from a single hardcoded
  `qr-minnorm` target into a descriptor-backed harness.
- Preserved existing `qr-minnorm` fixture, row IDs, output directory, summary
  scope, and success path.
- Added descriptor `qr-compatible-ls` for fixture
  `qr_overdetermined_compatible_5x3` with:
  - output root `build/comparison/qr_compatible_ls/`;
  - operation `least_squares_solve`;
  - RHS `[2.0, -2.5, 4.0, -0.5, 2.0]`;
  - expected solution `[1.0, -2.0, 0.5]`;
  - expected solution norm `2.2912878474779199`;
  - residual and solution tolerances `1e-10`;
  - solve mode `sparse_qr_factor` plus `sparse_qr_solve`;
  - six selected rows from the Day 3 contract.
- Updated baseline parsing so `qr-minnorm` still requires `OK 6`, while
  `qr-compatible-ls` requires `OK 4` and computes solution norm from the
  returned solution values.
- Updated project probe generation so the compatible family uses
  `sparse_qr_factor` and `sparse_qr_solve`, not
  `sparse_qr_solve_minnorm`.
- Updated study-row construction, selected-row validation, summary generation,
  and self-checks to use the active target descriptor.
- Generated local artifacts for both targets:
  - `build/comparison/qr_minnorm/`;
  - `build/comparison/qr_compatible_ls/`.
- Day 5 validation passed:
  - `python3 -m py_compile scripts/run_external_comparison.py`;
  - `python3 scripts/run_external_comparison.py --self-check`;
  - `python3 scripts/run_external_comparison.py --target qr-minnorm`;
  - `python3 scripts/run_external_comparison.py --target qr-compatible-ls`.
- Observed `qr-compatible-ls` selected rows: six pass rows, residual delta
  `1.7342238036525468e-15`, solution-norm delta
  `4.4408920985006262e-16`, and solution max absolute delta
  `4.4408920985006262e-16`.
- Day 6 handoff: integrate the generated comparison family with maintained
  corpus/report metadata and decide whether fixture metadata additions are
  needed before report-index freshness enforcement.

### Day 6: Corpus And Report Metadata Integration

- Reviewed QR corpus fixture manifests and confirmed
  `qr_overdetermined_compatible_5x3` is not a generated corpus-manifest
  fixture; it is maintained by `tests/qr_external_dense_reference.py` and
  proof-owned by `tests/test_qr_solve.c`.
- Decided not to add a generated corpus fixture row on Day 6 because doing so
  would imply `run_corpus_oracle.py` ownership and generator hashes that the
  selected comparison family does not use.
- Added a source-controlled report-family metadata row for
  `comparison/qr_compatible_ls` in
  `tests/corpus/manifests/report_families.tsv`.
- Kept the new report-family row `generated_local`, `local_only`, and
  `generated_compare_inputs` with artifact pattern
  `build/comparison/qr_compatible_ls/study.tsv`.
- Preserved explicit non-claims for broad QR parity, optional external-library
  parity, raw QR basis identity, Q sign/orientation, global rank-threshold
  policy, broad rank-deficient solve, hosted CI proof, platform portability,
  package-manager support, shared-library ABI, performance superiority, and
  state-of-the-art status.
- Found that adding the metadata row made the existing strict comparison
  freshness gate see 12 generated comparison rows while
  `scripts/normalize_report_index.py` still expected only the original six
  `qr-minnorm` rows.
- Closed that consistency gap immediately:
  - updated `Makefile` so `make report-index-comparison-freshness` regenerates
    both `qr-minnorm` and `qr-compatible-ls`;
  - updated `scripts/normalize_report_index.py` selected comparison row set to
    include both six-row families;
  - updated `tests/test_normalize_report_index.py` focused fixtures and
    expectations for the 12-row selected comparison set split across
    `build/comparison/qr_minnorm/` and
    `build/comparison/qr_compatible_ls/`.
- Day 6 validation passed:
  - `python3 -m py_compile scripts/run_external_comparison.py scripts/normalize_report_index.py tests/test_normalize_report_index.py`;
  - `python3 scripts/validate_corpus_schema.py`;
  - `python3 scripts/run_external_comparison.py --self-check`;
  - `python3 tests/test_normalize_report_index.py`;
  - `make report-index-comparison-freshness`.
- Day 7 handoff: design focused proof-owner tests around the new 12-row
  selected comparison policy, preserving current normalizer failure coverage
  and deciding whether additional harness-specific tests are needed.

### Day 7: Focused Proof-Owner Test Design

- Reviewed the touched Sprint 160 surfaces after Day 6:
  `scripts/run_external_comparison.py`, `Makefile`,
  `scripts/normalize_report_index.py`, `tests/test_normalize_report_index.py`,
  `tests/corpus/manifests/report_families.tsv`, and the existing QR proof
  owner `tests/test_qr_solve.c`.
- Classified existing coverage:
  - `python3 scripts/run_external_comparison.py --self-check` covers internal
    descriptor invariants and selected-row validation helpers;
  - target smoke commands cover `qr-minnorm` and `qr-compatible-ls`
    generation;
  - `python3 tests/test_normalize_report_index.py` already covers complete,
    missing, unexpected, duplicate, stale, fail, and defer selected comparison
    rows for the 12-row policy;
  - `make report-index-comparison-freshness` remains the end-to-end selected
    freshness owner because it regenerates both targets before strict
    normalization;
  - `python3 scripts/validate_corpus_schema.py` remains the report metadata
    owner.
- Decided no new C proof-owner test is required for Day 8 because the current
  sprint changes are harness/report metadata changes, while
  `tests/test_qr_solve.c` already owns the compatible overdetermined 5x3 QR
  solve behavior.
- Identified the missing focused Day 8 test surface: add harness-level script
  tests for unsupported target diagnostics, target-specific generated files,
  expected selected row IDs, subfamily/fixture/operation metadata, and all-pass
  selected rows for both comparison targets.
- Recorded row-state expectations: only complete current selected rows with
  `status=pass` count as evidence; missing, unexpected, duplicate, stale,
  fail, defer/skip, malformed, project-probe failure, or tolerance-miss states
  must fail closed.
- Recorded validation by changed-file type:
  docs-only hygiene, Python compile/self-check/target tests, corpus schema
  validation, Make freshness validation, normalizer tests, and full
  `make format && make lint && make test` only if `.c` or `.h` files change.
- Day 8 handoff: implement a small focused harness test without broadening QR
  C tests or duplicating normalizer row-state coverage.

### Day 8: Focused Tests Implementation

- Added `tests/test_run_external_comparison.py` as a focused black-box CLI test
  for `scripts/run_external_comparison.py`.
- Covered unsupported target handling:
  - invalid target returns nonzero;
  - diagnostics include `ERROR unsupported_target:`;
  - diagnostics list both supported targets, `qr-minnorm` and
    `qr-compatible-ls`.
- Covered `qr-minnorm` and `qr-compatible-ls` generation through isolated
  `--output-dir` directories:
  - required files exist: `project_observations.tsv`,
    `baseline_observations.tsv`, `dependency_status.tsv`, `study.tsv`,
    `summary.md`, and `manifest.tsv`;
  - manifest target, fixture key, and resolved study path match the invoked
    target;
  - each target emits exactly six selected study rows;
  - selected row IDs, metrics, `status=pass`, report family, subfamily,
    fixture key, operation, support tier, and artifact path match the
    descriptor-backed contract.
- Covered optional dependency context by asserting NumPy and SciPy rows remain
  `defer`, `optional_package_baseline_not_selected`, `required=no`, and
  `deferred rows are not pass evidence`.
- Preserved the Day 7 C proof-owner decision: no C/header tests were added
  because solver behavior and fixture helpers were not changed.
- First focused test run exposed that the runner records resolved output paths
  in `manifest.tsv`; adjusted the test to compare against the resolved output
  directory so it matches the actual CLI contract on platforms where `/tmp`
  may traverse a symlink.
- Day 8 validation passed:
  - `python3 -m py_compile scripts/run_external_comparison.py scripts/normalize_report_index.py tests/test_normalize_report_index.py tests/test_run_external_comparison.py`;
  - `python3 scripts/run_external_comparison.py --self-check`;
  - `python3 tests/test_run_external_comparison.py`;
  - `python3 tests/test_normalize_report_index.py`;
  - `python3 scripts/validate_corpus_schema.py`;
  - `make report-index-comparison-freshness`;
  - `git diff --check`.
- Day 9 handoff: use the now-tested runner contract when designing final
  report integration and freshness wording for the two selected comparison
  targets.

### Day 9: Report Integration Design

- Reviewed the now-tested selected comparison contract:
  - target `qr-minnorm`, subfamily `qr_minnorm`, fixture
    `qr_underdetermined_minnorm_2x4`, operation `minnorm_solve`, artifact
    `build/comparison/qr_minnorm/study.tsv`;
  - target `qr-compatible-ls`, subfamily `qr_compatible_ls`, fixture
    `qr_overdetermined_compatible_5x3`, operation `least_squares_solve`,
    artifact `build/comparison/qr_compatible_ls/study.tsv`.
- Defined the normalized row design as two source-controlled comparison
  contract rows plus 12 selected generated rows:
  - six selected `qr_underdetermined_minnorm_2x4` rows;
  - six selected `qr_overdetermined_compatible_5x3` rows.
- Kept native `comparison_row_id` values as normalized row IDs so freshness
  diagnostics and review references remain stable.
- Confirmed `make report-index-comparison-freshness` is the required selected
  comparison freshness gate and must regenerate both selected targets before
  running strict comparison freshness normalization.
- Preserved fail-closed row semantics: missing, unexpected, duplicate, stale,
  non-pass, skip, defer, malformed, project-probe failure, or tolerance-miss
  selected rows cannot count as evidence.
- Preserved support-tier classification:
  - both selected generated comparison families remain `local_only`;
  - hosted execution, if present, proves only that the reviewed Linux selected
    gate passed;
  - no broad QR, external-library, platform, package, ABI, performance,
    release, or state-of-the-art claim is added.
- Identified one Day 10 implementation gap: selected-comparison freshness
  errors in `scripts/normalize_report_index.py` still name only
  `build/comparison/qr_minnorm/study.tsv`; diagnostics should name both
  selected study artifacts now that the selected row set is split across two
  files.
- Identified documentation surfaces that still describe the comparison gate as
  QR minimum-norm-only and need later alignment:
  `docs/maintainer_guide.md`, `docs/solver_selection.md`, `README.md`, and
  `tests/corpus/README.md`.
- Day 10 handoff: update report diagnostics/tests and align selected
  comparison gate wording with the two-family design while keeping non-claims
  and `local_only` support tiers intact.

### Day 10: Report Integration Implementation

- Updated `scripts/normalize_report_index.py` selected comparison diagnostics:
  - added constants for both selected study artifacts;
  - row-set mismatch diagnostics now report
    `build/comparison/qr_minnorm/study.tsv` and
    `build/comparison/qr_compatible_ls/study.tsv`;
  - non-pass selected-row diagnostics now report both selected study artifacts.
- Updated `tests/test_normalize_report_index.py` to assert the two-artifact
  diagnostic wording for selected comparison row-set mismatch and non-pass
  selected-row failures.
- Updated selected comparison gate documentation in:
  - `docs/maintainer_guide.md`;
  - `docs/solver_selection.md`;
  - `README.md`;
  - `tests/corpus/README.md`.
- Replaced stale minimum-norm-only wording with the current two-family selected
  QR comparison contract:
  `qr_underdetermined_minnorm_2x4` minimum-norm and
  `qr_overdetermined_compatible_5x3` compatible least-squares.
- Preserved `local_only` generated row support tiers and explicit non-claims
  for broad QR parity, external-library parity, hosted/platform proof,
  package/ABI proof, performance, release, and state-of-the-art status.
- Day 10 validation passed:
  - `python3 -m py_compile scripts/run_external_comparison.py scripts/normalize_report_index.py tests/test_normalize_report_index.py tests/test_run_external_comparison.py`;
  - `python3 scripts/run_external_comparison.py --self-check`;
  - `python3 tests/test_run_external_comparison.py`;
  - `python3 tests/test_normalize_report_index.py`;
  - `python3 scripts/validate_corpus_schema.py`;
  - `make report-index-comparison-freshness`;
  - `git diff --check`.
- Day 11 handoff: review QR corpus, maintainer, solver-selection, and public
  non-claim wording as one surface and draft the Sprint 161 partial-SVD
  comparison handoff.

### Day 11: Documentation Alignment

- Reviewed QR comparison wording across `tests/corpus/README.md`,
  `docs/maintainer_guide.md`, `docs/solver_selection.md`, and `README.md`.
- Added a selected QR comparison freshness section to
  `tests/corpus/README.md` with:
  - target `qr-minnorm`, fixture `qr_underdetermined_minnorm_2x4`, artifact
    `build/comparison/qr_minnorm/study.tsv`;
  - target `qr-compatible-ls`, fixture
    `qr_overdetermined_compatible_5x3`, artifact
    `build/comparison/qr_compatible_ls/study.tsv`;
  - the six selected metrics each family contributes;
  - explicit non-claims for broad QR parity, raw basis identity, sign/orienting
    identity, global rank-threshold behavior, broad rank-deficient solve,
    external-library parity, platform/package/ABI/performance proof, and
    state-of-the-art status.
- Corrected stale maintainer wording that referred to hand-running “two
  underlying commands”; the selected comparison gate now regenerates both
  selected targets and then runs required comparison freshness normalization.
- Confirmed `docs/solver_selection.md` and `README.md` use selected QR
  comparison wording instead of minimum-norm-only wording.
- Preserved `local_only` generated row semantics: hosted execution can prove
  the selected gate passed on reviewed Linux, but it does not broaden the
  generated rows into platform, release, package, ABI, performance,
  external-library, or state-of-the-art evidence.
- Drafted the Sprint 161 partial-SVD comparison handoff:
  - start with a low-risk source-controlled target such as
    `partial_svd_diag6_k2`;
  - use descriptor-backed target definitions, source-controlled report-family
    metadata, focused runner tests, normalizer row-state tests, and strict
    selected freshness before public wording;
  - avoid raw singular-vector identity, vector sign/order identity,
    repeated-spectrum overclaims, convergence-rate claims, broad partial-SVD
    correctness, external-library parity, and platform/package/performance/ABI
    claims.
- Day 11 validation passed:
  - `git diff --check -- tests/corpus/README.md docs/maintainer_guide.md docs/planning/EPIC_14/SPRINT_160`;
  - `rg -n '[ \t]+$' tests/corpus/README.md docs/maintainer_guide.md docs/planning/EPIC_14/SPRINT_160`.
- Day 12 handoff: run the focused local validation pass for the full Sprint
  160 changed-file surface.

### Day 12: Local Validation Pass

- Ran the focused validation gate for the full Sprint 160 changed-file surface:
  Python scripts/tests, Makefile report-freshness wiring, corpus/report
  metadata, documentation, and Sprint 160 artifacts.
- Day 12 validation passed:
  - `python3 -m py_compile scripts/run_external_comparison.py scripts/normalize_report_index.py tests/test_normalize_report_index.py tests/test_run_external_comparison.py`;
  - `python3 scripts/run_external_comparison.py --self-check`;
  - `python3 tests/test_run_external_comparison.py`;
  - `python3 tests/test_normalize_report_index.py`;
  - `python3 scripts/validate_corpus_schema.py`;
  - `make report-index-comparison-freshness`;
  - `git diff --check`.
- Observed selected comparison freshness success:
  - `external-comparison: qr-minnorm project-vs-baseline comparison passed`;
  - `external-comparison: qr-compatible-ls project-vs-baseline comparison passed`;
  - `normalize-report-index: freshness ok (14 rows)`;
  - `report-index-comparison-freshness: passed (local-only generated comparison freshness)`.
- Removed generated Python bytecode cache directories from `scripts/` and
  `tests/` after validation.
- Confirmed no `.c` or `.h` files changed for Sprint 160 Day 12; the full
  `make format && make lint && make test` gate was not required.
- Day 13 handoff: trace each selected QR comparison claim to fixture,
  generated row, test, and documentation evidence; confirm support-tier and
  skip/defer wording; finalize Sprint 161 partial-SVD comparison handoff.

### Day 13: Evidence And Claim Review

- Traced selected QR comparison claims end to end:
  - `qr_underdetermined_minnorm_2x4` minimum-norm comparison maps to six
    generated rows under `build/comparison/qr_minnorm/study.tsv`;
  - `qr_overdetermined_compatible_5x3` compatible least-squares comparison
    maps to six generated rows under
    `build/comparison/qr_compatible_ls/study.tsv`;
  - both families are covered by `tests/test_run_external_comparison.py`,
    `tests/test_normalize_report_index.py`, and
    `make report-index-comparison-freshness`;
  - documentation anchors are `README.md`, `docs/maintainer_guide.md`,
    `docs/solver_selection.md`, and `tests/corpus/README.md`.
- Confirmed selected generated rows remain `local_only`; hosted execution can
  prove only that the selected gate passed on the reviewed Linux surface, not
  broad platform, package, ABI, release, performance, external-library, or
  state-of-the-art evidence.
- Confirmed skip/defer interpretation:
  - optional NumPy/SciPy dependency rows are `defer`, `required=no`, and
    `deferred rows are not pass evidence`;
  - selected comparison rows with skip/defer status are rejected by required
    freshness;
  - public and maintainer docs preserve this non-proof wording.
- Found one stale diagnostic path in `scripts/normalize_report_index.py`:
  required-generated comparison missing-family output still named only
  `build/comparison/qr_minnorm/study.tsv`.
- Fixed the diagnostic to reuse the two-artifact selected comparison
  diagnostic string and updated `tests/test_normalize_report_index.py` to
  assert it in the missing-family case.
- Focused Day 13 validation passed:
  - `python3 -m py_compile scripts/normalize_report_index.py tests/test_normalize_report_index.py`;
  - `python3 tests/test_normalize_report_index.py`;
  - `git diff --check`.
- Finalized the Sprint 161 partial-SVD handoff:
  - recommended first target `partial_svd_diag6_k2`;
  - carry forward descriptor-backed targets, source-controlled report-family
    metadata, focused runner tests, normalizer row-state tests, and strict
    selected freshness;
  - avoid raw singular-vector identity, sign/order identity, repeated-spectrum
    overclaims, convergence-rate claims, external-library parity, and
    platform/package/performance/ABI claims.
- Day 14 handoff: run final targeted checks, update closeout artifacts, review
  stale paths and non-claims one last time, and prepare retrospective inputs.

### Day 14: Closeout And Retrospective Prep

- Ran the final focused Sprint 160 validation set:
  - `python3 -m py_compile scripts/run_external_comparison.py scripts/normalize_report_index.py tests/test_normalize_report_index.py tests/test_run_external_comparison.py`;
  - `python3 scripts/run_external_comparison.py --self-check`;
  - `python3 tests/test_run_external_comparison.py`;
  - `python3 tests/test_normalize_report_index.py`;
  - `python3 scripts/validate_corpus_schema.py`;
  - `make report-index-comparison-freshness`;
  - `git diff --check`.
- Final validation passed with:
  - `external-comparison: qr-minnorm project-vs-baseline comparison passed`;
  - `external-comparison: qr-compatible-ls project-vs-baseline comparison passed`;
  - `normalize-report-index: freshness ok (14 rows)`;
  - `report-index-comparison-freshness: passed (local-only generated comparison freshness)`.
- Reviewed changed files for stale single-artifact diagnostics and unsupported
  claim wording:
  - no live `artifact=build/comparison/qr_minnorm/study.tsv`-only diagnostic
    remains;
  - remaining `minimum-norm-only` text is historical Sprint 160 artifact
    context describing stale wording that was fixed;
  - current public and maintainer docs preserve two-family selected comparison
    wording and non-claims.
- Confirmed no `.c` or `.h` files changed in Sprint 160; the full
  `make format && make lint && make test` gate was not required.
- Removed generated Python bytecode cache directories from `scripts/` and
  `tests/` after validation.
- Wrote the Day 14 closeout artifact with final validation, selected/deferred
  row closeout, claim review, cleanup notes, and retrospective inputs.
- Sprint 160 closeout state:
  - `qr-compatible-ls` is implemented and selected;
  - selected comparison freshness covers both QR comparison families;
  - tests and docs match the implemented report surface;
  - Sprint 161 partial-SVD comparison handoff is ready.
