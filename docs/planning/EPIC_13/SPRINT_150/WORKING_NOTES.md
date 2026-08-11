# Sprint 150 Working Notes

## Goal

Sprint 150 expands QR maintained corpus coverage beyond the single Sprint 139
fixture while keeping the claim bounded to selected fixture families, explicit
oracle semantics, focused proof-owner tests, and generated-local report
evidence.

## Starting Evidence

- Sprint 147 defines the corpus-family evidence gate for Sprint 150:
  source-controlled fixture rows, generator rows, expected-result rows,
  proof-owner tests, oracle/report rows, validation commands, and bounded
  documentation are prerequisites for any promoted corpus-family claim.
- Sprint 139 closes one fixture-local QR claim for
  `qr_rank_deficient_6x4_nullspace_v1`: rank `3`, nullity `1`, and
  normalized solver-produced nullspace residual at or below `1e-10`.
- Sprint 149 leaves a Windows package-lane boundary that must not be
  reinterpreted as QR corpus, platform, package, ABI, or performance proof.
- Current source-controlled QR corpus ownership consists of one fixture row,
  one generator row, one expected-result file, the focused
  `tests/test_qr_corpus.c` proof owner, helpers in `tests/test_qr_helpers.h`,
  and the opt-in `scripts/run_corpus_oracle.py --include-solver-qr` path.
- Existing non-corpus QR tests already cover many owner-local QR behaviors:
  rank-deficient rectangular cases, residual-only least-squares cases,
  nullspace projector/subspace cases, underdetermined minimum-norm cases, and
  COLAMD/reordered QR cases.

## Item-To-Day Owner Map

| Sprint 150 Item | Primary Days | Closeout Owner |
| --- | --- | --- |
| Item 1: QR Family Selection | Days 1-3 | Day 1 records the baseline, Day 2 audits candidate families, Day 3 selects the families and claim scopes. |
| Item 2: Fixture Metadata Batch | Days 4-5 | Day 4 designs rows, Day 5 implements source-controlled metadata. |
| Item 3: Oracle Semantics | Days 6-7 | Day 6 designs oracle semantics, Day 7 implements expected data and generation paths. |
| Item 4: Proof-Owner Tests | Days 8-9 | Day 8 designs focused proof owners, Day 9 implements and registers tests. |
| Item 5: Report Integration | Days 10-11 | Day 10 designs QR report rows, Day 11 implements report/index integration. |
| Item 6: Documentation Alignment | Day 12 | Day 12 aligns corpus, solver-selection, README, tutorial/cookbook, and maintainer guidance. |
| Item 7: Validation | Days 13-14 | Day 13 runs integrated validation; Day 14 closes with Sprint 151 handoff. |

## Stop Conditions

- A proposed QR expected row requires raw Q/R basis equality, sign,
  orientation, column order, or raw basis-vector identity.
- A selected family cannot name fixture keys, generators, expected rows,
  tolerances, support tier, proof owner, validation command, and non-claims.
- A source-controlled expected row is cited as observed solver pass evidence.
- Generated oracle/report rows omit command, commit, platform, compiler,
  configuration, support tier, claim scope, or non-claims.
- Optional-data skip/defer rows are counted as QR solver pass evidence.
- Documentation widens selected fixture-family evidence into broad QR,
  external-library, platform, package, performance, or state-of-the-art claims.
- Sprint 149 Windows CMake install/downstream evidence is reused as QR platform
  proof.
- Required focused tests, corpus schema checks, oracle/report checks, or full C
  quality gates fail after implementation changes.

## Daily Log

### Day 1: QR Intake

- Re-read the Sprint 150 section of
  `docs/planning/EPIC_13/PROJECT_PLAN.md`.
- Reviewed Sprint 139 QR retrospective and closeout context, Sprint 147
  corpus-family evidence gate, and Sprint 149 Windows package-lane handoff.
- Created the Sprint 150 artifact directory and Day 1 QR intake artifact.
- Inventoried QR implementation files:
  `include/sparse_qr.h`, `src/sparse_qr.c`,
  `src/sparse_qr_householder.c`, and `src/sparse_qr_internal.h`.
- Inventoried QR tests and proof owners:
  `tests/test_qr.c`, `tests/test_qr_solve.c`,
  `tests/test_qr_corpus.c`, `tests/test_qr_helpers.h`,
  `tests/test_colamd.c`, and `tests/qr_external_dense_reference.py`.
- Inventoried current source-controlled corpus rows:
  `tests/corpus/manifests/fixtures.tsv`,
  `tests/corpus/manifests/generators.tsv`,
  `tests/corpus/expected/qr_rank_deficient_6x4_nullspace_v1.tsv`, and
  `tests/corpus/manifests/report_families.tsv`.
- Confirmed the only current QR maintained corpus fixture is
  `qr_rank_deficient_6x4_nullspace_v1`; many other QR fixtures live as
  owner-local test evidence but not yet source-controlled corpus families.
- Captured Day 1 stop conditions for raw-basis identity claims, unowned
  fixtures, stale generated reports, optional-data pass confusion, and
  unsupported Windows/package inference.
- Day 2 handoff: audit candidate QR families for rank-deficient rectangular,
  underdetermined minimum-norm, and reorder/COLAMD closure value, metadata
  needs, oracle readiness, and implementation risk.

### Day 2: Family Audit

- Created the QR family candidate audit artifact in
  `artifacts/day2-qr-family-candidate-audit.md`.
- Audited rank-deficient rectangular candidates across `tests/test_qr.c`,
  `tests/test_qr_solve.c`, `tests/test_qr_helpers.h`,
  `tests/test_qr_corpus.c`, `tests/qr_external_dense_reference.py`, and the
  current QR corpus rows.
- Audited underdetermined minimum-norm candidates across `tests/test_qr_solve.c`
  and `tests/test_colamd.c`, including exact 2x4, 3x6, 5x10, COLAMD,
  rank-deficient, zero-row, refinement, and pseudoinverse cross-check
  owner-local lanes.
- Audited reorder/COLAMD QR candidates across `tests/test_colamd.c` and
  `tests/test_qr.c`, separating residual/status evidence from fill,
  performance, and ordering-optimality non-claims.
- Scored candidate families:
  rank-deficient rectangular `16`, underdetermined minimum-norm `15`, and
  reorder/COLAMD QR `9`.
- Identified common metadata gaps: source-controlled fixture rows, generator
  rows, expected rows, family claim scopes, tolerances, oracle rows, report
  rows, and focused proof-owner tests.
- Day 3 handoff: select two or three families, likely rank-deficient
  rectangular plus underdetermined minimum-norm, with reorder/COLAMD only if
  its claim can remain status/residual scoped.

### Day 3: Family Selection

- Created the family-selection and claim-scope artifact in
  `artifacts/day3-family-selection-claim-scope.md`.
- Selected two QR families for Sprint 150 complete closure:
  rank-deficient rectangular QR and underdetermined minimum-norm QR.
- Deferred reorder/COLAMD QR because its current evidence mixes residual,
  status, permutation, fill, optional SuiteSparse, and performance-adjacent
  semantics that would require a separate narrow product decision.
- Defined rank-deficient rectangular claim scope around fixture shape, `nnz`,
  rank, nullity, QR factorization success, residual, and subspace-safe
  projector/residual comparisons.
- Defined underdetermined minimum-norm claim scope around fixture shape, RHS
  policy, solve success, residual, solution norm, and exact solution entries
  only where source-controlled expected rows explicitly own them.
- Preserved non-claims for raw QR basis equality, Q-sign/orientation/column
  order parity, global rank-threshold policy, broad QR correctness, broad
  minimum-norm behavior, SVD-pseudoinverse-as-global oracle, external-library
  parity, platform, package, ABI, performance, and state-of-the-art claims.
- Mapped selected families to Days 4-14 metadata, oracle, proof-owner, report,
  documentation, validation, and closeout owners.
- Day 4 handoff: design concrete fixture, generator, expected-result,
  tolerance, claim-scope, and non-claim rows for selected rank-deficient
  rectangular and underdetermined minimum-norm fixtures, keeping the set small
  enough for complete closure.

### Day 4: Metadata Design

- Created the fixture metadata design artifact in
  `artifacts/day4-fixture-metadata-design.md`.
- Confirmed the selected Sprint 150 rows fit the existing fixture, generator,
  expected-result, and report-family schemas without adding new columns.
- Chose the Day 5 metadata batch:
  `qr_rankdef_duplicate_5x4_v1`,
  `qr_rankdef_dependent_row_4x3_v1`,
  `qr_underdetermined_minnorm_2x4`,
  `qr_minnorm_3x6_exact_values`, and
  `qr_minnorm_5x10_exact_values`.
- Kept `qr_rank_deficient_6x4_nullspace_v1` as the existing QR maintained
  corpus seed row.
- Deferred `qr_rankdef_wide_3x5_nullspace_subspace_v1`,
  `qr_minnorm_rankdef_2x4`, `qr_minnorm_zero_row_2x4`, and reorder/COLAMD QR
  rows so Sprint 150 can fully close the selected families before expanding the
  claim surface.
- Designed fixture-row fields, generator-row fields, expected-result row
  suffixes, tolerances, claim scopes, and non-claims for both selected
  families.
- Confirmed projector-oriented rows can use existing
  `comparison_kind=subspace_distance`, `expected_result_kind=subspace_distance`,
  and `tolerance_kind=projector` schema values.
- Day 5 handoff: implement deterministic generator builders/hashes, add
  fixture and generator manifest rows, add expected-result TSV files, and run
  `python3 scripts/validate_corpus_schema.py` before moving to oracle
  semantics.

### Day 5: Metadata Batch

- Created the fixture metadata batch artifact in
  `artifacts/day5-fixture-metadata-batch.md`.
- Added deterministic generator builders for the selected Sprint 150 QR
  fixtures to `scripts/validate_corpus_schema.py`.
- Added fixture rows for `qr_rankdef_duplicate_5x4_v1`,
  `qr_rankdef_dependent_row_4x3_v1`,
  `qr_underdetermined_minnorm_2x4`,
  `qr_minnorm_3x6_exact_values`, and
  `qr_minnorm_5x10_exact_values`.
- Added generator rows and canonical structure/value hashes for the five new
  generated QR fixtures.
- Added expected-result skeleton TSV files for rank/nullity, nullspace
  residual, subspace distance, minimum-norm status, residual, solution norm,
  and exact solution entries as applicable.
- Kept the rows at `support_tier=local_only` and preserved non-claims for raw
  QR basis equality, broad QR correctness, external-library parity, platform,
  package, ABI, performance, and state-of-the-art claims.
- Confirmed Day 5 did not require report-family row changes; generated-local
  report integration remains owned by Days 10-11 after oracle semantics and
  proof-owner tests exist.
- Ran `python3 scripts/validate_corpus_schema.py`; it passed.
- Day 6 handoff: define oracle semantics for rank/nullity, normalized
  nullspace residual, projector/subspace distance, minimum-norm status,
  residual, solution norm, exact-value comparisons, and downgrade rules for
  brittle exact-value rows.

### Day 6: Oracle Semantics Design

- Created the QR oracle semantics design artifact in
  `artifacts/day6-oracle-semantics-design.md`.
- Reviewed `scripts/run_corpus_oracle.py` and confirmed the current executable
  QR oracle path is hard-coded to `qr_rank_deficient_6x4_nullspace_v1`.
- Identified that the existing generic `value` comparison helper is currently
  partial-SVD `top_k` oriented, so Day 7 must extend it for QR scalar and
  vector value rows.
- Defined rank-deficient rectangular oracle semantics for exact rank, exact
  nullity, normalized nullspace residual, and projector/subspace distance.
- Defined underdetermined minimum-norm oracle semantics for solve status,
  residual norm, solution norm, and exact solution values.
- Defined observed-result encodings, expected-result normalization rules,
  failure classes, tolerance rationale, and downgrade rules for brittle exact
  solution-value rows.
- Rejected raw Q/R basis equality, raw nullspace basis equality,
  sign/orientation/scale/column-order parity, global rank-threshold policy,
  broad QR correctness, broad minimum-norm behavior, external-library parity,
  platform, package, ABI, performance, and state-of-the-art claims.
- Day 7 handoff: generalize the QR solver probe, add fixture-key mappings,
  extend `compare()` for QR value rows, normalize minimum-norm expected rows to
  key/value encodings, and run schema/oracle validation.

### Day 7: Oracle Data Implementation

- Created the oracle data implementation artifact in
  `artifacts/day7-oracle-data-implementation.md`.
- Extended `scripts/run_corpus_oracle.py --include-solver-qr` with fixture
  tables for the five Sprint 150 QR fixtures.
- Added generalized temporary C probes for rank-deficient rectangular
  rank/nullity/nullspace/projector observations and underdetermined
  minimum-norm status/residual/norm/vector observations.
- Extended the `value` comparator to support QR `solution_norm` and
  `solution_values` rows while preserving the existing partial-SVD `top_k`
  comparison path.
- Normalized minimum-norm expected rows to key/value encodings.
- Ran `python3 scripts/run_corpus_oracle.py --include-solver-qr`; it generated
  `26` oracle rows with `26` pass statuses, including `23` solver-backed QR
  rows covering the existing seed and five Sprint 150 fixtures.
- Ran oracle report-index checks:
  `python3 scripts/normalize_report_index.py --family oracle --require-generated oracle --check-freshness --check`
  and
  `python3 scripts/normalize_report_index.py --family corpus --family oracle --check`;
  both passed.
- Preserved local-only support tier and non-claims for broad QR correctness,
  raw basis identity, sign/orientation/column-order parity, external-library
  parity, platform/package/ABI, performance, and state-of-the-art claims.
- Day 8 handoff: design focused QR corpus proof-owner tests that mirror the
  executable oracle semantics and produce fixture-key-oriented diagnostics.

### Day 8: Proof-Owner Test Design

- Created the proof-owner test design artifact in
  `artifacts/day8-proof-owner-test-design.md`.
- Confirmed `tests/test_qr_corpus.c` is already the focused QR corpus proof
  owner and is registered in both Make and CMake.
- Designed Day 9 as an extension of `tests/test_qr_corpus.c`, not a new test
  binary or CI lane.
- Defined rank-deficient rectangular proof assertions for shape, `nnz`, rank,
  nullity, normalized nullspace residual, projector distance, and deterministic
  reference-null-vector residual.
- Defined underdetermined minimum-norm proof assertions for shape, `nnz`,
  `sparse_qr_solve_minnorm()` status, residual, solution norm, and exact
  solution values.
- Defined helper functions, fixture builders, fixture-key-oriented diagnostics,
  and Day 9 test case names.
- Preserved non-claims for broad QR correctness, raw basis identity,
  sign/orientation/column-order parity, global rank-threshold policy, broad
  minimum-norm behavior, external-library parity, platform/package/ABI,
  performance, and state-of-the-art claims.
- Day 9 handoff: implement the focused C tests, run `make build/test_qr_corpus`
  and `./build/test_qr_corpus`, rerun corpus/oracle/report checks, and because
  C changes will be made, run `make format && make lint && make test`.

### Day 9: Proof-Owner Test Implementation

- Created the proof-owner implementation artifact in
  `artifacts/day9-proof-owner-implementation.md`.
- Extended `tests/test_qr_corpus.c` rather than adding a new target, preserving
  the focused QR corpus proof-owner surface selected on Day 8.
- Added helper coverage for fixture-key-oriented shape checks, rank/nullity,
  normalized nullspace residuals, projector/subspace distance, minimum-norm
  residuals, solution norms, and exact deterministic solution values.
- Added executable proof coverage for `qr_rankdef_duplicate_5x4_v1`,
  `qr_rankdef_dependent_row_4x3_v1`,
  `qr_underdetermined_minnorm_2x4`,
  `qr_minnorm_3x6_exact_values`, and
  `qr_minnorm_5x10_exact_values`.
- Preserved the existing `qr_rank_deficient_6x4_nullspace_v1` seed tests for
  Sprint 139 continuity.
- Confirmed no Make/CMake registration change was required because
  `test_qr_corpus` is already registered in both build surfaces.
- Ran focused validation:
  `make build/test_qr_corpus && ./build/test_qr_corpus`; it passed with
  14 tests, 0 failures, and 258 assertions.
- Ran corpus/oracle/report validation:
  `python3 scripts/validate_corpus_schema.py`,
  `python3 scripts/run_corpus_oracle.py --include-solver-qr`,
  `python3 scripts/normalize_report_index.py --family corpus --family oracle --check`,
  and
  `python3 scripts/normalize_report_index.py --family oracle --require-generated oracle --check-freshness --check`;
  all passed, with the freshness check retaining existing advisory generated
  row warnings.
- Because Day 9 modified `tests/test_qr_corpus.c`, ran the required full C
  gate: `make format`, `make lint`, and `make test`; all passed.
- Preserved non-claims for broad QR correctness, raw basis identity,
  sign/orientation/column-order parity, global rank-threshold policy, broad
  minimum-norm behavior, external-library parity, platform/package/ABI,
  performance, and state-of-the-art claims.
- Day 10 handoff: design report/index rows around the bounded proof-owner
  tests and generated-local oracle outputs without treating generated-local
  report rows as source-controlled freshness claims.

### Day 10: Report Integration Design

- Created the report integration design artifact in
  `artifacts/day10-report-integration-design.md`.
- Inspected the existing report-family contracts in
  `tests/corpus/manifests/report_families.tsv` and confirmed the current
  `corpus` and `oracle` families can express the Sprint 150 QR rows without a
  new report family.
- Inspected `scripts/normalize_report_index.py` and confirmed generated-local
  oracle rows are derived from `build/corpus/oracle/*.tsv`, split into
  generated-reference and solver-backed rows by `solver_family`.
- Inspected current local generated outputs under `build/corpus/oracle/` and
  `build/corpus-reports/` as evidence shape only; no generated `build/`
  artifact is source-controlled by Day 10.
- Designed the Day 11 generated-output expectations for the selected QR
  family: six fixture keys, `23` solver-backed QR rows, source-controlled
  fixture/generator/expected owners, and generated-local oracle/report rows.
- Defined normalized report-index expectations for `corpus_fixture_*`,
  `corpus_generator_*`, `corpus_expected_*`, and generated-local `oracle_*`
  rows.
- Defined freshness rules that keep generated-local rows tied to their recorded
  command, commit, branch, platform, compiler, configuration, support tier, and
  artifact path.
- Preserved non-claims for broad QR correctness, raw basis identity,
  sign/orientation/column-order parity, global rank-threshold policy, broad
  rank-deficient solve behavior, broad minimum-norm behavior,
  external-library parity, platform/package/ABI, performance, and
  state-of-the-art claims.
- Mapped Day 12 documentation surfaces to the Day 11 implementation evidence:
  `README.md`, `docs/maintainer_guide.md`, corpus/sprint artifacts,
  solver-selection documentation, and the Sprint 150 retrospective.
- Day 11 handoff: run the local QR oracle command, confirm the report manifest
  names the selected fixtures and `solver_qr_row_count=23`, run the corpus and
  oracle report-index checks, and update source-controlled report contracts or
  generator behavior only if selected QR rows are missing or incorrectly
  scoped.

### Day 11: Report Integration Implementation

- Created the report integration implementation artifact in
  `artifacts/day11-report-integration-implementation.md`.
- Ran `python3 scripts/run_corpus_oracle.py --include-solver-qr`; it generated
  the QR oracle file, report index, skip report, and manifest under `build/`.
- Found that `scripts/normalize_report_index.py` reads
  `build/corpus/oracle/*.tsv`, so stale ignored oracle files from older local
  runs could duplicate QR rows or surface stale partial-SVD rows during
  normalization.
- Updated `scripts/run_corpus_oracle.py` to reset generated oracle/report
  outputs before writing the current run: prior oracle TSVs, report index, skip
  report, and manifest are removed from the generated-local output surface.
- Re-ran `python3 scripts/run_corpus_oracle.py --include-solver-qr` after the
  cleanup fix and confirmed the generated oracle directory contains only
  `build/corpus/oracle/qr_rank_deficient_6x4_nullspace_v1.oracle.tsv`.
- Confirmed the generated manifest reports six selected QR fixture keys,
  `oracle_row_count=26`, `solver_qr_row_count=23`, `partial_svd_row_count=0`,
  and `support_tier=local_only`.
- Ran `python3 -m py_compile scripts/run_corpus_oracle.py`; it passed.
- Ran `python3 scripts/normalize_report_index.py --family corpus --family oracle --check`;
  it passed with `78` rows.
- Ran
  `python3 scripts/normalize_report_index.py --family oracle --require-generated oracle --check-freshness --check`;
  it passed with `28` rows and the expected advisory
  `generated_present_unchecked` warnings for generated-local oracle rows.
- Preserved source-controlled ownership in the corpus manifests, expected rows,
  report-family contracts, proof-owner test, oracle script, and report-index
  normalizer; no generated `build/` report output was added to source control.
- No `.c` or `.h` files changed on Day 11, so the full C gate was not required
  for this day.
- Preserved non-claims for broad QR correctness, raw basis identity,
  sign/orientation/column-order parity, global rank-threshold policy, broad
  rank-deficient solve behavior, broad minimum-norm behavior,
  external-library parity, platform/package/ABI, performance, and
  state-of-the-art claims.
- Day 12 handoff: update user-facing and maintainer-facing documentation with
  the selected QR fixture keys, `23` solver-backed QR generated-local rows,
  normalized report-index validation, and generated-local freshness boundary.

### Day 12: Documentation Alignment

- Created the documentation alignment artifact in
  `artifacts/day12-documentation-alignment.md`.
- Updated `README.md` to describe the maintained Sprint 139/Sprint 150 QR
  corpus family instead of a single QR seed fixture.
- Updated `docs/cookbook.md` and `docs/algorithm.md` so QR workflow and
  algorithm guidance names the bounded six-fixture family, proof owner, oracle
  command, and local `23` solver-backed QR rows.
- Updated `docs/maintainer_guide.md` to rename the QR corpus section,
  list the selected Sprint 150 fixture keys, update focused proof expectations
  to `14` passing `test_qr_corpus` tests, and update report interpretation to
  `26` oracle rows, `23` solver-backed QR rows, and
  `partial_svd_row_count=0` for QR-only runs.
- Updated `tests/corpus/README.md` with the selected rank-deficient
  rectangular and underdetermined minimum-norm fixture families, generated
  output expectations, stale-report signals, and `solver_qr_row_count=23`.
- Updated `tests/corpus/schemas/oracle_fields.md` with the Sprint 150 expected
  row families and the maintained QR oracle command
  `python3 scripts/run_corpus_oracle.py --include-solver-qr`.
- Ran focused stale-claim searches for old QR row-count wording. Remaining
  `solver_qr_row_count=3` hits are historical Sprint 139 planning artifacts or
  the Sprint 150 Day 1 baseline artifact, not current guidance.
- Preserved non-claims for broad QR correctness, raw basis identity,
  sign/orientation/column-order parity, global rank-threshold policy, broad
  rank-deficient solve behavior, broad minimum-norm behavior,
  external-library parity, platform/package/ABI, performance, and
  state-of-the-art claims.
- No `.c` or `.h` files changed on Day 12, so the full C gate was not required
  for this day.
- Day 13 handoff: run integrated schema, focused QR proof-owner,
  oracle/report, documentation, and quality-gate checks; because Sprint 150 has
  earlier C changes, Day 13 should decide whether to rerun the full C gate as
  part of integrated validation.

### Day 13: Integrated Validation

- Created the integrated validation artifact in
  `artifacts/day13-integrated-validation.md`.
- Ran `python3 scripts/validate_corpus_schema.py`; it passed and reported
  `tests/corpus ok`.
- Ran focused QR proof-owner validation with
  `make build/test_qr_corpus && ./build/test_qr_corpus`; it passed with
  14 tests, 0 failures, 0 skips, and 258 assertions.
- Confirmed focused QR proof details for the selected six-fixture family:
  the Sprint 139 seed fixture plus two rank-deficient rectangular fixtures and
  three underdetermined minimum-norm fixtures.
- Ran `python3 scripts/run_corpus_oracle.py --include-solver-qr`; it
  regenerated the QR oracle TSV, report index, skip report, and manifest under
  `build/`.
- Ran `python3 -m py_compile scripts/run_corpus_oracle.py scripts/validate_corpus_schema.py scripts/normalize_report_index.py`;
  it passed.
- Ran `python3 scripts/normalize_report_index.py --family corpus --family oracle --check`;
  it passed with `78` rows.
- Ran
  `python3 scripts/normalize_report_index.py --family oracle --require-generated oracle --check-freshness --check`;
  it passed with `28` rows and the expected advisory
  `generated_present_unchecked` warnings for generated-local oracle rows.
- Ran focused stale current-doc searches for old QR row-count wording,
  Sprint 139-only current guidance, and obsolete proof-owner counts; no
  current-doc hits were found.
- Ran trailing-whitespace scans and `git diff --check`; both passed.
- Because Sprint 150 includes earlier `.c` changes, ran the full required C
  quality gate: `make format && make lint && make test`; it passed.
- Removed generated Python bytecode cache files after validation.
- Preserved non-claims for broad QR correctness, raw basis identity,
  sign/orientation/column-order parity, global rank-threshold policy, broad
  rank-deficient solve behavior, broad minimum-norm behavior,
  external-library parity, platform/package/ABI, performance, and
  state-of-the-art claims.
- Day 14 handoff: reconcile the sprint plan, working notes, and artifacts;
  document deferred reorder/COLAMD QR work; and prepare the sprint closeout
  around the completed six-fixture maintained QR corpus family.

### Day 14: Closeout And Sprint 151 Handoff

- Created the closeout and Sprint 151 handoff artifact in
  `artifacts/day14-closeout-handoff.md`.
- Reconciled the sprint closure against the Day 1-13 artifacts and confirmed
  Sprint 150 closed the selected six-fixture QR corpus family:
  `qr_rank_deficient_6x4_nullspace_v1`,
  `qr_rankdef_duplicate_5x4_v1`,
  `qr_rankdef_dependent_row_4x3_v1`,
  `qr_underdetermined_minnorm_2x4`,
  `qr_minnorm_3x6_exact_values`, and
  `qr_minnorm_5x10_exact_values`.
- Recorded the Day 13 integrated validation as the primary full-gate evidence:
  corpus schema, focused QR proof-owner tests, QR oracle generation, script
  compile checks, report normalization, oracle freshness, stale-claim scans,
  whitespace/diff checks, and `make format && make lint && make test` all
  passed.
- Ran Day 14 closeout checks:
  `python3 scripts/validate_corpus_schema.py`,
  `python3 scripts/normalize_report_index.py --family corpus --family oracle --check`,
  `python3 scripts/normalize_report_index.py --family oracle --require-generated oracle --check-freshness --check`,
  trailing-whitespace scans, and `git diff --check`; all passed.
- Confirmed the oracle freshness check still emits expected
  `generated_present_unchecked` warnings for generated-local rows while
  reporting freshness ok for `28` rows.
- Ran stale-reference searches for old QR row-count and proof-owner wording.
  Hits remain only in historical Sprint 150 baseline/validation artifacts that
  explicitly describe the old count as historical; current docs are aligned to
  the six-fixture family and `solver_qr_row_count=23`.
- Recorded Sprint 150 residuals: reorder/COLAMD QR corpus promotion, strict
  generated-row freshness comparison, broad rank-threshold policy, and any
  downstream package/platform/ABI/installed-consumer proof remain out of scope.
- Prepared the Sprint 151 partial-SVD handoff around family selection,
  subspace-safe comparison contracts, metadata, focused proof-owner tests,
  oracle/report integration, documentation alignment, and validation.
- No `.c` or `.h` files were modified on Day 14 itself, so the Day 13 full C
  gate remains the required quality-gate evidence for the existing Sprint 150
  code changes.
