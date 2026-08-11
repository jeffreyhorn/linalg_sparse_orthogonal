# Sprint 151 Working Notes

## Goal

Sprint 151 expands partial-SVD maintained corpus coverage beyond the single
Sprint 140 fixture while keeping the claim bounded to selected fixture
families, subspace-safe comparison semantics, focused proof-owner tests, and
generated-local report evidence.

## Starting Evidence

- Sprint 147 defines the corpus-family evidence gate: source-controlled
  fixture rows, generator rows, expected-result rows, proof-owner tests,
  oracle/report rows, validation commands, and bounded documentation are
  prerequisites for any promoted corpus-family claim.
- Sprint 140 closes one partial-SVD fixture-local claim for
  `partial_svd_clustered_repeated_diag8x6_k3_v1`: top-3 singular values,
  left/right top-k subspace projectors, triplet residuals, orthogonality,
  default-budget success, tight-budget fail-closed behavior, and no partial
  arrays on tight-budget failure.
- Sprint 150 leaves a corpus expansion pattern for Sprint 151: select a small
  closable family, define comparison semantics before rows, add focused proof
  owners, reset generated-local oracle/report outputs before writing current
  output, and keep current-document searches separate from historical planning
  artifacts.
- Current source-controlled partial-SVD corpus ownership consists of one
  fixture row, one generator row, one expected-result file, the focused
  `tests/test_svd_partial_corpus.c` proof owner, helpers in
  `tests/test_svd_partial_shared_helpers.h`, and the opt-in
  `scripts/run_corpus_oracle.py --include-partial-svd` path.
- Existing non-corpus SVD tests already cover many owner-local candidate lanes:
  external singular-value fixtures, vector-residual fixtures,
  rank-deficient range-projector behavior, low-rank Frobenius behavior,
  sparse low-rank output, tight-budget fail-closed behavior, full-k paths,
  nonsymmetric rectangular paths, and SuiteSparse smoke checks.

## Item-To-Day Owner Map

| Sprint 151 Item | Primary Days | Closeout Owner |
| --- | --- | --- |
| Item 1: Partial-SVD Family Selection | Days 1-3 | Day 1 records the baseline, Day 2 audits candidate families, Day 3 selects the families and claim scopes. |
| Item 2: Comparison Contract | Days 4-5 | Day 4 defines comparison semantics, Day 5 maps them into metadata design. |
| Item 3: Fixture Metadata Batch | Days 5-7 | Day 5 designs rows, Day 6 implements metadata, Day 7 implements expected data and oracle inputs. |
| Item 4: Proof-Owner Tests | Days 8-9 | Day 8 designs focused proof owners, Day 9 implements and validates tests. |
| Item 5: Report Integration | Days 10-11 | Day 10 designs report rows, Day 11 implements oracle/report integration. |
| Item 6: Documentation Alignment | Day 12 | Day 12 aligns SVD, corpus, README, cookbook, and maintainer guidance. |
| Item 7: Validation | Days 13-14 | Day 13 runs integrated validation; Day 14 closes with Sprint 152 handoff. |

## Stop Conditions

- A proposed expected row requires raw singular-vector identity, sign,
  orientation, phase, arbitrary basis order, or raw vector equality for a
  degenerate or repeated singular subspace.
- A selected family cannot name fixture keys, generators, expected rows,
  tolerances, support tier, proof owner, validation command, and non-claims.
- A generated-reference oracle row is cited as solver-backed hosted-platform
  pass evidence.
- Generated oracle/report rows omit command, commit, branch, platform,
  compiler/configuration, support tier, artifact path, or non-claims.
- Optional-data skip/defer rows are counted as partial-SVD pass evidence.
- Documentation widens selected fixture-family evidence into broad SVD,
  partial-SVD, external-library, platform, package, ABI, performance, or
  state-of-the-art claims.
- Sprint 150 QR corpus evidence is reused as SVD or partial-SVD evidence.
- Required focused tests, corpus schema checks, oracle/report checks, or full C
  quality gates fail after implementation changes.

## Daily Log

### Day 1: SVD Intake

- Re-read the Sprint 151 section of
  `docs/planning/EPIC_13/PROJECT_PLAN.md`.
- Reviewed Sprint 140 partial-SVD retrospective and closeout context, Sprint
  147 corpus evidence gate, and Sprint 150 QR corpus/report handoff.
- Created the Sprint 151 artifact directory and Day 1 partial-SVD intake
  artifact.
- Inventoried partial-SVD implementation files:
  `include/sparse_svd.h`, `src/sparse_svd.c`,
  `src/sparse_svd_partial.c`, and `src/sparse_svd_internal.h`.
- Inventoried partial-SVD tests and proof owners:
  `tests/test_svd_partial_corpus.c`, `tests/test_svd.c`,
  `tests/test_svd_partial_helpers.h`,
  `tests/test_svd_partial_shared_helpers.h`,
  `tests/test_svd_helpers.h`, and `tests/svd_external_dense_reference.py`.
- Inventoried current source-controlled partial-SVD corpus rows:
  `tests/corpus/manifests/fixtures.tsv`,
  `tests/corpus/manifests/generators.tsv`,
  `tests/corpus/expected/partial_svd_clustered_repeated_diag8x6_k3_v1.tsv`,
  and `tests/corpus/manifests/report_families.tsv`.
- Confirmed the only current maintained partial-SVD corpus fixture is
  `partial_svd_clustered_repeated_diag8x6_k3_v1`; many other partial-SVD
  fixtures live as owner-local test evidence but not yet source-controlled
  corpus families.
- Captured Day 1 stop conditions for raw-vector identity claims, unowned
  fixtures, stale generated reports, optional-data pass confusion, and
  unsupported platform/package/performance inference.
- Day 2 handoff: audit candidate partial-SVD families for repeated spectra,
  rank-deficient rectangular projectors, sparse low-rank output, and
  convergence/fail-closed closure value, metadata needs, oracle readiness, and
  implementation risk.

### Day 2: Family Audit

- Created the partial-SVD family candidate audit artifact in
  `artifacts/day2-partial-svd-family-candidate-audit.md`.
- Audited rank-deficient rectangular range-projector evidence in
  `tests/test_svd_partial_helpers.h`, especially
  `test_partial_svd_rankdef_diag6x4_k2_range_projector`.
- Audited sparse low-rank output evidence across `tests/test_svd.c`,
  `example_svd_lowrank.c`, and low-rank helper assertions.
- Audited convergence and fail-closed evidence in
  `test_partial_svd_max_iter_fail_closed_diag6_k2` and compared it with the
  Sprint 140 tight-budget corpus lane.
- Audited optional external dense-reference vector-residual fixtures for
  square, tall, and nonsymmetric rectangular partial-SVD cases.
- Audited repeated-spectrum follow-through and found Sprint 140 already owns
  the strongest current repeated/clustered corpus seed.
- Scored candidate families:
  rank-deficient rectangular range projector `17`, sparse low-rank output
  `15`, convergence/fail-closed behavior `14`, external dense-reference vector
  residuals `12`, and repeated spectra beyond Sprint 140 `11`.
- Identified common metadata gaps: fixture rows, generator rows, expected
  rows, support tiers, claim scopes, non-claims, oracle rows, report rows, and
  focused proof-owner mappings.
- Day 3 handoff: likely select rank-deficient rectangular range-projector
  closure, consider one bounded sparse low-rank output fixture, and add a
  fail-closed convergence fixture only if its claim is distinct from Sprint
  140. Defer external reference and repeated-spectrum expansion unless Day 3
  finds a narrow, non-duplicative claim.

### Day 3: Family Selection

- Created the family-selection and claim-scope artifact in
  `artifacts/day3-family-selection-claim-scope.md`.
- Selected three deterministic partial-SVD families for Sprint 151 complete
  closure:
  `partial_svd_rankdef_diag6x4_k2_range_projector_v1`,
  `partial_svd_lowrank_rect5x7_k3_sparse_output_v1`, and
  `partial_svd_fail_closed_diag6_k2_v1`.
- Defined rank-deficient rectangular range-projector scope around default
  success, top-2 singular values, rank, left/right projectors, triplet
  residuals, and U/V orthogonality.
- Defined sparse low-rank output scope around status, output shape, selected
  values, dense low-rank Frobenius error, and sparse-vs-dense low-rank
  Frobenius difference at `drop_tol=0`.
- Defined non-repeated convergence fail-closed scope around tight-budget
  `SPARSE_ERR_NOT_CONVERGED`, no partial arrays on failure, default-budget
  recovery, top-2 singular values, and default triplet residuals.
- Deferred optional external dense-reference vector-residual fixtures because
  optional data, provenance, Windows skip behavior, and broad parity wording
  would widen the sprint surface.
- Deferred repeated-spectrum expansion beyond Sprint 140 because the existing
  Sprint 140 clustered/repeated fixture already closes the strongest current
  repeated-spectrum corpus seed.
- Preserved non-claims for raw singular-vector identity, sign/orientation/phase
  parity, arbitrary basis ordering, broad partial-SVD correctness, broad
  rank-deficient/null-space behavior, broad sparse-output/drop-tolerance
  optimality, convergence rates, portable iteration counts, external-library
  parity, platform/package/ABI, performance, and state-of-the-art claims.
- Defined rollback rules for unstable tolerances, raw-vector requirements,
  sparse-output brittleness, convergence instability, generator hash drift,
  validation failures, and documentation overclaiming.
- Day 4 handoff: define exact comparison contracts and expected-result
  encodings for the selected singular-value, rank, projector, residual,
  sparse-output, Frobenius, status, and diagnostic rows.

### Day 4: Contract Design

- Created the comparison-contract artifact in
  `artifacts/day4-comparison-contract-design.md`.
- Reviewed existing oracle comparison semantics in
  `scripts/run_corpus_oracle.py` and the source-controlled oracle field schema.
- Confirmed current corpus conventions for exact rank rows, sorted `top_k`
  singular-value rows, projector-distance rows, residual-norm rows,
  status-only rows, and diagnostic rows.
- Defined the row-level contract for
  `partial_svd_rankdef_diag6x4_k2_range_projector_v1`: default success,
  sorted top-2 singular values, exact rank, left/right projector distances,
  triplet residuals, and U/V orthogonality.
- Defined the row-level contract for
  `partial_svd_lowrank_rect5x7_k3_sparse_output_v1`: sparse-output success,
  exact shape diagnostic, exact retained nonzero count, selected coordinate
  values, dense low-rank Frobenius error, and sparse-vs-dense Frobenius
  difference at `drop_tol=0`.
- Defined the row-level contract for
  `partial_svd_fail_closed_diag6_k2_v1`: tight-budget non-convergence,
  no partial arrays on failure, default-budget recovery, default singular
  values, and default triplet residuals.
- Explicitly rejected raw singular-vector identity, sign parity, orientation
  parity, phase parity, arbitrary basis-order parity, broad sparse-output
  optimality, convergence-rate claims, portable iteration-count claims, and
  partial-result guarantees after non-convergence.
- Identified one required oracle extension for later sprint work:
  `comparison_kind=value` should parse `selected_values` as a comma-separated
  numeric vector, parallel to existing `solution_values` handling.
- Day 5 handoff: translate the Day 4 comparison contract into fixture,
  generator, expected-row, proof-owner, and oracle metadata design without
  widening the selected fixture-family claims.

### Day 5: Metadata Design

- Created the metadata-design artifact in
  `artifacts/day5-metadata-design.md`.
- Reviewed current maintained corpus manifest schemas and confirmed the
  selected Sprint 151 rows fit existing fixture, generator, expected-result,
  and report-family columns.
- Designed fixture rows for
  `partial_svd_rankdef_diag6x4_k2_range_projector_v1`,
  `partial_svd_lowrank_rect5x7_k3_sparse_output_v1`, and
  `partial_svd_fail_closed_diag6_k2_v1`.
- Designed deterministic generator rows for the three selected diagonal
  fixtures, including stable algorithm names, ordered parameter strings,
  canonical format, floating policies, regeneration commands, and generator
  change policy.
- Designed expected-result row IDs, operations, comparison kinds,
  expected-result kinds, expected values, tolerance kinds, and tolerance
  values for rank-deficient, sparse low-rank output, and fail-closed families.
- Kept all planned rows at `support_tier=local_only` and preserved explicit
  non-claims for raw vector identity, sign/orientation/phase parity,
  arbitrary basis order, broad partial-SVD correctness, broad sparse-output
  optimality, convergence rates, portable iteration counts, external-library
  parity, platform/package/ABI, performance, and state-of-the-art claims.
- Confirmed no `tests/corpus/manifests/report_families.tsv` change is planned
  for Day 6 because existing `corpus/*` and `oracle/solver_backed` rows cover
  source-controlled metadata and generated-local partial-SVD oracle rows.
- Preserved the single known schema/comparator follow-up from Day 4:
  `selected_values` needs a narrow `comparison_kind=value` parser extension
  unless Day 6 replaces it with scalar supported rows.
- Day 6 handoff: add deterministic generator builders and hashes, fixture
  rows, generator rows, and expected-result TSV files, then run
  `python3 scripts/validate_corpus_schema.py`.

### Day 6: Metadata Batch

- Created the metadata-batch implementation artifact in
  `artifacts/day6-metadata-batch.md`.
- Added deterministic generator builders and registered generator contracts in
  `scripts/validate_corpus_schema.py` for:
  `partial_svd_rankdef_diag6x4_k2_range_projector_generator_v1`,
  `partial_svd_lowrank_rect5x7_k3_sparse_output_generator_v1`, and
  `partial_svd_fail_closed_diag6_k2_generator_v1`.
- Computed and populated canonical structure/value hashes for the three new
  partial-SVD generator rows.
- Added fixture rows to `tests/corpus/manifests/fixtures.tsv` for the selected
  rank-deficient projector, sparse low-rank output, and fail-closed
  convergence families.
- Added generator rows to `tests/corpus/manifests/generators.tsv` for the same
  selected families.
- Added expected-result TSV files for all three selected fixtures under
  `tests/corpus/expected/`.
- Kept all new source-controlled metadata rows at `support_tier=local_only`
  with fixture-local claim scopes and explicit non-claims.
- Ran `python3 scripts/validate_corpus_schema.py`; it passed.
- Ran
  `python3 scripts/normalize_report_index.py --family corpus --family oracle --check`;
  it passed with `102` normalized rows.
- No `.c` or `.h` files changed, so the C quality gate was not required.
- Day 7 handoff: update partial-SVD oracle inputs for the new fixtures,
  resolve the `selected_values` comparator follow-up or replace it with scalar
  supported rows, and keep generated oracle evidence local-only.

### Day 7: Oracle Data

- Created the oracle-data implementation artifact in
  `artifacts/day7-oracle-data-implementation.md`.
- Updated `scripts/run_corpus_oracle.py` so
  `python3 scripts/run_corpus_oracle.py --include-partial-svd` emits
  generated-reference rows for the existing Sprint 140 partial-SVD fixture and
  all three Sprint 151 partial-SVD fixtures.
- Added generated observations for the rank-deficient projector fixture:
  default success, sorted top-2 singular values, exact rank, left/right
  projector distances, triplet residuals, and orthogonality.
- Added generated observations for the sparse low-rank output fixture:
  sparse-output status, shape, retained nonzero count, selected coordinate
  values, dense Frobenius absolute error, and sparse-vs-dense Frobenius
  difference.
- Added generated observations for the fail-closed fixture: tight-budget
  non-convergence, no partial arrays on failure, recovery success, default
  singular values, and default triplet residuals.
- Extended `comparison_kind=value` parsing to support `selected_values` as a
  comma-separated numeric vector with optional `max_abs_error` validation.
- Ran `python3 scripts/run_corpus_oracle.py --include-partial-svd`; it wrote
  `29` total oracle rows, including `26` partial-SVD rows across four
  maintained partial-SVD fixtures, all with `comparison_status=pass`.
- Ran `python3 scripts/validate_corpus_schema.py`; it passed.
- Ran
  `python3 scripts/normalize_report_index.py --family corpus --family oracle --check`;
  it passed with `105` normalized rows.
- Ran `python3 scripts/normalize_report_index.py --family oracle --check-freshness`;
  it exited successfully with expected `generated_present_unchecked` warnings
  for generated-local rows.
- Ran `python3 -m py_compile scripts/run_corpus_oracle.py scripts/validate_corpus_schema.py`
  and `git diff --check`; both passed.
- No `.c` or `.h` files changed, so the C quality gate was not required.
- Day 8 handoff: design focused proof-owner tests mapping every selected
  expected row to executable assertions while preserving raw-vector,
  sign/orientation, broad sparse-output, and convergence-rate non-claims.

### Day 8: Test Design

- Created the proof-owner test design artifact in
  `artifacts/day8-proof-owner-test-design.md`.
- Inspected `tests/test_svd_partial_corpus.c` and confirmed it is the right
  focused proof-owner file for the Sprint 151 selected partial-SVD corpus
  fixtures.
- Inspected `tests/test_svd_partial_shared_helpers.h` and chose to reuse the
  existing residual and coordinate-range projector helpers rather than adding
  a broad fixture framework.
- Inspected existing broader owner-local evidence in
  `tests/test_svd_partial_helpers.h` for the rank-deficient projector and
  fail-closed diagonal fixtures.
- Inspected existing sparse low-rank dense/sparse consistency evidence in
  `tests/test_svd.c`.
- Designed focused Day 9 tests for the rank-deficient rectangular projector,
  sparse low-rank output, and non-repeated fail-closed convergence fixtures.
- Added a fixture-key diagnostic plan so failures identify the selected
  fixture and observed metrics without relying on raw singular-vector output.
- Mapped every selected Sprint 151 expected-result row to an executable
  assertion, including singular values, rank, projectors, residuals,
  orthogonality, sparse-output shape/values/Frobenius checks, and
  convergence/fail-closed status/diagnostics.
- Preserved non-claims for raw singular-vector identity,
  sign/orientation/phase parity, arbitrary basis order, broad partial-SVD
  correctness, broad sparse-output optimality, convergence rates, portable
  iteration counts, external-library parity, platform/package/ABI,
  performance, and state-of-the-art support.
- Day 9 handoff: extend `tests/test_svd_partial_corpus.c` with the focused
  proof-owner tests, run focused/affected tests and schema/oracle/report
  checks, then run `make format && make lint && make test` because C files
  will change.

### Day 9: Test Implementation

- Created the proof-owner test implementation artifact in
  `artifacts/day9-proof-owner-test-implementation.md`.
- Extended `tests/test_svd_partial_corpus.c` with focused corpus proof-owner
  tests for the rank-deficient rectangular projector, sparse low-rank output,
  and non-repeated fail-closed convergence fixtures.
- Added fixture-keyed diagnostics for the three selected Sprint 151 fixtures
  so failures identify the exact fixture and observed metric.
- Generalized the sorted top-k singular-value error helper in
  `tests/test_svd_partial_corpus.c`.
- Added executable proof coverage for all selected Sprint 151 expected-result
  rows: status, singular values, rank, projectors, residuals, orthogonality,
  sparse-output shape, sparse-output nnz, selected values, Frobenius error,
  sparse-vs-dense difference, fail-closed arrays, and recovery status.
- Ran `make build/test_svd_partial_corpus && ./build/test_svd_partial_corpus`;
  it passed with `10` tests and `247` assertions.
- Ran `make build/test_svd && ./build/test_svd`; it passed with `114` tests
  and `2067` assertions.
- Ran `python3 scripts/validate_corpus_schema.py`; it passed.
- Ran `python3 scripts/run_corpus_oracle.py --include-partial-svd`; it passed
  and refreshed generated local oracle/report outputs under ignored `build/`
  paths.
- Ran
  `python3 scripts/normalize_report_index.py --family corpus --family oracle --check`;
  it passed with `105` rows.
- Ran `make format && make lint && make test`; the full required C gate passed.

### Day 10: Report Integration Design

- Created the report integration design artifact in
  `artifacts/day10-report-integration-design.md`.
- Re-inspected `scripts/run_corpus_oracle.py`,
  `scripts/normalize_report_index.py`, `tests/test_normalize_report_index.py`,
  `tests/corpus/manifests/report_families.tsv`, and
  `tests/corpus/schemas/report_index_fields.md`.
- Confirmed the generated partial-SVD oracle command emits `29` oracle rows,
  including `26` partial-SVD rows across the Sprint 140 fixture and the three
  Sprint 151 fixtures.
- Confirmed the normalized corpus/oracle index currently writes `105` rows
  when generated-local oracle output is present.
- Defined the Day 11 normalized-row target matrix for the Sprint 151 fixtures:
  one source-controlled fixture row, one source-controlled generator row,
  source-controlled expected rows, and generated-local oracle rows per
  selected fixture.
- Recorded the required generated oracle row counts for Day 11 tests:
  `7` rank-deficient rectangular rows, `6` sparse low-rank output rows, and
  `5` fail-closed convergence rows.
- Defined freshness rules for absent, current-commit, and stale generated
  partial-SVD oracle rows, including strict-generated failure behavior.
- Preserved the report non-claims: generated-local rows remain local-only
  fixture evidence and do not claim broad partial-SVD correctness, raw
  singular-vector identity, external-library parity, hosted CI proof,
  platform/package/ABI support, performance, or state-of-the-art support.
- Ran Python/report validation:
  `python3 -m py_compile scripts/run_corpus_oracle.py scripts/normalize_report_index.py scripts/validate_corpus_schema.py tests/test_normalize_report_index.py`;
  `python3 tests/test_normalize_report_index.py`;
  `python3 scripts/validate_corpus_schema.py`;
  `python3 scripts/run_corpus_oracle.py --include-partial-svd`;
  `python3 scripts/normalize_report_index.py --family corpus --family oracle --check`;
  and `python3 scripts/normalize_report_index.py --family oracle --check-freshness`.
  All passed; freshness emitted the expected current strict-oracle warnings
  while exiting successfully with `31` oracle-family rows.
- No `.c` or `.h` files changed on Day 10, so the C quality gate was not
  required.

### Day 11: Report Integration Implementation

- Created the report integration implementation artifact in
  `artifacts/day11-report-integration-implementation.md`.
- Extended `tests/test_normalize_report_index.py` with explicit Sprint 151
  partial-SVD generated oracle row-count expectations:
  `7` rank-deficient rectangular rows, `6` sparse low-rank output rows, and
  `5` fail-closed convergence rows.
- Added normalized-row assertions that generated Sprint 151 partial-SVD oracle
  rows remain under `oracle/solver_backed`, `generated_local`, and
  `local_only` while preserving `solver_family=partial_svd`,
  `fixture_key=...`, `proof_owner=generated_partial_svd_reference`, and
  `solver_execution=none`.
- Added stale partial-SVD oracle freshness coverage: default oracle freshness
  warns on stale generated rows, while `--strict-generated --check-freshness`
  fails stale strict oracle evidence.
- Confirmed no `tests/corpus/manifests/report_families.tsv` update was needed;
  the existing `oracle/solver_backed` contract covers these generated-local
  partial-SVD rows with the correct local-only support tier.
- Ran Day 11 validation:
  `python3 tests/test_normalize_report_index.py`;
  `python3 -m py_compile scripts/run_corpus_oracle.py scripts/normalize_report_index.py scripts/validate_corpus_schema.py tests/test_normalize_report_index.py`;
  `python3 scripts/validate_corpus_schema.py`;
  `python3 scripts/run_corpus_oracle.py --include-partial-svd`;
  `python3 scripts/normalize_report_index.py --family corpus --family oracle --check`;
  `python3 scripts/normalize_report_index.py --family oracle --check-freshness`;
  `python3 scripts/normalize_report_index.py --family oracle --strict-generated --check-freshness`;
  and `git diff --check`. All passed.
- No `.c` or `.h` files changed on Day 11, so the C quality gate was not
  required.

### Day 12: Documentation Alignment

- Created the documentation alignment artifact in
  `artifacts/day12-documentation-alignment.md`.
- Updated `README.md`, `docs/solver_selection.md`, `docs/cookbook.md`,
  `docs/algorithm.md`, `docs/maintainer_guide.md`,
  `tests/corpus/README.md`, `tests/corpus/schemas/oracle_fields.md`, and
  `tests/corpus/expected/README.md`.
- Replaced active Sprint 140-only partial-SVD corpus wording with the maintained
  Sprint 140/Sprint 151 fixture set:
  `partial_svd_clustered_repeated_diag8x6_k3_v1`,
  `partial_svd_rankdef_diag6x4_k2_range_projector_v1`,
  `partial_svd_lowrank_rect5x7_k3_sparse_output_v1`, and
  `partial_svd_fail_closed_diag6_k2_v1`.
- Documented the current generated-local partial-SVD oracle row count of `26`
  and the per-fixture row counts `8`, `7`, `6`, and `5`.
- Added maintainer stale-report rules for partial-SVD oracle rows, including
  changed source files, mismatched command/commit/branch/platform/compiler/
  configuration, missing row counts, non-pass maintained rows, and strict
  freshness failures.
- Preserved non-claims across the updated docs: no broad partial-SVD
  correctness, raw singular-vector identity, broad sparse-output optimality,
  external-library parity, hosted CI proof, platform/package/ABI support,
  performance, or state-of-the-art support.
- No `.c` or `.h` files changed on Day 12, so the C quality gate was not
  required.

### Day 13: Integrated Validation

- Created the integrated validation artifact in
  `artifacts/day13-integrated-validation.md`.
- Ran `python3 scripts/validate_corpus_schema.py`; it passed.
- Ran `make build/test_svd_partial_corpus && ./build/test_svd_partial_corpus`;
  it passed with `10` focused partial-SVD corpus tests and `247` assertions.
- Ran `make build/test_svd && ./build/test_svd`; it passed with `114` tests
  and `2067` assertions.
- Ran `python3 tests/test_normalize_report_index.py`; it passed.
- Ran `python3 scripts/run_corpus_oracle.py --include-partial-svd`; it
  refreshed generated-local oracle/report outputs under ignored `build/`
  paths.
- Ran
  `python3 scripts/normalize_report_index.py --family corpus --family oracle --check`;
  it passed with `105` normalized rows.
- Ran `python3 scripts/normalize_report_index.py --family oracle --check-freshness`;
  it passed with `31` oracle-family rows and expected advisory
  `generated_present_unchecked` warnings for generated-local strict-oracle
  rows.
- Confirmed active documentation no longer contains stale Sprint 140-only
  partial-SVD wording or stale `partial_svd_row_count=8` claims.
- Ran the required full C quality gate because the branch includes C test
  changes: `make format && make lint && make test`; it passed.
- Removed transient Python cache output created by validation scripts.
- Day 14 handoff: use the Day 13 validation artifact as the Sprint 151
  closeout baseline and keep the maintained partial-SVD claim bounded to the
  four selected fixtures and `26` generated-local partial-SVD oracle rows.

### Day 14: Closeout And Sprint 152 Handoff

- Created the closeout and Sprint 152 handoff artifact in
  `artifacts/day14-closeout-and-sprint-152-handoff.md`.
- Finalized Sprint 151 completion inputs: selected families, comparison
  boundaries, fixture metadata, expected-result rows, proof-owner tests,
  oracle/report integration, documentation alignment, validation evidence,
  residuals, and generated-report handoff.
- Recorded the final maintained partial-SVD corpus shape: four fixtures and
  `26` generated-local partial-SVD oracle rows, with `29` total generated
  oracle rows when QR rows are included by
  `python3 scripts/run_corpus_oracle.py --include-partial-svd`.
- Preserved Sprint 151 claim boundaries: selected fixture-family evidence only;
  no raw singular-vector identity, sign/orientation/phase parity, broad
  partial-SVD correctness, broad sparse-output optimality, external-library
  parity, hosted CI proof, package/ABI support, performance, or
  state-of-the-art claim.
- Assigned the main residual to Sprint 152: decide generated report freshness
  publication policy for generated-local oracle rows, including whether any
  partial-SVD rows become required, strict, or hosted-CI claim evidence.
- Ran final closeout validation:
  `python3 scripts/validate_corpus_schema.py`;
  `python3 scripts/run_corpus_oracle.py --include-partial-svd`;
  `python3 scripts/normalize_report_index.py --family corpus --family oracle --check`;
  `python3 scripts/normalize_report_index.py --family oracle --check-freshness`;
  active-doc stale wording search;
  `git diff --check`;
  transient Python cache search;
  and `git status --short`.
- Full C gate status remains the Day 13 baseline:
  `make format && make lint && make test` passed after the branch's C test
  changes.
