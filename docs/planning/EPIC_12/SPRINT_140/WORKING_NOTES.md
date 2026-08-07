# Sprint 140 Working Notes

## Sprint Goal

Completely close the selected partial-SVD residual with deterministic
edge-case fixtures, comparison/oracle semantics, convergence-budget proof,
focused helper ownership, validation evidence, and bounded documentation.

## Initial Constraints

- Close one bounded partial-SVD residual completely rather than partially
  covering many residual families.
- Reuse the Sprint 138 corpus/oracle/report architecture and Sprint 139 QR
  fixture-local closure pattern, but define partial-SVD-specific fixture keys,
  expected rows, oracle rows, proof owner, tolerances, and non-claims.
- Use residual, projector, subspace, singular-value, status, or diagnostic
  comparisons where valid; do not rely on raw singular-vector identity when
  sign, scale, basis rotation, or repeated/clustered singular-value ambiguity
  is valid.
- Keep generated oracle/report outputs under ignored `build/` paths.
- Keep optional external data as skip/defer evidence until a reviewed
  support-tier gate promotes it.
- Preserve public non-claims for broad SVD/partial-SVD correctness,
  external-library parity, broad corpus completeness, platform parity,
  performance, package/ABI behavior, and state-of-the-art status.

## Inherited Evidence Inventory

| Surface | Current evidence | Sprint 140 use |
| --- | --- | --- |
| `docs/planning/EPIC_12/PROJECT_PLAN.md` | Defines Sprint 140 Items 1-7 and the 168-hour scope. | Sprint-level scope and estimate authority. |
| `docs/planning/EPIC_12/SPRINT_138/RETROSPECTIVE.md` | Leaves partial-SVD clustered/repeated singular-value fixture lanes to Sprint 140. | Confirms corpus architecture is available but partial-SVD closure is open. |
| `docs/planning/EPIC_12/SPRINT_139/RETROSPECTIVE.md` | Hands off partial-SVD-specific fixture, comparison, ambiguity, proof-owner, support-tier, and wording requirements. | Primary Sprint 140 readiness source. |
| `tests/corpus/` | Maintained manifest/schema/expected-result architecture currently contains the QR first lane. | Target home for new partial-SVD fixture metadata and expected rows. |
| `scripts/validate_corpus_schema.py` | Validates corpus TSVs, generator hashes, expected rows, and false-pass guardrails. | Must validate any new fixture/generator/expected-result rows. |
| `scripts/run_corpus_oracle.py` | Emits QR generated-reference and opt-in solver-backed QR rows plus report indexes and manifests. | Candidate pattern for partial-SVD oracle/report rows. |
| `tests/test_svd.c` | Broad SVD and partial-SVD owner with full SVD, low-rank, sparse low-rank, rank, pseudoinverse, and partial-SVD test registrations. | Current proof owner; may receive a focused extraction or a new adjacent owner. |
| `tests/test_svd_partial_helpers.h` | Large helper/test header covering partial-SVD singular values, vector residuals, range projectors, low-rank, and fail-closed max-iteration behavior. | Main source of candidate fixtures and helper ownership risk. |
| `tests/svd_external_dense_reference.py` | Bounded dense-reference helper for named SVD/partial-SVD fixtures. | Possible reference pattern; not broad NumPy/SciPy parity. |
| `include/sparse_svd.h` | Public SVD and partial-SVD API, including `max_iter`, `tol`, `compute_uv`, and low-rank surfaces. | Public contract boundary; changes require careful claim and quality gates. |
| `src/sparse_svd.c` and `src/sparse_svd_partial.c` | Full and partial SVD implementation surfaces. | Implementation surfaces only if residual closure reveals a solver defect. |
| `README.md`, `docs/algorithm.md`, `docs/cookbook.md`, `docs/solver_selection.md`, `docs/maintainer_guide.md`, `examples/example_svd_lowrank.c` | Current public and maintainer SVD wording and examples. | Update only after fixture/oracle/proof evidence exists. |

## Current Partial-SVD Evidence

| Evidence family | Present today | Gap for Sprint 140 |
| --- | --- | --- |
| External dense references | `partial_svd_diag6_k2`, `partial_svd_tall_diag_8x5_k3`, and `partial_svd_nonsym_rect10x8_k3` singular-value/vector-residual lanes. | Not maintained corpus rows; does not close clustered/repeated-spectrum or new convergence-budget residual. |
| Vector residuals | Helper tests compute `A*v ~= sigma*u` and `A^T*u ~= sigma*v` residuals for bounded fixtures. | Need selected edge-case fixture-local comparison and oracle semantics. |
| Range projectors | `partial_svd_rankdef_diag6x4_k2_range_projector` exists. | Sprint 140 must decide whether rank-deficient range-projector follow-through is the selected residual or a deferred subcase. |
| Low-rank Frobenius | `partial_svd_lowrank_diag6x4_k2_frobenius_optimality` and sparse/dense low-rank consistency tests exist. | Broad low-rank optimality remains a non-claim unless selected and closed. |
| Convergence budget | `partial_svd_max_iter_fail_closed_diag6_k2` tests fail-closed behavior and default-budget recovery. | Need selected residual-specific budget proof and diagnostics without masking non-convergence. |
| Repeated/clustered spectra | Some full SVD repeated-value tests exist. | Maintained partial-SVD clustered/repeated singular-value corpus lane remains open. |

## Item-To-Day Owner Map

| Project-plan item | Day owner(s) | Notes |
| --- | --- | --- |
| Item 1: Partial-SVD Residual Reaudit | Days 1-3 | Intake, candidate ranking, selected residual, backup, and closure contract. |
| Item 2: Edge-Case Fixture Batch | Days 4-5 | Fixture design and source-controlled corpus implementation. |
| Item 3: Comparison Semantics | Days 6-7 | Oracle semantics design and implementation. |
| Item 4: Convergence-Budget Tests | Days 8-9 | Proof-owner design and implementation. |
| Item 5: Proof-Owner Cleanup | Day 10 | Focused helper/test ownership cleanup only. |
| Item 6: Validation | Day 12, with focused checks on implementation days | Full quality gate if `.c` or `.h` files change. |
| Item 7: Docs and Closeout | Days 11, 13, 14 | Public/maintainer docs, claim closure, Sprint 141 handoff, closeout. |

## Initial Validation Expectations

| Touched surface | Required checks |
| --- | --- |
| Corpus manifests, expected rows, generator rows, schemas, or oracle outputs | `python3 scripts/validate_corpus_schema.py`, focused oracle/report command, TSV width checks, generated-artifact hygiene. |
| Python oracle/reference scripts | `python3 -m py_compile <script>` plus focused command validation for touched paths. |
| SVD or partial-SVD `.c`/`.h` tests/helpers | Focused SVD/partial-SVD test target, Make/CMake source-list parity when needed, then `make format && make lint && make test`. |
| SVD implementation or public API files | Focused SVD/partial-SVD tests followed by `make format && make lint && make test`. |
| Public or maintainer documentation | `git diff --check`, trailing-whitespace scan, focused Markdown link/path validation, and claim-boundary scan. |
| Generated reports under `build/` | Confirm ignored/untracked; never commit without an explicit future promotion gate. |

## Initial Non-Claim Register

| Non-claim | Boundary |
| --- | --- |
| Broad SVD or partial-SVD correctness | Sprint 140 closes one selected fixture-local residual only. |
| Raw singular-vector parity | Sign, scale, and basis rotations may differ; use safe metrics. |
| Broad repeated/clustered-spectrum coverage | Only selected fixtures may be claimed after proof and validation. |
| Broad rank-deficient null-space or range-space behavior | Fixture-local projector/subspace claims only if selected and proved. |
| Convergence-rate or performance claim | Budget diagnostics are correctness/fail-closed evidence, not performance evidence. |
| Partial-result guarantee | Only explicit status/diagnostic rows may describe partial results. |
| External-library parity | Dense references are bounded helper evidence, not broad LAPACK/NumPy/SciPy parity. |
| Optional SuiteSparse or external data pass evidence | Remains skip/defer until reviewed support-tier proof exists. |
| Platform, package, ABI, or state-of-the-art claim | Out of scope for Sprint 140. |

## Stop Conditions

- Stop and ask if candidate selection requires broad external-library parity
  or optional data that cannot be locally validated.
- Stop and ask if comparison semantics cannot distinguish valid singular-vector
  ambiguity from a solver defect.
- Stop and ask if a required quality gate fails.
- Stop and ask before broadening public SVD/partial-SVD claims beyond named
  fixture-local evidence.

## Day 1 Notes

- Created Sprint 140 working notes and artifact directory structure.
- Re-read the Sprint 140 project-plan section and Sprint 140 day-by-day plan.
- Reviewed Sprint 138 corpus handoff and Sprint 139 QR closure/handoff
  artifacts.
- Inventoried current partial-SVD evidence across corpus architecture, oracle
  runner, SVD tests, partial-SVD helpers, external dense-reference helper,
  public API, implementation files, docs, and example surfaces.
- Mapped Sprint 140 Items 1-7 to day-level owners.
- Recorded initial validation expectations, non-claims, and stop conditions
  before fixture or code changes begin.

## Day 2 Notes

- Created
  `docs/planning/EPIC_12/SPRINT_140/artifacts/day2-partial-svd-residual-reaudit.md`.
- Re-ranked partial-SVD residual candidates across user-facing risk,
  determinism, fixture complexity, validation fit, and Sprint 140 closure fit.
- Selected `partial_svd_clustered_repeated_subspace_budget_v1` as the priority
  residual family for complete Sprint 140 closure.
- Defined the selected closure boundary as singular-value, projector/subspace,
  vector-residual, orthogonality, fail-closed tight-budget, and default-budget
  recovery proof on one deterministic fixture family.
- Kept `partial_svd_rankdef_range_projector_budget_v1` as the backup candidate
  because the current helper evidence already covers much of that behavior.
- Deferred broad external-library parity, optional SuiteSparse data, near-zero
  rank-threshold policy, low-rank productization, performance proof, report
  index normalization, and raw vector-identity comparison for repeated spectra.
- Confirmed Day 3 can proceed from one bounded behavior family without
  broadening unsupported partial-SVD claims.

## Day 3 Notes

- Created
  `docs/planning/EPIC_12/SPRINT_140/artifacts/day3-partial-svd-closure-design.md`.
- Replaced the Day 2 working residual name with concrete fixture key
  `partial_svd_clustered_repeated_diag8x6_k3_v1`.
- Defined an 8-by-6 generated sparse diagonal fixture with requested `k=3` and
  singular values `{10.0, 10.0, 9.999999, 4.0, 1.0, 0.0}`.
- Set fixture-local success rows for top-k singular values, left/right
  projector subspace distance, vector residuals, U/V orthogonality, and default
  convergence status.
- Set tight-budget diagnostic rows for `SPARSE_ERR_NOT_CONVERGED` and no
  published partial `sigma`, `U`, or `Vt` arrays.
- Defined comparison semantics that accept sign flips and basis rotations while
  rejecting wrong singular values, wrong top-k subspaces, bad residuals, bad
  orthogonality, or false convergence.
- Scoped the preferred proof owner to a future focused
  `tests/test_svd_partial_corpus.c` surface, with reusable helper extraction
  only if needed for readability.
- Confirmed Day 4 can proceed by adding fixture and generator metadata before
  expected rows or solver-backed tests are implemented.

## Day 4 Notes

- Created
  `docs/planning/EPIC_12/SPRINT_140/artifacts/day4-partial-svd-fixture-batch-design.md`.
- Designed the source-controlled corpus fixture row for
  `partial_svd_clustered_repeated_diag8x6_k3_v1`.
- Designed generator metadata for
  `partial_svd_clustered_repeated_diag8x6_generator_v1`, including exact
  canonical structure/value text and SHA-256 hashes.
- Corrected the fixture metadata design to use `rank_status=rank_deficient`
  with `expected_rank=5` and `nullity=1`.
- Defined eight expected-result rows covering top-k singular values, left/right
  subspace projectors, vector residuals, orthogonality, default success status,
  tight-budget non-convergence status, and no partial arrays on failure.
- Confirmed no optional-data skip row is needed for the generated primary
  fixture.
- Left backup, optional SuiteSparse, near-zero threshold, and low-rank sparse
  approximation residuals deferred with explicit reasons.
- Recorded a Day 5 implementation checklist for corpus registry, fixture row,
  generator row, expected TSV, schema compatibility, and validation.

## Day 5 Notes

- Created
  `docs/planning/EPIC_12/SPRINT_140/artifacts/day5-partial-svd-fixture-implementation.md`.
- Added the deterministic generated fixture
  `partial_svd_clustered_repeated_diag8x6_generator_v1` to
  `scripts/validate_corpus_schema.py`.
- Made known-generator parameter validation data-driven so the existing QR
  generator and the new partial-SVD generator both validate from registry
  metadata.
- Added the source-controlled fixture row
  `partial_svd_clustered_repeated_diag8x6_k3_v1` to
  `tests/corpus/manifests/fixtures.tsv`.
- Added the generator manifest row with the Day 4 canonical structure and value
  hashes to `tests/corpus/manifests/generators.tsv`.
- Added
  `tests/corpus/expected/partial_svd_clustered_repeated_diag8x6_k3_v1.tsv`
  with eight fixture-local expected-result rows.
- Preserved the generated-output policy: no oracle or report outputs under
  `build/` were committed as pass evidence.

## Day 6 Notes

- Created
  `docs/planning/EPIC_12/SPRINT_140/artifacts/day6-partial-svd-comparison-semantics.md`.
- Mapped the eight Day 5 partial-SVD expected rows to solver family,
  operation, comparison kind, observed-result format, and pass condition.
- Defined observed-result parsing conventions for `value`,
  `subspace_distance`, `residual_norm`, `status`, and `diagnostic` rows.
- Required descending top-k singular-value comparison while still allowing
  sign flips and basis rotations in the repeated leading singular-value block.
- Defined projector/subspace, vector-residual, and orthogonality semantics that
  avoid raw singular-vector identity.
- Defined convergence-budget interpretation: default-budget rows require
  successful factors, tight-budget rows require `SPARSE_ERR_NOT_CONVERGED` and
  no visible partial arrays.
- Recorded failure-class mapping for mismatches, malformed rows, generator
  mismatches, stale reports, and optional-data non-applicability.
- Confirmed Day 7 should implement a focused partial-SVD oracle path while
  keeping QR behavior backward-compatible.

## Day 7 Notes

- Created
  `docs/planning/EPIC_12/SPRINT_140/artifacts/day7-partial-svd-oracle-implementation.md`.
- Updated `scripts/run_corpus_oracle.py` with fixture-specific expected-row
  loading so QR and partial-SVD rows can validate against their own expected
  TSVs.
- Added generic comparison parsing for `value`, `subspace_distance`,
  `residual_norm`, `status`, and `diagnostic` rows while preserving QR rank,
  nullity, and residual comparisons.
- Added the opt-in `--include-partial-svd` command path for generated-reference
  rows tied to `partial_svd_clustered_repeated_diag8x6_k3_v1`.
- Kept default `python3 scripts/run_corpus_oracle.py` behavior on the existing
  QR-only oracle output path.
- Added combined-output behavior for opt-in partial-SVD runs at
  `build/corpus/oracle/corpus.oracle.tsv`.
- Updated manifest metadata to record fixture keys, partial-SVD row count, and
  broader fixture-local non-claims.
- Preserved generated-output policy: oracle, report, skip, and manifest files
  under `build/` remain local generated evidence and are not committed.

## Day 8 Notes

- Created
  `docs/planning/EPIC_12/SPRINT_140/artifacts/day8-partial-svd-convergence-proof-design.md`.
- Reviewed the existing `test_svd` owner, partial-SVD helper coverage, and
  Make/CMake test registration surfaces.
- Chose a new focused proof owner,
  `tests/test_svd_partial_corpus.c`, instead of expanding the broad
  `tests/test_svd.c` owner further.
- Defined five focused tests for default success, projector/subspace checks,
  vector residual and orthogonality checks, tight-budget fail-closed behavior,
  and recovery after failure.
- Defined local projector-distance semantics for the exact first-three-coordinate
  left and right top-k subspaces.
- Mapped helper ownership so Day 9 can reuse small existing helper APIs where
  clean while keeping the corpus fixture builder and projector logic local.
- Identified build-system touch points in `Makefile` and `CMakeLists.txt`.
- Recorded the Day 9 validation plan, including focused test execution,
  corpus/oracle checks, CMake registration check, and the full C quality gate
  because Day 9 will touch `.c` files.

## Day 9 Notes

- Created
  `docs/planning/EPIC_12/SPRINT_140/artifacts/day9-partial-svd-proof-implementation.md`.
- Added focused proof owner `tests/test_svd_partial_corpus.c`.
- Registered `test_svd_partial_corpus` in `Makefile` and `CMakeLists.txt`.
- Implemented the generated 8x6 clustered/repeated diagonal fixture locally in
  the new proof owner.
- Added default-budget success, projector/subspace, vector residual and
  orthogonality, tight-budget fail-closed, and recovery-after-failure tests.
- Preserved existing `test_svd` and `tests/test_svd_partial_helpers.h`
  coverage without moving or deleting existing tests.

## Day 10 Notes

- Created
  `docs/planning/EPIC_12/SPRINT_140/artifacts/day10-proof-owner-cleanup.md`.
- Added `tests/test_svd_partial_shared_helpers.h` for reusable partial-SVD
  residual and coordinate-projector checks.
- Moved the existing residual/projector helper implementations out of
  `tests/test_svd_partial_helpers.h` and behind the shared helper header.
- Updated `tests/test_svd_partial_corpus.c` to reuse the shared helper APIs
  instead of carrying local duplicate triplet-residual and projector logic.
- Kept fixture construction, default/tight-budget setup, and expected singular
  values local to the Sprint 140 corpus proof owner.
- Preserved the public API, corpus manifests, oracle schema, and support-tier
  claim boundaries.

## Day 11 Notes

- Created
  `docs/planning/EPIC_12/SPRINT_140/artifacts/day11-documentation-update.md`.
- Updated SVD-facing documentation across `README.md`, `docs/cookbook.md`,
  `docs/tutorial.md`, `docs/solver_selection.md`, `docs/algorithm.md`,
  `examples/README.md`, `tests/corpus/README.md`,
  `docs/maintainer_guide.md`, and `include/sparse_svd.h`.
- Added earned fixture-local wording for
  `partial_svd_clustered_repeated_diag8x6_k3_v1`: generated 8x6 diagonal
  fixture, `k=3`, top-3 singular values, top-k subspace projectors, triplet
  residuals, orthogonality, default-budget success, tight-budget fail-closed
  behavior, and no partial arrays on tight-budget failure.
- Preserved explicit non-claims for broad partial-SVD correctness, raw
  singular-vector identity, broad repeated-spectrum coverage, external-library
  parity, platform, performance, package, ABI, partial-result, and
  state-of-the-art behavior.

## Day 12 Notes

- Created
  `docs/planning/EPIC_12/SPRINT_140/artifacts/day12-validation-pass.md`.
- Ran corpus schema validation and opt-in partial-SVD oracle generation.
- Ran the focused `test_svd_partial_corpus` target: 5 tests, 0 failures, and
  107 assertions.
- Ran `make source-list-check` and `make quality-review-cmake-compile`; CMake
  registered `test_svd_partial_corpus` and reported 59 tests matching the 59
  Makefile tests.
- Ran the required full quality gate because Sprint 140 changed `.c` and `.h`
  files: `make format && make lint && make test`.
- Ran script compilation, diff whitespace, targeted trailing-whitespace,
  targeted TSV width, and targeted markdown link checks.
- Removed local Python bytecode generated by `py_compile`.
- Kept generated oracle, report, skip, manifest, and CMake outputs under
  ignored `build/` paths and did not promote support-tier, performance,
  platform, package, ABI, broad correctness, partial-result, or state-of-the-art
  claims.

## Day 13 Notes

- Created
  `docs/planning/EPIC_12/SPRINT_140/artifacts/day13-claim-closure.md`.
- Closed the selected Sprint 140 residual as a fixture-local partial-SVD claim
  for `partial_svd_clustered_repeated_diag8x6_k3_v1`.
- Traced the closed claim to fixture manifests, expected rows, oracle semantics,
  focused compiled proof owner, helper cleanup, public wording, and Day 12
  validation.
- Published the remaining SVD and partial-SVD non-claims in one place so Day 14
  closeout and Sprint 141 planning do not infer broader support.
- Defined Sprint 141 report-index handoff requirements for normalized statuses,
  freshness metadata, stale-report detection, generated-output policy, and
  support-tier interpretation.
- Confirmed Day 13 is documentation-only after the Day 12 full quality gate;
  final Day 13 validation uses diff, whitespace, and markdown-link hygiene.

## Day 14 Notes

- Created
  `docs/planning/EPIC_12/SPRINT_140/artifacts/day14-closeout-validation-summary.md`.
- Reviewed Sprint 140 project-plan Items 1-7 and mapped each item to residual,
  corpus, oracle, test, helper, documentation, validation, or handoff evidence.
- Confirmed all Sprint 140 daily artifacts exist and are represented in the
  closeout inventory.
- Summarized the closed fixture-local partial-SVD residual and deferred
  non-claims for retrospective input.
- Preserved Sprint 141 handoff requirements for report-index normalization,
  freshness metadata, stale-report detection, generated-output policy, and
  support-tier interpretation.
- Ran final Day 14 validation for the branch's changed `.c`/`.h` surfaces:
  `make format && make lint && make test`.
- Ran final Day 14 documentation hygiene checks with `git diff --check`,
  targeted trailing-whitespace scan, and targeted markdown link validation.
- Confirmed Sprint 140 is ready for retrospective creation and pull-request
  packaging.
