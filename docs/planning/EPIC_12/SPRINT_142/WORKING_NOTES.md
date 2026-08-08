# Sprint 142 Working Notes

## Sprint Goal

Convert runtime/backend behavior into a clearer maintained contract and
expand sentinels where they provide useful local regression evidence without
creating portable timing, platform, package, ABI, or broad backend claims.

## Initial Constraints

- Consume the Sprint 141 `runtime_backend` defer row as a policy handoff, not
  as unfinished report-index work.
- Audit current controls before changing runtime/backend behavior.
- Preserve existing public behavior unless a sprint artifact records a
  bounded contract change with validation.
- Prefer typed options for user-facing controls and explicitly classify
  environment-only controls as compatibility, diagnostic, or maintainer-only
  unless promoted.
- Keep local benchmark and sentinel rows local/advisory unless a reviewed
  hard gate already owns them.
- Do not turn runtime/backend sentinel rows into portable performance claims.
- Do not broaden package-manager, shared-library, ABI, platform, broad
  backend-portability, or state-of-the-art claims.

## Inherited Sprint 141 Handoff

| Surface | Sprint 141 result | Sprint 142 use |
| --- | --- | --- |
| `tests/corpus/manifests/report_families.tsv` | Added `runtime_backend/governance/deferred_governance` row. | Source-controlled handoff row for runtime/backend governance. |
| `scripts/normalize_report_index.py` | Emits and freshness-checks the `runtime_backend` defer row. | Validation and report-index context for Sprint 142 rows. |
| `tests/test_normalize_report_index.py` | Covers deferred runtime/backend row behavior. | Baseline for any sentinel/report-index integration tests. |
| Sprint 141 Day 14 closeout | Routes runtime/backend governance and precedence policy to Sprint 142. | Scope authority for the sprint. |
| Sprint 141 retrospective | Defines readiness fields: runtime audit, precedence contract, typed decisions, sentinel expansion, docs, and validation. | Planning checklist for Sprint 142. |

## Initial Runtime/Backend Surface Map

| Surface | Initial owner area | Current signal |
| --- | --- | --- |
| Cholesky backend dispatch | `include/sparse_cholesky.h`, `src/sparse_cholesky.c`, `src/sparse_chol_csc.c`, related tests | `sparse_cholesky_opts_t::backend`, `SPARSE_CHOL_BACKEND_*`, `SPARSE_CSC_THRESHOLD`, and `used_csc_path`. |
| LDLT backend dispatch | `include/sparse_ldlt.h`, `src/sparse_ldlt.c`, CSC LDLT sources, related tests | `sparse_ldlt_opts_t::backend`, `SPARSE_LDLT_BACKEND_*`, `SPARSE_CSC_THRESHOLD`, and `used_csc_path`. |
| Dense helper backend selection | `src/sparse_chol_csc_supernodal.c`, `src/sparse_ldlt_dense.c`, supernodal tests | `SPARSE_CHOL_DENSE_BACKEND`, `SPARSE_LDLT_DENSE_BACKEND`, builtin/external/accelerate requests, and fallback descriptors. |
| Eigensolver backend selection | `include/sparse_eigs.h`, `src/sparse_eigs*.c`, eigensolver tests and benchmarks | `sparse_eigs_opts_t::backend`, `SPARSE_EIGS_BACKEND_*`, AUTO thresholds, preconditioner routing, and `backend_used`. |
| OpenMP controls | Make/CMake flags, iterative/eigs/matvec tests, README | `SPARSE_OPENMP`, `OMP_NUM_THREADS`, OpenMP-enabled SpMV and reorthogonalization thresholds. |
| Graph/ND and FM environment controls | `src/sparse_graph*.c`, reorder/graph tests, maintainer docs | `SPARSE_ND_*`, `SPARSE_FM_*`, typed analysis-time routing/coarsening controls, and compatibility env overrides. |
| Direct-solver analysis controls | `include/sparse_analysis.h`, `src/sparse_analysis.c`, reorder/LDLT/Cholesky tests | `sparse_analysis_opts_t`, reorder options, supernodal etree postorder, ND controls, and compatibility overrides. |
| Runtime benchmark and sentinel reports | `Makefile`, `scripts/performance_sentinels.sh`, `scripts/large_matrix_guardrails.sh`, `benchmarks/README.md` | `make performance-sentinels`, `make large-matrix-guardrails`, wall-check hard gate, threshold-free local rows, backend request/selection/fallback fields. |
| Normalized report-index integration | `scripts/normalize_report_index.py`, `tests/corpus/manifests/report_families.tsv` | Existing benchmark, sentinel, guardrail, and `runtime_backend` rows with freshness/non-claim boundaries. |

## Item-To-Day Owner Map

| Project-plan item | Day owner(s) | Notes |
| --- | --- | --- |
| Item 1: Runtime Control Audit | Days 1-3 | Intake, canonical inventory, backend dispatch/fallback audit, and coverage map. |
| Item 2: Precedence Contract | Days 4-5 | Contract design and mechanical implementation for selected precedence paths. |
| Item 3: Typed-Control Batch | Days 6-7 | Candidate scoring, selected promotion/deferral batch, implementation, and tests. |
| Item 4: Sentinel Expansion | Days 8-9 | Sentinel design and implementation for selected runtime/backend rows. |
| Item 5: Docs and Examples | Day 10 | Public/maintainer docs and examples affected by earned runtime/backend changes. |
| Item 6: Validation | Days 11-12 | Focused validation, report/freshness checks, generated-output hygiene, and full quality gate if C/header files changed. |
| Item 7: Closeout | Days 13-14 | Earned claims, non-claims, Sprint 143 package/ABI handoff, final validation, and closeout. |

## Initial Validation Expectations

| Touched surface | Required checks |
| --- | --- |
| Documentation and planning artifacts only | `git diff --check`, trailing-whitespace scan, and focused path/reference review. |
| Python report-index or corpus scripts/tests | `python3 -m py_compile`, focused script tests, `python3 scripts/validate_corpus_schema.py`, and normalized report-index/freshness checks. |
| Sentinel shell scripts or report fixtures | Focused script execution or syntax review, normalized report-index/freshness checks for affected families, and ignored generated-output checks. |
| C or header files | Focused tests for touched backend/runtime behavior, then `make format && make lint && make test`. |
| Build-system registrations | Source-list/build parity checks and relevant Make/CMake focused compile or test target. |
| Benchmark/sentinel docs | Documentation hygiene plus command/path consistency checks. |

## Initial Non-Claim Register

| Non-claim | Boundary |
| --- | --- |
| Portable performance | Runtime and sentinel rows remain local unless a reviewed gate explicitly owns the claim. |
| Broad backend portability | Backend behavior is scoped to documented options, build flags, platform, and fallback context. |
| Package/ABI support | Runtime governance does not imply package-manager, shared-library, dynamic-loader, or ABI support. |
| Hosted platform proof | Local runtime rows and generated sentinel reports do not become hosted CI evidence. |
| Broad solver correctness | Backend dispatch evidence is not a broad solver correctness or corpus-completeness proof. |
| Environment variables as public API | Environment controls remain compatibility/maintainer-only unless explicitly promoted to typed controls. |
| State-of-the-art status | Runtime governance and local sentinels are not competitive status evidence. |

## Stop Conditions

- Stop and ask if a runtime/backend decision requires product-policy scope
  beyond Sprint 142.
- Stop and ask if a typed-control promotion would imply ABI or package support
  that belongs to Sprint 143.
- Stop and ask if a sentinel row would require a portable performance claim or
  machine-class guarantee not already reviewed.
- Stop and ask if generated local reports would need to be committed as proof.
- Stop and ask if required quality gates fail.

## Day 1 Notes

- Created Sprint 142 working notes and artifact directory structure.
- Re-read the Sprint 142 project-plan section and Sprint 142 day-by-day plan.
- Reviewed Sprint 141 closeout and retrospective handoff material.
- Confirmed the `runtime_backend` row is a governance handoff, not a
  report-index implementation gap.
- Ran initial scans across source, headers, tests, scripts, benchmarks,
  README, benchmark docs, maintainer guide, and Make targets to seed the
  runtime/backend surface map.
- Identified initial surfaces: Cholesky and LDLT backend dispatch,
  dense-helper environment selection, eigensolver backend AUTO routing,
  OpenMP controls, graph/ND/FM environment controls, analysis typed options,
  performance sentinels, large-matrix guardrails, and normalized report-index
  integration.
- Mapped Sprint 142 Items 1-7 to day-level owners.
- Recorded initial validation expectations, non-claims, and stop conditions
  before audit or implementation work begins.

## Day 2 Notes

- Created
  `docs/planning/EPIC_12/SPRINT_142/artifacts/day2-runtime-control-inventory.md`
  as the canonical runtime/backend control inventory.
- Confirmed public typed controls currently include Cholesky backend dispatch,
  LDLT backend dispatch, eigensolver backend dispatch, and analysis/reorder
  policy.
- Confirmed build-time controls include `SPARSE_OPENMP`, `SPARSE_MUTEX`,
  `SPARSE_CSC_THRESHOLD`, eigensolver AUTO thresholds, `SPARSE_DROP_TOL`, and
  `SPARSE_NODES_PER_SLAB`.
- Confirmed `OMP_NUM_THREADS` is external OpenMP runtime context, not a
  library-owned per-call thread-control API.
- Confirmed analysis typed fields use the intended precedence:
  explicit typed value, then legacy compatibility env override when the field
  is unspecified, then internal default.
- Classified dense helper env selectors (`SPARSE_CHOL_DENSE_BACKEND` and
  `SPARSE_LDLT_DENSE_BACKEND`) as maintainer/compatibility controls pending
  explicit promotion or deferral.
- Added the SVD low-rank env selector (`SPARSE_SVD_LOWRANK_OUTER`) to the
  Day 3 audit queue because it affects runtime behavior but was not in the
  Day 1 priority map.
- Separated FM/ND/QG diagnostic and strategy env vars from public typed
  analysis controls.
- Mapped sentinel and report commands as maintainer evidence controls, not
  runtime API: `make performance-sentinels`, `make wall-check`,
  `make large-matrix-guardrails`, and `make bench-canonical-report`.
- Recorded Day 3 risks around backend selected-vs-fallback vocabulary, dense
  helper disposition, exact ND/FM precedence, package/link handoffs, Windows
  staged exclusions, and local-only sentinel claim boundaries.

## Day 3 Notes

- Created
  `docs/planning/EPIC_12/SPRINT_142/artifacts/day3-backend-dispatch-audit.md`
  as the backend dispatch and fallback audit.
- Confirmed Cholesky top-level dispatch has no post-selection linked-list
  fallback: AUTO uses `SPARSE_CSC_THRESHOLD`, explicit backends force the
  requested path, invalid backend enum values return `SPARSE_ERR_BADARG`, and
  `used_csc_path` is published immediately after dispatch selection.
- Confirmed LDLT top-level dispatch mirrors Cholesky except for `n == 0`,
  where forced CSC remains linked-list because the CSC scalar pre-pass has no
  meaningful empty input.
- Clarified LDLT vocabulary for Day 4: "CSC selected" may still complete via
  the CSC pipeline's resolved scalar-prepass fallback when the batched
  supernodal completion rejects the cached pivot pattern.
- Confirmed eigensolver AUTO priority: eligible preconditioned large/block
  solves route to LOBPCG before the Lanczos threshold rule; otherwise AUTO
  chooses thick-restart above `SPARSE_EIGS_THICK_RESTART_THRESHOLD` and
  grow-m below it.
- Confirmed `benchmarks/bench_eigs.c` mirrors the public eigensolver
  preconditioner gating so benchmark rows report the backend and preconditioner
  actually used.
- Confirmed Cholesky dense helper fallback is covered by focused env tests;
  LDLT dense helper request/selected/fallback is visible in
  `bench_refactor_csc`, but focused invalid-env test parity is weaker.
- Classified `SPARSE_SVD_LOWRANK_OUTER` as a maintainer/runtime env selector
  with existing SVD tests, not yet a runtime/backend sentinel row.
- Identified candidate sentinel expansions for dispatch-only Cholesky/LDLT
  route snapshots, eigensolver AUTO backend snapshots, shift-invert LDLT route
  snapshots, LDLT dense helper fallback rows, and explicit SVD env deferral.
- Captured Day 4 inputs: define selected backend versus internal completion
  path, keep dense helper env selectors maintainer-only unless promoted,
  state typed option precedence over AUTO thresholds, and keep OpenMP/thread
  settings as build/report context rather than library-owned runtime policy.

## Day 4 Notes

- Created
  `docs/planning/EPIC_12/SPRINT_142/artifacts/day4-precedence-contract-design.md`
  as the maintained precedence contract draft.
- Defined global precedence ordering: explicit public typed option, typed
  AUTO/DEFAULT semantics, documented compatibility env override, compile-time
  threshold/feature flag, internal default, then maintainer diagnostic/report
  context.
- Confirmed explicit Cholesky/LDLT/eigensolver backend selectors take
  precedence over AUTO thresholds, while invalid enum values remain
  `SPARSE_ERR_BADARG`.
- Preserved the LDLT `n == 0` forced-CSC exception and separated top-level
  backend selection from internal CSC completion fallback vocabulary.
- Defined analysis/reorder precedence as explicit typed field first,
  compatibility env only when the typed field is DEFAULT/unspecified, and
  internal default last.
- Kept FM strategy/debug/profile variables, dense helper selectors,
  `SPARSE_SVD_LOWRANK_OUTER`, and test opt-ins out of public API wording
  unless Day 6 explicitly selects them for promotion.
- Defined OpenMP and `OMP_NUM_THREADS` as build/report context rather than a
  library-owned runtime thread policy.
- Added fallback/failure rules for invalid typed values, unrecognized env
  values, unavailable dense backends, internal CSC completion fallback, OpenMP
  build availability, and sentinel timing rows.
- Drafted Day 5+ validation scenarios for backend dispatch precedence,
  analysis typed-vs-env precedence, dense-helper fallback, OpenMP report-only
  context, and sentinel row claim boundaries.
- Day 5 guidance is intentionally conservative: implement documentation and
  focused tests first, avoid ABI/package metadata changes, and leave typed
  promotion decisions to Day 6.

## Day 5 Notes

- Created
  `docs/planning/EPIC_12/SPRINT_142/artifacts/day5-precedence-contract-implementation.md`
  as the validation-backed precedence implementation ledger.
- Audited the Day 4 validation scenarios against existing mechanical owners
  and found that public typed backend precedence, AUTO routing, analysis
  typed-vs-env precedence, and sentinel row boundaries already have executable
  owners.
- Did not change C/header behavior on Day 5. The contract surface was landed
  as documentation plus focused validation because the audited behavior
  already matches the Day 4 precedence contract.
- Did not add a forced-CSC empty LDLT public test: `sparse_create(0,0)` is not
  a public matrix fixture, and fabricating an internal `SparseMatrix` shell
  would harden private state rather than public behavior.
- Built focused precedence owners with:
  `make build/test_chol_csc build/test_ldlt_backend_dispatch build/test_eigs_thick_restart build/test_eigs_lobpcg build/test_reorder_nd`.
- Ran focused C validation:
  - `build/test_chol_csc`: 92 tests passed.
  - `build/test_ldlt_backend_dispatch`: 22 tests passed.
  - `build/test_eigs_thick_restart`: 23 tests passed.
  - `build/test_eigs_lobpcg`: 29 tests passed.
  - `build/test_reorder_nd`: 35 tests passed, 1 known skip unrelated to a
    failed precedence assertion.
- Ran `python3 tests/test_normalize_report_index.py`; it passed and keeps
  sentinel hard-gate/advisory row boundaries validated.
- Deferred dense helper typed promotion, LDLT dense invalid-env parity,
  `SPARSE_SVD_LOWRANK_OUTER`, FM/debug/profile env vars, dispatch-only
  sentinel expansion, and package/link metadata changes to their planned owner
  days.

## Day 6 Notes

- Created
  `docs/planning/EPIC_12/SPRINT_142/artifacts/day6-typed-control-selection.md`
  as the typed-control candidate matrix and selection artifact.
- Scored candidate promotions by user value, implementation risk, validation
  readiness, documentation burden, and claim risk.
- Selected a conservative Day 7 batch: make maintainer-only deferrals explicit,
  avoid public ABI/API expansion, preserve dispatch-only sentinel candidates
  for Days 8-9, and optionally add LDLT dense invalid-env fallback validation
  only if it can remain test-only and low-risk.
- Explicitly deferred public typed promotion for
  `SPARSE_CHOL_DENSE_BACKEND`, `SPARSE_LDLT_DENSE_BACKEND`,
  `SPARSE_SVD_LOWRANK_OUTER`, FM strategy/debug/profile env vars, OpenMP
  runtime policy, package/link metadata, and test/benchmark opt-ins.
- Kept `SPARSE_OPENMP`, `SPARSE_MUTEX`, `SPARSE_CSC_THRESHOLD`,
  eigensolver AUTO thresholds, `SPARSE_DROP_TOL`, and
  `SPARSE_NODES_PER_SLAB` as build-time or compile-time controls, not new
  runtime typed API.
- Deferred dispatch-only Cholesky/LDLT, eigensolver AUTO, and shift-invert
  LDLT route snapshots to Day 8 sentinel design and Day 9 sentinel
  implementation.
- Confirmed the Day 7 work should not broaden package, ABI, shared-library,
  platform, portable performance, or state-of-the-art claims.

## Day 7 Notes

- Created
  `docs/planning/EPIC_12/SPRINT_142/artifacts/day7-typed-control-batch.md`
  as the selected typed-control implementation artifact.
- Implemented the Day 6 selected batch as an explicit non-expansion and
  deferral ledger: public typed controls stay limited to existing Cholesky,
  LDLT, eigensolver, and analysis/reorder surfaces.
- Confirmed the optional LDLT dense invalid-env fallback proof already exists
  in `tests/test_ldlt.c` through
  `test_ldlt_dense_backend_invalid_env_falls_back_to_builtin`, so no duplicate
  C test was added.
- Reaffirmed that dense-helper selectors, `SPARSE_SVD_LOWRANK_OUTER`,
  FM/debug/profile variables, OpenMP runtime context, package/link metadata,
  and test/benchmark opt-ins remain outside public typed API claims.
- Preserved dispatch-only Cholesky/LDLT, eigensolver AUTO, shift-invert LDLT,
  and optional dense-helper report-row candidates for Days 8-9 sentinel work.
- No C/header files changed on Day 7; full `make format && make lint &&
  make test` remains reserved for later code-changing days.

## Day 8 Notes

- Created
  `docs/planning/EPIC_12/SPRINT_142/artifacts/day8-runtime-sentinel-design.md`
  as the sentinel expansion design artifact.
- Reviewed the existing `make performance-sentinels` producer,
  `scripts/performance_sentinels.sh`, `scripts/normalize_report_index.py`,
  `tests/test_normalize_report_index.py`, `tests/corpus/manifests/report_families.tsv`,
  and benchmark report documentation.
- Selected a narrow Day 9 implementation path: keep `S5` as the only hard
  local gate, keep `S2` as the existing advisory Cholesky CSC row, and add
  `S3` as an advisory LDLT KKT row sourced from
  `bench_refactor_csc --indefinite-kkt --repeat 1`.
- Deferred eigensolver AUTO, shift-invert LDLT, OpenMP runtime, package, and
  standalone dense-helper sentinel rows because they would require broader
  report schemas or would imply claims Sprint 142 has not earned.
- Confirmed the existing normalizer already preserves sentinel hard/advisory
  row separation and backend request/selected/fallback fields in the
  normalized `configuration` field.
- No code or header files changed on Day 8; the design defines the concrete
  script/test edits for Day 9.

## Day 9 Notes

- Created
  `docs/planning/EPIC_12/SPRINT_142/artifacts/day9-sentinel-implementation.md`
  as the sentinel implementation artifact.
- Extended `scripts/performance_sentinels.sh` to accept
  `bench_refactor_csc`, run `bench_refactor_csc --indefinite-kkt --repeat 1`,
  and emit advisory `S3` rows from the generated KKT CSV metrics.
- Updated the `Makefile` `performance-sentinels` target so
  `$(BUILDDIR)/bench_refactor_csc` is built and passed to the sentinel script.
- Added synthetic `S3` coverage to `tests/test_normalize_report_index.py`,
  including assertions for normalized advisory status and backend
  request/selected/fallback preservation.
- Updated `benchmarks/README.md` with the `S3` row meaning,
  `bench_refactor_csc_kkt.csv` generated artifact, and ignored generated-output
  policy under `build/`.
- Updated `docs/maintainer_guide.md` so current maintainer guidance names S3
  as threshold-free LDLT KKT backend report context.
- Changed the sentinel manifest source commit from short SHA to full SHA so
  generated sentinel rows compare cleanly against normalized report-index
  freshness checks.
- Preserved existing semantics: `S5` is still the only hard local gate, `S2`
  and `S3` remain threshold-free advisory rows, and no portable performance or
  platform claim was added.
- Validation passed:
  `python3 tests/test_normalize_report_index.py`,
  `make build/bench_chol_csc build/bench_refactor_csc build/bench_amd_qg build/bench_reorder`,
  `make performance-sentinels`,
  `python3 scripts/normalize_report_index.py --family sentinel --output build/report-index/normalized-index.tsv`,
  `python3 scripts/normalize_report_index.py --family sentinel --check-freshness`,
  `make format && make lint`, `bash -n scripts/performance_sentinels.sh`, and
  `git diff --check`.

## Day 10 Notes

- Created
  `docs/planning/EPIC_12/SPRINT_142/artifacts/day10-runtime-docs-alignment.md`
  as the documentation alignment artifact.
- Updated `README.md` with a runtime/backend control boundary that separates
  public typed options from maintainer, build, runtime-context, and report
  controls.
- Updated current-facing benchmark/sentinel wording in `README.md`,
  `docs/cookbook.md`, `docs/algorithm.md`, and `docs/maintainer_guide.md` so
  `make performance-sentinels` consistently describes `S5` as the hard
  wall-check gate and `S2`/`S3` as threshold-free local context.
- Updated `tests/corpus/schemas/report_index_fields.md` to clarify that
  selected Sprint 142 sentinel rows belong under the `sentinel` family, while
  unresolved runtime/backend policy can remain deferred.
- Did not change public headers or examples because no public typed options
  changed in the Day 7-9 work.
- Preserved package, ABI, platform, optional-backend availability, portable
  performance, and state-of-the-art non-claims.

## Day 11 Notes

- Created
  `docs/planning/EPIC_12/SPRINT_142/artifacts/day11-focused-runtime-validation.md`
  as the focused validation artifact.
- Built focused backend/precedence binaries with
  `make build/test_chol_csc build/test_ldlt_backend_dispatch build/test_eigs_thick_restart build/test_eigs_lobpcg build/test_reorder_nd build/test_ldlt`.
- Ran focused C validation:
  - `./build/test_chol_csc`: 92 tests passed.
  - `./build/test_ldlt_backend_dispatch`: 22 tests passed.
  - `./build/test_eigs_thick_restart`: 23 tests passed.
  - `./build/test_eigs_lobpcg`: 29 tests passed.
  - `./build/test_reorder_nd`: 35 tests passed, 1 known skip.
  - `./build/test_ldlt`: 89 tests passed.
- Ran sentinel/report-index validation:
  - `python3 tests/test_normalize_report_index.py`: passed.
  - `bash -n scripts/performance_sentinels.sh`: passed.
  - `python3 scripts/validate_corpus_schema.py`: passed.
  - `make performance-sentinels`: passed and generated the S3 KKT artifact.
  - `python3 scripts/normalize_report_index.py --family sentinel --output build/report-index/normalized-index.tsv`: passed with 21 rows.
  - `python3 scripts/normalize_report_index.py --family sentinel --check-freshness`: passed.
  - `python3 scripts/normalize_report_index.py --family benchmark --family sentinel --family guardrail --check-freshness`: passed with 25 rows.
- Inspected generated-output hygiene with
  `git ls-files --others --exclude-standard`; generated `build/` report
  outputs were ignored as intended.
- Removed the Python cache file produced by schema validation
  (`scripts/__pycache__/validate_corpus_schema.cpython-314.pyc`).
- No scoped runtime/backend repair was needed on Day 11.

## Day 12 Notes

- Created
  `docs/planning/EPIC_12/SPRINT_142/artifacts/day12-quality-gate.md`
  as the full quality-gate artifact for the touched Sprint 142 surface.
- Confirmed the current diff does not include any `*.c` or `*.h` files, so
  the Sprint 142 conditional `make test` requirement was not triggered.
- Ran script and Python checks:
  - `bash -n scripts/performance_sentinels.sh`: passed.
  - `python3 -m py_compile tests/test_normalize_report_index.py scripts/normalize_report_index.py scripts/validate_corpus_schema.py`: passed.
  - `python3 tests/test_normalize_report_index.py && python3 scripts/validate_corpus_schema.py`: passed.
- Recorded one invalid validation attempt in the Day 12 artifact:
  `python3 -m py_compile scripts/performance_sentinels.sh ...` failed because
  the sentinel producer is a shell script, not Python. The corrected shell
  syntax and Python compile checks passed.
- Ran `make format && make lint`; it passed, including formatting,
  benchmark/example tooling build, strict compile, clang-tidy, and cppcheck.
- Ran report-index validation:
  - `python3 scripts/normalize_report_index.py --family sentinel --output build/report-index/normalized-index.tsv`: passed with 21 rows.
  - `python3 scripts/normalize_report_index.py --family sentinel --check-freshness`: passed with 21 rows.
  - `python3 scripts/normalize_report_index.py --family benchmark --family sentinel --family guardrail --check-freshness`: passed with 25 rows.
- Ran `git diff --check`; it passed.
- Removed generated Python cache directories from `scripts/__pycache__` and
  `tests/__pycache__`.

## Day 13 Notes

- Created
  `docs/planning/EPIC_12/SPRINT_142/artifacts/day13-claim-closure-and-sprint143-handoff.md`
  as the claim closure and Sprint 143 handoff artifact.
- Compared Sprint 142 outcomes against the original item map:
  - runtime control audit: complete.
  - precedence contract: complete.
  - typed-control batch: complete as an explicit non-expansion/deferral batch.
  - sentinel expansion: complete with advisory `S3` LDLT KKT rows.
  - docs/examples alignment: complete for the touched docs surface.
  - validation: complete through Day 11 focused checks and Day 12 quality
    gate.
  - final closeout: remains assigned to Day 14.
- Published earned runtime/backend claims only where backed by specific
  artifacts:
  - existing typed backend/reorder controls are the public caller-facing
    control surface.
  - explicit typed values retain precedence over AUTO/default or compatibility
    behavior where the existing API supports explicit selection.
  - maintainer/env/build/report controls remain outside public typed API
    unless separately promoted.
  - `make performance-sentinels` now emits local advisory `S3` LDLT KKT
    backend context.
  - `S5` remains the only hard local sentinel gate; `S2` and `S3` remain
    threshold-free advisory rows.
  - normalized report-index freshness can discover and check the selected
    runtime/backend sentinel rows.
- Reaffirmed non-claims for shared-library ABI, dynamic loader behavior,
  package-manager availability, broad backend portability, platform parity,
  portable performance, optional dense-kernel availability, and
  state-of-the-art status.
- Routed concrete Sprint 143 prerequisites:
  static-first install baseline audit, CMake/pkg-config downstream consumer
  proof, explicit shared-library product decision, runtime sentinel non-claim
  preservation, package/link control classification, and platform-tier
  separation for Sprint 144.

## Day 14 Notes

- Created
  `docs/planning/EPIC_12/SPRINT_142/artifacts/day14-closeout-validation-summary.md`
  as the final closeout validation and handoff summary.
- Reconfirmed the final changed surface:
  `Makefile`, runtime/backend docs, benchmark docs,
  `scripts/performance_sentinels.sh`, report-index schema/test files, and the
  Sprint 142 planning artifact package.
- Confirmed no `*.c` or `*.h` files are present in the final diff, so the
  conditional full `make test` requirement for C/header changes was not
  triggered.
- Re-ran final validation after Day 13 claim/handoff updates:
  - `bash -n scripts/performance_sentinels.sh`: passed.
  - `python3 -m py_compile tests/test_normalize_report_index.py scripts/normalize_report_index.py scripts/validate_corpus_schema.py`: passed.
  - `python3 tests/test_normalize_report_index.py`: passed.
  - `python3 scripts/validate_corpus_schema.py`: passed.
  - `python3 scripts/normalize_report_index.py --family sentinel --output build/report-index/normalized-index.tsv`: passed with 21 rows.
  - `python3 scripts/normalize_report_index.py --family sentinel --check-freshness`: passed with 21 rows.
  - `python3 scripts/normalize_report_index.py --family benchmark --family sentinel --family guardrail --check-freshness`: passed with 25 rows.
- Reviewed the artifact package for consistency with implemented behavior:
  audit, precedence, typed-control deferral, sentinel expansion, docs,
  validation, claim closure, and package/ABI handoff all have explicit day
  artifacts.
- Reconfirmed claim boundaries:
  `S5` remains the only hard local sentinel gate; `S2` and `S3` remain local
  advisory rows; Sprint 142 does not claim shared-library ABI, package-manager
  availability, platform parity, portable performance, broad backend
  portability, optional dense-kernel availability, or state-of-the-art status.
- Routed remaining work forward:
  package/ABI decision and validation to Sprint 143, platform promotion to
  Sprint 144, and any future environment-control promotion or new runtime
  sentinel rows to future owners with explicit evidence requirements.
