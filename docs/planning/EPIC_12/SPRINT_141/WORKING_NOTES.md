# Sprint 141 Working Notes

## Sprint Goal

Normalize maintained report metadata across corpus, benchmark, sentinel,
guardrail, coverage, dead-code, package, install, and oracle lanes where row
meaning can be preserved honestly, then add freshness checks that identify
stale generated reports without turning local measurements into release proof.

## Initial Constraints

- Preserve the Sprint 138 corpus architecture and the Sprint 139/140
  fixture-local proof pattern.
- Normalize only report families whose row meaning can be represented without
  distortion.
- Treat local benchmark, sentinel, coverage, and dead-code output as
  branch-local or advisory evidence unless a reviewed lane explicitly promotes
  it.
- Keep generated reports under ignored `build/` or tool-owned report
  directories unless a future sprint deliberately promotes reviewed artifacts.
- Preserve optional-data skip/defer semantics.
- Do not broaden package, ABI, platform, backend, performance, corpus
  completeness, external-library parity, or state-of-the-art claims.

## Inherited Evidence Inventory

| Surface | Current evidence | Sprint 141 use |
| --- | --- | --- |
| `docs/planning/EPIC_12/PROJECT_PLAN.md` | Defines Sprint 141 Items 1-7 and the 168-hour scope. | Sprint-level scope and estimate authority. |
| `docs/planning/EPIC_12/SPRINT_138/RETROSPECTIVE.md` | Establishes corpus manifests, expected-result rows, generated-reference rows, support tiers, optional-data skips, and generated-output policy. | Baseline for corpus/oracle row metadata. |
| `docs/planning/EPIC_12/SPRINT_139/RETROSPECTIVE.md` | Leaves stale-report diagnostics and freshness normalization to Sprint 141 after QR closure. | QR handoff for generated-reference versus solver-backed rows. |
| `docs/planning/EPIC_12/SPRINT_140/RETROSPECTIVE.md` | Hands off generated-reference, solver-backed, skip, stale, unsupported, freshness, support-tier, and claim-boundary requirements. | Primary readiness source for Sprint 141. |
| `tests/corpus/` | Contains schemas, manifests, fixture metadata, generator metadata, expected rows, and documentation for QR and partial-SVD corpus lanes. | Source-controlled report-family row source. |
| `scripts/validate_corpus_schema.py` | Validates corpus TSVs, generator hashes, expected rows, and skip/defer guardrails. | Candidate validation integration point. |
| `scripts/run_corpus_oracle.py` | Emits generated-reference rows, optional solver-backed QR/partial-SVD rows, report index metadata, and manifest context under `build/`. | Candidate normalized index input and pattern. |
| `scripts/bench_canonical_report.sh` | Generates canonical benchmark CSV reports, `index.tsv`, and `manifest.txt` under `build/bench-reports/canonical/`. | Benchmark report-family input. |
| `scripts/performance_sentinels.sh` | Generates local sentinel bundle with hard wall-check gate context and threshold-free report rows. | Sentinel report-family input. |
| `scripts/large_matrix_guardrails.sh` | Generates large-matrix structural guardrail reports and manifest/index artifacts. | Guardrail report-family input. |
| `scripts/deadcode_report.py` and `scripts/deadcode_workflow.sh` | Generate classified dead-code report artifacts under `build/deadcode/`. | Dead-code report-family input. |
| `Makefile` | Owns report-producing targets: `bench-canonical-report`, `performance-sentinels`, `large-matrix-guardrails`, `deadcode-report`, `deadcode-check`, `coverage`, `install`, and package checks. | Command inventory and future validation hook surface. |
| `.github/workflows/` | Defines reviewed Linux, macOS, and Windows CI lanes plus supplemental package/install/report paths. | CI-summary and platform-support metadata source. |
| `README.md`, `docs/cookbook.md`, `docs/maintainer_guide.md`, `benchmarks/README.md`, `INSTALL.md` | Describe report interpretation, local benchmark/report non-claims, package/install validation, and maintainer workflows. | Documentation surfaces for later alignment. |

## Initial Report Family Inventory

| Report family | Current producer or source | Current output or row source | Initial normalization risk |
| --- | --- | --- | --- |
| Corpus manifests | `tests/corpus/manifests/*.tsv` | source-controlled fixture, generator, and optional-data rows | Low; row meaning is already structured. |
| Corpus expected results | `tests/corpus/expected/*.tsv` | source-controlled expected-result rows | Low; row meaning is fixture-local and schema-checked. |
| Oracle generated-reference rows | `python3 scripts/run_corpus_oracle.py` | ignored generated rows under `build/` | Medium; generated rows need freshness/source context. |
| Solver-backed corpus rows | `scripts/run_corpus_oracle.py --include-solver-qr --include-partial-svd` | ignored generated report rows under `build/` | Medium; must distinguish solver-backed local proof from hosted-platform evidence. |
| Canonical benchmarks | `make bench-canonical-report` | `build/bench-reports/canonical/index.tsv` and `manifest.txt` | Medium; local measurement rows must not become release/performance claims. |
| Performance sentinels | `make performance-sentinels` | `build/bench-reports/sentinels/` bundle | High; one hard gate plus advisory rows need separate semantics. |
| Large-matrix guardrails | `make large-matrix-guardrails` | `build/bench-reports/large-matrix-guardrails/` reports | Medium; optional/large data and structural rows need support-tier fields. |
| Dead-code reports | `make deadcode-report` and `make deadcode-check` | `build/deadcode/report.md` and `report.tsv` | Medium; completeness checks are not a zero-findings guarantee. |
| Coverage reports | `make coverage`, `coverage-lcov`, `coverage-gcovr` | `coverage/coverage-src.info` and `coverage/html/` | High; backend/tool/platform differences affect row meaning. |
| Package metadata | Make install, CMake install/export, `sparse.pc.in`, generated `sparse.pc` | installed files and generated package metadata | Medium; platform/package-manager claims remain deferred. |
| Install validation | `tests/test_install.sh`, CMake install/downstream CI snippets | local and CI install proof logs | Medium; static-first install proof is platform-scoped. |
| CI summary lanes | `.github/workflows/*.yml` | hosted job checks and environment metadata | High; report index can reference lanes, but job logs are not source-controlled rows. |

## Item-To-Day Owner Map

| Project-plan item | Day owner(s) | Notes |
| --- | --- | --- |
| Item 1: Report Family Inventory | Days 1-2 | Intake, canonical family inventory, commands, row meanings, owners, and unknowns. |
| Item 2: Shared Metadata Contract | Days 3 and 5 | Contract design and implementation for common/family-specific fields. |
| Item 3: Normalized Index Generator | Days 4, 6, 7, 8, 9 | Generator design, implementation, and family integrations. |
| Item 4: Stale-Report Gate | Days 10-11 | Freshness design and implementation. |
| Item 5: Documentation Alignment | Day 12 | Maintainer, benchmark, corpus, package, and user-facing docs. |
| Item 6: Validation | Day 13, with focused checks on implementation days | Report generators, freshness checks, scripts, docs, and required quality gates. |
| Item 7: Closeout | Day 14 | Final evidence, working-note closure, and Sprint 142 runtime/backend handoff. |

## Initial Validation Expectations

| Touched surface | Required checks |
| --- | --- |
| Documentation and planning artifacts only | `git diff --check`, trailing-whitespace scan, and focused Markdown path/link review. |
| Corpus TSVs, schemas, or manifests | `python3 scripts/validate_corpus_schema.py`, TSV width checks, and generated-artifact hygiene. |
| Python scripts | `python3 -m py_compile <script>` plus focused command tests for touched paths. |
| Report generator or freshness output | deterministic-output comparison, missing/stale/defer fixtures, and ignored generated-output checks. |
| Make targets or CI workflow report surfaces | focused target dry run or syntax review, plus path/command consistency checks. |
| C or header files | focused test target, then `make format && make lint && make test`. |

## Initial Non-Claim Register

| Non-claim | Boundary |
| --- | --- |
| Broad report completeness | Sprint 141 starts with an inventory and normalizes only families with preservable row meaning. |
| Fresh local measurements as release proof | Benchmark, sentinel, coverage, and dead-code outputs remain local/advisory unless reviewed gates explicitly promote them. |
| Portable performance | Benchmark and sentinel rows describe current machine/compiler/configuration only. |
| Hosted platform parity | CI lane metadata may be indexed, but local generated rows do not become hosted proof. |
| Package-manager, ABI, or shared-library support | Static-first package/install evidence remains scoped to existing reviewed lanes. |
| Broad corpus correctness | Corpus rows remain fixture-local. |
| External-library parity | Oracle/helper outputs are bounded evidence, not broad third-party parity claims. |
| Runtime/backend governance closure | Runtime/backend-specific rows that need policy decisions are handed off to Sprint 142. |
| State-of-the-art status | Report normalization is evidence governance, not a claim of competitive status. |

## Stop Conditions

- Stop and ask if a report family cannot preserve row meaning but would need to
  be represented as a pass/fail proof.
- Stop and ask if freshness checks would require committing machine-local
  generated measurements.
- Stop and ask if runtime/backend rows require product-policy decisions that
  belong to Sprint 142.
- Stop and ask if a required validation or quality gate fails.
- Stop before broadening public claims beyond source-controlled or explicitly
  generated fixture-local evidence.

## Day 1 Notes

- Created Sprint 141 working notes and artifact directory structure.
- Re-read the Sprint 141 project-plan section and Sprint 141 day-by-day plan.
- Reviewed Sprint 138, Sprint 139, and Sprint 140 handoff artifacts for
  report rows, generated-output policy, stale-report guidance, support-tier
  semantics, and non-claim boundaries.
- Inventoried current candidate report families across corpus, oracle,
  benchmark, sentinel, guardrail, coverage, dead-code, package, install, and
  CI surfaces.
- Mapped Sprint 141 Items 1-7 to day-level owners.
- Recorded initial validation expectations, normalization boundaries,
  non-claims, and stop conditions before schema or generator implementation.

## Day 2 Notes

- Created
  `docs/planning/EPIC_12/SPRINT_141/artifacts/day2-report-family-inventory.md`.
- Inspected live report-producing surfaces in `Makefile`, `scripts/`,
  `tests/corpus/`, `.github/workflows/`, `README.md`,
  `docs/cookbook.md`, `docs/maintainer_guide.md`,
  `benchmarks/README.md`, and `INSTALL.md`.
- Promoted the Day 1 candidate list into a canonical report-family inventory
  with producer command, inputs, outputs, row identity, row meaning, owner,
  support tier, regeneration policy, and normalization risk.
- Separated source-controlled evidence rows from local generated report and
  measurement outputs.
- Flagged high-risk or non-normalizable surfaces before contract design:
  coverage backend/tool differences, CI job logs, runtime/backend sentinel
  details, package/install platform scope, and supplemental guardrail rows.
- Recorded metadata-contract questions for Day 3 around family taxonomy,
  row identity, support tier, freshness inputs, generated output status,
  warning/error severity, and Sprint 142 runtime/backend handoff.

## Day 3 Notes

- Created
  `docs/planning/EPIC_12/SPRINT_141/artifacts/day3-shared-metadata-contract.md`.
- Defined the shared metadata contract around stable `report_family`,
  `subfamily`, `row_id`, `native_row_id`, `row_origin`, command, source
  revision, platform, compiler, configuration, support tier, status,
  row meaning, claim scope, non-claims, freshness status, freshness inputs,
  and skip/defer reason fields.
- Defined a row-meaning taxonomy that preserves fixture targets, observed
  oracle comparisons, solver-backed fixture proof, local benchmark
  measurements, sentinel hard gates, sentinel advisory rows, guardrail lanes,
  dead-code classification rows, coverage summaries, package/install proof
  rows, CI lane definitions, skip/defer policy rows, and documentation
  advisory rows.
- Split fields into common required, generated-row required, source-controlled
  row required, optional, and family-specific groups.
- Defined freshness semantics for source-controlled rows, generated local
  reports, hosted CI lane definitions, optional-data rows, and local
  measurement rows.
- Defined validation severities: `error`, `warning`, `defer`, `skip`,
  `unsupported`, and `advisory`.
- Answered the Day 2 metadata-contract questions with a conservative contract:
  native family row IDs are preserved, normalized IDs are added for indexing,
  local measurement rows warn rather than fail on staleness unless an enforced
  gate owns them, and runtime/backend details are carried as metadata with
  broader governance deferred to Sprint 142.

## Day 4 Notes

- Created
  `docs/planning/EPIC_12/SPRINT_141/artifacts/day4-normalized-index-generator-design.md`.
- Chose a standalone script owner:
  `scripts/normalize_report_index.py`, with optional later Make integration
  once generator behavior is proven.
- Defined the CLI shape: `--corpus-root`, `--build-root`, `--output`,
  `--family`, `--include-generated`, `--require-generated`, `--check`, and
  `--format=tsv`.
- Defined deterministic discovery rules for source-controlled corpus metadata,
  corpus/oracle generated reports, canonical benchmark reports, performance
  sentinel reports, large-matrix guardrails, dead-code reports, coverage
  reports, package/install proof surfaces, CI workflow definitions, and
  documentation advisory rows.
- Drafted normalized output columns mapped directly to the Day 3 metadata
  contract.
- Defined deterministic row ordering by `report_family`, `subfamily`,
  `row_origin`, `row_meaning`, `native_row_id`, and `artifact_path`.
- Defined missing generated report behavior: default advisory
  `not_generated` rows, opt-in `--require-generated` failures for selected
  families, and explicit `skip`/`defer` rows for optional or non-normalizable
  families.
- Designed unit, smoke, and golden-output tests that can run without
  platform-specific benchmark, coverage, package, or hosted CI outputs.

## Day 5 Notes

- Created
  `docs/planning/EPIC_12/SPRINT_141/artifacts/day5-metadata-contract-implementation.md`.
- Added `tests/corpus/manifests/report_families.tsv` as the
  source-controlled report-family contract consumed by the future normalized
  index generator.
- Added `tests/corpus/schemas/report_index_fields.md` to document the
  contract fields, freshness policies, and guardrails.
- Extended `scripts/validate_corpus_schema.py` with report-family required
  fields, row-origin/status/freshness/row-meaning vocabularies, duplicate
  identity checks, lowercase snake-case checks, support-tier validation, and
  false-pass guardrails.
- Updated `tests/corpus/README.md` with layout, ownership, and interpretation
  notes for report-family contract rows.
- Preserved existing corpus/oracle schema compatibility: fixture, generator,
  optional-data, and expected-result validation paths remain unchanged except
  for reading the additional report-family manifest.
- Validated the implementation with `python3 -m py_compile
  scripts/validate_corpus_schema.py` and
  `python3 scripts/validate_corpus_schema.py`.
- Left generator row emission, generated-report ingestion, and Make target
  integration to Day 6 and later Sprint 141 implementation days.

## Day 6 Notes

- Created
  `docs/planning/EPIC_12/SPRINT_141/artifacts/day6-normalized-index-generator-implementation.md`.
- Added `scripts/normalize_report_index.py` as the standalone normalized
  report index generator.
- Implemented CLI options for alternate corpus/build roots, output path,
  repeatable family filters, generated-artifact inclusion/exclusion,
  required-generated checks, check mode, and TSV output.
- Implemented deterministic row IDs, deterministic row sorting, duplicate
  normalized row ID validation, contract rows, generated-artifact presence
  rows, explicit `not_generated` rows, and deferred runtime/backend rows.
- Added `tests/test_normalize_report_index.py` to cover current-repository
  smoke output, sorting, family filtering, required missing generated rows,
  alternate temporary roots, generated artifact presence, and deferred
  governance.
- Ran `python3 -m py_compile scripts/normalize_report_index.py
  tests/test_normalize_report_index.py`, `python3
  tests/test_normalize_report_index.py`, `python3
  scripts/normalize_report_index.py --no-generated --output
  build/report-index/normalized-index.tsv`, and `python3
  scripts/validate_corpus_schema.py`.
- Confirmed the smoke generator wrote 26 rows under ignored `build/` output
  and did not manufacture pass evidence for missing generated report families.
- Left deeper native corpus/oracle row mapping, benchmark/sentinel native row
  parsing, stale-report gates, and Make/CI integration to later Sprint 141
  days.

## Day 7 Notes

- Created
  `docs/planning/EPIC_12/SPRINT_141/artifacts/day7-corpus-oracle-index-integration.md`.
- Extended `scripts/normalize_report_index.py` to emit native corpus fixture,
  generator, optional-data, and expected-result rows rather than only
  family-level contract rows.
- Added generated oracle TSV ingestion for `build/corpus/oracle/*.tsv`,
  preserving native oracle row IDs, fixture keys, solver family, operation,
  comparison kind, command, source revision, generated time, platform,
  compiler, support tier, claim scope, non-claims, and skip/defer reason.
- Kept `--no-generated` deterministic by preventing it from reading existing
  ignored oracle outputs.
- Made generated oracle normalized row IDs include artifact-path context so
  overlapping local oracle files do not collide while `native_row_id` keeps the
  original oracle row ID.
- Updated `tests/test_normalize_report_index.py` to assert source-controlled
  corpus fixture, expected-result, and optional-data rows are emitted.
- Added a temp-build integration test that runs `scripts/run_corpus_oracle.py
  --include-partial-svd` and verifies normalized QR and partial-SVD oracle
  rows preserve fixture-local status, configuration, freshness, and non-claim
  boundaries.
- Validated with Python compile checks, corpus schema validation, focused
  generator tests, deterministic `--no-generated --check`, and default
  generated-discovery `--check`.

## Day 8 Notes

- Created
  `docs/planning/EPIC_12/SPRINT_141/artifacts/day8-benchmark-sentinel-guardrail-index-integration.md`.
- Extended `scripts/normalize_report_index.py` with generated-row parsers for
  canonical benchmark reports, performance sentinels, and large-matrix
  guardrails.
- Mapped canonical benchmark rows to advisory local measurements with report
  label, command, platform, compiler, build mode, thread count, artifact, and
  threshold-free non-claim context.
- Split sentinel rows by native `claim_boundary`: `local_wall_gate` rows map
  to `sentinel_hard_gate`, while threshold-free/report rows map to
  `sentinel_advisory_measurement`.
- Preserved sentinel backend request, selected backend, fallback,
  dense-kernel, panel-solver, value, baseline, and threshold metadata inside
  normalized configuration text.
- Mapped large-matrix guardrail lanes by `lane_id`, preserving reviewed pass
  rows, supplemental advisory/report rows, supplemental skips, command,
  artifact, notes, and manifest context.
- Added synthetic runtime-report fixtures to `tests/test_normalize_report_index.py`
  so benchmark/sentinel/guardrail parsing is tested without running
  machine-dependent benchmark commands.
- Validated with Python compile checks, corpus schema validation, focused
  generator tests, deterministic `--no-generated --check`, and default
  generated-discovery `--check`.

## Day 9 Notes

- Created
  `docs/planning/EPIC_12/SPRINT_141/artifacts/day9-quality-package-index-integration.md`.
- Extended `scripts/normalize_report_index.py` with coverage, dead-code, and
  package proof-owner row emitters.
- Mapped `coverage/coverage-src.info` to an advisory generated-local coverage
  row when present, with missing coverage represented by deterministic
  `not_generated` rows.
- Mapped `build/deadcode/report.tsv` rows by bucket/tool/symbol/path/line,
  preserving classification, disposition, advisory status, and
  zero-dead-code non-claims.
- Expanded the package contract into source-controlled proof-owner rows for
  `tests/test_install.sh`, `tests/test_cmake_install.sh`, `sparse.pc.in`,
  `cmake/SparseConfig.cmake.in`, and
  `scripts/static_package_deferral_check.sh`.
- Added synthetic quality/package fixtures to `tests/test_normalize_report_index.py`
  to verify package proof-owner rows, generated dead-code rows, missing
  coverage rows, and required coverage diagnostics.
- Confirmed `python3 scripts/normalize_report_index.py --family coverage
  --no-generated --require-generated coverage --check` fails with
  `required generated family missing: coverage`, as expected.

## Day 10 Notes

- Created
  `docs/planning/EPIC_12/SPRINT_141/artifacts/day10-freshness-gate-design.md`.
- Defined freshness inputs for normalized rows: source commit, branch,
  generator command, generated timestamp, platform, compiler, configuration,
  artifact path, freshness status, family policy, optional-data state, and
  corpus generator/expected-row hashes.
- Defined freshness states: `source_controlled`,
  `generated_present_unchecked`, `fresh`, `stale`, `not_generated`,
  `optional_data_skip`, `deferred`, and `unsupported`.
- Defined severity levels and exit behavior: `error`, `warning`,
  `advisory`, `skip`, `defer`, and `unsupported`.
- Created a family behavior matrix covering corpus metadata, expected rows,
  oracle generated rows, benchmarks, sentinel hard/advisory rows, guardrails,
  coverage, dead-code, package/install proof owners, CI lane definitions,
  documentation advisories, report-index missing rows, and runtime/backend
  governance.
- Designed the Day 11 CLI shape around explicit freshness checking,
  `--require-generated`, `--strict-generated`, `--advisory-ok`, and existing
  family filters.
- Defined deterministic diagnostic format:
  `freshness: <severity>: <row_id>: <state>: <reason>`.
- Designed tests for source-controlled rows, missing generated rows,
  required-generated failures, stale commit mismatch, hard-gate failure,
  advisory measurements, optional-data skips, deferred governance,
  unsupported rows, and duplicate IDs.
- Left implementation to Day 11 as planned.

## Day 11 Notes

- Created
  `docs/planning/EPIC_12/SPRINT_141/artifacts/day11-freshness-gate-implementation.md`.
- Added freshness evaluation to `scripts/normalize_report_index.py` behind
  explicit `--check-freshness`.
- Added `--strict-generated` and `--advisory-ok` CLI flags, while preserving
  existing `--check` and `--require-generated` behavior.
- Implemented deterministic diagnostics in the Day 10 format:
  `freshness: <severity>: <row_id>: <state>: <reason>`.
- Implemented freshness-state handling for source-controlled,
  generated-present, fresh, stale, not-generated, optional-data skip,
  deferred, and unsupported rows.
- Added policy fallback by report family and row meaning so generated native
  rows classify correctly even when row-specific configuration replaces the
  source-controlled `freshness_policy` field.
- Extended `tests/test_normalize_report_index.py` to cover missing oracle
  warnings, required oracle errors, stale oracle warnings/errors,
  stale benchmark advisory behavior, runtime/backend defers, and sentinel
  hard-gate failure errors.
- Confirmed `python3 scripts/normalize_report_index.py --family oracle
  --no-generated --require-generated oracle --check-freshness` emits
  expected `freshness: error` diagnostics and returns nonzero.

## Day 12 Notes

- Created
  `docs/planning/EPIC_12/SPRINT_141/artifacts/day12-documentation-alignment.md`.
- Updated `docs/maintainer_guide.md` with the normalized report-index
  workflow, freshness diagnostic interpretation, dead-code normalized row
  semantics, and package proof-owner row semantics.
- Updated `benchmarks/README.md` with benchmark/sentinel/guardrail
  normalized-index commands and local-measurement non-claims.
- Updated `tests/corpus/README.md` with corpus/oracle normalized-index
  commands, `--check-freshness`, and `--require-generated oracle`
  interpretation.
- Updated `INSTALL.md` with normalized package proof-owner rows and
  static-first source-controlled scope.
- Updated `README.md` and `docs/cookbook.md` with compact maintainer-facing
  normalized-index commands and non-claim reminders.
- Preserved the distinction between source-controlled metadata,
  generated-local report rows, advisory measurements, required generated rows,
  and Sprint 142 runtime/backend defers.

## Day 13 Notes

- Created
  `docs/planning/EPIC_12/SPRINT_141/artifacts/day13-validation-and-quality-gates.md`.
- Ran the Sprint 141 validation pass across Python compile checks, corpus
  schema validation, focused normalized-index tests, deterministic
  source-controlled index generation, generated-aware index checks,
  freshness checks, documentation hygiene, and generated-output hygiene.
- Confirmed source-controlled normalized output writes
  `build/report-index/normalized-index.tsv` with `47` rows and remains ignored
  under `build/`.
- Confirmed default generated-aware index validation reports `59` rows ok.
- Confirmed default freshness review reports stale local oracle rows as
  warnings, missing advisory coverage/dead-code/benchmark/sentinel rows as
  advisory diagnostics, optional-data rows as skips, and runtime/backend
  governance as a Sprint 142 defer.
- Confirmed strict required-generated behavior by running
  `python3 scripts/normalize_report_index.py --family coverage
  --require-generated coverage --check-freshness`; the wrapper observed the
  expected nonzero exit and `freshness: error` for missing generated coverage.
- Confirmed no C or header files changed during Day 13, so the required gate
  for this day remained script/docs/report validation rather than
  `make format && make lint && make test`.
- Cleaned Python `__pycache__` directories created by compile/test checks.

## Day 14 Notes

- Created
  `docs/planning/EPIC_12/SPRINT_141/artifacts/day14-closeout-and-sprint142-handoff.md`.
- Reran final Sprint 141 validation after the Day 13 validation artifact and
  Day 14 closeout work:
  - `python3 -m py_compile scripts/validate_corpus_schema.py
    scripts/normalize_report_index.py tests/test_normalize_report_index.py`
  - `python3 scripts/validate_corpus_schema.py`
  - `python3 tests/test_normalize_report_index.py`
  - `python3 scripts/normalize_report_index.py --no-generated --output
    build/report-index/normalized-index.tsv`
  - `python3 scripts/normalize_report_index.py --no-generated --check`
  - `python3 scripts/normalize_report_index.py --check`
  - `python3 scripts/normalize_report_index.py --check-freshness`
  - `python3 scripts/normalize_report_index.py --family runtime_backend
    --check-freshness`
  - `python3 scripts/normalize_report_index.py --family coverage --family
    deadcode --family package --check-freshness`
  - `python3 scripts/normalize_report_index.py --family benchmark --family
    sentinel --family guardrail --check-freshness`
- Confirmed source-controlled deterministic index output remains `47` rows,
  generated-aware index output remains `59` rows, and default freshness exits
  successfully with advisory, warning, skip, and defer diagnostics as
  designed.
- Confirmed strict required-generated oracle behavior by running
  `python3 scripts/normalize_report_index.py --family oracle --no-generated
  --require-generated oracle --check-freshness`; the wrapper observed the
  expected nonzero exit and `freshness: error` diagnostics for missing oracle
  generated rows.
- Confirmed `build/report-index/normalized-index.tsv` remains ignored.
- Reviewed Sprint 141 artifacts and report-family rows for claim-boundary
  consistency. The closeout preserves non-claims for portable performance,
  broad solver/corpus correctness, package-manager/shared-library/ABI
  support, hosted CI proof from local rows, zero-dead-code status, coverage
  completeness, and state-of-the-art status.
- Closed Sprint 141 with one narrow Sprint 142 handoff:
  runtime/backend governance and precedence policy.
