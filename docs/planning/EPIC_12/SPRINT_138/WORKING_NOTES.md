# Sprint 138 Working Notes

## Sprint Goal

Build the maintained numerical corpus architecture and first durable
oracle/report lane before adding broad fixture volume.

Sprint 138 is the first Epic 12 implementation sprint. It must convert the
Sprint 137 evidence contracts into maintained repository structure, schema,
validation, and documentation for one deterministic corpus lane. The sprint
must not broaden corpus, solver, report, package, platform, performance, or
state-of-the-art claims before fixture-local evidence exists.

## Starting Constraints

- Start from the Sprint 137 selected target: maintained corpus/oracle contract
  with one durable deterministic fixture lane and explicit skip/defer
  semantics.
- Use the Sprint 137 Day 8 corpus/oracle templates rather than redefining row
  meanings.
- Preserve Day 12 public claim freeze: corpus/oracle rows are fixture-local
  evidence only.
- Keep optional external data disabled or skipped by default; skip/defer rows
  are policy evidence, not solver pass evidence.
- Keep generated outputs out of source control unless this sprint explicitly
  promotes a maintained artifact.
- If any `.c` or `.h` file changes, run `make format && make lint && make
  test`. Documentation-only changes require `git diff --check` and focused
  Markdown hygiene/link checks.

## Input Artifact Inventory

| Input | Role in Sprint 138 |
| --- | --- |
| `docs/planning/EPIC_12/PROJECT_PLAN.md` Sprint 138 | Defines Sprint 138 items, deliverables, prerequisites, and 168-hour estimate. |
| `docs/planning/EPIC_12/SPRINT_138/PLAN.md` | Provides day-level execution order and validation expectations. |
| `docs/planning/EPIC_12/SPRINT_137/RETROSPECTIVE.md` | Summarizes Sprint 137 selected targets, residuals, validation, and Sprint 138 handoff. |
| `docs/planning/EPIC_12/SPRINT_137/artifacts/day7-gap-selection-decision.md` | Selects the Sprint 138 maintained corpus/oracle target and claim boundaries. |
| `docs/planning/EPIC_12/SPRINT_137/artifacts/day8-corpus-oracle-evidence-templates.md` | Defines fixture metadata, generated-matrix metadata, optional-data skip/defer rows, oracle rows, and failure interpretation. |
| `docs/planning/EPIC_12/SPRINT_137/artifacts/day9-report-index-freshness-templates.md` | Defines report/freshness fields that Sprint 138 rows should support for Sprint 141. |
| `docs/planning/EPIC_12/SPRINT_137/artifacts/day11-quality-surface-map.md` | Defines required and supplemental validation by touched surface. |
| `docs/planning/EPIC_12/SPRINT_137/artifacts/day12-public-claim-freeze.md` | Freezes public claim boundaries before corpus implementation. |
| `docs/planning/EPIC_12/SPRINT_137/artifacts/day13-handoff-synthesis-sprint138-readiness.md` | Provides Sprint 138 minimum implementation checklist, stop conditions, and later-sprint handoff notes. |
| `docs/planning/EPIC_12/SPRINT_137/artifacts/day14-closeout-and-sprint138-readiness.md` | Publishes final Sprint 138 readiness criteria and residual register. |
| `README.md`, `INSTALL.md`, `docs/maintainer_guide.md`, `benchmarks/README.md` | Current public and maintainer claim surfaces that must not be widened without evidence. |
| `tests/`, `scripts/`, `benchmarks/`, `examples/`, `src/`, `include/` | Candidate implementation, validation, corpus, report, and adoption surfaces. |

## Day-Level Ownership

| Day | Owner focus | Project-plan items |
| --- | --- | --- |
| 1 | Scope setup, Sprint 137 handoff inventory, validation expectations, non-claims | Items 1-7 |
| 2 | Fixture taxonomy draft and first-lane class selection | Item 1 |
| 3 | Taxonomy review, promotion gates, and claim boundaries | Item 1 |
| 4 | Corpus storage, manifest, optional-data, expected-result, and report path design | Item 2 |
| 5 | Corpus directory and manifest skeleton implementation | Item 2 |
| 6 | Oracle row schema and comparison semantics design | Item 3 |
| 7 | Oracle schema implementation and validation helper | Item 3 |
| 8 | First deterministic fixture lane design and generator metadata | Item 4 |
| 9 | First corpus lane implementation | Item 4 |
| 10 | Maintained oracle/report command implementation | Item 4 |
| 11 | Optional-data skip/defer semantics implementation | Item 5 |
| 12 | Focused corpus/oracle validation and required quality gates | Item 6 |
| 13 | Corpus documentation and Sprint 139 QR handoff | Item 7 |
| 14 | Sprint 138 closeout, residuals, validation summary, and working-notes completion | Item 7 |

## Initial Validation Expectations

| Change type | Required validation |
| --- | --- |
| Sprint 138 planning artifacts only | `git diff --check`, trailing-whitespace scan under `docs/planning/EPIC_12/SPRINT_138`, and focused Markdown link/path validation under `docs/planning/EPIC_12`. |
| Corpus manifests, schemas, oracle rows, or generated report indexes | Schema/field validation when available, `git diff --check`, and corpus/report non-claim scan. |
| Public or maintainer documentation | `git diff --check`, focused Markdown link/path validation, and claim-boundary scan against Sprint 137 Day 12 non-claims. |
| Python scripts | `python3 -m py_compile <script>` plus focused command validation when feasible. |
| Shell scripts | `bash -n <script>` plus focused command validation when feasible. |
| Makefile, CMake, pkg-config, install, or package edits | Relevant package/install/CMake proof commands plus static/shared support-boundary review. |
| CI workflow edits | Workflow syntax or structural review plus hosted-runner support-tier notes; do not treat unrun hosted lanes as passed local evidence. |
| Benchmark or generated report execution | Capture command, platform, compiler/configuration, source commit, row meaning, freshness, support tier, and skip/defer status. |
| `.c` or `.h` edits | `make format && make lint && make test` after focused tests needed for the touched implementation. |

## Sprint-Level Non-Claim Register

| Non-claim | Sprint 138 boundary |
| --- | --- |
| Broad corpus completeness | Sprint 138 implements one durable corpus lane, not broad fixture volume. |
| SuiteSparse or external corpus parity | Optional external data remains skip/defer-gated unless configured and proven; no broad SuiteSparse claim is earned. |
| Broad QR behavior | Sprint 138 prepares corpus/oracle inputs for Sprint 139; it does not close QR residuals. |
| Broad partial-SVD behavior | Sprint 138 prepares corpus/oracle inputs for Sprint 140; it does not close partial-SVD residuals. |
| Report index as release proof | Sprint 138 rows may feed future report indexes; they do not prove release readiness, broad correctness, coverage completeness, or performance. |
| Package/ABI/platform support | Corpus implementation must not imply package, ABI, loader, package-manager, macOS, Windows, or platform-parity support. |
| Portable performance | Corpus/report rows are fixture-local evidence, not portable timing, speedup, scalability, or memory claims. |
| Coverage/dead-code completeness | Corpus validation does not widen coverage completeness or dead-code removal-readiness claims. |
| State of the art | No state-of-the-art claim is earned by the first corpus lane. |

## Day 1 Notes

- Created the Sprint 138 working-notes baseline and artifact directory.
- Re-read the Sprint 138 section of `docs/planning/EPIC_12/PROJECT_PLAN.md`.
- Re-read the Sprint 137 Sprint 138 readiness handoff and closeout artifacts.
- Mapped Sprint 138 Items 1-7 to day-level owners across Days 1-14.
- Recorded inherited Sprint 137 evidence contracts before taxonomy or storage
  implementation begins.
- Recorded validation expectations for planning docs, corpus schemas, public
  docs, scripts, build/package changes, CI workflows, generated reports, and
  `.c`/`.h` changes.
- Recorded sprint-level non-claims that keep Sprint 138 bounded to
  fixture-local corpus/oracle evidence.
- No source files, public documentation, workflows, scripts, build files,
  corpus data, or support claims were changed on Day 1 beyond Sprint 138
  planning artifacts.
- No `.c` or `.h` files changed, so the full C quality gate was not required.

## Day 2 Notes

- Wrote
  `docs/planning/EPIC_12/SPRINT_138/artifacts/day2-fixture-taxonomy-draft.md`.
- Drafted maintained matrix-class axes for symmetry, definiteness, rank,
  rectangularity, conditioning, scaling, sparsity pattern, graph shape, RHS
  policy, expected behavior, and data provenance.
- Mapped the taxonomy back to Sprint 137 Day 8 fixture metadata fields so Day 3
  can review candidate extensions instead of reconciling new row meanings.
- Marked `shape_class`, `graph_shape`, and `expected_failure_class` as
  candidate review fields rather than implemented schema changes.
- Selected `qr_rank_deficient_6x4_nullspace_v1` as the first durable fixture
  lane candidate: generated, rectangular 6x4, expected rank 3, nullity 1,
  structured sparse, moderate conditioning, unit scale, and fixture-local
  success behavior.
- Recorded first-lane non-claims for broad QR correctness, raw basis parity,
  minimum-norm or least-squares closure, SuiteSparse/external parity, SVD
  correctness, corpus completeness, and public state-of-the-art claims.
- Recorded out-of-scope residual classes for optional external data,
  partial-SVD clustered spectra, direct-solver expansions, random generators,
  performance sentinels, graph/order fixtures, parser failures, iterative
  non-convergence, and package/platform fixtures.
- Captured QR, partial-SVD, and report-index dependency notes for Sprint 139,
  Sprint 140, and Sprint 141.
- No `.c` or `.h` files changed, so the full C quality gate was not required.

## Day 3 Notes

- Wrote
  `docs/planning/EPIC_12/SPRINT_138/artifacts/day3-taxonomy-review-claim-boundaries.md`.
- Compared the Day 2 taxonomy against current QR tests, SVD/partial-SVD tests,
  external dense-reference helpers, Matrix Market fixtures, bundled
  SuiteSparse-style data, examples, and Sprint 137 evidence templates.
- Finalized the Sprint 138 taxonomy by accepting Sprint 137 Day 8 fields as
  the implementation contract and keeping `shape_class`, `graph_shape`, and
  `expected_failure_class` out of the stored first-lane schema.
- Confirmed `qr_rank_deficient_6x4_nullspace_v1` as the first durable fixture
  lane with generator key `qr_rank_deficient_6x4_nullspace_generator_v1`.
- Recorded promotion gates for taxonomy fit, stable identity,
  reproducibility, oracle semantics, support-tier evidence, skip/defer
  handling, claim boundaries, validation paths, and documentation handoff.
- Wrote fixture-local claim boundaries for the selected QR lane and future
  SVD, direct-solver, optional external-data, graph/order, and report-index
  lanes.
- Preserved QR, partial-SVD, and report-index handoff notes for Sprints 139,
  140, and 141 before Day 4 storage layout begins.
- No `.c` or `.h` files changed, so the full C quality gate was not required.

## Day 4 Notes

- Wrote
  `docs/planning/EPIC_12/SPRINT_138/artifacts/day4-corpus-storage-layout-design.md`.
- Designed the maintained source layout under `tests/corpus/` with manifest,
  expected-result, schema, README, and future promoted-fixture paths.
- Designed generated output paths under ignored `build/corpus/` and
  `build/corpus-reports/`, following the existing `build/bench-reports/...`
  convention without mixing corpus rows into benchmark outputs.
- Chose `tests/corpus/manifests/fixtures.tsv`,
  `tests/corpus/manifests/generators.tsv`, and
  `tests/corpus/manifests/optional_data.tsv` as the maintained row paths.
- Chose `tests/corpus/expected/<fixture_key>.tsv` as the committed
  expected-result path pattern for stable, small, deterministic rows.
- Chose `SPARSE_CORPUS_OPTIONAL_DATA_DIR` as the optional external-data root
  and kept optional payloads outside the source-controlled corpus tree.
- Defined naming rules for fixture keys, fixture families, generator keys,
  optional-data keys, oracle row IDs, report row IDs, and expected-result
  files.
- Recorded the Day 5 implementation checklist, including the need to update
  `.gitignore` only if a future committed corpus `.mtx` fixture path is
  promoted.
- No `.c` or `.h` files changed, so the full C quality gate was not required.

## Day 5 Notes

- Wrote
  `docs/planning/EPIC_12/SPRINT_138/artifacts/day5-corpus-storage-layout-implementation.md`.
- Added the maintained corpus skeleton under `tests/corpus/` with README,
  manifest, expected-result, schema, and future promoted-fixture paths.
- Added `tests/corpus/manifests/fixtures.tsv` with the first-lane
  `qr_rank_deficient_6x4_nullspace_v1` placeholder row.
- Added `tests/corpus/manifests/generators.tsv` with the first-lane
  `qr_rank_deficient_6x4_nullspace_generator_v1` placeholder row.
- Added `tests/corpus/manifests/optional_data.tsv` as a header-only
  optional external-data policy skeleton.
- Added expected-result placeholders for first-lane rank, nullity, and future
  projector/subspace residual comparison.
- Added `tests/corpus/schemas/fixture_fields.md` for fixture, generator, and
  optional-data field definitions while leaving oracle schema finalization to
  Day 6.
- Confirmed no generated `build/corpus/` or `build/corpus-reports/` outputs,
  optional external payloads, or committed corpus Matrix Market files were
  added.
- No `.c` or `.h` files changed, so the full C quality gate was not required.

## Day 6 Notes

- Wrote
  `docs/planning/EPIC_12/SPRINT_138/artifacts/day6-oracle-row-schema-design.md`.
- Defined the separation between fixture manifest rows, source-controlled
  expected-result rows, and generated observed oracle rows.
- Designed observed oracle TSV fields for row identity, fixture key, solver
  family, operation, comparison kind, command, commit, branch, timestamp,
  platform, compiler, configuration, support tier, expected and observed
  results, tolerance, comparison status, failure class, skip/defer reason,
  claim scope, and non-claims.
- Defined comparison kinds for value, residual norm, rank, nullity, subspace
  distance, status, diagnostic, and local measurement rows.
- Defined tolerance kinds for exact, absolute, relative, mixed, projector,
  status-only, and not-applicable comparisons.
- Defined comparison statuses and failure classes so skip, defer,
  unsupported, and xfail rows cannot be counted as solver passes.
- Designed first-lane QR oracle rows for rank, nullity, and
  projector/subspace residual comparison without raw QR basis parity.
- Recorded serialization rules, validation expectations, and the Day 7
  implementation handoff.
- No `.c` or `.h` files changed, so the full C quality gate was not required.

## Day 7 Notes

- Wrote
  `docs/planning/EPIC_12/SPRINT_138/artifacts/day7-oracle-schema-implementation.md`.
- Added `tests/corpus/schemas/oracle_fields.md` for observed oracle fields,
  comparison status semantics, failure classes, and first-lane row IDs.
- Updated the first-lane expected-result TSV with `oracle_row_id` and
  `comparison_kind` columns so the rank, nullity, and projector/subspace rows
  match the Day 6 schema design.
- Added `scripts/validate_corpus_schema.py` to mechanically validate corpus
  TSV widths, required fields, selected enum values, fixture/generator
  references, expected-result fixture references, and placeholder status
  boundaries.
- Updated corpus README files with oracle schema and validation-helper
  ownership notes.
- Confirmed no generated oracle rows, corpus report outputs, optional external
  data, committed corpus Matrix Market files, or pass evidence were added.
- No `.c` or `.h` files changed, so the full C quality gate was not required.
