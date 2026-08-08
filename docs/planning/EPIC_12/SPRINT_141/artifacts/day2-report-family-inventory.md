# Day 2 Report Family Inventory

## Purpose

Day 2 converts the Day 1 intake into a canonical report-family inventory. The
inventory names each current report-producing surface, the command or source
that owns it, where rows or artifacts appear, what each row means, who owns
the surface, and which families are safe to normalize now versus which need
warnings, defer rows, or Sprint 142 runtime/backend handoff.

This remains a documentation and planning artifact. It does not change report
scripts, schemas, tests, CI workflows, generated artifacts, or public claims.

## Command-To-Output Map

| Family | Producer command or source | Primary inputs | Current output | Row identity |
| --- | --- | --- | --- | --- |
| Corpus fixture manifests | source-controlled `tests/corpus/manifests/fixtures.tsv` | fixture metadata, generator key, validation command, support tier | committed TSV rows | `fixture_key` |
| Corpus generator manifests | source-controlled `tests/corpus/manifests/generators.tsv` | deterministic generator metadata, canonical text hashes | committed TSV rows | `generator_key` |
| Optional data policy | source-controlled `tests/corpus/manifests/optional_data.tsv` | optional external-data availability and defer policy | committed TSV rows | `optional_data_key` |
| Corpus expected results | source-controlled `tests/corpus/expected/*.tsv` | fixture-local expected numerical or diagnostic rows | committed TSV rows | `oracle_row_id` |
| Corpus schema validation | `python3 scripts/validate_corpus_schema.py` | corpus manifests, expected rows, schema vocabulary | console validation result | validator error or success state |
| Corpus/oracle reports | `python3 scripts/run_corpus_oracle.py` | corpus manifests, expected rows, optional data, generated fixture registry | ignored `build/corpus/oracle/*.tsv`, `build/corpus-reports/index.tsv`, `build/corpus-reports/skips.tsv`, `build/corpus-reports/manifest.txt` | `oracle_row_id` and `report_row_id` |
| Solver-backed QR rows | `python3 scripts/run_corpus_oracle.py --include-solver-qr` | built static library, QR corpus fixture, expected rows | ignored corpus oracle/report rows | `_qr_*` solver-backed `oracle_row_id` |
| Solver-backed partial-SVD rows | `python3 scripts/run_corpus_oracle.py --include-partial-svd` | generated partial-SVD fixture and expected rows | ignored corpus oracle/report rows | partial-SVD expected `oracle_row_id` |
| Canonical benchmarks | `make bench-canonical-report` | `bench_refactor_csc`, `bench_chol_csc`, `bench_iterative_reuse`, `bench_eigs_reuse` | ignored `build/bench-reports/canonical/*.csv`, `index.tsv`, `manifest.txt` | `artifact` in `index.tsv` |
| Performance sentinels | `make performance-sentinels` | `bench_chol_csc`, `bench_amd_qg`, `bench_reorder`, wall-check baseline | ignored `build/bench-reports/sentinels/sentinels.tsv`, `manifest.txt`, optional command output files | `sentinel_id` plus metric |
| Large-matrix guardrails | `make large-matrix-guardrails` | `test_graph`, `test_reorder_nd`, `test_reorder_amd_qg`, `bench_reorder`, `bench_amd_qg` | ignored `build/bench-reports/large-matrix-guardrails/index.tsv`, `manifest.txt`, test logs, CSVs | `lane_id` |
| Dead-code report | `make deadcode-report` and `make deadcode-check` | CMake compile database, `deadcode_workflow.sh`, `deadcode_report.py`, cppcheck/xunused evidence | ignored `build/deadcode/report.md`, `report.tsv`, raw evidence files | dead-code finding/classification row |
| Coverage report | `make coverage`, `make coverage-lcov`, `make coverage-gcovr` | instrumented test build, lcov or gcovr backend | `coverage/coverage-src.info`, `coverage/html/index.html`, summary output | file/line coverage summary rows |
| Make install/pkg-config proof | `bash tests/test_install.sh` | Make static library, install target, generated version header, `sparse.pc` | temporary install tree and console pass/fail log | named install assertion |
| CMake install/export proof | `bash tests/test_cmake_install.sh` and CI CMake snippets | CMake build/install/export metadata and maintained example | temporary install tree and console pass/fail log | named CMake install assertion |
| Static package deferral proof | `bash scripts/static_package_deferral_check.sh` | `CMakeLists.txt`, `cmake/SparseConfig.cmake.in`, `sparse.pc.in`, README, INSTALL, maintainer docs | console pass/fail log | named package-contract assertion |
| CI summary lanes | `.github/workflows/*.yml` | workflow definitions, hosted runner/toolchain setup | hosted check status and logs | workflow job name |

## Canonical Inventory

| Family | Owner | Support tier today | Row meaning | Regeneration policy | Normalization assessment |
| --- | --- | --- | --- | --- | --- |
| Corpus manifests | Corpus maintainer plus solver-owner review for numerical semantics | source-controlled fixture-local | Eligible fixture/generator/optional-data evidence lane, not observed pass evidence | update rows only with explicit fixture/generator revision | Safe to normalize. |
| Corpus expected rows | Corpus maintainer plus solver owner | source-controlled fixture-local | Expected result targets for a named fixture and operation | update with fixture or comparison-semantics revision | Safe to normalize. |
| Corpus/oracle generated rows | Corpus/report maintainer | `local_only` unless later promoted | Local observed comparison rows tied to command, commit, platform, compiler, configuration, support tier, and claim scope | regenerate after source, manifest, expected row, generator, command, platform, or compiler changes | Safe to normalize with freshness fields. |
| Solver-backed QR and partial-SVD rows | Solver owner plus corpus/report maintainer | `local_only` generated proof | Local solver-backed observed rows for named fixtures only | regenerate after solver, proof-owner, corpus, expected-row, generator, or command changes | Safe to normalize with explicit generated/support-tier status. |
| Canonical benchmarks | Benchmark maintainer | threshold-free local measurement | Artifact map for current-machine benchmark CSV snapshots | regenerate per branch/run; compare across local or CI artifacts only | Safe to normalize as measurement rows, not claims. |
| Performance sentinels | Benchmark/performance maintainer | mixed: one reviewed thresholded wall gate plus threshold-free local context | Sentinel rows with pass/fail/skip/report state and metric context | regenerate before interpreting local performance context | Normalize with separate hard-gate and advisory-row semantics. |
| Large-matrix guardrails | Graph/reorder maintainer plus benchmark maintainer | reviewed structural lanes plus supplemental opt-in rows | Structural test/report lane map; supplemental rows are threshold-free local context | regenerate after graph/reorder, named-matrix, command, or supplemental-mode changes | Normalize reviewed lanes now; mark supplemental rows explicitly. |
| Dead-code report | Maintainer quality owner | Linux enforced report-completeness lane plus local generated artifacts | Classified static-analysis findings and completeness checks, not zero-findings proof | regenerate through serialized dead-code targets after source/build changes | Normalize as quality report, not removal-ready claim. |
| Coverage report | Quality owner | Linux supplemental, local tree-mutating | Line coverage summary for active backend/toolchain/test surface | regenerate from clean tree after source/test changes or backend change | High risk; normalize only with backend/tool/platform fields and advisory semantics. |
| Install/package proofs | Package maintainer | Linux reviewed static-first, macOS/Windows supplemental confidence as documented | Static archive install/export/package metadata/downstream consumer assertions | rerun after build/install/package metadata/docs changes | Normalize with platform and static-first scope. |
| CI summary lanes | CI maintainer | workflow-defined reviewed/supplemental split | Hosted job status and configured lane intent | GitHub Actions generated; source repo owns job definitions only | Index lane definitions; do not treat job logs as source-controlled report rows. |

## Source-Controlled Versus Generated Evidence

| Category | Examples | Interpretation |
| --- | --- | --- |
| Source-controlled metadata | `tests/corpus/manifests/*.tsv`, `tests/corpus/expected/*.tsv`, schemas, workflow YAML, package templates | Defines eligible evidence, configuration, or lane intent; does not become observed pass evidence by itself. |
| Source-controlled proof owners | `tests/test_qr_corpus.c`, `tests/test_svd_partial_corpus.c`, install test scripts, package deferral script | Compiled/scripted checks can produce pass evidence when run in a documented environment. |
| Ignored generated reports | `build/corpus-reports/`, `build/bench-reports/`, `build/deadcode/`, `coverage/` | Local or CI artifacts tied to command, commit, platform, compiler, configuration, support tier, and freshness. |
| Hosted CI logs | GitHub Actions job output | Authoritative for that run, but not a source-controlled row unless summarized by a maintained artifact. |

## Non-Normalizable Or High-Risk Semantics

| Surface | Risk | Day 3 contract implication |
| --- | --- | --- |
| Coverage reports | lcov/gcovr backend, compiler, and tree-mutating build mode change row meaning | Require backend, compiler, platform, threshold, and advisory/support-tier fields. |
| Performance sentinel rows | `S5` hard wall-check gate and `S2` threshold-free rows have different status semantics | Require `row_meaning`, `gate_kind`, and advisory-versus-enforced fields. |
| Large-matrix supplemental rows | Opt-in threshold-free measurements are useful but not reviewed pass evidence | Require supplemental support tier and skip/defer status preservation. |
| Dead-code report | Completeness check is not zero-findings or removal-ready proof | Require quality-report row meaning and non-claim fields. |
| Package/install proofs | Linux reviewed, macOS supplemental, and Windows CMake-first confidence are different evidence tiers | Require platform, workflow lane, package surface, and static-first scope fields. |
| CI job logs | Logs are generated outside the repo and may not map cleanly to row artifacts | Index workflow lane definitions first; defer log normalization unless a source-controlled summary exists. |
| Runtime/backend fields | Backend request/selected/fallback fields are product-governance sensitive | Preserve raw metadata but hand broader governance to Sprint 142. |

## Metadata-Contract Questions For Day 3

1. Should `report_family` use a small enum such as `corpus`, `oracle`,
   `benchmark`, `sentinel`, `guardrail`, `coverage`, `deadcode`, `package`,
   `install`, and `ci`, with a separate `subfamily` field?
2. Should row identity be normalized as `row_id` plus family-specific
   `source_row_id`, or should each family keep its native identity field?
3. Which fields are required for every generated row: command, source commit,
   branch, platform, compiler, configuration, generated time, support tier,
   status, row meaning, claim scope, and non-claims?
4. How should the freshness gate compare source-controlled input hashes
   against generated reports without requiring generated reports to be
   committed?
5. Should missing optional-data rows produce `skip`, `defer`, or advisory
   statuses in the normalized index?
6. Which families can fail CI on staleness, and which should warn because they
   are local measurement or supplemental report context?
7. What is the minimum schema that lets Sprint 142 consume runtime/backend
   rows without reinterpreting Sprint 141 evidence?

## Day 2 Completion Criteria

| Criterion | Status | Evidence |
| --- | --- | --- |
| Benchmark, sentinel, guardrail, coverage, dead-code, package, corpus, and oracle outputs are accounted for. | Complete | Command-to-output map and canonical inventory above. |
| Each family has a documented row meaning or a reason it cannot be normalized yet. | Complete | Canonical inventory and high-risk semantics tables above. |
| Inventory is concrete enough to drive schema design. | Complete | Metadata-contract questions and required field implications are recorded for Day 3. |
