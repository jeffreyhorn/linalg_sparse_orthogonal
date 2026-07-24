# Sprint 131 Day 6 - Report Index Requirements

## Purpose

Day 6 defines requirements for indexes over benchmark, coverage, dead-code,
large-matrix, oracle, guardrail, and planning-report artifacts. The goal is to
make report ownership, freshness, input corpus, output class, support tier,
and failure meaning visible without changing benchmark semantics, CI policy,
or public performance claims.

This is a documentation-only requirements artifact. It does not add a report
generator, alter existing report scripts, change benchmark commands, change
coverage targets, change dead-code workflow behavior, or promote any report
row to a broader public claim.

## Report-Family Requirements Matrix

| Report family | Current source | Current outputs | Primary audience | Required index fields | Strategy |
| --- | --- | --- | --- | --- | --- |
| Canonical benchmark report | `make bench-canonical-report`, `scripts/bench_canonical_report.sh` | `build/bench-reports/canonical/*.csv`, `index.tsv`, `manifest.txt` | Maintainer local comparison and CI artifact capture | surface, category, report label, timestamp, git commit/branch, artifact, relative path, command, support tier, non-claim note | Use existing generated `index.tsv`; enrich only through a higher-level curated index until script changes are justified. |
| Performance sentinel bundle | `make performance-sentinels`, `scripts/performance_sentinels.sh` | `sentinels.tsv`, `manifest.txt`, optional `wall_check.txt`, optional `bench_chol_csc_nos4.csv` | Maintainer local regression triage | sentinel id, status, command, build mode, OMP threads, fixture, metric, value, baseline, threshold, notes, support tier, local-only flag | Curated first; generated schema already exists but differs from canonical benchmark index. |
| Large-matrix guardrails | `make large-matrix-guardrails`, `scripts/large_matrix_guardrails.sh` | `index.tsv`, `manifest.txt`, reviewed test logs, bounded CSV shape report, optional supplemental CSVs | Guardrail and release-readiness review | lane id, status, category, command, artifact, notes, supplemental mode, reviewed/supplemental split, input corpus | Use existing generated `index.tsv`; Day 7 should consider this as a first generated index candidate because reviewed/supplemental semantics are already explicit. |
| Coverage reports | `make coverage`, `make coverage-lcov`, `make coverage-gcovr` | `coverage/coverage.info`, `coverage/coverage-src.info`, `coverage/html/`, gcovr HTML, summary output | Coverage review and risk triage | command, backend, tree-mutating flag, threshold, aggregate percentage, source filters, generated artifacts, reset requirement, support tier | Defer generated index until Day 8 coverage architecture defines risk ranking and reviewed/supplemental split. |
| Dead-code reports | `make deadcode-report`, `make deadcode-check`, `scripts/deadcode_report.py` | `build/deadcode/report.md`, `report.tsv`, raw `coverage-notes.txt`, `xunused.txt`, `cppcheck.txt` | Maintainer dead-code and coverage-gap triage | command, artifacts dir, compile database, TU counts, bucket counts, coverage gaps, check status, serial-execution requirement | Curated first; generated index should wait until Day 9 dead-code architecture confirms bucket ownership. |
| External-reference oracle artifacts | `tests/*_external_dense_reference.py`, Sprint 120-130 artifacts | Helper fixture keys, helper outputs, planning artifacts | Claim-gate and fixture ownership review | fixture key, helper path, output class, owner test, skip behavior, oracle source, support tier, claim boundary | Curated index candidate; no current generated report artifact exists. |
| Planning sprint artifacts | `docs/planning/EPIC_11/SPRINT_*/artifacts/*.md` | Markdown evidence, policy, validation, closeout artifacts | Maintainer historical traceability | sprint/day, artifact path, evidence class, owner focus, validation status, residual handoff, non-claims | Curated or generated later from filenames plus headings; defer until report schema is stable. |
| Benchmark-local CSV outputs | `benchmarks/*.c` direct runs | Individual CSV/stdout outputs | Benchmark owners | benchmark, command, fixture, schema owner, output fields, threshold status, support tier | Defer broad indexing; canonical and guardrail scripts already provide bounded report surfaces. |

## Generated Versus Curated Decisions

| Candidate | Decision | Reason | Day 7 implication |
| --- | --- | --- | --- |
| Existing large-matrix guardrail `index.tsv` | Generated candidate | Already emits reviewed/supplemental rows, stable lane IDs, artifacts, commands, and notes. | Strong first generated/index artifact candidate. |
| Existing canonical benchmark `index.tsv` | Generated but narrow | Already stable but benchmark-only and intentionally threshold-free. | Useful reference schema, but not enough for corpus/coverage architecture. |
| Existing performance sentinel `sentinels.tsv` | Curated first | Metric-level schema differs from artifact-level index rows and includes a hard wall-check status. | Normalize later only after report row type is explicit. |
| Coverage index | Deferred | Coverage is tree-mutating and needs Day 8 risk architecture before row meanings are stable. | Do not generate before reviewed/supplemental coverage split exists. |
| Dead-code index | Deferred | Dead-code bucket ownership and coverage-gap relationship need Day 9 architecture. | Do not generate before bucket semantics are confirmed. |
| External-reference helper index | Curated first | Helper fixture metadata is available, but no generated artifact exists and output classes need schema enforcement. | Possible curated artifact or later script after schema stabilizes. |
| Planning artifact index | Deferred | High value for traceability, but lower priority than runtime/report surfaces. | Revisit after first report index validates schema. |

## Index Field Schema

### Required Artifact-Level Fields

| Field | Meaning |
| --- | --- |
| `report_key` | Stable report family or lane key. |
| `row_type` | `artifact`, `metric`, `fixture`, `expected-error`, `skip`, `policy`, or `gap`. |
| `source_command` | Make target, script invocation, helper invocation, or curated source. |
| `artifact_path` | Path relative to report directory or repository root. |
| `report_owner` | Script, Makefile target, helper, or docs artifact owner. |
| `input_corpus` | Fixture key, matrix path, generated family, report-only input, or `none`. |
| `output_class` | CSV, TSV, manifest, HTML, helper vector, helper scalar, log, markdown, or policy row. |
| `support_tier` | Reviewed, smoke, supplemental, benchmark, experimental, deferred, or unsupported. |
| `status` | pass, fail, report, skip, deferred, stale, unknown, or not-run. |
| `freshness_rule` | Timestamp, git commit/branch, source command, checksum, validation command, or curated review date. |
| `failure_meaning` | What failure/staleness means and what it does not mean. |
| `claim_boundary` | One-sentence bounded interpretation and non-claim. |

### Optional Fixture/Metric Fields

| Field | Meaning |
| --- | --- |
| `solver_family` | Primary owner solver/report family. |
| `fixture_owner` | Test/helper owner for fixture rows. |
| `oracle_source` | Analytic, external helper, product observed, none, or unknown. |
| `oracle_output` | Solve vector, singular values, projector values, rank scalar, report row, or not applicable. |
| `metric_name` | Metric field for sentinel/coverage/dead-code metric rows. |
| `metric_value` | Captured metric value. |
| `baseline` | Baseline when the report has a threshold gate. |
| `threshold` | Threshold when applicable. |
| `platform_context` | Platform, compiler, backend, OMP, or tree-mutating context. |
| `docs_owner` | Documentation path if wording changes depend on the row. |

## Owner And Freshness Policy

| Report family | Owner | Freshness requirement | Stale or missing meaning |
| --- | --- | --- | --- |
| Canonical benchmark report | `scripts/bench_canonical_report.sh`, `benchmarks/README.md` | Generated timestamp plus git commit/branch in `index.tsv` and `manifest.txt`. | Stale artifact blocks local comparison; it is not a performance failure. |
| Performance sentinels | `scripts/performance_sentinels.sh`, `scripts/wall_check.sh` | Generated timestamp, platform, compiler, build mode, OMP, backend env, and wall-check baseline. | Failed S5 is existing wall-check failure; skipped S2 is report incompleteness, not correctness failure. |
| Large-matrix guardrails | `scripts/large_matrix_guardrails.sh`, Makefile target | Generated timestamp, git commit/branch, platform, compiler, supplemental flag, lane rows. | Reviewed lane failure is guardrail failure; supplemental skip is expected unless opt-in enabled. |
| Coverage reports | Makefile coverage targets | Command/backend, tree-mutating status, threshold, output path, and reset requirement. | Stale/missing output means no current coverage evidence; not solver behavior failure. |
| Dead-code reports | `scripts/deadcode_workflow.sh`, `scripts/deadcode_report.py` | Serialized run through `make deadcode-report` and `make deadcode-check`; report completeness validation. | Check failure means report contract or uncategorized finding issue; not automatic removal proof. |
| External-reference helper index | Helper script plus owner test file | Helper path, fixture key, output class, platform skip behavior, owner test validation. | Helper skip is environmental when declared; helper error is protocol failure. |
| Planning artifacts | Sprint artifact path | Curated review date or generated from committed artifact paths. | Missing artifact is planning traceability gap, not code failure. |

## Non-Goals And No-Claim Notes

Day 6 requirements do not:

- change benchmark commands, CSV schemas, timing thresholds, or benchmark
  interpretation;
- make `bench-canonical-report` a pass/fail portability gate;
- make `performance-sentinels` a portable performance proof beyond the
  existing wall-check lane;
- make `large-matrix-guardrails` a broad scalability, memory, or performance
  proof;
- make coverage percentages a behavioral parity claim;
- make dead-code reports removal-ready proof;
- make helper indexes broad dense-library, SuiteSparse, LAPACK, NumPy, SciPy,
  PETSc, Trilinos, Eigen, ARPACK, vendor-backend, or ecosystem parity;
- change CI policy, required PR checks, Makefile target membership, CMake
  membership, package behavior, public API, or public solver-selection wording.

## Completion Criteria Status

| Criterion | Status | Evidence |
| --- | --- | --- |
| Every report family has an index strategy or explicit deferral. | Complete | Requirements matrix and generated-versus-curated table cover benchmark, sentinel, guardrail, coverage, dead-code, oracle, planning, and direct benchmark outputs. |
| Index requirements do not change benchmark or CI semantics. | Complete | Non-goals explicitly preserve benchmark semantics, timing thresholds, CI policy, and public wording. |
| Freshness and owner fields have clear interpretation. | Complete | Owner and freshness policy defines owner, freshness rule, and stale/missing meaning per report family. |

## Day 7 Handoff

Day 7 should choose the first index candidate. The strongest implementation
candidate is the existing large-matrix guardrail index because it already has
stable lane IDs, reviewed/supplemental categories, explicit skip rows, and a
manifest. A curated external-reference index is also useful, but should not be
generated until output classes and expected-error rows are stable.

