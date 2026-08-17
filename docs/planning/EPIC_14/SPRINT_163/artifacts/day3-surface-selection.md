# Sprint 163 Day 3 Surface Selection

## Selection Goal

Choose the smallest benchmark and sentinel row set that can support a
methodology-bound performance publication without implying portable performance
superiority, broad platform parity, package proof reuse, or state-of-the-art
claims.

## Scoring Rubric

Candidates were scored qualitatively across:

- reproducibility;
- methodology completeness;
- maintenance cost;
- runtime cost;
- hosted suitability;
- user value;
- claim risk.

Rows with stable maintained targets, generated artifacts, local environment
metadata, and existing claim-boundary language scored highest. Rows with broad
execution scope, ambiguous methodology, or non-performance evidence ownership
were deferred.

## Selected Surface

Sprint 163 selects two maintained report commands:

1. `make bench-canonical-report`
2. `make performance-sentinels`

This keeps the selected surface narrow enough to close in one sprint while
covering both threshold-free maintained benchmark measurements and the existing
local thresholded timing gate.

## Selected Row Classification

| Row Family | Selected Rows | Classification | Command Owner | Generated Outputs | Validation Requirement | Claim Boundary |
| --- | --- | --- | --- | --- | --- | --- |
| Canonical benchmark report | `bench_refactor_csc`, `bench_chol_csc`, `bench_iterative_reuse`, `bench_eigs_reuse` | Published threshold-free local measurements | `make bench-canonical-report`, `scripts/bench_canonical_report.sh` | `build/bench-reports/canonical/*.csv`, `index.tsv`, `manifest.txt` | Run `make bench-canonical-report`; inspect `index.tsv` and `manifest.txt`; preserve row-to-command mapping. | Local artifact-friendly snapshot only; no portable performance, speedup, OpenMP, external-library, or state-of-the-art claim. |
| Sentinel S5 wall-check | bcsstk14 QG-AMD, Pres_Poisson AMD, Pres_Poisson ND | Published thresholded local regression gate | `make performance-sentinels`, `scripts/performance_sentinels.sh`, `scripts/wall_check.sh` | `build/bench-reports/sentinels/sentinels.tsv`, `wall_check.txt`, `manifest.txt` | Run `make performance-sentinels`; verify S5 status, baseline, threshold, fixture, metric, and artifact fields. | Local wall-check regression gate only; baseline/threshold provenance must remain visible. |
| Sentinel S2 Cholesky CSC | `bench_chol_csc tests/data/suitesparse/nos4.mtx --repeat 1` parsed metrics | Published threshold-free local context | `make performance-sentinels`, `scripts/performance_sentinels.sh` | `sentinels.tsv`, `bench_chol_csc_nos4.csv`, `manifest.txt` | Run `make performance-sentinels`; verify S2 rows preserve backend request, selected backend, dense kernel, panel solver, metric, and artifact fields. | Threshold-free local backend-context report only; no pass/fail or superiority wording. |
| Sentinel S3 LDLT KKT | `bench_refactor_csc --indefinite-kkt --repeat 1` parsed metrics | Published threshold-free local context | `make performance-sentinels`, `scripts/performance_sentinels.sh` | `sentinels.tsv`, `bench_refactor_csc_kkt.csv`, `manifest.txt` | Run `make performance-sentinels`; verify S3 rows preserve backend request, selected backend, fallback, metric, residual, and artifact fields. | Threshold-free local LDLT backend-context report only; fallback context limits comparisons. |

## Local-Only And Advisory Rows

| Row / Surface | Classification | Reason |
| --- | --- | --- |
| Raw canonical CSV timing rows | Local-only source data for selected publication | Generated under ignored `build/` paths; publish methodology and row identity, not hand-edited timing files. |
| Raw sentinel CSV and wall-check text | Local-only source data for selected publication | Supports selected `sentinels.tsv` rows; should be regenerated from maintained commands. |
| `bench-fast` rows | Advisory runtime confidence | Useful CI/local signal but not selected as a publication report owner. |
| `bench-reorder-sprint86` rows | Advisory bounded historical lane | Branch-local ND evidence, not part of the selected Sprint 163 publication surface. |
| Normalized report-index rows | Advisory navigation/freshness context | May link selected benchmark and sentinel rows, but does not convert them into release proof. |

## Deferred Register

| Deferred Surface | Blocker / Reason |
| --- | --- |
| `make bench` full benchmark run | Broad exploratory mix and higher runtime cost; too large for one sprint's methodology-bound publication claim. |
| `make bench-suitesparse` | Useful benchmark command, but not part of the compact maintained canonical report surface. |
| `make bench-eigs` | Broader eigensolver benchmark sweep with higher claim risk than selected reuse report row. |
| `make large-matrix-guardrails` | Structural guardrail evidence with reviewed/supplemental lane meanings; defer unless a future sprint selects memory/fill guardrail publication. |
| `make report-index-oracle-freshness` | Report freshness/correctness evidence, not performance evidence. |
| `make report-index-comparison-freshness` | Fixture-local comparison freshness, not performance publication evidence. |
| Package/install validation | Sprint 162 package proof stays separate from performance evidence. |
| Corpus, residual, and solver tests | Correctness and coverage owners; useful context but not timing publication rows. |
| API reference and generated API HTML freshness | Documentation/adoption evidence, not performance evidence. |

## Command Map

| Command | Selected Role | Required Follow-Through |
| --- | --- | --- |
| `make bench-canonical-report` | Generate selected canonical benchmark report bundle. | Day 4 methodology contract must define required manifest/index fields and caveats before any report edits. |
| `make performance-sentinels` | Generate selected sentinel bundle and run S5 thresholded gate. | Day 4 methodology contract must preserve S5/S2/S3 classification and baseline/fallback context. |
| `python3 scripts/normalize_report_index.py --family benchmark --family sentinel --output build/report-index/normalized-index.tsv` | Optional navigation check after selected bundles exist. | Do not treat normalized rows as release proof; preserve support tier and claim boundary fields. |

## Stop Conditions

Stop and revise the selected surface if:

- `make bench-canonical-report` cannot emit command, artifact, build mode,
  branch/commit, platform, compiler, and thread context;
- `make performance-sentinels` collapses S5, S2, and S3 meanings into one
  pass/fail status;
- S5 rows lack visible baseline or threshold provenance;
- S2 or S3 rows are described as passing, failing, or proving backend
  superiority;
- any selected row requires broad benchmark execution to validate;
- documentation describes the selected rows as portable performance,
  state-of-the-art performance, broad platform parity, or package proof.

## Completion Check

- Selected row set is limited to the canonical report and sentinel bundles.
- Selected rows can be validated with narrow maintained commands.
- Deferred rows have explicit blockers.
- Performance publication remains separate from correctness, package,
  freshness, documentation, and broad platform evidence.
