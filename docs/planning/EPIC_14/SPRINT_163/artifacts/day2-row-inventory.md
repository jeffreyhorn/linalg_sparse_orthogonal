# Sprint 163 Day 2 Row Inventory

## Purpose

Day 2 inventories candidate benchmark and sentinel rows for Sprint 163
publication. This is not the final surface selection; it is the source-backed
register Day 3 will use to choose rows that can carry methodology fields
without implying portable performance superiority.

## Source Inputs

- `Makefile`
- `benchmarks/README.md`
- `scripts/bench_canonical_report.sh`
- `scripts/performance_sentinels.sh`
- `scripts/wall_check.sh`
- `.github/workflows/ci.yml`
- `.github/workflows/macos-ci.yml`
- Sprint 162 working notes and retrospective

## Candidate Row Register

| Candidate | Owner / Command | Current Artifact | Evidence Type | Methodology Fields Already Present | Blockers Before Publication | Publication Risk |
| --- | --- | --- | --- | --- | --- | --- |
| `bench_refactor_csc` canonical row | `make bench-canonical-report`; script command: `bench_refactor_csc tests/data/suitesparse/nos4.mtx --repeat 1` | `build/bench-reports/canonical/bench_refactor_csc.csv`; indexed by `index.tsv` | Threshold-free direct repeated-run measurement | report label, timestamp, commit, branch, platform, compiler, build mode, `OMP_NUM_THREADS`, command, artifact | Raw CSV schema and row-level interpretation need methodology wording before publication. | Medium: easy to overread as solver speedup if detached from local context. |
| `bench_chol_csc` canonical row | `make bench-canonical-report`; script command: `bench_chol_csc tests/data/suitesparse/nos4.mtx --repeat 1` | `build/bench-reports/canonical/bench_chol_csc.csv`; indexed by `index.tsv` | Threshold-free Cholesky CSC measurement | report label, timestamp, commit, branch, platform, compiler, build mode, `OMP_NUM_THREADS`, command, artifact | Needs clear separation between path measurability and broad Cholesky superiority. | Medium: backend and panel-solver descriptors can be mistaken for portable backend proof. |
| `bench_iterative_reuse` canonical row | `make bench-canonical-report`; script command: `bench_iterative_reuse` | `build/bench-reports/canonical/bench_iterative_reuse.csv`; indexed by `index.tsv` | Threshold-free iterative public-handle reuse measurement | report label, timestamp, commit, branch, platform, compiler, build mode, `OMP_NUM_THREADS`, command, artifact | Needs row identity and fixture semantics documented before any published comparison. | Low to medium: narrow public-handle reuse surface is already bounded. |
| `bench_eigs_reuse` canonical row | `make bench-canonical-report`; script command: `bench_eigs_reuse` | `build/bench-reports/canonical/bench_eigs_reuse.csv`; indexed by `index.tsv` | Threshold-free eigensolver public-handle reuse measurement | report label, timestamp, commit, branch, platform, compiler, build mode, `OMP_NUM_THREADS`, command, artifact | Needs residual/convergence context kept separate from timing interpretation. | Medium: eigensolver rows are easy to overstate as algorithmic superiority. |
| Sentinel S5 bcsstk14 QG-AMD row | `make performance-sentinels`; wraps `make wall-check` / `bench_amd_qg --only bcsstk14` | `build/bench-reports/sentinels/sentinels.tsv`; raw `wall_check.txt` | Thresholded local regression gate | sentinel id, status, support tier, claim boundary, command, build mode, `OMP_NUM_THREADS`, fixture, metric, value, baseline, threshold, artifact | Baseline provenance and machine-class framing must be visible wherever published. | Medium: thresholded rows can be misread as broad performance promise. |
| Sentinel S5 Pres_Poisson AMD row | `make performance-sentinels`; wraps `make wall-check` / `bench_reorder --only Pres_Poisson` | `build/bench-reports/sentinels/sentinels.tsv`; raw `wall_check.txt` | Thresholded local regression gate | sentinel id, status, support tier, claim boundary, command, build mode, `OMP_NUM_THREADS`, fixture, metric, value, baseline, threshold, artifact | Same S5 baseline and local gate framing required. | Medium. |
| Sentinel S5 Pres_Poisson ND row | `make performance-sentinels`; wraps `make wall-check` / `bench_reorder --only Pres_Poisson` | `build/bench-reports/sentinels/sentinels.tsv`; raw `wall_check.txt` | Thresholded local regression gate | sentinel id, status, support tier, claim boundary, command, build mode, `OMP_NUM_THREADS`, fixture, metric, value, baseline, threshold, artifact | Same S5 baseline and local gate framing required; ND threshold differs from AMD threshold. | Medium. |
| Sentinel S2 Cholesky CSC rows | `make performance-sentinels`; script command: `bench_chol_csc tests/data/suitesparse/nos4.mtx --repeat 1` | `build/bench-reports/sentinels/sentinels.tsv`; raw `bench_chol_csc_nos4.csv` | Threshold-free backend-context report | sentinel id, support tier, claim boundary, command, build mode, thread setting, fixture, metric, value, artifact, backend request, backend selected, dense kernel, panel solver, notes | Must stay threshold-free; no pass/fail or superiority wording. | Low to medium if support tier and claim boundary are preserved. |
| Sentinel S3 LDLT KKT rows | `make performance-sentinels`; script command: `bench_refactor_csc --indefinite-kkt --repeat 1` | `build/bench-reports/sentinels/sentinels.tsv`; raw `bench_refactor_csc_kkt.csv` | Threshold-free LDLT backend-context report | sentinel id, support tier, claim boundary, command, build mode, thread setting, fixture, metric, value, artifact, backend request, backend selected, backend fallback, notes | Must retain fallback context and avoid backend superiority claims. | Medium because fallback fields change how rows can be compared. |

## Rejected Or Deferred Candidates

| Candidate | Reason |
| --- | --- |
| `make bench` full benchmark run | Too broad for a methodology-bound Sprint 163 publication surface; includes exploratory rows that are not the compact maintained benchmark face. |
| `make bench-fast` runtime subset | Useful supplemental runtime confidence, but not currently the canonical report owner and not a publication artifact by itself. |
| `make bench-reorder-sprint86` | Historical bounded ND lane; branch-local evidence but not part of the canonical maintained measurement surface. |
| `make bench-suitesparse` and `make bench-eigs` | Useful direct benchmark commands, but broader or exploratory compared with the maintained canonical report rows. |
| `make large-matrix-guardrails` rows | Structural guardrail and supplemental local report rows; adjacent evidence unless a later day explicitly selects them. |
| `make report-index-oracle-freshness` | Generated oracle freshness is correctness/report hygiene evidence, not performance evidence. |
| `make report-index-comparison-freshness` | Fixture-local comparison freshness is correctness/comparison hygiene evidence, not performance publication evidence. |
| Package/install validation rows | Sprint 162 package proof is intentionally separate from Sprint 163 performance proof. |
| API docs and generated API HTML freshness rows | Documentation/adoption evidence, not performance evidence. |
| Corpus tests and solver residual tests | Correctness and coverage evidence; useful context but not timing publication rows. |

## Performance Versus Non-Performance Separation

Performance-publication candidates must be benchmark or sentinel rows with:

- exact command ownership;
- fixture or workload identity;
- generated artifact identity;
- local environment context;
- row type classification as thresholded gate or threshold-free report;
- claim-boundary language that prevents portable superiority claims.

Rows are excluded from Sprint 163 performance publication when they primarily
prove:

- correctness;
- package/install behavior;
- ABI or shared-library policy;
- report freshness;
- corpus coverage;
- external comparison correctness;
- documentation generation;
- broad platform parity.

## Day 3 Selection Blockers

- Canonical rows need a methodology contract that names fixture, repeat count,
  generated artifact, timing interpretation, and local context requirements.
- S5 rows need explicit baseline provenance and threshold framing before any
  public-facing publication text.
- S2 and S3 rows must remain threshold-free and preserve backend request,
  selected backend, fallback, dense-kernel, and panel-solver context.
- The selected surface should avoid long-running broad benchmark commands.
- Generated rows should be regenerated from maintained commands, not edited by
  hand or treated as source-controlled timing proof.

## Completion Check

- Candidate rows are backed by maintained targets and scripts.
- Non-performance rows are excluded from performance publication.
- Commands, owners, artifacts, blockers, and publication risks are recorded
  before Day 3 selection.
