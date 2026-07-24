# Sprint 132 Day 2 - Hot Path Inventory

## Purpose

Day 2 inventories hot compressed, direct, iterative, eigensolver, SVD, reorder,
and backend/runtime paths with current benchmark, sentinel, canonical report,
and guardrail visibility.

This is a documentation-only inventory artifact. It does not change benchmark
code, report scripts, Makefile targets, generated artifacts, maintainer
wording, or public performance claims.

## Report Surface Inventory

| Surface | Command | Current outputs | Interpretation |
| --- | --- | --- | --- |
| Full benchmark execution | `make bench` | Direct execution of all benchmark binaries | Developer-side opt-in timing investigation; broad and potentially expensive. |
| Benchmark compile coverage | `make bench-build` | All benchmark binaries under `build/` | Compile-only tooling coverage; no runtime evidence. |
| Fast benchmark subset | `make bench-fast` | Fast benchmark stdout plus `bench_reorder --skip-factor` | Bounded runtime regression signal, not canonical performance proof. |
| Canonical benchmark report | `make bench-canonical-report` | `build/bench-reports/canonical/*.csv`, `index.tsv`, `manifest.txt` | Threshold-free maintained benchmark snapshot for local/CI-friendly comparison. |
| Performance sentinels | `make performance-sentinels` | `sentinels.tsv`, `manifest.txt`, optional `wall_check.txt`, optional `bench_chol_csc_nos4.csv` | Local sentinel bundle; hard gate limited to existing wall-check lane. |
| Large-matrix guardrails | `make large-matrix-guardrails` | `index.tsv`, `manifest.txt`, reviewed logs, bounded CSV shape report, supplemental skip/report rows | Structural and CSV-shape guardrail evidence; not broad timing/scalability proof. |
| Reorder Sprint 86 slice | `make bench-reorder-sprint86` | `bench_reorder --sprint86-slice --skip-factor` stdout/CSV | Bounded historical named-fixture slice, not canonical benchmark surface. |
| Individual benchmark binaries | `build/bench_*` | Binary-specific stdout or CSV | Focused local measurement or exploratory comparison. |

## Benchmark Binary Inventory

| Binary | Hot-path family | Current maintained category | Current report/sentinel visibility | Notes |
| --- | --- | --- | --- | --- |
| `bench_main` | One-shot LU, Cholesky, SpMV, iterative | Exploratory/broad comparison | `make bench`; `bench-suitesparse` smoke target noted in docs | Broad harness; not a compact maintained sentinel. |
| `bench_scaling` | LU scaling and dense/sparse generated inputs | Regression-sensitive runtime lane | `make bench-fast`, `make bench` | Fast-lane runtime signal; not report-indexed as canonical. |
| `bench_fillin` | Fill-in and reorder quality | Regression-sensitive runtime lane | `make bench-fast`, `make bench` | Runtime/fill exploration; not sentinel metadata owner. |
| `bench_convergence` | Iterative solver convergence rates | Exploratory/broader comparison | `make bench` | Important iterative behavior, but no current sentinel/report row. |
| `bench_svd` | Sparse SVD and bidiagonalization | Exploratory/broader comparison | `make bench` | SVD hot path lacks current sentinel/report-index visibility. |
| `bench_refactor` | Direct repeated-run Cholesky lifecycle | Direct repeated-run lifecycle | `make bench` | Adoption/performance measurement surface; not in canonical bundle. |
| `bench_refactor_csc` | CSC refactor, Cholesky default, LDLT KKT mode, backend request/selection/fallback | Canonical maintained measurement surface | `make bench-canonical-report` | Strong backend/runtime metadata candidate. |
| `bench_colamd` | QR/COLAMD ordering quality | Regression-sensitive runtime lane | `make bench-fast`, `make bench` | Reorder/QR comparison; no current sentinel row. |
| `bench_bicgstab` | BiCGSTAB convergence | Exploratory/broader comparison | `make bench` | Iterative hot path with no current sentinel/report row. |
| `bench_chol_csc` | Cholesky CSC linked-list, scalar CSC, supernodal CSC, dense-kernel and panel-solver fields | Canonical maintained measurement surface and sentinel S2 source | `make bench-canonical-report`, `make performance-sentinels` | Current sentinel visibility is threshold-free S2 on `nos4`. |
| `bench_ldlt_csc` | LDLT linked-list versus CSC, dispatch, dense-factor runtime seam | One-shot/backend comparison | `make bench` | Backend hot path lacks canonical/sentinel report visibility. |
| `bench_eigs` | Symmetric eigensolver backend sweep and preconditioner comparison | Exploratory/broader comparison | `make bench`; `make bench-eigs` noted in docs | Backend-rich, but broad and not current sentinel. |
| `bench_eigs_reuse` | Eigensolver public handle reuse | Canonical maintained measurement surface | `make bench-canonical-report` | Current canonical visibility, no hard sentinel. |
| `bench_reorder` | Reorder, ND, named fixtures, skip-factor modes | Regression-sensitive runtime and guardrail CSV-shape lane | `make bench-fast`, `make bench-reorder-sprint86`, `make large-matrix-guardrails` | Current guardrail G4 validates bounded CSV shape/fill rows. |
| `bench_amd_qg` | qg-AMD, generated-banded report, wall-check input | Regression-sensitive adjacent lane and guardrail supplemental lane | `make bench-fast`, `make performance-sentinels` wall-check, `make large-matrix-guardrails` supplemental S2 | Wall-check uses this path; supplemental guardrail row remains opt-in. |
| `bench_iterative_reuse` | CG, GMRES, MINRES public handle reuse | Canonical maintained measurement surface | `make bench-canonical-report` | Current canonical visibility, no hard sentinel. |

## Hot-Path Coverage Map

| Hot-path family | Representative paths | Current visibility | Support tier | Gap status |
| --- | --- | --- | --- | --- |
| Compressed/direct Cholesky CSC | `bench_chol_csc`, `bench_refactor_csc`, Cholesky CSC source paths | Canonical report plus performance sentinel S2 threshold-free row | Benchmark/supplemental local report | Visible; hard threshold limited to wall-check, not Cholesky timing. |
| Direct repeated-run lifecycle | `bench_refactor`, `bench_refactor_csc` | Canonical only for `bench_refactor_csc`; direct `bench_refactor` remains benchmark-local | Benchmark | Partial visibility; public repeated-run path is visible but not fully indexed. |
| LDLT CSC backend/runtime | `bench_ldlt_csc`, `bench_refactor_csc --indefinite-kkt` fields | Backend fields in `bench_refactor_csc`; `bench_ldlt_csc` benchmark-local | Benchmark/deferred sentinel | Partial visibility; no sentinel/report bundle row dedicated to LDLT CSC backend comparison. |
| Iterative reuse | `bench_iterative_reuse` | Canonical report | Benchmark | Visible in canonical report; no sentinel threshold or runtime-risk classification yet. |
| Iterative convergence/BiCGSTAB | `bench_convergence`, `bench_bicgstab` | `make bench` only | Exploratory/deferred | Missing maintained sentinel/report visibility. |
| Eigensolver reuse | `bench_eigs_reuse` | Canonical report | Benchmark | Visible in canonical report; no sentinel threshold. |
| Eigensolver backend sweep | `bench_eigs` | `make bench` / `make bench-eigs` | Exploratory/deferred | Backend-rich surface lacks sentinel/report governance. |
| SVD/bidiag | `bench_svd` | `make bench` only | Exploratory/deferred | Missing maintained sentinel/report visibility. |
| Reorder/qg-AMD | `bench_reorder`, `bench_amd_qg` | `bench-fast`, wall-check, guardrail G1/G4, supplemental S1/S2 opt-in | Reviewed structural guardrail plus supplemental/local timing | Visible; timing remains local context, fill/shape are stronger than timing claims. |
| Graph/ND structural guardrails | `test_graph`, `test_reorder_nd`, `test_reorder_amd_qg` | `make large-matrix-guardrails` reviewed G1-G3 | Reviewed structural guardrail | Visible as structure/invariant checks, not benchmark timing. |
| Backend runtime observability | Dense backend env vars, backend selected/fallback fields, OpenMP build mode, `OMP_NUM_THREADS` | Canonical and sentinel manifests/rows include partial metadata | Supplemental metadata | Partial visibility; Day 4-5 must define complete contract. |
| OpenMP build/runtime | Makefile `SPARSE_OPENMP`, `make omp`, sentinel `build_mode`, `OMP_NUM_THREADS` | Build-mode and thread-count context in sentinel bundle | Runtime context/deferred governance | Partial visibility; no per-call public thread-control surface. |

## Reviewed Versus Supplemental Lane Notes

| Lane family | Current classification | Notes |
| --- | --- | --- |
| Guardrail `G1`-`G3` | Reviewed structural guardrail | Test pass/fail lanes, not timing claims. |
| Guardrail `G4` | Reviewed bounded CSV-shape/fill report | Validates `bench_reorder` CSV shape and named-fixture fill rows; timing remains local context. |
| Guardrail `S1` and `S2` | Supplemental opt-in reports | Skipped by default; not recurring reviewed evidence. |
| Sentinel `S5` wall-check | Local threshold gate | Existing hard timing gate with machine-class baseline; should not expand without new baseline policy. |
| Sentinel `S2` Cholesky CSC | Threshold-free report | Local Cholesky CSC timing context with backend env and dense-kernel metadata. |
| Canonical benchmark rows | Benchmark report | Threshold-free maintained snapshot; compare across branches/runs only with matching context. |
| Benchmark-local binaries | Exploratory or focused measurement | Useful for investigation, not generated recurring assurance until indexed or documented. |

## Missing-Sentinel Queue

| Gap | Current coverage | Blocker | Future owner |
| --- | --- | --- | --- |
| Dedicated LDLT CSC backend/runtime sentinel | Benchmark-local `bench_ldlt_csc`; partial backend fields via `bench_refactor_csc` | Need runtime metric, fixture, backend request/selected/fallback semantics, and threshold/report-only decision. | `benchmark-report-owner` and backend runtime owner. |
| Iterative convergence and BiCGSTAB sentinel | `bench_convergence` and `bench_bicgstab` only under broad `make bench` | Need stable fixture, metric, runtime budget, variance policy, and support tier. | Iterative benchmark owner. |
| SVD/bidiag sentinel | `bench_svd` only under broad `make bench` | Need bounded SVD fixture, metric, runtime budget, and non-claim boundary. | SVD benchmark owner. |
| Eigensolver backend sweep sentinel | `bench_eigs` broad backend sweep | Need scoped backend/runtime lane; full sweep may be too broad/slow for recurring sentinel. | Eigensolver benchmark owner. |
| OpenMP runtime observability row | Sentinel records build mode and `OMP_NUM_THREADS`; no complete contract yet | Need Day 4 backend/runtime contract and Day 5 metadata design. | Runtime governance owner. |
| Canonical report backend metadata completeness | Canonical report has branch/commit and per-artifact command; backend fields live inside CSVs unevenly | Need metadata field matrix before script changes. | `report-index-owner` and benchmark report owner. |
| Supplemental large-matrix recurring validation | Supplemental guardrail lanes skip by default | Need runtime and support-tier policy before recurring validation. | `large-matrix-guardrails`. |

## Owner and Validation Surface Map

| Owner area | Files | Validation surface |
| --- | --- | --- |
| Canonical report owner | `scripts/bench_canonical_report.sh`, canonical benchmark binaries | `make bench-canonical-report` if touched. |
| Sentinel owner | `scripts/performance_sentinels.sh`, `scripts/wall_check.sh`, `bench_chol_csc`, `bench_amd_qg`, `bench_reorder` | `make performance-sentinels` if touched. |
| Guardrail owner | `scripts/large_matrix_guardrails.sh`, graph/reorder tests, reorder benchmarks | `make large-matrix-guardrails` if touched. |
| Direct/backend benchmark owner | `bench_refactor_csc`, `bench_chol_csc`, `bench_ldlt_csc`, backend helper header | Focused benchmark command plus full C quality if `.c`/`.h` changes. |
| Iterative benchmark owner | `bench_iterative_reuse`, `bench_convergence`, `bench_bicgstab` | Focused benchmark command plus full C quality if `.c`/`.h` changes. |
| Eigensolver benchmark owner | `bench_eigs_reuse`, `bench_eigs` | Focused benchmark command plus full C quality if `.c`/`.h` changes. |
| SVD benchmark owner | `bench_svd` | Focused benchmark command plus full C quality if `.c`/`.h` changes. |
| Runtime/OpenMP owner | `Makefile`, runtime docs, benchmark/sentinel metadata | `make omp` or documented unavailable-runtime check if touched; full C quality if code changes. |
| Benchmark docs owner | `benchmarks/README.md`, `docs/maintainer_guide.md` | Docs hygiene and non-claim scan if touched. |

## Day 3 Handoff

Day 3 should rank the missing and partial sentinel gaps by:

- user-facing workflow value;
- expected runtime and flake risk;
- backend and OpenMP sensitivity;
- whether a hard threshold is defensible or only threshold-free reporting is
  safe;
- current report metadata readiness;
- claim impact if the lane is over-interpreted.

The strongest initial candidates for ranking are LDLT CSC backend/runtime,
iterative convergence/BiCGSTAB, SVD/bidiag, eigensolver backend sweep, OpenMP
runtime observability, canonical backend metadata completeness, and
supplemental large-matrix validation policy.

## Completion Criteria

| Criterion | Status | Evidence |
| --- | --- | --- |
| Every high-value hot path has current coverage status or explicit unknown status. | Complete | Benchmark inventory and hot-path coverage map cover direct, iterative, eigensolver, SVD, reorder, backend, OpenMP, sentinel, and guardrail surfaces. |
| Timing rows are not confused with correctness or portable performance proof. | Complete | Reviewed/supplemental lane notes classify timing rows as benchmark, supplemental, local, or exploratory evidence. |
| Missing coverage is recorded as an owner queue, not left implicit. | Complete | Missing-sentinel queue assigns blockers and future owners to each unresolved gap. |
