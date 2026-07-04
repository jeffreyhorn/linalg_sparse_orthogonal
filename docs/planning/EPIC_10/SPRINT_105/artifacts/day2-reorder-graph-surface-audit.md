# Sprint 105 Day 2 Reorder and Graph Surface Audit

## Purpose

Day 2 audits the live reorder, graph, fill, and large-matrix evidence surfaces
before Sprint 105 defines metric contracts or edits implementation code. The
goal is to rank real gaps from current owners, not from old sprint assumptions.

## Live Owner Inventory

### Public Headers and API Owners

| owner | surface | current role | audit note |
|---|---|---|---|
| `include/sparse_reorder.h` | `sparse_reorder_rcm`, `sparse_reorder_amd`, `sparse_reorder_nd`, `sparse_reorder_colamd`, `sparse_permute`, `sparse_bandwidth` | public reorder API | clearly separates symmetric reorderings from COLAMD column ordering |
| `include/sparse_analysis.h` | `sparse_analysis_opts_t.reorder`, `sparse_analysis_reorder_opts_t` | public analysis-time reorder controls | typed ND and supernodal-postorder controls are the preferred surface; legacy env vars are compatibility only |
| `include/sparse_matrix.h` and factor headers | reorder options and matrix permutation state | factorization integration | documentation warns about reordered/factored matrix lifecycle interactions |

### Source Owners

| owner | algorithm family | current role | audit note |
|---|---|---|---|
| `src/sparse_reorder.c` | RCM, AMD wrapper, permutation, bandwidth | public reorder entry points and common helpers | AMD wrapper delegates to quotient-graph implementation |
| `src/sparse_reorder_amd_qg.c` | quotient-graph AMD | production AMD implementation | scalable replacement for deleted bitset AMD; still has history-heavy design comments |
| `src/sparse_colamd.c` | COLAMD | column ordering for QR/unsymmetric paths | separate from symmetric ordering; rectangular support lives here |
| `src/sparse_reorder_nd.c` | nested dissection | ND recursion, leaf AMD, policy/env compatibility, profiling | high-value surface with many historical comments and policy knobs |
| `src/sparse_graph_core.c` | graph construction and ownership | graph-from-sparse, subgraph, lifecycle | extracted ownership from older monolithic graph file |
| `src/sparse_graph_coarsen.c` | graph coarsening | HEM/HCC coarsening and override state | part of ND multilevel pipeline |
| `src/sparse_graph_bisect.c` | coarsest bisection and spectral path | brute/GGGP/spectral bisection | includes spectral fallback and large graph runtime caveats |
| `src/sparse_graph_refine.c` | FM refinement | bucket/refinement helpers and strategy state | extracted helper owner |
| `src/sparse_graph_separator.c` | separator lifting | edge-to-vertex separator and lift strategy | extracted helper owner |
| `src/sparse_graph.c` | graph orchestration | uncoarsening and top-level partition orchestration | still carries a large design block and residual historical notes |
| `src/sparse_analysis.c` | symbolic analysis dispatch | applies reorder policy through analysis path | owns analyze-time dispatch and supernodal postorder composition |

### Test Owners

| owner | proof surface | current role | audit note |
|---|---|---|---|
| `tests/test_reorder.c` | public permutation, bandwidth, RCM, AMD, COLAMD behavior | core reorder unit coverage | broad public API owner |
| `tests/test_colamd.c` | COLAMD-specific behavior | rectangular and QR-oriented column-order proof | keeps COLAMD separate from symmetric ordering |
| `tests/test_reorder_amd_qg.c` | AMD wrapper and quotient-graph stress | delegation, fill equality, 10k banded stress | strongest current large regular-structure guardrail |
| `tests/test_reorder_nd.c` | ND policy, fill, integration, env/typed controls | main ND proof owner | very large owner with many historical comments and generated fixtures |
| `tests/test_graph.c` | graph construction, coarsening, bisection, FM, separator, stress | main graph proof owner | very large owner; many generated graph fixtures and policy variants |
| `tests/test_graph_fm_buckets.c` | FM bucket helpers | helper-level proof | cleanly split helper proof |
| `tests/test_etree.c` | etree/colcount support | symbolic analysis adjacency | indirect fill-analysis owner |

### Benchmark and Reporting Owners

| owner | current fields | current interpretation |
|---|---|---|
| `benchmarks/bench_reorder.c` | `matrix`, `n`, `reorder`, `nnz_L`, `reorder_ms`, `factor_ms`, `reorder_path`, `fixture_slice`, `nd_base_threshold` | strongest current reorder/fill report; local timing only |
| `benchmarks/bench_fillin.c` | human-readable LU fill rows with `nnz_before`, `nnz_after`, `ratio` | useful smoke/runtime lane, but not artifact-friendly or canonical |
| `benchmarks/bench_amd_qg.c` | `matrix`, `n`, `impl`, `reorder_ms`, `peak_rss_mb`, `nnz_L` | AMD quotient-graph vs deleted bitset foil; historical and local |
| `benchmarks/bench_colamd.c` | human-readable QR `nnz(R)` for none/AMD/COLAMD | useful COLAMD comparison, but not a stable machine-readable report |
| `make bench-reorder-sprint86` | `bench_reorder --sprint86-slice --skip-factor` | bounded two-fixture reorder/fill artifact |
| `make wall-check` | qg-AMD and Pres_Poisson reorder timing threshold | current hard timing gate used by performance sentinels |
| `make performance-sentinels` | S5 wall-check rows and S2 Cholesky CSC local rows | local regression evidence; hard-fail behavior remains S5 only |

### Documentation Owners

| owner | current role | audit note |
|---|---|---|
| `README.md` | public feature and API summary | names AMD, COLAMD, ND, RCM and warns about local reorder use |
| `benchmarks/README.md` | benchmark surface map and reorder field definitions | already documents `bench_reorder` fields; Sprint 105 can tighten canonical contract |
| `docs/maintainer_guide.md` | reviewed/supplemental ownership and Sprint 98 reorder/fill snapshot | explicitly says `nnz_L` is fill and `reorder_ms` is local timing context |
| Sprint 98 artifacts | prior reorder/fill assurance topology | identifies `bench-reorder-sprint86` as bounded two-fixture calibration |
| Sprint 104 handoff | runtime and benchmark claim constraints | local timing and sentinel constraints carry into Sprint 105 |

## Current Evidence by Algorithm Family

### AMD and Quotient-Graph AMD

Current evidence:

- public API docs describe quotient-graph AMD and its soft workspace target;
- `sparse_reorder_amd` delegates to `sparse_reorder_amd_qg`;
- `tests/test_reorder_amd_qg.c` owns wrapper delegation, fill equality, and a
  10 000 x 10 000 banded stress fixture;
- `bench_amd_qg` still preserves a deleted bitset implementation as a local
  benchmark foil and emits `reorder_ms`, `peak_rss_mb`, and `nnz_L`;
- `wall-check` and `performance-sentinels` use AMD/qg-AMD timing in narrow
  regression roles.

Gaps:

- AMD memory evidence is split between header prose, `bench_amd_qg`, and the
  10k test; there is no single Sprint 105 metric contract naming workspace or
  memory-proxy interpretation.
- `bench_amd_qg` is intentionally historical and should not become the
  canonical public AMD report without a fresh contract.

### COLAMD

Current evidence:

- public API clearly separates COLAMD as a column ordering that supports
  rectangular matrices;
- `sparse_colamd.c` owns the implementation;
- `tests/test_colamd.c` and `bench_colamd` own correctness and QR fill context;
- `benchmarks/README.md` points users to `bench_colamd` and `example_colamd`
  instead of `bench_main --reorder`.

Gaps:

- `bench_colamd` emits human-readable text, not the canonical field set used
  by `bench_reorder`.
- QR fill fields are not yet aligned with the future Sprint 105 fixture and
  metric naming contract.

### Nested Dissection

Current evidence:

- `sparse_reorder_nd.c` owns ND recursion, leaf AMD, policy/env compatibility,
  and profiling;
- `tests/test_reorder_nd.c` owns the large ND proof surface, including grids,
  SuiteSparse fixtures, policy/default behavior, and analyze dispatch;
- `bench_reorder` includes ND rows and records `nd_base_threshold`;
- `bench-reorder-sprint86` keeps a bounded bcsstk14/Pres_Poisson calibration
  slice.

Gaps:

- ND comments and tests contain extensive sprint-history blocks that make the
  current contract harder to read.
- generated graph families exist inside tests, but their naming and report
  roles are not standardized for benchmark artifacts.
- large-matrix behavior is partly covered by selected fixtures and historical
  timing notes, not by a fresh Sprint 105 guardrail contract.

### Graph Partition and Separator Paths

Current evidence:

- graph code is decomposed into core, coarsen, bisect, refine, separator, and
  orchestration owners;
- `tests/test_graph.c` covers graph construction, HEM/HCC, bisection, FM,
  separator lifting, spectral paths, env/typed policies, and stress fixtures;
- `tests/test_graph_fm_buckets.c` owns the FM bucket helper layer;
- `sparse_reorder_nd.c` consumes graph partition output through the ND driver.

Gaps:

- `tests/test_graph.c` is a very large proof owner with many generated
  fixtures and historical explanations. It is strong coverage but difficult to
  scan.
- generated-family coverage is test-local; Sprint 105 needs to decide which
  generated families should also appear in report artifacts.

### Fill and Runtime Report Surfaces

Current evidence:

- `bench_reorder` is the most artifact-ready reorder/fill report.
- `bench_fillin`, `bench_colamd`, and `bench_amd_qg` provide useful adjacent
  context but have different output schemas and levels of machine-readability.
- `docs/maintainer_guide.md` already treats `nnz_L` as fill and `reorder_ms`
  as local timing context only.

Gaps:

- there is no single canonical contract for matrix/fixture identity, algorithm
  label, fill count, fill ratio, runtime field, memory proxy, skip status, and
  reviewed/supplemental status across reorder/fill artifacts.
- current reports mix CSV, human-readable text, thresholded wall-check output,
  and local benchmark foils.

## Ranked Gap List

| rank | gap | value | determinism | validation cost | claim risk | recommendation |
|---:|---|---|---|---|---|---|
| 1 | No single fill/runtime/memory fixture contract across reorder artifacts | high | high | low | high if left vague | fix in Days 3-5 |
| 2 | `bench_reorder` is strong but still tied to historical fixture-slice naming and local field prose | high | high | medium | medium | fix in Days 3-6 |
| 3 | Generated graph families are proof-local, not report-contract-owned | high | high | medium | medium | fix in Days 4 and 7 |
| 4 | Large-matrix guardrails are split across AMD 10k stress, ND fixture tests, and benchmark comments | high | medium | medium/high | high | design Days 8-9 |
| 5 | COLAMD and LU fill reports are human-readable and schema-inconsistent with `bench_reorder` | medium | high | medium | medium | fix or defer after Day 3 contract |
| 6 | Graph and ND proof owners contain heavy sprint-history comments | medium | high | medium | low/medium | cleanup only when touched, Days 10/13 |
| 7 | `bench_amd_qg` preserves deleted bitset code as a historical foil | medium | medium | medium | medium/high if overpromoted | keep bounded; do not make canonical without fresh contract |
| 8 | Windows/POSIX CMake count impacts are easy to miss when adding tests | medium | high | low | medium | track in every test-touch day |

## Initial Fix-Now Queue

1. Define the Sprint 105 fill and fixture contract before editing benchmark
   output.
2. Treat `bench_reorder` as the first implementation target because it already
   has stable CSV rows and the bounded Sprint 98/104 documentation path.
3. Decide whether generated graph-family evidence belongs in `bench_reorder`,
   a small helper, or planning artifacts before adding new fixtures.
4. Define large-matrix guardrails as structural or memory/runtime smoke checks
   rather than broad timing thresholds.
5. If touching graph/ND files, remove stale history-heavy comments only where
   the cleanup clarifies current ownership.

## Deferred Queue

| deferred item | reason |
|---|---|
| Promote `bench_amd_qg` to canonical AMD report | it is intentionally a historical bitset-vs-qg foil and needs a fresh contract first |
| Convert every reorder-adjacent benchmark to CSV in one batch | too broad for early Sprint 105; start with `bench_reorder` and contract first |
| Split `tests/test_graph.c` or `tests/test_reorder_nd.c` immediately | high blast radius; cleanup should follow touched implementation boundaries |
| Add new hard performance thresholds for reorder/fill lanes | Sprint 104 says new thresholds need baseline design and machine-class assumptions |
| Claim broad ND superiority over AMD | current evidence is fixture-specific and policy-specific |
| Claim COLAMD broad QR superiority | current report is small and human-readable; needs a maintained evidence contract |

## Day 3 Starting Point

Day 3 should define the canonical field contract for:

- fixture identity;
- generated family labels;
- ordering algorithm labels;
- `nnz_L` or QR/LU fill fields;
- fill ratios and comparison baselines;
- local runtime fields;
- memory proxy fields;
- skip/error status fields;
- reviewed versus supplemental lane status;
- explicit non-claims for local timing and generated fixtures.

The first candidate implementation lane after the contract is
`bench_reorder`, with `bench_colamd`, `bench_fillin`, and `bench_amd_qg`
remaining adjacent until the contract decides whether they need schema
alignment in Sprint 105.
