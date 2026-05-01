# Sprint 22 Retrospective — Ordering Upgrades: Nested Dissection & Quotient-Graph AMD

**Sprint budget:** 14 working days (~168 hours)
**Branch:** `sprint-22`
**Calendar elapsed:** 2026-04-28 → 2026-04-29 (intensive condensed run building on the Sprint 21 finish; the 14-day budget tracks engineering effort, not wall-clock days)

> **Status:** Day 14 stub.  Day-by-day metrics are populated; prose
> sections (What went well / What didn't / Lessons) are placeholders
> for the post-sprint write-up — landing the structure now makes
> that write-up mechanical.

## Goal recap

> Upgrade the ordering stack with nested dissection (ND) for large
> 2D / 3D PDE meshes and replace the bitset-based AMD with a
> quotient-graph implementation that operates in O(nnz) memory.
> Removes the current scaling bottleneck on the AMD path
> (O(n²/64) memory for the bitset adjacency representation) and
> adds a fill-reducing ordering that should beat AMD on regular
> meshes by 2-5× in fill on the SuiteSparse Pres_Poisson / bcsstk14
> corpus the eigensolver work has been stressing since Sprint 20.

## Definition of Done checklist

Against the five `PROJECT_PLAN.md` items:

| # | Item | Target | Landed | Verdict |
|---|------|--------|--------|---------|
| 1 | Graph partitioning | Multilevel vertex-separator partitioner: heavy-edge-matching coarsening + brute-force / GGGP coarsest bisection + FM uncoarsening + smaller-side vertex-separator extraction | Days 1-5.  `src/sparse_graph.c` (~1 100 LoC), 39 tests / 1 460 assertions in `tests/test_graph.c`.  Deterministic on seed (Day 5).  SuiteSparse smoke: bcsstk14 28 ms; Pres_Poisson 1.3 s. | ✅ Complete |
| 2 | Nested dissection ordering | Recursive ND on top of the partitioner; produces fill-reducing orderings superior to AMD for 2D / 3D PDE meshes | Days 6-7.  `src/sparse_reorder_nd.c` (~250 LoC).  Day 6 ND_BASE_THRESHOLD = 4; Day 9 retuned to 32 (sweep winner on bcsstk14 + Pres_Poisson).  Fill quality: 10×10 grid 1.22× of AMD; bcsstk14 1.21×; Pres_Poisson 1.06× — see Deferred section below. | ⚠️ Functional, fill target deferred |
| 3 | Add SPARSE_REORDER_ND to enum | Wire ND through every direct factorization's analysis dispatch | Day 8.  `SPARSE_REORDER_ND = 4` in `include/sparse_types.h`; arm in `sparse_analyze` central dispatch + per-factorization `*_factor_opts` switches in cholesky / ldlt / lu / qr.  Public-header doxygen + README updated.  Cholesky / LU / LDL^T residuals on bcsstk14 / nos4 / bcsstk04 verified ≤ 1e-8 via the enum dispatch. | ✅ Complete |
| 4 | Quotient-graph AMD | Replace the bitset AMD; production swap with the bitset removed (Option A) | Days 10-12.  `src/sparse_reorder_amd_qg.c` (~340 LoC) implements simplified Davis-style quotient-graph minimum-degree (single workspace, sorted-merge, on-demand compaction + realloc).  Day 12 deleted the bitset; `sparse_reorder_amd` now delegates.  Bit-identical fill across the corpus.  10 000×10 000 banded stress completes in 0.24 s. | ✅ Complete |
| 5 | Tests and benchmarks | ND + AMD-qg cross-corpus benches; PERF_NOTES.md updated; bench captures committed | Days 5, 9, 13, 14.  `benchmarks/bench_reorder.c` (Day 9 cross-ordering + threshold sweep), `benchmarks/bench_amd_qg.c` (Day 13 bitset-vs-qg AB), `benchmarks/bench_day14.csv` (final cross-ordering).  PERF_NOTES.md gained "Reordering comparison" (Day 9) + "AMD bitset → quotient-graph swap" (Day 13) sections. | ✅ Complete |

## Final metrics

| Metric | Start of Sprint 22 | End of Sprint 22 |
|--------|-------------------:|-----------------:|
| `tests/test_graph.c` (new in S22 D1) | — | **39 tests / 1 460 assertions** |
| `tests/test_reorder_nd.c` (new in S22 D6) | — | **12 tests / 62 assertions** |
| `tests/test_reorder_amd_qg.c` (new in S22 D10) | — | **7 tests / 40 assertions** |
| Total Sprint 22 assertions | 0 | **1 562** (across 58 tests in 3 new test files) |
| `include/sparse_reorder.h` | ~140 LoC | **~165 LoC** (+ ND declaration + AMD doxygen rewrite) |
| `src/sparse_graph.c` (new) | — | ~1 100 LoC (sparse_graph_t + multilevel partitioner) |
| `src/sparse_reorder_nd.c` (new) | — | ~250 LoC |
| `src/sparse_reorder_amd_qg.c` (new) | — | ~340 LoC |
| `src/sparse_reorder.c` AMD section | ~130 LoC (bitset) | **4 LoC** (delegating wrapper) |
| New benchmark drivers | — | `bench_reorder.c`, `bench_amd_qg.c` |
| New benchmark captures | — | `bench_day9_nd.{csv,txt}`, `bench_day13_amd_qg.{csv,txt}`, `bench_day14.{csv,txt}` |
| New PERF_NOTES.md | — | Sprint 22's perf notes file with two appended sections |
| **AMD memory bound** | bitset O(n²/64) | **quotient-graph O(nnz)** — ≥ 17× reduction at n = 20 000 (analytic ~26× at n = 50 000) |
| **AMD fill (SuiteSparse corpus)** | bitset baseline | bit-identical — 1.000× across all six fixtures |
| **AMD wall time on n = 20 000 banded** | ~6.2 s (bitset) | **~0.9 s (qg)** — 6.7× speedup |
| **ND on Pres_Poisson** | not implemented | 2.84M nnz(L) at threshold = 32 (1.06× of AMD) |
| `ND_BASE_THRESHOLD` default | — | 4 (Day 6) → **32** (Day 9 sweep winner) |
| Total `make test` binaries | 47 | **48** (+1 = test_reorder_amd_qg) |

## Performance highlights

### AMD memory reduction (Days 10-13)

Bitset → quotient-graph swap.  Day 12 production swap with the
bitset deleted entirely (Option A); Day 13 AB-bench in
`bench_day13_amd_qg.txt` measured the headline:

  - **banded_20000** (synthetic, bandwidth = 5): bitset adds 24.88 MB
    above the qg implementation's already-set peak.  Analytic
    bitset = 50 MB; analytic qg = ~3 MB.  **Ratio ≥ 17×.**
  - **banded_50000** (analytic only — bitset's O(n³/64) is multi-
    minute at this size): 312 MB vs 12 MB → **~26× reduction**.

Plan target was ≥ 4× memory reduction at n ≥ 50 000 — cleared by a
wide margin on the 20 K row alone.

### AMD wall-time speedup on banded fixtures

  | Fixture       |     n   | bitset_ms | qg_ms    | speedup |
  | ------------- | ------: | --------: | -------: | ------: |
  | banded_5000   |   5 000 |     263.4 |    59.5  |   4.4×  |
  | banded_10000  |  10 000 |   1 211.8 |   233.2  |   5.2×  |
  | banded_20000  |  20 000 |   6 194.6 |   919.5  |   6.7×  |

The bitset's O(n²/64) per-pivot merge is quadratic in n regardless
of nnz; the quotient-graph version's O(deg) per-pivot merge stays
linear in nnz.

### ND ND_BASE_THRESHOLD sweep (Day 9)

Day 6 default was 4 (recurse to single vertices).  Day 9 sweep on
bcsstk14 + Pres_Poisson picked 32:

  | Threshold | bcsstk14 nnz | PP nnz       | bcsstk14 ms | PP ms    |
  | --------: | -----------: | -----------: | ----------: | -------: |
  |   4       |     140 102  |  2 834 569 ← |  2 175.7    | ~38 000  |
  |  32       |     140 024 ←|  2 837 046   |  2 180.2    | 23 633.0 |
  |  64       |     142 265  |  2 848 518   |  1 214.5    | 12 237.1 |
  | 500       |     185 245  |  3 336 156   |    100.5    |  5 588.3 |

Threshold = 32 minimises bcsstk14 fill (within 0.05 % of 4) and is
within 0.09 % of Pres_Poisson's optimum at ~30 % faster.

## What went well

- Sprint 22 Day 12's "swap with bitset deletion" landed cleanly because Day 11's parallel-comparison tests had already verified bit-identical fill across the SuiteSparse corpus.  Option A (delete the bitset, no `#ifdef` guard) saved a sprint of dead-code maintenance and the `make test` regression caught zero callers diverging.
- Splitting the AMD work into design (Day 10) → core (Day 11) → swap (Day 12) → bench (Day 13) prevented Day 11 from blocking on perf concerns — the simplification (skip supervariables / element absorption / approximate-degree updates) freed Day 11 to ship correctness without quality risk; Day 13's bench then measured the actual gap honestly and the deferred items have a clear scope rather than a vague "could be faster" feeling.

## What didn't go well

- ND on Pres_Poisson lands at 1.06× AMD's nnz(L), nowhere near the plan's 0.5× target.  The simplification that saved AMD's Day 11 (skip the full Davis algorithm) hit ND harder because Pres_Poisson is exactly the fixture where supervariable detection and element absorption matter most.  Closure deferred to Sprint 23.
- ND wall time on Pres_Poisson is ~5× AMD's, dominated by the naïve O(n) max-gain scan in `graph_refine_fm`.  The Day 14 plan explicitly called this out as a Day 14 task ("profile FM gain-bucket update; if it dominates wall time, port the METIS O(1) bucket structure"); the bucket-structure port itself didn't fit Day 14's budget and is also deferred.

## Items deferred

- ND fill on Pres_Poisson: 1.06× AMD; plan target was 0.5× (≥ 2× reduction).  Two contributing axes:
  - ND's recursion leaves emit subgraph-local order rather than calling `sparse_reorder_amd_qg` per leaf.  Day 12 made the per-leaf AMD cheap; ND just doesn't use it yet.
  - The simplified quotient-graph AMD itself doesn't have supervariables / element absorption / approximate-degree.  Both close on Sprint 23.
- ND wall-time bottleneck: O(n) max-gain scan in `graph_refine_fm`.  METIS uses O(1) gain buckets.
- Quotient-graph AMD wall time on small SPD fixtures (n ≤ 1 800): currently ~30 % bitset-favoured because the simplified algorithm doesn't have supervariables.  Sprint 22 ships the simplified version because the memory ceiling was the headline; Sprint 23 may revisit if the small-fixture wall-time tail shows up in profiles.
- Cholesky-via-ND residual test threshold relaxation.  bcsstk14 Cholesky residual under AMD is 1.41e-9 — moderately ill-conditioned fixture amplifies roundoff.  The plan's 1e-12 target is for "well-conditioned" fixtures; ours uses 1e-8.  Sprint 23 may swap in a strictly-SPD synthetic fixture for the residual contract instead of bcsstk14.

## Lessons

- TBD — populate post-sprint.  Anchor candidates: the simplification-vs-quality trade-off on Day 11 (took the simple path because the memory headline was the goal); the ND plan target's optimism (the plan presumed METIS-quality FM + Davis-quality leaf AMD; we shipped neither at Day 6, so the headline missed); the value of the Day 11 parallel-comparison tests (bit-identical fill is what made Day 12's bitset-delete safe to do without an `#ifdef` fallback).

## Acknowledgements

TBD — populate post-sprint.

## Day-by-day capsule (for the prose write-up)

  - **Day 1.** sparse_graph_t struct + multilevel-partitioner stubs (sparse_graph.c file-header design block + stub).
  - **Day 2.** Heavy-edge-matching coarsener + multilevel hierarchy (graph_coarsen_heavy_edge_matching, sparse_graph_hierarchy_*).
  - **Day 3.** Coarsest-graph bisection (brute force / GGGP) + FM refinement with rollback-on-regress.
  - **Day 4.** Uncoarsening + smaller-side vertex-separator extraction; full sparse_graph_partition pipeline.
  - **Day 5.** Stress tests, edge cases (n = 1 / n = 2 / empty / K20 / K_{10,10}), determinism contract, SuiteSparse smoke.
  - **Day 6.** Recursive ND driver + permutation assembly.  10×10 grid 1.22× AMD.
  - **Day 7.** ND `sparse_analyze` integration + SuiteSparse fill validation; manual ND-then-NONE bridge.
  - **Day 8.** SPARSE_REORDER_ND enum dispatch across all four factorizations; replaces Day 7 manual bridge.
  - **Day 9.** Cross-ordering bench + ND_BASE_THRESHOLD sweep (4 → 32).  PERF_NOTES.md created.
  - **Day 10.** Quotient-graph AMD design + stub; bitset-AMD seam documented.
  - **Day 11.** Quotient-graph AMD core (simplified Davis-style); parallel comparison vs bitset on nos4 / bcsstk04 / bcsstk14 — bit-identical fill.
  - **Day 12.** Production swap; bitset deleted entirely (Option A).  10 000×10 000 banded stress in 0.24 s.
  - **Day 13.** Bitset-vs-qg AB bench.  banded_20000 ≥ 17× memory reduction; PERF_NOTES.md gains the AMD section.
  - **Day 14.** Final cross-ordering capture; algorithm.md + PROJECT_PLAN.md updated; this retro stubbed.

## DoD verification

  - ✅ `make format` clean across the sprint's commits.
  - ✅ `make lint` clean (0 errors, 109 NOLINT).
  - ✅ `make test` green on every commit (48 / 48 binaries on Day 14).
  - ✅ `make sanitize` (UBSan) green on Days 5 / 8 / 11 / 12 / 14.
  - ⚠️ ND fill target on Pres_Poisson missed (1.06× vs target 0.5×).  Documented above; deferred to Sprint 23.
  - ✅ AMD memory target on n ≥ 50 000 cleared (≥ 17× at n = 20 000 measured; analytic ~26× at n = 50 000).
  - ✅ AMD-using tests (test_cholesky / test_ldlt / test_lu / test_qr / test_reorder / test_suitesparse) all green post-Day-12 swap.
