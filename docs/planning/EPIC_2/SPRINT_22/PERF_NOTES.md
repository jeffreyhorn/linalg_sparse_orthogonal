# Sprint 22 — Performance Notes

Performance measurements collected during Sprint 22's nested-dissection
work.  This file accumulates as the sprint proceeds; today's entry is
the Day 9 cross-ordering benchmark.

## Reordering comparison (Day 9)

`benchmarks/bench_reorder.c` runs the four fill-reducing orderings
(RCM / AMD / COLAMD / ND) against six SuiteSparse fixtures and
reports `nnz(L)` from `sparse_analyze`'s symbolic Cholesky factor
plus the wall-clock reorder time.  Full capture lives at
`bench_day9_nd.txt` (human-readable) and `bench_day9_nd.csv`.

### Headline ND-vs-AMD nnz(L) ratios

| Fixture       |     n   | AMD nnz(L) | ND nnz(L)  | ND/AMD |
| ------------- | ------: | ---------: | ---------: | -----: |
| nos4          |     100 |       637  |     1 091  |  1.71× |
| bcsstk04      |     132 |     3 143  |     3 683  |  1.17× |
| Kuu           |   7 102 |   406 264  |   943 463  |  2.32× |
| bcsstk14      |   1 806 |   116 071  |   140 024  |  1.21× |
| s3rmt3m3      |   5 357 |   474 609  |   481 904  |  1.02× |
| Pres_Poisson  |  14 822 | 2 668 793  | 2 837 046  |  1.06× |

ND is currently 1.0× – 2.3× of AMD's nnz(L) across the corpus.  On
the canonical 2D-PDE fixture (Pres_Poisson) the ratio is 1.06× — far
from the Sprint 22 plan's < 0.5× target.

### Why ND under-performs at Day 9

Two contributors that Day 12 and Day 14 are scoped to address:

1. **Natural-order base case.**  When the recursion reaches a
   subgraph of `n ≤ ND_BASE_THRESHOLD` (default 32 from the Day 9
   sweep below), the leaves emit vertex indices in their
   subgraph-local order — adequate for the separator-last rule to
   dominate fill at the upper recursion levels but well below AMD's
   minimum-degree quality at the leaves.  Day 12 swaps the leaf
   orderer for the new quotient-graph AMD that Days 10-11 ship.

2. **FM refinement throughput.**  Day 5's smoke test measured the
   Day-6 ND at ~38 s on Pres_Poisson — ~5× slower than AMD's
   reorder pass on the same fixture.  The naïve O(n) max-gain scan
   in `graph_refine_fm` is the prime suspect; METIS uses an O(1)
   bucket structure that Day 14's profiling pass is expected to
   port over.

### ND_BASE_THRESHOLD sweep

The Day-6 default was 4 (recurse to single vertices).  The Day 9
sweep on bcsstk14 + Pres_Poisson picks 32:

| Threshold | bcsstk14 nnz(L) | Pres_Poisson nnz(L) | bcsstk14 ms | PP ms     |
| --------: | --------------: | ------------------: | ----------: | --------: |
| 4         |     140 102     |       2 834 569 ←   |    2 175.7  | ~38 000   |
| 8         |     140 100     |              —      |    2 236.7  |     —     |
| 16        |     140 100     |              —      |    2 303.9  |     —     |
| 32        |     140 024 ←   |       2 837 046     |    2 180.2  | 23 633.0  |
| 64        |     142 265     |       2 848 518     |    1 214.5  | 12 237.1  |
| 128       |     146 967     |       2 906 863     |      565.8  |  8 653.0  |
| 200       |     163 016     |       2 991 452     |      135.8  |  7 212.2  |
| 500       |     185 245     |       3 336 156     |      100.5  |  5 588.3  |

Threshold = 32 is the minimum-fill choice on bcsstk14 (within 0.05 %
of 4 / 8 / 16) and within 0.09 % of the minimum on Pres_Poisson at
~30 % faster.  The natural break is between 32 and 64: deeper
recursion keeps separator-last dominating the leaves' natural
ordering, while shallower recursion starts to leave fill on the
table as leaf subgraphs grow large.

Day 12's quotient-graph AMD swap will likely move this higher (the
leaf orderer becomes substantially better than natural ordering, so
the recursion can afford larger leaf subgraphs).

## AMD bitset → quotient-graph swap (Day 13)

`benchmarks/bench_amd_qg.c` keeps the pre-Day-12 bitset as a static
bench-local helper alongside the production `sparse_reorder_amd`
(now a thin wrapper around `sparse_reorder_amd_qg`).  Full capture
lives at `bench_day13_amd_qg.txt` (human-readable) and
`bench_day13_amd_qg.csv`.

### Headline: memory reduction on n ≥ 20 000

Plan target: ≥ 4× memory reduction on a synthetic large fixture
(n ≥ 50 000).  The 20 K row already clears it:

| Fixture       |     n   | bitset bytes (analytic) | qg bytes (analytic) | ratio |
| ------------- | ------: | ----------------------: | ------------------: | ----: |
| banded_20000  |  20 000 |             50 MB       |             ~3 MB   |   17× |
| banded_50000  |  50 000 |            312 MB       |            ~12 MB   |   26× |

The 50 K row is analytic only — the bitset's O(n³/64) elimination
loop runs in multi-minute territory at that size and produces no
information beyond the memory ratio.  The 20 K row is measured (via
`getrusage` ru_maxrss delta around the call) and the bitset added
24.88 MB above the quotient-graph's already-set peak; the
quotient-graph version's own delta was 0 MB, consistent with its
analytic ~3 MB sitting under the bench's startup RSS.

### nnz(L) parity (every fixture)

| Fixture       |     n   | bitset nnz(L) | qg nnz(L)     |
| ------------- | ------: | ------------: | ------------: |
| nos4          |     100 |         637   |         637   |
| bcsstk04      |     132 |       3 143   |       3 143   |
| bcsstk14      |   1 806 |     116 071   |     116 071   |
| Kuu           |   7 102 |     406 264   |     406 264   |
| s3rmt3m3      |   5 357 |     474 609   |     474 609   |
| Pres_Poisson  |  14 822 |   2 668 793   |   2 668 793   |
| banded_5000   |   5 000 |      29 985   |      29 985   |
| banded_10000  |  10 000 |      59 985   |      59 985   |
| banded_20000  |  20 000 |     119 985   |     119 985   |

Bit-identical fill across the corpus + synthetic banded — both
implementations are exact minimum-degree on the same graph with the
same lowest-vertex-id tie-break.

### Wall-time speedup

The quotient-graph implementation's O(deg) per-pivot merge stays
linear in nnz, while the bitset's O(n²/64) merge is quadratic in n
regardless of nnz.  This shows up as 4-7× speedup on banded
fixtures (n ≥ 5 000):

| Fixture       |     n   | bitset_ms | qg_ms    | speedup |
| ------------- | ------: | --------: | -------: | ------: |
| banded_5000   |   5 000 |     263.4 |    59.5  |   4.4×  |
| banded_10000  |  10 000 |   1 211.8 |   233.2  |   5.2×  |
| banded_20000  |  20 000 |   6 194.6 |   919.5  |   6.7×  |
| Kuu           |   7 102 |     702.4 |   396.9  |   1.8×  |

On small SPD corpus fixtures (n ≤ 1 800) the bitset wins by ~30 %
on wall time — its inner loop is bit-twiddling against pre-allocated
words, while the quotient-graph version pays for sorted-merge
intermediate buffers.  The crossover sits around n = 5 000, matching
the analytic memory crossover.

### What's left for Sprint 23 (or later)

The current quotient-graph implementation is a Davis-style "simple
quotient-graph minimum-degree" — it skips supervariable detection,
element absorption, and approximate-degree updates that the full
SuiteSparse AMD reference uses.  The full algorithm would tighten
the wall-time gap on small SPD fixtures (currently 30 % bitset-
favoured) and likely close the absolute gap to METIS-AMD on PDE-
mesh corpora.  Sprint 22 ships the simplified version because it
already lifts the memory ceiling that was blocking n ≥ 50 000 use
cases; the wall-time tail is a Sprint-23 optimisation if it shows
up in profiles.
