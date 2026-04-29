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
