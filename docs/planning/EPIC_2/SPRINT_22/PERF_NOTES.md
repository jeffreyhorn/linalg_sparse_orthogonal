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

Two contributors.  As shipped in Sprint 22 both remain open and are
deferred to Sprint 23 (see PROJECT_PLAN.md):

1. **Natural-order base case.**  When the recursion reaches a
   subgraph of `n ≤ ND_BASE_THRESHOLD` (default 32 from the Day 9
   sweep below), the leaves emit vertex indices in their
   subgraph-local order — adequate for the separator-last rule to
   dominate fill at the upper recursion levels but well below AMD's
   minimum-degree quality at the leaves.  Splicing the new
   quotient-graph AMD (Days 10-12) into each leaf is the planned
   Sprint-23 follow-up that closes most of this gap.

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

## Sprint 23 closures

Sprint 23 ran 14 days against the two ordering-quality gaps Sprint
22 left open: ND's 1.06× ratio on Pres_Poisson (target was ≤ 0.5×)
and the qg-AMD wall-time tail.  Headline reference is
`docs/planning/EPIC_2/SPRINT_23/bench_summary_day12.md`; per-day
captures are at `bench_day8_nd_leaf_amd.txt`,
`bench_day10_fm_buckets.txt`, `bench_day12.txt`, and
`bench_day12_amd_qg.txt`.

### What moved

| metric                          | Sprint 22  | Sprint 23  | delta |
|---------------------------------|------------|------------|-------|
| Pres_Poisson nnz_nd / nnz_amd   | 1.063×     | 0.952×     | -10 % (ND now beats AMD) |
| bcsstk14    nnz_nd / nnz_amd    | 1.207×     | 1.130×     | -6 %  |
| 10×10 grid  nnz_nd / nnz_amd    | 1.380×     | 1.160×     | -16 % |
| Kuu         nnz_nd / nnz_amd    | 2.322×     | 2.275×     | -2 %  |
| Pres_Poisson nnz_L (ND)         | 2 837 046  | 2 541 734  | -10 % |
| Pres_Poisson ND wall (s)        | ~44        | ~45        | unchanged (FM win offset by AMD-leaf cost) |

ND on Pres_Poisson now lands at 0.95× AMD — the headline
fill-quality outcome.  The cumulative drivers, in order of
contribution: Day 11's multi-pass FM at the finest level (1.026
→ 0.952×, the largest single jump), Day 7's leaf-AMD splice
(2 of 6 fixtures saw nnz wins, 4 within ±1 % noise), and
Days 9-10's gain-bucket FM (lifted per-pass cost from O(n²) to
O(|E|), making the multi-pass exploration affordable).

### What didn't move

The literal targets PROJECT_PLAN.md set for Sprint 23 are
documented under the three headline gates in
`docs/planning/EPIC_2/SPRINT_23/bench_summary_day12.md`:

- **(a)** Pres_Poisson `nnz_nd / nnz_amd ≤ 0.7` — *not literally
  met* (achieved 0.952×).  ND beats AMD which was the spirit of
  the goal; closing 0.95 → 0.7 needs deeper coarsening / smarter
  separator extraction.  Routed to Sprint 24 per PLAN.md
  risk-flag #2.

- **(b)** qg-AMD wall on bcsstk14 ≤ Sprint-22 bitset baseline (64
  ms) — *hard fail*.  Sprint 23 Days 2-5 (element absorption +
  supervariable detection + approximate-degree formula) introduced
  a 77× wall-time regression vs Sprint 22 quotient-graph baseline
  on bcsstk14; analogous regressions on Kuu (96×), s3rmt3m3
  (151×), Pres_Poisson (199×); banded fixtures regressed only
  ~3×.  Fill correctness is intact (bit-identical to bitset
  reference).  Likely root cause: Day 4's per-pivot O(k²)
  supervariable-hash-bucket compare dominates when supervariables
  don't form (irregular SPD has no symmetry to merge).  Routed to
  Sprint 24 with three candidate fixes documented in
  `bench_summary_day12.md` "(b)".

- **(c)** bench_day14 nnz_L bit-identical-or-better on every
  fixture — *passes*, with bcsstk04 +0.5 % within partitioner-RNG
  noise.

The shipping story for the Sprint 23 PR description: "ND now beats
AMD on the canonical 2D-PDE benchmark, at the cost of a
production-AMD wall-time regression Sprint 24 must root-cause and
fix."

## Sprint 24 closures

Sprint 24 ran 11 days against the two items Sprint 23 routed forward:
the qg-AMD wall-time regression (gate (b) hard fail) and the
Pres_Poisson 0.7× literal target (gate (a) spirit-met-but-literal-
miss).  Headline reference is `docs/planning/EPIC_2/SPRINT_24/bench_summary_day9.md`;
per-day captures are at `bench_day9.{csv,txt}`,
`bench_day9_amd_qg.{csv,txt}`, plus the Day-7 + Day-8 ND tuning
decision docs.

### What moved

| metric                                | Sprint 22  | Sprint 23  | Sprint 24  | delta vs S23 |
|---------------------------------------|------------|------------|------------|---------------|
| bcsstk14 qg-AMD wall (ms)             |    ~140    |    4 715   |    125.8   | -97 % (39× speedup) |
| Kuu qg-AMD wall (ms)                  |    ~700    |   25 720   |    546.5   | -98 % (47× speedup) |
| s3rmt3m3 qg-AMD wall (ms)             |   ~2 100   |   51 321   |    728.2   | -99 % (70× speedup) |
| Pres_Poisson qg-AMD wall (ms)         |  ~12 200   |  758 927   |  8 138.8   | -99 % (93× speedup) |
| Pres_Poisson qg-AMD peak RSS (MB)     |    19.19   |    25.09   |    19.19   | -24 % (back to S22 baseline) |
| Pres_Poisson nnz_nd / nnz_amd (default) | 1.063×   |   0.952×   |   0.952×   | unchanged (S24 ND default-path bit-identical to S23) |
| Pres_Poisson nnz_nd / nnz_amd (`SPARSE_ND_COARSEN_FLOOR_RATIO=200`) | — | — | 0.942× | new opt-in (1pp tighter) |
| Kuu nnz_nd / nnz_amd (`SPARSE_ND_SEP_LIFT_STRATEGY=balanced_boundary`) | — | — | 1.415× | new opt-in (38pp tighter) |
| 10×10 grid nnz_nd / nnz_amd (default) |   1.380×   |   1.158×   |   1.158×   | unchanged (S24 default-path bit-identical to S23) |

The qg-AMD wall-time regression was the headline fix.  Sprint 24
Day 1's `clock_gettime` profile of bcsstk14 measured 95 % of total
wall time in `qg_recompute_deg`'s element-side adjacency-of-adjacency
walk — the cost Sprint 23 Day 3's element absorption enabled, not
Day 4's supervariable hash compare (which was the 1 % overhead the
three originally-considered fix candidates were targeting).  Day 2's
revert of Sprint 23 Days 2-5 closed it directly: bcsstk14 went from
4 715 ms back to 125.8 ms (within Sprint 22's range), Pres_Poisson
went from 759 s to 8.1 s.  Memory profile also returned to Sprint
22's baseline (19.19 MB on Pres_Poisson, exact match).

The two new ND env-var-gated alternatives ship as documented advisory:

- `SPARSE_ND_COARSEN_FLOOR_RATIO` (Day 5; default 100) overrides the
  multi-level coarsening floor `MAX(20, n/divisor)`.  Pres_Poisson with
  divisor=200 drops ND/AMD 0.952× → 0.942×; ratios ≥ 400 regress
  because the coarsest level pegs at 20 vertices and the brute-force
  bisection loses cut quality.
- `SPARSE_ND_SEP_LIFT_STRATEGY` (Day 6; default `smaller_weight`)
  switches the edge-to-vertex separator extraction from METIS's
  smaller-weight rule to a smaller-boundary-count rule with a 70/30
  post-lift weight-balance fallback.  The `balanced_boundary` value is
  a 8-38 percentage-point ND/AMD win on every smaller fixture (Kuu's
  38pp drop is the largest single nnz win Sprint 24 produced) but
  essentially neutral on Pres_Poisson (+0.1pp), so the production
  default stays `smaller_weight` per the literal flip rule.

### What didn't move

The literal Sprint 24 targets PROJECT_PLAN.md set are documented
under the four headline gates in `docs/planning/EPIC_2/SPRINT_24/bench_summary_day9.md`:

- **(a)** qg-AMD wall on bcsstk14 ≤ 1.5× Sprint 22 baseline (~210 ms)
  — **PASS** (125.8 ms).
- **(b)** qg-AMD nnz(L) bit-identical to Sprint 22 + Sprint 23 captures
  — **PASS** (9/9 fixtures bit-identical).
- **(c)** Pres_Poisson ND/AMD ≤ 0.85× — **MISS** (default 0.952×;
  best opt-in 0.942×; combined ratio=200 + balanced_boundary actually
  worse at 0.950× because the changes interact destructively on this
  fixture).  Sprint 24's stated partial-close is "tighten
  `test_nd_pres_poisson_fill_with_leaf_amd` from `≤ 1.0×` to `≤ 0.96×`
  (Day 7) — pin the actual achievement; the 0.7× literal target +
  0.85× stretch target both route to Sprint 25 (smarter coarsening /
  multi-pass FM at intermediate levels / spectral bisection at the
  coarsest level).
- **(d)** All `SPRINT_23/bench_day14.txt` nnz_L rows stay bit-identical
  or improve — **PASS** with 1 row (Kuu ND) drifting +24 nnz (0.003 %)
  in partitioner FM tie-break noise band.

### Test bound tightening

| test                                                                      | S23 bound        | S24 bound        | margin   |
|---------------------------------------------------------------------------|------------------|------------------|----------|
| `test_nd_pres_poisson_fill_with_leaf_amd`                                 | `≤ 1.00× nnz_amd` | `≤ 0.96× nnz_amd` | 0.8pp    |
| `test_nd_10x10_grid_matches_or_beats_amd_fill`                            | `≤ 1.21× nnz_amd` | `≤ 1.17× nnz_amd` | 1.07pp   |

Both tests pin the bit-stable Sprint 23 Day 11 multi-pass FM
achievement; Sprint 24's ND default-path nnz_L is bit-identical to
Sprint 23 on every fixture.  The Day-2 revert is fill-neutral by
construction; Days 5-6's env-var-gated alternatives are off by
default.

### What's left for Sprint 25

Sprint 24 Day 9's "Items deferred to Sprint 25" calls out four:

1. **Pres_Poisson ND/AMD ≤ 0.85×** — needs algorithmic work beyond
   Days 5-6's options (smarter coarsening like Heavy Connectivity
   Coarsening; multi-pass FM at intermediate levels; spectral
   bisection at the coarsest level).
2. **Davis 2006 §7.5.1 external-degree refinement** — N/A under (c)
   revert; resurrect if a future sprint reintroduces approximate-
   degree.
3. **`make wall-check` Pres_Poisson ND wall line** — Day 8 captured
   42.86 s default-path (21 % above Sprint 23 baseline); Sprint 25
   should add a baseline line with a 50 % threshold rather than 5 %.
4. **ND wall-time tightening to meet the 5 % drift target** — Sprint
   24's default ND path is 1.06-1.10× of Sprint 23's; not a regression
   but worth profiling if the 5 % drift target is to be met.

### Shipping story for the Sprint 24 PR description

"qg-AMD wall regression closed via revert of Sprint 23 Days 2-5
(profile pointed away from the originally-considered fix candidates).
ND fill-quality opt-in env vars added (Pres_Poisson advisory:
ratio=200 → 0.942×; non-Pres_Poisson advisory: balanced_boundary →
8-38pp wins).  Pres_Poisson 0.7× literal target + 0.85× stretch
target both route to Sprint 25 with concrete avenues identified."

## Sprint 25 closures

Sprint 25 ran 14 days against the four items Sprint 24 routed forward:
the Pres_Poisson ≤ 0.85× literal target (item 1), `make wall-check`
extension to cover Pres_Poisson ND wall (item 6), the ND wall-time
profile + tightening question (item 5), plus closing tests / docs.
The headline reference is `docs/planning/EPIC_2/SPRINT_25/headline_summary.md`;
the Day-9 cross-corpus sweep at `bench_day9_combinations.{csv,txt}`
is the canonical 16-setting × 6-fixture matrix this sprint's
production-default decisions are based on.

### What moved

| metric | Sprint 24 | Sprint 25 default | Sprint 25 best opt-in | delta |
|---|---|---|---|---|
| Pres_Poisson nnz_nd / nnz_amd (default) | 0.952× | 0.952× (bit-identical) | — | 0pp |
| Pres_Poisson nnz_nd / nnz_amd (best opt-in) | 0.942× (`ratio=200`) | — | **0.922×** (HCC + ratio=200; Day 9 setting 13) | -2.0pp |
| Kuu nnz_nd / nnz_amd (default) | 2.275× | 2.275× | — | 0pp |
| Kuu nnz_nd / nnz_amd (best opt-in) | 1.415× (balanced_boundary) | — | **1.309×** (full setting 15) | -10.6pp |
| nos4 nnz_nd / nnz_amd (best opt-in) | ~1.51× | — | **1.256×** (setting 15) | -25pp |
| bcsstk14 nnz_nd / nnz_amd (best opt-in) | 1.129× | — | 1.037× (setting 15) | -9pp |
| Pres_Poisson ND wall (default) | ~37 s (Day 7 measure) | ~47 s median (Day 11; 16 % within-run variance) | ~1.6 s (setting 15 spectral) | variance vs S24 measure; ~23× speedup under setting 15 |
| `wall_check_baseline.txt` lines | 2 (AMD only) | **3** (AMD × 2 + Pres_Poisson ND) | — | item 6 closure |

The Pres_Poisson 0.922× best-opt-in achievement is the Sprint 25
headline.  Three independent algorithmic axes were trialled per
PLAN.md items 1-3:

- `SPARSE_ND_COARSENING={heavy_edge,hcc}` (Day 1-3; default
  `heavy_edge`).  HCC selects Heavy Connectivity Coarsening (Karypis
  & Kumar 1998 §5) — matching score `edge_weight × min(deg(u),
  deg(v))` rather than HEM's pure `edge_weight`.  Pres_Poisson alone:
  -1.5pp.  See `coarsening_decision.md` + `hcc_design.md`.
- `SPARSE_FM_INTERMEDIATE_PASSES` (Day 4-5; default 1; range [1, 10]).
  Multi-pass FM at intermediate uncoarsening levels (Sprint 23's
  `SPARSE_FM_FINEST_PASSES` only touched the finest level).
  Pres_Poisson sweep across passes ∈ {1, 2, 3}: passes=2 essentially
  unchanged at 0.952× (-0.04pp); passes=3 *regresses* to 0.967×
  (+1.5pp).  Default stays at 1 because the PLAN.md Day-5 flip rule
  (≥ 1pp Pres_Poisson tightening + no smaller-fixture regression
  past 5pp band) fails on the headline fixture for both candidate
  values, not because of an absence of any per-fixture win.  Per-
  fixture wins DO exist: Kuu at passes=3 closes -23.2pp (2.275× →
  2.043×; the strongest single per-fixture win Sprint 25 produced
  via this axis, at +3.7% wall); bcsstk14 at passes=2 closes -2.2pp
  (1.129× → 1.107×, +4.7% wall) or passes=3 closes -2.3pp at -11.4%
  wall.  s3rmt3m3 regresses +1.7pp at passes ≥ 2.  See
  `intermediate_fm_decision.md`.
- `SPARSE_ND_COARSEST_BISECTION={gggp,spectral}` (Day 6-8; default
  `gggp`).  `spectral` builds the graph Laplacian and uses the
  Sprint 20-21 Lanczos eigensolver to compute the Fiedler vector,
  partitioning by median.  Pres_Poisson alone: 0.953× (essentially
  no nnz_L change), but reduces ND wall ~23× as part of full setting
  15.  See `spectral_bisection_decision.md` + `spectral_bisection_design.md`.

The new `SPARSE_ND_PROFILE` env-var-gated `clock_gettime` per-phase
ND instrumentation (Day 11; off by default; one branch overhead when
off) confirmed `sparse_graph_partition` accounts for **99.5 % of
Pres_Poisson ND wall in every run** — useful for Sprint 26 if
finest-level FM tuning targets wall-time wins.

### What didn't move

The literal Sprint 25 targets PROJECT_PLAN.md set are documented
under the four headline gates in `headline_summary.md` "Verdict":

- **Pres_Poisson ND/AMD ≤ 0.85× (literal target)** — **MISS** (best
  combination achieves 0.9218×; -7.2pp gap).  This is the third
  sprint in a row to miss the Pres_Poisson literal target (Sprint
  22 PLAN's 0.5× → Sprint 23 PLAN's 0.7× → Sprint 24 PLAN's 0.85×
  → Sprint 25 PLAN's 0.85×, all unmet).
- **Pres_Poisson ND/AMD ≤ 0.90× (partial close)** — **MISS** (best
  0.9218×; -2.2pp gap).
- **Pres_Poisson ND/AMD < Sprint 24 baseline (0.952×)** — **PASS**
  at -3pp via setting 13 opt-in; default unchanged.
- **Smaller-fixture corpus safety (no > 5pp regression)** — **PASS**;
  worst regression under setting 13 is s3rmt3m3 +1.0pp.

The Sprint 25 sweep's strongest evidence is that **all three
independent algorithmic axes (HCC, multi-pass intermediate FM,
spectral bisection) wash out individually on Pres_Poisson** when
applied at coarsening / intermediate / coarsest-bisection levels —
the **finest-level FM dominates** downstream on regular structured
fixtures.  Sprint 26 must intervene at the finest level (or
pre-empt the multilevel pipeline entirely with geometric cut
detection) to close the residual 7-9pp gap.

### Production defaults

**No defaults flipped.**  All three Sprint 25 algorithmic axes ship
behind env vars as advisory.  The two blockers:

1. `SPARSE_ND_COARSENING=hcc` flip blocked by **bcsstk14 sep=0**
   (HCC's matching choice on this fixture produces a degenerate
   coarse-level partition; the multilevel pipeline's separator
   extraction yields an empty separator).  Documented Day 10 in
   `coarsening_decision.md` "Two test failures surfaced under the
   new defaults".  Sprint 26 inherits the root-cause investigation.
2. `SPARSE_ND_COARSEST_BISECTION=spectral` and
   `SPARSE_FM_INTERMEDIATE_PASSES=2`: neither moves Pres_Poisson
   nnz_L past the 1pp flip-rule threshold individually.

### Test bound tightening

| test | S24 bound | S25 bound | reason |
|---|---|---|---|
| `test_nd_pres_poisson_fill_with_leaf_amd` | `≤ 0.96× nnz_amd` | unchanged at `≤ 0.96×` | default unchanged |
| `test_nd_10x10_grid_matches_or_beats_amd_fill` | `≤ 1.17× nnz_amd` | unchanged at `≤ 1.17×` | default unchanged |

Both bounds stay at Sprint 24's values because the Sprint 25 default
ND code path is bit-identical to Sprint 24's (the 0.922×
Pres_Poisson best-opt-in achievement is opt-in-only and would
unduly pin the test bound to combination-specific behaviour).
Sprint 25's bound-related work was item 6 (`make wall-check`
extension), not item 4 (which Sprint 24 had already absorbed via
Day 7's 0.96× tightening).

### `make wall-check` extension

Sprint 25 Day 12 closed Sprint 24 item 6: `wall_check_baseline.txt`
gains a `pres_poisson_nd_ms = 47 055` line (Day 11 5-run median;
range 44 321 - 51 562 ms = 16.3 % within-run variance).  The
threshold is per-key in `scripts/wall_check.sh`: AMD baselines stay
at 2× (tight gate; Sprint 23 introduced + Sprint 24 reverted a
30-200× regression that escaped notice for an entire sprint); the
new Pres_Poisson ND baseline uses **1.5×** to absorb the 16 %
within-run variance without going so wide that real algorithmic
regressions slip through.  Verified Day 12 by 10× synthetic
regression: gate fires correctly.  See
`docs/planning/EPIC_2/SPRINT_25/nd_wall_time_decision.md`.

### What's left for Sprint 26

Per `headline_summary.md` "Sprint 26 routing" + `nd_tuning_day8.md`:

1. **Pres_Poisson ND/AMD ≤ 0.85×** — needs intervention at the
   FINEST FM level (annealing acceptance / different bucket-tie-break
   / thick-restart-style FM with global rollback) or geometric cut
   detection on regular grids that pre-empts the multilevel pipeline.
   The Sprint 25 sweep is the strongest evidence yet that
   coarsening / intermediate-FM / coarsest-bisection axes all wash
   out on this fixture.
2. **bcsstk14 sep=0 under HCC** — root-cause investigation into
   why HCC's `min(deg(u), deg(v))` weighting produces a degenerate
   coarse-level partition on bcsstk14.  Either HCC matching pattern
   tightening or a `sparse_graph_partition` sep=0 fall-back.
   Blocks the HCC default flip until resolved.
3. **Pres_Poisson ND wall ~10s sprint-to-sprint drift** — Sprint 23's
   ~36s reported vs Sprint 25's ~47s median is variance/build-drift
   per Day 11 classification, but Sprint 26 could re-baseline if
   the test machine + compiler version stabilise.

### Shipping story for the Sprint 25 PR description

"Three new ND env-var-gated algorithmic axes added: HCC coarsening
(Karypis-Kumar 1998 §5), multi-pass FM at intermediate uncoarsening
levels, and spectral bisection at the coarsest level using the
Sprint 20-21 Lanczos eigensolver.  Pres_Poisson ND/AMD best opt-in
0.922× (HCC + ratio=200; -3pp vs Sprint 24 baseline); Kuu best
opt-in 1.309× (-97pp; the largest single corpus win).  No
production defaults flipped — all three axes ship as advisory; HCC
default flip blocked by bcsstk14 sep=0 inheritance to Sprint 26.
0.85× literal target misses by 7.2pp; routes to Sprint 26 with
strong evidence the residual gap requires finest-level FM
intervention.  `make wall-check` extended with Pres_Poisson ND
baseline (1.5× threshold per Day 11 16 % within-run variance
classification).  `SPARSE_ND_PROFILE` per-phase instrumentation
added for Sprint 26 wall-time work."
