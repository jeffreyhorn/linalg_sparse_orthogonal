# Davis 2006 §7 — AMD reference notes (Sprint 23 prep)

Reference: Davis, T. A. (2006). *Direct Methods for Sparse Linear Systems*. SIAM
Fundamentals of Algorithms, vol. 2.  ISBN 978-0-89871-613-9.  Chapter 7 covers
AMD; the original paper is Amestoy, Davis, Duff (2004) "Algorithm 837: AMD, an
Approximate Minimum Degree Ordering Algorithm", *ACM TOMS* 30:381–388 (the
TOMS paper compresses §7 into a tighter form; both worth keeping handy).

These notes capture the parts of §7 that Sprint 23 Days 2-6 implement.  Page
numbers cite the SIAM book; algorithm-line numbers cite Algorithm 7.x in the
chapter.

---

## §7.3 — Element absorption (Days 2-3)

**The mechanism.**  As elimination proceeds, each pivot creates a new
*element* — the clique formed by the pivot's neighbours.  When a variable's
remaining variable-side adjacency reduces to a single element `e`, that
variable is absorbed into `e`: it disappears from the active set, and its
workspace slot is reclaimed by the next compaction pass.

**Why it matters.**  Without absorption, the active variable count stays at
`n - eliminated_so_far` throughout elimination.  With absorption, the active
set shrinks faster — on PDE meshes, the typical end-of-elimination active
count is `O(√n)` vertices (the supervariables of the dense top of the
elimination tree), not `O(n)`.  This is what bounds the per-pivot cost from
`O(n)` (Sprint 22) to `O(active_count)` ≈ `O(supervariable_count)`.

**Workspace layout (Algorithm 7.4 / book p.140).**  Each variable `i` owns a
slice of `iw[]` of length `len[i]`.  The slice is split into two regions:

```
iw[xadj[i] .. xadj[i] + len[i] - elen[i] - 1]   variable-side adjacency
iw[xadj[i] + len[i] - elen[i] .. xadj[i] + len[i] - 1]   element-side adjacency
```

Where `elen[i]` is the per-variable count of element-side adjacency entries
(new state Sprint 23 adds).  At init time `elen[i] = 0` for every variable —
the entire slice is variable-side.  As pivots eliminate, neighbours' slices
get a new element-ID appended on the element-side; their `elen[i]`
increments; on the next compaction the slice is repacked with the
`len[i] - elen[i]` variable entries first, then the `elen[i]` element
entries.

**Algorithm 7.4 hot loop (when pivot p eliminates):**
1. Form the new element `e` (reusing `p`'s slot — Davis's index-recycling
   convention).  Write `e`'s variable-adjacency = ⋃ neighbours' variable
   adjacency \ {p, eliminated} into `iw[]`'s elements-side region.
2. For each neighbour `u` of `p`: append `e` to `u`'s element-side
   adjacency; remove `p` from `u`'s variable-side adjacency.
3. For each neighbour `u`: if `u`'s variable-side adjacency is now empty
   (`len[u] - elen[u] == 0`) **and** `u`'s element-side has converged to a
   single element, mark `u` absorbed.  The next compaction skips `u` and
   reclaims its slot.

**Pres_Poisson / mesh fixtures expectation.**  ~5–10× active-set shrinkage
toward the end of elimination.  bcsstk14: smaller (~2–3×) — structural
mechanics matrices have less geometric regularity.

**Day 3 measurement (post-implementation, basic absorption only).**
Strict variable absorption per Davis §7.3 (variable-side empty AND
element-side has exactly one element) fires zero times on the
SuiteSparse corpus (nos4, bcsstk04, bcsstk14, banded_10000):

```
qg-probe n=100   iw_peak=4188   iw_size=4259   absorbed=0 (0.0%)   nos4
qg-probe n=132   iw_peak=25468  iw_size=25669  absorbed=0 (0.0%)   bcsstk04
qg-probe n=1806  iw_peak=445976 iw_size=445985 absorbed=0 (0.0%)   bcsstk14
qg-probe n=10000 iw_peak=649710 iw_size=779791 absorbed=0 (0.0%)   banded_10000
```

Why: under basic absorption alone, `elen[u]` grows monotonically as
pivots eliminate.  For `elen[u] == 1` to trigger absorption, `u` must
have touched exactly one prior pivot and have no remaining variable
neighbours — essentially a degree-1 leaf vertex at elimination time,
which doesn't occur in dense corpus fixtures.

The mechanism that makes absorption fire is **aggressive element
absorption** (Davis §7.3.4): when a new element `e` is created and an
older element `e'` is a subset of `e`'s variable-set, `e'` is absorbed
into `e` and `e'`'s ID is removed from every variable's element-side
that referenced it.  This *can* drive `elen[u]` down to 1 across
elimination.  Aggressive absorption is deferred (per `davis_notes.md`
"What Sprint 23 deliberately doesn't implement"); it's a Sprint 24+
candidate.

Fill-quality consequence: bit-identical to Sprint 22 (verified —
nos4=637, bcsstk04=3143, bcsstk14=116071 all match `bench_day14.txt`).
The Davis representation is in place; the workspace win arrives when
Day 5's approximate-degree formula is wired up plus aggressive
absorption follows.

---

## §7.4 — Supervariable detection (Day 4)

**The mechanism.**  Two variables `i` and `j` are *indistinguishable* if
they have the same variable-side adjacency *and* the same element-side
adjacency.  Indistinguishable variables can be merged into a single
*supervariable* and eliminated together — pivot order within a supervariable
doesn't change anything downstream.

**Why it matters.**  The minimum-degree pivot scan in `qg_pick_min_deg` is
the dominant per-pivot cost in Sprint 22's qg AMD (it's an O(active_count)
scan).  Operating on supervariables instead of variables shrinks the scan by
the supervariable factor.  On PDE meshes the factor is 5–20× by mid-
elimination; on irregular sparsity it's typically 2–5×.

**Hash signature (book p.144).**  Davis uses a sum-of-IDs signature:

```
hash(i) = sum_{v in adj_var(i)} v + sum_{e in adj_elem(i)} e   (mod n_buckets)
```

Cheap to compute (`O(len[i])`), cheap to update incrementally (when an entry
is added/removed from `i`'s adjacency, just add/subtract the entry's ID
from the running hash).  Collision rate: low on real fixtures because
adjacency-list contents tend to be locally-clustered, so two variables
needing to collide are usually genuinely indistinguishable.

**Algorithm 7.5 collision handling (book p.145).**  After a pivot
eliminates, walk the affected variables:
1. Bucket them by `hash(i) mod n_buckets`.
2. Within each bucket, do an O(k²) full-list compare on every pair.  When
   a pair compares equal, merge them: the smaller-supervariable
   representative folds into the larger.
3. Update `super[i] = representative_id` and `super_size[representative] +=
   super_size[i]`.

**Fold direction (book p.145, footnote).**  Folding the smaller into the
larger keeps the union-find paths short — important because the
post-merge degree query reads through `super[]`.

**Sprint 23 simplification.**  Davis recommends a fixed bucket array of
size `n` reused across pivots.  We can do the same — `qg_t` already has
the per-vertex temporary arrays needed; allocate the bucket array once in
`qg_init` instead of per-pivot.

---

## §7.5 — Approximate-degree formula (Day 5)

### Day 5 production-default decision: exact-degree (opt-in approx)

The Sprint 23 plan called for switching the production deg-update
from exact recompute to Davis's approximate formula.  Day 5
implemented the switch but measured significant fill regression on
PDE meshes (1.45–2.84× of Sprint 22 baseline; details below).
Achieving the textbook "few-percent" quality requires the
*external-degree* refinement from Davis §7.5.1, which Sprint 23
deliberately defers (see "What Sprint 23 deliberately doesn't
implement" below).

To preserve the Sprint 22 / `bench_day14.txt` fill baseline while
landing the Day-5 framework, Day 5 ships the approximate-degree
formula behind an opt-in env var:

```
SPARSE_QG_USE_APPROX_DEG=1   ./build/test_reorder_amd_qg
                             # production path uses approximate
                             # degree (regresses fill, faster
                             # per-pivot)
SPARSE_QG_VERIFY_DEG=1       # always run both, assert
                             # d_approx >= d_exact (Davis's
                             # conservative-bound contract)
```

The default production path uses exact recompute (matching Sprint
22 fill bit-identically).  The framework + parity test land for
Sprint 24's external-degree extension to plug into.

### Day-6 measurement: bcsstk14 wall time (probe)

| Mode                                         | Full test suite | Notes                              |
| -------------------------------------------- | --------------: | ---------------------------------- |
| Default (exact-degree)                       |       11.84 s   | Sprint 22 baseline behaviour       |
| `SPARSE_QG_USE_APPROX_DEG=1`                 |       57.51 s   | Davis approx-degree (this commit)  |

The USE_APPROX run is 4.9× slower than exact-default for the full
test suite, contradicting the Sprint 23 plan's "expected ~2×
speedup (Days 2-5 cumulative)" claim.  Root cause: both formulas
have the same per-pivot complexity (walk variable-side + walk
each element's variable-set), but the approximate formula's
looseness produces a worse pivot order which cascades into more
fill which means more iw[] entries to walk on subsequent pivots.
Net wall-time is *worse*, not better.

The plan's wall-time win presumed the approximate formula would
match exact's fill quality (within "few percent") so the per-pivot
saving would translate to total savings.  Without external-degree
tracking (Davis §7.5.1, deferred), the textbook formula's
coarser pivot order erases the per-pivot saving.

### Day-6 measurement: cap-fire counts

The `cap_fired` probe (added Day 6) counts firings of the
`d > qg->n` cap inside `qg_compute_deg_approx`.  Across the
existing test suite:

```
nos4 (n=100, exact prod):       cap_fired=0   (cap path inactive)
bcsstk04 (n=132, exact prod):   cap_fired=0
bcsstk14 (n=1806, exact prod):  cap_fired=0
50-vert  (n=50, VERIFY only):   cap_fired=83  (approx side runs alongside exact)
200-vert (n=200, USE_APPROX):   cap_fired=0   (approx-driven elimination)
hub      (n=200, USE_APPROX):   cap_fired=0   (approx-driven elimination)
```

The cap fires on the 50-vertex random graph (sparsified random
edges) when running as a parity-check alongside exact-driven
elimination.  Under USE_APPROX (approx drives the elimination),
the cap doesn't fire on these synthetic fixtures — likely
because approx-driven elimination follows different pivots that
don't accumulate the cross-element overlap needed to push d > n.
The cap remains a defensive guard against the qg_pick_min_deg
sentinel constraint rather than a frequently-exercised path.

### Day-5 measurement: fill regression on PDE meshes

Switching `qg_eliminate` from exact-degree recompute to Davis's
approximate formula produced these `nnz(L)` shifts:

| Fixture       |     n   | Sprint 22 nnz(L) | Day 5 nnz(L) | ratio |
| ------------- | ------: | ---------------: | -----------: | ----: |
| nos4          |     100 |             637  |         819  | 1.29× |
| bcsstk04      |     132 |           3 143  |       3 761  | 1.20× |
| bcsstk14      |   1 806 |         116 071  |     168 413  | 1.45× |
| Pres_Poisson  |  14 822 |       2 668 793  |   7 581 360  | 2.84× |
| 100×100 bw=5  |     100 |             585  |         585  | 1.00× |

The formula is fill-equivalent for banded matrices (where every
variable has small adjacency and few elements form) but degrades
progressively on PDE meshes — the "few-percent" textbook claim
holds when external-degree tracking (Davis §7.5.1) is layered on
top, which Sprint 23 deliberately defers.

Wall-time impact on Pres_Poisson: ND-via-AMD fill increases drive
proportionally longer Cholesky factorizations downstream — the
test_reorder_nd suite went from ~25 minutes total in Sprint 22 to
~7 hours in Day 5.  The per-pivot AMD cost itself didn't regress
(approx and exact have same complexity); the regression is
entirely from the worse pivot order producing more fill.

### Sprint-24+ candidate: external-degree tracking

**The formula (book p.147, eqn 7.5).**

```
d_approx(i) = |adj_var(i, V \ {pivot})|
            + |L_pivot \ {i}|
            + sum_{e in adj_elem(i) \ {L_pivot}} |adj(e, V \ {pivot}) \ adj(i)|
```

Where `L_pivot` is the new element formed by the just-eliminated pivot,
`V` is the current active variable set, and the third term is a "fresh
contribution from each older element" — it counts variables reachable via
element `e` that aren't already in `i`'s variable-side adjacency.

**Why it's an upper bound.**  The exact degree `d(i) = |adj(i, V)|`
counts each neighbour exactly once.  The approximate formula counts
neighbours reachable via multiple elements multiple times — the
set-difference in the third term reduces but doesn't eliminate the
double-counting.  Tight on regular meshes (the "double-counting"
overlap is small when elements have nearly-disjoint adjacency); looser
on dense rows.

**Cost reduction.**  Exact degree: `O(|adj(i, V)|)` per neighbour, so
`O(adj²)` per pivot.  Approximate: `O(|adj_elem(i)|)` per neighbour
(reads element adjacency lists, doesn't expand them), so `O(adj * elem)`
per pivot — which on PDE meshes is `O(adj * O(1))` because most
variables connect to a small constant number of elements.

**Dense-row skip (book p.148).**  Davis recommends skipping the degree
update entirely when `d_approx(i) > 10·√n`.  Rationale: the formula is
least accurate for dense rows (lots of double-counting), and dense rows
dominate the pivot scan anyway — they're not chosen until late, by
which time the absorbed-into-element accounting has cleaned up the
double-counting.  Use `qg->skipped_dense_count` as a probe Sprint 23
Day 6 verifies.

**Conservative-bound contract.**  `d_approx(i) >= d_exact(i)` for every
non-skipped vertex.  This is what Sprint 23 Day 6's parity test pins —
the approximate formula must never under-estimate, or AMD's
minimum-degree pivot loop would pick the wrong pivot and fill quality
would degrade.

---

## Workspace size implications (Sprint 23 Day 2 sizing)

Sprint 22's initial `iw_size = 5·nnz + 6·n + 1` accommodated variable-side
adjacency only.  Davis 2006 §7 uses `iw_size = 7·nnz + 8·n + 1` — the
extra `2·nnz + 2·n` is the elements-side reserve.  The compaction +
realloc growth path stays in place; the bigger initial size just defers
the first realloc by a few pivots.

---

## What Sprint 23 deliberately *doesn't* implement

- **Aggressive absorption (book p.142, §7.3.4).**  Davis describes a
  multi-pass absorption that compacts the same variable across multiple
  elements.  Sprint 23 sticks with single-pass: a variable absorbs the
  first time its variable-side empties.  Absorbed-multiple-times-in-one-
  pivot is a Sprint 24+ optimization; the per-pivot work-save is small.

- **External degree (book p.149, §7.5.1).**  Davis describes a tighter
  bound that uses the actual neighbour-count of each element rather
  than the degree-count.  Sprint 23 uses the simpler approximate
  formula above; tightening to external degree is a Sprint 24+ closure
  if AMD wall-time still trails SuiteSparse reference.

- **Multiple eliminations per pivot step (book §7.6).**  Davis describes
  eliminating all of a supervariable in a single batched step.  Sprint
  23's supervariable detection just merges and pivots them as a unit,
  which gives the same result without the batch-bookkeeping
  complexity.

These three are noted here so they're not lost — Sprint 24's "AMD
parity with SuiteSparse" follow-up (if PROJECT_PLAN.md acquires one)
would consume them.

### Day-11 finding: multi-pass FM at the finest level

PLAN.md §11.4 budgeted a 2-hour exploration of multi-pass FM —
the question is whether Sprint 22's single-pass-per-uncoarsening-
level FM is leaving cut quality on the table.  Davis 2006 §4.2
notes that FM converges to a local minimum after 2-3 passes from
a given starting partition; METIS's reference implementation runs
multi-pass FM at the finest level by default for the same reason.

The Day-9 / Day-10 gain-bucket FM makes each pass cheap enough to
afford running multiple passes — Sprint 22's O(n²) FM made even
single-pass expensive on Pres_Poisson.  Sweep result on Pres_Poisson
(n = 14 822, end-to-end `sparse_reorder_nd` + symbolic Cholesky):

| `SPARSE_FM_FINEST_PASSES` | nnz(L)    | ND/AMD ratio | ND wall |
|---------------------------|-----------|--------------|---------|
| 1 (Sprint 22 default)     | 2 737 253 |   1.026 ×    | 47.3 s  |
| 2                         | 2 556 617 |   0.958 ×    | 41.4 s  |
| 3 (chosen)                | 2 541 734 |   0.952 ×    | 40.5 s  |
| 5                         | 2 543 161 |   0.953 ×    | 41.2 s  |

Pass 2 captures most of the win (ratio jumps 1.026 → 0.958 ×); pass
3 tightens further (→ 0.952 ×); pass 5 sits at the same ratio (no
further win — converged at the FM local optimum).  Wall time
*drops* 6 seconds with 3 passes vs 1: a tighter partition produces
fewer cross-edges to chase during the recursive-ND descent, so the
cumulative downstream work shrinks faster than the extra passes add.

Decision: adopt 3-pass at the finest level by default.  Override
via `SPARSE_FM_FINEST_PASSES` env var (1..16) for regression
bisection.  Intermediate-level passes stay at 1 — those levels see
mostly-converged partitions from coarsening, and adding passes
there is wall-time cost without measurable fill win.

Pres_Poisson now lands at 0.95 × AMD — the headline fill-quality
gate from Sprint 22 onwards.  PLAN.md Day-8's literal target was
≤ 0.7 ×; not achieved (and remains Sprint-24 territory per the
risk-flag #2 fallback) but ND now consistently *beats* AMD on this
2D-PDE benchmark, which was the intent behind the relaxed
≤ 0.7 × stretch goal.

Smaller fixtures (10×10 grid, bcsstk04) see no measurable change
from multi-pass FM — their partitions are already converged after
single-pass.  Same picture for bcsstk14 / Kuu in the post-multilevel
experiment that informed this decision (`/tmp/test_multipass_after`,
not committed — the `SPARSE_FM_FINEST_PASSES` env var subsumes it
as the canonical multi-pass mechanism).
