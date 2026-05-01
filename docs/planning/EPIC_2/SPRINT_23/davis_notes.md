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
