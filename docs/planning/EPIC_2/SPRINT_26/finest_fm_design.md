# Sprint 26 Day 6 — FINEST FM Sub-Axis Design

## Decision

**Sub-axis (b) bucket-tie-break, FIFO variant** — change FM's
`fm_bucket_pop_max` to pop from the *tail* of the cursor bucket's
linked list (the *first-inserted* vertex among the equal-gain tie
group) instead of the head (most-recently inserted).  Gated via
`SPARSE_FM_FINEST_STRATEGY=fifo` env var; default `baseline`
preserves Sprint 23's existing LIFO-on-insertion-order pop bit-
identically.  Day 7 implements the FIFO `pop_max` variant by
adding `tails[]` to `fm_bucket_array_t` + a `pop_max_tail` helper;
Day 8 sweeps + decides whether to flip default.

Sub-axes (a) annealing acceptance and (c) thick-restart-style
global rollback are **rejected for Sprint 26** based on Day 4's
per-recursion-depth profile.  Documented below for traceability;
Sprint 27+ can revisit.

## Why bucket-tie-break (LIFO) — Day 4 profile-driven

Day 4's per-recursion-level profile
(`per_recursion_profile_day4.md`) showed cost concentrating at
depths 6-9 (88 % of partition cost on 169 small-subgraph calls).
Each of those calls invokes the multilevel pipeline + FINEST FM at
the leaf level.  Three sub-axes evaluated against this concentration:

| sub-axis | per-call wall cost | leverage at depths 6-9 | risk |
|---|---|---|---|
| (a) annealing | +20-50 % (more passes) | broad — every pass benefits | destabilises convergence on already-tight cuts |
| **(b) bucket-tie-break (LIFO)** | **0 % — same op count** | **high — 169 indep. invocations × different exploration** | **low — single new pop-from-tail variant** |
| (c) thick-restart | +200-300 % (re-run from anchor) | concentrated targeting | breaches 1.5× wall-check ceiling at 2× wall expansion |

Sub-axis (b) is the only one with **zero runtime cost expansion**.
The wall-check 1.5× ceiling is preserved; Day 7-8 can experiment
freely without risking the gate.

The Sprint 25 Day 5 saturation finding ("passes ≥ 5 saturate at
0.952× on Pres_Poisson") is exactly the symptom LIFO is best
suited to break: each successive FIFO pass picks the same
tie-break choice, converging deterministically to the same local
optimum regardless of how many passes run.  LIFO gives the FM
cascade a *different exploration direction per call* — at depths
6-9 with 169 independent FM invocations, this is 169 different
tie-break orderings cascading through 3-pass FM each.

## Algorithm contract

### Current `fm_bucket_pop_max` (Sprint 23 Day 9-10 — LIFO on insertion order)

```c
void fm_bucket_insert(fm_bucket_array_t *arr, idx_t vertex, idx_t gain) {
    /* Insert at HEAD: new vertex becomes the new head;
     * old head hangs off `next[vertex]`. */
    ...
    arr->heads[bucket] = vertex;
    ...
}

sparse_err_t fm_bucket_pop_max(fm_bucket_array_t *arr, ...) {
    idx_t v = arr->heads[bucket];     /* HEAD = most-recently inserted */
    fm_bucket_remove(arr, v, g);
    ...
}
```

Insert-at-head + pop-from-head means within an equal-gain tie
group, the **most-recently inserted-or-gain-updated** vertex
wins.  This is **LIFO with respect to insertion order** —
contradicting the natural English reading of "FIFO bucket queue".

### Proposed FIFO variant

```c
sparse_err_t fm_bucket_pop_max_tail(fm_bucket_array_t *arr, ...) {
    idx_t v = arr->tails[bucket];    /* TAIL = first-inserted */
    fm_bucket_remove(arr, v, g);
    ...
}
```

Insert-at-head + pop-from-tail means within an equal-gain tie
group, the **first-inserted-or-gain-updated** vertex wins (FIFO
semantics).

### Bucket-array storage extension (Day 7 work)

Add a `tails[]` array parallel to `heads[]` in `fm_bucket_array_t`:

```c
typedef struct {
    idx_t *heads;        /* unchanged: per-bucket newest insertion */
    idx_t *tails;        /* NEW: per-bucket oldest insertion; -1 when empty */
    idx_t *next;         /* unchanged */
    idx_t *prev;         /* unchanged */
    idx_t *counts;       /* unchanged */
    /* ... */
} fm_bucket_array_t;
```

`fm_bucket_insert` updates: when inserting into an empty bucket,
`tails[bucket] = vertex`; when inserting into a non-empty bucket,
`tails[bucket]` is unchanged (the existing tail vertex stays at
the tail end of the chain).

`fm_bucket_remove` updates: when removing the tail (`next[v] ==
EMPTY`), set `tails[bucket] = prev[v]`.  O(1) per remove.

### Day 6 / Day 7 split

- **Day 6** (this commit): parser + dispatch *skeleton*.  Reads
  `SPARSE_FM_FINEST_STRATEGY={baseline,fifo,annealing,thick_restart}`;
  default `baseline`; all on-values fall through to baseline
  behavior today (no semantic change from current master).
- **Day 7** (next): implement `fifo` semantics.  Add `tails[]` to
  `fm_bucket_array_t`; thread an `enum fm_strategy` parameter
  through `graph_refine_fm` (or use a thread-local override
  matching Sprint 26 Day 3's `force_hem_override` pattern); when
  strategy = fifo, use `pop_max_tail` instead of `pop_max`.
- **Day 8**: cross-corpus sweep + flip-rule application.  If FIFO
  closes Pres_Poisson ≤ 0.85× without smaller-fixture regression
  past 5pp, flip default to `fifo`.

## Rejected alternatives

### Sub-axis (a) annealing acceptance

Reject reason: per-call wall expands by 20-50 % (more passes
needed), which compounds across 169 partition calls at depths 6-9.
At 88 % cost concentration, even a +20 % per-call expansion would
push total wall to ~14.6 s (vs Day-5's 12.2 s post-flip baseline) —
acceptable, but the fill-quality benefit is uncertain (annealing
with too-aggressive temperature destabilises Sprint 23's
rollback-on-regress contract).  Sprint 27+ can revisit if (b)
falls short.

### Sub-axis (c) thick-restart-style global rollback

Reject reason: 2-3× per-call wall expansion would push Day-5's
12.2 s post-flip Pres_Poisson ND wall to 24-36 s, which exceeds the
1.5× wall_check_baseline.txt ceiling of 70 583 ms only if the
baseline drops too — but more importantly, breaches the
implicit "Day 5 wall improvement should not be eaten by Day 7
fill-quality work" contract.  Sprint 27+ as last resort.

## Sweep dimensions (Day 7-8)

Day 7 implements `fifo`; Day 8 sweeps:

1. **Pres_Poisson nnz_L under `fifo` (alone)**: target ≤ 0.85×
   literal vs Sprint 25 Day 9's setting-13 best of 0.9218×.
   Hypothesis: tail-pop breaks the FIFO-equivalent saturation;
   measured outcome could be anywhere in [0.92×, 0.85×].
2. **Pres_Poisson nnz_L under `fifo` + Sprint 25 setting 13
   (HCC + ratio=200)**: composes with the existing best opt-in.
   Target ≤ 0.85× literal.
3. **Smaller-fixture corpus safety**: nos4, bcsstk04, Kuu,
   bcsstk14, s3rmt3m3 must not regress past 5pp under `fifo`.
4. **Determinism**: same input → same output across two `fifo`
   runs.  Pin via a determinism unit test on a synthetic fixture.
5. **Wall time**: must stay within Day-5's 12.2 s Pres_Poisson ND
   floor (or only modestly above; the 1.5× ceiling is 70 583 ms
   so there's headroom).

If `fifo` lands the 0.85× Pres_Poisson target with acceptable
corpus + wall profile, **Day 8 flips the default to `fifo`**.
Otherwise ships as advisory + Sprint 27+ inheritance.

## Day 6 deliverables

- `SPARSE_FM_FINEST_STRATEGY={baseline,fifo,annealing,thick_restart}`
  env-var parser stub in `src/sparse_graph.c::graph_uncoarsen`.
  Default `baseline`; out-of-range / non-numeric / unrecognized →
  `baseline`.  Day 6 commits the parser + a *no-op dispatch* (all
  values fall through to baseline behavior, since the `fifo`
  semantics aren't implemented yet).
- `tests/test_graph.c::test_finest_fm_strategy_fifo_smoke`
  stub-and-pin: under `SPARSE_FM_FINEST_STRATEGY=fifo` on a known
  fixture, prints "fifo not yet implemented" (Day 6) or compares
  the cut against baseline (Day 7).  Today the test prints +
  passes; Day 7 tightens.

## References

- `docs/planning/EPIC_2/SPRINT_26/PLAN.md` Day 6 + Day 7 + Day 8
- `docs/planning/EPIC_2/SPRINT_26/per_recursion_profile_day4.md` —
  the per-depth profile that drove this sub-axis selection
- `docs/planning/EPIC_2/SPRINT_25/RETROSPECTIVE.md` "Sprint 26
  inputs" #1 — the three sub-axis candidates this design
  down-selects from
- `docs/planning/EPIC_2/SPRINT_25/intermediate_fm_decision.md` —
  the saturation-at-3-passes finding the bucket-tie-break
  hypothesis is built on
- `src/sparse_graph_fm_buckets.h` — Day 7's bucket-array
  extension target (add `tails[]`)
- `src/sparse_graph.c::fm_bucket_pop_max` (line 1623) — current
  FIFO pop (head); Day 7 unchanged but `fm_bucket_insert` flips
  to tail-insert under `tail_insert` flag
- `src/sparse_graph.c::graph_uncoarsen` (~line 1937) — finest-FM
  dispatch site; Day 6 adds the `SPARSE_FM_FINEST_STRATEGY` parser
  here
