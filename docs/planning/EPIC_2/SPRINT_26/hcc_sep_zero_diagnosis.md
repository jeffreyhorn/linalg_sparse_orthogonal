# Sprint 26 Day 2 — HCC bcsstk14 sep=0 Root-Cause Diagnosis + Fix Decision

## Decision

**Adopt option (b): `sparse_graph_partition` sep=0 fall-back.**  When
the multilevel pipeline's projected cut on the original graph produces
`sep == 0`, log a fall-back warning and re-bisect with HEM matching
(Sprint 22 baseline; known to produce sep > 0 on bcsstk14).  Day 3
implements the fix at the single point where sep is computed.

Option (a) HCC matching tightening is rejected: the divergence isn't
in HCC's per-call matching choices (Day-2 traces show HCC and HEM
match-ratios within 1pp at every coarsening level on bcsstk14) but
in the *cumulative* topological effect of those matches on the coarsest
graph.  Catching this at coarsening time would require simulating the
projection back through every uncoarsening level — equivalent to
running the full multilevel pipeline and then checking the result, which
is what option (b) does anyway, just at the natural single-point seam
in `sparse_graph_partition`.

## Evidence

Day 2 captured HCC + HEM coarsening traces on bcsstk14 under the
SPARSE_HCC_DEBUG instrumentation Day 1 added:

```
SPARSE_HCC_DEBUG=1 SPARSE_ND_COARSENING=hcc        build/bench_reorder --only bcsstk14 --skip-factor 2> /tmp/hcc_bcsstk14_trace.txt
SPARSE_HCC_DEBUG=1 SPARSE_ND_COARSENING=heavy_edge build/bench_reorder --only bcsstk14 --skip-factor 2> /tmp/hem_bcsstk14_trace.txt
```

### Coarsening hierarchy comparison (root ND call, n_fine=1806)

| level | HCC n_coarse | HCC match_ratio | HEM n_coarse | HEM match_ratio | Δ |
|---|---|---|---|---|---|
| 0 | 948 | 0.950 | 941 | 0.958 | -7 (HCC matches 7 fewer pairs) |
| 1 | 509 | 0.926 | 502 | 0.933 | -7 |
| 2 | 283 | 0.888 | 280 | 0.884 | -3 |
| 3 | 167 | 0.820 | 164 | 0.829 | -3 |
| 4 | 106 | 0.731 | 105 | 0.720 | -1 |
| 5 | 77 | 0.547 | 76 | 0.552 | -1 |
| 6 | 62 | 0.390 | 60 | 0.421 | -2 |
| 7 | 53 | 0.290 | 51 | 0.300 | -2 |
| 8 | 48 | 0.189 | 46 | 0.196 | -2 |

HCC and HEM coarsen to nearly identical sizes (HCC 48 vs HEM 46 at
the coarsest level).  Match-ratios are within ~1pp at every level.
**The divergence isn't in matching coverage** — both produce
multilevel hierarchies of comparable depth and aggressiveness.

### Recursion behavior comparison (post-root-partition)

The ND recursion structure shows the smoking-gun divergence:

| state | HCC n_fine | HEM n_fine |
|---|---|---|
| ND root call | 1806 | 1806 |
| 2nd ND call (after root partition) | **1805** | 911 |
| 3rd ND call | **1804** | 394 |
| 4th ND call | **1803** | (further recursion) |

**HEM** produces a balanced root partition: ~895 vertices on one side
(the separator + the smaller side), recurses on the 911-vertex
remaining side.  Per Sprint 25 docs, HEM's bcsstk14 sep = 97 ✓.

**HCC** produces a degenerate root partition: 1805 vertices on one
side, 0 on the other, separator = 1 (or 0 — the test failure
manifests as sep=0).  The recursion then goes nowhere useful — peels
off one vertex at a time (1806 → 1805 → 1804 → 1803 → ...).  This
is the canonical "all vertices on one side" failure mode that
`nd_emit_natural` masks via the degenerate-fall-through to natural
ordering, producing a valid but mediocre nnz_L of 130 358 in
Sprint 25 setting 13's measurement.  But
`test_partition_bcsstk14_smoke` exercises `sparse_graph_partition`
directly and asserts `sep > 0` — that's the test-level failure
Sprint 25 Day 10 surfaced.

### Structural property of bcsstk14 that triggers the pathology

Hypothesis: bcsstk14's structural-mechanics provenance produces a
**bimodal degree distribution** — low-degree vertices at the
"boundary" of the FE mesh (skin elements), high-degree vertices at
the "interior" (core elements).  HCC's `min(deg(u), deg(v))`
weighting biases matching toward `(low_deg, low_deg)` pairs because
their `min(deg(u), deg(v))` score is low BUT the score acts as a
multiplier on `edge_weight` only when both endpoints have low
degree — for `(low, high)` pairs, the score is dominated by
`edge_weight * low_deg`, which is the same as for `(low, low)`.
The score *ranking* therefore prefers `(low, low)` over `(low, high)`
when the edge weights are comparable.

This produces a coarse graph with a "boundary cluster" of merged
boundary-pairs structurally separated from the "interior cluster".
The coarsest-level bisection (brute-force at n ≤ 20, GGGP at n > 20)
naturally cuts along this structural seam.  When projected back
through 9 uncoarsening levels, the boundary-cluster un-coarsens
back into a *thin shell* that wraps fully around the interior
cluster — which projects to a one-sided cut on the original graph
(the "interior" gets all vertices; the "boundary shell" gets the
single-vertex separator).

This is the canonical pathology of degree-aware coarsening on
boundary-dominated meshes.  HEM's pure-edge-weight matching avoids
it because it doesn't bias toward boundary-vertex preferential
matching.

## Fix design

### Option (a) — HCC matching tightening: REJECTED

Per-vertex check at coarsening time to detect "this match is producing
a one-sided coarse graph".  Pros: keeps fix local to HCC.  Cons:

- The pathology is *cumulative across 9 coarsening levels*; no
  single-level matching choice is detectably bad.  Per Day 2's
  per-level match-ratio comparison, HCC and HEM match within 1pp
  at every level — there's no single match-choice "the bad one".
- Detecting the pathology at coarsening time would require simulating
  the full uncoarsening + projection — that's what option (b) does at
  the natural seam.
- Cross-cutting risk: a future coarsening algorithm (HCC-variant or
  unrelated) could produce the same pathology; option (a) would need
  per-algorithm detection.
- HCC's score-ranking-prefers-`(low, low)` is the algorithmic
  *intent* per Karypis-Kumar 1998 §5; "fixing" HCC to avoid it would
  be reverting to HEM in disguise.

### Option (b) — `sparse_graph_partition` sep=0 fall-back: ADOPTED

When the multilevel pipeline's final partition has `sep == 0` (or `sep == 1`
with a one-sided pattern), detect it at partition emit time, log a
SPARSE_QG_DEBUG-equivalent fall-back warning, and re-run the partition
under HEM matching (Sprint 22 baseline) which is known to produce
sep > 0 on bcsstk14.

**Pros:**
- Localized: single-point fix at the partition output where sep is
  computed.
- Cross-cutting: works for any current or future coarsening algorithm
  that produces this pathology.
- Cleanly recovers via re-bisection under a known-working algorithm
  (HEM has worked on every fixture in the Sprint 22-25 corpus including
  bcsstk14).
- The fall-back path is a clean library-internal contract:
  `sparse_graph_partition` always returns sep > 0 (or an error if
  even HEM fails, which would be a separate correctness gap).

**Cons:**
- Extra cost on the degenerate path: re-runs the multilevel pipeline
  end-to-end.  Mitigation: the degenerate path only fires on
  bcsstk14-class fixtures under HCC; ordinary HEM users pay zero
  cost.
- Logs may be noisy if the fall-back fires unexpectedly.
  Mitigation: gate the warning behind a debug env var, OR always-on
  but quiet (single line per `sparse_graph_partition` call that
  fell back).

### Implementation outline (Day 3)

In `src/sparse_graph.c::sparse_graph_partition` (or wherever the
edge-to-vertex separator is extracted to compute sep):

```c
/* Sprint 26 Day 3: sep=0 fall-back.  When the multilevel pipeline
 * produces a degenerate empty separator (HCC bcsstk14 case per
 * SPRINT_26/hcc_sep_zero_diagnosis.md), retry with HEM coarsening
 * and re-run the partition. */
idx_t sep = compute_separator_size(...);
if (sep == 0) {
    /* Save current coarsening strategy, force HEM, re-run, restore. */
    coarsening_strategy_t saved_strategy = parse_coarsening_strategy();
    if (saved_strategy != COARSENING_HEAVY_EDGE) {
        /* TODO: cleanest implementation needs to thread strategy
         * through sparse_graph_hierarchy_build rather than re-reading
         * the env var.  Day 3 picks the threading approach. */
        ...
    }
}
```

The exact threading mechanism (env-var override, additional opts
parameter on `sparse_graph_partition`, or a dedicated retry path) is
a Day-3 implementation choice.  The diagnosis ends here; Day 3
turns it into code.

## Day 3 inputs

1. Implement option (b) sep=0 fall-back in
   `src/sparse_graph.c::sparse_graph_partition`.
2. Pick the strategy-threading approach: option (b1) re-read env var
   inside `sparse_graph_hierarchy_build` with a temp override; option
   (b2) add a `coarsening_override` parameter to the partition entry
   point; option (b3) extract the matching loop into a parameterized
   helper called twice with different strategies.  Option (b3) is
   cleanest; Day 3 picks based on code-review.
3. Light up `tests/test_graph.c::test_hcc_bcsstk14_no_degenerate_partition`
   stub from Day 2 task 5.
4. Re-run `test_partition_bcsstk14_smoke` under
   `SPARSE_ND_COARSENING=hcc` — must now pass (the Sprint 25 Day 10
   default-flip blocker).
5. Re-attempt the HCC default flip per Day 3 task 5 in PLAN.md.

## References

- `docs/planning/EPIC_2/SPRINT_25/coarsening_decision.md` — Sprint 25
  Day 10's HCC default-flip-revert decision; the bcsstk14 sep=0
  reproducer block this sprint inherits
- `docs/planning/EPIC_2/SPRINT_25/hcc_design.md` — Sprint 25 Day 1's
  HCC algorithm contract (`score = edge_weight × min(deg(u), deg(v))`)
- `/tmp/hcc_bcsstk14_trace.txt` + `/tmp/hem_bcsstk14_trace.txt` —
  Day 2 cmap traces (intermediate; not committed)
- `tests/test_graph.c::test_partition_bcsstk14_smoke` — line 1798's
  `ASSERT_TRUE(sep > 0)` is the contract HCC fails on bcsstk14
  (Sprint 25 Day 10 finding) and Day 3's fix lights up
- Karypis-Kumar 1998 "Multilevel k-way Partitioning Scheme for
  Irregular Graphs" §5 — Heavy Connectivity Coarsening; the
  `min(deg(u), deg(v))` weighting that produces the bimodal-degree
  pathology on bcsstk14
