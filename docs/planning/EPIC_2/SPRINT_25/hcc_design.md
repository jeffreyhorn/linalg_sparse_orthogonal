# Sprint 25 Day 1 — Heavy Connectivity Coarsening (HCC) Design

## Context

Sprint 24 closed the qg-AMD wall-time regression but missed the
Pres_Poisson ND/AMD ≤ 0.85× literal target (achieved 0.952×
default; 0.942× best opt-in under
`SPARSE_ND_COARSEN_FLOOR_RATIO=200`).  The Sprint 24 retrospective
"Sprint 25 inputs" #1 routes three algorithmic candidates to
Sprint 25; Heavy Connectivity Coarsening is the first.

The current coarsener in `src/sparse_graph.c::graph_coarsen_heavy_edge_matching`
(Sprint 22 Day 2) implements **heavy-edge matching (HEM)**: walk
vertices in shuffled order, match each unmatched vertex to its
heaviest unmatched neighbour by edge weight alone:

```
score_HEM(u, v) = edge_weight(u, v)
```

HEM works well on banded structural-mechanics matrices but
under-explores the cut-quality-vs-coarsening-rate tradeoff on 2D
PDE meshes like Pres_Poisson, where many neighbours have equal
edge weight (1 for unweighted finite-difference stencils) and
HEM falls back to first-encountered tie-break.

## Karypis-Kumar 1998 §5 — Heavy Connectivity Coarsening

Karypis & Kumar (1998), "Multilevel k-way Partitioning Scheme for
Irregular Graphs", J. Parallel Distrib. Comput. 48:96-129.

§5 ("Heavy Connectivity Matching", page ~110-112 in the journal
version) describes a family of matching schemes that incorporate
vertex-degree information into the edge-selection score.  The
canonical scoring function for **HCC** as adopted by METIS for ND
is:

```
score_HCC(u, v) = edge_weight(u, v) * min(degree(u), degree(v))
```

i.e. weight the edge by its lower-degree endpoint.  The
intuition: matching two low-degree vertices contracts the graph
faster (fewer surviving edges in the coarse adjacency), which is
the opposite of what HEM optimizes for.  HEM picks the
heaviest-weight neighbour; HCC picks the heaviest-weight
*low-connectivity* neighbour.  On regular meshes (Pres_Poisson),
this preserves the geometric structure of the cut better through
the multilevel hierarchy because low-degree vertices are typically
boundary or near-boundary vertices in a 2D / 3D grid; matching
them first keeps the interior structure intact for the deeper
coarsening steps.

### Variant: HEM vs. HCC vs. SHEM

KK1998 §5 defines several variants:

- **HEM** (Heavy-Edge Matching): `score = edge_weight`.  Sprint 22's current behavior.
- **HCC** (Heavy-Connectivity Coarsening): `score = edge_weight * min(deg(u), deg(v))` — what Sprint 25 implements.
- **SHEM** (Sorted Heavy-Edge Matching): HEM with vertices visited in degree-ascending order rather than shuffled.

Sprint 25 implements HCC specifically because:
1. It's the variant METIS uses for its ND defaults (per KK1998 §5 + the METIS source).
2. Its scoring function is simple — a single multiplier added to the existing HEM scoring.
3. The `min()` formulation handles asymmetric degrees naturally (matches the symmetric-graph contract of `sparse_graph_t`).

SHEM is not implemented in Sprint 25; if the HCC sweep on Day 2-3 shows promise without closing the gap, SHEM becomes a Sprint 26 candidate.

### Tie-break order

Equal-score edges (common on unit-weighted regular meshes) need a
deterministic tie-break to preserve the
`test_partition_determinism_*` contract.  HCC tie-break (in
Sprint 25's implementation):

1. Higher score wins.
2. On equal score: lower-id neighbour wins (preserves Sprint 22's
   shuffled-walk-order tie-break property for fair comparisons —
   the shuffle still controls outer-vertex visit order; the
   tie-break only fires within a single vertex's neighbour scan).

This deviates from Sprint 22's HEM tie-break (first-encountered)
in a controlled way:  HEM scans `fine->adjncy[v]` in storage
order; the first neighbour with `w > best_wt` wins.  HCC switches
the comparison to `w > best_score` AND adds the tie-break "or (w == best_score AND u < best_nbr)".  This makes HCC's match selection
deterministic regardless of adjacency storage order — necessary
because HCC's score is more often tied than HEM's weight.

### Visit order over outer vertices

Sprint 22's `graph_coarsen_heavy_edge_matching` uses a
splitmix64-seeded Fisher-Yates shuffle (`fisher_yates_shuffle`,
`src/sparse_graph.c:330ish`).  Sprint 25's HCC inherits the same
shuffle to preserve determinism: the shuffle is purely a function
of `seed`, which `sparse_graph_hierarchy_build` already passes
through unchanged.

Two visit-order alternatives KK1998 mentions:
- **Degree-ascending sort** (SHEM-style): match low-degree vertices
  first.  Considered for Sprint 26 if HCC alone doesn't close the
  gap.
- **Random** (HEM-current): preserves entropy across runs with the
  same seed perturbation per level.

Sprint 25 stays with HEM's random shuffle to isolate the
score-function change from the visit-order change.

### Matched-vertex collapse rule

Identical to Sprint 22's HEM:
- For each matched pair (v, best_nbr), assign both `cmap[v] = cmap[best_nbr] = n_coarse`.
- Coarse vertex's weight = sum of matched fine vertices' weights.
- Coarse vertex's adjacency = union of matched fine vertices' adjacencies (with duplicate filtering).

The collapse rule is a separate concern from the matching score;
HCC only changes which pairs get matched, not what happens after.

## `SPARSE_ND_COARSENING` env-var gate

### Contract

```
SPARSE_ND_COARSENING={hcc,heavy_edge}
```

- `heavy_edge` (default): Sprint 22 behavior, bit-identical.
- `hcc`: Sprint 25's HCC implementation.
- Unset or unrecognized value: falls back to `heavy_edge`.

### Validation pattern

Mirrors Sprint 24 Day 5's `SPARSE_ND_COARSEN_FLOOR_RATIO` parser
(strtol + range-check + fallback), adapted for string values:

```c
typedef enum {
    COARSENING_HEAVY_EDGE = 0,  /* Sprint 22 default */
    COARSENING_HCC        = 1,  /* Sprint 25 Day 1-3 */
} coarsening_strategy_t;

static coarsening_strategy_t parse_coarsening_strategy(void) {
    const char *env = getenv("SPARSE_ND_COARSENING");
    if (env && strcmp(env, "hcc") == 0)
        return COARSENING_HCC;
    /* Default + unrecognized + "heavy_edge" all fall through. */
    return COARSENING_HEAVY_EDGE;
}
```

### Entry-point identification

Day 1 confirmed: the matching loop is `graph_coarsen_heavy_edge_matching`
(`src/sparse_graph.c:390-476`).  The hierarchy builder
`sparse_graph_hierarchy_build` (line 664+) calls it at line 748:

```c
graph_coarsen_heavy_edge_matching(prev, seed + (uint32_t)level, &coarse, cmap);
```

Day 2's HCC implementation will:
1. Add a sibling function `graph_coarsen_hcc(fine, seed, coarse_out, cmap_out)` with the HCC score.
2. Read `SPARSE_ND_COARSENING` once at the top of `sparse_graph_hierarchy_build` (cached as a `coarsening_strategy_t` local).
3. Branch on the cached strategy at line 748: `COARSENING_HEAVY_EDGE` → existing call (no behavior change); `COARSENING_HCC` → new HCC call.

The dispatch happens in `sparse_graph_hierarchy_build` rather than
inside `graph_coarsen_heavy_edge_matching` so the score-function
choice is per-hierarchy rather than per-call (consistent across all
levels of the same hierarchy build).

### Why default `heavy_edge` for Day 1

Day 1 ships only the design doc + skeleton + test stubs.  The
default stays `heavy_edge` because:
1. Day 2 lands the HCC matching function but defers default-flip
   to Day 10 after Day 9's cross-corpus sweep produces the
   per-fixture deltas.
2. The literal flip rule (PROJECT_PLAN.md Sprint 25 item 1):
   "flip default if HCC produces a clear corpus-wide win on
   Pres_Poisson + small fixtures".

Until that rule is empirically tested, `heavy_edge` is the
conservative default.

## Modified-vs-replaced delta from Sprint 22

What HCC keeps from Sprint 22's `graph_coarsen_heavy_edge_matching`:

- The function signature: `(const sparse_graph_t *fine, uint32_t seed, sparse_graph_t *coarse_out, idx_t *cmap_out)`.
- The Fisher-Yates shuffle on outer vertices for visit order.
- The matched-vertex collapse rule (cmap assignment, weight summation, adjacency union).
- The bail-out paths (empty graph, n_coarse <= 0 sentinel).
- The clang-analyzer guards on `n_coarse` (the `if (n_coarse <= 0)` early return after the matching loop).

What HCC modifies:

- **Score function** (the one-line change): `w` → `(int64_t)w * (int64_t)min(deg(u), deg(v))`.  Wider arithmetic to match Sprint 24 Day 11's `int64_t` overflow-safety pattern in `graph_edge_separator_to_vertex_separator`.
- **Tie-break** (additional clause): on equal score, lower-id neighbour wins.

What HCC needs to track that HEM doesn't:

- Vertex degrees: precomputed once before the matching loop as `deg[i] = fine->xadj[i+1] - fine->xadj[i]` (O(n) memory + O(n) compute).  No persistent state beyond the function call.

## Day 1 deliverables checklist

- [x] Read Sprint 24 retro "Sprint 25 inputs" + nd_coarsen_floor_decision.md + nd_sep_strategy_decision.md (Day-1 task 1)
- [x] Document KK1998 §5 algorithm contract + scoring formula + tie-break + visit order (this doc, sections above)
- [x] Identify entry point: `graph_coarsen_heavy_edge_matching` (line 390); dispatch in `sparse_graph_hierarchy_build` (line ~748)
- [x] Sketch `SPARSE_ND_COARSENING` env-var gate (parser + enum)
- [x] Cross-reference Sprint 22 HEM for the modified-vs-replaced delta

Day 2 picks up: implement `graph_coarsen_hcc` + wire dispatch + run corpus parity check under `SPARSE_ND_COARSENING=heavy_edge` (default; bit-identical to Sprint 22).

## References

- Karypis, G. and Kumar, V. (1998).  "Multilevel k-way Partitioning Scheme for Irregular Graphs."  J. Parallel Distrib. Comput. 48(1):96-129.  §5 ("Heavy Connectivity Matching"), pages ~110-112.
- METIS source code (METIS v5.1, `libmetis/coarsen.c::Match_HEM`, `Match_SHEM`) — informs the tie-break + visit-order choices for the C-level implementation.
- Sprint 22 `src/sparse_graph.c::graph_coarsen_heavy_edge_matching` (lines 390-476) — the function HCC mirrors with the score-function swap.
- Sprint 24 `docs/planning/EPIC_2/SPRINT_24/nd_coarsen_floor_decision.md` — documents why deeper coarsening (smaller floor) helps Pres_Poisson; HCC is the orthogonal axis (better matching quality at the same floor).
- Sprint 24 `docs/planning/EPIC_2/SPRINT_24/RETROSPECTIVE.md` "Sprint 25 inputs" #1 — routes HCC as the first algorithmic candidate.
