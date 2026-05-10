# Root-Level Spectral Bisection — Design (Sprint 27 Day 7)

## Background

Sprint 25 Day 6-8 implemented spectral bisection at the **coarsest** level of the multilevel pipeline (`SPARSE_ND_COARSEST_BISECTION=spectral`).  The implementation uses Sprint 20-21's `sparse_eigs_sym` Lanczos eigensolver to compute the smallest two eigenpairs of the graph Laplacian L = D - A; extracts the Fiedler vector v_1 (eigenvector of λ_1, the second-smallest eigenvalue); bisects at the median of v_1; falls back to GGGP on Lanczos failure or 60/40 balance violation.

Sprint 27 PLAN.md item 5 extends this to the **root** level: skip the multilevel coarsening pipeline entirely, run Lanczos directly on the full graph Laplacian.  The hypothesis is that the multilevel pipeline's coarsening + bisection + uncoarsening + FM cascade introduces approximation errors that compound as the graph shrinks; running Lanczos directly on the root graph captures the true Fiedler structure without those losses.

This document captures Day 7's design + Lanczos cost characterisation + Days 8-9 implementation plan.

## Reproducer (post-Days-8-9)

```
SPARSE_ND_ROOT_BISECT=spectral build/bench_reorder --only Pres_Poisson --skip-factor
SPARSE_ND_ROOT_BISECT=spectral SPARSE_ND_ROOT_BISECT_MAX_N=20000 ...
```

Day 7 ships only the env-var skeleton; Day 8-9 wires the actual spectral path.

## Design Decisions

### (i) Lanczos Cost Characterisation

Sprint 21 Day 5's scaling measurement:
- `sparse_eigs_sym` shift-invert at σ ≈ 0+ε, k=2, tol=1e-8: ~O(n^1.5) on Laplacians of regular meshes (FE matrices ~O(n^1.3)).

Day-7 estimate at corpus sizes:
- nos4 (n=100): ~0.1 ms — negligible.
- bcsstk04 (n=132): ~0.2 ms — negligible.
- Kuu (n=7102): ~3 s.
- bcsstk14 (n=1806): ~0.5 s.
- s3rmt3m3 (n=5357): ~2 s.
- **Pres_Poisson (n=14822): ~5-10 s** — the headline-fixture cost.

**Comparison with multilevel pipeline cost** (Sprint 27 Day 3 default):
- Pres_Poisson ND wall = 7.1 s (Sprint 27 Day 3 measurement).
- Lanczos at root level = 5-10 s (estimated).
- After spectral bisection at root, the two halves are still partitioned via the multilevel pipeline (just on n/2 graphs).  Each half's multilevel partition is ~7.1 s × (n/2 / n) × (multilevel cost scales O(n^1.05) per Sprint 23 Day 12 measurement) ≈ 3-3.5 s per half.

**Total root-spectral wall on Pres_Poisson**: ~7-10 s Lanczos + 2 × ~3-3.5 s multilevel = ~13-17 s.  This is 2× the current Sprint 27 Day 3 default (7.1 s) but still well within the 70.5 s `wall-check` 1.5× ceiling.  Affordable.

### (ii) Fallback Threshold

Two failure modes:
- **Lanczos failure**: `sparse_eigs_sym` returns non-OK or `n_converged < 2`.  Fall through to multilevel pipeline (existing path).
- **Size threshold**: at n > 50000, Lanczos cost (~30 s+) likely exceeds the multilevel pipeline's cost.  Fall through to multilevel without invoking Lanczos.

`SPARSE_ND_ROOT_BISECT_MAX_N` env var (default **50000**):
- Wide enough to include Pres_Poisson (n=14 822).
- Small enough to skip production-scale fixtures where Lanczos cost would dominate.
- Tunable for users who know their workload size + Lanczos performance characteristics.

Range: [1, 100 000 000]; out-of-range / non-numeric / missing → default.

## Algorithm (Days 8-9 Implementation)

```
nd_recurse(G, vertex_id_map, perm, next_pos, depth):
  if (depth == 0):
    strategy = parse_nd_root_bisect_strategy()
    max_n = parse_nd_root_bisect_max_n()
    if (strategy == ND_ROOT_BISECT_SPECTRAL && G.n <= max_n):
      part_2way = malloc(G.n)
      rc = graph_bisect_coarsest_spectral(G, part_2way)
        # Sprint 25 Day 7 — reuse without modification.
        # Returns 2-way partition (0/1) on the input graph;
        # falls back to GGGP internally on Lanczos failure.
      if (rc == SPARSE_OK):
        # Convert 2-way to 3-way separator via existing
        # graph_edge_separator_to_vertex_separator.
        rc = graph_edge_separator_to_vertex_separator(G, part_2way)
        # part_2way[] is now {0, 1, 2}.
        # Continue with the existing nd_recurse split-and-recurse
        # logic (write separator vertices last; recurse on each side).
        # Skip the multilevel sparse_graph_partition call below.
        ...

  # Existing multilevel path (default; bit-identical to Sprint 27 Day 6).
  rc = sparse_graph_partition(G, part, &sep_count)
  ...
```

The cleanest implementation:
1. Build the Laplacian from G via `graph_build_laplacian` (already exposed; Sprint 25 Day 7).
2. Call `graph_bisect_coarsest_spectral(G, part)` (Sprint 25 Day 7; static helper — needs to be promoted to non-static or re-implemented in `sparse_reorder_nd.c` via re-exposing).  Actually the helper is already deterministic + falls back internally.
3. Apply `graph_edge_separator_to_vertex_separator(G, part)` to convert 2-way → 3-way.
4. Skip the existing `sparse_graph_partition` call; jump directly to the split-and-recurse phase below.

**Key implementation note**: `graph_bisect_coarsest_spectral` is currently declared `static` in `src/sparse_graph.c`.  Days 8-9 will need to either:
- (A) Promote it to a header-declared internal function (add to `sparse_graph_internal.h`).
- (B) Re-implement the spectral path in `sparse_reorder_nd.c` (duplicates code).
- (C) Add a new public-internal entry point `sparse_graph_bisect_spectral` that wraps the static helper.

**Choice for Days 8-9: Option (A)** — minimal duplication; keeps the spectral logic in `sparse_graph.c` where Laplacian-build + Lanczos invocation lives.  Add the function declaration to `sparse_graph_internal.h` so `sparse_reorder_nd.c` can call it.

## Per-Fixture Hypothesised Outcomes

| Fixture | n | Current default | Root-spectral hypothesis |
|---|---:|---:|---|
| nos4 | 100 | 0.756× | Bit-stable (n < threshold but worth running) |
| bcsstk04 | 132 | 1.184× | Bit-stable / minor delta |
| Kuu | 7 102 | 1.882× | Possibly mild improvement (irregular structure; Fiedler may find better cut than HCC+multilevel) |
| bcsstk14 | 1 806 | 1.124× | Mild improvement (the bcsstk-class advisory) |
| s3rmt3m3 | 5 357 | 1.024× | Mild improvement |
| **Pres_Poisson** | **14 822** | **0.923×** | **Headline target: → 0.85× ?** |

Pres_Poisson is the fixture the spectral path most plausibly closes to 0.85×.  Pres_Poisson is a 2D-Poisson FE mesh — exactly the structure the Fiedler vector captures cleanly (smooth eigenfunctions on regular grids).  The multilevel pipeline's coarsening loses some of this geometric structure; running Lanczos at the root level preserves it.

If item 5 lands Pres_Poisson ≤ 0.85×, the headline target closes after 4 sprints of attempts.

## Day 7 Skeleton Implementation

### `src/sparse_reorder_nd.c`

Added `nd_root_bisect_strategy_t` enum + `parse_nd_root_bisect_strategy()` + `parse_nd_root_bisect_max_n()`.  Added a depth-0 branch in `nd_recurse` that reads the env vars but currently no-ops (TODO marker for Days 8-9).

### `tests/test_reorder_nd.c`

Stubbed `test_nd_root_spectral_pres_poisson_smoke` (RUN_TEST commented out for Day 7; pins the Days 8-9 contract that `SPARSE_ND_ROOT_BISECT=spectral` produces a different ND output than `multilevel` on Pres_Poisson).

### Bit-Identicality Under Default-Off

`SPARSE_ND_ROOT_BISECT` unset / `multilevel` → existing path → same output as Sprint 27 Day 6 default.  Verified by `make test` passing without changes to fixture-output assertions.

## Days 8-9 Plan

### Day 8

1. Promote `graph_bisect_coarsest_spectral` to internal-API: add prototype to `src/sparse_graph_internal.h`; remove `static` qualifier in `src/sparse_graph.c`.
2. Wire the root-level dispatch in `nd_recurse` at `depth == 0`: call `graph_bisect_coarsest_spectral(G, part_2way)`; convert via `graph_edge_separator_to_vertex_separator`; integrate with the existing split-and-recurse phase.
3. Run the existing 39 partition + reorder tests under `SPARSE_ND_ROOT_BISECT=multilevel` (default) — should all pass bit-identically to Day 7.
4. Run the same tests under `SPARSE_ND_ROOT_BISECT=spectral` — most pass; some determinism contracts may need re-verification (Lanczos is deterministic given the same Laplacian + tolerance).
5. Capture interim Pres_Poisson + corpus measurements.
6. Light up `test_nd_root_spectral_pres_poisson_smoke` (uncomment RUN_TEST).

### Day 9

1. Full corpus sweep under `SPARSE_ND_ROOT_BISECT=spectral`.
2. Apply flip rule for `SPARSE_ND_ROOT_BISECT` default.
3. If headline closes (Pres_Poisson ≤ 0.85×): tighten `test_nd_pres_poisson_fill_with_leaf_amd` bound.
4. Document in `root_spectral_decision.md`.
5. Combine with Sprint 27 Day 6 annealing-best — does `annealing + spectral` close any further?  Sprint 27 PLAN.md Day 9 task 3 specifies this exploration.

## Files Generated Day 7

- `docs/planning/EPIC_2/SPRINT_27/root_spectral_design.md` — this document
- `src/sparse_reorder_nd.c` — env-var skeleton (parsers + no-op dispatch stub)
- `tests/test_reorder_nd.c` — `test_nd_root_spectral_pres_poisson_smoke` stub (RUN_TEST commented out)

## Headline Status After Day 7 Item-5 Design Kickoff

- Annealing FM (Item 4) ships as advisory; default unchanged.  Item 5 carries the remaining 7.3pp 0.85× gap.
- Root-spectral skeleton lands; default behaviour bit-identical.  Days 8-9 implement.
- 7.3pp gap remains to literal 0.85× target.
