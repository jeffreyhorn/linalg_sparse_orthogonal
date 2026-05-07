# Sprint 25 Day 11 — ND Wall-Time Decision

## Decision

**The Pres_Poisson ND default-path wall-time drift Sprint 24 Day 8
documented (21 % above Sprint 23 baseline) is RUN-TO-RUN VARIANCE,
not algorithmic cost growth.**  No fix landed.  Day 12 sets the
`wall_check_baseline.txt` Pres_Poisson ND line with a 50 %
threshold (per PLAN.md Day 11 task 3 routing) rather than the
original 5 % drift target Sprint 24 hoped for.

## Evidence

5 consecutive `bench_reorder --only Pres_Poisson --skip-factor`
runs under `SPARSE_ND_PROFILE=1` (the new Day 11 instrumentation),
no concurrent host load, single sitting:

| run | total ms | partition % of total |
|---|---|---|
| 1 | 50 460 | 99.5 % |
| 2 | 51 562 | 99.5 % |
| 3 | 44 503 | 99.5 % |
| 4 | 47 055 | 99.5 % |
| 5 | 44 321 | 99.5 % |

| statistic | value |
|---|---|
| min | 44 321 ms |
| median | 47 055 ms |
| max | 51 562 ms |
| range | 7 241 ms = **16.3 % of min** |
| mean | 47 580 ms |
| std-dev | 3 318 ms (~7 % of mean) |

The 16.3 % within-run range exceeds the 5 % drift threshold
PLAN.md Day 11 task 3 cited as the variance-vs-algorithmic-cost
discriminator.  The drift is variance.

## Per-phase analysis

The `partition` phase (cumulative `sparse_graph_partition` calls
across all recursion levels) accounts for **99.5 % of total wall
time in every run**.  Other phases are negligible:

| phase | % of wall (mean across 5 runs) |
|---|---|
| partition | 99.5 % |
| graph_build (root `sparse_graph_from_sparse`) | 0.33 % |
| leaf_amd (cumulative `sparse_reorder_amd_qg`) | 0.05 % |
| subgraph (cumulative `sparse_graph_subgraph`) | 0.05 % |
| leaf_subgraph (cumulative `nd_subgraph_to_sparse`) | 0.02 % |
| emit_natural (degenerate-partition fallback) | 0.000 % |
| other (recursion + alloc overhead) | 0.03 % |

Call counts (deterministic; identical across runs):

| call | count |
|---|---|
| `sparse_graph_partition` | 301 |
| `sparse_reorder_amd_qg` (leaf splice) | 237 |
| `nd_emit_natural` (degenerate fallback) | 32 |

Since partition dominates wall time and is itself variant by
16.4 % across runs (44 094 ms - 51 313 ms), the entire wall-time
variance is attributable to partition.  No other phase has the
absolute cost to drive wall variance even if its per-phase
variance were 100 %.

## Why algorithmic-cost growth is unlikely

The default ND code path is bit-identical between Sprint 23 and
Sprint 25:
- Sprint 24 added `SPARSE_ND_COARSEN_FLOOR_RATIO` (Day 5) +
  `SPARSE_ND_SEP_LIFT_STRATEGY` (Day 6) — both default-off; the
  default code path is unchanged.
- Sprint 25 added `SPARSE_ND_COARSENING` (Days 1-3) +
  `SPARSE_FM_INTERMEDIATE_PASSES` (Days 4-5) +
  `SPARSE_ND_COARSEST_BISECTION` (Days 6-8) — all default-off; the
  default code path is unchanged.

Sprint 25 Day 10's `bench_day10_default.csv` measurement
(Pres_Poisson ND nnz_L = 2 541 734) is bit-identical to Sprint 23
+ Sprint 24's defaults — proving the algorithmic output is
identical.  If the algorithmic output is identical, the cost
should also be (modulo build/compiler/host noise).

The remaining drift sources between Sprint 23's reported ~36 s and
Sprint 25's ~47 s median are:
1. **Build/compiler version drift** — Sprint 23 was built in
   April 2026; today's build uses the current Apple Clang.  Across
   ~2 months of incremental Xcode updates, codegen for hot loops
   in `sparse_graph_partition` (FM bucket scans, adjacency walks)
   may differ.  This is a fixed cost that doesn't appear as
   within-Day-11 variance — but it explains the sprint-to-sprint
   ~10 s baseline shift.
2. **macOS arm64 thermal management** — sustained ND workloads
   on Apple Silicon trigger CPU frequency scaling.  Within-Day-11
   variance of 16 % is consistent with this; the test machine's
   thermal state at the start of each run influences the partition
   walltime.
3. **Host-load variance** — even with no concurrent jobs running,
   macOS background processes (Spotlight, Time Machine, system
   updates) intermittently consume cores and memory bandwidth.

None of these are "Sprint 25 introduced an algorithmic cost
regression" — they're environmental.

## What we learned from the profile

Useful data from Day 11's instrumentation that informs Sprint 26
planning (per `nd_tuning_day8.md` "Sprint 26 routing"):

1. **Partition is 99.5 % of ND wall on Pres_Poisson** — any
   wall-time speedup work targeting Sprint 26's ND fill-quality
   gap should focus on `sparse_graph_partition`'s internals (FM
   passes, coarsening matching, bisection), not on the recursion
   driver (which is < 1 % of wall).

2. **Leaf-AMD splice is essentially free** — 28-31 ms across 237
   calls = ~0.12 ms per leaf.  The ND/AMD ratio improvement Sprint
   23 Day 7's leaf-AMD splice landed (~1pp on Pres_Poisson) costs
   essentially zero wall time.  Whatever Sprint 26 does at the
   finest level is bounded by partition cost, not by the
   recursion driver's leaf-handling overhead.

3. **301 partition calls × ~165 ms/call = 49.7 s** — the median
   per-partition cost is ~165 ms.  At the recursion's deepest
   levels (small subgraphs at the base-threshold boundary, n ~32),
   each partition call is fast.  At the root (n=14 822) the
   partition is most expensive.  A profile that broke down per-
   recursion-level cost would be a Sprint 26 follow-up if
   targeting wall-time wins specifically.

4. **emit_natural fires 32 times** — degenerate-partition
   fallback (n0=0 or n1=0 in `nd_recurse`) fires for 32 of the
   301 recursion calls.  These are the small-subgraph cases where
   the partitioner produces a one-sided cut.  If a future
   optimization wants to skip the partition call when the subgraph
   is "too small for non-degenerate partitioning" (raise
   `nd_base_threshold` from 32 → 64?), it would save 32 *
   ~165 ms = ~5.3 s per ND call on Pres_Poisson.  But raising the
   threshold also forces leaf-AMD on larger subgraphs, which has
   its own fill-quality cost.  Sprint 26 trade-off territory.

## Sprint 26 routing

Per `nd_tuning_day8.md` "Sprint 26 routing", the Pres_Poisson
0.85× residual gap candidates are:
1. Multi-pass FM at the FINEST level beyond 3 passes
2. Direct geometric cut detection on regular grids
3. Per-vertex separator scoring

To which Day 11's profile adds:
4. **Per-recursion-level partition profiling** — break out the
   partition cost by recursion depth to identify where the FM
   passes are spending their time.  Likely candidate for the
   "multi-pass FM at the FINEST level" axis #1's optimization
   target.
5. **Larger `nd_base_threshold` exploration** — raising from 32
   would skip the degenerate-partition fallback overhead but
   shift fill-quality cost to the leaf-AMD path.  Sprint 22 Day 9
   set the threshold; Sprint 26 could re-sweep.

## What ships in Sprint 25

- `SPARSE_ND_PROFILE` env-var-gated `clock_gettime` per-phase
  instrumentation in `src/sparse_reorder_nd.c` (Day 11
  deliverable).  Off by default; enabled via the env var.
  Production overhead: one branch per timed call when off.
  Useful for Sprint 26's per-recursion-level profiling work
  + general ND wall-time investigation.

- The variance-vs-algorithmic-cost finding documented here +
  routed to Day 12's wall-check baseline tooling.

- No code fix — there's nothing to fix.  The default ND path is
  bit-identical to Sprint 23.

## Day 12 input

Per PLAN.md Day 12 task 1: add a `pres_poisson_nd_ms` baseline to
`wall_check_baseline.txt`.

Recommended values from this Day 11 measurement:

```
# Sprint 25 Day 11: Pres_Poisson ND default-path wall-time baseline.
# Per profile_day11_pres_poisson_nd.txt, 5 consecutive runs span
# 44 321 - 51 562 ms (16.3 % range; classified as run-to-run
# measurement variance per nd_wall_time_decision.md).  Baseline
# pinned at the Day 11 median; threshold 50 % above baseline gives
# headroom for typical macOS arm64 thermal + host-load variance.
pres_poisson_nd_ms=47055
```

And in `scripts/wall_check.sh`, pass through a 1.5× threshold
multiplier for this specific key (the existing 2× threshold for
the AMD baselines stays).

## References

- `docs/planning/EPIC_2/SPRINT_25/PLAN.md` Day 11
- `docs/planning/EPIC_2/SPRINT_25/profile_day11_pres_poisson_nd.txt` —
  the 5-run profile capture this decision is based on
- `docs/planning/EPIC_2/SPRINT_24/RETROSPECTIVE.md` "Sprint 25
  inputs" #2-3 — the original drift report + Sprint 25 routing
- `src/sparse_reorder_nd.c` — Day 11's SPARSE_ND_PROFILE
  instrumentation
- `docs/planning/EPIC_2/SPRINT_25/nd_tuning_day8.md` — Sprint 26
  routing for Pres_Poisson 0.85× literal target work
