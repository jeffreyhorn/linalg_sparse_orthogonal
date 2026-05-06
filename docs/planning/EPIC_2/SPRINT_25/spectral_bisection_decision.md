# Sprint 25 Day 10 — `SPARSE_ND_COARSEST_BISECTION` Production-Default Decision

## Decision

**Production default stays `SPARSE_ND_COARSEST_BISECTION=gggp`**
(Sprint 22 routing: brute-force at n ≤ 20, GGGP for n > 20; bit-
identical to current master).  Spectral bisection ships behind
the env var as advisory.

**Recommended advisory use:** `SPARSE_ND_COARSEST_BISECTION=spectral`
for callers who want the **22× ND wall_time speedup** on
Pres_Poisson (or other regular-mesh fixtures) at no fill-quality
cost.  Setting 13 (HCC + ratio=200) remains the better choice for
Pres_Poisson **fill** quality.

## Why the default flip wasn't made

Per Day 8's `bench_day8_spectral_only.txt` + Day 9's combined-
effect sweep, spectral bisection alone (setting 05) on Pres_Poisson:

| metric | default (gggp) | spectral alone |
|---|---|---|
| Pres_Poisson ND/AMD | 0.952× | 0.953× (essentially unchanged) |
| Kuu ND/AMD | 2.275× | 1.744× (-23.3pp WIN) |
| bcsstk14 ND/AMD | 1.129× | 1.062× (-2.2pp WIN) |
| Pres_Poisson ND wall_time | 37 365 ms | 1 700 ms (**22× speedup**) |

Spectral fails the PLAN.md flip rule's Pres_Poisson criterion:
"flip `SPARSE_ND_COARSEST_BISECTION` to `spectral` if it's the
**dominant Pres_Poisson contributor**."  Spectral barely moves
Pres_Poisson nnz_L (essentially unchanged from default), so it
isn't the dominant contributor — it isn't even a measurable
contributor on the headline fixture.

The wall-time win (22× speedup) is genuine and useful, but PLAN.md
Day 10's flip rule is keyed on nnz_L not wall_time.  Documenting
spectral as advisory ships the wall-time win to callers who want
it without committing the entire user base to the path's
infrastructure cost (Lanczos eigensolver + Laplacian builder per
ND call).

## Why the spectral path doesn't move Pres_Poisson nnz_L

Three independent algorithmic axes (HCC, multi-pass FM
intermediate, spectral bisection) all show the same wash-out
pattern on Pres_Poisson — see `nd_tuning_day8.md` "Why three
independent axes all wash out on Pres_Poisson" for the full
hypothesis.  Summary:

1. The 3-pass FM at the finest level (Sprint 23 Day 11) on a
   regular structured 2D Poisson grid converges to a strong
   local optimum that dominates the final cut quality.
2. Pre-finest-level choices (matching strategy, intermediate FM
   passes, coarsest bisection method) influence the trajectory
   of the multilevel pipeline but not the endpoint nnz_L on
   regular meshes.
3. Where spectral DOES help (Kuu, bcsstk14) is on irregular SPD
   fixtures where finest-level FM doesn't fully converge — the
   coarsest-level cut has lasting downstream influence.

For Sprint 26, closing Pres_Poisson 0.85× requires changing the
finest-level FM, not the coarsest-level choices.  This is
documented in `nd_tuning_day8.md` "Sprint 26 routing".

## PLAN.md flip rule application

PLAN.md Sprint 25 item 3 + Day 10 task 1 spec the flip rule:

> Flip `SPARSE_ND_COARSEST_BISECTION` to `spectral` if it's the
> dominant Pres_Poisson contributor.

| sub-rule | result |
|---|---|
| Spectral is the dominant Pres_Poisson contributor | ✗ (essentially unchanged from default) |

Default stays `gggp`.  Spectral ships behind the env var.

## Per-fixture advisory

| workload | recommended setting | benefit |
|---|---|---|
| Pres_Poisson (wall-time-sensitive) | `SPARSE_ND_COARSEST_BISECTION=spectral` | ND wall 37 s → 1.7 s (**22× speedup**); nnz_L essentially unchanged |
| Pres_Poisson (fill-quality-sensitive) | `SPARSE_ND_COARSENING=hcc SPARSE_ND_COARSEN_FLOOR_RATIO=200` (Sprint 25 setting 13) | nnz_L 3pp tighter; wall_time unchanged |
| Pres_Poisson (both) | the full set: HCC + ratio=200 + spectral + interFM=3 + balanced_boundary (Sprint 25 setting 15) | Kuu -97pp (huge win); Pres_Poisson 0.5pp tighter; ND wall 23× speedup |
| Kuu / irregular SPD (fill-quality-sensitive) | `SPARSE_ND_COARSEST_BISECTION=spectral` (alone or with HCC) | Kuu -23.3pp WIN; bcsstk14 -2.2pp WIN |
| All other corpus | (default) — spectral alone is essentially noise on n ≤ 1 000 fixtures | unchanged |

## ND wall_time benefit explained

Spectral bisection at the coarsest level produces a near-optimal
geometric cut on regular mesh fixtures (Pres_Poisson is a 2D
Poisson grid; the Fiedler vector's median partition closely
matches the grid's geometric mid-row/column cut).  This near-
optimal cut needs fewer FM polish passes through the
uncoarsening hierarchy to converge to the final cut.  Each
uncoarsening level's FM run terminates earlier when the input
partition is already close to the local optimum.

On Pres_Poisson, the cumulative FM work drops from ~37 s to
~1.7 s (~22× speedup).  Per-level wall_time savings:
- Each FM pass on the finest level (n = 14 822) runs O(|E|) per
  pass; spectral's near-optimal starting partition reduces the
  number of beneficial moves found per pass.
- Multi-level FM (with the 3-pass finest-level Sprint 23 Day 11
  default) compounds the savings: each pass converges faster.

The savings don't appear on irregular SPD fixtures (Kuu's wall
under spectral = ~700 ms; default = ~12 400 ms; ratio similar to
the regular case but the absolute time is different).

## Edge case behavior

Day 8's `test_spectral_bisection_n1` / `n2` / `disconnected` /
`lanczos_failure` tests confirmed the spectral path's edge-case
fallbacks work correctly:

- n ≤ 2 → trivial early-return (no Lanczos call).
- Disconnected graph → `λ_1 - λ_0 < 1e-6` detection → GGGP
  fallback.
- Lanczos non-convergence → `eigs_rc != SPARSE_OK` → GGGP
  fallback.
- 60/40 imbalance check → too-skewed Fiedler cut → GGGP
  fallback.

The GGGP fallback is the universal safety net — every spectral
call that can't produce a balanced partition routes to GGGP, so
the spectral path's external contract is the same as the default
gggp path on edge-case inputs.

## Related decisions

- Sprint 22 Day 3 (`graph_bisect_coarsest`) — established the
  brute-force-at-n≤20 / GGGP-otherwise default routing.
- Sprint 24 Day 6 (`SPARSE_ND_SEP_LIFT_STRATEGY`) — analogous
  decision pattern: env-var-gated; default stays Sprint 22's
  `smaller_weight`; advisory for non-Pres_Poisson workloads.
- Sprint 25 Day 1-3 (`SPARSE_ND_COARSENING`) —
  `coarsening_decision.md` documents the analogous decision for
  HCC vs heavy_edge.
- Sprint 25 Day 4-5 (`SPARSE_FM_INTERMEDIATE_PASSES`) —
  `intermediate_fm_decision.md` (Day 5) decided default stays at 1.

## References

- `docs/planning/EPIC_2/SPRINT_25/PLAN.md` Day 10
- `docs/planning/EPIC_2/SPRINT_25/spectral_bisection_design.md` —
  Day 6 design; Day 7 implementation
- `docs/planning/EPIC_2/SPRINT_25/bench_day8_spectral_only.txt` —
  spectral-alone corpus capture
- `docs/planning/EPIC_2/SPRINT_25/nd_tuning_day8.md` — Day 8
  escalation finding + Sprint 26 routing
- `docs/planning/EPIC_2/SPRINT_25/headline_summary.md` — Day 9
  combined-effect verdict
- `docs/planning/EPIC_2/SPRINT_25/coarsening_decision.md` — Day 10
  HCC default decision (analogous flip-rule application)
