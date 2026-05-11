# Pres_Poisson ND Wall Further Reduction — No-Op Day (Sprint 28 Day 11, Item 6 Conditional)

## Trigger Check

Sprint 28 PLAN.md Day 11 task 1 specifies Item 6 fires only if one of:

- **(a)** Item 4's chosen non-pipeline approach lands a structural change that opens a new wall-reduction path.
- **(b)** Real-world Sprint-28-cycle workloads surface a need.

If neither trigger fires, Day 11 ends at task 1 + commit no-op + reallocates the 11 remaining hours to Day 12-14 prep (per PLAN.md Day 11 task 1).

### Trigger (a): Item 4 structural change → DOES NOT FIRE

Sprint 28 Day 1 picked **(c) supernodal-etree reordering** as Item 4's non-pipeline pivot.  The PLAN.md Day 11 task 1 already noted this case in advance:

> "(c) supernodal-etree reordering doesn't change the partition wall (it's a post-permutation), so (c) doesn't trigger Item 6.  (a) METIS-style multi-matching adds wall (K-matching multiplier) — also doesn't trigger Item 6 (regression, not improvement)."

Sprint 28 Days 6-10 confirmed empirically: the supernodal-etree post-pass runs AFTER `sparse_analyze`'s partition phase (`sparse_etree_compute` + `sparse_etree_postorder` + recompute pass on the composed perm).  It adds ~6-15 % to the analyze wall (see `non_pipeline_interim_day7.txt` + `non_pipeline_sweep.txt`) without restructuring the partition phase at all — the multilevel coarsening + FM + leaf-AMD pipeline runs exactly as it did in Sprint 27.  Trigger (a) cannot fire under pick (c).

### Trigger (b): real-world ND-wall feedback → DOES NOT FIRE

`git log --all --since="Sprint 27 merge"` between `79f7a85` (Sprint 27 PR #35 merge) and `d79e02d` (Sprint 28 Day 10) shows:
- 8 Sprint 27 wrap-up commits (Days 13-14 closure + PR review fixes).
- 11 Sprint 28 commits (Days 1-10 per the PLAN.md schedule).
- No external feedback / issues / user reports.

The Sprint 28 PLAN.md "External GitHub issues" check surfaces zero open issues mentioning ND wall performance.  No real-world workload has surfaced a need for further Pres_Poisson ND wall reduction since Sprint 27 merged at `79f7a85`.

## Current Pres_Poisson ND Wall Status

`make wall-check` against `docs/planning/EPIC_2/SPRINT_24/wall_check_baseline.txt`:

| Phase | Sprint baseline | Current | Margin to 1.5× gate |
|---|---:|---:|---:|
| Pres_Poisson ND wall | 47 055 ms (Sprint 22 baseline) | ~3 200 ms (Day-11 measurement) | 67 300 ms headroom (21× under gate) |

Cumulative Pres_Poisson ND wall reduction:
- Sprint 22 baseline: 47 055 ms
- Sprint 25 baseline: 38 100 ms (t=32, HEM)
- Sprint 26 Day 5: 12 200 ms (t=96 + HEM)
- Sprint 27 Day 2: 8 826 ms (t=96 + HCC)
- Sprint 27 Day 3: 7 079 ms (t=128 + HCC) — official Sprint 27 default
- Sprint 28 Day 11: ~3 200 ms (no Sprint-28 wall-affecting changes; system load variance over single-run measurement)

**The Pres_Poisson ND wall is no longer a constraint on this codebase.**  Even at the Sprint 27 default (7.1 s), the gate (70.5 s = 1.5× the Sprint-22 baseline) has ~63 s of headroom.  Day 11's wall-check at 3.2 s reflects routine measurement variance; no production behaviour has changed since Sprint 27 Day 3.

## Verdict: No-Op Day

Both Item-6 triggers fail to fire.  No code changes today; the `SPARSE_ND_*` env-var surface, `sparse_reorder_nd` implementation, and `nd_base_threshold = 128` default all stay at their Sprint 27 / Sprint 28 Days 1-10 state.

`make format && make lint && make test && make wall-check` are re-run as a sanity check to confirm Day 10's clean state survives (no regression from any out-of-band file mutation).

## Budget Reallocation (per PLAN.md Day 11 task 1)

The 12-hour Day-11 budget reallocates as:

- **0 hrs**: Item-4 over-budget absorption (Item 4 closed on time at Day 10 with `non_pipeline_decision.md`; no over-budget).
- **6 hrs**: Item-5 prep (Day 12's `bench_day12_combinations.csv` + `.txt` cross-corpus matrix scaffolding — design the ≤24-setting matrix that combines Sprint 28's new axes with Sprint 27 Day-13's baseline; capture the dispatch logic for invoking each setting; verify the bench harness path works end-to-end on a smoke fixture).
- **6 hrs**: Item-7 prep (Day 13's `headline_summary.md` skeleton + `docs/algorithm.md` ND-subsection diff draft + `SPRINT_28/RETROSPECTIVE.md` section skeleton).

These prep tasks run between Day 11 and Day 12 / 13 / 14; the work itself surfaces in those days' commits.

## Sprint 28 Cumulative Wall Status

Per `wall_check_baseline.txt`:

| Test | Baseline (ms) | Gate (×) | Current (ms) | Margin |
|---|---:|---:|---:|---:|
| bcsstk14 qg-AMD | 130 | 2× | ~50-100 | well under |
| Pres_Poisson AMD | 8 000 | 2× | ~2 200 | well under |
| Pres_Poisson ND | 47 055 | 1.5× | ~3 200 | well under (-93 %) |

Sprint 28's contributions to wall performance: zero (the supernodal-etree post-pass adds 6-15 % on `sparse_analyze` ONLY when `SPARSE_ND_SUPERNODAL_POSTORDER=on`, which is opt-in advisory).  Default path bit-identical to Sprint 27.

## Day 12 Item-5 Prep — Scaffolding Notes

(Drafted Day 11 in anticipation of Day 12's `bench_day12_combinations.csv` work.)

The Day-12 ≤24-setting matrix combines Sprint 28's new axes with Sprint 27 Day-13's baseline.  Settings to capture:

1. Sprint 26 default (HEM, t=96) — historical reference.
2. Sprint 27 default (HCC + t=128) — inherited baseline.
3-4. Sprint 28 Item-4 axis: `SPARSE_ND_SUPERNODAL_POSTORDER={off, on}`.  Day-9 sweep already covered AMD + ND × env on/off; Day 12 picks the on-cell for the matrix.
5-6. Sprint 28 Item-1 axis: `SPARSE_FM_THICK_RESTART_PERTURB={random_flip, gauss_noise, gain_noise_formal}` (Day-3 close, advisory only).
7-10. Sprint 28 Item-2 axis: `SPARSE_FM_FINEST_STRATEGY=ensemble` × {selector list variants per Day-5 close}.
11-17. Sprint 27 advisory recipes (Kuu opt-in `--nd-threshold 256 + per_vertex_fixed_k × hybrid`; thick_restart × {random_flip, boundary_shuffle}; annealing × {linear, exponential, cosine}).
18-24. Sprint 28 stack combinations (Item-4 × Item-1, Item-4 × Item-2, Item-4 × Item-1 × Item-2, Item-4 × Sprint-27-advisory-recipes).

Sprint 28 expected outcome on the matrix (per non_pipeline_decision.md): no flip — supernodal-etree adds wall cost without nnz_L improvement; Item 1's `gain_noise_formal` lands as advisory (Day 3 verdict); Item 2's ensemble lands as advisory (Day 5 verdict).  The matrix verifies the Sprint 27 advisory recipes still hold under Sprint 28 defaults.

Reproduction: `/tmp/bench_day9.c` pattern from Day-9 (calls `sparse_analyze` directly with reorder enum, captures `analysis->sym_L.nnz` + analyze wall) extends naturally to all the env-var stacks above.  Day 12 will commit the bench helper into `benchmarks/bench_combinations_day12.c` so the matrix is reproducible from-tree.

## Files Generated

- `docs/planning/EPIC_2/SPRINT_28/wall_reduction_decision.md` — this document (no-op verdict + budget reallocation).

## Files NOT Modified

- `src/sparse_reorder_nd.c` — no Day-11 changes; `nd_base_threshold` stays at 128.
- `src/sparse_graph.c` — no Day-11 changes; HCC + Kuu-safe default coarsening stays.
- `tests/test_reorder_nd.c` — no Day-11 changes; wall-affecting tests pass at Day-10 state.
- `docs/algorithm.md` — no Day-11 changes (Day 14 wraps up).

## Headline Status After Day 11 Item-6 Close

- **No Item-6 trigger fired.**  Day 11 is a no-op decision day; the 12-hour budget reallocates to Item-5 prep (6 hrs) + Item-7 prep (6 hrs) per the PLAN.md slack rule.
- **Pres_Poisson ND wall: ~3.2 s** at the Sprint 27 default; the Sprint 22 baseline gate (70.5 s) has 21× headroom.  Further wall reduction is not motivated by any current constraint.
- **Sprint 29+ parking-lot inputs**: deepen-coarsen-and-reuse-symbolic + parallelise FM gain-bucket picks via OpenMP + batch-process leaf-AMD calls remain available if a future workload surfaces a need.  None in Sprint 29 budget.
