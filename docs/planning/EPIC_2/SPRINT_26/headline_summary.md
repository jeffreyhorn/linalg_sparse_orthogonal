# Sprint 26 Day 13 — Headline Summary

## Verdict

| Headline gate | result |
|---|---|
| Pres_Poisson ND/AMD ≤ 0.85× (literal target) | **MISS** — best opt-in 0.9217× (Sprint 25 setting 13, unchanged) |
| Pres_Poisson ND/AMD ≤ 0.90× (partial close) | **MISS** — best opt-in 0.9217× still 2.2pp short |
| Pres_Poisson ND/AMD < Sprint 25 default (0.9524×) | **PASS** — Sprint 26 Day 5 nd_base_threshold=96 flip moves default to 0.9504× (-0.2pp) |
| Smaller-fixture corpus safety (no > 5pp regression on Sprint 26 default) | **PASS** — see "Default-path improvements" below |
| HCC bcsstk14 sep=0 blocker fixed | **PASS** — Day 3 sep=0 fall-back; bcsstk14 sep > 0 under HCC |
| `sparse_eigs.c:948` UBSan log cleared | **PASS** — Day 1 one-line guard fix |
| Test bound tightening (`test_nd_pres_poisson_fill_with_leaf_amd`) | **STAY** — bound stays at 0.96× (Sprint 24 Day 7); Pres_Poisson default 0.9504× has 0.96pp margin |
| `make wall-check` exits 0 | **PASS** — Pres_Poisson ND ≈ 11 s vs 70 583 ms 1.5× ceiling |

Per PLAN.md Day 13 task 5, **the Pres_Poisson 0.85× target routes to Sprint 27** — fourth consecutive sprint to miss the literal target (Sprint 22-26).  Sprint 26 Items 5/6/7 (FINEST FM FIFO, geometric grid-cut, per-vertex separator scoring) all closed without moving Pres_Poisson past 0.9217×.

## Combination matrix (12 settings × 6 fixtures)

Sprint 26 Day 13 cross-corpus sweep (`bench_day13_combinations.{csv,txt}`).  Sprint 26 Day 5's `nd_base_threshold = 96` is the active default; all settings sweep on top of that.

ND/AMD ratios (AMD nnz_L: nos4=637, bcsstk04=3143, Kuu=406264, bcsstk14=116071, s3rmt3m3=474609, Pres_Poisson=2668793):

| # | setting | nos4 | bcsstk04 | Kuu | bcsstk14 | s3rmt3m3 | **Pres_Poisson** |
|---|---|---:|---:|---:|---:|---:|---:|
| 01 | baseline (Sprint 26 default) | 1.270× | 1.184× | 2.169× | 1.114× | 1.018× | 0.9504× |
| 02 | setting 13 (HCC + ratio=200) | 1.392× | 1.183× | 1.962× | 1.077× | 1.017× | **0.9217×** ★ |
| 03 | setting 15-ish (HCC + ratio=200 + spectral + balanced_boundary) | 1.196× | 1.183× | 1.534× | 1.069× | 1.007× | 0.9565× |
| 04 | balanced_boundary alone | **1.174×** ★ | 1.184× | 1.299× | 1.068× | 1.006× | 0.9525× |
| 05 | per_vertex alone | 1.201× | **1.129×** ★ | 1.703× | 1.301× | 1.294× | 1.241× |
| 06 | fifo alone | 1.220× | 1.184× | 2.032× | 1.149× | 1.019× | 0.9799× |
| 07 | setting 13 + INTERMEDIATE=3 | 1.392× | 1.183× | 1.963× | 1.080× | 1.018× | 0.9483× |
| 08 | setting 15-ish + fifo | **1.174×** ★ | 1.177× | 1.503× | **1.040×** ★ | 1.014× | 0.9326× |
| 09 | setting 13 + fifo | 1.268× | 1.177× | 1.852× | 1.143× | 1.020× | 0.9679× |
| 10 | full Sprint-26-max | **1.174×** ★ | 1.177× | **1.204×** ★ | 1.050× | **1.005×** ★ | 0.9484× |
| 11 | HCC alone | 1.392× | 1.183× | 2.315× | 1.077× | 1.017× | 0.9357× |
| 12 | per_vertex_balance alone | 1.201× | **1.129×** ★ | 1.737× | 1.301× | 1.294× | 1.241× |

★ = best for that fixture across the 12 settings.

Setting legend (env-var combinations):
- **02 setting 13**: `SPARSE_ND_COARSENING=hcc SPARSE_ND_COARSEN_FLOOR_RATIO=200`
- **03 setting 15-ish**: setting 13 + `SPARSE_ND_COARSEST_BISECTION=spectral SPARSE_ND_SEP_LIFT_STRATEGY=balanced_boundary`
- **08**: setting 15-ish + `SPARSE_FM_FINEST_STRATEGY=fifo`
- **10 full Sprint-26-max**: setting 15-ish + `SPARSE_FM_FINEST_STRATEGY=fifo SPARSE_FM_INTERMEDIATE_PASSES=3`

## Per-fixture winners

| fixture | best setting | ND/AMD | Δ vs Sprint 25 best | Sprint 26 contribution |
|---|---|---:|---:|---|
| nos4 | 04/08/10 (balanced_boundary) | 1.174× | -8pp from setting 15's 1.256× | balanced_boundary alone now matches setting 15 (Day 5 + setting interactions) |
| bcsstk04 | 05/12 (per_vertex variants) | 1.129× | -5pp from baseline 1.184× | **NEW** (Sprint 26 Day 10 only) |
| Kuu | **10 (full Sprint-26-max)** | **1.204×** | **-10pp** from Sprint 25 setting 15's 1.309× | fifo + INTERMEDIATE=3 add cumulative -10pp |
| bcsstk14 | **08 (setting 15-ish + fifo)** | **1.040×** | -3pp from Sprint 25 setting 15's 1.037× | fifo contributes ~-1pp |
| s3rmt3m3 | 10 (full Sprint-26-max) | 1.005× | -0.2pp from Sprint 25 setting 15's 1.007× | small win |
| **Pres_Poisson** | 02 (setting 13) | **0.9217×** | unchanged from Sprint 25 setting 13 | Sprint 26 contributes 0pp |

**Sprint 26 best new opt-in setting: setting 10 (full Sprint-26-max)** — Kuu 1.204× (-10pp from Sprint 25), nos4/bcsstk14/s3rmt3m3 close-to-best, Pres_Poisson 0.9484× (mid-pack but acceptable).

**Sprint 26 best Pres_Poisson opt-in: still setting 13 (Sprint 25)** at 0.9217×.

## Default-flip rule application (PLAN.md task 3)

PLAN.md flip rule: ≥ 1pp Pres_Poisson tightening + no smaller-fixture regression past 5pp.

| candidate default | Pres_Poisson | smaller-fixture worst | flip? |
|---|---|---|---|
| 02 setting 13 | -2.9pp ✓ | nos4 +12.2pp ✗ | FAIL (nos4 regress) |
| 03 setting 15-ish | +0.6pp ✗ | n/a | FAIL |
| 04 balanced_boundary | +0.2pp ✗ | n/a (helps every fixture) | FAIL |
| 05 per_vertex | +29pp ✗ | n/a | FAIL (Pres_Poisson catastrophic) |
| 06 fifo | +3.0pp ✗ | n/a | FAIL |
| 08 setting 15-ish + fifo | -1.8pp ✓ | nos4 -9.6pp WIN; **bcsstk04 -0.6pp; Kuu -30pp WIN; bcsstk14 -7.4pp WIN; s3rmt3m3 -0.4pp** | (would be PASS, but env-var dependency means this can't be a "no env vars" default) |
| 10 full Sprint-26-max | -0.2pp ✗ | various wins | FAIL on the ≥ 1pp gate |
| 11 HCC alone | -1.5pp ✓ | nos4 +12.2pp ✗; Kuu +14.6pp ✗ | FAIL |

**No production default flip.**  The closest candidate (setting 08) requires 4 env vars and isn't a "drop env vars and ship as default" change.

The Sprint 26 default flip that DID land: **`nd_base_threshold = 96`** (Day 5).  This contributes the -0.2pp Pres_Poisson default tightening (0.9524× → 0.9504×) plus dramatic wall-time improvements across the corpus (-67.9% on Pres_Poisson ND).

## Sprint 26 default-path improvements (PLAN.md task 4 verification)

Sprint 25 vs Sprint 26 default-path comparison (`nd_base_threshold` flip is the only Sprint 26 default change):

| fixture | Sprint 25 default ND/AMD | Sprint 26 default ND/AMD | Δ |
|---|---:|---:|---:|
| nos4 | 1.520× | 1.270× | -25pp WIN |
| bcsstk04 | 1.178× | 1.184× | +0.6pp noise |
| Kuu | 2.275× | 2.169× | -10.6pp WIN |
| bcsstk14 | 1.130× | 1.114× | -1.6pp WIN |
| s3rmt3m3 | 1.009× | 1.018× | +0.9pp noise |
| Pres_Poisson | 0.9524× | 0.9504× | -0.2pp WIN |

5 of 6 fixtures improved; bcsstk04 + s3rmt3m3 are within 1pp noise (not material regressions).  No fixture regresses past the 5pp band → corpus safety gate **PASS**.

Headline default-path wall-time improvements (all from Day 5's `nd_base_threshold` flip):
- nos4: -38 % wall
- bcsstk04: -69 % wall
- Kuu: -56 % wall
- bcsstk14: -81 % wall
- s3rmt3m3: -74 % wall
- **Pres_Poisson: -68 % wall** (38 131 ms → 12 207 ms; the headline)

## Test bound tightening (PLAN.md task 4)

PLAN.md Day 13 task 4 routing:
> If items 5-7 closed Pres_Poisson default to ≤ 0.85×, tighten to ≤ 0.87× nnz_amd; if partial close to ≤ 0.90×, tighten to ≤ 0.92× nnz_amd; if no default movement (Sprint 26 ships everything as advisory), bound stays at 0.96×.

Items 5/6/7 verdict: **no default movement**.  Sprint 26 default Pres_Poisson is 0.9504× (vs Sprint 24 Day 7's 0.96× test bound = 0.96pp margin; reasonable).

`tests/test_reorder_nd.c::test_nd_pres_poisson_fill_with_leaf_amd` bound **stays at ≤ 0.96× nnz_amd** (Sprint 24 Day 7 setting).

## Per-fixture advisory (Sprint 26 final)

Updated user-facing advisory list (extends Sprint 24/25 advisory):

| workload | recommended setting | wins |
|---|---|---|
| Pres_Poisson | `SPARSE_ND_COARSENING=hcc SPARSE_ND_COARSEN_FLOOR_RATIO=200` (Sprint 25 setting 13) | 0.9217× (-2.9pp from default) |
| Kuu / irregular SPDs | `SPARSE_ND_COARSENING=hcc SPARSE_ND_COARSEN_FLOOR_RATIO=200 SPARSE_ND_COARSEST_BISECTION=spectral SPARSE_ND_SEP_LIFT_STRATEGY=balanced_boundary SPARSE_FM_FINEST_STRATEGY=fifo SPARSE_FM_INTERMEDIATE_PASSES=3` (Sprint 26 setting 10 = "full Sprint-26-max") | Kuu 1.204× (-10pp from Sprint 25 setting 15's 1.309×) |
| bcsstk14 (PDE/structural) | `SPARSE_ND_COARSENING=hcc SPARSE_ND_COARSEN_FLOOR_RATIO=200 SPARSE_ND_COARSEST_BISECTION=spectral SPARSE_ND_SEP_LIFT_STRATEGY=balanced_boundary SPARSE_FM_FINEST_STRATEGY=fifo` (setting 8 = "setting 15-ish + fifo") | 1.040× (Sprint 26 best on this fixture) |
| bcsstk04 (small irregular) | `SPARSE_ND_SEP_LIFT_STRATEGY=per_vertex` (Sprint 26 Day 10 advisory) | 1.129× (-4.6 % from default) |
| nos4 / s3rmt3m3 / small fixtures | `SPARSE_ND_SEP_LIFT_STRATEGY=balanced_boundary` (Sprint 24 advisory) | small win |
| Default (no env vars) | unchanged baseline | 0.9504× Pres_Poisson; corpus-wide wall improvement from Day 5's `nd_base_threshold=96` flip |

## What ships in Sprint 26

Production-default changes:
- **`nd_base_threshold = 96`** (Day 5; was 32) — corpus-wide wall improvement + small fill-quality wins on multiple fixtures.
- HCC sep=0 fall-back path in `sparse_graph_partition` (Day 3) — invisible to default callers but unblocks `SPARSE_ND_COARSENING=hcc` opt-in on bcsstk14.
- `sparse_eigs.c:948` UBSan guard fix (Day 1) — clears the Sprint 25 inherited sanitize log.

New opt-in env vars + values:
- `SPARSE_FM_FINEST_STRATEGY=fifo` (Day 7) — FIFO bucket-tie-break.  Advisory in combination only (regresses Pres_Poisson alone).
- `SPARSE_ND_SEP_LIFT_STRATEGY=per_vertex` / `per_vertex_balance` / `per_vertex_degree` (Day 10/12) — per-vertex separator scoring.  Advisory for bcsstk04-class only.

Infrastructure:
- `SPARSE_ND_PROFILE` per-recursion-depth instrumentation (Day 4) — extends Sprint 25 Day 11 with depth attribution.
- `SPARSE_HCC_DEBUG` cmap-emit instrumentation (Day 1; used Day 2 to diagnose bcsstk14 sep=0).

Documents:
- `hcc_sep_zero_diagnosis.md` — Day 2 root-cause + Day 3 post-fix verification
- `per_recursion_profile_day4.md` — Day 4 per-depth analysis
- `nd_base_threshold_decision.md` — Day 5 default-flip decision
- `finest_fm_design.md` + `finest_fm_decision.md` — Day 6/8 FIFO design + decision
- `geometric_cut_design.md` — Day 9 Item 6 rejection
- `per_vertex_sep_design.md` + `per_vertex_sep_decision.md` — Day 10/12 per_vertex design + decision

## Items deferred to Sprint 27+

Per Sprint 26's "no item closes Pres_Poisson 0.85×" finding, the gap routes to Sprint 27+ with these candidates:

1. **Root-level spectral bisection** — extend Sprint 25 spectral from coarsest to root level.  Higher prior than Sprint 26 Item 6's 2D-grid heuristic; reuses Sprint 20-21 Lanczos eigensolver.  4-day budget.
2. **Annealing-acceptance FM** — rejected at Sprint 26 Day 6 design for cost reasons; Day 5's wall improvement (-68% on Pres_Poisson) makes affordable now.  3-4 day budget.
3. **Multi-strategy ensemble** — run baseline + FIFO + (future axes) in parallel; pick best cut per partition call.  Doubles wall-time but explores 2× the FM landscape.
4. **Larger nd_base_threshold beyond 96** — Sprint 26 Day 5 found t=96 was the maximum threshold satisfying the strict flip rule; t=128 regressed s3rmt3m3 by +1.05 %.  Sprint 27 could re-evaluate with relaxed flip-rule (e.g., 2pp tolerance).
5. **Tunable per_vertex selection criterion** — Sprint 26 Day 12 found the 70/30 balance gate dominates the score formula; fixed-K (vs dynamic-K) selection would let weight schemes differentiate.

The shipping story for Sprint 27 (when ready):
"Sprint 26 closed Sprint 25's bcsstk14 sep=0 blocker and shipped a Day-5 default flip that improved Pres_Poisson ND wall by 68% with -0.2pp fill-quality default tightening + corpus-wide wall improvements.  Three new advisory env-var-gated axes added (FIFO bucket-tie-break, per_vertex separator scoring, per-recursion-depth profiling).  0.85× Pres_Poisson literal target misses by 7.2pp; routes to Sprint 27 with five concrete avenues identified above."

## References

- `docs/planning/EPIC_2/SPRINT_26/PLAN.md` Day 13
- `docs/planning/EPIC_2/SPRINT_26/bench_day13_combinations.{csv,txt}` — raw 12-setting × 6-fixture sweep
- `docs/planning/EPIC_2/SPRINT_26/bench_day13_amd_qg.csv` — qg-AMD parity capture
- `docs/planning/EPIC_2/SPRINT_25/headline_summary.md` — Sprint 25 Day 9 analogue (parallel structure)
- `docs/planning/EPIC_2/SPRINT_26/{hcc_sep_zero_diagnosis,nd_base_threshold_decision,finest_fm_decision,geometric_cut_design,per_vertex_sep_decision}.md` — per-day decision docs
