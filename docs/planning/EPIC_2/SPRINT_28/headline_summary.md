# Sprint 28 Day 12 Cross-Corpus Re-Bench — Headline Summary (started Day 12)

Day 12 ran the 24-setting × 6-fixture cross-corpus matrix; this document captures the top-5 lists + intersection.  Day 13 finishes the full verdict + production-default decisions + test-bound calibration (per PLAN.md Day 12-13 sequence).

## Methodology

Sprint 28 PLAN.md Day 12 task 1: ≤24 representative combinations × 6 corpus fixtures × nnz_L + analyze wall = ≤288 measurements.  Captured in `bench_day12_combinations.csv` + `.txt`.  Each combination run via `sparse_analyze(REORDER_ND)` directly (not via `bench_reorder`'s pre-applied-perm + REORDER_NONE path, because the latter doesn't fire the Sprint 28 Item-4 `SPARSE_ND_SUPERNODAL_POSTORDER` dispatch — the dispatch gate is `analysis->perm != NULL`).  The `analyze_ms` field therefore captures full `sparse_analyze` wall (reorder + etree + colcount + symbolic), not the Sprint 27 Day-13 `sparse_reorder_nd`-only wall.

Default Sprint 28 production state (Sprint 27 inherited): `SPARSE_ND_COARSENING=hcc` (Sprint 27 Day 2 flip), `nd_base_threshold=128` (Sprint 27 Day 3 flip), `SPARSE_FM_FINEST_STRATEGY=baseline` (Sprint 27 + Sprint 28 advisory-only), `SPARSE_ND_SUPERNODAL_POSTORDER=off` (Sprint 28 Day 10 advisory-only).  All other axes opt-in via env var.

Sprint 28's three new axes:
- **Item 1** (Day 2): `SPARSE_FM_THICK_RESTART_PERTURB=gain_noise_formal` with `SPARSE_FM_ANNEALING_SCHEDULE={linear, exponential}`.  Advisory after Day 3.
- **Item 2** (Day 4): `SPARSE_FM_FINEST_STRATEGY=ensemble` with `SPARSE_FM_ENSEMBLE_STRATEGIES` selector list.  Advisory after Day 5.
- **Item 4** (Day 7): `SPARSE_ND_SUPERNODAL_POSTORDER=on`.  Advisory after Day 10.

## Headline: Sprint 27 default remains the Pres_Poisson best

**Settings 2 (Sprint 27 default; HCC + t=128) AND setting 3 (Sprint 28 item-4 SUPERNODAL_POSTORDER=on) are tied for the lowest Pres_Poisson nnz_L at 0.9226×.**

This is the expected outcome — Item-4's supernodal-etree post-pass reorders columns within the existing symbolic Cholesky fill pattern, which is a symmetric-permutation-invariant by construction.  No other Sprint 28 axis or combination beats 0.9226× on Pres_Poisson.

### Pres_Poisson top 8 (lowest ND/AMD ratio)

| Setting | Description | Pres_Poisson ratio | Geomean |
|---:|---|---:|---:|
| **2** | **Sprint 27 default (HCC + t=128)** | **0.9226×** | 1.1551 |
| 3 | Sprint 28 item-4 SUPERNODAL_POSTORDER=on | 0.9226× | 1.1551 |
| 6 | Sprint 28 item-2 ensemble × baseline,fifo,annealing (default) | 0.9374× | 1.1633 |
| 9 | Sprint 28 item-2 ensemble × 4-way (with thick_restart) | 0.9374× | 1.1633 |
| 18 | Sprint 28 stack: item-4 × item-2 (SUPERNODAL=on + ensemble) | 0.9374× | 1.1633 |
| 19 | Sprint 28 stack: item-1 × item-2 (gain_noise + ensemble) | 0.9374× | 1.1666 |
| 20 | Sprint 28 stack: item-4 × item-1 × item-2 (all three) | 0.9374× | 1.1666 |
| 8 | Sprint 28 item-2 ensemble × fifo,annealing (drop baseline) | 0.9407× | 1.1707 |

### Pres_Poisson worst 6 (catastrophic regress; > 1.0× — ND becomes worse than AMD)

| Setting | Description | Pres_Poisson ratio |
|---:|---|---:|
| 24 | Sprint 28 kitchen-sink lite (item-4 + t=256 + fixed_k + spectral) | 1.6473× |
| 10 | Sprint 27 advisory: per_vertex_fixed_k × hybrid | 1.4022× |
| 22 | Sprint 28 stack: item-4 × per_vertex_fixed_k × hybrid | 1.4022× |
| 16 | Sprint 27 advisory: t=256 + per_vertex_fixed_k × hybrid | 1.3596× |
| 23 | Sprint 28 kitchen-sink (all axes + t=256 + fixed_k) | 1.3356× |
| 4 / 5 / 17 | Sprint 28 item-1 thick_restart × gain_noise_formal | 1.1656× |

The fixed_k×hybrid and thick_restart × gain_noise_formal axes catastrophically regress Pres_Poisson.  Item-4's SUPERNODAL_POSTORDER=on is invariant under permutation (bit-equals the base axis on Pres_Poisson nnz_L for every combination it joins).

## Corpus-wide best (geometric mean of 6 fixture ratios)

| Setting | Description | Geomean | Pres_Poisson | Kuu |
|---:|---|---:|---:|---:|
| **15** | **Sprint 27 advisory: t=256 Kuu opt-in** | **1.1156** | 0.9475× | 1.7716× |
| 21 | Sprint 28 stack: item-4 × Kuu opt-in (SUPERNODAL=on + t=256) | 1.1156 | 0.9475× | 1.7716× |
| 2 | Sprint 27 default (HCC + t=128) | 1.1551 | 0.9226× | 1.8822× |
| 3 | Sprint 28 item-4 SUPERNODAL_POSTORDER=on | 1.1551 | 0.9226× | 1.8822× |
| 12 | Sprint 27 advisory: annealing × linear | 1.1588 | 0.9424× | 1.8830× |

**Setting 15 (Sprint 27 advisory `--nd-threshold 256`) remains the corpus-wide best — bit-equal to Sprint 28 setting 21 (item-4 layered on top).**  Item-4 doesn't move Kuu either; the Kuu win comes entirely from `t=256`.  This confirms Sprint 27's Day-13 corpus-wide-best verdict survives Sprint 28's new axes intact.

## Intersection — Pres_Poisson best AND corpus-wide best

**No single setting wins both.**  The Pres_Poisson-best (settings 2 + 3) and corpus-wide-best (settings 15 + 21) are different settings.  This matches the Sprint 27 Day-13 finding ("clear bimodal-class-vs-FE-mesh tradeoff — no single setting wins on both fronts").  Sprint 28 produced ZERO new winners on either front.

## Kuu best (lowest ND/AMD ratio)

| Setting | Description | Kuu ratio | vs Sprint 27 default |
|---:|---|---:|---:|
| **23** | **kitchen-sink (item-4 + item-1 + item-2 + t=256 + fixed_k hybrid)** | **1.1934×** | −36.6% |
| 16 | Sprint 27 advisory: t=256 + per_vertex_fixed_k × hybrid | 1.2170× | −35.4% |
| 10 / 22 | per_vertex_fixed_k × hybrid (+ item-4 SUPERNODAL=on) | 1.2286× | −34.7% |
| 24 | kitchen-sink lite (item-4 + t=256 + fixed_k + spectral) | 1.3350× | −29.1% |

Setting 23 (Sprint 28 kitchen-sink) is the new corpus-wide Kuu best at 1.1934× — a -1pp improvement over Sprint 27's setting 16 (1.2170×).  But setting 23 catastrophically regresses Pres_Poisson (1.3356×; +41pp from Sprint 27 default 0.9226×), so it's not a viable default flip — purely advisory for workloads that look like Kuu.

## Sprint 28 Default (Setting 2) Per-Fixture

| Fixture | n | AMD nnz_L | ND nnz_L | ratio | analyze_ms |
|---|---:|---:|---:|---:|---:|
| nos4 | 100 | 637 | 637 | 1.000× | 0.3 |
| bcsstk04 | 132 | 3 143 | 3 722 | 1.184× | 142.7 |
| Kuu | 7 102 | 406 264 | 764 664 | 1.882× | 5 247 |
| bcsstk14 | 1 806 | 116 071 | 130 422 | 1.124× | 421 |
| s3rmt3m3 | 5 357 | 474 609 | 487 832 | 1.028× | 4 538 |
| **Pres_Poisson** | 14 822 | 2 668 793 | **2 462 201** | **0.923×** | 3 418 |

## Sprint 28 Headline Verdict (Day 12 preview; Day 13 finalises)

**Pres_Poisson: 0.923× of AMD — 7.3pp from the literal 0.85× target.**

Sprint 28's three new advisory axes (Item 1 gain_noise_formal; Item 2 ensemble; Item 4 supernodal-postorder) all FAIL to move Pres_Poisson:
- Item 1 (gain_noise_formal): regresses Pres_Poisson by +24pp (1.166× ratio); already documented advisory by `gain_noise_decision.md`.
- Item 2 (ensemble): regresses Pres_Poisson by +1.5pp (0.937× ratio); already documented advisory by `ensemble_fm_decision.md`.
- Item 4 (supernodal-postorder): bit-identical to default by symmetric-permutation invariance; already documented advisory by `non_pipeline_decision.md`.

**Day-12 cross-corpus matrix CONFIRMS the empirical conclusion: no Sprint 28 axis or combination lands Pres_Poisson ≤ 0.85×.**  Item-4's bit-equality with the default reinforces the Day-10 retirement of the literal 0.85× target.

The 5-sprint trajectory:
- Sprint 22: 1.063× ND/AMD (regress vs AMD baseline)
- Sprint 23: 0.952× (leaf-AMD splice; first ND-beats-AMD)
- Sprint 24: 0.942× (qg-AMD + threshold)
- Sprint 25: 0.922× (spectral coarsest + per-vertex lift)
- Sprint 26: 0.9217× (HCC matching)
- Sprint 27: 0.9226× (HCC default flip + t=128 + FINEST FM axes)
- **Sprint 28: 0.9226× (non-pipeline supernodal-etree pivot)**

The gap stops closing at Sprint 25's 0.922× and stays there ±0.05pp for three sprints.  Sprint 28's empirical evidence (the only sprint to attempt a non-pipeline-level intervention) confirms the floor is structural — neither the multilevel pipeline nor a post-permutation can move below it.  `non_pipeline_decision.md` formally retired the literal 0.85× target on Day 10.

## Day-13 Follow-Up Tasks

Day 13 closes Item 5 with:
- Final production-default decisions per axis (per `non_pipeline_decision.md` Day 10 + `ensemble_fm_decision.md` Day 5 + `gain_noise_decision.md` Day 3, all three: stay default).
- `test_nd_pres_poisson_fill_with_leaf_amd` bound calibration: stays at 0.94× (Sprint 27 ratio + 2pp; Sprint 28 didn't close the literal 0.85× target).
- Item 7 prep: `docs/algorithm.md` ND subsection draft updates + `RETROSPECTIVE.md` section skeleton.
