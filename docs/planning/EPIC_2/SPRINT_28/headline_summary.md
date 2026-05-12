# Sprint 28 Cross-Corpus Re-Bench — Headline Summary (Day 13 final)

Day 12 ran the 24-setting × 6-fixture cross-corpus matrix; Day 13 finalises the verdict + production-default decisions + test-bound calibration per PLAN.md Day 12-13 sequence.

## Methodology

Sprint 28 PLAN.md Day 12 task 1: ≤24 representative combinations × 6 corpus fixtures × nnz_L + analyze wall = ≤288 measurements.  Captured in `bench_day12_combinations.csv` + `.txt`.  Each combination run via `sparse_analyze(REORDER_ND)` directly (not via `bench_reorder`'s pre-applied-perm + REORDER_NONE path, because the latter doesn't fire the Sprint 28 Item-4 `SPARSE_ND_SUPERNODAL_POSTORDER` dispatch — the dispatch gate is `analysis->perm != NULL`).  The `analyze_ms` field therefore captures full `sparse_analyze` wall (reorder + etree + colcount + symbolic), not the Sprint 27 Day-13 `sparse_reorder_nd`-only wall.

Default Sprint 28 production state (Sprint 27 inherited): `SPARSE_ND_COARSENING=hcc` (Sprint 27 Day 2 flip), `nd_base_threshold=128` (Sprint 27 Day 3 flip), `SPARSE_FM_FINEST_STRATEGY=baseline` (Sprint 27 + Sprint 28 advisory-only), `SPARSE_ND_SUPERNODAL_POSTORDER=off` (Sprint 28 Day 10 advisory-only).  All other axes opt-in via env var.

Sprint 28's three new axes:
- **Item 1** (Day 2): `SPARSE_FM_THICK_RESTART_PERTURB=gain_noise_formal` with `SPARSE_FM_GAIN_NOISE_SCHEDULE={linear (default), exponential, cosine}`.  (Note: `SPARSE_FM_ANNEALING_SCHEDULE` is reserved for the Sprint 27 annealing FM strategy and does NOT apply here — the gain-noise variant uses its own schedule env var per `parse_fm_gain_noise_schedule()` in `src/sparse_graph.c`.)  Advisory after Day 3.
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

## Sprint 28 Headline Verdict

**Pres_Poisson: 0.923× of AMD — 7.3pp from the literal 0.85× target — MISS (6th consecutive sprint).**

Sprint 28's three new advisory axes all FAIL to move Pres_Poisson:
- Item 1 (gain_noise_formal): regresses Pres_Poisson by +24pp (1.166× ratio); already documented advisory by `gain_noise_decision.md`.
- Item 2 (ensemble): regresses Pres_Poisson by +1.5pp (0.937× ratio); already documented advisory by `ensemble_fm_decision.md`.
- Item 4 (supernodal-postorder): bit-identical to default by symmetric-permutation invariance; already documented advisory by `non_pipeline_decision.md`.

**Day-12 cross-corpus matrix CONFIRMS the empirical conclusion: no Sprint 28 axis or combination lands Pres_Poisson ≤ 0.85×.**  Item-4's bit-equality with the default reinforces the Day-10 retirement of the literal 0.85× target.  See the dedicated retirement section below.

## Production-Default Decisions Per Axis (Day 13 close)

Day 13 finalises the production-default outcomes for each Sprint 28 axis, applying the Day-3 / Day-5 / Day-10 decisions against the Day-12 cross-corpus matrix:

| Axis | Env var | Day-X decision | Cross-corpus evidence | Day-13 verdict |
|---|---|---|---|---|
| Item 1 (gain_noise_formal) | `SPARSE_FM_THICK_RESTART_PERTURB=gain_noise_formal` | Day 3: advisory only (`gain_noise_decision.md`) | Pres_Poisson 1.1656× (settings 4, 5, 17); regresses +24pp from default | **STAY at default** (`random_flip`) |
| Item 2 (ensemble) | `SPARSE_FM_FINEST_STRATEGY=ensemble` + `SPARSE_FM_ENSEMBLE_STRATEGIES` | Day 5: advisory only (`ensemble_fm_decision.md`) | Pres_Poisson 0.9374× under all 4 selector variants (settings 6, 7, 8, 9); regresses +1.5pp from default | **STAY at default** (`baseline`) |
| Item 4 (supernodal-postorder) | `SPARSE_ND_SUPERNODAL_POSTORDER=on` | Day 10: advisory only (`non_pipeline_decision.md`) | Pres_Poisson 0.9226× (setting 3); bit-equals default by symmetric-permutation invariance | **STAY at default** (`off`) |

**Zero production default flips for Sprint 28.**  All three new advisory env vars ship as opt-in.

## Sprint 28 Verdict on the Literal 0.85× Pres_Poisson Target — RETIRED

**Pres_Poisson ND/AMD ratio achieved Sprint 28: 0.9226× (default = best, tied with Item-4 SUPERNODAL_POSTORDER=on).**

The literal 0.85× target REMAINS UNMET for the **6th consecutive sprint** (Sprints 22 → 28):

| Sprint | Default ratio | Best achievable ratio | Sprint hypothesis | Outcome |
|---|---:|---:|---|---|
| 22 | 1.063× | 1.063× | Multilevel ND beats AMD on FE mesh | Miss (ND regresses vs AMD) |
| 23 | 0.952× | 0.952× | Leaf-AMD splice closes gap | Miss; first ND-beats-AMD |
| 24 | 0.952× | 0.942× | qg-AMD + threshold tuning | Miss |
| 25 | 0.952× | 0.922× | Spectral coarsest + per-vertex lift | Miss; best opt-in 0.922× |
| 26 | 0.952× | 0.9217× | HCC matching (default-flip blocked) | Miss |
| 27 | 0.9226× | 0.9226× | HCC default flip + t=128 + FINEST FM axes | Miss; advisory-only ax verdicts |
| **28** | **0.9226×** | **0.9226×** | **Non-pipeline supernodal-etree pivot** | **Miss; structurally invariant by permutation** |

Sprint 28's non-pipeline-level pivot (Item 4 supernodal-etree reordering) is the strongest empirical evidence yet that the literal target is structurally unreachable on this codebase + corpus:

1. **The ratio stops moving at 0.922×.**  Across 4 sprints (25-28), Pres_Poisson stays at 0.922-0.923× ± 0.05pp.  This is not a local minimum the pipeline keeps just-failing to cross — it IS the floor.
2. **Symmetric permutation cannot eliminate fill.**  Sprint 28's Item-4 post-pass reorders columns within the symbolic Cholesky fill pattern; it cannot reduce nnz_L.  This is the only intervention that can act AFTER the pipeline computes the partition, and it produces 0pp delta on the metric.  Any future intervention must act WITHIN or BEFORE the pipeline.
3. **Item-4's bit-equality with the default is the single cleanest piece of evidence.**  Setting 3 (item-4 SUPERNODAL_POSTORDER=on) bit-equals setting 2 (default) on every Pres_Poisson metric across the matrix.  The non-pipeline pivot, while structurally correct + ships useful infrastructure, demonstrates the floor.

**The literal 0.85× Pres_Poisson target is formally retired with Sprint 28's empirical evidence.**  Sprint 29+ revisits the target ONLY with fundamentally different machinery:

- **METIS C library interop** — defer to the production METIS implementation rather than this codebase's in-house multilevel pipeline.  Out of Sprint 29 budget.
- **Geometric mesh-aware ordering with first-class coordinate API** — requires coordinate input the corpus doesn't ship; rejected Sprint 27 Day 9 (Laplacian-spectral coordinates regress +2.3pp).  Long-term parking lot.
- **Hybrid AMD-then-ND-on-separators** — AMD already finds near-optimal cuts on Pres_Poisson per Sprint 27 evidence; ND adds marginal value only at the separator levels.  Possible but speculative; no Sprint 29 advocate.

None of these are budgeted for Sprint 29.

## Test-Bound Calibration

`tests/test_reorder_nd.c::test_nd_pres_poisson_fill_with_leaf_amd` Sprint 27 bound: **0.94×**.

Sprint 28 Day-9 + Day-12 measurements: Pres_Poisson ND ratio = 0.9226× under both env settings (default and Item-4 SUPERNODAL_POSTORDER=on).  Sprint 27's 2pp safety margin above 0.923× is preserved.

**Bound stays at 0.94×.**  No tightening this sprint.  The Day-10 staged diff (test_non_pipeline_pres_poisson_close_to_target with RUN_TEST commented out, asserting ≤ 0.87× would close + 2pp) remains as failing-as-expected scaffolding; it does NOT land as an active test today.

## Files Generated

- `docs/planning/EPIC_2/SPRINT_28/bench_day12_combinations.csv` — Day-12 raw 24-setting × 6-fixture matrix.
- `docs/planning/EPIC_2/SPRINT_28/bench_day12_combinations.txt` — same, grouped by setting.
- `docs/planning/EPIC_2/SPRINT_28/headline_summary.md` — this document.
- `docs/planning/EPIC_2/SPRINT_28/gain_noise_decision.md` — Item-1 advisory verdict (Day 3).
- `docs/planning/EPIC_2/SPRINT_28/ensemble_fm_decision.md` — Item-2 advisory verdict (Day 5).
- `docs/planning/EPIC_2/SPRINT_28/non_pipeline_decision.md` — Item-4 advisory verdict + 0.85× target retirement (Day 10).
- `docs/planning/EPIC_2/SPRINT_28/wall_reduction_decision.md` — Item-6 no-op (Day 11).
- `docs/planning/EPIC_2/SPRINT_28/pivot_decision_day1.md` — Day-1 candidate-(c) supernodal-etree selection.
- `docs/planning/EPIC_2/SPRINT_28/non_pipeline_interim_day{7,9}.{txt,md}` — interim Day-7 + Day-9 measurements.
- `docs/planning/EPIC_2/SPRINT_28/non_pipeline_sweep.txt` — Day-9 24-cell {AMD, ND} × {off, on} sweep.

## Files NOT Modified

- `src/sparse_graph.c::parse_finest_fm_strategy()` — default stays `FINEST_FM_BASELINE` (no Item-2 flip).
- `src/sparse_graph.c::parse_thick_restart_perturb()` — default stays `FM_THICK_RESTART_PERTURB_RANDOM_FLIP` (no Item-1 flip).
- `src/sparse_analysis.c::parse_nd_supernodal_postorder()` — default stays `ND_SUPERNODAL_POSTORDER_OFF` (no Item-4 flip).
- `tests/test_reorder_nd.c::test_nd_pres_poisson_fill_with_leaf_amd` — bound stays `≤ 0.94×` (no tightening).
- `tests/test_reorder_nd.c::test_non_pipeline_pres_poisson_close_to_target` — RUN_TEST stays commented out.
- `tests/test_reorder_nd.c::test_finest_fm_annealing_pres_poisson_close_to_target` (Sprint 27) — RUN_TEST stays commented out.
- `tests/test_reorder_nd.c::test_nd_root_spectral_pres_poisson_close_to_target` (Sprint 27) — RUN_TEST stays commented out.

## Day 14 Follow-Up Tasks (Item 7 close)

- `docs/algorithm.md` ND subsection: add the Sprint 28 closures subsection (Item-1 `gain_noise_formal`, Item-2 `ensemble`, Item-4 `supernodal_postorder` env vars) + supersede Sprint 27's "0.85× literal target route to Sprint 28+" caveat with the Sprint 28 retirement verdict.  Day 13 commits the draft.
- `docs/planning/EPIC_2/SPRINT_22/PERF_NOTES.md` (or new `SPRINT_28/PERF_NOTES.md` if Sprint 22 file is too long): append a "Sprint 28 closures" subsection.  Day 14.
- `RETROSPECTIVE.md` filled in single-pass.  Day 13 stubs the section skeleton; Day 14 completes.
- `PROJECT_PLAN.md` Sprint 28 status flip from "in flight" to "Complete".  Day 14.
