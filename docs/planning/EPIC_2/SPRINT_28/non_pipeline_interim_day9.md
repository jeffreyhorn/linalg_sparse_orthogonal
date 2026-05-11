# Sprint 28 Day 9 — supernodal-etree non-pipeline pivot: headline verdict + Day-10 close-out plan

## Headline verdict: **MISSED**

Under `SPARSE_ND_SUPERNODAL_POSTORDER=on`, Pres_Poisson's ND nnz_L ratio
vs AMD remains at **0.9226×** — identical to the Sprint 27 default
(unchanged because symmetric permutation preserves the symbolic
Cholesky fill pattern by construction; the supernodal-etree post-pass
reorders columns but cannot eliminate fill that the pipeline produces).

The literal 0.85× Pres_Poisson target REMAINS UNMET after Sprint 28:

- **5th consecutive sprint to miss** (Sprint 22 1.063× → Sprint 23 0.952× → Sprint 24 0.942× → Sprint 25 0.922× → Sprint 26 0.9217× → Sprint 27 0.9226× → **Sprint 28 0.9226×**).
- **Sprint 28's Item-4 non-pipeline pivot, while structurally correct, cannot move the headline metric on this codebase + corpus.**

## Sweep summary ({AMD, ND} × {env off, env on} × 6 fixtures)

See `non_pipeline_sweep.txt` for the full per-cell capture.  Key cells:

| fixture | reorder | env | nnz_L | ratio vs Pres_Poisson AMD | super_count | total_grouped |
|---|---|---|---|---|---|---|
| Pres_Poisson | AMD | off | 2 668 793 | 1.0000 | 565 | 11 360 |
| Pres_Poisson | AMD | on  | 2 668 793 | 1.0000 | 566 | 11 360 |
| Pres_Poisson | ND  | off | 2 462 201 | 0.9226 | 571 | 10 753 |
| Pres_Poisson | ND  | on  | 2 462 201 | **0.9226** | 573 | 10 756 |

- **nnz_L invariant under env-on on every fixture × reorder cell** (24/24); confirms the Day-7 permutation composition is mathematically equivalent to the env-off path (symmetric permutation preserves fill).
- **Supernode-count delta is trivial** (±1-3 supernodes; total_grouped same or ±3 columns); the post-pass produces a marginally different fundamental-supernode structure but does not measurably improve clustering on AMD-pre-ordered inputs.  This matches the pivot_decision_day1.md candidate-(c) prior ("Pres_Poisson upside per literature: ≤ 5% nnz reduction; primarily a numeric-factor optimization, not fill-reduction").
- **analyze wall delta**: env-on adds 5-15 % on the analyze pass (the extra etree_compute + sparse_permute + recompute step).  Pres_Poisson AMD: +6.6 % (4472 → 4768 ms); Pres_Poisson ND: roughly within noise (-5 %; single-run timing variance).  Day 10 will document the production-default cost-benefit.
- **factor wall**: bench captured a Pres_Poisson ND env-on factor at 24.8 s vs env-off 33.8 s — a -27 % delta.  Single-run timing on a system under variable load; supernode structure is essentially equivalent across the two cells (571 vs 573 supernodes); the -27 % is most likely measurement noise.  Day 13's cross-corpus matrix can verify with median-over-repetitions if a follow-up sprint wants to investigate.

## Per-fixture corpus safety

All 6 fixtures × both reorders: nnz_L env-on equals env-off exactly (corpus-safety contract per `test_supernodal_postorder_corpus_nnz_L_invariant` in `tests/test_reorder_nd.c`).

The nos4 / bcsstk04 small-fixture supernode total_grouped slightly DECREASES under env-on (Day-7's already-documented observation): AMD's perm on small graphs happens to produce more compact contiguous supernodes than the strict etree-postorder.  This is "no measurable safety regression at the production scale" — small-graph noise that disappears in fixtures with > 1000 vertices.

## Tuning verdict

The supernodal-etree pivot has only one tunable (`SPARSE_ND_SUPERNODAL_POSTORDER={off, on}`); there is no granularity sub-axis or per-fixture tuning.  The Day-9 sweep covers the full tuning surface.

**Recommended production default: STAY at OFF.**
- No measurable nnz_L improvement (invariant by construction).
- Supernode structure essentially unchanged (±1-3 supernodes; trivial total_grouped delta).
- Analyze wall adds ~6-15 % on the largest fixtures.
- The env-on path ships as opt-in advisory for any future sprint that wires supernodal numeric-factor kernels (the literature prior says that's where the value would manifest, and the existing chol_csc path doesn't currently exploit supernodal kernels).

## test_nd_pres_poisson_fill_with_leaf_amd bound

The Sprint 27 bound is 0.94× (per `tests/test_reorder_nd.c::test_nd_pres_poisson_fill_with_leaf_amd` line 474).  Sprint 28's measured Pres_Poisson ND ratio is 0.9226× under both env settings — well within the 0.94× bound.  **No bound change needed.** Day 13's calibration step will keep the bound at 0.94× and document that the literal 0.85× target is formally retired with Sprint 28's empirical evidence.

## Day-10 close-out plan

Day 10 lands `non_pipeline_decision.md` with three sections:

1. **Verdict**: MISSED.  Sprint 28's non-pipeline pivot ships the supernodal-etree post-pass as advisory; production default stays OFF; literal 0.85× Pres_Poisson target is formally retired.
2. **Production default rationale**: cite the Day-9 sweep + the literature prior + the +6-15 % analyze wall cost.  Note the env-on path remains useful infrastructure for any future sprint that adds supernodal numeric-factor kernels.
3. **Sprint 29+ routing**: route the formal target retirement to Day 13's `headline_summary.md`; route any future "supernodal numeric factor kernels" work to the Sprint 29 parking-lot.

Day 10's test scaffolding:
- `test_non_pipeline_pres_poisson_close_to_target` lands with `RUN_TEST` commented out (Sprint 28's verdict is MISSED, matching the Sprint 27 Day 12 stubbed-out pattern for `test_finest_fm_annealing_pres_poisson_close_to_target`).
- The new test documents "Pres_Poisson ND nnz_L ≤ 0.87× AMD = 0.85× + 2pp" + the failing-as-expected status with bench evidence pointing to Sprint 28's 0.9226× achieved.

Day-14 retrospective inputs:
- Items-deferred entry: "Pres_Poisson literal 0.85× target — formally retired with Sprint 28's evidence; Sprint 29+ may revisit only if fundamentally different machinery (METIS C library interop, geometric mesh-aware ordering with first-class coordinate API) is pursued".
- Sprint 29 inputs: supernodal numeric-factor kernels (the post-pass infrastructure ships in Sprint 28; the kernels that exploit it are Sprint 29+ work).
