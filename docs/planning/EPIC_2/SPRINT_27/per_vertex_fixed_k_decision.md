# Per-Vertex Fixed-K Selection Mode + 3-Scheme Sweep — Decision (Sprint 27 Day 4)

## Background

Sprint 26 Day 12 implemented three per-vertex separator-lift weight schemes:

- `SPARSE_ND_SEP_LIFT_STRATEGY=per_vertex` (HYBRID; cross_deg-priority + balance tie-break)
- `SPARSE_ND_SEP_LIFT_STRATEGY=per_vertex_balance` (BALANCE; balance-priority + cross_deg tie-break)
- `SPARSE_ND_SEP_LIFT_STRATEGY=per_vertex_degree` (DEGREE; low-total-degree-priority + balance tie-break)

Sprint 26 Day 12's empirical finding was that **all three weight schemes converge to bit-identical outputs on 5 of 6 fixtures** because the dynamic-K + 70/30 balance gate dominates the score formula — the gate fires early enough that the score formula doesn't get to differentiate vertices before the lift loop terminates.

Sprint 27 PLAN.md Day 4 adds a **fixed-K** termination mode that bypasses the 70/30 gate, terminating after exactly K = `min(boundary_count[0], boundary_count[1])` lifts regardless of balance state.  The orthogonal `SPARSE_ND_SEP_LIFT_WEIGHT={hybrid (default), balance, degree}` axis stacks with fixed-K so the three weight schemes can express their character.

## Implementation

### `src/sparse_graph.c`

1. **New strategy enum value** `SEP_LIFT_PER_VERTEX_FIXED_K` (5).  Recognised by `parse_sep_lift_strategy()` as `SPARSE_ND_SEP_LIFT_STRATEGY=per_vertex_fixed_k`.
2. **New weight enum** `sep_lift_weight_t` with values `SEP_LIFT_WEIGHT_{HYBRID, BALANCE, DEGREE}`.  Recognised by new `parse_sep_lift_weight()` reading `SPARSE_ND_SEP_LIFT_WEIGHT` (default hybrid).
3. **Score-formula resolution**: the four legacy `per_vertex_*` strategies hardcode their weight schemes (preserves Sprint 26 advisory env-var contract).  Only `SEP_LIFT_PER_VERTEX_FIXED_K` reads `SPARSE_ND_SEP_LIFT_WEIGHT`.
4. **Termination predicate**: legacy strategies use the dynamic-K + 70/30 balance gate (Sprint 26 Day 10 contract).  `SEP_LIFT_PER_VERTEX_FIXED_K` uses fixed-K = min(boundary_count[0], boundary_count[1]) — the lift loop terminates after exactly K iterations regardless of balance state.

### `tests/test_graph.c`

Added `test_per_vertex_fixed_k_differs_from_dynamic_k` (RUN_TEST registered): runs `sparse_graph_partition` on a 30×30 grid (n=900) under `SPARSE_ND_SEP_LIFT_STRATEGY=per_vertex` (dynamic-K hybrid) and `SPARSE_ND_SEP_LIFT_STRATEGY=per_vertex_fixed_k` (fixed-K, default hybrid weight); asserts the resulting partitions differ (separator count or `part_io[]` byte-level).

## Day 4 Test Placement

Sprint 27 PLAN.md Day 4 task 5 named **Pres_Poisson** as the fixture for the smoke test.  At n=14 822, Pres_Poisson takes ~7 s per partition call — too slow for unit-test scope (would dominate the `make test` runtime).  Day 4 deviates: uses the existing 30×30 grid (n=900, ~ms per partition) which Sprint 26 Day 7 also used for env-var differentiation tests (`test_finest_fm_strategy_fifo_smoke`).  Plan-spec-vs-implementation deviation noted here for traceability; no functional change.

## Sweep: 3 weight schemes × 6 fixtures × {fixed-K, dynamic-K}

(Captured in `per_vertex_fixed_k_sweep.txt`; ND nnz_L only — wall-time deltas omitted from this table for clarity.)

| Fixture | Sprint 27 default (smaller_weight) | dyn-K hybrid (per_vertex) | dyn-K balance | dyn-K degree | **fixed-K hybrid** | fixed-K balance | fixed-K degree |
|---|---:|---:|---:|---:|---:|---:|---:|
| nos4 |        637 |        637 |        637 |        637 |        637 |        637 |        637 |
| bcsstk04 |    3 722 |      3 549 |      3 549 |      3 549 |      3 679 |      4 469 |      4 613 |
| **Kuu** | **764 664** |    724 358 |    728 166 |    726 362 | **499 127 win** |  1 035 335 |  2 920 633 |
| bcsstk14 |  130 422 |    152 900 |    152 900 |    152 900 |    282 996 |    214 778 |    277 394 |
| s3rmt3m3 |  487 832 |    621 658 |    621 658 |    621 658 |    830 461 |    526 194 |  1 032 359 |
| Pres_Poisson | 2 462 201 |  3 256 763 |  3 256 763 |  3 256 763 |  3 742 309 |  3 265 627 |  5 573 256 |

### Key findings

1. **Sprint 26 Day 12's hypothesis VALIDATED.**  Under dynamic-K, the three weight schemes are bit-identical on 5 of 6 fixtures (nos4, bcsstk04, bcsstk14, s3rmt3m3, Pres_Poisson all show identical 637 / 3549 / 152900 / 621658 / 3256763 across hybrid/balance/degree — the 70/30 gate dominates).  Only Kuu shows a tiny differentiation under dynamic-K (724358 / 728166 / 726362 — sub-1pp).
2. **Under fixed-K, the three weight schemes massively differentiate.**  bcsstk04 spread is ±15-25%; Kuu spread is from 499K to 2.9M (6×); Pres_Poisson spread is 50-126%.
3. **Fixed-K + hybrid is the headline win for Kuu**: 499 127 vs 764 664 default = **−34.7 % nnz_L on Kuu** (ratio 1.229× of AMD; was 1.902× under Day 2's HCC default).  Best Kuu opt-in by a wide margin.
4. **Fixed-K is catastrophic for regular meshes** (Pres_Poisson +52 % under hybrid, +33 % under balance, +126 % under degree) — confirms that the 70/30 gate's early termination is *correct* for mesh-like fixtures and only *suboptimal* for bimodal-degree fixtures.

### Per-fixture-class advisory

| Fixture class | Best fixed-K opt-in | Default behaviour |
|---|---|---|
| Regular meshes (Pres_Poisson, s3rmt3m3, bcsstk14) | n/a — fixed-K regresses | Stay default `smaller_weight` |
| Tiny (n ≤ 200; nos4, bcsstk04) | Indifferent (637 / 3679 — small) | Stay default |
| **Bimodal-degree solid mechanics (Kuu)** | **`per_vertex_fixed_k` × `hybrid` (-34.7 %)** | Stay default; advisory opt-in |

The advisory pattern matches Sprint 27 Day 3's per-fixture-class advisory for `nd_base_threshold = 256` on Kuu.  Combining the two opt-ins (`SPARSE_ND_SEP_LIFT_STRATEGY=per_vertex_fixed_k SPARSE_ND_SEP_LIFT_WEIGHT=hybrid` plus `--nd-threshold 256`) is a natural Sprint 28+ exploration if Kuu workloads become a corpus priority.

## Decision: Stay At Default `smaller_weight`; Ship Fixed-K As Advisory

### Why not flip default to `per_vertex_fixed_k + hybrid`

Pres_Poisson is the headline fixture and would regress +52 % under fixed-K hybrid (2 462 201 → 3 742 309).  This decisively fails any flip-rule formulation.  Pres_Poisson + Kuu trade-off is the inverse of what the corpus default needs — fixed-K is bimodal-class-specific.

### What ships

1. `SPARSE_ND_SEP_LIFT_STRATEGY=per_vertex_fixed_k` env-var value — opt-in; default behaviour unchanged.
2. `SPARSE_ND_SEP_LIFT_WEIGHT={hybrid (default), balance, degree}` env var — orthogonal weight axis for the fixed-K mode.
3. `tests/test_graph.c::test_per_vertex_fixed_k_differs_from_dynamic_k` smoke test pinning the fixed-K-vs-dynamic-K differentiation.
4. Per-fixture-class advisory documented here + (Day 13) in `docs/algorithm.md` Sprint 27 closure subsection: workloads dominated by bimodal-degree solid-mechanics SPDs (Kuu's class) get a `~-35 %` opt-in win via `SPARSE_ND_SEP_LIFT_STRATEGY=per_vertex_fixed_k`.

### Sprint 26 Day 12's open question — answered

Sprint 26 Day 12 documented the three-weight-scheme convergence as "the 70/30 balance gate dominates the score formula" but didn't have a way to test the hypothesis (the dynamic-K mode IS the gate).  Sprint 27 Day 4's fixed-K mode is the natural test: removing the gate makes the three weight schemes diverge by 6× on Kuu and 1.5-2× on smaller fixtures.  **Hypothesis confirmed.**

## Files Generated

- `docs/planning/EPIC_2/SPRINT_27/per_vertex_fixed_k_sweep.txt` — raw 3-weight × 6-fixture × {fixed-K, dynamic-K reference} sweep
- `docs/planning/EPIC_2/SPRINT_27/per_vertex_fixed_k_decision.md` — this document
- `src/sparse_graph.c` — `SEP_LIFT_PER_VERTEX_FIXED_K` enum value; `sep_lift_weight_t` enum + `parse_sep_lift_weight()`; weight-scheme switch refactored to read from `weight` variable; termination-predicate split (dynamic-K 70/30 gate vs fixed-K = min(boundary_count) cap)
- `tests/test_graph.c` — `test_per_vertex_fixed_k_differs_from_dynamic_k` smoke test

## Headline Status After Day 4

- Fixed-K mode lands behind `SPARSE_ND_SEP_LIFT_STRATEGY=per_vertex_fixed_k`; default behaviour unchanged.
- **Kuu opt-in win: −34.7 % nnz_L** under `per_vertex_fixed_k` × `hybrid` (1.229× of AMD; was 1.902× under Sprint 27 Day 2 default).
- Pres_Poisson default unchanged (Day 4 doesn't move the headline; items 4-6 carry the 0.85× target).
- Sprint 26 Day 12's "70/30 gate dominates the score formula" hypothesis empirically validated.
- Quality gates clean (format, lint, test, wall-check).
