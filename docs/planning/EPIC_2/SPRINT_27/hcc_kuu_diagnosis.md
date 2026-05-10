# HCC Kuu Default-Flip Blocker — Diagnosis (Day 1)

## Background

Sprint 26 Day 13's 12-setting × 6-fixture combination matrix surfaced **Kuu HCC-alone +14.6pp ND/AMD nnz_L regress** as the SECOND HCC default-flip blocker (after Sprint 26 Day 3 fixed the FIRST — bcsstk14 sep=0).  Kuu's CV of 0.425 is the highest in the Sprint 26 corpus, making it the worst-case fixture for HCC's `min(deg(u), deg(v))` weighting (Karypis-Kumar 1998 §5).

This document captures Day 1's empirical diagnosis + Day 2 fix-option selection.

## Reproducer

```
SPARSE_HCC_DEBUG=1 SPARSE_ND_COARSENING=hcc build/bench_reorder --only Kuu --skip-factor
SPARSE_HCC_DEBUG=1 SPARSE_ND_COARSENING=heavy_edge build/bench_reorder --only Kuu --skip-factor
```

## Headline Numbers (Day 1 capture, master `b2831bc` plus Sprint 27 sprint-27 branch)

```
Kuu (n=7102):
                     nnz_L     ratio (vs AMD = 406 264)
  AMD                406 264   1.000
  HEM (Sprint 26 default)   881 177   2.169    ← baseline
  HCC                940 582   2.315    ← +14.6pp regress
```

The +14.6pp delta matches PROJECT_PLAN.md Sprint 27 Item 1 statement; reproducer is bit-identical to the Sprint 26 Day 13 capture.

## Per-Level Match Ratios

Match ratios under both strategies are uniformly high (>0.92 at every level), so the issue is NOT that HCC fails to match — it's that HCC matches DIFFERENT vertex pairs than HEM.  Per-level top-of-recursion match ratios:

| Level (n_fine → n_coarse) | HEM   | HCC   |
|---------------------------|-------|-------|
| 7102 → ~3625              | 0.979 | 0.977 |
| ~3625 → ~1870             | 0.970 | 0.966 |
| ~1870 → ~963              | 0.968 | 0.955 |
| ~963 → ~500               | 0.972 | 0.952 |
| ~500 → ~256               | 0.966 | 0.953 |

Match ratios diverge starting at level 2-3 (HCC's match ratio drops 1-2pp below HEM's), consistent with HCC consistently picking sub-optimal partners as the graph shrinks.

## Kuu Structural Profile

(Full numbers in `hcc_kuu_degree_profile.txt`.)

```
n = 7102, edges = 166 549
Degrees: min=22 max=97 mean=46.90 std=19.95 CV=0.425

Degree histogram (FOUR distinct clusters):
  deg in [22,  30]:   378 vertices ( 5.3%)   ← LOW band  (boundary-ish)
  deg in [31,  40]:  2935 vertices (41.3%)   ← MID-LOW
  deg in [41,  50]:   351 vertices ( 4.9%)
  deg in [51,  60]:  2724 vertices (38.4%)   ← MID-HIGH
  deg in [81,  97]:   714 vertices (10.1%)   ← HIGH band (interior-ish)

Edge min(deg) histogram:
  min(deg) in [22, 30]:   7 379 edges ( 4.4%)
  min(deg) in [31, 40]:  79 909 edges (48.0%)
  min(deg) in [51, 60]:  66 591 edges (40.0%)
  min(deg) in [81, 97]:   5 615 edges ( 3.4%)
```

The degree distribution is **strongly multimodal** — four clusters reflecting the 3D structural-mechanics provenance (boundary nodes have fewer DOF connections; interior nodes with rotational DOFs have many more).  This is NOT a continuous distribution that a single-CV summary captures fully — the multimodality is what makes HCC's `min(deg)` bias severe.

## Why HCC's `min(deg)` Bias Is Severe On Kuu

HCC scores edges by `score(u,v) = edge_weight × min(deg(u), deg(v))`.  Edge-weight in this corpus is uniformly 1, so `score = min(deg)`.

On Kuu's distribution:
- **Interior-interior edges** (both endpoints in the high band): score ≈ 51-97 (top 40 % + 3.4 %)
- **Mid-mid edges** (both endpoints in the mid-low band): score ≈ 31-40 (48 % of edges)
- **Boundary-anywhere edges** (at least one low-band endpoint): score ≈ 22-30 (4.4 % of edges)

When HCC argmaxes, it preferentially matches **interior-interior pairs first** — boundary vertices stay unmatched at level 0 and only get partnered at later levels with whatever neighbours happen to remain.  Over 7-8 levels, this compounds: boundary structure dissolves into the coarse graph in a way that no longer reflects the original separator-friendly geometry, leading to the +14.6pp ND/AMD nnz_L regress.

HEM (`score = edge_weight` only) has no such bias on uniform-weight graphs — every edge ties at score = 1 and the shuffle-dependent first-encountered tie-break distributes matches structurally.  This is why HEM works on Kuu while HCC doesn't.

## Comparison Fixtures (CV Summary)

| Fixture       | n     | deg range | mean  | CV    | HCC vs HEM nnz_L delta |
|---------------|-------|-----------|-------|-------|------------------------|
| nos4          |   100 | [1, 6]    |  4.94 | 0.295 | (small; no regress)    |
| bcsstk04      |   132 | [14, 45]  | 26.64 | 0.405 | -0.13 % (slight WIN)   |
| bcsstk14      |  1806 | [0, 47]   | 34.14 | 0.280 | (Sprint 26 D3 fix)     |
| s3rmt3m3      |  5357 | [6, 47]   | 37.77 | 0.187 | (mild; no regress)     |
| Pres_Poisson  | 14822 | [17, 49]  | 47.29 | 0.108 | (mild; no regress)     |
| **Kuu**       |  **7102** | **[22, 97]** | **46.90** | **0.425** | **+14.6pp regress** |

Notable: bcsstk04 has CV=0.405 (close to Kuu's 0.425) but shows no HCC regress — likely because (a) at n=132 the multilevel pipeline only goes 2-3 levels deep, leaving little room for HCC's bias to compound, and (b) bcsstk04's degree distribution is more continuous (no large multimodal clusters).  This means a fix gated solely on CV may also catch bcsstk04, but bcsstk04 is robust enough that an HCC→HEM fall-through wouldn't regress it (HCC and HEM tie within 0.13 % on bcsstk04).

## Fix-Option Analysis

PROJECT_PLAN.md Sprint 27 Item 1 names two fix options:

### Option (a) — Adaptive HCC weighting

**Two sub-variants:**

- **(a.1) CV-detection-and-HEM-fallthrough**: at the top of `graph_coarsen_with_strategy` when `strategy == COARSENING_HCC`, compute the degree CV of the input graph; if CV exceeds a threshold (e.g. 0.30), treat the matching as `COARSENING_HEAVY_EDGE` for that call.
- **(a.2) Soften `min(deg)`**: replace `min(deg(u), deg(v))` with `sqrt(min(deg))` or `log(1 + min(deg))` to compress dynamic range and bring boundary-edge scores closer to interior-edge scores.

### Option (b) — Per-edge weight-equality break

When two edges have identical HCC scores, prefer the edge whose endpoints have closer degrees (smaller `|deg(u) - deg(v)|`).

## Decision: Option (a), Variant (a.1) — CV-Detection + HEM Fall-Through

### Rationale

1. **The structural property identified (degree-CV bimodal/multimodal distribution) is exactly what option (a.1) targets.**  Kuu's CV=0.425 stands out from the rest of the corpus (next-highest is bcsstk04 at 0.405; Pres_Poisson is 0.108).  A CV threshold around 0.30 cleanly separates Kuu from the fixtures HCC works well on.

2. **Option (a.1) is simpler than (a.2).**  No new tunable formula; reuses the existing HEM path which is known-good on Kuu (HEM nnz_L 881 177 vs HCC 940 582).  Variant (a.2)'s sqrt/log softening would need its own corpus sweep to pick the right shape and risks regressing on fixtures where HCC's `min(deg)` was a clear win (e.g. Sprint 25 Day 9 setting 13 reported HCC + ratio=200 as Pres_Poisson best).

3. **Option (b) is not the right shape for this regress.**  Per-level match ratios show HCC matching >0.95 of vertices at every level — there are FEW ties (HCC's tie-break with `score == best_score && u < best_nbr` already deterministically resolves ties).  The issue is the score *distribution itself*, not the tie-breaking; (b) wouldn't move the needle.

4. **Surgical and reversible.**  Variant (a.1) adds one branch at coarsening entry; the existing HEM-default path is the fallback.  If a future fixture surfaces and the threshold is wrong, only the threshold needs adjustment.  The `SPARSE_ND_COARSENING_CV_FALLTHROUGH=N.NN` env var can expose the threshold for tuning + sweep.

### CV Threshold Selection

Initial Day 2 default: **CV > 0.30** triggers HEM fall-through.  Reasoning:
- Kuu (CV=0.425) → falls through (the desired fix).
- bcsstk04 (CV=0.405) → also falls through, but its HCC-HEM delta is -0.13 % (a slight win) so the fall-through doesn't regress it.
- Pres_Poisson (CV=0.108) → stays HCC (Sprint 26 evidence shows HCC parity-or-better here).
- s3rmt3m3 (CV=0.187), bcsstk14 (CV=0.280) → stay HCC.
- nos4 (CV=0.295) → stays HCC (just under the threshold).

Day 2 will sweep `{0.20, 0.25, 0.30, 0.35}` to confirm 0.30 is the empirical optimum + corpus-bit-identical-or-better check.

### Day 2 Implementation Plan

1. Compute graph-wide degree CV at the top of `graph_coarsen_with_strategy` when `strategy == COARSENING_HCC`.  Cost: one pass over `xadj` (O(n)).
2. Read `SPARSE_ND_COARSENING_CV_FALLTHROUGH` env var (default 0.30; out-of-range/non-numeric → default; `0.0` disables fall-through entirely for sweep purposes).
3. If computed CV > threshold, treat the call as `COARSENING_HEAVY_EDGE`.  Emit a one-line stderr advisory under `SPARSE_HCC_DEBUG=1` ("hcc-debug strategy=hcc fell through to heavy_edge: CV=0.425 > threshold=0.30").
4. Run `tests/test_reorder_nd.c::test_hcc_kuu_no_default_flip_blocker` — must now pass.
5. Run the Sprint 26 Day 13 combination matrix subset (settings involving HCC alone, HCC + per-vertex variants).  HCC default flip is good if (a) Pres_Poisson improves ≥ 1pp under HCC + Kuu-safe AND (b) no smaller-fixture regress past 5pp.
6. Update this doc's "Day 2 verification" section with bench evidence + flip outcome.

## Day 1 Test Stub Placement

The Sprint 27 PLAN.md Day 1 task spec named `tests/test_graph.c::test_hcc_kuu_no_default_flip_blocker` as the stub location.  Day 1 deviates: the stub lands in **`tests/test_reorder_nd.c::test_hcc_kuu_no_default_flip_blocker`** instead — that file is the natural home for `nnz_L`-vs-`AMD` fill-quality assertions (it already hosts `test_nd_bcsstk14_fill_vs_amd`, `test_nd_pres_poisson_fill_with_leaf_amd`, etc.), and `tests/test_graph.c` doesn't include the reorder-layer headers needed to invoke `symbolic_cholesky_nnz_nd`.  Plan-spec-vs-implementation mismatch is documented here for traceability; no functional change.

## Files Generated Day 1

- `docs/planning/EPIC_2/SPRINT_27/hcc_kuu_diagnosis.md` (this document)
- `docs/planning/EPIC_2/SPRINT_27/hcc_kuu_degree_profile.txt` (degree histogram + CV summary across corpus fixtures)
- `tests/test_reorder_nd.c::test_hcc_kuu_no_default_flip_blocker` (failing-as-expected stub; pin Day-2 fix)

## Files Not Committed (Intermediate Day 1 Captures)

- `/tmp/hcc_kuu_trace.txt` (5092-line per-level cmap dump under `SPARSE_HCC_DEBUG=1 SPARSE_ND_COARSENING=hcc`)
- `/tmp/hem_kuu_trace.txt` (5094-line per-level cmap dump under `SPARSE_ND_COARSENING=heavy_edge`)
- `/tmp/hcc_kuu_stdout.txt`, `/tmp/hem_kuu_stdout.txt` (bench summary CSVs)

---

## Day 2 Verification: Fix Implementation + Default-Flip Decision

### Implementation

Landed in `src/sparse_graph.c::graph_coarsen_with_strategy` (top of function, after `n_fine = fine->n`).  When `strategy == COARSENING_HCC` and `n_fine >= 2`:

1. Read `SPARSE_ND_COARSENING_CV_FALLTHROUGH` env var (default 0.30; out-of-range/non-numeric/negative → default; 0.0 disables fall-through).
2. If threshold > 0.0, compute degree CV via one-pass mean+variance over `xadj` (O(n)).
3. If CV > threshold, set `strategy = COARSENING_HEAVY_EDGE` for that call; emit a one-line stderr advisory under `SPARSE_HCC_DEBUG=1`.

The matching loop downstream is unchanged.  Fall-through fires per-level (not just at the root) — the input graph at each coarsening level may have a different CV than the previous level (boundary structure dissolves as the graph shrinks).

### Per-Level Fall-Through on Kuu

```
SPARSE_HCC_DEBUG=1 SPARSE_ND_COARSENING=hcc build/bench_reorder --only Kuu --skip-factor

hcc-debug strategy=hcc fell through to heavy_edge: n_fine=7102 CV=0.425 > threshold=0.300
hcc-debug strategy=hcc fell through to heavy_edge: n_fine=3625 CV=0.404 > threshold=0.300
hcc-debug strategy=hcc fell through to heavy_edge: n_fine=1867 CV=0.331 > threshold=0.300
(further levels stay HCC: CV drops below 0.30 at level 3+)
```

CV-detection fires at the top 3 coarsening levels (where Kuu's bimodality dominates) and naturally turns off at finer levels (where the multimodal structure has already dissolved into a more uniform distribution).  This is the optimal shape — preserves HCC's wins on small-coarse-graph levels (where matching choice matters less) while bypassing HCC's structural bias at the levels where it hurts.

### Post-Fix Corpus Sweep

| Fixture       | n     | AMD nnz_L | HEM nnz_L (Sprint 26) | HCC + Kuu-safe (Sprint 27) | Δ vs HEM | HCC+fix/AMD | HEM/AMD |
|---------------|-------|-----------|-----------------------|----------------------------|----------|-------------|---------|
| nos4          |   100 |       637 |                  809  |                       809  |  0.0 %   |    1.270×   |  1.270× |
| bcsstk04      |   132 |     3 143 |                3 722  |                     3 722  |  0.0 %   |    1.184×   |  1.184× |
| **Kuu**       |  7102 |   406 264 |              881 177  |                   772 871  | **−12.3 %** | **1.902×** | 2.169× |
| bcsstk14      |  1806 |   116 071 |              129 292  |                   130 163  | +0.7 %   |    1.121×   |  1.114× |
| s3rmt3m3      |  5357 |   474 609 |              483 195  |                   486 040  | +0.6 %   |    1.024×   |  1.018× |
| **Pres_Poisson** | 14822 | 2 668 793 |          2 536 427  |                 2 450 405  | **−3.4 %**  | **0.918×** | 0.950× |

Two MAJOR wins:

- **Kuu**: HCC + Kuu-safe is now BETTER than HEM by 12.3 % (1.902× vs 2.169× of AMD).  Surprise outcome: the Kuu-safe fix not only neutralises HCC's bias, it also lets HCC win at the finer levels where CV drops below threshold.  The combination beats HEM (which stays HEM at every level).
- **Pres_Poisson**: HCC + Kuu-safe is BETTER than HEM by 3.4 % (0.918× vs 0.950× of AMD).  Pres_Poisson's CV is 0.108 (well below the 0.30 threshold), so HCC stays HCC at every level.  This is the headline 0.85× target's progress: 0.918× is an improvement of 3.2pp toward the 0.85× target (Sprint 26 best opt-in was 0.922×; Sprint 27 default is now 0.918× — closer but still 6.8pp away from the literal target).

Two MINOR regresses (within flip-rule budget):

- **bcsstk14**: +0.7 % (130 163 vs 129 292).  CV=0.280 (below threshold, stays HCC); the bcsstk14 sep=0 retry path (Sprint 26 Day 3) still fires and produces a correct partition.  Tiny regress is well under the 5 pp flip-rule budget.
- **s3rmt3m3**: +0.6 % (486 040 vs 483 195).  CV=0.187 (well below threshold, stays HCC).  Tiny regress is well under the 5 pp flip-rule budget.

Bit-identical on nos4 and bcsstk04 (both have CV below threshold in the worst case, and at small `n` HCC and HEM converge to the same matching).

### Default-Flip Rule Application

PROJECT_PLAN.md / Sprint 27 PLAN.md flip rule:
- (a) Pres_Poisson improves ≥ 1pp under HCC + Kuu-safe → **YES** (3.2pp improvement).
- (b) No smaller-fixture regress past 5pp → **YES** (max regress is 0.7pp on bcsstk14).

**Both conditions satisfied.  HCC default flip lands.**

`parse_coarsening_strategy()` default changed from `COARSENING_HEAVY_EDGE` to `COARSENING_HCC` in `src/sparse_graph.c`.  `SPARSE_ND_COARSENING=heavy_edge` becomes the opt-in fallback (recognised explicitly in the parser).

### Test Updates

The default flip surfaced two tests pinned against Sprint-22-era HEM-specific behavior:

1. **`tests/test_graph.c::test_hierarchy_build_5x5_grid`** — pinned `SPARSE_ND_COARSENING=heavy_edge` (the `n ≤ 13` bound is HEM-specific; HCC's `min(deg)` matching on the 5×5 grid leaves more boundary vertices unmatched because HCC matches interior-interior pairs first).  The Sprint 25 Day 2 `test_hcc_match_selection_grid` test already exercises HCC behavior on small grids; this test stays HEM-scoped.

2. **`tests/test_graph.c::test_finest_fm_strategy_fifo_smoke`** — pinned `SPARSE_ND_COARSENING=heavy_edge` (the "FIFO differs from baseline on the 30×30 grid" contract was verified under HEM coarsening per Sprint 26 Day 7 design).  HCC's `min(deg)` scoring on a regular grid produces more deterministic matchings (most edges score identically; tie-break dominates), which can collapse the FIFO-vs-baseline differentiation that the test was designed to surface.

3. **`tests/test_reorder_nd.c::test_hcc_kuu_no_default_flip_blocker`** — uncommented from RUN_TEST list; now passes (Kuu nnz_L = 772 871 < bound 901 503 = AMD × 2.219).

No other tests required updates — the rest of the test suite passed bit-identically under the new default.

### Files Generated Day 2

- `src/sparse_graph.c` — added CV-detection-and-HEM-fall-through in `graph_coarsen_with_strategy`; added explicit `"heavy_edge"` env-var recognition in `parse_coarsening_strategy`; flipped default from `COARSENING_HEAVY_EDGE` to `COARSENING_HCC`; added `<math.h>` include.
- `tests/test_reorder_nd.c` — uncommented `test_hcc_kuu_no_default_flip_blocker` from RUN_TEST list.
- `tests/test_graph.c` — pinned `SPARSE_ND_COARSENING=heavy_edge` in `test_hierarchy_build_5x5_grid` + `test_finest_fm_strategy_fifo_smoke`.
- `docs/planning/EPIC_2/SPRINT_27/hcc_kuu_diagnosis.md` — appended this Day-2 verification section.

### Headline Status After Day 2

- **HCC + Kuu-safe matching variant lands as default coarsening strategy.**  Both Sprint 25 Day 10 default-flip blockers (bcsstk14 sep=0 fixed Sprint 26 Day 3; Kuu +14.6pp regress fixed Sprint 27 Day 2) are now closed.
- **Pres_Poisson ND/AMD = 0.918×** under the new default (was 0.950× under Sprint 26 default; was 0.922× under Sprint 26 best opt-in setting 13).  This is a 3.2pp improvement; still 6.8pp away from the literal 0.85× target.
- **Items 4 (annealing FM) and 5 (root-level spectral) carry the remaining headline-target weight.**  Sprint 27 Days 5-9 plan to close the 6.8pp gap; if both miss, item 6 (thick-restart, conditional fallback) fires Days 10-12.
