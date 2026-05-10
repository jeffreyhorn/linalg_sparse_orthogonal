# Sprint 28 Day 1 — Non-Pipeline-Level Pivot Decision

## Decision

**Pick (c) supernodal-etree reordering as Item 4's non-pipeline-level
pivot.**  Sprint 28 absorbs the empirical-floor calibration from
fallback (d) into Day 13's Item 5 work — the literal 0.85×
Pres_Poisson target is retired with documented 5-sprint evidence
that the multilevel pipeline + leaf-AMD reaches the empirical
floor, and any non-pipeline approach with literature-backed Pres_Poisson
upside is structurally blocked (b: needs coordinates the corpus
lacks) or doesn't move nnz_L by the required magnitude (c: ≤ 5pp,
gap is 7.3pp).  Sprint 28 ships (c) as the worked non-pipeline-level
pivot — new infrastructure that operates on the elimination tree
abstraction (post-AMD/ND ordering) rather than the partition graph,
demonstrably non-pipeline-level even though it cannot close the
literal target alone.

The fallback (d) "all three look infeasible" path is **not selected**:
all three candidates are feasible within Item 4's 60-hour budget per
the LOC estimates below.  The right framing is "all three are
feasible but none has a credible path to closing the literal 0.85×
target on Pres_Poisson; pick the one that best fits the Sprint 28
charter (non-pipeline-level pivot) and ships useful infrastructure".

This pattern matches Sprint 26 Day 9's `geometric_cut_design.md`
honest-rejection precedent: when the empirical evidence contradicts
the original-design premise, document the rejection clearly and
re-allocate the budget to a worked alternative.

## Background — Empirical priors

Sprint 22-27 cumulative attempt at the 0.85× Pres_Poisson literal
target landed at 0.923× of AMD nnz_L (-7.3pp from target; 5th
consecutive sprint to miss).  Sprint 27 Day 13's 24-setting × 6-fixture
cross-corpus matrix surfaced two key empirical priors that constrain
Sprint 28's pivot choice:

1. **The multilevel pipeline + leaf-AMD reaches the empirical floor
   on Pres_Poisson.**  Three independent algorithmic axes acting at
   the coarsening (HCC), intermediate-FM, coarsest-bisection, and
   FINEST-FM levels (across Sprints 22-27, ~200 measurements + 24
   combination matrix) all wash out individually or actively
   regress.  The `headline_summary.md` Sprint 27 verdict: "no
   combination of Sprint 27's advisory axes lands Pres_Poisson ≤
   0.85×".

2. **Pres_Poisson is a high-order FE-mesh, not a 2D grid.**  Sprint
   26 Day 9 (`geometric_cut_design.md`) measured Pres_Poisson with
   mean degree 47.3 and CV=0.108 (most regular fixture by
   adjacency variance); the matrix represents a P2/P3 finite-element
   discretization of a 2D Poisson problem on a triangulated mesh.
   The corpus's `.mtx` file ships only the matrix; no vertex
   coordinates are included.

These priors set the constraint envelope for Sprint 28's pivot
choice.

## Per-candidate dossier

| Candidate | LOC estimate | Pres_Poisson literature upside | Charter fit (non-pipeline) | Risk |
|---|---:|---:|---|---|
| **(a) METIS-style multi-matching coarsening** | 200-350 | 5-15pp (Karypis-Kumar 1998 §5.4) | **Low** — still pipeline-level (K parallel coarsenings) | Wall blowup ~K×; Sprint 27 evidence says pipeline-level intervention doesn't move this fixture |
| **(b) Geometric domain decomposition** | 400-600 | 10-30pp on regular FE meshes (Hendrickson-Leland 1995) | **High** — operates on coordinates, not adjacency | **Structural blocker:** corpus has no coordinates for Pres_Poisson; coordinate synthesis from Laplacian spectrum is essentially Sprint 27's root-level spectral approach (Sprint 27 Day 9: regressed +2.3pp) |
| **(c) Supernodal-etree reordering** | 150-250 | ≤ 5pp on nnz_L (Liu 1990; Davis 2006 §6.5) — primarily a numeric-factor optimization | **High** — operates on the elimination tree post-AMD/ND ordering | Cannot close 7.3pp gap on its own; primary value is numeric-factor wall reduction not nnz_L reduction |

### Candidate (a) METIS-style multi-matching coarsening

**Integration surface (per Day-1 codebase research):**
- `src/sparse_graph.c::graph_coarsen_with_strategy` lines 739-796 (391 LOC) — current single-pass HCC vs HEM matching dispatch with CV-fallthrough adaptive weighting (Sprint 27 Day 2)
- `src/sparse_graph.c::sparse_graph_hierarchy_build` lines 1078-1246 (169 LOC) — per-level coarsening loop with seed perturbation `seed + (uint32_t)level` (line 1177)
- `src/sparse_graph.c::graph_bisect_coarsest_spectral` lines 1549-1668 (120 LOC) — re-usable per-matching trial-bisection scoring path (calls `graph_build_laplacian` → `sparse_eigs_sym` for the scoring metric)

**LOC estimate:** 200-350 (matching-loop unroll for K parallel matchings + per-matching trial-bisection scoring + best-cut selection + env-var gates `SPARSE_ND_MULTI_MATCHING={off,on}` + `SPARSE_ND_MULTI_MATCHING_K=3`).

**Pres_Poisson upside per literature:** Karypis-Kumar 1998 §5.4 ("Multiple-Matching Coarsening") cites 5-15% nnz reduction on FE meshes vs single-matching.  This translates to roughly 5-15pp on the ND/AMD ratio scale — plausibly enough to close the 7.3pp gap.

**Charter fit:** **Low.**  Multi-matching is fundamentally pipeline-level — it runs K parallel multilevel coarsenings and picks the best.  Sprint 27 retrospective lesson #3: "Sprint 28 should NOT replicate Sprint 27's algorithmic-axis exploration on Pres_Poisson.  The empirical evidence is conclusive that pipeline-level interventions don't move this fixture."  Multi-matching is exactly the kind of pipeline-level intervention Sprint 27's evidence rejects.  The K=3 matchings (HEM, HCC, random-shuffle) may all converge to similar cuts on Pres_Poisson because the graph's high regularity (CV=0.108) doesn't give matching choices much room to differentiate.

**Risk:** Wall blowup ~K× per coarsening level (Sprint 27 Pres_Poisson default 10 s × 3 = 30 s — still under the 70.5 s wall-check ceiling but consumes Sprint 27's wall headroom).  Worse risk: even if it lands, the Sprint 27 empirical evidence suggests it lands neutrally on Pres_Poisson.  Shipping a 200-350 LOC feature for "marginal-or-zero Pres_Poisson improvement + wall regression" is a poor budget choice.

### Candidate (b) Geometric domain decomposition

**Integration surface:**
- New public API `sparse_reorder_nd_with_coords(A, coords, opts, perm)` — Pres_Poisson `.mtx` file does NOT ship coordinates; the API would have no caller in the existing corpus
- `sparse_eigs_sym()` (Sprint 20 Day 6 / `include/sparse_eigs.h:524`) for principal-axis projection via Lanczos on coordinate covariance OR Laplacian (the two approaches are NOT equivalent — covariance-PCA on coordinates captures geometric structure; Laplacian-spectral captures graph structure, which Sprint 27 already exhausted)
- `src/sparse_graph.c::graph_bisect_coarsest_spectral` lines 1549-1668 — re-usable for the spectral-projection step IF we use Laplacian-spectral (degenerate to Sprint 27's root-level spectral approach)
- `graph_edge_separator_to_vertex_separator` lines 2706-2850 — re-usable for the lift-edge-as-separator step

**LOC estimate:** 400-600 (new public API + coordinate-input plumbing + PCA-projection helper + recursive call with coordinate subsetting + auto-mode coordinate detection).

**Pres_Poisson upside per literature:** Hendrickson-Leland 1995 ("Multilevel algorithms for partitioning irregular graphs" §3 "Geometric methods") cites 10-30% nnz reduction on regular FE meshes — promising magnitude but with caveats.  Pres_Poisson's high-order FE structure (mean degree 47.3) means the principal-axis projection still tracks the dominant geometric direction but couples each vertex to many neighbours; the cut produced by median-axis-bisection lifts a wider boundary than for low-order grids.

**Charter fit:** **High** — operates on coordinates, not the partition graph.  Genuinely non-pipeline-level.

**Risk: structural blocker.**  Pres_Poisson's `.mtx` corpus file does NOT include coordinates.  Three options for obtaining them:

1. **Synthesize coordinates from the Laplacian spectrum** (Hall 1970 / Koren 2003 spectral graph drawing): compute the two smallest non-trivial Laplacian eigenvectors, project vertices into 2D.  But this is essentially Sprint 27 Day 9's root-level spectral approach with two eigenvectors instead of one.  Sprint 27 Day 9 found root-level spectral REGRESSED Pres_Poisson +2.3pp; going from 1 eigenvector to 2 is a marginal change unlikely to flip from regress to close.

2. **Acquire coordinates from upstream sources** (the SuiteSparse Pres_Poisson originator, if locatable; recreate the FE mesh from a Pres_Poisson PDE specification).  External-dependency risk; not reproducible across users; not in the Sprint 28 budget.

3. **Synthesize coordinates from a hand-crafted 2D embedding** (e.g. force-directed layout + truncation).  Heuristic-on-heuristic; unlikely to add fundamental information beyond what the graph adjacency already encodes.

Option (1) collapses to Sprint 27's root-level spectral path which already regressed.  Options (2) and (3) introduce dependencies / heuristics outside the Sprint 28 charter.  The structural conclusion: **(b)'s theoretical case for Pres_Poisson does not survive the corpus reality**.

### Candidate (c) Supernodal-etree reordering

**Integration surface (per Day-1 codebase research):**
- `src/sparse_chol_csc.c::chol_csc_detect_supernodes` lines 1258-1289 (32 LOC) — already exists; outputs `super_starts`, `super_sizes`, `count`; no postorder; iterates columns 0..n via `columns_in_same_supernode()` helper
- `src/sparse_analysis.c::sparse_analyze` lines 27-205 (179 LOC) — already computes etree (`analysis->etree`, parent-pointer array, line 117) and postorder (`analysis->postorder`, line 118) via `sparse_etree_compute` + `sparse_etree_postorder` (lines 125-132)
- `include/sparse_analysis.h` lines 113-116 — `sparse_analysis_t.etree` and `.postorder` are public fields (length n each)
- Insertion point: post-pass between `sparse_analyze()` completion and `sparse_factor_numeric()` call (line 278 in `sparse_analysis.c::sparse_factor_numeric` reads `analysis->perm`); compose a supernode-grouping permutation with the existing AMD/ND `perm` to ensure supernodes appear contiguously in the final order

**LOC estimate:** 150-250 (supernode-aware postorder helper + composition with the existing `analysis->perm` + env-var gate `SPARSE_ND_SUPERNODAL_POSTORDER={off,on}` + corpus-safety tests).

**Pres_Poisson upside per literature:** Liu 1990 ("The role of elimination trees in sparse factorization") + Davis 2006 §6.5 ("Postordering") cite ≤ 5% nnz reduction from etree postorder.  Supernodal-etree reordering's PRIMARY value is numeric-factor performance (dense-block kernels' efficiency on contiguous supernodes via batched BLAS calls), not fill-reduction.  **Cannot close 7.3pp gap on Pres_Poisson alone.**

**Charter fit:** **High** — operates on the elimination tree abstraction (post-AMD/ND ordering).  Genuinely non-pipeline-level: the partition graph + multilevel coarsening + FM is unchanged; the new code path runs after `sparse_analyze` produces the etree and reorders within supernodes only.

**Risk:** The headline-target gap (7.3pp on Pres_Poisson) is approximately 1.5× the literature-cited maximum (≤ 5pp).  (c) cannot reach the literal target on its own; Sprint 28 must explicitly retire the target as part of Item 5.

## Why (c) is the right pick despite the headline-target shortfall

Three converging reasons:

1. **Sprint 28 charter alignment.**  PROJECT_PLAN.md Sprint 28 framed the sprint as "pivot to a fundamentally different approach (METIS interop, geometric ND, supernodal reordering)".  (a) is pipeline-level (charter mismatch); (b) is structurally blocked on Pres_Poisson (corpus reality mismatch); (c) is the only candidate that fits the charter AND has a credible path to landing.

2. **Empirical evidence for retiring the literal target.**  After 5 sprints + ~200 pipeline-level measurements + 24-combination matrix exploration, Sprint 27 is the strongest empirical case yet that the literal 0.85× target on Pres_Poisson is unreachable via the pipeline this codebase implements.  The honest Sprint 28 outcome is to retire the target (Item 5 / Day 13 absorbs the (d) fallback's calibration content) and ship infrastructure that's USEFUL even though it doesn't close that one specific target.  (c) is that infrastructure: supernodal-etree reordering ships value on numeric-factor wall (5-15% on supernodal-heavy fixtures per Liu 1990) regardless of nnz_L outcome.

3. **Lowest-LOC + highest-clarity integration.**  (c) at 150-250 LOC is the smallest of the three; the integration point (post `sparse_analyze`) is the cleanest; the env-var gate (`SPARSE_ND_SUPERNODAL_POSTORDER={off,on}`) is the simplest; corpus-safety tests (numeric-factor residuals stay ≤ 1e-8) are well-defined.  Sprint 28 has 60 hrs allocated for Item 4 implementation; (c) at 150-250 LOC fits in 30-40 hrs of focused work, leaving 20-30 hrs of slack for the comparative numeric-factor wall benchmark + composition with AMD/ND ordering edge cases.

## Sprint 28 implications

With (c) picked:

- **Days 6-10 (Item 4 implementation):** as PLAN.md describes; gates via `SPARSE_ND_SUPERNODAL_POSTORDER={off,on}`; lands inside `sparse_analyze`'s postorder pass; composes with the existing `analysis->perm` to ensure supernodes appear contiguously in the final order; corpus-safety tests confirm numeric-factor residuals stay ≤ 1e-8 after the supernode-grouping permutation; Pres_Poisson nnz_L expected ≤ 5pp delta from Sprint 27 default (advisory; doesn't close 0.85×).

- **Day 11 (Item 6, conditional):** unlikely to fire under (c) — supernodal-etree reordering doesn't open new wall-reduction paths since it's a post-permutation that doesn't change the partition phase.  The 12 hrs becomes slack for Item 4 over-budget absorption or Item 5 prep.

- **Days 12-13 (Item 5 cross-corpus re-bench + production-default decisions + test-bound calibration):** the cross-corpus matrix verifies (c) lands as advisory (not a default flip); test bound `test_nd_pres_poisson_fill_with_leaf_amd` STAYS at Sprint 27's 0.94× (no tightening); literal 0.85× target is FORMALLY RETIRED in `headline_summary.md` + `docs/algorithm.md` with Sprint 28 documenting the 5-sprint evidence; Sprint 29+ can revisit if a fundamentally different approach (e.g. METIS C library interop, geometric mesh-aware ordering with first-class coordinate API) is pursued.

- **Day 14 (Item 7 retrospective):** documents the literal-target retirement; ships supernodal-etree as the worked non-pipeline-level pivot example; Sprint 29 inputs section routes any future numeric-factor wall-improvement work that the (c) infrastructure enables.

## Why not (d) empirical-floor calibration alone

Fallback (d) shifts items 4-6's 84 hrs into Sprint 29 wrap-up.  This option is rejected because:

1. **(c) is feasible within budget** (150-250 LOC fits in 60 hrs comfortably; the LOC estimate from Day-1 research is conservative).  The (d) fallback was for "all three look infeasible" — the empirical investigation today shows (c) is feasible.

2. **Sprint 28 productivity.**  Bailing to (d) leaves Sprint 28 at items 1 + 2 + 7 = 48 hrs across 14 days = a thin sprint.  Picking (c) ships meaningful new infrastructure within the same retirement-of-literal-target framing — strictly more useful for the same engineering effort.

3. **Sprint 29 setup.**  If Sprint 29 wants to revisit the literal 0.85× target with a fundamentally different machinery (e.g. METIS C library interop), the supernodal-etree infrastructure (c) provides a worked example of how a non-pipeline-level pivot integrates with the existing analyze + factor + reorder pipeline.  This is the kind of incremental scaffolding that compounds across sprints.

## Items 4-6 budget allocation under (c)

The Sprint 28 PLAN.md Day 6-10 budget (60 hrs Item 4 + 12 hrs Item 6 conditional) re-allocates as:

- **Days 6-9 (Item 4 implementation):** 48 hrs.  (c)'s 150-250 LOC translates to ~30-40 hrs focused work + 8-18 hrs corpus-safety tests + composition edge cases.  Day 9's interim verdict: Pres_Poisson nnz_L delta vs Sprint 27 default; expected ≤ 5pp range.
- **Day 10 (Item 4 close + decision doc):** 12 hrs.  `non_pipeline_decision.md` documents (c)'s flip-or-stay outcome; expected outcome is "advisory" (ships behind env-var gate; doesn't flip default; doesn't close 0.85×).
- **Day 11 (Item 6 conditional):** triggers don't fire under (c); 12 hrs becomes slack for Item 5 prep + Item 7 prep.

The Sprint 28 plan's Day 11 conditional logic accommodates this naturally — no plan revision needed.

## What ships in Sprint 28 Day 1

- `docs/planning/EPIC_2/SPRINT_28/pivot_decision_day1.md` (this doc) — empirical investigation, per-candidate dossier, chosen-pivot rationale, Sprint 28 implications.
- No source changes.  Default code path bit-identical.
- All quality checks clean.

## References

- `docs/planning/EPIC_2/PROJECT_PLAN.md` Sprint 28 section (lines 704-751) — the Sprint 28 charter
- `docs/planning/EPIC_2/SPRINT_28/PLAN.md` — Day 1 task breakdown
- `docs/planning/EPIC_2/SPRINT_27/RETROSPECTIVE.md` "Items deferred" #1 + "Sprint 28 inputs" #1, #3 — empirical priors
- `docs/planning/EPIC_2/SPRINT_27/headline_summary.md` — Day-13 24-setting × 6-fixture matrix verdict
- `docs/planning/EPIC_2/SPRINT_26/geometric_cut_design.md` — high-order FE-mesh empirical finding (Pres_Poisson mean degree 47.3, CV=0.108) + Sprint 26 grid-cut rejection precedent
- Karypis-Kumar 1998 §5.4 "Multiple-Matching Coarsening" — candidate (a) literature reference
- Hendrickson-Leland 1995 "Multilevel algorithms for partitioning irregular graphs" §3 "Geometric methods" — candidate (b) literature reference
- Liu 1990 "The role of elimination trees in sparse factorization" + Davis 2006 §6.5 "Postordering" — candidate (c) literature reference
