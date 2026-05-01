# Sprint 23 Plan: Ordering Quality Follow-Ups (Sprint 22 Deferrals)

**Sprint Duration:** 14 days
**Goal:** Close the two ordering-quality gaps Sprint 22 left open — ND's fill ratio on Pres_Poisson (currently 1.06× of AMD's nnz(L); the Sprint 22 plan's "≥ 2× reduction over AMD" target is unmet) and the simplified quotient-graph AMD's wall-time tail on small SPD corpus matrices (currently ~30 % bitset-favoured at n ≤ 1 800).  Two algorithmic fronts: bring `sparse_reorder_amd_qg` up to the full Davis 2006 reference algorithm (element absorption + supervariable detection + approximate-degree updates), and port METIS's O(1) gain-bucket structure into `graph_refine_fm` to lift FM from O(n²) to O(|E|) per pass.  Adds the per-leaf AMD call inside `nd_recurse` that Sprint 22's ND driver doesn't yet make, and swaps the `Cholesky-via-ND` residual test fixture for one whose conditioning lets the Sprint 22 plan's 1e-12 residual target become assertable.

**Starting Point:** Sprint 22 (PR #30, merged at `6a76f2c`) shipped: a multilevel `sparse_graph_partition` (`src/sparse_graph.c` — coarsen / bisect / uncoarsen / vertex-separator extraction, with single-pass Fiduccia-Mattheyses refinement at each level), the recursive `sparse_reorder_nd` driver (`src/sparse_reorder_nd.c` — separator-last with a natural-ordering base case at `n ≤ ND_BASE_THRESHOLD = 32`), the `SPARSE_REORDER_ND = 4` enum value wired through every factorization's `sparse_analyze` dispatch, and a *simplified* quotient-graph AMD (`src/sparse_reorder_amd_qg.c` — exact minimum-degree on a single integer workspace; element absorption / supervariable detection / approximate-degree updates all skipped).  The bitset AMD is gone; `sparse_reorder_amd` is now a one-line wrapper that forwards to the quotient-graph helper.  Sprint 22's `bench_day14.txt` cross-ordering capture documents ND/AMD = 1.06× on Pres_Poisson and the qg-AMD's ~1.5–2× wall-time penalty on bcsstk14 — both call out the algorithmic deferrals this sprint closes.

**End State:** `sparse_reorder_amd_qg` runs the full Davis 2006 algorithm: element absorption shrinks the active variable set as elimination proceeds (releasing workspace slots and bounding the per-pivot cost), supervariable detection merges variables with identical adjacency hashes so the minimum-degree pivot operates on supervariables (5–20× active-set shrinkage on PDE meshes), and the approximate-degree formula replaces the Sprint-22 exact recompute (per-pivot cost from O(adjacency) to O(adjacency-of-adjacency), plus a dense-row skip for vertices whose post-pivot degree exceeds `10·√n`).  `nd_recurse`'s base case calls `sparse_reorder_amd_qg` on each leaf-sized subgraph instead of emitting natural ordering, splicing the per-leaf permutation into the global `perm[]` via the existing `vertex_id_map`.  `graph_refine_fm` swaps its O(n) max-gain scan for METIS's O(1) gain-bucket structure, lifting FM from O(n²) to O(|E|) per pass and removing the Pres_Poisson wall-time penalty.  The `Cholesky-via-ND` residual test fixture is a strictly diagonally-dominant synthetic SPD instead of bcsstk14, so the 1e-12 residual target the Sprint 22 fixture's conditioning fought against becomes assertable.  Cross-corpus re-bench captures land in `docs/planning/EPIC_2/SPRINT_23/bench_*.{csv,txt}`, verifying ND/AMD on Pres_Poisson now reaches ≤ 0.7× and qg-AMD wall time on bcsstk14 is at or below the Sprint-22 bitset baseline.

**Time budget:** Each day caps at 12 hours.  The day budgets below sum to ~98 hours — about 10 hours above the 88-hour PROJECT_PLAN.md estimate, providing a similar safety buffer to Sprint 22 (which estimated 124 hrs, shipped at ~134 hrs).  Risk concentration is items 2 and 3 (the Davis-AMD upgrades): item 2 extends a load-bearing kernel that every Cholesky / LDL^T / LU / QR call routes through, and item 3's approximate-degree formula reads element-adjacency lists that item 2 must populate first — if item 2 ships partial, item 3 stalls.  Item 5 (FM gain-bucket) is independent and can be reshuffled if the AMD upgrades blow their budget.

---

## Day 1: Sprint Kickoff — Test Fixture Swap & AMD Reading

**Theme:** Land the small Item-1 fixture swap up front so Days 2-8's AMD work has a tight residual gate to validate against, then read the Davis 2006 algorithm and sketch the workspace layout the next three days will iterate on.

**Time estimate:** 6 hours

### Tasks
1. Re-read PROJECT_PLAN.md Sprint 23 section (lines 460-493), `docs/planning/EPIC_2/SPRINT_22/PERF_NOTES.md` "Why ND under-performs at Day 9" + "Sprint 23 closures" stub, and `docs/planning/EPIC_2/SPRINT_22/bench_day14.txt` for the Sprint-22 baseline numbers items 6 will measure against.
2. Replace bcsstk14 in `tests/test_reorder_nd.c::test_cholesky_via_nd_residual_bcsstk14` with a strictly diagonally-dominant synthetic SPD fixture (rename to `test_cholesky_via_nd_residual_spd_synth`).  Construction: take a 256×256 banded matrix with bandwidth 8, set `A[i][i] = 100.0` (forcing diagonal dominance), set off-diagonals to `0.5`.  Symmetric, SPD, condition number bounded by the 100/8 = 12.5 ratio — well within float64 1e-15 unit roundoff.  Restore the Sprint 22 plan's 1e-12 residual threshold (the previous 1e-8 was a fixture-conditioning workaround).  Document the fixture choice inline so the conditioning rationale is captured at the test, not in the commit message.
3. Read Davis (2006) "Direct Methods for Sparse Linear Systems" §7 (the AMD chapter): focus on §7.3 (element absorption), §7.4 (supervariable detection — hash-then-compare), §7.5 (approximate degree formula).  Take notes in a scratch file `docs/planning/EPIC_2/SPRINT_23/davis_notes.md` (kept in-tree as design rationale; deleted on retrospective day if it's just margin notes that never get cited from production code).  Key items to capture: the `iw[]` slice layout in the SuiteSparse AMD reference (variables-side vs elements-side regions; per-vertex `len[i]` / `elen[i]` accounting), the supervariable hash signature (sum-of-vertex-IDs is sufficient if combined with a full-list compare on collision), and the approximate-degree formula's three terms (variable adjacency, element adjacency, set-difference-with-pivot).
4. Sketch the `qg_t` extension in a comment block at the top of `src/sparse_reorder_amd_qg.c`: which extra slices of `iw[]` host the element-side adjacency, where `super[]` and `elen[]` per-vertex arrays live, and how the existing compaction walk in `qg_compact` extends to skip absorbed variables.  No code yet — design block only.
5. Run `make format && make lint && make test && make sanitize`.

### Deliverables
- New SPD synthetic fixture in `tests/test_reorder_nd.c` with the 1e-12 residual threshold restored
- `docs/planning/EPIC_2/SPRINT_23/davis_notes.md` capturing the Davis 2006 §7 reference notes
- File-header design block extension in `src/sparse_reorder_amd_qg.c` sketching the iw[] layout for items 2-3
- All quality checks clean

### Completion Criteria
- New SPD synthetic fixture's Cholesky-via-ND residual asserts < 1e-12 (locked in for the rest of the sprint)
- Davis-notes file references §7.3 / §7.4 / §7.5 explicitly with page numbers
- Design block in `sparse_reorder_amd_qg.c` shows where `super[]` / `elen[]` live and how compaction handles absorbed variables
- `make format && make lint && make test && make sanitize` clean

---

## Day 2: Quotient-Graph AMD — Element Absorption (Design + Workspace)

**Theme:** Land the workspace-layout extension that element absorption needs: per-vertex `elen[]` accounting and the elements-side region of `iw[]`.  No absorption logic yet — just the bookkeeping infrastructure that Days 3 will build absorption on top of.

**Time estimate:** 8 hours

### Tasks
1. Extend `qg_t` in `src/sparse_reorder_amd_qg.c`: add `idx_t *elen` (per-vertex count of element-side adjacency), and reserve a tail region of `iw[]` for element adjacency lists.  Update `qg_init` to allocate `elen[]` of length `n` and grow the initial `iw_size` from `5·nnz + 6·n + 1` to `7·nnz + 8·n + 1` (Davis 2006 §7's reference size, which buffers the elements-side region).  Update `qg_free` and the size-cap check.
2. Extend the layout doc comment to describe the new structure: each vertex's `iw[]` slice now has two regions — the variable-adjacency prefix (length `len[i] - elen[i]`) followed by the element-adjacency suffix (length `elen[i]`).  No element list exists yet — every vertex starts with `elen[i] = 0`, so this is structurally invisible until Day 3 populates it.
3. Update `qg_compact` to walk the new layout: when relocating a vertex's `iw[]` slice, the move is still `memmove(&iw[new_pos], &iw[old_xadj], len[i] * sizeof(idx_t))` because the variable-side and element-side regions are contiguous within `len[i]` entries.  Add a debug assertion that `elen[i] ≤ len[i]` before each relocation.
4. Run the existing `tests/test_reorder_amd_qg.c` — fill should stay bit-identical (this day is structural-only; no algorithmic change).  Capture the workspace size in a quick stderr print to confirm the new `iw_size` is what we expect.
5. Add a unit test `test_qg_workspace_extension_no_regression`: builds a 100×100 banded fixture, runs the existing `sparse_reorder_amd_qg`, asserts the resulting permutation matches the Sprint-22 baseline byte-for-byte (use a saved permutation array as the golden).
6. Run `make format && make lint && make test && make sanitize`.

### Deliverables
- `qg_t` extended with `elen[]` and the larger initial workspace
- Layout doc comment describes the variable/element split
- Compaction walk updated for the new layout (no behaviour change yet)
- `test_qg_workspace_extension_no_regression` pinning bit-identical fill
- All quality checks clean

### Completion Criteria
- nos4 / bcsstk04 / bcsstk14 produce identical permutations to the Sprint-22 baseline (no behaviour change)
- New workspace size is `7·nnz + 8·n + 1` (verified by stderr probe)
- Compaction debug assertion (`elen[i] ≤ len[i]`) holds on every elimination step
- `make format && make lint && make test && make sanitize` clean

---

## Day 3: Quotient-Graph AMD — Element Absorption (Implementation)

**Theme:** Wire element absorption into `qg_eliminate`: when a pivot is eliminated, record the new element it forms; when a neighbour's variable adjacency reduces to a single element `e`, mark the neighbour absorbed and remove its variable-side state.  Validate against the Sprint-22 fill baseline.

**Time estimate:** 8 hours

### Tasks
1. Implement element creation in `qg_eliminate`: when pivot `p` is eliminated, the neighbours' merged adjacency forms a new element.  Allocate a fresh element ID (reusing the eliminated pivot's slot — Davis's convention), write its variable-adjacency list into `iw[]`'s element-side region, and update each neighbour's `elen[i]` to include the new element ID.
2. Implement variable absorption: after each elimination, walk the neighbours; for any vertex `i` whose remaining variable-side adjacency is empty (i.e. `len[i] == elen[i]`) and whose element-side has converged to a single element, mark `i` absorbed (a new `qg->absorbed[i]` byte array).  Absorbed vertices skip the next `qg_pick_min_deg` scan and have their workspace slots released by the next compaction.
3. Update `qg_pick_min_deg` to skip both eliminated and absorbed vertices.
4. Update `qg_compact` to skip absorbed vertices when packing surviving adjacency, releasing their slots.  This is what makes the absorption a *workspace* win — without compaction, absorbed slots would leak.
5. Validate against existing parity tests: `test_amd_qg_delegation_*` on nos4 / bcsstk04 / bcsstk14 should still pass (fill is *exact* minimum degree either way; element absorption shrinks the active set but doesn't change pivot order on these fixtures).  If fill diverges by even 1 nnz, root-cause before moving on — element absorption can subtly affect tie-breaking on degenerate degree-equal pivots.
6. Probe the workspace high-water on bcsstk14: print `iw_used / iw_size` peak per pivot.  Sprint-22 baseline was a steady climb; with absorption the slope should flatten toward the end of elimination.
7. Run `make format && make lint && make test && make sanitize`.

### Deliverables
- Element absorption logic in `qg_eliminate` + `qg_pick_min_deg` + `qg_compact`
- New `qg->absorbed[]` byte array tracking absorbed-vertex state
- Stderr probe documenting workspace high-water shrinkage on bcsstk14
- All quality checks clean

### Completion Criteria
- All `tests/test_reorder_amd_qg.c` parity tests pass with bit-identical fill (or a documented +1 / -1 nnz drift on a single fixture, root-caused to tie-breaking and inconsequential for fill quality)
- Absorbed-vertex count after final pivot equals `n - n_supernodes` (where supernodes will be measured Day 4) — sanity check that absorption is engaged
- `iw_used` peak on bcsstk14 reduces vs the Sprint-22 baseline (any reduction is a win; we don't pin a specific ratio yet)
- `make format && make lint && make test && make sanitize` clean

---

## Day 4: Quotient-Graph AMD — Supervariable Detection

**Theme:** Detect variables with identical adjacency and merge them into supervariables so the minimum-degree pivot operates on supervariable count instead of variable count.  PDE-mesh fixtures see 5-20× active-set shrinkage from this — the headline performance win for items 2-3.

**Time estimate:** 8 hours

### Tasks
1. Add `qg->super[]` (per-vertex array; `super[i]` = the supervariable representative this vertex belongs to, initially `super[i] = i`) and `qg->super_size[i]` (count of variables in supervariable `i`).  Allocate in `qg_init`, free in `qg_free`.
2. Implement the hash signature in `qg_supervariable_hash`: sum-of-IDs over a vertex's variable-side adjacency XOR sum-of-IDs over its element-side adjacency.  Cheap to compute, cheap to compare, low collision rate on PDE meshes.
3. Implement the merge step in `qg_merge_supervariables` (called once per pivot, after element absorption settles): bucket post-pivot non-eliminated vertices by hash; within each bucket, do an O(k²) full-list compare to confirm identical adjacency; merge confirmed pairs by adjusting `super[]` and `super_size[]`.  The smaller-supervariable representative folds into the larger.
4. Update `qg_pick_min_deg` to scan over supervariable representatives only (skip vertices where `super[i] != i`).  The "degree" of a supervariable is the variable count of its non-self adjacency, weighted by the merged-in `super_size[]`.
5. Update `qg_eliminate` to mark every vertex in the pivot's supervariable as eliminated together (this is the supervariable's defining property — they all eliminate at the same step).  Append all of them to `perm[]` in some deterministic order (sorted by original vertex ID).
6. Add `test_qg_supervariable_synthetic`: build a fixture where 4 vertices have identical adjacency (e.g., 4 leaves of a star joined at a common centre); assert that after the centre is eliminated, the 4 leaves merge into a single supervariable; assert the final permutation contains all 4 in a contiguous block.
7. Re-validate corpus parity: `test_amd_qg_delegation_*` on nos4 / bcsstk04 / bcsstk14 should still produce fill within 1.000× of the bitset-AMD baseline (saved in `bench_day14.txt`).  Supervariable detection is fill-neutral (supervariables eliminate together; the per-vertex pivot order within a supervariable doesn't change what gets merged).
8. Run `make format && make lint && make test && make sanitize`.

### Deliverables
- `qg->super[]` / `super_size[]` infrastructure
- `qg_supervariable_hash` + `qg_merge_supervariables`
- `qg_pick_min_deg` and `qg_eliminate` updated for supervariable-aware iteration
- New `test_qg_supervariable_synthetic` test pinning the merge contract
- Corpus parity tests pass with fill ≤ 1.000× of Sprint-22 baseline
- All quality checks clean

### Completion Criteria
- 4-leaves-of-a-star fixture merges into a single supervariable after centre elimination
- nos4 / bcsstk04 / bcsstk14 fill ≤ 1.000× of Sprint-22 baseline (bit-identical preferred; +1 nnz drift acceptable if root-caused to tie-breaking)
- Supervariable count on bcsstk14 (n=1806) measurable and < n by a factor ≥ 2 (sanity check that detection is engaged on a real fixture)
- `make format && make lint && make test && make sanitize` clean

---

## Day 5: Quotient-Graph AMD — Approximate-Degree Formula

**Theme:** Replace the per-pivot exact-degree recompute with Davis's approximate degree formula.  Cuts per-pivot cost from O(adjacency) to O(adjacency-of-adjacency), which is the asymptotic win behind SuiteSparse AMD's wall-time advantage.

**Time estimate:** 6 hours

### Tasks
1. Implement `qg_approximate_degree(qg, vertex)` per Davis 2006 §7.5: `d_approx(i) = |adj(i, V)| + Σ_{e ∈ adj(i, E)} |adj(e, V) \ {pivot}|`, where the pivot is the variable just eliminated.  Reads element-side adjacency lists populated by Day 3's element absorption.
2. Replace the exact-degree recompute call in `qg_eliminate` (the one that ran `qg->deg[u] = qg_compute_exact_degree(qg, u)` per neighbour) with a call to `qg_approximate_degree`.  Keep the exact-degree function around as `qg_exact_degree_for_test` (used by Day 6's parity test).
3. Add the dense-row skip from Davis 2006 §7.5: vertices whose post-pivot approximate degree exceeds `10·√n` skip the update entirely (their degree stays at the previous value; the formula is least-accurate for dense rows and they'd dominate the pivot scan anyway).
4. Verify the approximation is conservative (never under-estimates the true degree) on a tiny synthetic — Davis's formula is known to be an upper bound, but it's worth pinning that with an `assert(d_approx ≥ d_exact)` in a debug test on a 50-vertex random graph.
5. Run `make format && make lint && make test && make sanitize`.

### Deliverables
- `qg_approximate_degree` implementing Davis's formula
- Dense-row skip threshold at `10·√n`
- `qg_exact_degree_for_test` retained for Day 6 parity verification
- Conservative-bound assertion on the 50-vertex synthetic
- All quality checks clean

### Completion Criteria
- All existing corpus parity tests pass (fill quality unchanged — approximate-degree is a tie-breaking heuristic; exact-MD is preserved on the corpus)
- Conservative-bound assertion holds on 50-vertex random graph for 100 random pivot sequences
- `make format && make lint && make test && make sanitize` clean

---

## Day 6: Quotient-Graph AMD — Approximate-Degree Validation & Dense-Row Tests

**Theme:** Pin the approximate-degree formula's output against an exact-degree reference on a controlled synthetic, and verify the dense-row skip kicks in on a fixture engineered to have one.  This is the test-side sibling to Day 5's algorithmic change.

**Time estimate:** 6 hours

### Tasks
1. New test `test_qg_approx_degree_parity`: 200-vertex synthetic graph (10×20 grid with random edge sparsification — varied degree distribution); run `sparse_reorder_amd_qg` while recording, at every pivot, both the approximate degree (from `qg_approximate_degree`) and the exact degree (from `qg_exact_degree_for_test`) of every neighbour.  Assert: `d_approx ≥ d_exact` always (conservative bound), and `d_approx ≤ 1.5 × d_exact` for ≥ 90% of pivots (Davis's bound is tight in practice on regular meshes).
2. New test `test_qg_dense_row_skip`: build a fixture with one row of degree `n - 1` (a vertex connected to all others) on a `n = 200` graph; run elimination; assert the dense vertex's degree update is skipped (gets a `qg->skipped_dense_count` probe added in Day 5).  Assert the skip count grows monotonically as pivots eliminate dense neighbours.
3. Verify Pres_Poisson fill on the new approximate-degree path: load `tests/data/suitesparse/Pres_Poisson.mtx`, run `sparse_reorder_amd_qg`, assert nnz(L) ≤ 1.05× of the Sprint-22 baseline (saved in `bench_day14.txt`).  This is the corpus-scale parity check — approximate-degree is a heuristic, so a tiny drift is acceptable, but a > 5% regression means the formula or the dense-row threshold is mis-tuned.
4. Quick wall-time probe: time `sparse_reorder_amd_qg` on bcsstk14 (n=1806) with `clock()`; expected speedup vs Sprint-22 baseline is ~2× (Days 2-5 cumulative).  Capture in a stderr print; if the speedup is < 1.5×, profile and capture the hot spot in `davis_notes.md` for Day 13's bench-rerun day to revisit.
5. Run `make format && make lint && make test && make sanitize`.

### Deliverables
- `test_qg_approx_degree_parity` pinning the conservative-bound + tight-bound contract
- `test_qg_dense_row_skip` pinning the dense-row skip behaviour
- Pres_Poisson fill within 5% of Sprint-22 baseline (assertion)
- bcsstk14 wall-time probe captured (stderr or `davis_notes.md`)
- All quality checks clean

### Completion Criteria
- Conservative-bound contract holds on 200-vertex synthetic (100% of pivots)
- Tight-bound contract holds for ≥ 90% of pivots on the same synthetic
- Dense-row skip count > 0 on the engineered fixture
- Pres_Poisson nnz(L) ≤ 1.05× Sprint-22 baseline
- bcsstk14 wall-time speedup ≥ 1.5× vs Sprint-22 (or root-caused regression note in `davis_notes.md`)
- `make format && make lint && make test && make sanitize` clean

---

## Day 7: Nested Dissection — Per-Leaf AMD Splice (Implementation)

**Theme:** Replace `nd_emit_natural`'s leaf-base-case behaviour with a `sparse_reorder_amd_qg` call on the leaf subgraph.  This is the Sprint-22 deferral that made `bench_day14.txt`'s ND/AMD ratio = 1.06× instead of the plan's < 0.5× target — the leaves emit natural ordering when they should emit AMD.

**Time estimate:** 6 hours

### Tasks
1. Extract the leaf-subgraph-to-SparseMatrix conversion: `nd_subgraph_to_sparse(graph_t *G_leaf, SparseMatrix **A_leaf)` builds a temporary square `SparseMatrix` from a `sparse_graph_t`'s CSR adjacency.  Each edge becomes a value=1.0 entry; the matrix is symmetric and pattern-only (numeric values don't matter — AMD only reads the symbolic structure).  Lives in `src/sparse_reorder_nd.c` as a static helper.
2. Replace `nd_emit_natural`'s leaf-base-case path in `nd_recurse`: at `n ≤ ND_BASE_THRESHOLD`, build the temporary `SparseMatrix` via `nd_subgraph_to_sparse`, allocate a per-leaf `idx_t *leaf_perm` of length `n_leaf`, call `sparse_reorder_amd_qg(A_leaf, leaf_perm)`, then splice the leaf permutation into the global `perm[]` via `vertex_id_map`: `perm[*next_pos + i] = vertex_id_map[leaf_perm[i]]`.  Free the temporary `SparseMatrix` and `leaf_perm` before returning.
3. Handle the edge case where `sparse_reorder_amd_qg` fails on a leaf (allocation failure, invariant violation): fall back to `nd_emit_natural` for that one leaf and propagate the rc up so the caller can surface the failure in debug builds.  Document the fallback in a comment.
4. Re-validate `tests/test_reorder_nd.c::test_nd_4x4_grid_valid_permutation` still passes — the 4×4 grid's separator-last contract is unchanged (the separator vertices still go last; only the leaf interior changes from natural to AMD-ordered).  The current test's `ND_BASE_THRESHOLD = 4` override stays — it forces partitioning to actually run, and the separator-last assertion is unaffected by the leaf-AMD change.
5. Run `make format && make lint && make test && make sanitize`.

### Deliverables
- `nd_subgraph_to_sparse` helper in `src/sparse_reorder_nd.c`
- `nd_recurse` base case calls `sparse_reorder_amd_qg` instead of `nd_emit_natural`
- Fallback path on per-leaf AMD failure
- All quality checks clean

### Completion Criteria
- Existing 4×4 grid separator-last contract test still passes
- Existing 1D-path / singleton / null-args / rejects-rectangular tests still pass
- `make format && make lint && make test && make sanitize` clean

---

## Day 8: Nested Dissection — Validation & 10×10 Grid Tightening

**Theme:** Tighten the test bounds the per-leaf AMD splice should now hit.  Sprint 22's 10×10 grid test asserted `ND ≤ 1.5× AMD`; with leaves now AMD-ordered, ND should be at or below AMD on this fixture.  Pres_Poisson fill probe is the canonical validation — the Sprint-22 1.06× ratio should drop to ≤ 0.7×.

**Time estimate:** 6 hours

### Tasks
1. Tighten `tests/test_reorder_nd.c::test_nd_10x10_grid_beats_amd_fill`: replace the current `ND ≤ 1.5× AMD` assertion (`(long long)nnz_nd * 2 <= (long long)nnz_amd * 3`) with `ND ≤ 1.0× AMD` (`nnz_nd ≤ nnz_amd`).  Update the inline comment to reflect the new bound and reference the Sprint 23 leaf-AMD splice.  Rename the test to `test_nd_10x10_grid_matches_or_beats_amd_fill` to match the new contract.
2. Add `test_nd_pres_poisson_fill_with_leaf_amd`: load Pres_Poisson, run ND with the new leaf-AMD path, assert nnz(L) ≤ 0.7× of AMD's nnz(L) on the same fixture (the Sprint 22 plan's relaxed target — full closure to 0.5× may need multi-pass FM, deferred if budget allows).  Skip cleanly if the fixture isn't loadable (matches the Sprint-22 pattern in this file).
3. Run a quick determinism re-check: `test_nd_determinism_public_api` should still pass bit-identically — leaf-AMD is deterministic given the same input subgraph, so the global permutation should still be deterministic.
4. Wall-time probe: time ND on Pres_Poisson; capture in stderr.  Sprint-22 measured ~38 s on the natural-leaf path; the leaf-AMD splice trades a bit of leaf-side compute for upper-level fill.  Expected: ≤ 50 s (a small regression is acceptable; the headline is fill quality, not wall time — Day 9-12's FM gain-bucket work closes the wall-time side).
5. Re-run the full corpus cross-ordering bench (`benchmarks/bench_reorder.c`) and capture stderr to `docs/planning/EPIC_2/SPRINT_23/bench_day8_nd_leaf_amd.txt` as an interim checkpoint.  Compare row-by-row against `bench_day14.txt`; the ND nnz(L) row should drop on every fixture (or stay flat on fixtures already AMD-quality).
6. Run `make format && make lint && make test && make sanitize`.

### Deliverables
- 10×10 grid test bound tightened to `ND ≤ AMD`
- New `test_nd_pres_poisson_fill_with_leaf_amd` pinning the ≤ 0.7× ratio
- `bench_day8_nd_leaf_amd.txt` interim cross-ordering capture
- Wall-time probe documented
- All quality checks clean

### Completion Criteria
- 10×10 grid: `nnz_nd ≤ nnz_amd` (was 1.25×; should now be ≤ 1.0×)
- Pres_Poisson: `nnz_nd / nnz_amd ≤ 0.7` (was 1.06×; SQQRT 22's < 0.5× target relaxed for this sprint)
- ND determinism contract still bit-identical
- bench_day8 capture committed; nnz_L drops vs `bench_day14.txt` on every fixture (or root-cause note inline)
- `make format && make lint && make test && make sanitize` clean

---

## Day 9: FM Gain-Bucket — Design & Bucket Structure

**Theme:** Stand up the gain-bucket data structure that Day 10 will integrate into `graph_refine_fm`.  Bucket array indexed by gain value; each bucket is a doubly-linked list of vertex IDs; an O(1) max-find uses a "highest non-empty bucket" cursor.  The infrastructure day before the swap.

**Time estimate:** 8 hours

### Tasks
1. Read METIS source (`Lib/refine.c` in the upstream METIS reference, or §4 of the Karypis-Kumar 1998 paper if the source isn't accessible) for the bucket structure's exact shape: bucket array size = `2 · max_gain + 1` (gains range from `-max_gain` to `+max_gain`), bucket offset = `max_gain` (so bucket index `bucket_offset + gain ∈ [0, 2·max_gain]`), each bucket is a doubly-linked list head pointer plus a count.
2. Add `src/sparse_graph_fm_buckets.h` (internal header) declaring `fm_bucket_array_t`: bucket head pointers, count per bucket, the `max_gain` parameter, plus the highest-non-empty cursor.
3. Implement `fm_bucket_array_init(arr, n_vertices, max_gain)` — allocates the bucket head array (length `2·max_gain + 1`) plus per-vertex `prev`/`next` linkage arrays (length `n_vertices` each).  All buckets initially empty; cursor at `-1` (sentinel).
4. Implement `fm_bucket_insert(arr, vertex, gain)` and `fm_bucket_remove(arr, vertex, gain)` as O(1) doubly-linked-list operations.  Update the highest-non-empty cursor on insert (cursor jumps up if the new gain exceeds current cursor) and on remove (cursor walks down past empty buckets if removing the cursor-bucket's last element).
5. Implement `fm_bucket_pop_max(arr, vertex_out, gain_out)`: returns the head of the highest non-empty bucket, removes it from that bucket, walks the cursor down if needed.  This is the O(1) max-find that replaces Sprint 22's O(n) scan.
6. Implement `fm_bucket_array_free`.
7. New unit test `tests/test_graph_fm_buckets.c`: pin the API contract — insert 100 vertices with random gains in `[-50, +50]`; pop them all out via `fm_bucket_pop_max`; assert the popped sequence is non-increasing in gain (proves max-find correctness); insert/remove cycle of 1000 random operations; assert no double-free / no UB under ASan.
8. Wire the new file into Makefile + CMakeLists.txt.
9. Run `make format && make lint && make test && make sanitize`.

### Deliverables
- `src/sparse_graph_fm_buckets.h` + `.c` (or inline in `sparse_graph.c`) implementing the bucket structure
- 4-function API (`init` / `insert` / `remove` / `pop_max` / `free`)
- `tests/test_graph_fm_buckets.c` pinning the API contract
- All quality checks clean

### Completion Criteria
- 1000-operation insert/remove cycle is clean under ASan
- Pop sequence on 100-vertex test is non-increasing in gain
- Cursor invariant (always at the highest non-empty bucket) holds across the test cycle
- `make format && make lint && make test && make sanitize` clean

---

## Day 10: FM Gain-Bucket — Integration with `graph_refine_fm`

**Theme:** Swap Sprint 22's O(n) max-gain scan for the new bucket structure.  This is the algorithmic change that lifts FM from O(n²) to O(|E|) per pass and removes the Pres_Poisson wall-time penalty.

**Time estimate:** 8 hours

### Tasks
1. Refactor `graph_refine_fm` in `src/sparse_graph.c`: replace the gain-array + linear-scan pattern with the bucket structure.  Boundary-vertex gain initialization becomes `fm_bucket_insert(boundary_buckets, v, gain[v])` for each boundary vertex.
2. Replace the per-step max-find: was `for (i) if (gain[i] > best_g && !locked[i]) ...`; becomes `fm_bucket_pop_max(boundary_buckets, &v, &g)`.  Locked-vertex handling: `pop_max` returns the highest-gain vertex; if it happens to be locked (shouldn't be, since we removed it on lock), the loop continues — but to be defensive, add an `assert(!locked[v])` and document the invariant.
3. Implement gain-update propagation: when vertex `v` moves to the other partition, every neighbour `u` whose gain changes needs `fm_bucket_remove(buckets, u, old_gain) + fm_bucket_insert(buckets, u, new_gain)`.  This is the per-move O(1) work that replaces Sprint-22's full re-scan.
4. Compute `max_gain` upfront for the bucket array sizing: `max_gain = max(degree[i])` over all vertices in the boundary.  Use that to size the bucket array; if a gain exceeds `max_gain` mid-pass (shouldn't happen — gain is bounded by degree — but document the contract), assert and abort the pass.
5. Run all existing partitioning tests in `tests/test_graph.c`: `test_partition_*` (2D mesh / 3D mesh / disconnected fixtures from Sprint 22 Day 4-5).  All must pass — the partitions should be at least as good as Sprint-22's, since FM with O(1) max-find can do more work per pass under the same wall-clock budget.
6. Re-run the cross-ordering bench (`bench_reorder.c`) on Pres_Poisson; capture wall time for the ND reorder phase.  Sprint-22 baseline was ~38 s; expected: ≤ 10 s (gain-bucket FM should remove the dominant cost).  Save to `bench_day10_fm_buckets.txt` as an interim checkpoint.
7. Run `make format && make lint && make test && make sanitize`.

### Deliverables
- `graph_refine_fm` refactored to use the bucket structure
- Gain-update propagation integrated; per-move work is O(1)
- All Sprint-22 partition tests still pass
- `bench_day10_fm_buckets.txt` Pres_Poisson wall-time capture
- All quality checks clean

### Completion Criteria
- All `test_partition_*` tests pass with bit-identical or improved separator sizes vs Sprint-22 baseline
- Pres_Poisson ND reorder wall time ≤ 10 s (was ~38 s)
- `make format && make lint && make test && make sanitize` clean

---

## Day 11: FM Gain-Bucket — Stress Tests & Edge Cases

**Theme:** Push the bucket-FM against pathological fixtures (degree-skewed graphs, gain-overflow scenarios, single-vertex boundaries) and lock in the deterministic-seed contract that ND's reproducibility depends on.

**Time estimate:** 8 hours

### Tasks
1. Edge-case tests in `tests/test_graph_fm_buckets.c`:
   - Single-vertex boundary: bucket insert + pop_max + assert empty.  No infinite loop, no UB.
   - All-zero-gain boundary: every bucket index `bucket_offset` (gain=0); pop order is FIFO (insertion order); deterministic.
   - All-equal-positive-gain boundary: same as above but gain=+5; cursor stays at one bucket throughout.
   - Mixed positive + negative gains: pop order is gain-descending; ties broken by insertion order.
2. Determinism contract: same input + same seed → same partition.  Re-test `tests/test_graph.c::test_partition_determinism` — bucket-FM must preserve this.  If the linked-list ordering within a bucket is implementation-defined (it should be FIFO), document that as the determinism source.
3. Stress test on a 1000-vertex random graph: insert all 1000 vertices with random gains, do 5000 random insert/remove/pop cycles, assert the bucket invariant (cursor at highest non-empty) holds throughout.  Run under ASan.
4. Multi-pass FM exploration (budget-permitting, last 2 hours): Sprint 22's FM is single-pass per partition; Davis 2006 §4.2 / METIS reference run 2-3 passes for additional cut quality.  Try a 3-pass version on Pres_Poisson; assert the 3rd pass either matches the 2nd or improves; document the result in `davis_notes.md`.  If 3-pass doesn't measurably improve cut quality (likely on regular meshes), revert to single-pass and note the finding.
5. Run `make format && make lint && make test && make sanitize`.

### Deliverables
- 4+ edge-case tests in `tests/test_graph_fm_buckets.c`
- Stress-test on 1000-vertex random graph passes under ASan
- Determinism contract verified post-bucket-FM
- Multi-pass FM exploration result documented in `davis_notes.md`
- All quality checks clean

### Completion Criteria
- All edge cases produce expected pop order without crashes / asserts
- 5000-operation stress cycle clean under ASan; cursor invariant holds throughout
- Determinism contract: same input + same seed → bit-identical partition
- Multi-pass exploration finding documented (single-pass retained or multi-pass adopted with rationale)
- `make format && make lint && make test && make sanitize` clean

---

## Day 12: Cross-Corpus Re-Bench — Initial Captures

**Theme:** Re-run `benchmarks/bench_reorder.c` and `benchmarks/bench_amd_qg.c` against the Sprint-22 baselines.  Capture nnz(L) and wall time for every fixture × ordering combination.  This is the headline measurement day.

**Time estimate:** 6 hours

### Tasks
1. Run `benchmarks/bench_reorder.c` against the full SuiteSparse corpus (nos4, bcsstk04, Kuu, bcsstk14, s3rmt3m3, Pres_Poisson) with all five orderings (NONE / RCM / AMD / COLAMD / ND).  Capture stdout to `docs/planning/EPIC_2/SPRINT_23/bench_day12.csv` and a human-readable rendering to `bench_day12.txt`.  Use the same `column -t -s,` pipe Sprint-22 used so visual diffs are clean.
2. Run `benchmarks/bench_amd_qg.c` (the bitset comparison foil from Sprint 22 Day 13) — capture to `bench_day12_amd_qg.txt`.  This is the qg-AMD wall-time + memory check; the bitset reference is unchanged from Sprint 22, so this run measures Sprint 23's AMD upgrades (Days 2-6) head-to-head.
3. Side-by-side compare against `bench_day14.txt` and `bench_day13_amd_qg.txt` from Sprint 22.  Build a small markdown table in `docs/planning/EPIC_2/SPRINT_23/bench_summary_day12.md` showing nnz(L) and wall-time deltas per fixture × ordering.
4. Headline checks (Sprint 23's deliverable gates from PROJECT_PLAN.md item 6):
   - **(a)** Pres_Poisson ND/AMD ≤ 0.7×.  Was 1.06×.
   - **(b)** qg-AMD wall time on bcsstk14 ≤ Sprint-22 bitset baseline.  Was ~30 % bitset-favoured.
   - **(c)** `bench_day14.txt` nnz(L) row stays bit-identical or improves on every fixture.
5. If any of (a)/(b)/(c) miss, root-cause and document in `bench_summary_day12.md` — Days 13's budget is for the closing tests + retro stub, but if a hard regression turns up here we'd need to flag it as a Sprint 24 follow-up rather than land Sprint 23 broken.
6. Run `make format && make lint && make test && make sanitize` (no source changes today, but the bench captures are committed).

### Deliverables
- `bench_day12.csv` + `bench_day12.txt` cross-ordering capture
- `bench_day12_amd_qg.txt` qg-AMD vs bitset capture
- `bench_summary_day12.md` side-by-side table vs Sprint-22 baselines
- All three headline checks (a)/(b)/(c) status documented
- All quality checks clean

### Completion Criteria
- Headline check (a): Pres_Poisson ND/AMD ≤ 0.7×
- Headline check (b): qg-AMD wall time on bcsstk14 ≤ Sprint-22 bitset baseline
- Headline check (c): nnz(L) bit-identical-or-better on every `bench_day14.txt` row
- `bench_summary_day12.md` committed with the deltas
- `make format && make lint && make test && make sanitize` clean (no behaviour change today)

---

## Day 13: Closing Tests & Documentation Sweep

**Theme:** Add the supervariable + approximate-degree contract tests called out in PROJECT_PLAN.md item 7, refresh `docs/algorithm.md`'s AMD subsection to describe the now-full Davis algorithm, and append a "Sprint 23 closures" subsection to `SPRINT_22/PERF_NOTES.md` with the Day-12 numbers.

**Time estimate:** 8 hours

### Tasks
1. New tests in `tests/test_reorder_amd_qg.c`:
   - `test_qg_supervariable_synthetic_corpus`: 4-row stencil fixture with 16 vertices arranged so 4 explicitly-known supervariables form (vertices 0-3, 4-7, 8-11, 12-15 each share identical adjacency post-elimination of a centre vertex).  Assert `qg_merge_supervariables` produces exactly 4 supervariable representatives after the centre eliminates.  Stronger contract than Day 4's spot test.
   - `test_qg_approx_degree_parity_corpus`: extend Day 6's parity test to bcsstk14 + Pres_Poisson; assert the conservative-bound contract holds across the full elimination on a real corpus matrix (not just the 200-vertex synthetic).
2. Update `docs/algorithm.md`'s AMD subsection: rewrite the "Quotient-Graph AMD (Sprint 22)" paragraph as "Quotient-Graph AMD (Davis 2006)".  Cover the four mechanisms now in place: element absorption, supervariable detection, approximate-degree updates, and the dense-row skip.  Reference Days 2-6 commits as the implementation history.  Drop the Sprint-22 caveat that called out the simplifications as deferred.
3. Append a "Sprint 23 closures" subsection to `docs/planning/EPIC_2/SPRINT_22/PERF_NOTES.md`: the table from `bench_summary_day12.md` plus a 1-2 paragraph narrative explaining what Sprint 23 actually moved (ND/AMD ratio, qg-AMD wall time, FM wall time on Pres_Poisson) and what it didn't (the < 0.5× ND/AMD on Pres_Poisson — defer to Sprint 24 if the multi-pass FM Day 11 explored doesn't get there).
4. Stub `docs/planning/EPIC_2/SPRINT_23/RETROSPECTIVE.md` with the same eight-section structure as `SPRINT_22/RETROSPECTIVE.md`: Goals & Outcomes, What Shipped, What Slipped, What Surprised Us, Items Deferred, Time Tracking, Lessons Learned, Sprint 24 Inputs.  Headers only; the post-sprint write-up fills the body.
5. Update `docs/planning/EPIC_2/PROJECT_PLAN.md`'s Sprint 23 section: add a "**Status: Complete**" line + the actual hours-spent total (from time-tracking accumulated through the sprint).
6. Decide on `davis_notes.md`: if it's been useful as design-rationale citation in commit messages and code comments, retain it under `docs/planning/EPIC_2/SPRINT_23/`; if it's mostly margin notes, delete it.  Document the call in the retro stub.
7. Run `make format && make lint && make test && make sanitize`.

### Deliverables
- 2 new corpus-scale contract tests in `tests/test_reorder_amd_qg.c`
- `docs/algorithm.md` AMD subsection refreshed for the full Davis algorithm
- `SPRINT_22/PERF_NOTES.md` Sprint-23-closures subsection added
- `SPRINT_23/RETROSPECTIVE.md` stub committed
- `PROJECT_PLAN.md` Sprint 23 marked complete with actual hours
- `davis_notes.md` retained or removed with rationale
- All quality checks clean

### Completion Criteria
- 4-supervariable synthetic test passes
- Approximate-degree parity holds on bcsstk14 + Pres_Poisson
- `algorithm.md` AMD subsection no longer mentions Sprint-22 simplifications
- `PERF_NOTES.md` closures subsection cites Day 12's numbers
- Retrospective stub has all 8 section headers
- `make format && make lint && make test && make sanitize` clean

---

## Day 14: Soak, Final Bench Capture & Sprint Retrospective

**Theme:** Final cross-ordering capture as the end-of-sprint headline, full corpus regression run, retrospective body, and PR open.  This is the day that gates the merge.

**Time estimate:** 8 hours

### Tasks
1. Final capture: re-run `benchmarks/bench_reorder.c` and `benchmarks/bench_amd_qg.c` once more (Day 12's runs were the working capture; this is the closing one with any Day-13 doc/test changes baked in).  Save to `bench_day14.csv` / `bench_day14.txt` / `bench_day14_amd_qg.txt`.  Sanity-check that nothing regressed since Day 12 — should be bit-identical on the bench output since Day 13 was tests + docs only.
2. Run `make sanitize` against the full test suite (ASan + UBSan).  Sprint 22's sanitize pass took ~5 minutes; budget similarly.  Any new warnings since Sprint 22's clean baseline are a hard gate — investigate before the retro write-up.
3. Run `make tsan` against the OpenMP-parallelised tests.  Sprint 22 added thread-safety guards; verify Sprint 23's algorithmic changes don't introduce new races (they shouldn't — the AMD upgrades are single-threaded; only the `sparse_reorder_nd_base_threshold` global is shared, and Sprint 22's bench/test header documented its non-thread-safe contract).
4. Fill in the Sprint 23 retrospective (`docs/planning/EPIC_2/SPRINT_23/RETROSPECTIVE.md` body): for each of the 8 sections, write 1-2 paragraphs.  Headline material: did the AMD upgrades hit the 0.7× ND/AMD ratio on Pres_Poisson?  Did the FM gain-bucket wall time match expectations?  What slipped (multi-pass FM, if it didn't move the needle)?  What surprised us (likely: supervariable detection's active-set shrinkage being less than the textbook 5-20× on the SuiteSparse corpus, since these matrices have less geometric regularity than synthetic PDE meshes).
5. Open the Sprint 23 PR (`gh pr create`) targeting master.  PR description summarizes the seven items + the day-by-day commits + the headline numbers from `bench_day14.txt` vs Sprint 22's `bench_day14.txt`.
6. Run `make format && make lint && make test && make sanitize && make tsan`.

### Deliverables
- `bench_day14.csv` / `.txt` / `_amd_qg.txt` final captures
- `make sanitize` + `make tsan` clean
- `RETROSPECTIVE.md` body filled in (all 8 sections)
- Sprint 23 PR opened
- All quality checks clean

### Completion Criteria
- Final cross-ordering capture matches Day 12's output bit-identically (no regressions in Day 13's doc/test sweep)
- `make sanitize` + `make tsan` clean against the full test suite
- Retrospective body written; all 8 sections have content (not stubs)
- PR opened; description references the headline ND/AMD + qg-AMD wall-time numbers
- `make format && make lint && make test && make sanitize && make tsan` clean

---

## Sprint 23 Summary

**Total estimated hours:** 6 + 8 + 8 + 8 + 6 + 6 + 6 + 6 + 8 + 8 + 8 + 6 + 8 + 8 = 100 hours

**Item-to-day mapping:**
| Item | Days | Hours |
|------|------|-------|
| 1: Cholesky-via-ND fixture swap | Day 1 | 4 |
| 2: Element absorption + supervariable detection | Days 2-4 | 24 |
| 3: Approximate-degree update | Days 5-6 | 12 |
| 4: ND leaves call qg AMD | Days 7-8 | 12 |
| 5: FM gain-bucket structure | Days 9-11 | 24 |
| 6: Cross-corpus re-bench | Day 12 | 6 |
| 7: Tests + docs + retrospective stub | Days 1, 13, 14 | 18 |

**Headline gates (must pass on Day 14):**
- Pres_Poisson `nnz_nd / nnz_amd ≤ 0.7` (was 1.06× in Sprint 22)
- qg-AMD wall time on bcsstk14 ≤ Sprint-22 bitset baseline (was ~30 % bitset-favoured)
- All `bench_day14.txt` nnz(L) rows bit-identical-or-better vs Sprint 22

**Risk flags:**
- Item 2 (24 hrs across Days 2-4) is the load-bearing kernel rewrite; if the per-day cap forces a 4th day on element absorption, Day 5's approximate-degree work slips to Day 6 and Item 3's two-day budget compresses
- Multi-pass FM (Day 11 exploration) is budgeted as exploratory; if it doesn't move the needle on Pres_Poisson cut quality, the < 0.5× ND/AMD ratio that PROJECT_PLAN.md mentions as the Sprint 22 plan target is *not* in scope and gets deferred to Sprint 24
- The 100-hour estimate has a 12-hour cushion against the 88-hour PROJECT_PLAN.md figure; this is similar to Sprint 22's actual-vs-estimate (134 vs 124) and is the right magnitude for a sprint touching a load-bearing factorisation kernel
