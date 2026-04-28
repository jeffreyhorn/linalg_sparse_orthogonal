# Sprint 22 Plan: Ordering Upgrades — Nested Dissection & Quotient-Graph AMD

**Sprint Duration:** 14 days
**Goal:** Upgrade the ordering stack with nested dissection (ND) for large 2D/3D PDE meshes and replace the bitset-based AMD with a quotient-graph implementation that operates in O(nnz) memory.  Removes the current scaling bottleneck on the AMD path (O(n²/64) memory for the bitset adjacency representation) and adds a fill-reducing ordering that beats AMD on regular meshes by 2-5× in fill on the SuiteSparse Pres_Poisson / bcsstk14 corpus the eigensolver work has been stressing since Sprint 20.

**Starting Point:** A sparse linear algebra library with three fill-reducing reorderings — `SPARSE_REORDER_NONE` / `_RCM` / `_AMD` / `_COLAMD` — defined in `include/sparse_types.h:89-93` and dispatched from the per-factorization opts blocks (`sparse_cholesky_opts_t::reorder`, `sparse_ldlt_opts_t::reorder`, `sparse_lu_opts_t::reorder`, `sparse_qr_opts_t::reorder`).  AMD lives in `src/sparse_reorder.c:421` (`sparse_reorder_amd`) and uses an n×n bitset adjacency representation (O(n²/64) memory, O(n³/64) time).  COLAMD (Sprint 15) provides the column-AMD pattern this sprint mirrors for the ND enum extension — see `src/sparse_colamd.c` for the integration pattern.  The Sprint 14 `sparse_analyze` symbolic phase consumes a permutation from any of the listed orderings and feeds it to the numeric kernel; ND will plug into the same seam.  No graph-partitioning infrastructure exists in the codebase yet — Sprint 22's items 1-2 build it from scratch.

**End State:** `sparse_reorder_nd()` is exposed through a new `SPARSE_REORDER_ND = 4` enum value and routes through every factorization's opts block (Cholesky, LDL^T, LU, QR — the same way COLAMD does today).  The implementation rests on a new `src/sparse_graph.c` (or `_partition.c`) that ships a multilevel vertex-separator algorithm (heavy-edge-matching coarsening + Kernighan-Lin / Fiduccia-Mattheyses refinement + edge-to-vertex-separator conversion) with a `sparse_graph_partition()` entry point.  Recursive nested dissection orders interior nodes of each partition first, then separator nodes, accumulating a full n-element permutation; small subgraphs (≤ a documented threshold) fall back to the new quotient-graph AMD.  The bitset AMD in `src/sparse_reorder.c:421` is replaced by the quotient-graph implementation (Amestoy / Davis / Duff 2004; SuiteSparse AMD reference) with element absorption, supervariable detection, and approximate-degree updates running in O(nnz) memory and O(nnz · α(n)) time on typical fixtures.  All four orderings (RCM / AMD-quotient-graph / COLAMD / ND) are benchmarked head-to-head on the SuiteSparse corpus (nos4, bcsstk04, bcsstk14, Pres_Poisson, Kuu, s3rmt3m3) with fill-in / memory / wall-time numbers captured in `docs/planning/EPIC_2/SPRINT_22/bench_day14.txt`, and the AMD bitset → quotient-graph swap shows ≥ 4× memory reduction on n ≥ 1000 fixtures.

**Time budget:** Each day caps at 12 hours.  The day budgets below sum to ~134 hours — about 10 hours above the 124-hour PROJECT_PLAN.md estimate, providing a safety buffer within the 168-hour (14 × 12) hard ceiling.  The buffer tracks Sprint 21's actual-vs-estimate experience (estimated 124 hrs, shipped at ~133 hrs — the cushion was consumed by the LOBPCG BLOPEX P-update revert on Day 9 and the dispatch-debug round when stale objects briefly produced swapped numbers).  Sprint 22's risk concentration is items 1 and 4: the multilevel partitioner is novel infrastructure (no precedent in the codebase) and the quotient-graph AMD swap touches a load-bearing kernel that every Cholesky / LDL^T / LU / QR call routes through — regression risk is high if the quotient-graph implementation diverges in fill-reduction quality from the bitset version on any corpus matrix.

---

## Day 1: Graph Partitioning — Design & `sparse_graph_t` Infrastructure

**Theme:** Read the multilevel-partitioning literature, design the working graph representation Sprint 22 will iterate on, and stub the `sparse_graph_partition` entry point so Days 2-4 have a target.

**Time estimate:** 10 hours

### Tasks
1. Read the Sprint 14 / Sprint 15 reorder integration: `include/sparse_types.h:89-93` for the enum, `include/sparse_reorder.h` for the public API shape, `src/sparse_reorder.c:421` for the existing bitset AMD, `src/sparse_colamd.c` for the COLAMD-style file layout, and `include/sparse_analysis.h` for how `sparse_analyze` consumes a permutation.  Note the two integration seams Sprint 22 will use: the standalone `sparse_reorder_nd(A, perm)` public function (matches `sparse_reorder_amd` / `_rcm` / `_colamd`) and the `SPARSE_REORDER_ND` enum dispatch in each factorization's analysis phase.
2. Read the multilevel partitioning literature: Karypis & Kumar (1998) "A Fast and Highly Quality Multilevel Scheme" (METIS paper) and George (1973) "Nested Dissection of a Regular Finite Element Mesh".  Key takeaway: the standard pipeline is **coarsen → partition coarsest → uncoarsen with refinement → extract vertex separator from edge separator**, and the partitioner returns a vertex separator (3-way: left part / right part / separator) which nested dissection then orders separator-last.
3. Design `sparse_graph_t` working representation in a new `src/sparse_graph_internal.h`:
   - `n` (vertex count), `xadj` / `adjncy` CSR-like adjacency (O(nnz) — no bitset).
   - Optional `vwgt` (vertex weights — used by the coarsening to track collapsed-vertex masses) and `ewgt` (edge weights — likewise for the multilevel hierarchy).
   - Helpers `graph_from_sparse(const SparseMatrix *)` (symmetric pattern; A + A^T for unsymmetric A is out of scope this sprint — ND ships symmetric-only, mirroring AMD's contract), `graph_free`, `graph_subgraph(parent, vertex_set, child_out)` for the recursive ND.
4. Design `sparse_graph_partition(const sparse_graph_t *G, idx_t *part_out, idx_t *sep_out)` signature: writes `part_out[i] ∈ {0, 1, 2}` (0 = left part, 1 = right part, 2 = separator) and the count of separator vertices via `*sep_out`.  Internal — lives in `sparse_graph_internal.h`.
5. Write the file-header design block in `src/sparse_graph.c` (new file): explain the multilevel three-phase pipeline, the heavy-edge-matching coarsening choice (favoured over random matching because it preserves spectral structure; cited in METIS paper §4), the FM refinement choice (vs Kernighan-Lin: FM is single-pass and asymptotically faster), and the small-graph base case (n ≤ 20 falls through to a brute-force minimum-cut bisection).
6. Add compile-ready stubs: `sparse_graph_partition` returning `SPARSE_ERR_BADARG` (the codebase's "stub in progress" signal — no `SPARSE_ERR_NOT_IMPL` exists), wired into the Makefile / CMakeLists.txt (new source + new test file).
7. Run `make format && make lint && make test` — all clean (no behavior change yet).

### Deliverables
- `src/sparse_graph_internal.h` declaring `sparse_graph_t`, `sparse_graph_partition`, and the conversion helpers
- `src/sparse_graph.c` with the design block + `graph_from_sparse` / `graph_free` implementations + `sparse_graph_partition` stub
- New empty `tests/test_graph.c` skeleton wired into Makefile / CMakeLists.txt
- Compile-ready stubs returning `SPARSE_ERR_BADARG`

### Completion Criteria
- Design block references Karypis & Kumar (1998) and George (1973); the three-phase pipeline is sketched with the heavy-edge-matching / FM / vertex-separator-extraction choices justified
- `graph_from_sparse(nos4)` round-trips through `graph_free` cleanly under ASan (smoke test)
- Makefile + CMakeLists.txt updated; both build the new source and skeleton test
- `make format && make lint && make test` clean

---

## Day 2: Graph Partitioning — Coarsening (Heavy-Edge Matching)

**Theme:** Implement the multilevel hierarchy's first phase — collapse heavy-weight edges to produce a sequence of progressively smaller graphs whose coarsest level is small enough for an exact partition.

**Time estimate:** 10 hours

### Tasks
1. Implement `graph_coarsen_heavy_edge_matching`: walk vertices in random order (deterministic seed for reproducibility — match Sprint 21 LOBPCG's golden-ratio convention), for each unmatched vertex pick the unmatched neighbour with the heaviest connecting edge, and collapse the pair.  Returns the coarsened `sparse_graph_t` and a `cmap` array mapping each fine-vertex to its coarse-vertex index (needed by the uncoarsening phase on Day 4).
2. Implement vertex/edge weight aggregation under matching:
   - `vwgt_coarse[c] = sum of vwgt_fine[i] for i mapped to c`
   - `ewgt_coarse[c1, c2] = sum of ewgt_fine[i, j] for matching pairs` — handle parallel edges via a hash-or-sort merge in the adjacency construction.
3. Build the multilevel hierarchy: keep coarsening until `n_coarsest ≤ MAX(20, n_orig / 100)` or a coarsening pass fails to halve n (METIS's stop condition).  Cap the number of levels at `log2(n) + 5` as a defensive bound.  Store the hierarchy as an array of `sparse_graph_t *` plus the `cmap` array per level.
4. Helper `graph_hierarchy_free` to release the hierarchy in reverse order.
5. Tests in `tests/test_graph.c`:
   - 2D 5×5 grid (n=25, 80 edges): coarsening reduces n by ~half per level until base case; vertex-weight sum is conserved across levels.
   - 1D path graph (n=20): coarsening still halves n correctly even when the matching is forced (every other edge is heavy).
   - Disconnected fixture (two cliques joined by a single edge): heavy-edge matching prefers within-clique edges first.
6. Run `make format && make lint && make test && make sanitize` — all clean.

### Deliverables
- `graph_coarsen_heavy_edge_matching` + `graph_hierarchy_free` in `src/sparse_graph.c`
- 3+ unit tests in `tests/test_graph.c`
- Vertex/edge weight aggregation invariants documented in the function-level doc comments

### Completion Criteria
- 5×5 grid coarsens deterministically (seed-controlled) to a graph with n ≤ 13 in one step, ≤ 7 in two
- Vertex-weight sum is preserved across all coarsening levels (asserted in tests)
- `make format && make lint && make test && make sanitize` clean

---

## Day 3: Graph Partitioning — Initial Partition & FM Refinement

**Theme:** Bisect the coarsest graph and define the Fiduccia-Mattheyses refinement pass that the Day 4 uncoarsening will replay at every level.

**Time estimate:** 10 hours

### Tasks
1. Implement the coarsest-graph bisection.  At n_coarsest ≤ 20, brute-force minimum-cut bisection is tractable:
   - Enumerate balanced partitions (|P0| - |P1| within ±1 of vertex-weight balance) by walking the partition number 0..2^n_coarsest - 1; track the cut weight; pick the minimum.  O(2^n) but n ≤ 20 means ~10^6 evaluations.
   - For n_coarsest in [21, 40], fall back to a greedy graph-growing partition (start at a peripheral vertex, BFS until half the vertex weight is consumed) — this is the GGGP heuristic from METIS §3.
2. Implement the FM (Fiduccia-Mattheyses) refinement pass.  FM is single-pass per partition, complexity O(|E|), and improves the cut by moving boundary vertices that yield the largest gain (gain = decrease in cut weight if vertex is moved to the other partition).  Bookkeeping:
   - `gain_buckets[]` — boundary-vertex gain bucket structure (hash or fixed-size array indexed by bounded gain).
   - `locked[]` — vertices already moved this pass cannot move again.
   - Pass terminates when no positive-gain unlocked move exists; rollback to the best partition seen during the pass if the final state is worse.
3. Tests in `tests/test_graph.c`:
   - Brute-force bisection on a known 8-vertex graph: verify the cut equals the analytically-computed minimum.
   - FM refinement on a 30-vertex random graph: starting from a deliberately-bad random partition, FM reduces the cut to within 10% of the brute-force minimum on at least 9 of 10 seeds (single-pass FM is not optimal — multi-pass would close the gap, but Day 4's uncoarsening replays FM at every level which serves the same purpose).
4. Run `make format && make lint && make test && make sanitize` — all clean.

### Deliverables
- `graph_bisect_coarsest` (brute-force + GGGP fallback)
- `graph_refine_fm` (single-pass Fiduccia-Mattheyses with rollback-on-regress)
- 2+ unit tests covering both routines

### Completion Criteria
- Brute-force bisection on the 8-vertex fixture matches the analytic minimum exactly
- FM refinement reduces a deliberately-bad random partition's cut weight on every test seed
- Refinement pass complexity is O(|E|) per pass (asserted via no nested vertex iteration)
- `make format && make lint && make test && make sanitize` clean

---

## Day 4: Graph Partitioning — Uncoarsening & Vertex-Separator Extraction

**Theme:** Project the coarsest-level partition back through the multilevel hierarchy, refining at each level via FM, then convert the resulting edge separator into the vertex separator that nested dissection consumes.

**Time estimate:** 10 hours

### Tasks
1. Implement `graph_uncoarsen`: walk the hierarchy from coarsest back to the original fine graph.  At each level, project the partition through the `cmap` (each coarse vertex becomes its fine-vertex preimage) and re-run `graph_refine_fm` to clean up the projected boundary.  Total complexity: O(|E_orig|) summed over levels (geometric).
2. Implement `graph_edge_separator_to_vertex_separator`: given a fine-grained 2-way partition, compute the boundary vertices on the smaller side of the cut and call them the vertex separator.  Returns the 3-way partition (`part[i] ∈ {0, 1, 2}`).  Choose the smaller-side convention (matches METIS): a separator on the smaller part minimises the recursive ND tree's height inflation.
3. Wire `sparse_graph_partition` end-to-end: `graph_coarsen` (Day 2) → `graph_bisect_coarsest` (Day 3) → `graph_uncoarsen` (this day) → `graph_edge_separator_to_vertex_separator` (this day).  Replace the `SPARSE_ERR_BADARG` stub from Day 1 with the real implementation.
4. Tests in `tests/test_graph.c`:
   - 2D 10×10 grid: partition produces |separator| ≈ 10 (the natural row/column cut), |left| ≈ |right| ≈ 45.  Asserted with ±20% tolerance to absorb FM stochasticity.
   - 3D 5×5×5 mesh (n=125): partition produces |separator| ≈ 25 (a planar cut through the centre).
   - Disconnected fixture (two K_10 cliques + single edge): partition cuts the bridge edge, |separator| = 1.
5. Verify partition invariant: every edge of the original graph either has both endpoints in the same `part[i]` or has at least one endpoint in the separator (no left-right edges).  Asserted in tests.
6. Run `make format && make lint && make test && make sanitize` — all clean.

### Deliverables
- `graph_uncoarsen` projecting partitions through the hierarchy with per-level FM refinement
- `graph_edge_separator_to_vertex_separator` (smaller-side convention)
- `sparse_graph_partition` end-to-end (real implementation replacing the Day 1 stub)
- 3+ unit tests covering 2D mesh / 3D mesh / disconnected fixtures

### Completion Criteria
- 10×10 grid partition produces |separator| ≤ 12 on every test seed (10 is optimal, +20% headroom for FM noise)
- 3D mesh partition produces a balanced 3-way split with |separator| ≤ 30 on the 5×5×5 fixture
- Partition invariant (no left-right edges) holds on every test fixture
- `make format && make lint && make test && make sanitize` clean

---

## Day 5: Graph Partitioning — Stress Tests, Edge Cases, Determinism

**Theme:** Push the partitioner against pathological fixtures (singletons, disconnected, dense, bipartite) and lock in the deterministic-seed contract that nested dissection will rely on for reproducible orderings.

**Time estimate:** 8 hours

### Tasks
1. Edge-case tests in `tests/test_graph.c`:
   - n = 1 (single vertex): partition trivially returns `part = [2]` (the one vertex is the separator) and `*sep_out = 1`.  No coarsening occurs.
   - n = 2 with one edge: partition returns `part[0] = 0, part[1] = 1, *sep_out = 0` (no separator needed).
   - Empty graph (n vertices, 0 edges): partition splits arbitrarily — assert `|part0| + |part1| + |sep| == n` and `|sep| == 0`.
   - Complete graph K_20: separator can't be small — assert `|sep| ≥ 5` (the dense connectivity forces a large cut).
   - Bipartite K_{10,10}: separator is one of the bipartition sides minus one vertex — assert `|sep| ≤ 11`.
2. Determinism contract: same input + same seed → same partition.  Test by running `sparse_graph_partition` twice on the same fixture and asserting `memcmp(part1, part2, n * sizeof(idx_t)) == 0`.  Document the seed source in the function-level doc comment.
3. SuiteSparse smoke: run `sparse_graph_partition` on the bcsstk14 (n = 1806) and Pres_Poisson (if loadable) symmetric patterns; assert convergence (no allocation failures, finite separator sizes).  Capture timing — should complete in < 100ms on each.
4. Profiling pass: if `sparse_graph_partition` exceeds 100ms on bcsstk14, profile with `sample` (macOS) or `perf` (Linux) and identify the hot spot.  Most likely culprit: the FM gain-bucket update — METIS's bucket structure is O(1) amortised; a naive linked-list version is O(|E|) per move.  Defer the optimisation to Day 14 if it doesn't blow the budget here.
5. Run `make format && make lint && make test && make sanitize` — all clean.

### Deliverables
- 5+ edge-case tests in `tests/test_graph.c`
- Deterministic-partition contract test
- SuiteSparse smoke results documented in `tests/test_graph.c` comments

### Completion Criteria
- All 5 edge cases produce expected partitions without crashes / asserts
- Determinism contract verified (same input + same seed → bit-identical partition)
- SuiteSparse smoke completes in < 100 ms on bcsstk14 (or has an explicit profiling note in `bench_day14.txt` deferred follow-up)
- `make format && make lint && make test && make sanitize` clean

---

## Day 6: Nested Dissection — Recursive Driver & Permutation Assembly

**Theme:** Compose the recursive ND algorithm on top of `sparse_graph_partition`: order interior nodes of each partition first, then separator nodes, accumulating an n-element permutation.  Includes the small-graph base case (fall through to AMD) and a 2D-grid sanity test.

**Time estimate:** 10 hours

### Tasks
1. Design `sparse_reorder_nd_recursive` in `src/sparse_reorder.c` (or a new `src/sparse_reorder_nd.c` file alongside `src/sparse_colamd.c`):
   - Inputs: a `sparse_graph_t *` subgraph + a `vertex_id_map[]` (parent-graph indices for the subgraph's vertices) + an output `perm[]` cursor + a `next_perm_pos` counter passed by reference.
   - If `n_subgraph ≤ ND_BASE_THRESHOLD` (provisional 100; will tune in Day 9): call the new quotient-graph AMD on the subgraph and append its permutation to `perm[]` at `next_perm_pos`, advance the cursor, return.  (Until Day 12, AMD here means the existing bitset AMD.)
   - Otherwise: `sparse_graph_partition` to get `part[]` + `*sep_out`; build the two child subgraphs via `graph_subgraph`; recurse left, recurse right, then append the separator vertices to `perm[]` at the end (this is the "separator last" rule that gives ND its fill-reducing power).
2. Implement the public-facing `sparse_reorder_nd(const SparseMatrix *A, idx_t *perm)`: build the root graph via `graph_from_sparse(A)`, allocate the permutation cursor, call `sparse_reorder_nd_recursive` from position 0, free the graph.  Same signature as `sparse_reorder_amd` / `sparse_reorder_rcm`.
3. Tests in a new `tests/test_reorder_nd.c`:
   - 2D 4×4 grid (n=16): verify the produced permutation is a valid permutation (`memset(seen, 0, n); for (i) seen[perm[i]] = 1; for (i) assert(seen[i])`); spot-check the separator block is at the tail.
   - 2D 10×10 grid: ND fill-in (count of nnz in the symbolic Cholesky factor under the produced permutation, via `sparse_analyze`) is strictly less than AMD fill-in on the same matrix.  Target: ≥ 1.5× reduction (METIS-like grids show 2-3×).
   - 1D path graph (n=20): ND degenerates gracefully — separators are single vertices and the resulting permutation may not beat AMD, but it must not crash and must remain a valid permutation.
4. Wire the new test file into Makefile / CMakeLists.txt.
5. Run `make format && make lint && make test && make sanitize` — all clean.

### Deliverables
- `sparse_reorder_nd_recursive` (internal) and `sparse_reorder_nd` (public function in `src/sparse_reorder.c` or `_nd.c`)
- New `tests/test_reorder_nd.c` with 3+ tests
- Test files wired into Makefile + CMakeLists.txt
- `ND_BASE_THRESHOLD` documented inline (provisional 100; tuned in Day 9)

### Completion Criteria
- 4×4 grid produces a valid permutation with the separator block at the tail
- 10×10 grid ND fill-in is ≥ 1.5× lower than AMD fill-in on the symbolic Cholesky factor
- `make format && make lint && make test && make sanitize` clean

---

## Day 7: Nested Dissection — `sparse_analyze` Integration & SuiteSparse Validation

**Theme:** Wire the new ND ordering into the Sprint 14 symbolic-analysis machinery so every direct factorization (Cholesky / LDL^T / LU / QR) can request it via the same opts-block dispatch the existing orderings already use.  Validate fill reduction on the SuiteSparse Pres_Poisson fixture (the project's canonical 2D-PDE benchmark target).

**Time estimate:** 10 hours

### Tasks
1. Read `src/sparse_analysis.c` to understand how the existing `SPARSE_REORDER_RCM` / `_AMD` / `_COLAMD` enum values are dispatched in `sparse_analyze` (probably a switch on `opts->reorder` calling the right `sparse_reorder_*` function).  Sketch the diff for a new `case SPARSE_REORDER_ND: return sparse_reorder_nd(A, perm);` arm — Day 8 actually adds the enum value; today the wiring is ad-hoc (call `sparse_reorder_nd` directly in tests).
2. Tests in `tests/test_reorder_nd.c`:
   - **Pres_Poisson fill reduction.** Load `tests/data/suitesparse/Pres_Poisson.mtx` if available (skip with a documented note if not).  Compute the symbolic Cholesky factor under each of `SPARSE_REORDER_NONE` / `_RCM` / `_AMD` / `sparse_reorder_nd`; report the four nnz counts.  Assert `nnz_nd < 0.5 × nnz_amd` (Pres_Poisson is a 2D Poisson-on-irregular-grid matrix where ND's geometric advantage is large).
   - **bcsstk14 fill reduction.** Same comparison on bcsstk14 (n = 1806; structural mechanics matrix).  Looser threshold: `nnz_nd ≤ 1.2 × nnz_amd` — bcsstk14 has irregular sparsity that doesn't favour ND as strongly as Pres_Poisson does.
   - **Cross-permutation determinism.** `sparse_reorder_nd` produces the same permutation on the same input across calls (re-run the determinism contract check from Day 5 at the public-API level).
3. Wire ND into one factorization end-to-end as a smoke test — pick Cholesky (the simplest):
   - In `tests/test_reorder_nd.c`, call `sparse_cholesky_factor_opts` with `opts.reorder = SPARSE_REORDER_AMD` (the existing path) and then with a manual `sparse_reorder_nd` followed by a `SPARSE_REORDER_NONE` factor (since the enum value isn't wired until Day 8).  Compare the resulting solve residuals — both should be ≤ 1e-12 (the factorisation is numerically equivalent regardless of ordering).
   - This is the bridge: Day 8 will replace the manual ND-then-NONE bridge with proper enum dispatch.
4. Run `make format && make lint && make test && make sanitize` — all clean.

### Deliverables
- 3+ Pres_Poisson / bcsstk14 / determinism tests in `tests/test_reorder_nd.c`
- Cholesky-via-ND smoke test (manual ordering, factor with SPARSE_REORDER_NONE) confirming numerical equivalence

### Completion Criteria
- Pres_Poisson `nnz_nd < 0.5 × nnz_amd` passes (or a documented note in the test if Pres_Poisson is unavailable)
- bcsstk14 `nnz_nd ≤ 1.2 × nnz_amd` passes
- Determinism check at the public-API level passes
- Cholesky-via-ND solve residual ≤ 1e-12 on a known fixture
- `make format && make lint && make test && make sanitize` clean

---

## Day 8: `SPARSE_REORDER_ND` Enum Wiring

**Theme:** Add the public enum value, wire it through every factorization's analysis dispatch, update the public headers and README so callers can request ND the same way they request AMD today.

**Time estimate:** 8 hours

### Tasks
1. Add `SPARSE_REORDER_ND = 4` to the enum in `include/sparse_types.h` (line 92 area).  Document with the same one-liner style as the existing values (`/**< Nested Dissection (multilevel vertex separator) ordering */`).
2. Add `sparse_reorder_nd(const SparseMatrix *A, idx_t *perm)` to `include/sparse_reorder.h` next to the existing `sparse_reorder_amd` declaration (line 78 area).  Doxygen-comment matching the AMD entry's style.
3. Wire the enum dispatch into every factorization's analysis phase:
   - `src/sparse_analysis.c` — the central dispatch.  Add `case SPARSE_REORDER_ND: return sparse_reorder_nd(A, perm);`.
   - `src/sparse_cholesky.c` — verify `sparse_cholesky_factor_opts` already routes through `sparse_analyze` (Sprint 14 shipped this).  If yes: free.  If it has a parallel switch: add the ND arm.
   - `src/sparse_ldlt.c`, `src/sparse_lu.c`, `src/sparse_qr.c` — same checks.
4. Update the public-header doxygen examples that enumerate orderings (`include/sparse_cholesky.h` line 22 area, `include/sparse_ldlt.h` line 25 area, `include/sparse_lu.h` line 42 area, `include/sparse_qr.h` if it has one).  Add `SPARSE_REORDER_ND` to each list with a one-line note ("for 2D / 3D PDE meshes; see `sparse_reorder.h` for details").
5. Update the README ordering table.  Look for the `### Reordering` or similar section; add an `ND` row with the fill-reduction characteristic ("best on regular meshes; falls back to AMD on small subgraphs").
6. Replace the Day 7 manual-ordering smoke test in `tests/test_reorder_nd.c` with the proper enum-dispatch form: `sparse_cholesky_opts_t opts = { .reorder = SPARSE_REORDER_ND }; sparse_cholesky_factor_opts(A, &opts, ...)`.  Confirm the same numerical result.
7. Add a new test in `tests/test_reorder_nd.c`: `sparse_lu_opts_t opts = { .reorder = SPARSE_REORDER_ND }; sparse_lu_factor_opts(...)` — verify ND dispatch works on the LU path too (the analyze phase is the same code, but cheap insurance against typos in the `case` arm).
8. Run `make format && make lint && make test && make sanitize` — all clean.

### Deliverables
- `SPARSE_REORDER_ND = 4` in `include/sparse_types.h`
- `sparse_reorder_nd` in `include/sparse_reorder.h`
- Enum dispatch arm in `src/sparse_analysis.c` (and any parallel sites in the per-factorization sources)
- Updated public-header doxygen examples
- README ordering section refreshed
- 2+ enum-dispatch tests (Cholesky + LU)

### Completion Criteria
- All four factorizations (Cholesky / LDL^T / LU / QR) accept `opts.reorder = SPARSE_REORDER_ND` and produce a numerically-correct factorisation on a known fixture
- README ordering table lists the four orderings with one-liner characteristics
- `make format && make lint && make test && make sanitize` clean

---

## Day 9: Nested Dissection — Cross-Ordering Benchmark & Threshold Tuning

**Theme:** Capture fill-in / wall-time / memory numbers for the four orderings (RCM / AMD-bitset / COLAMD / ND) on the SuiteSparse corpus, and tune `ND_BASE_THRESHOLD` based on the measured AMD-vs-ND crossover.

**Time estimate:** 8 hours

### Tasks
1. Extend the existing `benchmarks/bench_fillin.c` (or create `benchmarks/bench_reorder.c` if it doesn't exist) with an ND row.  For each of nos4 / bcsstk04 / bcsstk14 / Kuu / Pres_Poisson / s3rmt3m3 (the SuiteSparse corpus the eigensolver work uses), report:
   - n
   - For each of NONE / RCM / AMD / COLAMD / ND: nnz(L) under symbolic Cholesky, ordering wall_ms, factor wall_ms.
2. Run the bench and capture to `docs/planning/EPIC_2/SPRINT_22/bench_day9_nd.txt` (CSV + a human-readable table).  Schema: `matrix,n,reorder,nnz_L,reorder_ms,factor_ms`.
3. Tune `ND_BASE_THRESHOLD`.  The Day 6 stub set this to 100; the actual crossover (where AMD beats ND on a small subgraph because the multilevel overhead outweighs the geometric gain) depends on the matrix family.  Sweep ND_BASE_THRESHOLD ∈ {20, 50, 100, 200, 500} on bcsstk14 and Pres_Poisson; pick the value that minimises total fill on both.  Document the sweep in `bench_day9_nd.txt`.
4. Update `docs/planning/EPIC_2/SPRINT_17/PERF_NOTES.md` (or create `SPRINT_22/PERF_NOTES.md` if a new file is preferred) with a new "Reordering comparison" section summarising the Day 9 capture.  Cite the matrices and the headline ND-vs-AMD fill ratios.
5. Run `make format && make lint && make test` — all clean.

### Deliverables
- `benchmarks/bench_fillin.c` (or `bench_reorder.c`) extended with an ND row across the four orderings
- `docs/planning/EPIC_2/SPRINT_22/bench_day9_nd.txt` capture (CSV + human-readable)
- `ND_BASE_THRESHOLD` tuned with sweep data documented
- PERF_NOTES.md (Sprint 17 or new) updated with the reordering-comparison section

### Completion Criteria
- All six SuiteSparse fixtures load and produce the four-ordering comparison row
- ND beats AMD on Pres_Poisson by ≥ 2× nnz(L) (the canonical 2D-PDE benchmark)
- ND_BASE_THRESHOLD sweep data shows a clear crossover (or documents that no crossover was found and the default holds)
- `make format && make lint && make test` clean

---

## Day 10: Quotient-Graph AMD — Design & Read the Reference

**Theme:** Read the Amestoy / Davis / Duff (2004) quotient-graph AMD paper and SuiteSparse's `AMD` source as the reference; design the in-place data structures the new `src/sparse_reorder.c` AMD will use; identify the seam where the bitset implementation will be replaced.

**Time estimate:** 10 hours

### Tasks
1. Read Amestoy / Davis / Duff (2004) "Algorithm 837: AMD, an approximate minimum degree ordering algorithm" (TOMS).  Key mechanisms to internalise:
   - **Quotient graph representation** — vertices and "elements" (clusters of eliminated variables) stored in a single integer-array workspace shared across the algorithm.  No bitset.
   - **Element absorption** — when a vertex's adjacency reduces to a single element, the vertex is absorbed into that element (compaction).
   - **Supervariable detection** — vertices with identical adjacency are merged; the minimum-degree decision is made on supervariables, dramatically shrinking the active set on PDE-like matrices.
   - **Approximate degree update** — exact minimum-degree update is O(nnz) per pivot; AMD's approximation is O(adjacency-of-adjacency), much cheaper on average.
2. Read the existing bitset AMD in `src/sparse_reorder.c:421` (`sparse_reorder_amd`).  Identify:
   - Where the n×n bitset is allocated (the O(n²/64) memory hotspot).
   - The pivot-selection loop (where minimum degree is computed).
   - The adjacency-update loop after each pivot (where the O(n³/64) time cost lives).
   - The output: a permutation array.  This is the contract the quotient-graph version must preserve.
3. Design the new internal layout in a fresh `src/sparse_reorder_amd_qg.c` file (or reuse `_reorder.c` with a clear section header — TBD based on file size).  Single-workspace design (Davis 2006 style):
   - `iw[]` integer workspace of size ≈ 5 × nnz + 2n; holds the quotient-graph adjacency lists, the element lists, the vertex-degree array, the supervariable-link array, and the elimination-order output.
   - Helper offsets / pointers into `iw[]` (no separate per-array mallocs — Davis's design avoids them for cache locality).
4. Write the file-header design block in the new file: cite Amestoy/Davis/Duff (2004), explain the quotient-graph representation vs the bitset baseline, document the O(nnz) memory bound and the approximate-degree-update rationale, and call out the validation strategy (Day 12 — verify the new and old produce *equivalent fill*, not necessarily bit-identical permutations, since AMD's pivot tie-breaking differs).
5. Add compile-ready stubs: a new internal entry point `sparse_reorder_amd_qg(const SparseMatrix *A, idx_t *perm)` returning `SPARSE_ERR_BADARG`.  Don't touch `sparse_reorder_amd` yet — the swap happens on Day 12.
6. Run `make format && make lint && make test` — all clean.

### Deliverables
- New `src/sparse_reorder_amd_qg.c` with file-header design block + stub
- Sprint 22 / Day 10 design notes (inline in the source file or as a sidebar in `bench_day9_nd.txt`'s notes)
- Identification of the bitset-AMD seam in `src/sparse_reorder.c`

### Completion Criteria
- Design block cites Amestoy/Davis/Duff (2004) and SuiteSparse AMD; the quotient-graph / element-absorption / supervariable-detection / approximate-degree mechanisms are each described in 2-3 sentences
- Stub compiles and links into the library; no behavior change yet
- `make format && make lint && make test` clean

---

## Day 11: Quotient-Graph AMD — Core Algorithm Implementation

**Theme:** Implement the quotient-graph elimination loop end-to-end.  The longest task day in the sprint — the algorithm has four interlocking subsystems (quotient-graph maintenance, element absorption, supervariable detection, approximate-degree updates) that must all work together for the output to be correct.

**Time estimate:** 12 hours

### Tasks
1. Implement the workspace allocator: a single `malloc(iw_size)` with documented offsets for each sub-array.  iw_size = `5 * nnz + 6 * n + 1` (Davis 2006 §7); over-allocate by 20% as scratch headroom.
2. Implement quotient-graph maintenance:
   - `qg_init` — populate the initial vertex adjacency lists from the symmetric pattern of A.
   - `qg_absorb_element` — when vertex i's adjacency reduces to a single element e, mark i as absorbed and remove it from the active set.
   - `qg_compact` — periodic compaction of the workspace when fragmentation crosses 50% (Davis's heuristic).
3. Implement element absorption:
   - When pivot vertex p is eliminated, its adjacency becomes a new element E_p.
   - Absorb older elements into E_p when their members are all in p's adjacency.
   - Update remaining vertices' adjacency lists to reference E_p instead of p's individual neighbours.
4. Implement supervariable detection:
   - After a pivot, scan vertices with identical hash-of-adjacency (Davis's hash-based bucket).
   - Merge supervariables in O(|supervariable| · |adjacency|) per merge.
   - The minimum-degree pivot selection then operates on supervariables, not individual vertices — major asymptotic win on PDE-like matrices.
5. Implement approximate-degree updates:
   - Davis's "AMD-AT" approximate degree formula: `d_approx(i) = |adj(i, V)| + |adj(i, E)|` where the second term sums over elements adjacent to i, capped to avoid double-counting.
   - Cheap to update incrementally after each pivot; substantially cheaper than exact degree, with negligible fill-quality loss on the SuiteSparse benchmark corpus.
6. Replace the `SPARSE_ERR_BADARG` stub from Day 10 with the full implementation in `sparse_reorder_amd_qg`.  Output: the same n-element permutation array contract that `sparse_reorder_amd` (bitset) produces.
7. Wire a *parallel* test in `tests/test_reorder.c` (or a new `tests/test_reorder_amd_qg.c`):
   - Call both `sparse_reorder_amd` (existing bitset) and `sparse_reorder_amd_qg` (new) on nos4 / bcsstk04 / bcsstk14.
   - Compute symbolic Cholesky nnz under each permutation.
   - Assert `nnz_qg ≤ 1.05 × nnz_bitset` (within 5% — AMD pivot tie-breaking differs across implementations, so bit-identical permutations are not expected, but fill quality should match).
8. Run `make format && make lint && make test && make sanitize` — all clean.  The bitset AMD is still the production path; today's work runs alongside it for validation.

### Deliverables
- `sparse_reorder_amd_qg` end-to-end implementation in `src/sparse_reorder_amd_qg.c`
- Parallel-comparison test in `tests/test_reorder_amd_qg.c` (or extension of `test_reorder.c`)
- Memory bound documented inline (workspace size formula + 20% headroom)

### Completion Criteria
- `sparse_reorder_amd_qg` produces a valid permutation on every corpus matrix (nos4 / bcsstk04 / bcsstk14)
- `nnz_qg ≤ 1.05 × nnz_bitset` on every corpus matrix (fill-quality parity check)
- Workspace allocation is a single malloc; verified under ASan with no leaks
- `make format && make lint && make test && make sanitize` clean

---

## Day 12: Quotient-Graph AMD — Replace the Bitset, Cross-Validate

**Theme:** Make the quotient-graph implementation the production AMD path.  The bitset version is removed (or kept behind a debug compile flag for one sprint as a fallback).  All existing AMD-using tests must pass unchanged.

**Time estimate:** 10 hours

### Tasks
1. Replace `sparse_reorder_amd`'s body in `src/sparse_reorder.c:421` with a call to `sparse_reorder_amd_qg`.  Preserve the public API contract (signature unchanged; identical permutation-array output convention).
2. Decide on the bitset-AMD fallback:
   - **Option A (recommended):** delete the bitset implementation entirely.  Simpler tree, no dead code.  The Sprint 22 retrospective documents the swap; if a regression surfaces post-merge, `git revert` brings the bitset back.
   - **Option B:** keep the bitset behind `#ifdef SPARSE_REORDER_AMD_BITSET_FALLBACK` as a one-sprint safety net.  Adds maintenance cost but lets us A/B in production for a sprint.
   - Pick Option A unless something during Day 11 testing made you nervous.  Document the choice in the commit message.
3. Run the full test suite: every existing AMD-using test (`tests/test_cholesky.c`, `tests/test_ldlt.c`, `tests/test_lu.c`, `tests/test_qr.c`, the SuiteSparse fixtures) must pass unchanged.  Failures here mean the quotient-graph implementation diverged from the bitset in a way the Day 11 fill-parity check missed — investigate before continuing.
4. Update existing AMD-related comments to reflect the swap.  Search for "bitset" in `src/sparse_reorder.c` and `include/sparse_reorder.h`; update or delete stale references.
5. Update README's known-limitations section: the "AMD uses O(n²/64) memory" caveat is gone.  Replace with a positive note about the new O(nnz) bound.
6. Run `make format && make lint && make test && make sanitize` — all clean.  Run `make tsan` if available (TSan validates that the new AMD's workspace allocation is single-threaded-clean; the existing AMD never had TSan coverage).
7. Stress test: run AMD on a synthetic 10000×10000 banded matrix that the bitset version would have refused (or taken minutes on).  Should complete in seconds with ≤ 100 MB peak memory under the new implementation.

### Deliverables
- `sparse_reorder_amd` body now calls `sparse_reorder_amd_qg`
- Bitset implementation removed (Option A) or guarded (Option B) — choice documented
- All existing AMD-using tests pass unchanged
- README known-limitations section updated

### Completion Criteria
- `make test` is fully green — no AMD-using test regresses
- `make sanitize` passes with the new AMD as the production path
- 10000×10000 banded stress test completes in < 5 sec, ≤ 100 MB RSS
- `make format && make lint && make test && make sanitize` clean

---

## Day 13: Quotient-Graph AMD — Bench, Memory, Scaling

**Theme:** Quantify the win — capture the AMD bitset → quotient-graph speedup and memory reduction across the SuiteSparse corpus, with a focus on the n ≥ 1000 fixtures where the bitset's O(n²/64) was painful.

**Time estimate:** 8 hours

### Tasks
1. Restore the bitset AMD temporarily for benchmarking (revert the Day 12 swap on a scratch branch, or use Option B's `#ifdef` if you took that path).  Build two binaries: one with bitset AMD, one with quotient-graph AMD.  Use `git stash` + `make clean && make` toggling for the AB if working from a single checkout.
2. Benchmark bitset vs quotient-graph on each of nos4 / bcsstk04 / bcsstk14 / Pres_Poisson / Kuu / s3rmt3m3 / a synthetic 5000×5000 banded.  Capture for each:
   - Wall time of `sparse_reorder_amd`
   - Peak RSS during the call (via `getrusage(RUSAGE_SELF)` snapshot before / after; document the macOS ru_maxrss-in-bytes vs Linux ru_maxrss-in-kbytes difference inline)
   - Resulting nnz(L) under symbolic Cholesky (sanity: should be within 5% across implementations)
3. Capture to `docs/planning/EPIC_2/SPRINT_22/bench_day13_amd_qg.txt` (CSV + human-readable).  Schema: `matrix,n,impl,reorder_ms,peak_rss_mb,nnz_L`.
4. Sanity-check the headline numbers:
   - bcsstk14 (n = 1806): bitset RSS ≈ n² / 64 / 2^20 ≈ 0.4 MB (small fixture, bitset wins on tiny matrices).  Don't expect a memory win here; do expect ≤ 2× wall-time delta.
   - Synthetic 5000×5000: bitset RSS ≈ 5000² / 64 / 2^20 ≈ 0.4 MB still.  Not impressive.
   - Synthetic 50000×50000 (if loadable): bitset RSS ≈ 39 MB; quotient-graph should be ≤ 5 MB.  The headline 8× reduction.
   - If the corpus doesn't include a fixture large enough to surface the bitset cost, generate a synthetic one and document the comparison.
5. Update PERF_NOTES.md (the file touched on Day 9) with an "AMD memory reduction" sub-section.  Include the synthetic-large-matrix measurement explicitly — that's the fixture where the swap actually moves the needle.
6. Restore the production state: quotient-graph as the only AMD implementation (commit the Day 12 state cleanly).
7. Run `make format && make lint && make test && make sanitize` — all clean.

### Deliverables
- `bench_day13_amd_qg.txt` capture (CSV + human-readable)
- PERF_NOTES.md updated with the AMD memory-reduction headline
- Synthetic large-fixture comparison documented (the one that actually shows the bitset's failure mode)

### Completion Criteria
- Bitset vs quotient-graph numbers captured for all corpus fixtures plus at least one synthetic large fixture
- nnz(L) parity within 5% across implementations on every corpus row
- Synthetic large fixture (n ≥ 50000 or whichever surfaces the bitset's O(n²) blow-up) shows ≥ 4× memory reduction for quotient-graph
- `make format && make lint && make test && make sanitize` clean

---

## Day 14: Final Integration, Cross-Ordering Capture, Retrospective Stub

**Theme:** Close out Sprint 22 with the four-ordering head-to-head capture (RCM / AMD-quotient-graph / COLAMD / ND), update the project plan / README / `docs/algorithm.md` to reflect the new orderings, and seed the Sprint 22 retrospective with the day-by-day metrics so the post-sprint write-up is mechanical.

**Time estimate:** 10 hours

### Tasks
1. Run the final cross-ordering benchmark sweep — same `bench_fillin.c` / `bench_reorder.c` driver from Day 9 but now with the production quotient-graph AMD instead of the bitset.  Capture to `docs/planning/EPIC_2/SPRINT_22/bench_day14.txt` (the canonical sprint capture).  Schema unchanged from Day 9: `matrix,n,reorder,nnz_L,reorder_ms,factor_ms`.  Six matrices × four orderings = 24 rows.
2. Compare the Day 14 numbers to the Day 9 numbers (which used bitset AMD).  The ND and COLAMD rows should be identical (those code paths didn't change); the AMD rows should show:
   - Identical or near-identical nnz(L) (fill-quality parity)
   - Equal-or-faster wall time (quotient-graph is asymptotically faster on large matrices)
   - Substantially lower peak RSS on the n ≥ 1000 fixtures (already captured in `bench_day13_amd_qg.txt` — cross-reference here).
3. Update `docs/algorithm.md` with two new sub-sections:
   - **Nested dissection.** Describe the multilevel pipeline (heavy-edge matching → coarsest bisection → uncoarsen with FM → vertex-separator extraction → recursive ordering with separator-last), the `ND_BASE_THRESHOLD` cutover, and the SuiteSparse fill-reduction headline.
   - **Quotient-graph AMD.** Describe the swap-out from bitset to quotient-graph, the memory bound move from O(n²/64) to O(nnz), the element-absorption / supervariable / approximate-degree mechanisms, and the synthetic-large-matrix headline from `bench_day13_amd_qg.txt`.
4. Update `docs/planning/EPIC_2/PROJECT_PLAN.md`:
   - Mark the Sprint 22 row in the Summary table with **Complete** and the actual hours.
   - Update the Sprint 22 detail section with status ticks per item if your project plan format uses them (Sprint 21's section has them; mirror the convention).
5. Stub `docs/planning/EPIC_2/SPRINT_22/RETROSPECTIVE.md` with the standard headers (Sprint budget / Calendar elapsed / Goal recap / What went well / What went poorly / Lessons / DoD verification) and the day-by-day metrics from this sprint.  Leave the prose for post-sprint — the goal here is to make the post-sprint write-up mechanical.
6. Run `make format && make lint && make test && make sanitize` — all clean.  Run `make tsan` if available — full eigensolver + ordering combo should remain TSan-clean (the sprint touched no concurrency code, but the extended workspace allocations are worth a sanitiser pass).
7. Push the sprint-22 branch and open the PR (keep the format the round-1 PR for sprint-21 used: PR title under 70 chars, body covers what changed and the test plan).

### Deliverables
- `docs/planning/EPIC_2/SPRINT_22/bench_day14.txt` final cross-ordering capture
- `docs/algorithm.md` updated with ND + quotient-graph AMD subsections
- `docs/planning/EPIC_2/PROJECT_PLAN.md` Sprint 22 row marked **Complete** with actual hours
- `docs/planning/EPIC_2/SPRINT_22/RETROSPECTIVE.md` stubbed with day-by-day metrics
- sprint-22 branch pushed; PR opened against master

### Completion Criteria
- Day 14 cross-ordering CSV has 24 rows (6 matrices × 4 orderings) and at least one ND row beats the corresponding AMD row by ≥ 2× nnz(L) on Pres_Poisson
- `docs/algorithm.md` has the two new subsections; PROJECT_PLAN summary row reflects sprint outcome
- RETROSPECTIVE.md stub is structurally complete (all standard headers present)
- `make format && make lint && make test && make sanitize` clean
- PR opened (URL pasted in the closing commit's body)

---

## Sprint Budget Recap

| Day | Theme | Hours |
|-----|-------|-------|
| 1   | Graph partitioning — design + `sparse_graph_t` + stubs | 10 |
| 2   | Coarsening (heavy-edge matching, multilevel hierarchy) | 10 |
| 3   | Initial coarsest-graph bisection + FM refinement | 10 |
| 4   | Uncoarsening + vertex-separator extraction + end-to-end partitioner | 10 |
| 5   | Stress tests, edge cases, determinism contract | 8 |
| 6   | Nested dissection — recursive driver + permutation assembly | 10 |
| 7   | ND `sparse_analyze` integration + Pres_Poisson / bcsstk14 validation | 10 |
| 8   | `SPARSE_REORDER_ND` enum wiring across all four factorizations | 8 |
| 9   | Cross-ordering benchmark + `ND_BASE_THRESHOLD` tuning | 8 |
| 10  | Quotient-graph AMD — read references + design + stub | 10 |
| 11  | Quotient-graph AMD — core algorithm implementation | 12 |
| 12  | Quotient-graph AMD — swap into production + cross-validate | 10 |
| 13  | Quotient-graph AMD — bench bitset vs QG, memory + scaling | 8 |
| 14  | Final integration, cross-ordering capture, retrospective stub | 10 |

**Total:** 134 hours (about 10 hours above the 124-hour PROJECT_PLAN.md estimate; well within the 168-hour 14×12 ceiling).
