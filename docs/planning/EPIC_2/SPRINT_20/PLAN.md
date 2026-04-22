# Sprint 20 Plan: LDL^T Completion & Symmetric Lanczos Eigensolver

**Sprint Duration:** 14 days
**Goal:** Close out the final Sprint 19 follow-ups for the batched supernodal LDL^T path — the `_with_analysis` shim that enables indefinite matrices and a transparent size-based dispatch through `sparse_ldlt_factor_opts` — then begin the eigensolver work by landing the symmetric Lanczos eigensolver with shift-invert mode.

**Starting Point:** Sprint 19 shipped the SPD path of the batched supernodal LDL^T kernel (`ldlt_csc_detect_supernodes`, `ldlt_dense_factor`, `ldlt_csc_supernode_extract` / `_writeback` / `_eliminate_diag` / `_eliminate_panel`, `ldlt_csc_eliminate_supernodal`) plus the scalar `LdltCsc` row-adjacency index.  Indefinite matrices (KKT-style saddle points, matrices with non-trivial off-block structure) currently fall back to the scalar `ldlt_csc_eliminate` path because the heuristic CSC fill produced by `ldlt_csc_from_sparse` does not cover the supernodal cmod fill rows — the same lesson the Cholesky side learned on Sprint 19 Day 6 for the Kuu regression.  On the eigensolver side, the library today exposes only a tridiagonal QR kernel (`sparse_tridiag_eigvals`) and has no routines for sparse symmetric eigenpairs.

**End State:** `ldlt_csc_from_sparse_with_analysis` pre-allocates the full `sym_L` pattern from `sparse_analysis_t` and is consumed by `ldlt_csc_eliminate_supernodal` so the batched path works end-to-end on indefinite SuiteSparse matrices.  `sparse_ldlt_factor_opts` grows a `backend` selector (AUTO / LINKED_LIST / CSC) and a `used_csc_path` reporting field, mirroring the Sprint 18 Cholesky dispatch; AUTO routes to the CSC supernodal path above `SPARSE_CSC_THRESHOLD` with a structure-aware fallback to the scalar kernel when the indefinite pattern defeats batching.  A new `sparse_eigs.h` public header defines `sparse_eigs_t` + `sparse_eigs_sym()`; the Lanczos backend implements thick-restart Lanczos with full reorthogonalization and shift-invert mode (via the new LDL^T dispatch for symmetric indefinite shifts).  Lanczos is tested on diagonal / tridiagonal / SuiteSparse SPD fixtures, cross-checked against existing SVD (eigenvalues of A^T*A = singular values squared), exercises the new indefinite LDL^T path through shift-invert, and is documented in the README with a new example program.

**Time budget:** Each day caps at 12 hours.  The day budgets below sum to 142 hours — 6 hours above the 136-hour PROJECT_PLAN.md estimate, providing a safety buffer within the 168-hour (14 × 12) hard ceiling.  The budgets track Sprint 19's actual-vs-estimate experience (shipped at ~140 hrs vs 168 hrs estimated, ~17% under).

---

## Day 1: `ldlt_csc_from_sparse_with_analysis` — Design & Symbolic Pattern

**Theme:** Design the LDL^T mirror of `chol_csc_from_sparse_with_analysis` and produce the symbolic pattern the batched supernodal kernel needs on indefinite inputs

**Time estimate:** 10 hours

### Tasks
1. Read `chol_csc_from_sparse_with_analysis` in `src/sparse_chol_csc.c` as the template — understand how it consumes `analysis->sym_L`, scatters A's lower-triangle entries into the pre-allocated slots, and sets `csc->sym_L_preallocated = 1` so `chol_csc_gather`'s fast path fires.
2. Review the indefinite-scope NOTE in `tests/test_sprint19_integration.c` (around line 235-264) that explains why `ldlt_csc_from_sparse`'s heuristic `fill_factor` pattern silently drops supernodal cmod fill rows on KKT-style inputs.  Read the `ldlt_csc_eliminate_supernodal` writeback path in `src/sparse_ldlt_csc.c` to understand which fill rows need to be pre-populated.
3. Decide the symbolic approach:
   - **Option A:** Reuse `analysis->sym_L` (computed for Cholesky) as-is and rely on the fact that the Cholesky pattern is a superset of the LDL^T pattern with 1×1 pivots only.  Fails on 2×2 pivots because symmetric swaps can introduce rows not in the Cholesky pattern.
   - **Option B:** Run a dedicated LDL^T symbolic pass that accounts for potential 2×2 pivot fill.  More correct but requires new infrastructure.
   - **Option C (recommended):** Use `analysis->sym_L` plus a per-column buffer over-allocation (2× the Cholesky row count) to cover BK 2×2 fill.  Matches Sprint 19 Day 4's observation that 2×2 pivots produce bounded extra fill.  Document the bound in the design block.
4. Design the new function signature: `sparse_err_t ldlt_csc_from_sparse_with_analysis(const SparseMatrix *mat, const sparse_analysis_t *analysis, LdltCsc **out);`.  Mirror the existing `ldlt_csc_from_sparse` return-code and ownership conventions.
5. Declare the function in `src/sparse_ldlt_csc_internal.h` with a short header comment pointing back to the design rationale.
6. Write the design block (file-header-style comment) above the function stub in `src/sparse_ldlt_csc.c` explaining: the Option C rationale, the 2×2-pivot fill bound, how it integrates with the supernodal writeback, and the failure mode the heuristic path produces on KKT saddle points.
7. Stub body returning `SPARSE_ERR_NOT_IMPL` so Day 2 has a compiling target.
8. Run `make format && make lint && make test` — all clean.

### Deliverables
- Function signature and design block committed to `src/sparse_ldlt_csc.c`
- Header declaration in `src/sparse_ldlt_csc_internal.h`
- Design block explains the chosen symbolic approach with rationale

### Completion Criteria
- Design block reviewed against `chol_csc_from_sparse_with_analysis`
- Function signature compiles with an `SPARSE_ERR_NOT_IMPL` stub body
- `make format && make lint && make test` clean

---

## Day 2: `ldlt_csc_from_sparse_with_analysis` — Implementation

**Theme:** Implement the pattern pre-allocation + scatter so the new shim produces a valid `LdltCsc` on both SPD and indefinite inputs

**Time estimate:** 10 hours

### Tasks
1. Implement the sym_L pattern pre-allocation:
   - Allocate `L->col_ptr`, `L->row_idx`, `L->values` sized per `analysis->sym_L->col_ptr[n]` with the Day 1 2×2-pivot safety factor applied.
   - Copy the row-index layout from `analysis->sym_L` into `L->row_idx`; zero-initialize `L->values`.
   - Set `csc->sym_L_preallocated = 1` so the LDL^T equivalent of `chol_csc_gather`'s fast path will fire (to be added or confirmed on Day 3).
2. Scatter A's lower-triangle entries into the pre-allocated slots using the same bsearch-into-row-range pattern `chol_csc_from_sparse_with_analysis` uses.  For LDL^T we also need the upper triangle mirrored into the packed storage where `ldlt_csc_symmetric_swap` can find it — confirm by reading `ldlt_csc_scatter_symmetric` and matching its expectations.
3. Initialize `row_adj` / `row_adj_count` / `row_adj_cap` from Sprint 19 Day 8 to the pre-allocated pattern so the scalar fallback path composes cleanly if it needs to run.
4. Unit tests in `tests/test_ldlt_csc.c`:
   - Round-trip: build `LdltCsc` via `ldlt_csc_from_sparse_with_analysis` on a small SPD fixture, compare the resulting `col_ptr` / `row_idx` against `analysis->sym_L` for identity.
   - Scatter correctness: after scatter, reading `A(i, j)` via `ldlt_csc_get` equals the input value for every stored entry.
   - Indefinite smoke: build via the new shim on a small KKT fixture (n ≈ 10, 2×2 saddle structure) and verify `col_ptr` covers the expected fill pattern.
5. Run `make format && make lint && make test && make sanitize` — all clean.

### Deliverables
- `ldlt_csc_from_sparse_with_analysis` implementation producing a valid `LdltCsc` with pre-allocated sym_L
- 3+ unit tests in `test_ldlt_csc.c` covering pattern identity + scatter + indefinite smoke

### Completion Criteria
- All three unit tests pass
- Round-trip on 5+ existing SPD fixtures produces the same factored state (after `ldlt_csc_eliminate_supernodal`) as the heuristic path does on SPD
- `make format && make lint && make test && make sanitize` clean

---

## Day 3: `ldlt_csc_from_sparse_with_analysis` — Indefinite Supernodal Enablement

**Theme:** Wire the new shim into the batched supernodal LDL^T path so indefinite matrices factor through the batched kernel end-to-end

**Time estimate:** 10 hours

### Tasks
1. Identify where `ldlt_csc_eliminate_supernodal` writeback currently drops fill rows on indefinite inputs (the symptom: residual jumps from 1e-15 to 1e-2..1e-6).  Add an assertion or counter in the writeback path to confirm the failure mode before fixing.
2. Update `ldlt_csc_eliminate_supernodal` (or its callees) to take the Day 1 "sym_L is pre-allocated" fast path when `csc->sym_L_preallocated` is set:
   - Writeback writes into existing slots instead of growing columns.
   - Mirrors `chol_csc_gather`'s Sprint 19 Day 6 fast-path: in-place write + zero-pad for dropped positions, `col_ptr` immutable across elimination.
3. Identify a small KKT-style indefinite fixture for cross-check (either the Sprint 18 Day 13 `test_s18_ldlt_csc_kkt_indefinite` pattern or a dedicated Sprint 20 fixture of dim ≈ 20 with both definite and indefinite blocks).  Confirm scalar `ldlt_csc_eliminate` on this fixture produces a residual ≤ 1e-10 — that is the reference.
4. Cross-check batched vs scalar on the indefinite fixture:
   - Factor via `ldlt_csc_from_sparse_with_analysis` + `ldlt_csc_eliminate_supernodal` (batched).
   - Factor via `ldlt_csc_from_sparse` + `ldlt_csc_eliminate` (scalar reference).
   - Assert `s19_ldlt_factor_state_matches` agreement and `‖A·x - b‖ / ‖b‖ ≤ 1e-10` on both paths.
5. Expand `tests/test_ldlt_csc.c` with 3+ indefinite batched tests: pure KKT, mixed definite/indefinite blocks, random indefinite n = 40 with `ldlt_csc_from_sparse_with_analysis` explicitly exercised.
6. Run `make format && make lint && make test && make sanitize` — all clean.

### Deliverables
- `ldlt_csc_eliminate_supernodal` fast-path that respects `sym_L_preallocated` on indefinite fill
- 3+ indefinite batched-supernodal cross-check tests
- Before/after residual numbers on the KKT fixture committed to `docs/planning/EPIC_2/SPRINT_20/bench_day3_indefinite.txt`

### Completion Criteria
- Batched supernodal LDL^T produces round-off residuals on the KKT fixture (previously 1e-2..1e-6)
- Scalar vs batched factors match under `s19_ldlt_factor_state_matches` tolerance
- `make format && make lint && make test && make sanitize` clean

---

## Day 4: Transparent LDL^T Dispatch — API & Backend Selector

**Theme:** Extend `sparse_ldlt_factor_opts` with the AUTO / LINKED_LIST / CSC backend selector and wire the basic dispatch skeleton

**Time estimate:** 10 hours

### Tasks
1. Read `sparse_cholesky_factor_opts` and `sparse_cholesky_opts_t` in `include/sparse_cholesky.h` + `src/sparse_cholesky.c` as the template (Sprint 18 Day 10-11).  The key elements to mirror are:
   - `backend` field with enum `SPARSE_CHOL_BACKEND_AUTO` / `_LINKED_LIST` / `_CSC`.
   - `used_csc_path` result field populated on return.
   - AUTO routes to CSC when `n >= SPARSE_CSC_THRESHOLD`, otherwise linked-list.
   - CSC→linked-list writeback preserves the external `SparseMatrix *L` contract.
2. Extend `sparse_ldlt_opts_t` in `include/sparse_ldlt.h`:
   - Add `sparse_ldlt_backend_t backend` field (enum `SPARSE_LDLT_BACKEND_AUTO` / `_LINKED_LIST` / `_CSC`).
   - Add `int used_csc_path` reporting field.
   - Document every field in the public header.
3. Update `sparse_ldlt_factor_opts` in `src/sparse_ldlt.c` to route based on the backend selector:
   - `LINKED_LIST` → existing path unchanged.
   - `CSC` → goes through the new CSC pipeline (stub body returning `SPARSE_ERR_NOT_IMPL` for now; full implementation on Day 5).
   - `AUTO` → routes CSC above threshold; currently falls back to LINKED_LIST otherwise.
4. Add the enum values to any other affected public-header places (check `include/sparse_ldlt.h` for existing backend-selection patterns).
5. Wire a skeleton test in `tests/test_ldlt.c` that constructs `sparse_ldlt_opts_t` with each backend value and confirms the opts struct accepts the new fields.
6. Run `make format && make lint && make test` — all clean.

### Deliverables
- `sparse_ldlt_opts_t` with `backend` selector + `used_csc_path` reporting
- `sparse_ldlt_factor_opts` dispatches on `backend` (CSC path still stubbed)
- Public-header documentation for every new field

### Completion Criteria
- Every existing test using `sparse_ldlt_factor_opts` still passes (backward compat)
- Explicit `SPARSE_LDLT_BACKEND_LINKED_LIST` routes to the existing path without behavior change
- `make format && make lint && make test` clean

---

## Day 5: Transparent LDL^T Dispatch — CSC Path & Writeback

**Theme:** Implement the CSC branch of the dispatch (sparse_analyze → `_with_analysis` → supernodal eliminate → writeback) and fold in the structural fallback

**Time estimate:** 10 hours

### Tasks
1. Implement the full CSC path in `sparse_ldlt_factor_opts`:
   - Run `sparse_analyze` with AMD reordering (or the opts-requested reorder) to populate `sparse_analysis_t`.
   - Build the CSC via `ldlt_csc_from_sparse_with_analysis`.
   - Factor via `ldlt_csc_eliminate_supernodal` (Day 3 enabled it for indefinite inputs).
   - Write the factored state back to the public `SparseMatrix *L` via the LDL^T equivalent of `chol_csc_writeback_to_sparse` — if no such function exists yet, add it to `src/sparse_ldlt_csc.c` mirroring the Cholesky helper's contract.
   - Set `opts->used_csc_path = 1` on success.
2. Implement the structural fallback: if `ldlt_csc_eliminate_supernodal` returns an error indicating the batched path cannot handle the input (e.g., all-scalar pivots, supernode detection produces no batched opportunities), fall back to `ldlt_csc_eliminate` on the same CSC, writeback, and set `used_csc_path = 1` still (CSC path was used — just the scalar variant).
3. Update `AUTO` branch: route CSC when `n >= SPARSE_CSC_THRESHOLD` and the input's structure isn't pathological (single tridiagonal column, etc. — match Cholesky's heuristic).
4. Cross-backend agreement test in `tests/test_ldlt.c`:
   - Same SPD matrix factored via `SPARSE_LDLT_BACKEND_LINKED_LIST` and `SPARSE_LDLT_BACKEND_CSC` — the resulting `L` matrices must produce solutions that agree to round-off (residuals ≤ 1e-10).
5. Run `make format && make lint && make test && make sanitize` — all clean.

### Deliverables
- Full CSC path in `sparse_ldlt_factor_opts` end-to-end (analyze → from_sparse_with_analysis → eliminate_supernodal → writeback)
- Structural fallback to `ldlt_csc_eliminate` on the CSC side when batching defeats
- Cross-backend agreement test

### Completion Criteria
- `SPARSE_LDLT_BACKEND_CSC` produces valid solves on 5+ existing LDL^T fixtures
- `used_csc_path` is set correctly on both batched and fallback CSC runs
- `make format && make lint && make test && make sanitize` clean

---

## Day 6: Transparent LDL^T Dispatch — Cross-Threshold Tests & Bench

**Theme:** Validate AUTO dispatch on realistic corpora, land a new integration test file, and capture bench numbers in `PERF_NOTES.md`

**Time estimate:** 8 hours

### Tasks
1. New integration tests in `tests/test_sprint20_integration.c` (create and register in `Makefile` + `CMakeLists.txt`):
   - Below-threshold dispatch: n < `SPARSE_CSC_THRESHOLD` with `SPARSE_LDLT_BACKEND_AUTO` — `used_csc_path == 0`.
   - Above-threshold dispatch: n ≥ `SPARSE_CSC_THRESHOLD` SPD — `used_csc_path == 1`, residual ≤ 1e-10.
   - Indefinite above-threshold: KKT-style at n = 150 — `used_csc_path == 1`, residual ≤ 1e-10 (this is the new capability).
   - Forced LINKED_LIST on a large matrix — always uses linked-list path regardless of n.
   - Forced CSC on a small matrix — uses CSC path below threshold.
2. Extend `bench_ldlt_csc` with a `--dispatch` mode that runs the same corpus under `SPARSE_LDLT_BACKEND_AUTO` and reports the chosen path + timing per matrix.  Adds a `backend` column to the CSV.
3. Run `./build/bench_ldlt_csc --dispatch` on the Sprint 19 corpus (bcsstk14, bcsstk04, s3rmt3m3, nos4) plus one KKT fixture; capture to `docs/planning/EPIC_2/SPRINT_20/bench_day6_dispatch.txt`.
4. Extend `docs/planning/EPIC_2/SPRINT_17/PERF_NOTES.md` with a new "LDL^T transparent dispatch" section showing: per-matrix chosen path, factor time via AUTO vs explicit LINKED_LIST, residuals.  Confirm AUTO's choice matches the measured crossover.
5. Run `make format && make lint && make test && make sanitize && make bench` — all clean.

### Deliverables
- `tests/test_sprint20_integration.c` with 5+ dispatch tests
- `bench_ldlt_csc --dispatch` mode + CSV capture
- `PERF_NOTES.md` extended with LDL^T dispatch section

### Completion Criteria
- Every dispatch test passes with `used_csc_path` reporting the expected path
- Indefinite fixture at n ≥ threshold produces a round-off residual via the AUTO path (Day 3 enablement validated end-to-end)
- `make format && make lint && make test && make sanitize && make bench` clean

---

## Day 7: Eigenvalue API Design

**Theme:** Design `sparse_eigs_t` / `sparse_eigs_sym` / the `sparse_eigs.h` public header so Days 8-12 have a stable target to implement against

**Time estimate:** 12 hours

### Tasks
1. Read the existing dense `sparse_tridiag_eigvals` in `src/sparse_dense.c` and the Sprint 16 iterative API in `include/sparse_iterative.h` / `src/sparse_iterative.c` as naming / style templates.  The public API should feel consistent with existing solver opts/result conventions.
2. Design `sparse_eigs_t`:
   - `idx_t n_requested` (what the caller asked for)
   - `idx_t n_converged` (how many Ritz pairs converged within tolerance)
   - `double *eigenvalues` (caller-owned buffer, length `n_requested`)
   - `double *eigenvectors` (optional — caller-owned buffer, `n × n_requested` column-major, NULL if not requested)
   - `idx_t iterations` (Lanczos iterations performed)
   - `double residual_norm` (max ‖A·v - λ·v‖ / ‖v‖ across returned pairs)
   - `sparse_err_t status` (SPARSE_OK / SPARSE_ERR_NOT_CONVERGED / ...)
3. Design `sparse_eigs_opts_t`:
   - `sparse_eigs_which_t which` — enum `LARGEST`, `SMALLEST`, `NEAREST_SIGMA`
   - `double sigma` — shift point (used with `NEAREST_SIGMA`)
   - `idx_t max_iterations`
   - `double tol` — convergence tolerance (relative)
   - `int reorthogonalize` — 0 / 1 flag for full reorthogonalization (default 1)
   - `int compute_vectors` — 0 / 1 flag (default 0 = eigenvalues only)
   - `sparse_eigs_backend_t backend` — enum `AUTO`, `LANCZOS`, future `LOBPCG`
4. Design `sparse_eigs_sym`:
   - Signature: `sparse_err_t sparse_eigs_sym(const SparseMatrix *A, idx_t k, const sparse_eigs_opts_t *opts, sparse_eigs_t *result);`.
   - Precondition documentation: A must be symmetric (checked at entry via `sparse_is_symmetric`); k must be between 1 and n.
5. Write `include/sparse_eigs.h` with the full public API + doxygen comments + preconditions + error codes.
6. Add header to `CMakeLists.txt`'s install set and to the Makefile's public-header list (if any).
7. Stub `src/sparse_eigs.c` with the function returning `SPARSE_ERR_NOT_IMPL` — compile-ready target for Day 8.
8. Update `docs/algorithm.md` API overview table to include the new `sparse_eigs_sym` entry (placeholder "Sprint 20" row).
9. Run `make format && make lint && make test` — all clean.

### Deliverables
- `include/sparse_eigs.h` with `sparse_eigs_t` + `sparse_eigs_opts_t` + `sparse_eigs_sym()` fully documented
- `src/sparse_eigs.c` stub compiling and registered in the build
- `docs/algorithm.md` API table placeholder entry

### Completion Criteria
- Every public API element has a doxygen comment
- `sparse_eigs_opts_t` and `sparse_eigs_t` designs reviewed against Sprint 16 iterative solver conventions for consistency
- Stub call from a smoke test returns `SPARSE_ERR_NOT_IMPL` cleanly
- `make format && make lint && make test` clean

---

## Day 8: Lanczos — Iteration Core (3-Term Recurrence)

**Theme:** Implement the basic Lanczos iteration that builds the tridiagonal T matrix and Lanczos vectors V for a symmetric A

**Time estimate:** 10 hours

### Tasks
1. Design block at the top of `src/sparse_eigs.c` covering: 3-term recurrence, the tridiagonal T matrix, Lanczos vector orthogonality loss in finite precision, and the roadmap (core today, reorthogonalization Day 9, thick-restart Day 10, Ritz extraction Day 11).
2. Implement `lanczos_iterate_basic`:
   - Input: matrix A, starting vector v₀ (random normalized), max iterations m.
   - Output: V = [v₀, v₁, ..., vₘ₋₁] (n × m), α (diagonal of T, length m), β (sub/superdiagonal of T, length m-1).
   - 3-term recurrence: w = A·vₖ - βₖ₋₁·vₖ₋₁; αₖ = <w, vₖ>; w = w - αₖ·vₖ; βₖ = ‖w‖; vₖ₊₁ = w / βₖ.
   - Early exit on βₖ == 0 (invariant subspace found — can't continue).
3. Unit tests on deterministic inputs:
   - Diagonal matrix A = diag(1, 2, 3, ..., n): Lanczos T matrix has the same spectrum as A after n iterations.
   - Tridiagonal A already in Lanczos form: T should equal A bit-for-bit after n iterations.
4. Implement a small helper `tridiag_eigen_symmetric` (or reuse the existing `sparse_tridiag_eigvals`) to extract eigenvalues from T — this is the Day 11 Ritz-extraction prerequisite but we need the signature now.
5. Expose `lanczos_iterate_basic` as a static internal helper in `src/sparse_eigs.c`; no public header changes.
6. Run `make format && make lint && make test && make sanitize` — all clean.

### Deliverables
- `lanczos_iterate_basic` producing V and T from A and a starting vector
- 2+ unit tests validating the recurrence on deterministic inputs
- Integration with the existing tridiagonal eigensolver for Ritz extraction

### Completion Criteria
- Diagonal and tridiagonal unit tests pass with T spectrum matching A's to 1e-10
- No memory leaks under `make sanitize`
- `make format && make lint && make test && make sanitize` clean

---

## Day 9: Lanczos — Full Reorthogonalization

**Theme:** Add full reorthogonalization against all prior Lanczos vectors so the iteration maintains V^T·V ≈ I under finite precision

**Time estimate:** 10 hours

### Tasks
1. Extend `lanczos_iterate_basic` → `lanczos_iterate` with an `opts->reorthogonalize` gate.  When enabled, after computing the tentative `w = A·vₖ - βₖ₋₁·vₖ₋₁ - αₖ·vₖ`, subtract the projection of w onto every stored vⱼ (j < k):
   - `for j in [0, k): w -= <w, vⱼ>·vⱼ`.
   - Renormalize: βₖ = ‖w‖; vₖ₊₁ = w / βₖ.
   - Document the classical Gram-Schmidt (CGS) vs modified Gram-Schmidt (MGS) tradeoff; use MGS for numerical stability on small-to-moderate k, optionally add a twice-MGS refinement pass if stability testing shows loss of orthogonality.
2. Write orthogonality-loss tests:
   - Dense SPD of n = 100 with widely-spaced eigenvalues (exponentially scaled, e.g. 1, 10, 100, ..., 10⁹⁹): Lanczos without reorth loses V^T·V ≈ I after ~30 iterations; Lanczos with full reorth maintains ‖V^T·V - I‖_max ≤ 1e-10.
   - Same test with `reorthogonalize = 0` — ghost eigenvalues should appear (the reference is the known Paige/Parlett behavior, not a regression); log but do not fail.
3. Benchmark reorthogonalization cost:
   - Factor a 500 × 500 SPD at m = 100 iterations with and without reorth; report the timing ratio.  Reorth should cost O(m²·n) per iteration vs O(m·n) basic.  Confirm the overhead is ≤ 3× on small m.
4. Capture numbers in `docs/planning/EPIC_2/SPRINT_20/bench_day9_reorth.txt`.
5. Run `make format && make lint && make test && make sanitize` — all clean.

### Deliverables
- `lanczos_iterate` with `opts->reorthogonalize` flag honored
- Orthogonality-loss tests confirming reorth works (and documenting the behavior when disabled)
- Benchmark capture of reorth overhead

### Completion Criteria
- Wide-spectrum SPD test shows ‖V^T·V - I‖_max ≤ 1e-10 with reorth enabled
- Both reorth-on and reorth-off paths exercised under sanitizers
- `make format && make lint && make test && make sanitize` clean

---

## Day 10: Lanczos — Thick-Restart Mechanism

**Theme:** Implement thick-restart Lanczos so the iteration can run indefinitely without the Krylov subspace growing unbounded, while preserving the converged Ritz pairs across restarts

**Time estimate:** 10 hours

### Tasks
1. Design block for thick-restart (Wu/Simon, Stathopoulos/Saad 2007): after m iterations, compute the Ritz pairs (θⱼ, yⱼ) from T = Y·Θ·Y^T; select the top k converged (or most-wanted) Ritz vectors; replace V with V·[y₁, ..., yₖ]; T becomes Θ (diagonal with θⱼ); β vector gets the `β_{m-1}·y_{m-1,j}` trailing terms that couple the restart to the next basis vector.
2. Implement `lanczos_restart`:
   - Input: current V (n × m), T (tridiag, m × m), β_last (scalar), target restart size k.
   - Compute Ritz pairs via `tridiag_eigen_symmetric` on T.
   - Select the k Ritz vectors matching `opts->which` (LARGEST / SMALLEST / NEAREST_SIGMA).
   - Form V_new = V · Y_k (n × k) via BLAS-like gemm (implement inline — no external BLAS dependency in this library).
   - Form T_new = diag(θ_k) with trailing-β coupling column for the restart.
   - Reset the iteration state so the next call to `lanczos_iterate` continues from v_{k+1} = (A·V_new's residual) normalized.
3. Wrap core iteration + restart in an outer loop in `sparse_eigs_sym`:
   - Run `lanczos_iterate` for m iterations (e.g. m = 2k + 10).
   - Check convergence: ‖A·vⱼ - θⱼ·vⱼ‖ ≤ tol·|θⱼ| for each Ritz pair.
   - If k converged or max_iterations reached → return result.
   - Otherwise → `lanczos_restart` and continue.
4. Unit test on a moderate-size SPD (n = 200) asking for k = 5 largest eigenvalues; assert the outer loop converges in < 100 iterations and the returned eigenvalues match the dense reference to 1e-10.
5. Run `make format && make lint && make test && make sanitize` — all clean.

### Deliverables
- `lanczos_restart` helper implementing Wu/Simon thick-restart
- Outer loop in `sparse_eigs_sym` composing iterate + restart
- Convergence test on n = 200 SPD confirming the mechanism

### Completion Criteria
- Restart preserves the converged Ritz pairs (their residual does not regress across a restart boundary)
- n = 200 SPD test converges to 5 eigenvalues in < 100 iterations with residual ≤ 1e-10
- `make format && make lint && make test && make sanitize` clean

---

## Day 11: Lanczos — Ritz Extraction, Convergence & Public Entry

**Theme:** Complete `sparse_eigs_sym` end-to-end: Ritz value/vector extraction, convergence bookkeeping, and the caller-visible result struct

**Time estimate:** 10 hours

### Tasks
1. Implement full Ritz extraction at the end of the outer loop:
   - Eigenvalues: `result->eigenvalues[j] = θⱼ` for the converged Ritz values ordered by `opts->which`.
   - Eigenvectors (if `opts->compute_vectors`): `result->eigenvectors[:, j] = V · yⱼ` — n-length vector in the original space.
2. Convergence bookkeeping:
   - Populate `result->n_converged`, `result->iterations`, `result->residual_norm`, `result->status`.
   - `SPARSE_OK` when n_converged ≥ n_requested; `SPARSE_ERR_NOT_CONVERGED` otherwise (still populate partial results).
3. Defensive input checks:
   - Symmetry check at entry (`sparse_is_symmetric(A)` → `SPARSE_ERR_NOT_SYMMETRIC` if fails).
   - k bounds check (1 ≤ k ≤ n → `SPARSE_ERR_BADARG`).
   - NULL result/opts checks → `SPARSE_ERR_NULL`.
4. `sparse_eigs_free(sparse_eigs_t *result)` helper that frees caller-owned eigenvalue / eigenvector buffers allocated inside `sparse_eigs_sym` when the caller passed NULLs.
5. Smoke tests in `tests/test_eigs.c` (new file, register in Makefile + CMake):
   - Diagonal A = diag(1, ..., 10), k = 3 largest → expect 10, 9, 8 within 1e-10.
   - Diagonal A = diag(1, ..., 10), k = 3 smallest → expect 1, 2, 3 within 1e-10.
   - Eigenvector correctness on the diagonal test: ‖A·vⱼ - λⱼ·vⱼ‖ / ‖vⱼ‖ ≤ 1e-10 for each returned pair.
   - Symmetry-rejection test: non-symmetric input → `SPARSE_ERR_NOT_SYMMETRIC`.
6. Run `make format && make lint && make test && make sanitize` — all clean.

### Deliverables
- `sparse_eigs_sym` end-to-end with Ritz extraction + convergence reporting
- `sparse_eigs_free` helper
- `tests/test_eigs.c` with 4+ smoke tests covering largest / smallest / eigenvector / precondition rejection

### Completion Criteria
- All 4 smoke tests pass
- Returned eigenvectors satisfy the eigen-equation within the requested tolerance
- `make format && make lint && make test && make sanitize` clean

---

## Day 12: Shift-Invert Lanczos

**Theme:** Add shift-invert mode so interior eigenvalues can be found via `(A - σ·I)^{-1}`, using the new transparent LDL^T dispatch for the symmetric case

**Time estimate:** 10 hours

### Tasks
1. Design block covering: why shift-invert converges to interior eigenvalues (eigenvalues of `(A - σ·I)^{-1}` are `1/(λ - σ)`, so eigenvalues near σ become largest in magnitude — ideal for Lanczos which converges to extreme eigenvalues fastest), what the factorization choice is (LDL^T for symmetric, LU fallback for general-on-symmetric-input), and the cost tradeoff (one factor + repeated solves vs iterating on A directly).
2. Extend the Lanczos iteration with a `matvec` callback parameter:
   - Default (non-shift-invert): `matvec(v) = A·v` via `sparse_matvec`.
   - Shift-invert: `matvec(v) = (A - σ·I)^{-1}·v` — one solve per Lanczos iteration against the pre-factored `A - σ·I`.
3. In `sparse_eigs_sym`, when `opts->which == NEAREST_SIGMA`:
   - Build `A - σ·I` as a copy of A with σ subtracted from the diagonal.
   - Factor via `sparse_ldlt_factor_opts` (the Day 4-6 dispatch; benefits from `SPARSE_LDLT_BACKEND_AUTO`).
   - Drive Lanczos with the shift-invert matvec.
   - Post-process the returned Ritz values: `λⱼ = σ + 1/θⱼ` (Ritz values of the shift-inverted operator are `1/(λ - σ)`).
4. Factorization failure handling: if LDL^T returns `SPARSE_ERR_SINGULAR` on `A - σ·I` (σ is exactly an eigenvalue), report `SPARSE_ERR_SHIFT_SINGULAR` and suggest the caller perturb σ slightly.
5. Shift-invert tests:
   - Diagonal A = diag(1, ..., 20), σ = 10.5, k = 3 → expect eigenvalues 10, 11, 9 (nearest to σ) within 1e-10.
   - Indefinite 2×2 saddle-point matrix with eigenvalues {-3, -1, +2, +5}, σ = 0, k = 2 → expect -1, +2 (nearest to 0).  Exercises the new LDL^T `_with_analysis` path indirectly.
   - Convergence comparison: finding the 3 middle eigenvalues of a wide-spectrum n = 200 SPD via direct Lanczos vs shift-invert — shift-invert should converge in ≤ half the iterations.
6. Run `make format && make lint && make test && make sanitize` — all clean.

### Deliverables
- Shift-invert mode in `sparse_eigs_sym` using the new LDL^T dispatch
- 3+ shift-invert tests covering diagonal, indefinite, convergence-rate comparison
- Failure-mode test for σ exactly equal to an eigenvalue

### Completion Criteria
- Shift-invert on the indefinite fixture exercises `sparse_ldlt_factor_opts` with `SPARSE_LDLT_BACKEND_AUTO` and succeeds
- Convergence-rate test shows shift-invert wins on interior eigenvalues
- `make format && make lint && make test && make sanitize` clean

---

## Day 13: Lanczos Tests — SuiteSparse, SVD Cross-Check & Indefinite Coverage

**Theme:** Broaden Lanczos validation across SuiteSparse matrices, cross-check against existing SVD, and exercise the indefinite LDL^T path through shift-invert

**Time estimate:** 12 hours

### Tasks
1. SuiteSparse SPD coverage in `tests/test_eigs.c`:
   - nos4 (n = 100): k = 5 largest, k = 5 smallest; compare each to the dense reference computed via the existing dense eigensolver (or LAPACK if linked); residuals ≤ 1e-8.
   - bcsstk04 (n = 132): k = 3 largest, k = 3 smallest; same validation.
   - bcsstk14 (n = 1806): k = 5 largest; skip smallest (likely many small eigenvalues, slow to converge — smoke test only).
2. SVD cross-check:
   - For a dense-ish rectangular A (e.g. 40 × 30), compute the eigenvalues of A^T·A via `sparse_eigs_sym` and compare to the squared singular values from `sparse_svd_economy`.  Must agree to 1e-10 for the largest few.
   - Dual: eigenvalues of A·A^T equal the same squared singular values — confirms the API handles both shapes.
3. Indefinite coverage (exercises Day 3 batched-supernodal LDL^T + Day 12 shift-invert end-to-end):
   - Sprint 18 Day 13 KKT fixture (or a Sprint 20 equivalent of dim ≥ 150 above threshold): find 3 eigenvalues nearest 0 via shift-invert; the LDL^T factor inside must route through the AUTO CSC path on this n ≥ threshold input.
   - Assert `opts.used_csc_path == 1` was observed on the inner LDL^T factor (expose via a test-only diagnostic or assert post-hoc).
4. Stability / regression tests:
   - Near-singular A (condition number 10⁸): `sparse_eigs_sym` should still return reasonable eigenvalues or cleanly report `SPARSE_ERR_NOT_CONVERGED` without crashing.
   - Zero matrix (all eigenvalues = 0): should return k zeros with warning-level convergence status.
5. Run `make format && make lint && make test && make sanitize` — all clean.
6. Capture Lanczos bench numbers (time vs n, time vs k, time with/without reorth, shift-invert vs direct) to `docs/planning/EPIC_2/SPRINT_20/bench_day13_lanczos.txt`.

### Deliverables
- 5+ SuiteSparse-based tests (nos4, bcsstk04, bcsstk14) for both SPD eigenvalue ends
- SVD consistency cross-check (eigenvalues of A^T·A = σⱼ²)
- 2+ indefinite shift-invert tests that exercise the AUTO LDL^T dispatch
- Lanczos bench capture

### Completion Criteria
- All SuiteSparse tests pass with residuals within tolerance
- SVD cross-check confirms the eigenvalue/singular-value relationship to 1e-10
- Indefinite shift-invert test confirms `used_csc_path == 1` on the inner LDL^T factor
- `make format && make lint && make test && make sanitize` clean

---

## Day 14: Documentation, Example Program & Sprint 20 Retrospective

**Theme:** Write the Lanczos documentation, add the example program, update project docs, and close out Sprint 20 with a retrospective

**Time estimate:** 10 hours

### Tasks
1. `README.md` updates (2 hrs):
   - Add a "Sparse symmetric eigensolver (Sprint 20)" subsection describing `sparse_eigs_sym`, the three `which` modes, shift-invert, and the LDL^T dispatch it composes with.
   - Extend the API overview table with `sparse_eigs_sym` / `sparse_eigs_t` / `sparse_eigs_opts_t`.
   - Short performance blurb pointing at `bench_day13_lanczos.txt`.
2. Example program (2 hrs):
   - New `examples/example_eigs.c` demonstrating: (a) 5 largest eigenvalues of a small SuiteSparse SPD, (b) 3 eigenvalues nearest σ via shift-invert on a KKT fixture, (c) eigenvectors with residual check.
   - Register in `examples/README.md`, `Makefile`, and `CMakeLists.txt`.
   - Ship with inline explanatory comments — this is the customer-facing onboarding doc, not an internal test.
3. `include/sparse_eigs.h` doxygen refresh (1 hr): tighten comments based on Days 8-13 learnings; add cross-references to `sparse_ldlt.h` (for shift-invert) and `sparse_svd.h` (for the cross-check).
4. `docs/algorithm.md` Lanczos section (2 hrs): sketch the 3-term recurrence, full reorthogonalization rationale, thick-restart mechanism, shift-invert semantics, and convergence heuristics — same pedagogical depth as the existing "Supernodal Detection" section from Sprint 17/18.  Point at `bench_day13_lanczos.txt` for measured performance.
5. `docs/planning/EPIC_2/PROJECT_PLAN.md` update (0.5 hr): mark Sprint 20 items 1-7 as complete, update the Summary table (Sprint 20 → **Complete**, actual hours).
6. Retrospective (2 hrs): `docs/planning/EPIC_2/SPRINT_20/RETROSPECTIVE.md` mirroring the Sprint 19 template — DoD checklist against the 7 PROJECT_PLAN.md items, final metrics (assertion count, test count delta, Lanczos vs shift-invert perf numbers), what went well / didn't, items deferred with rationale, lessons (especially any surprises in the Lanczos reorth or the LDL^T `_with_analysis` integration).
7. Final regression (0.5 hr):
   - `make clean && make format && make lint && make test && make sanitize && make bench && make examples` — all clean.
   - Verify total test count grew by at least 25 from the Sprint 19 baseline.

### Deliverables
- `README.md` + `docs/algorithm.md` refreshed with Lanczos content
- `examples/example_eigs.c` (customer-facing)
- `include/sparse_eigs.h` doxygen finalized
- `docs/planning/EPIC_2/PROJECT_PLAN.md` Sprint 20 marked **Complete**
- `docs/planning/EPIC_2/SPRINT_20/RETROSPECTIVE.md` with full DoD + metrics + deferrals

### Completion Criteria
- Every item (1-7) in the Sprint 20 PROJECT_PLAN.md table is marked ✅ or ⚠️ with explicit deferral rationale
- `make clean && make format && make lint && make test && make sanitize && make bench && make examples` all clean
- `examples/example_eigs.c` runs end-to-end and prints a sensible set of eigenvalues + residuals
- Retrospective honestly assesses both wins and misses, including any architectural surprises hit in Days 1-6 (LDL^T) or Days 8-12 (Lanczos)
