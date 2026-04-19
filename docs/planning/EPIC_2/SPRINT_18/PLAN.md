# Sprint 18 Plan: CSC Kernel Performance Follow-Ups

**Sprint Duration:** 14 days
**Goal:** Complete the Sprint 17 CSC numeric-backend work by replacing the linked-list delegations with native CSC kernels, exposing a transparent size-based dispatch through the public API, and validating scaling behavior on larger SuiteSparse problems.

**Starting Point:** Sprint 17 delivered the CSC working-format backends in `src/sparse_chol_csc.c` and `src/sparse_ldlt_csc.c` with the following caveats that this sprint closes out:

- `ldlt_csc_eliminate` is a **wrapper**: it expands the lower-triangular CSC into a full symmetric `SparseMatrix`, calls the linked-list `sparse_ldlt_factor`, and unpacks the result back into CSC. Factor body still runs through the linked-list kernel. Solve (`ldlt_csc_solve`) is already fully native.
- `chol_csc_eliminate_supernodal` **detects** supernodes via `chol_csc_detect_supernodes` and then **delegates** to the scalar kernel. The dense primitives `chol_dense_factor` / `chol_dense_solve_lower` ship as tested helpers but are not yet called per supernode.
- `sparse_cholesky_factor_opts` still dispatches exclusively to the linked-list kernel. `SPARSE_CSC_THRESHOLD` is defined (default 100 in `include/sparse_matrix.h`) but only used as documentation; callers have to reach into the internal `chol_csc_*` / `ldlt_csc_*` APIs to exercise the CSC path.
- The default benchmark fixtures cap at n = 132 (bcsstk04). The scalar CSC kernel's pointer-chasing win and the supernodal batched path's scaling both grow with n, so larger matrices are needed to demonstrate the effect.

**End State:** A native CSC Bunch-Kaufman LDL^T kernel replaces the Day 8 wrapper and produces bit-identical output on the existing `test_ldlt_csc` matrices; `chol_csc_eliminate_supernodal` calls `chol_dense_factor` + `chol_dense_solve_lower` per detected supernode and matches the scalar kernel to round-off; `sparse_cholesky_factor_opts` transparently dispatches matrices with `n >= SPARSE_CSC_THRESHOLD` to the CSC backend with a lossless CSC → linked-list writeback (preserving `factor_norm`, `reorder_perm`, the row/col permutation arrays, and the `factored` flag); and the default benchmark corpus expands with SuiteSparse SPD / symmetric indefinite matrices at `n >= 1000` so scaling is visible in `PERF_NOTES.md`.

---

## Day 1: Native CSC LDL^T Bunch-Kaufman — Design & Scaffolding

**Theme:** Replace the wrapper with a native kernel: design the on-CSC Bunch-Kaufman algorithm and stand up scaffolding

**Time estimate:** 8 hours

### Tasks
1. Design document in a new `/* native-kernel design */` block at the top of `src/sparse_ldlt_csc.c`:
   - What the wrapper does today (expand → `sparse_ldlt_factor` → unpack), why it is correct, why it is slow (double allocation for the full symmetric matrix; linked-list pointer chasing inside the factor).
   - What the native kernel must do: column-by-column Bunch-Kaufman directly on the packed CSC arrays, four-criteria pivot selection, in-place symmetric row/column swaps, element-growth tracking, and the same 1×1 / 2×2 pivot conventions (`pivot_size`, `D`, `D_offdiag`) that `ldlt_csc_solve` already consumes.
   - Cite `src/sparse_ldlt.c` as the behavioural reference — the target is bit-identical output on every test currently exercising the wrapper.
2. Add a feature-flag path so the two kernels can coexist during development:
   - Introduce `ldlt_csc_eliminate_native(LdltCsc *F)` alongside the existing `ldlt_csc_eliminate`.
   - Add an internal `LDLT_CSC_USE_NATIVE` compile-time switch (default off) so `ldlt_csc_eliminate` selects between the wrapper and the native path; tests can force either side via a hidden setter.
3. Workspace scaffolding mirroring the Sprint 17 `CholCscWorkspace` pattern:
   - Dense column buffer (length n) for the currently-active column and its pivot candidate.
   - Dense pattern / marker arrays for scatter-gather during cmod (same idiom as `chol_csc_eliminate`).
   - Per-step scratch for a 2×2 BK pivot candidate row (length n).
   - Allocation / free helpers (`ldlt_csc_workspace_alloc` / `_free`), guarded by the same `size_t` overflow checks already used in `chol_csc_workspace_alloc`.
4. Skeleton `ldlt_csc_eliminate_native` that currently returns `SPARSE_ERR_NOT_IMPLEMENTED` but sets up the column loop, the workspace, and the permutation bookkeeping (composes BK swaps into `F->perm` alongside any incoming fill-reducing perm, matching the wrapper's behavior today).
5. Port the Bunch-Kaufman pivot-selection helper from `src/sparse_ldlt.c` into a static function in `src/sparse_ldlt_csc.c` (four criteria with α = (1 + √17) / 8 ≈ 0.6404). Keep it numerically identical to the reference — no semantic changes — so the only variable later in testing is CSC vs linked-list storage, not the pivot policy.
6. Run `make format && make lint && make test` — all clean.

### Deliverables
- Design block in `src/sparse_ldlt_csc.c` explaining the wrapper → native migration
- `ldlt_csc_eliminate_native` skeleton plus workspace lifecycle
- Feature flag `LDLT_CSC_USE_NATIVE` with wrapper / native selector
- Ported Bunch-Kaufman pivot-selection helper (static, CSC-callable)

### Completion Criteria
- Library still builds with the wrapper path selected (default); existing tests pass unchanged
- Native path builds cleanly and is exercised by a single smoke test returning `SPARSE_ERR_NOT_IMPLEMENTED`
- `make format && make lint && make test` clean

---

## Day 2: Native CSC LDL^T Bunch-Kaufman — In-Place Symmetric Swaps

**Theme:** The hardest primitive: swap two rows **and** two columns of a packed CSC matrix in place

**Time estimate:** 8 hours

### Tasks
1. Understand why symmetric swap is hard in CSC: rows are not first-class citizens, so swapping row `i` with row `j` means touching every column's `row_idx[]` entry. Symmetric swap additionally reorders the column-ptr slots for columns `i` and `j`. Implementing this naively is O(nnz) per swap, which would wipe out the kernel's speedup on anything but the smallest matrices.
2. Implement `ldlt_csc_symmetric_swap(LdltCsc *F, idx_t i, idx_t j)`:
   - **Column swap**: exchange the contiguous slices `L->values[col_ptr[i]..col_ptr[i+1])` and `L->values[col_ptr[j]..col_ptr[j+1])` (and the matching `row_idx[]` slices). Update `col_ptr` so both columns end up in the right place; shift intermediate columns as needed.
   - **Row swap**: walk every column's `row_idx[]` in the range `[min(i,j), max(i,j)]` and rewrite any entry equal to `i` as `j` and vice versa; re-sort the affected column slices so the diagonal-first-in-column CSC invariant holds.
   - Maintain `F->perm` and `F->pivot_size` consistent with the swap.
3. Optimisation: bound the work by the active sub-matrix. Symmetric swaps during BK only happen between column `j` (the current step) and some `k > j`, so rows `< j` have already been finalised and do not need rewriting. Skip them in the row-walk.
4. Unit tests for the swap primitive on synthetic `LdltCsc` states (constructed directly, not via elimination):
   - Swap adjacent columns on a dense 5×5.
   - Swap non-adjacent columns on a tridiagonal pattern.
   - Symmetric swap that moves a zero diagonal onto the pivot seat (the target use-case).
   - No-op swap (i == j) — must leave storage untouched.
   - Validate with `ldlt_csc_validate` after every swap.
5. Stress test: random sequence of 50 symmetric swaps on a random 20×20 lower-triangular CSC; after each swap, convert to dense, swap symmetrically in dense, and check the two representations agree.
6. Run `make format && make lint && make test` — all clean.

### Deliverables
- `ldlt_csc_symmetric_swap` primitive, bounded to the active sub-matrix
- Round-trip validation between CSC swap and dense swap
- 5+ unit tests + 1 stress test

### Completion Criteria
- CSC swap matches dense swap exactly on the stress test (bit-identical `row_idx` after re-sort, `values` after permutation)
- `ldlt_csc_validate` passes after every swap
- `make format && make lint && make test` clean

---

## Day 3: Native CSC LDL^T Bunch-Kaufman — Pivot Selection & 1×1 Elimination

**Theme:** Wire the pivot selector into the native column loop and land the 1×1 pivot elimination step

**Time estimate:** 8 hours

### Tasks
1. Complete the main column loop inside `ldlt_csc_eliminate_native`:
   - For `j = 0..n-1`, scatter column `j` into the dense workspace (reuse the Sprint 17 `chol_csc_eliminate` scatter pattern).
   - Apply cmod updates from all prior columns `k < j` with `L[j,k] != 0` — but with the LDL^T form: `L[i,j] -= L[i,k] * D[k] * L[j,k]` for 1×1 pivots, and the 2×2 block form for 2×2 pivots at step k (the 2×2 step lands on Day 4; leave a guard here that falls back to `SPARSE_ERR_NOT_IMPLEMENTED` when `F->pivot_size[k] == 2` so tests exercising pure-1×1 cases can already run).
2. Invoke the ported Bunch-Kaufman pivot selector at column j:
   - Input: the dense workspace column and the largest off-diagonal magnitude.
   - Output: decision (1×1 at j, 1×1 with row swap, 2×2 at j / j+1 with symmetric swap).
3. Handle the 1×1 pivot branches fully:
   - **Straight 1×1**: write `D[j] = L[j,j]`, divide the below-diagonal entries by `D[j]`, set the stored diagonal slot to 1.0 (unit-L invariant), gather back into CSC, update `pivot_size[j] = 1`.
   - **1×1 with symmetric swap**: call `ldlt_csc_symmetric_swap(F, j, r)` for the chosen row r before the divide, update `F->perm[j]` with the swap, then fall through to the straight-1×1 path.
4. Element-growth tracking:
   - Maintain `F->factor_norm` and the running max-entry magnitude across completed columns (as `sparse_ldlt_factor` does).
   - Abort with `SPARSE_ERR_NOT_SPD` if the growth exceeds the configured threshold.
5. Tests targeting pure-1×1 matrices:
   - Diagonal indefinite matrix (D values matching the diagonal, no swaps required).
   - Tridiagonal indefinite (well-conditioned → 1×1 with no swap, but exercises the cmod loop).
   - Indefinite matrix where the first column forces a 1×1 swap (diagonal magnitude < α · max_off_diag).
   - For each, force the native path via the feature flag and compare `L`, `D`, `perm`, and `pivot_size` element-for-element with the wrapper's output (which currently runs through the linked-list reference).

### Deliverables
- Native column loop with cmod + 1×1 pivot elimination + element-growth tracking
- Pivot selector wired to the BK-chosen action
- 4+ tests on pure-1×1 matrices matching the wrapper's output bit-for-bit

### Completion Criteria
- Pure-1×1 tests pass under both wrapper and native paths with identical output
- Element-growth abort triggers on an adversarial growth test matrix
- `make format && make lint && make test` clean

---

## Day 4: Native CSC LDL^T Bunch-Kaufman — 2×2 Block Pivots

**Theme:** Land the 2×2 branch of Bunch-Kaufman on CSC storage

**Time estimate:** 8 hours

### Tasks
1. 2×2 pivot elimination at columns j / j+1:
   - After the selector picks a 2×2 pair, call `ldlt_csc_symmetric_swap` as needed so the chosen partner column sits in slot j+1.
   - Factor the 2×2 block: set `D[j]`, `D[j+1]`, `D_offdiag[j]` to the block diagonal entries (matching the `sparse_ldlt_t` convention already consumed by `ldlt_csc_solve`).
   - Compute `det = D[j] * D[j+1] - D_offdiag[j]^2` and use the inverse 2×2 to scale the below-diagonal rows of both columns j and j+1.
   - Write unit diagonals into the stored L slots, update `pivot_size[j] = pivot_size[j+1] = 2`.
2. cmod for subsequent columns contributed by a 2×2 pivot step:
   - A column `m > j+1` with nonzeros in both rows j and j+1 receives a *block* contribution: `L[i, m] -= ( L[i,j] L[i,j+1] ) · D_block · ( L[m,j] L[m,j+1] )^T`.
   - Handle the "only one of j / j+1 contributes" sparsity case (the other entry is structurally zero in column m) by zeroing the missing row in a scratch 2-vector before the multiply.
3. Update the guard in the Day 3 cmod loop so `pivot_size[k] == 2` branches actually execute the block-cmod instead of erroring.
4. Symmetric-permutation bookkeeping: after a 2×2 swap, compose the swap into `F->perm[j+1]` alongside the earlier `F->perm[j]` update. This matches the convention `sparse_ldlt_factor` uses today and keeps the solve path working unchanged.
5. Tests targeting 2×2 cases:
   - `[[0.1, 1], [1, 0.3]]` — the forced-2×2 used in Sprint 17 day-8 tests (both diagonals small → BK Criterion 4).
   - Larger symmetric indefinite matrices that force a mix of 1×1 and 2×2 pivots (port from existing `test_ldlt.c` fixtures).
   - SuiteSparse indefinite fixture already used by `test_ldlt_csc` (native vs wrapper must match D, D_offdiag, perm, and pivot_size bit-for-bit).
6. Run `make format && make lint && make test` — all clean.

### Deliverables
- 2×2 pivot elimination branch (swap + block factor + block scaling)
- Block cmod contributing to subsequent columns
- 4+ tests exercising 2×2 pivots, each cross-checked against the wrapper

### Completion Criteria
- Every `test_ldlt_csc` matrix factors identically under wrapper and native paths (L, D, D_offdiag, perm, pivot_size all bit-identical)
- Inertia (counts of positive / negative / zero entries in `D`) matches on every test matrix
- `make format && make lint && make test` clean

---

## Day 5: Native CSC LDL^T Bunch-Kaufman — Cross-Checks & Switchover

**Theme:** Enable the native path by default, retire the wrapper, update the benchmark baseline

**Time estimate:** 8 hours

### Tasks
1. Flip `LDLT_CSC_USE_NATIVE` default on so `ldlt_csc_eliminate` calls the native kernel. The wrapper path stays compiled for one sprint as a debugging fallback but is not reachable from production call sites.
2. Delete the now-unused `csc_to_full_symmetric_matrix` helper and the wrapper's `sparse_ldlt_factor` call site once the test suite is green; keep the file header comment updated to reflect reality ("native kernel, linked-list reference used only in dev-mode cross-check").
3. Expanded cross-check test `test_eliminate_matches_linked_list_indefinite`: iterate a list of 20+ random symmetric indefinite matrices (seeded RNG for determinism) and assert bit-identical L / D / perm / pivot_size between native CSC and `sparse_ldlt_factor` on the linked-list side.
4. Update `benchmarks/bench_ldlt_csc.c`:
   - Remove the "factor is delegated" asterisk from the header comment.
   - Add a column labelled `factor_csc_native_ms` alongside the existing `factor_csc_ms` (which, before today, was measuring the wrapper), and verify the new numbers are lower on every matrix.
   - Re-run the benchmark and capture numbers for the PERF_NOTES update later in the sprint.
5. ASan run on the new code paths: `make sanitize` (ASan + UBSan) across `test_ldlt_csc` and `bench_ldlt_csc` to catch any swap / scatter / gather overruns.
6. Run `make format && make lint && make test && make sanitize` — all clean.

### Deliverables
- Native kernel enabled by default; wrapper compiled but gated behind a dev-mode flag
- 20+ random cross-check test passing bit-identically
- Updated `bench_ldlt_csc.c` reporting native-kernel factor times
- ASan/UBSan clean on all new code paths

### Completion Criteria
- Every existing `test_ldlt_csc` test passes under the native path
- Random cross-check finds zero divergences
- `make format && make lint && make test && make sanitize` clean

---

## Day 6: Batched Supernodal Cholesky — Extract / Writeback Helpers

**Theme:** Build the plumbing that moves a supernode's diagonal block and panel between CSC storage and a dense column-major buffer

**Time estimate:** 8 hours

### Tasks
1. Review `chol_csc_detect_supernodes` output (from Sprint 17 Day 10) so the batched path consumes the existing `super_starts` / `super_count` arrays unchanged. No changes to detection this sprint — only to what happens after detection.
2. Design the extract/writeback buffer contract:
   - For a supernode starting at column `s_start` with `s_size = super_starts[k+1] - super_starts[k]` columns, allocate a dense column-major buffer holding `s_size` columns × `panel_height` rows, where `panel_height = s_size + |rows below the supernode that appear in the first column of the supernode|`.
   - The top `s_size × s_size` slab is the **diagonal block** (fed to `chol_dense_factor`); the remaining `(panel_height - s_size) × s_size` slab is the **panel** (fed to `chol_dense_solve_lower`).
3. Implement `chol_csc_supernode_extract(const CholCsc *csc, idx_t s_start, idx_t s_size, double *dense, idx_t lda, idx_t *row_map, idx_t *panel_height_out)`:
   - Walk the first column of the supernode to collect the row indices ≥ `s_start`; those are the rows of the panel in order. `row_map[i]` = CSC row index for panel row `i`; `panel_height_out` is set.
   - For each of the `s_size` columns, scatter the column's values into the dense buffer using `row_map` to place each value. Assumes the supernodal sparsity invariant (all columns in the supernode share the same below-diagonal pattern) — verify with an assert on debug builds.
4. Implement `chol_csc_supernode_writeback(CholCsc *csc, idx_t s_start, idx_t s_size, const double *dense, idx_t lda, const idx_t *row_map, idx_t panel_height)`:
   - Gather the dense buffer back into CSC using the same `row_map`. Apply the scalar-kernel drop tolerance (consistent with `chol_csc_eliminate`).
   - Assert the writeback stays within the pre-allocated column capacity (the scalar kernel already guarantees this via Sprint 14 symbolic analysis; the supernodal path must match).
5. Tests (no dense factoring yet, just round-trip):
   - Dense 6×6 SPD matrix with a single supernode covering all columns: extract → memcpy → writeback → `chol_csc_validate` passes and `chol_csc_to_sparse` reproduces the input.
   - Block-diagonal SPD (two 3×3 blocks): extract each supernode, round-trip, and verify independence (first supernode's writeback doesn't disturb the second's storage).
   - Column-major lda > panel_height (exercise padding).
6. Run `make format && make lint && make test` — all clean.

### Deliverables
- `chol_csc_supernode_extract` / `_writeback` helpers
- 3+ round-trip tests on canonical supernodal structures

### Completion Criteria
- Extract → writeback is exactly the identity on every supernode of the test matrices
- `chol_csc_validate` passes after every writeback
- `make format && make lint && make test` clean

---

## Day 7: Batched Supernodal Cholesky — Diagonal Block Factor Integration

**Theme:** Replace the per-column scalar `cdiv` sequence on the diagonal block with a single `chol_dense_factor` call

**Time estimate:** 8 hours

### Tasks
1. Introduce `chol_csc_supernode_eliminate_diag(CholCsc *csc, idx_t s_start, idx_t s_size, CholCscWorkspace *ws)`:
   - Extract the diagonal block using Day 6's helper.
   - First, apply scalar cmod from all prior (non-supernodal or previously-factored-supernodal) columns into the dense buffer — Cholesky is left-looking, so every column k < s_start with any `L[i, k] != 0` for some `i` in the supernode's row map contributes.
   - Call `chol_dense_factor` on the `s_size × s_size` diagonal slab with the configured tolerance.
   - Return `SPARSE_ERR_NOT_SPD` on failure so the caller can surface the error to the user.
2. Writeback only the diagonal slab back into CSC now; the panel slab stays untouched pending Day 8's dense triangular solve.
3. Wire the partial batched path into `chol_csc_eliminate_supernodal`:
   - For each detected supernode: run `chol_csc_supernode_eliminate_diag`, then fall back to the scalar kernel for the panel rows (this is still faster than pure scalar because the diagonal-block factor is now dense, but the full speedup comes tomorrow).
   - For columns not inside any supernode: use the existing scalar column-by-column path.
4. Tests:
   - Dense 8×8 SPD matrix: the diagonal block is the entire matrix, so the supernodal path is now one `chol_dense_factor` call; the panel is empty.
   - Block-diagonal SPD (6×6 of 3×3 blocks): each block is a supernode; verify the diagonal slab is factored dense and the rest via scalar.
   - Random SPD 20×20: compare against the scalar kernel's output; residual must match to round-off, identity `L` × `L^T` must match `A`.
   - Non-SPD (inserted negative eigenvalue): `SPARSE_ERR_NOT_SPD` surfaced correctly.
5. Run `make format && make lint && make test` — all clean.

### Deliverables
- `chol_csc_supernode_eliminate_diag` helper using `chol_dense_factor`
- Partial batched supernodal path (diagonal block batched; panel still scalar)
- 4+ tests against scalar-kernel residuals

### Completion Criteria
- Residuals match the scalar kernel to `1e-14` on every SPD test matrix
- Non-SPD detection path surfaces the correct error code
- `make format && make lint && make test` clean

---

## Day 8: Batched Supernodal Cholesky — Panel Triangular Solve

**Theme:** Finish the batched supernode with `chol_dense_solve_lower` on the panel

**Time estimate:** 8 hours

### Tasks
1. Implement `chol_csc_supernode_eliminate_panel(CholCsc *csc, idx_t s_start, idx_t s_size, const double *L_diag, idx_t lda_diag, double *panel, idx_t lda_panel, idx_t panel_rows)`:
   - The diagonal block is already factored (Day 7) and held as `L_diag` (lower triangular, unit-diagonal-for-Cholesky-style storage; consult `chol_dense_factor` conventions).
   - For each of the `s_size` columns in the panel: call `chol_dense_solve_lower` against `L_diag` to produce the below-diagonal L entries of that supernodal column.
   - (Cholesky's panel update for supernodes is a single dense triangular solve per column, not the symmetric rank-`s_size` update that LDL^T or LU would need; `chol_dense_solve_lower` is exactly the right primitive.)
2. Full integration inside `chol_csc_eliminate_supernodal`:
   - Extract → left-looking scalar cmod into the dense buffer (for prior contributions) → diagonal factor (Day 7) → panel solve (Day 8) → writeback.
   - Remove the scalar-panel fallback from Day 7.
3. Update the file header comment at the top of `src/sparse_chol_csc.c`: the "Supernodal extension" section currently says the batched path is follow-up work. Replace it with a description of the completed path, keeping the cdiv / cmod worked example intact for the scalar kernel.
4. Tests:
   - Dense 10×10 SPD: full batched path; residual vs scalar at round-off.
   - bcsstk01 / bcsstk04 (existing fixtures): batched residual matches scalar and matches the user's right-hand side to `1e-10`.
   - Random SPD generator (seeded) at n = 50, 100, 200: 30 matrices total; all match scalar path.
   - Supernode of size 1 (the minimum `min_size` case, which degenerates to scalar behaviour): must produce identical output.
5. Run `make format && make lint && make test` — all clean.

### Deliverables
- `chol_csc_supernode_eliminate_panel` using `chol_dense_solve_lower`
- Full batched supernodal path in `chol_csc_eliminate_supernodal`
- Updated file header comment removing the "follow-up work" language
- 30+ randomised cross-check tests

### Completion Criteria
- Residual `||A·x - b|| / ||b|| < 1e-10` on all SuiteSparse SPD fixtures via the batched path
- Batched vs scalar L differs by at most round-off on every random-SPD test
- `make format && make lint && make test` clean

---

## Day 9: Batched Supernodal Cholesky — Testing, ASan, and Benchmark Refresh

**Theme:** Harden the batched path and capture the factor-time improvement

**Time estimate:** 8 hours

### Tasks
1. Extend `tests/test_chol_csc.c`:
   - Add a parametrised test that runs each existing SPD fixture twice (once scalar, once batched) and asserts byte-identical CSC output (`col_ptr`, `row_idx`, `values` after drop-tolerance).
   - Add a "boundary supernode" test: a matrix whose supernode partitioning produces at least one single-column "supernode" and at least one supernode of size ≥ 4 — exercises both degenerate and non-degenerate branches in the same run.
2. Update `benchmarks/bench_chol_csc.c`:
   - The existing `speedup_csc_supernodal` column was measuring detection + delegation; it now measures the real batched kernel. Update the header comment and any accompanying docstring.
   - Re-run the benchmark on the Sprint 17 corpus (nos4, bcsstk04) and capture the new speedup numbers. These go into `PERF_NOTES.md` during Days 12–13 when the larger corpus lands.
3. Sanitizer pass: `make sanitize` on `test_chol_csc` and `bench_chol_csc`. Fix any finds immediately — do not defer.
4. Update `docs/algorithm.md`:
   - The "Supernodal extension" section's description can now accurately say the batched path uses `chol_dense_factor` + `chol_dense_solve_lower` per supernode. Remove the "detection ships; batched integration deferred" language.
5. Run `make format && make lint && make test && make sanitize` — all clean.

### Deliverables
- Parametrised cross-check tests asserting scalar == batched on existing fixtures
- Benchmark reporting real batched speedup (no longer the delegated-to-scalar value)
- ASan/UBSan-clean batched path
- `docs/algorithm.md` supernodal section updated

### Completion Criteria
- Scalar == batched byte-identical on every existing fixture
- ASan/UBSan produce zero findings on the batched path
- `make format && make lint && make test && make sanitize` clean

---

## Day 10: Transparent Dispatch — CSC → Linked-List Writeback

**Theme:** The hard half of transparent dispatch: turn a `CholCsc` result back into a linked-list `SparseMatrix` that looks identical to what the linked-list factor would have produced

**Time estimate:** 10 hours

### Tasks
1. Study the current `sparse_cholesky_factor_opts` to enumerate every field it mutates on the input `SparseMatrix`:
   - `factor_norm` (used by solve tolerance).
   - `reorder_perm` (the fill-reducing permutation, caller-visible).
   - `row_perm` / `col_perm` / `inv_row_perm` / `inv_col_perm`.
   - `factored` flag.
   - The L entries themselves (stored in the lower triangle of the linked-list structure).
   - Any cached norms / diagnostic counters.
   Capture this as a checklist in a code comment at the top of the new writeback function.
2. Implement `chol_csc_writeback_to_sparse(const CholCsc *L, SparseMatrix *mat, const idx_t *perm)`:
   - Walk the CSC column-by-column; for each nonzero `L[i, j]`, translate the permuted `(i, j)` back to user space using `perm` and call `sparse_insert(mat, user_i, user_j, value)`.
   - This is the inverse of `chol_csc_from_sparse` and must round-trip losslessly on all existing `test_chol_csc` fixtures.
3. Preserve non-L state:
   - Copy `L->factor_norm` into `mat->factor_norm`.
   - Set `mat->factored = true`.
   - Set `mat->reorder_perm` to the AMD permutation the factor used (or leave identity if no reorder was requested) — the caller-facing field must match what the linked-list path would have set.
   - Leave `row_perm` / `col_perm` as identity (Cholesky doesn't pivot; symmetric permutation is absorbed into `reorder_perm`, matching linked-list behavior).
4. Round-trip tests:
   - Factor a matrix via the linked-list path; capture the resulting `SparseMatrix` state as the reference.
   - Factor the same matrix via CSC then writeback; compare every field enumerated in step 1 against the reference.
   - For L entries, compare under a small tolerance (both paths can differ by round-off due to different floating-point orderings; a `1e-14` per-entry tolerance is acceptable).
5. Edge cases:
   - n = 0 (writeback is a no-op; `factored` still set correctly).
   - Caller supplied a matrix that was already marked `factored` (error `SPARSE_ERR_BADARG`).
   - Caller supplied a matrix with non-identity `row_perm` / `col_perm` (error — Cholesky expects unpermuted input).
6. Run `make format && make lint && make test` — all clean.

### Deliverables
- `chol_csc_writeback_to_sparse` with complete state preservation
- Round-trip tests comparing linked-list vs CSC-then-writeback output field-by-field
- 3+ edge-case tests

### Completion Criteria
- CSC path + writeback produces a `SparseMatrix` indistinguishable from the linked-list path's output (values within round-off, all other fields exact)
- Edge-case errors surfaced with the documented codes
- `make format && make lint && make test` clean

---

## Day 11: Transparent Dispatch — Wiring into `sparse_cholesky_factor_opts`

**Theme:** Land the threshold-based dispatch and exercise both sides transparently

**Time estimate:** 10 hours

### Tasks
1. Modify `sparse_cholesky_factor_opts` to dispatch based on `SPARSE_CSC_THRESHOLD`:
   - If `mat->rows < SPARSE_CSC_THRESHOLD`: keep the existing linked-list path (conversion overhead dominates for small matrices).
   - If `mat->rows >= SPARSE_CSC_THRESHOLD`: run `chol_csc_from_sparse_with_analysis` → `chol_csc_eliminate_supernodal` → `chol_csc_writeback_to_sparse` → free the intermediate `CholCsc`.
   - Make the threshold tunable at runtime via an entry in the opts struct so benchmarks and tests can force either side; default falls back to the compile-time `SPARSE_CSC_THRESHOLD`.
2. Update the Sprint 17 `SPARSE_CSC_THRESHOLD` comment in `include/sparse_matrix.h`: it currently says "matrices above this size benefit from CSC" — now the threshold is load-bearing, not a hint. Document the rationale for the default (100) and cite the benchmark that justifies it.
3. Respect the `reorder` opt:
   - `SPARSE_REORDER_AMD` / `SPARSE_REORDER_RCM`: compute the ordering with the existing helpers, pass into `chol_csc_from_sparse_with_analysis` as the external perm.
   - `SPARSE_REORDER_NONE`: identity perm.
4. Telemetry: surface which path was taken via an optional output field in the opts result struct (`used_csc_path: bool`). Useful for the Day 14 integration tests.
5. Tests that assert the right path is chosen:
   - n = 10: linked-list path (verified by `used_csc_path == false`).
   - n = 500 SPD: CSC path, residual matches linked-list to round-off.
   - Runtime threshold override: force CSC at n = 10 and linked-list at n = 500, verify both paths complete successfully and produce equivalent results on the same matrix.
6. Regression: run the entire existing Cholesky test suite (`test_cholesky.c`, `test_sprint8_integration.c`, any downstream users) — every test that called `sparse_cholesky_factor*` must still pass without modification. Capture the path taken for each and confirm no surprising regressions in residuals.
7. Run `make format && make lint && make test && make sanitize` — all clean.

### Deliverables
- `sparse_cholesky_factor_opts` with transparent CSC / linked-list dispatch
- Updated `SPARSE_CSC_THRESHOLD` documentation reflecting load-bearing role
- Runtime opt override + `used_csc_path` telemetry
- All pre-existing Cholesky tests passing unchanged

### Completion Criteria
- Existing tests pass unmodified under the new dispatch
- Residuals match the pre-Sprint-18 baseline to round-off
- Runtime override exercises both paths on the same matrix and they agree
- `make format && make lint && make test && make sanitize` clean

---

## Day 12: Larger SuiteSparse Corpus — Fixture Additions & Benchmark Runs

**Theme:** Expand the default benchmark corpus so the CSC kernel's scaling story is visible

**Time estimate:** 10 hours

### Tasks
1. Select the new fixtures (target: 4 SPD and 2 symmetric indefinite, all with n ≥ 1000):
   - SPD candidates: `bcsstk14` (n = 1806), `s3rmt3m3` (n = 5357), `Kuu` (n = 7102), `Pres_Poisson` (n ≈ 14k).
   - Symmetric indefinite candidates: `bloweybq` (n = 10000), `tuma1` (n ≈ 22k).
   - Verify each is already available under the project's SuiteSparse fixture arrangement, or add a download step to `tests/data/suitesparse/README.md` / the fixture fetch script.
2. Wire the new matrices into the benchmark corpus arrays in `benchmarks/bench_chol_csc.c` and `benchmarks/bench_ldlt_csc.c`. Keep the existing small matrices in the list so the before/after speedup comparison is directly readable.
3. Run the benchmarks:
   - `make bench BENCHMARKS=bench_chol_csc` (or the equivalent invocation in this repo).
   - `make bench BENCHMARKS=bench_ldlt_csc`.
   - Capture results to `docs/planning/EPIC_2/SPRINT_18/bench_day12.txt` (or similar) for reference during the write-up tomorrow.
4. Spot-check residuals on the new matrices: factor + solve for `Ax = b` with a synthetic x; assert `||A·x - b|| / ||b|| < 1e-10` for SPD and `< 1e-9` for indefinite (looser tolerance due to element growth).
5. If any new matrix fails (crash, unconverged, residual too large): investigate today. Common causes will be exhausted CSC capacity (fix by tightening the symbolic analysis) or an element-growth threshold set too aggressively for the new matrices.
6. Integration-test the dispatch path from Day 11 against the new fixtures: every new matrix (n ≥ 1000) must take the CSC path.
7. Run `make format && make lint && make test && make bench` — all clean.

### Deliverables
- New SPD and symmetric indefinite fixtures (n ≥ 1000) added to the default benchmark corpus
- Fresh benchmark numbers captured for Day 13's write-up
- Residual spot-check passing on every new matrix
- Dispatch path observed taking CSC on every large fixture

### Completion Criteria
- Benchmarks run to completion on all new matrices without crashes or capacity overruns
- Residuals meet the SPD / indefinite tolerances on every new matrix
- `make format && make lint && make test && make bench` clean

---

## Day 13: Larger SuiteSparse Corpus — Scaling Analysis & `PERF_NOTES.md` Update

**Theme:** Turn yesterday's numbers into the published scaling story

**Time estimate:** 10 hours

### Tasks
1. Update `docs/planning/EPIC_2/SPRINT_17/PERF_NOTES.md`:
   - Extend the existing table with the new matrices. Columns stay the same: matrix, n, nnz, linked-list factor ms, CSC scalar factor ms, CSC supernodal factor ms, speedup ratios.
   - Add a brief paragraph observing how the scalar-CSC speedup scales with n — the hypothesis from the Sprint 17 retrospective was that pointer-chasing overhead grows proportionally with n, so the ratio should increase on the new fixtures. Either confirm or disconfirm with the data.
   - Note any matrix where the batched supernodal path does not beat the scalar path (if any) and attribute the cause (e.g., small average supernode size, low fill, detection overhead).
2. Add a new "Scaling" section to `docs/algorithm.md` (or extend the existing performance section):
   - Plot-friendly table of `(n, linked-list ms, CSC ms, speedup)` for the enlarged corpus.
   - Short narrative on when CSC wins (large n, high fill, supernodal structure) and when linked-list wins (tiny matrices — below the threshold).
3. Update `README.md`:
   - Refresh the "Performance" section with the best numbers from the new corpus.
   - Mention the transparent dispatch (Day 11) by name — users don't need to do anything to get the speedup, it just happens at the threshold.
4. Retrospective on the threshold: if the Day 12 data shows the crossover n at a different value than 100 (either higher or lower), update the `SPARSE_CSC_THRESHOLD` default and document the change. Do not move the threshold without benchmark data to justify it.
5. Commit the captured benchmark output file from Day 12 to the repo (small text file) so future sprints can replay the exact numbers.
6. Run `make format && make lint && make test` — all clean.

### Deliverables
- Extended `PERF_NOTES.md` table + scaling narrative
- `docs/algorithm.md` scaling section
- `README.md` refreshed performance section
- Tuned `SPARSE_CSC_THRESHOLD` (if data supports a change)

### Completion Criteria
- `PERF_NOTES.md` accurately reflects the Day 12 runs
- README performance numbers are current, not stale
- Any threshold change is justified by a data point in the notes
- `make format && make lint && make test` clean

---

## Day 14: Integration Tests, Documentation Cleanup & Retrospective

**Theme:** Close out: integration tests across the threshold, remove wrapper/delegation language project-wide, write the retrospective

**Time estimate:** 12 hours

### Tasks
1. Integration tests (4 hrs):
   - A new `tests/test_sprint18_integration.c` (or additions to `test_sprint8_integration.c`) that, for each of ~10 representative matrices (mix of n < threshold and n ≥ threshold, SPD and indefinite): factors via `sparse_cholesky_factor_opts`, solves, verifies residual, asserts `used_csc_path` matches the expected path for that n.
   - A "force both paths" test that takes a single matrix near the threshold and factors it twice — once with CSC forced on, once with CSC forced off — asserting the two `SparseMatrix` results are bit-equivalent (up to round-off in L values).
2. Documentation cleanup (3 hrs):
   - `src/sparse_ldlt_csc.c` file header: replace the "Today's kernel is a wrapper" block with a clean description of the native Bunch-Kaufman kernel, keeping the worked solve example intact (the solve did not change this sprint).
   - `src/sparse_chol_csc.c` file header: the "Supernodal extension" and "Role of Sprint 14 symbolic analysis" sections both still describe the batched path as follow-up work in places; update both to past tense.
   - `include/sparse_cholesky.h`: the doc note about `SPARSE_CSC_THRESHOLD` should now describe actual dispatch, not internal-API call paths.
   - `docs/planning/EPIC_2/PROJECT_PLAN.md`: mark Sprint 18 items 1–5 as complete or explicitly deferred. Any deferrals need rationale paragraphs, not one-liners.
3. Final regression (1 hr):
   - `make clean && make format && make lint && make test`.
   - `make sanitize`.
   - `make examples` + run each (confirm none regressed under the new dispatch).
   - `make bench` one more time on the full corpus to capture final numbers.
4. Write `docs/planning/EPIC_2/SPRINT_18/RETROSPECTIVE.md` (3 hrs):
   - Definition of Done checklist (the 5 items from Sprint 18 in `PROJECT_PLAN.md`).
   - What went well: expected wins are the clean conversion of the Sprint 17 solve scaffolding into a native BK kernel, and the `chol_dense_*` primitives from Sprint 17 Day 11 slotting in exactly as planned.
   - What didn't: any items that slipped (e.g., a fixture that couldn't be wired into the corpus, a speedup that underperformed on a particular matrix family) — be specific.
   - Final metrics: test count, benchmark speedups before / after, integration test count.
   - Items deferred (if any, with rationale) — if a native LDL^T supernodal batched path is spotted as future work (cf. the Sprint 17 pattern on the Cholesky side), note it with a link to a future sprint.
   - Lessons for future "wrapper → native" migrations.
5. Metrics collection (1 hr):
   - Total test count.
   - Total Sprint 18 test additions.
   - Benchmark deltas: per-matrix speedup ratio before (Sprint 17 end) vs after (Sprint 18 end).
   - Line counts / complexity of the retired wrapper code (just to document what was removed).

### Deliverables
- `test_sprint18_integration.c` + cross-threshold integration tests
- All Sprint 17 "wrapper / delegation / follow-up" language removed from source, headers, and docs
- `docs/planning/EPIC_2/SPRINT_18/RETROSPECTIVE.md`
- Full clean regression (tests + sanitizers + examples + benchmarks)

### Completion Criteria
- Integration tests pass on both sides of the threshold, on SPD and indefinite matrices
- No code comment, header, or doc still refers to the Sprint 17 kernels as "wrappers" or "deferred"
- Retrospective written with an honest assessment of what hit and missed targets
- `make format && make lint && make test && make sanitize && make bench && make examples` clean
