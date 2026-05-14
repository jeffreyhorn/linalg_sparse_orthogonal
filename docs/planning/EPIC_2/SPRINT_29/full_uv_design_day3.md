# Sprint 29 Day 3 — Item 2: Full SVD U/V Output Beyond Economy Mode

## Decision

**Light up the existing `sparse_svd_opts_t.economy` field's `economy=0` path** instead of adding a new `opts.full_u_v` field as the PLAN.md task 2 wording suggested.

Rationale:
- The existing struct already documents `economy=0` as the full-mode toggle: `include/sparse_svd.h:34-35` says "Must be nonzero when compute_uv is set (only economy/thin SVD is implemented). Produces thin U (m×k) and V^T (k×n) where k = min(m,n)."  The phrasing "only economy/thin SVD is implemented" implies full mode is the natural unlit alternative.
- The `sparse_svd_t` result struct's `U` / `Vt` doc-comments already state the full-mode size contract: "Economy: m×k. Full: m×m." and "Economy: k×n. Full: n×n."  Sprint 11-28 left full mode unwired but the contract was documented up-front.
- Adding `full_u_v` would create a confusing duplicate of the existing `economy=0` semantics.
- PLAN.md Day 4 task 1's `test_svd_full_u_v_economy_mode_unchanged` becomes "with `economy=1`" — the test still pins economy-mode bit-identical-to-Sprint-28 behaviour, just using the existing field name.

## Algorithm

For an m × n input A with k = min(m, n):

1. **Run the existing economy SVD path unchanged.**  This produces:
   - `sigma[0..k-1]` — singular values (descending after sort).
   - `U_work` — m × k column-major; columns are orthonormal Lanczos/Householder-based left singular vectors.
   - `V_work` — n × k column-major; columns are orthonormal right singular vectors.

2. **If `compute_uv && !economy`, complete U and V to square orthonormal bases via MGS over canonical unit vectors:**
   - **U-pad** (only when m > k, i.e. tall matrix m > n where k = n): allocate `U_full` of size m × m column-major.  Copy `U_work` columns 0..k-1 into `U_full` columns 0..k-1.  For each new column j ∈ [k, m), find a canonical unit vector e_cand ∈ {e_0, e_1, ..., e_{m-1}} that is linearly independent of the basis so far (cols 0..j-1); MGS-subtract its projection onto the basis; normalize.  Accept the first non-degenerate candidate.  Worst-case n_pad × m candidate-tries × m basis cols × m vector len = O(m^3) per pad, but for a generic matrix the first canonical (cand = j) almost always works → expected O(m^2 · m_pad) = O(m^3) total which matches the dense SVD cost.
   - **V-pad** (only when n > k, i.e. wide matrix m < n where k = m): symmetric — allocate `V_full` of size n × n, copy economy V columns, pad columns k..n-1 via MGS-over-canonicals.
   - **Square case** (m == n == k): no padding needed; full ≡ economy in storage.

3. **Transpose `V_work` (n × k_v col-major) → `svd->Vt` (k_v × n col-major leading dim k_v).**  Economy: k_v = k.  Full: k_v = n.

## Storage Layout Contract

| Mode | `svd->U` | `svd->Vt` |
|------|----------|-----------|
| Economy | m × k col-major, leading dim m | k × n col-major, leading dim k |
| Full | m × m col-major, leading dim m | n × n col-major, leading dim n |

`svd->sigma` is unchanged: always length k = min(m, n), descending order.

The leading-dim distinction is the only structural difference.  Internal callers (`sparse_svd_lowrank_sparse`, `sparse_pinv`, etc.) all pass `opts = {.compute_uv = 1, .economy = 1}` explicitly — none of them index Vt with a non-`k` stride, so the full-mode storage change has zero impact on the existing call surface.

## Numeric Comparability

The economy-mode path (steps 1 + 3 with k_v = k) is **bit-identical** to Sprint 28: the same `bidiag_svd_iterate` runs on the same buffers; sigma is sorted with the same comparator; U columns + V columns are permuted in the same order.  The only operational change is moving the sort step ahead of the V→Vt transpose (the existing post-transpose-row-swap is mathematically equivalent to a pre-transpose-column-swap on V_work — same permutation applied to the same orthonormal columns).

The full-mode path adds an MGS step that operates ONLY on columns past index k of U_work / V_work.  Columns 0..k-1 are NOT touched by the pad step, so the singular triplets are bit-identical to economy mode.

## Reconstruction Identity

For full-mode output:

```
A = U_full · Σ_padded · Vt_full
```

where `Σ_padded` is the m × n rectangular diagonal with `sigma[0..k-1]` on the main diagonal and zeros elsewhere.  Concretely:

```
A[i, j] = sum_{s=0..k-1} sigma[s] · U_full[i, s] · Vt_full[s, j]
```

(only the first k columns of U_full and the first k rows of Vt_full enter the sum, since Σ has only k nonzeros).  The extra padded columns of U_full and rows of Vt_full are orthogonal to A's range and co-range; they don't affect the reconstruction.

This is why `test_svd_full_u_v_reconstruction` (Day 3 task 5) gets the same residual as economy-mode reconstruction.

## Test Coverage

- `test_svd_full_u_v_orthonormality`: 16 × 8 dense random A.  Request full SVD.  Assert `||U^T U - I_16||_F ≤ 1e-10` (full U is 16×16 orthonormal) and `||V V^T - I_8||_F ≤ 1e-10` (V^T is 8×8 orthonormal — square case, full ≡ economy).
- `test_svd_full_u_v_reconstruction`: same fixture.  Reconstruct A_k = sum_{s} sigma[s] · U[:,s] · Vt[s,:] and assert `||A - A_k||_F ≤ 1e-10`.
- `test_svd_full_u_v_economy_mode_unchanged` (Day 4): assert economy-mode output bit-identical to pre-Sprint-29 (`economy=1` path untouched).

## Rejection Rationale (alternatives considered)

- **(a) Add `opts.full_u_v` field**: rejected.  Duplicates the existing `economy` field's semantics.  Would require updating every internal caller to set both fields.  Net negative API ergonomics.
- **(b) Pad inline within `bidiag_svd_iterate` by allocating m × m U / n × n V from the start + running the QR iteration on the full-size buffers**: rejected.  The iteration only writes columns 0..k-1; passing a wider buffer wastes memory and adds no signal.  The post-iteration pad step is the cleanest separation of concerns.
- **(c) Reuse the bidiagonal Householder reflectors to span the left/right null spaces**: rejected.  The Householder reflectors used in `sparse_bidiag_factor` produce m × k U and n × k V (the economy basis); extending to m × m / n × n via reflectors would require re-running bidiagonalization with full-size workspaces.  MGS over canonicals is simpler, has the same asymptotic cost (O(m^3) / O(n^3)), and reuses the post-economy state.

## LOC Estimate

- `src/sparse_svd.c`: ~80 LOC (pad helper + dispatch + sort-restructure).
- `tests/test_svd.c`: ~120 LOC (two new tests + a small helper if needed).
- Total: ~200 LOC + design doc.

## What Ships in Sprint 29 Day 3

- `src/sparse_svd.c`: full-mode dispatch lit up; existing `economy=1` path bit-identical to Sprint 28.
- `tests/test_svd.c`: `test_svd_full_u_v_orthonormality` + `test_svd_full_u_v_reconstruction`.
- `docs/planning/EPIC_2/SPRINT_29/full_uv_design_day3.md` (this doc).
- All quality checks clean.

## References

- `docs/planning/EPIC_2/SPRINT_29/PLAN.md` Day 3 section.
- `docs/planning/EPIC_2/PROJECT_PLAN.md` Sprint 29 Item 2.
- `include/sparse_svd.h::sparse_svd_opts_t.economy` line 34 — existing API toggle.
- `include/sparse_svd.h::sparse_svd_t.U/Vt` lines 48-51 — existing full-mode size contract.
- `src/sparse_svd.c::sparse_svd_compute` lines 617-793 — the path being extended.
