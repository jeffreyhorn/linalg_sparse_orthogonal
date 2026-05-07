# Sprint 25 Day 6 — Spectral Bisection at Coarsest Level: Design

## Context

Sprint 25 Days 1-5 closed two of the three algorithmic axes
PROJECT_PLAN.md Sprint 25 routes from Sprint 24's deferred items:

- **Item 1 (HCC)** — Days 1-3 — contributes ~1.5pp on Pres_Poisson
  (0.952× → 0.937×).  Default-flip blocked by Kuu's +8.5 % nnz_L
  regression at HCC; HCC ships behind `SPARSE_ND_COARSENING=hcc` as
  advisory.
- **Item 2 (multi-pass FM intermediate)** — Days 4-5 — contributes
  ~0pp on Pres_Poisson (passes=1 stays default).  Default-flip
  blocked by Pres_Poisson failing the ≥ 1pp tightening rule; ships
  behind `SPARSE_FM_INTERMEDIATE_PASSES` as advisory for Kuu /
  bcsstk14.
- **Item 3 (spectral bisection)** — Days 6-8 — must close the
  remaining ~8.7pp on Pres_Poisson essentially alone (per
  `intermediate_fm_decision.md` headline-gate accounting).

This document is Day 6's design + Laplacian helper + stub-and-dispatch
landing.  Days 7-8 implement + validate.

## Algorithm

Per PLAN.md Day 7 task 1:

> Replace `graph_coarsest_bisection`'s brute-force-or-GGGP path
> (currently brute-force for n ≤ 20, GGGP for n > 20) with
> Fiedler-vector-based spectral bisection: compute the
> second-smallest eigenvalue's eigenvector of the graph Laplacian
> via the Sprint 20-21 Lanczos eigensolver (`sparse_eigs_sym` with
> shift-invert at σ ≈ 0+ε), partition by the median value with a
> balance-tolerance fallback to GGGP if the Fiedler cut is more
> imbalanced than 60/40.

### Steps

1. **Build the graph Laplacian** `L = D - A` where `D` is the
   diagonal degree matrix and `A` is the adjacency matrix.  For
   weighted graphs (`fine->ewgt != NULL`):  `L[i][i] = sum_k
   fine->ewgt[k]` over edges incident to vertex `i`; `L[i][j] =
   -fine->ewgt[k]` for `j = fine->adjncy[k]`.  For unit-weighted
   graphs (`ewgt == NULL`):  `L[i][i] = degree(i)`; `L[i][j] = -1`
   for `j` adjacent to `i`.

   Properties of L preserved by this construction:
   - Symmetric (since the underlying graph is symmetric).
   - Row sums to zero (the row's diagonal equals the sum of
     absolute off-diagonals).
   - Positive semi-definite (smallest eigenvalue λ_0 = 0; the
     constant vector is the eigenvector).
   - For a connected graph, λ_1 > 0 (algebraic connectivity);
     the corresponding eigenvector v_1 is the **Fiedler vector**.

2. **Solve for the smallest two eigenpairs** via `sparse_eigs_sym`:

   ```c
   sparse_eigs_opts_t opts = {
       .which            = SPARSE_EIGS_SMALLEST,
       .max_iterations   = 0,            /* library default */
       .tol              = 1e-8,         /* relaxed from default
                                          * 1e-10; small Laplacians
                                          * sometimes can't reach
                                          * 1e-10 */
       .reorthogonalize  = 1,            /* required for compute_vectors */
       .compute_vectors  = 1,            /* need v_1 = Fiedler vector */
       .backend          = SPARSE_EIGS_BACKEND_AUTO,
   };
   sparse_eigs_t result = {
       .eigenvalues   = vals,            /* length 2 */
       .eigenvectors  = vecs,            /* length 2 * n column-major */
   };
   sparse_err_t err = sparse_eigs_sym(L, /*k=*/2, &opts, &result);
   ```

   Note: `SPARSE_EIGS_SMALLEST` (algebraically smallest) is the right
   spectrum-selection mode for Laplacian work.  We do NOT use
   `SPARSE_EIGS_NEAREST_SIGMA` with σ ≈ 0+ε because:
   (a) shift-invert at σ near a numerically-zero eigenvalue
       returns SPARSE_ERR_SINGULAR per the API contract.
   (b) Lanczos converges fastest to extreme eigenvalues; the
       smallest end IS the extreme end on a Laplacian (which is
       PSD, so spectrum is bounded below by 0).
   (c) `SMALLEST` doesn't require an LDL^T factorization of
       (L - σI), so it's faster + sidesteps the singularity risk.

   The PLAN.md mentions "shift-invert at σ ≈ 0+ε" — that was a
   design rough draft.  After reading the API contract this Day 6,
   `SMALLEST` is the correct mode.

3. **Extract the Fiedler vector** as `v_1 = result.eigenvectors[n .. 2*n - 1]`
   (column 1 of the column-major buffer).  `v_1` has the property:
   - λ_0 ≈ 0 (within tolerance) and v_0 is approximately constant
     (1/sqrt(n)).
   - λ_1 > 0 (for connected graphs); v_1's components have a sign
     pattern that approximately separates the graph into two
     connected halves.

4. **Median-partition** by v_1:  Compute `median(v_1)`.  For each
   vertex `i`:  `part[i] = 0` if `v_1[i] < median`, else `part[i] = 1`.
   Tie-breaking on equal values uses vertex-id order
   (lower-id vertices go to side 0) for determinism.

5. **Balance check + GGGP fallback**:  After partitioning, count
   `n_0`, `n_1`.  If `min(n_0, n_1) / max(n_0, n_1) < 0.4` (i.e.
   the partition is more imbalanced than 60/40), the Fiedler cut
   is too skewed to use as ND's recursion-balance contract
   requires — fall back to `bisect_gggp` (the existing Sprint 22
   GGGP entry point).

   The 60/40 tolerance picks the worst case ND can recurse on
   without the depth blowing up: the recursion's `n` halves each
   step; under 60/40, depth is `log_{1/0.6}(n) ≈ 1.36 log_2(n)` —
   ~30 levels for `n = 14 822` (Pres_Poisson), comfortably under
   the level_cap of 64 in `sparse_graph_hierarchy_build`.
   Tighter than 60/40 (e.g. 50/50) would be theoretically
   preferable but would cause spectral to fall back too often on
   star-like fixtures where the Fiedler cut naturally puts the
   center alone.

## Edge cases (Day 8 implements; Day 6 stubs)

- `n == 1`: trivial; `part[0] = 0`; skip Lanczos.
- `n == 2`: trivial; `part[0] = 0; part[1] = 1`; skip Lanczos.
- **Disconnected graph**: the Laplacian has multiple zero
  eigenvalues; the "Fiedler vector" is degenerate.  Detect via
  `λ_1 ≈ 0` (within 1e-6 of `λ_0`); fall back to GGGP if so.
- **Lanczos non-convergence** (`result.n_converged < 2`): fall
  back to GGGP.

## `SPARSE_ND_COARSEST_BISECTION` env-var gate

```
SPARSE_ND_COARSEST_BISECTION={spectral,gggp,brute}
```

Default: route by `n` (Sprint 22 behavior):
- `n <= 20` → `bisect_brute_force` (still required for very
  small fixtures where Lanczos is overkill + GGGP can produce
  poor cuts on graphs that admit perfect bisection by
  enumeration).
- `n > 20` → `bisect_gggp` (Sprint 22 default).

When env var is set:
- `spectral` → `graph_bisect_coarsest_spectral` (Day 7 stub
  returns NOT-IMPL on Day 6; Day 7 implements).  On Lanczos
  failure or balance fallback → `bisect_gggp`.
- `gggp` → unconditionally `bisect_gggp` regardless of `n`.
- `brute` → `bisect_brute_force` for `n <= 20`; for `n > 20`
  fall back to `bisect_gggp` (brute on `n > 20` would be
  intractable: 2^(n-1) patterns).
- Unrecognized value → silent fallback to default routing,
  matching Sprint 24 Day 5's `SPARSE_ND_COARSEN_FLOOR_RATIO`
  validation pattern.

Validation pattern: `getenv` + `strcmp` against the three
known values.  Anything else → silent fallback.  Same shape as
Sprint 25 Day 1's `SPARSE_ND_COARSENING` parser.

## Modified-vs-replaced delta from Sprint 22

What spectral keeps from Sprint 22's `graph_bisect_coarsest`:
- Function signature: `(const sparse_graph_t *G, idx_t *part_out) → sparse_err_t`.
- The fallback chain to GGGP if spectral can't produce a
  balanced cut.

What spectral introduces:
- A new `graph_bisect_coarsest_spectral` static function that
  builds the Laplacian, calls `sparse_eigs_sym`, partitions by
  median Fiedler-vector value.
- A new `graph_build_laplacian` helper that constructs the
  Laplacian SparseMatrix.
- The `SPARSE_ND_COARSEST_BISECTION` env-var dispatch in
  `graph_bisect_coarsest`.

What spectral does NOT change:
- The Sprint 22 default-path behavior under unset env var:
  `n <= 20` → brute, `n > 20` → GGGP.  Bit-identical.

## Day 6 deliverables (this commit)

- ✓ This design doc (`spectral_bisection_design.md`).
- `graph_build_laplacian` helper (Day 6 lands; Day 7's spectral
  call uses it).
- `graph_bisect_coarsest_spectral` stub returning
  `SPARSE_ERR_NOT_IMPLEMENTED` (Day 7 lands the implementation).
- `SPARSE_ND_COARSEST_BISECTION` env-var gate in
  `graph_bisect_coarsest` with dispatch:
    - `spectral` → spectral stub (currently fails NOT-IMPL on Day 6;
      callers fall through to GGGP via the wrapper logic).
    - `gggp` → `bisect_gggp`.
    - `brute` → `bisect_brute_force` (clamped to n ≤ 20).
    - default / unrecognized → Sprint 22 routing (brute / GGGP by n).
- 2 stubbed tests in `tests/test_graph.c`:
    - `test_spectral_bisection_eigenvalue_ordering` — pin λ_0 ≈ 0,
      λ_1 > 0 on a connected graph (Day 7-8 lands the asserts).
    - `test_spectral_bisection_gggp_fallback` — pin GGGP fallback
      on a star-graph fixture (Day 7-8 lands the asserts).

## Day 7 reference (next-day handoff)

Day 7 (PLAN.md): implement `graph_bisect_coarsest_spectral` per
the algorithm above, including the median-partition + 60/40
balance fallback + Lanczos-failure fallback.  Land the eigenvalue-
ordering test + GGGP-fallback test asserts.  Verify end-to-end
on the 6-fixture corpus that the spectral path produces tighter
cuts than GGGP on regular meshes (Pres_Poisson + 10×10 grid) without
regressing on irregular SPD (Kuu + s3rmt3m3).

## References

- `include/sparse_eigs.h` — `sparse_eigs_sym` API surface;
  `sparse_eigs_opts_t` + `sparse_eigs_t` definitions; spectrum-
  selection enum; backend enum; per-field doxygen.
- Sprint 20 Day 7 — `sparse_eigs_sym` Lanczos backend introduction.
- Sprint 20 Days 8-13 — shift-invert + LDL^T integration; bench
  + thresholds.
- Sprint 21 Days 1-3 — Wu/Simon thick-restart backend.
- Sprint 21 Days 7-10 — LOBPCG backend + AUTO routing.
- Karypis-Kumar 1998 §3 — METIS spectral bisection (the
  algorithmic reference, though METIS uses MLevelRecursiveBisection
  + RB the Fiedler vector at each level, not just the coarsest).
- Spielman-Teng 2004 — spectral graph theory background; Fiedler
  vector convergence + the algebraic connectivity argument.
