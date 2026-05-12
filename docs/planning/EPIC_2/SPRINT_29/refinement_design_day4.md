# Sprint 29 Day 4 — Item 3: Eigenpair Refinement via Inverse Iteration (Design)

## Motivation — Sprint 20 Retrospective Citation

`docs/planning/EPIC_2/SPRINT_20/RETROSPECTIVE.md` "Items deferred" line 96:

> **Iterative refinement for eigenpairs.**  Not in scope.  The Wu/Simon
> residual is the accuracy bound users should trust; callers who need
> tighter answers can either bump `opts.tol` or apply inverse-iteration
> post-hoc on the returned eigenvector.

Sprint 29 Item 3 lifts that "post-hoc inverse-iteration" pattern into
the library as an opt-in post-pass.  Wu/Simon's `|β_m · y_{m-1, j}| /
max(|θ_j|, scale)` residual remains the **production accuracy gate**:
the library's default behaviour is unchanged.  Refinement is opt-in
for downstream callers (deflation pipelines, response-function
evaluation, sensitivity analysis) that need eigen-equation residuals
near `machine_eps · ||A||` rather than `1e-10`.

## Decision

**Pick Rayleigh-quotient iteration with a fresh LDL^T factor at each
inverse step.**  For each converged Ritz pair `(λ_j, v_j)`:

```
for iter in [0, opts.refine_max_iters):
    r = A·v_j - λ_j·v_j
    if ||r||_2 / |λ_j| ≤ machine_eps_tight: break
    factor: (A - λ_j·I) = L·D·L^T            via sparse_ldlt_factor_opts(AUTO)
    solve:  (A - λ_j·I) · y = v_j           via sparse_ldlt_solve
    v_j    = y / ||y||_2
    λ_j    = v_j^T · A · v_j                 (Rayleigh quotient)
    free factor
update result->eigenvalues[j] = λ_j
update result->eigenvectors[:, j] = v_j
update result->residual_norm = max_j ||A·v_j - λ_j·v_j||_2 / max(|λ_j|, scale)
```

## Alternatives Considered

- **(a) Reuse the Sprint-20 shift-invert factor.**  Rejected: the
  Sprint 20 NEAREST_SIGMA path factors `(A - σ·I)` once at the
  user-supplied shift `σ` and uses that factor throughout Lanczos.
  After convergence, the refinement shift is `λ_j` (the converged
  Ritz value), which is *not* equal to `σ` — applying `(A - σ·I)^{-1}`
  to `v_j` produces `θ_j · v_j` (a scalar multiple of `v_j` itself,
  by the eigen-equation `(A - σ·I)·v_j = (1/θ_j)·v_j`).  No
  refinement happens.  The Day 5 PLAN.md task 2 wording ("reuse the
  Sprint-20 shift-invert factored matrix") was optimistic; the
  algebra requires a new factor at each `λ_j`.

- **(b) Single shared factor at `λ_target = mean(λ_j)`.**  Rejected:
  refinement convergence per pair is set by the spectral gap to the
  nearest *other* eigenvalue when the shift is exact, and degrades
  quadratically with `|λ_target - λ_j|`.  A shared shift would give
  one or two converged pairs and leave the outliers unrefined.  Not
  worth the implementation complexity savings vs option (b)'s
  inability to deliver the contract.

- **(c) Plain inverse iteration without Rayleigh-quotient update.**
  Rejected as a strict subset: at each iter, `λ_j` stays at the
  Lanczos-emitted value, so the shift in `(A - λ_j·I)` doesn't
  tighten.  Rayleigh-quotient iteration's `λ_j ← v_j^T A v_j` after
  each inverse step is essentially free (one matvec + dot product)
  and gives cubic convergence near a simple eigenvalue.  No reason
  to skip it.

- **(d) Subspace iteration / locking.**  Rejected: out of scope for
  Item 3's "opt-in post-pass" framing.  If a caller needs block
  refinement they can already chain `sparse_eigs_sym(refine=true)`
  calls or implement their own.

## API Surface

Two new fields appended to `sparse_eigs_opts_t` (preserves Sprint 21
ABI break + designated-init back-compat per the Sprint 21 pattern):

```c
typedef struct {
    /* ... existing fields ... */

    /** Sprint 29 Day 5: opt-in eigenpair refinement post-pass.
     *  Nonzero enables Rayleigh-quotient iteration on each converged
     *  Ritz pair after the Lanczos / LOBPCG loop returns.  Each
     *  iteration solves (A - λ_j·I) · y = v_j via a fresh LDL^T
     *  factor at the current Ritz value, normalises y, then updates
     *  λ_j via the Rayleigh quotient v_j^T A v_j.  Terminates when
     *  the per-pair residual ||A·v_j - λ_j·v_j||_2 / max(|λ_j|, scale)
     *  drops below the tight tolerance (≈ 100 · machine_eps) or
     *  `refine_max_iters` is hit, whichever comes first.  Default 0
     *  (refinement off — Sprint 28 behaviour bit-identical).  Requires
     *  `compute_vectors = 1` (vectors are the input to inverse
     *  iteration); rejected with SPARSE_ERR_BADARG when set without
     *  vectors. */
    int refine;
    /** Sprint 29 Day 5: cap on refinement iterations per converged
     *  Ritz pair.  Default 0 selects the library default (5).
     *  Negative values rejected with SPARSE_ERR_BADARG.  Each
     *  iteration runs one LDL^T factor + solve + Rayleigh update —
     *  on the Pres_Poisson-class corpus that's ~5 s per iter, so a
     *  budget of 5 caps refinement at ~25 s per pair.  Rayleigh-
     *  quotient iteration converges cubically near simple
     *  eigenvalues so 2-3 iters typically suffice. */
    idx_t refine_max_iters;
} sparse_eigs_opts_t;
```

## Result-Struct Contract

When `refine = true`:
- `result->eigenvalues[j]` is updated to the post-refinement Rayleigh
  quotient for `j ∈ [0, n_converged)`.  The output ordering (LARGEST
  descending / SMALLEST ascending / NEAREST_SIGMA by `|λ - σ|`) is
  preserved — refinement only tightens existing values; it does not
  reorder them.
- `result->eigenvectors[:, j]` is updated to the post-refinement unit
  vector.
- `result->residual_norm` is the maximum post-refinement relative
  residual across the converged pairs (which is `≤ opts->tol` by the
  Wu/Simon gate + further reduced by refinement).
- `result->iterations` is unchanged (the Lanczos iteration count is
  the Wu/Simon convergence-path metric; the refinement loop count is
  not currently surfaced — add a `result->refine_iters` field in a
  future sprint if a caller needs it).

When `refine = false`: result struct unchanged from Sprint 28.

## Per-Pair Edge Cases

- **`λ_j` is exactly an eigenvalue of A**: `(A - λ_j·I)` is singular
  and `sparse_ldlt_factor_opts` returns `SPARSE_ERR_SINGULAR`.
  Handle by perturbing `λ_j ← λ_j + δ` where `δ = 100 · machine_eps ·
  max(|λ_j|, 1.0)` and retrying.  This is the same perturbation
  pattern Sprint 20 documented for the `sigma` user-input case.
- **Already-converged pair**: if the initial residual is already
  below the tight tolerance, refinement is a no-op (loop exits at
  iter = 0).  This is the common case for well-separated extremes
  on well-conditioned matrices.
- **Rayleigh-quotient stalls**: if `λ_j` moves by less than the
  tight tolerance across consecutive iters, we've converged to the
  numerical fixed point — break early even if `refine_max_iters`
  isn't exhausted.

## Numerical Comparability

With `refine = false`, the call is bit-identical to Sprint 28
(production default off + no code path entered).  This is the same
back-compat contract Sprint 28 used for the new advisory env vars.

With `refine = true`, results differ from Sprint 28 only in the
direction of *tighter* residuals.  Refinement cannot move `λ_j` to a
different eigenvalue (Rayleigh-quotient iteration is locally
convergent to the eigenvalue closest to the starting shift, which is
the Lanczos-converged Ritz value).

## Backend Coverage

Both Lanczos backends (`SPARSE_EIGS_BACKEND_LANCZOS` and
`SPARSE_EIGS_BACKEND_LANCZOS_THICK_RESTART`) and the LOBPCG backend
(`SPARSE_EIGS_BACKEND_LOBPCG`) share the same converged-pair
post-pass: after the backend writes `n_converged`, `eigenvalues[]`,
`eigenvectors[]`, and `residual_norm`, the refinement loop reads
those buffers and updates in-place.  Day 5 task 5 covers LOBPCG
specifically (`test_eigs_refine_lobpcg_backend`).

## What Ships in Sprint 29 Day 4

- `include/sparse_eigs.h`: `opts.refine` + `opts.refine_max_iters`
  trailing fields with the full doc-comments above.
- `src/sparse_eigs.c`: validation rejects `refine && !compute_vectors`
  with SPARSE_ERR_BADARG; refine path itself is a Day-1-style stub
  (returns SPARSE_ERR_BADARG explaining "lit up in Sprint 29 Day 5"
  for now — see below).  Actually, simpler: stub silently no-ops with
  a TODO note, so that `refine = true && compute_vectors = true`
  doesn't trip the validation rejection in Day 4 → returns the
  default-off-equivalent result for now.  Day 5 replaces the no-op
  with the real loop.
- `tests/test_eigs.c`: two failing-as-expected tests
  (`test_eigs_refine_default_off_unchanged` +
  `test_eigs_refine_tightens_residual`) — RUN_TEST commented out
  until Day 5.
- `docs/planning/EPIC_2/SPRINT_29/refinement_design_day4.md` (this
  doc).
- All quality checks clean.

## Sprint 29 Day 5 (planned)

- Implement the Rayleigh-quotient iteration loop in `src/sparse_eigs.c`.
- Light up the Day-4 tests + add 2 more (LOBPCG-backend coverage +
  `refine_max_iters` budget enforcement).
- Per Sprint 29 PLAN.md Day 5, ~12 hrs.

## LOC Estimate

- Day 4 (this commit): ~50 LOC (API + validation + no-op stub +
  failing-as-expected tests + design doc).
- Day 5: ~150 LOC (Rayleigh loop + matvec residual computation +
  factor-and-solve dispatch + edge-case handling) + ~120 LOC tests.

## References

- `docs/planning/EPIC_2/SPRINT_29/PLAN.md` Day 4 + Day 5 sections.
- `docs/planning/EPIC_2/PROJECT_PLAN.md` Sprint 29 Item 3.
- `docs/planning/EPIC_2/SPRINT_20/RETROSPECTIVE.md` line 96 — the
  original "callers can do this post-hoc" deferral.
- `src/sparse_eigs.c::sparse_eigs_sym` lines 580-1040 — the Lanczos
  dispatch the refinement post-pass hooks into.
- `src/sparse_eigs.c::s20_op_shift_invert` line 478 — the
  `sparse_ldlt_solve` wiring that the refinement step reuses (with a
  per-pair factor instead of the user-supplied `σ` factor).
- Parlett, *The Symmetric Eigenvalue Problem* (1980) §4.6 —
  Rayleigh-quotient iteration convergence theorem.
