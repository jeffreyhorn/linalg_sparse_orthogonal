## Sprint 47 Day 10: Example Safety Cleanup Batch

### Objective

Land the bounded Day 9 example cleanup by aligning `example_eigs.c` to the
current example allocation-helper seam, without broadening into larger
multi-demo examples or unrelated benchmark/script churn.

### Commands Run

1. Re-read the primary target and current helper seam:
   - `sed -n '1,340p' examples/example_eigs.c`
   - `sed -n '1,220p' examples/example_alloc_helpers.h`
   - `sed -n '341,420p' docs/planning/EPIC_4/SPRINT_47/PLAN.md`
2. Land the bounded Day 10 code batch:
   - `apply_patch` on:
     - `examples/example_eigs.c`
3. Run the required code-quality gate and the auxiliary build/runtime checks:
   - `make format`
   - `make lint`
   - `make test`
   - `make tooling-build`
   - `./build/example_eigs`

### Findings

#### 1. The right Day 10 move was shared-helper adoption, not example redesign

`example_eigs.c` already had a clear repeated allocation pattern:

- eigenvector bundles
- per-demo `A*v` scratch vectors
- the same raw `malloc` / `calloc` shape repeated across the nos4, KKT, and
  LOBPCG sections

The Sprint 47 fix was to route those allocations through
`examples/example_alloc_helpers.h` rather than redesign the example structure.

Interpretation:

- the batch stayed exactly in the helper/safety lane
- Sprint 47 did not turn a small example cleanup into algorithm churn

#### 2. `example_eigs.c` now follows the current example allocation seam

The file now includes:

- `example_alloc_helpers.h`

and the raw allocation sites now use:

- `example_calloc_array(...)`
- `example_malloc_array(...)`

This now covers all touched dynamic buffers in the example:

- `vecs`
- `Av`
- `kvecs`
- `KAv`
- `bvecs`
- `BAv`

Interpretation:

- the strongest direct shared-helper adoption target identified on Day 9 is now
  complete

#### 3. The cleanup also avoids pre-multiplied `idx_t` count drift

For the multi-vector bundles, the helper calls were written in a way that keeps
the count dimension as the row count and moves the fixed-vector width into the
element size:

- `example_calloc_array(n, sizeof(double[5]), ...)`
- `example_calloc_array(nk, sizeof(double[3]), ...)`
- `example_calloc_array(nb, sizeof(double[3]), ...)`

Interpretation:

- the helper adoption did not simply move unchecked count multiplication to a
  different line
- the touched example now better matches the current safety intent

#### 4. The public runtime behavior stayed unchanged

The direct `./build/example_eigs` rerun stayed green:

- nos4 largest-eigenvalue demo converged `5 / 5`
- KKT nearest-`sigma` demo converged `3 / 3`
- bcsstk04 IC(0)-preconditioned LOBPCG demo converged `3 / 3`

Representative reported values:

- nos4 top residual check stayed around `4.331e-14`
- KKT residual checks stayed around `1e-15`
- bcsstk04 LOBPCG reported residual stayed `8.808e-09`

Interpretation:

- Day 10 improved helper/safety consistency without changing the example’s
  public demonstration story

#### 5. The batch stayed narrow and did not reopen the deferred example queue

No Day 10 changes were needed in:

- `example_iterative.c`
- `example_matrix_free.c`
- `example_colamd.c`
- `example_analysis.c`
- `example_condition.c`
- `example_ic_minres.c`

Interpretation:

- the Day 9 defer/keep boundary held
- Sprint 47 still has an honest small-example batch rather than a broad sweep

### Bottom Line

Sprint 47 Day 10 successfully converted the strongest remaining small-example
allocation hotspot onto the shared helper seam:

- touched target completed:
  - `examples/example_eigs.c`
- already aligned examples remained untouched:
  - `example_iterative.c`
  - `example_matrix_free.c`
  - `example_colamd.c`
- larger raw-allocation examples remain explicitly deferred:
  - `example_analysis.c`
  - `example_condition.c`
  - `example_ic_minres.c`

The code-quality gate, auxiliary build surface, and direct example rerun all
passed.
