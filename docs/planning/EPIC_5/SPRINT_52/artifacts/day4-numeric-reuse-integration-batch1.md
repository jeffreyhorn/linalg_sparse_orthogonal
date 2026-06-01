# Sprint 52 Day 4: Numeric Reuse Integration Batch I

## Purpose

Day 4 lands the first bounded Phase 2 implementation batch for Sprint 52.
The target is the strongest shared-path seam from the Day 3 audit:
remove avoidable extra symbolic-analysis work from the repeated-run Cholesky
CSC path without reopening LU or the harder LDL^T permutation/pivot seam.

## Main Day 4 Conclusion

The shared direct repeated-run Cholesky path is now materially deeper than it
was at Sprint 52 start:

- `sparse_factor_numeric(...)` now reuses the caller's
  `sparse_analysis_t` directly when Cholesky routes to the CSC path
- the old second `sparse_analyze(...)` hidden inside the one-shot Cholesky CSC
  wrapper is no longer part of that repeated-run path
- the smaller linked-list Cholesky route remains intentionally unchanged

This stays inside the Sprint 52 scope fence:

- no public API redesign
- no raw internal storage exposure
- no one-shot API demotion
- no promise that reuse preserves old numeric factor contents
- no attempt to solve the deeper LDL^T BK/symmetric-permutation seam early

## Touched Code

### `src/sparse_analysis.c`

Day 4 adds the first shared-path numeric-reuse helper for direct factorization:

- `factor_cholesky_with_analysis_csc(...)`

That helper:

- builds the CSC working factor directly from
  `chol_csc_from_sparse_with_analysis(...)`
- reuses the caller's `analysis->perm`, etree, postorder, and symbolic pattern
  through the existing analysis-aware CSC conversion path
- runs `chol_csc_eliminate_supernodal(...)`
- writes the finished factor back into a fresh factor-owned
  `SparseMatrix`

The Cholesky case inside `sparse_factor_numeric(...)` now splits cleanly:

- `n >= SPARSE_CSC_THRESHOLD`
  - direct analysis-aware CSC path
- otherwise
  - existing linked-list `REORDER_NONE` delegated path

### `src/sparse_chol_csc_internal.h`

Day 4 moves the shared supernode cutoff ownership to the internal CSC header:

- `SPARSE_CSC_SUPERNODE_MIN_SIZE`

That lets the one-shot Cholesky path and the shared analysis/factor path use
the same supernodal threshold without duplicating the constant definition.

### `src/sparse_cholesky.c`

The local duplicate `SPARSE_CSC_SUPERNODE_MIN_SIZE` block was removed because
the shared internal header now owns it.

### `include/sparse_analysis.h`

The public repeated-run direct API description now reflects reality more
closely:

- the shared Cholesky CSC path already consumes analysis directly on larger
  repeated-run problems
- the remaining shared paths still delegate more heavily through one-shot
  family routines

## Important Contract Detail

Day 4 preserves the repeated-run direct ownership model from Sprint 50/51:

- `analysis->perm` remains the single published symmetric permutation for the
  repeated-run direct path
- the factor-owned `SparseMatrix` produced for Cholesky keeps
  `reorder_perm == NULL`
- this matches the old delegated `REORDER_NONE` repeated-run end state and
  keeps solve-path semantics stable

That means the batch changes internal reuse depth, not caller-visible
permutation ownership.

## Explicit Non-Landings

Day 4 intentionally does **not** do these yet:

- deepen LDL^T through the BK/symmetric-permutation CSC seam
- change LU routing or parameter exposure
- tighten `sparse_refactor_numeric(...)` yet
- add new public structs or new direct family APIs
- broaden into tutorial/README/example rewriting

## Validation

Because `*.c` / `*.h` changed, the full required code-day gate was run:

- `make format`
- `make lint`
- `make test`

All passed.

Because this was a substantial shared direct-lifecycle batch, the stronger
reviewed baseline was also run:

- `make quality-review-full`

That also passed.

## Truthfulness Anchors Preserved

The maintained reviewed baseline stayed exact:

- reviewed CMake parity remained `53`
- Makefile/CMake parity remained `53` vs `53`
- full reviewed CMake `ctest` passed `53 / 53`
- `Total Test time (real) = 354.23 sec`

## Focused Follow-Ons

The high-signal repeated-run direct follow-ons stayed clean:

- `./build/example_analysis`
  - residuals remained `4.44e-16`
- `./build/bench_refactor`
  - `tridiag-200` analyze-once speedup = `1.46x`
  - `bcsstk04` analyze-once speedup = `1.66x`
  - `nos4` analyze-once speedup = `1.57x`

## Day 4 Operational Result

Sprint 52 now has one real repeated-run integration improvement in code, not
just in planning notes. The next bounded choices are clearer:

1. deepen LDL^T if the BK-specific seam can be handled without broadening
   scope
2. tighten `sparse_refactor_numeric(...)` so the public refactor story is less
   shallow
3. keep LU as the intentionally bounded family-specific seam until later
