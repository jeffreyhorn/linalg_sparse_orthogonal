# Sprint 52 Day 5: Numeric Reuse Integration Batch II

## Purpose

Day 5 lands the second bounded Phase 2 implementation batch for Sprint 52.
The target is the harder shared-path seam left open by Day 4:
deepen the LDL^T CSC repeated-run path without broadening into LU or a larger
public direct-lifecycle redesign.

## Main Day 5 Conclusion

The shared direct repeated-run LDL^T path is now materially deeper than it was
at Sprint 52 start:

- `sparse_factor_numeric(...)` now reuses the caller's `sparse_analysis_t`
  directly when LDL^T routes to the CSC path and the scalar BK pre-pass does
  not introduce extra symmetric swaps beyond the caller's reorder
- when BK *does* add extra swaps, the path now rebuilds symbolic analysis only
  on the resulting pre-permuted matrix
- the smaller linked-list LDL^T route remains intentionally unchanged

This stays inside the Sprint 52 scope fence:

- no public API redesign
- no raw internal storage exposure
- no one-shot API demotion
- no LU routing change
- no promise that reuse preserves old numeric factor contents
- no broad refactor-path redesign yet

## Touched Code

### `src/sparse_analysis.c`

Day 5 adds the shared-path LDL^T CSC reuse helper:

- `perm_matches_analysis_reorder(...)`
- `factor_ldlt_with_analysis_csc(...)`

The helper takes a bounded two-stage approach:

1. run the existing scalar BK CSC pre-pass with the caller's reorder applied
2. compare the resulting symmetric permutation against the caller's analysis
3. if they match:
   - reuse `ldlt_csc_from_sparse_with_analysis(...)` directly from the
     caller's analysis
4. if they do not match:
   - build the BK-pre-permuted matrix
   - rerun `sparse_analyze(..., REORDER_NONE)` only on that matrix
   - feed the derived analysis into
     `ldlt_csc_from_sparse_with_analysis(...)`
5. seed the batched CSC factor with the scalar pre-pass pivot-size choices
6. run `ldlt_csc_eliminate_supernodal(...)`
7. fall back to the scalar pre-pass factor if the supernodal CSC path does not
   complete cleanly

The LDL^T case inside `sparse_factor_numeric(...)` now splits cleanly:

- `n >= SPARSE_CSC_THRESHOLD`
  - shared LDL^T CSC repeated-run path
- otherwise
  - existing linked-list delegated path

### `src/sparse_ldlt_csc_internal.h`

Day 5 makes the internal header self-contained for the new shared call site:

- include `sparse_ldlt.h` so the
  `ldlt_csc_writeback_to_ldlt(..., sparse_ldlt_t *)` declaration does not
  rely on include-order accidents

### `include/sparse_analysis.h`

The public repeated-run direct API description now reflects the live LDL^T
state more accurately:

- the shared Cholesky CSC path already consumes analysis directly on larger
  repeated-run problems
- the shared LDL^T CSC path now does the same when BK does not introduce extra
  swaps, and rebuilds analysis only when it has to

### `tests/test_integration.c`

The direct lifecycle parity test now proves the simplest important Day 5 case:

- on a large SPD path where BK adds no extra swaps, the one-shot LDL^T factor
  permutation and the shared repeated-run permutation both stay equal to the
  original `analysis.perm`

## Important Contract Detail

Day 5 preserves the repeated-run direct ownership model from Sprint 50/51:

- `analysis->perm` remains the single published symmetric permutation for the
  shared direct path
- the shared LDL^T path only reuses the caller's analysis directly when that
  permutation still matches the scalar BK pre-pass result
- when BK changes the symmetric permutation, the path does not pretend the old
  analysis is still authoritative; it derives a new one on the pre-permuted
  matrix instead

That means the batch deepens internal reuse without weakening the truthfulness
of the repeated-run direct contract.

## Explicit Non-Landings

Day 5 intentionally does **not** do these yet:

- redesign `sparse_refactor_numeric(...)`
- reopen LU routing or parameter exposure
- add new public structs or new direct family APIs
- broaden into tutorial/README/example rewriting
- expose raw CSC/native factor layout

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
- `Total Test time (real) = 220.30 sec`

## Focused Follow-Ons

The high-signal repeated-run direct follow-ons stayed clean:

- `./build/example_analysis`
  - residuals remained `4.44e-16`
- `./build/test_integration`
  - `29 / 29` passed
- `./build/test_ldlt`
  - `83 / 83` passed
- `./build/test_ldlt_csc`
  - `95 / 95` passed
- `./build/bench_refactor`
  - `tridiag-200` analyze-once speedup = `1.69x`
  - `tridiag-500` analyze-once speedup = `1.40x`
  - `bcsstk04` analyze-once speedup = `1.73x`
  - `nos4` analyze-once speedup = `1.48x`
- `./build/bench_refactor_csc tests/data/suitesparse/nos4.mtx --repeat 1`
  - `speedup_refactor = 1.70x`
  - `res_ll = 8.24e-16`
  - `res_csc = 7.06e-16`

## Day 5 Operational Result

Sprint 52 now has a real LDL^T repeated-run deepening batch in code, not just
in design notes. The next bounded choices are clearer:

1. tighten `sparse_refactor_numeric(...)` so the public refactor story is less
   shallow
2. expand factor-many proof on the strongest benchmark and direct-lifecycle
   regression surfaces
3. keep LU as the intentionally bounded family-specific seam until a later
   sprint chooses to reopen it explicitly
