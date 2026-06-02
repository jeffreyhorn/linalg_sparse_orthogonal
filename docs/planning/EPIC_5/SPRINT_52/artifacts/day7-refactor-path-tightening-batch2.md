# Sprint 52 Day 7: Refactor Path Tightening Batch II

## Purpose

Day 7 completes the bounded Sprint 52 refactor-tightening work without
expanding into a full structural-pattern verifier. Day 6 tightened the
existing `sparse_factors_t` replacement contract; Day 7 tightens the other
remaining shared seam: the analyzed matrix itself could still drift in obvious
ways before a later factor/refactor call.

## Main Day 7 Conclusion

The shared analysis/factor/refactor path now rejects the simplest important
kind of structure drift before later numeric work begins:

- `sparse_analysis_t` now remembers the analyzed matrix nonzero count
- `sparse_factor_numeric(...)` and `sparse_refactor_numeric(...)` both reject
  gross structure mismatch when `sparse_nnz(...)` no longer matches the
  analyzed matrix
- this remains a cheap boundary check, not a full same-pattern verifier
- the existing zero-init and old-factor-preservation behavior remains intact

This stays inside the Sprint 52 scope fence:

- no public API redesign
- no raw internal storage exposure
- no one-shot API demotion
- no LU routing expansion
- no broad structural-pattern verifier redesign

## Touched Code

### `include/sparse_analysis.h`

Day 7 extends the public repeated-run direct contract in the smallest useful
way:

- `sparse_analysis_t` now caches the analyzed matrix nonzero count as
  `source_nnz`
- the `sparse_refactor_numeric(...)` docs now state the final bounded behavior
  more truthfully:
  - same-pattern remains the caller contract
  - obvious gross structure mismatch is rejected
  - the function does not promise a full structural-pattern check

### `src/sparse_analysis.c`

Day 7 adds one shared validation seam:

- `sparse_validate_analysis_input_matrix(...)`

The helper now enforces, before later numeric work:

1. matrix dimensions still match the analysis
2. the matrix is still in original row/col state
3. the matrix nonzero count still matches `analysis->source_nnz`

That helper is now used by both:

- `sparse_factor_numeric(...)`
- `sparse_refactor_numeric(...)`

The important contract detail is that this is intentionally only a cheap,
bounded guard. It catches obvious gross drift while preserving the existing
same-pattern caller precondition rather than replacing it with an expensive
structural verifier.

### `tests/test_integration.c`

Day 7 adds the strongest direct proof for the new boundary:

- `test_public_lifecycle_refactor_rejects_nnz_drift_and_preserves_old_factors(...)`
  - build valid Cholesky factors
  - remove a symmetric off-diagonal pair so the matrix `nnz` changes
  - verify `sparse_refactor_numeric(...)` returns `SPARSE_ERR_BADARG`
  - verify the old factors still solve the original system afterward

The preexisting Day 6 tests remain in place, so the direct lifecycle proof now
covers:

- zero-init first-factorization
- mismatched existing factors
- failed numeric refactor on bad values
- obvious gross structure drift

## Important Contract Detail

Day 7 finalizes the bounded Sprint 52 refactor boundary as:

- accepted:
  - zero-init first factorization
  - repeated refactor/solve on stable structure
  - temporary-factor swap-on-success semantics
- rejected:
  - mismatched existing factors
  - malformed family payload
  - gross NNZ drift against the analyzed matrix
- still caller-owned:
  - full same-pattern truth beyond the cheap guard

That is stronger and more truthful than the Sprint 51 state without claiming
Sprint 52 has become a full structure-verifying direct-refactor layer.

## Explicit Non-Landings

Day 7 intentionally does **not** do these:

- add a full structural-pattern verifier
- redesign `sparse_factors_t`
- reopen LU routing
- add incremental numeric-update machinery
- broaden into benchmark-framework or tutorial redesign
- expose raw CSC/native factor layout

## Validation

Because `*.c` / `*.h` changed, the full required code-day gate was run:

- `make format`
- `make lint`
- `make test`

All passed.

Because this remained a substantial shared direct-lifecycle batch, the stronger
reviewed baseline was also run:

- `make quality-review-full`

That also passed.

## Truthfulness Anchors Preserved

The maintained reviewed baseline stayed exact:

- reviewed CMake parity remained `53`
- Makefile/CMake parity remained `53` vs `53`
- full reviewed CMake `ctest` passed `53 / 53`
- `Total Test time (real) = 242.05 sec`

## Focused Follow-Ons

The high-signal repeated-run direct follow-ons stayed clean:

- `./build/test_integration`
  - `32 / 32` passed
  - `Assertions: 2064`
- `./build/example_analysis`
  - residuals remained `4.44e-16`
- `./build/bench_refactor`
  - `tridiag-200` analyze-once speedup = `1.86x`
  - `tridiag-500` analyze-once speedup = `1.37x`
  - `bcsstk04` analyze-once speedup = `1.84x`
  - `nos4` analyze-once speedup = `1.72x`
- `./build/bench_refactor_csc tests/data/suitesparse/nos4.mtx --repeat 1`
  - `speedup_refactor = 1.72x`
  - `res_ll = 8.24e-16`
  - `res_csc = 7.06e-16`

## Day 7 Operational Result

Sprint 52 now has a materially more complete refactor contract in code:

1. zeroed-state first factorization still works
2. stale or mismatched existing factors are rejected
3. obvious gross structure drift is rejected
4. old factors still survive failed refactor attempts

That closes the bounded Sprint 52 refactor-tightening work cleanly enough for
the next day to focus on benchmark proof rather than reopening contract
ambiguity.
